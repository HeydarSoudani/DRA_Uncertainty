"""Train the uncertainty-aware agent (SFT cold start + GRPO).

The training sibling of ``experiments/dra_inference.py --agentic-model uncertainty_aware``.
The common training pipeline lives in ``src/training``; this file holds what is
specific to the uncertainty-aware agent and plugs it in:

  * ``UncertaintyAwareEnv``: one rollout of the uncertainty-aware agent's protocol (system prompt,
    one growing transcript in the user message, format re-asks, retrieval,
    doc novelty, criteria updater, ``<belief>`` block).  Its pieces are
    imported from the inference agent (``deep_research_agents.agents.
    uncertainty_aware_agent``, READ-ONLY), so the policy is trained on exactly the context
    it reads at inference; ``--selftest`` checks that against the inference
    agent itself.
  * the ``agent:`` config section: the ``UncertaintyAwareAgentConfig`` fields, with the
    inference defaults (``ua_*`` keys of ``dra_inference.yaml``).
  * the criteria updater: a frozen model (``agent.criteria_model``, default
    qwen3.6-27b on OpenRouter, the inference backbone), never the policy
    being trained.

Each policy turn is one training sample and gets the run's advantage; the
reward is the outcome only (answer correct: 1, else 0; see
``training.reward.outcome_reward``).

Stages
------
  sft  the teacher (``sft.teacher_model``) runs through UncertaintyAwareEnv; correct
       trajectories become per-turn chat samples (``{output_dir}/sft``).
  rl   GRPO on the policy served by vLLM (``rollout.server_url``).  The GRPO
       update (veRL) is still a TODO: ``trainer.backend: mock`` runs the whole
       loop (rollouts, rewards, advantages) without updating weights.

Examples
--------
  # CPU smoke test (mock policy / retrieval / updater / trainer):
  python experiments/dra_uncertainty_aware_train.py --smoke
  python experiments/dra_uncertainty_aware_train.py --smoke --mode async
  python experiments/dra_uncertainty_aware_train.py --smoke --stage sft
  # Parity of UncertaintyAwareEnv with the inference UncertaintyAwareAgent (CPU, scripted LLMs):
  python experiments/dra_uncertainty_aware_train.py --selftest
  # SFT data from the teacher on TRQA wiki1 validation (retriever on GPU):
  python experiments/dra_uncertainty_aware_train.py --stage sft --limit 200
  # RL rollouts against a vLLM policy server (mock trainer until veRL):
  python experiments/dra_uncertainty_aware_train.py --stage rl --server-url http://127.0.0.1:8000
"""

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── Ensure src/ is importable even without scripts/_activate.sh (defensive) ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging

from dotenv import load_dotenv
load_dotenv()   # OPENROUTER_API_KEY etc., as dra_inference.py loads them

from training.config import DEFAULT_POLICY_MODEL, TrainingConfig, load_config
from training import pipeline
from training.data.schema import PromptRecord
from training.rollout.env import AgentEnv, EnvFactory, GenParams, StepResult
from training.rollout.tool_env import ToolEnv

logger = logging.getLogger("dra_uncertainty_aware_train")

_CONFIG_DEFAULT = str(Path(__file__).resolve().parent / "configs" / "dra_uncertainty_aware_train.yaml")
DEFAULT_CRITERIA_MODEL = "openrouter/qwen/qwen3.6-27b"


def _agent_module():
    """The inference uncertainty-aware agent's module (heavy import: loads every agent)."""
    from deep_research_agents.agents import uncertainty_aware_agent
    return uncertainty_aware_agent


# ── Agent config ──────────────────────────────────────────────────────────────

def uncertainty_aware_config(agent: Dict[str, Any]):
    """``UncertaintyAwareAgentConfig`` from the ``agent:`` section (unknown keys rejected).

    ``criteria_mode`` stays unresolved (``auto`` is resolved per prompt, from
    its dataset, as inference resolves it from ``--dataset``).
    """
    b = _agent_module()
    known = {f.name for f in fields(b.UncertaintyAwareAgentConfig)}
    unknown = set(agent) - known
    if unknown:
        raise ValueError(f"unknown agent keys for the uncertainty-aware agent: {sorted(unknown)}")
    cfg = b.UncertaintyAwareAgentConfig(**{"criteria_mode": "auto", **agent})
    if not cfg.criteria_model:
        # Inference falls back to the policy backbone; in training the policy
        # changes every update, so the updater must be a fixed model.
        cfg.criteria_model = DEFAULT_CRITERIA_MODEL
    return cfg


# ── Criteria updater ──────────────────────────────────────────────────────────

class MockCriteriaLLM:
    """Stateless stand-in for the criteria updater (CPU tests).

    Init: three criteria.  Update: ticks one criterion, chosen from the update
    prompt's hash, to ``partial`` or ``covered``.  Answers in the JSON the real
    prompts ask for, so the tracker's parsing and delta logic run for real.
    """

    NAMES = ["the first condition", "the second condition", "the third condition"]

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        if '"actions"' not in system:
            return json.dumps({"reasoning": "mock init",
                               "criteria": [{"name": n, "status": "not_covered"} for n in self.NAMES]})
        h = int(hashlib.md5(user.encode()).hexdigest(), 16)
        name = self.NAMES[h % len(self.NAMES)]
        status = "covered" if (h >> 8) % 2 else "partial"
        return json.dumps({"reasoning": "mock update", "actions": [
            {"action": "tick", "name": name, "status": status, "evidence": "Source 1 (mock)"}]})


def build_criteria_llm(bcfg) -> Callable[[List[Dict[str, str]]], str]:
    if bcfg.criteria_model == "mock":
        return MockCriteriaLLM()
    from training.rollout.env_llm import TextLLM
    # As UncertaintyAwareAgent._criteria_complete: greedy, reasoning off, <think> stripped.
    return TextLLM(bcfg.criteria_model, temperature=0.0, max_tokens=bcfg.criteria_max_tokens,
                   disable_native_thinking=bcfg.disable_native_thinking, strip_think=True)


# ── Environment ───────────────────────────────────────────────────────────────

class UncertaintyAwareEnv(AgentEnv):
    """One uncertainty-aware-agent run; mirrors ``UncertaintyAwareAgent.inference`` step by step."""

    def __init__(self, tool: ToolEnv, criteria_llm: Callable, bcfg, seen_top_k: int) -> None:
        self.b = _agent_module()
        self.tool = tool
        self.criteria_llm = criteria_llm
        self.cfg = bcfg
        self.seen_top_k = seen_top_k

    async def reset(self, prompt: PromptRecord) -> None:
        b, cfg = self.b, self.cfg
        self.question = prompt.question
        self.mode = b.resolve_criteria_mode(cfg.criteria_mode, prompt.meta.get("dataset"))
        self.tracker = b.CriteriaTracker(
            self.criteria_llm, mode=self.mode, max_criteria=cfg.max_criteria,
            stabilization_window=cfg.stabilization_window,
            evidence_top_k=cfg.evidence_top_k, evidence_chars=cfg.evidence_chars)
        self.init = await asyncio.to_thread(self.tracker.initialize, prompt.question)
        self.last_summary = self.init
        self.show_belief = cfg.show_novelty or cfg.show_criteria
        self.system = b.render_system(cfg.max_turns, show_belief=self.show_belief)
        self.transcript = b.render_user(prompt.question)
        self.registry = b.DocRegistry()
        self.records: List[Dict[str, Any]] = []
        self.extra: List[Dict[str, str]] = []     # re-ask messages after a malformed turn
        self.attempts = 0                         # re-asks spent on the current turn
        self.turns = 0                            # policy turns (re-asks excluded)
        self.step_no = 0                          # searches (belief steps)
        self.end: Optional[str] = None
        self.prediction = ""

    def messages(self) -> List[Dict[str, str]]:
        return self.b.UncertaintyAwareAgent._messages(self.system, self.transcript) + self.extra

    def gen_params(self) -> GenParams:
        return GenParams(max_tokens=self.cfg.max_tokens_per_call, stop=list(self.b.STOP_SEQUENCES))

    def _record(self, turn, attempts: int) -> Dict[str, Any]:
        rec = {"turn": self.turns, "valid": turn.ok, "errors": turn.errors,
               "warnings": turn.warnings, "format_retries": attempts, "action": turn.action,
               "action_text": turn.action_text, "native_reasoning_chars": len(turn.native_reasoning)}
        self.records.append(rec)
        return rec

    async def step(self, text: str) -> StepResult:
        b, cfg = self.b, self.cfg
        turn = b.parse_turn(text)

        if not turn.ok:
            if self.attempts < cfg.max_format_retries:
                self.attempts += 1
                self.extra = [{"role": "assistant", "content": turn.text or text},
                              {"role": "user", "content": b.render_format_error(turn.errors)}]
                return StepResult("retry", valid=False,
                                  info={"errors": turn.errors, "warnings": turn.warnings})
            self._record(turn, self.attempts)
            self.turns += 1
            self.end = b.END_FORMAT
            return StepResult("stop", done=True, valid=False, end_reason=b.END_FORMAT,
                              info={"errors": turn.errors, "warnings": turn.warnings})

        rec = self._record(turn, self.attempts)
        self.attempts, self.extra = 0, []
        self.turns += 1
        info: Dict[str, Any] = {"target_text": turn.text, "warnings": turn.warnings}

        if turn.action == b.ACTION_ANSWER:
            self.prediction = turn.action_text
            self.end = b.END_ANSWERED
            return StepResult("answer", done=True, final_answer=turn.action_text,
                              end_reason=b.END_ANSWERED, info=info)

        # -- search, then the belief that closes this iteration --
        query = turn.action_text
        docs = await asyncio.to_thread(self.tool.search, query, original_query=self.question,
                                       reasoning=turn.think)
        shown = docs[:self.seen_top_k]
        novelty = self.registry.register(shown)
        information = b.render_information(shown, self.registry, cfg.max_passage_chars)
        summary = await asyncio.to_thread(self.tracker.update, self.step_no, shown, [query], self.question)
        self.last_summary = summary
        belief = b.render_belief(self.step_no,
                                 novelty if cfg.show_novelty else None,
                                 summary if cfg.show_criteria else None)
        self.transcript += f"\n\n{turn.text}\n{information}\n"
        if self.show_belief:
            self.transcript += f"{belief}\n"
        rec.update({"step": self.step_no, "novelty": b._r(novelty.nu), "novel_docs": novelty.novel,
                    "shown_docs": novelty.shown, "criteria": summary.to_dict(),
                    "criteria_error": summary.error})
        info.update({"step": self.step_no, "novelty": b._r(novelty.nu),
                     "covered": summary.num_covered, "partial": summary.num_partial,
                     "total": summary.total, "criteria_error": summary.error})
        self.step_no += 1

        done = self.turns >= cfg.max_turns
        if done:
            self.end = b.END_MAX_TURNS
        return StepResult("search", done=done, query=query,
                          end_reason=b.END_MAX_TURNS if done else None,
                          obs_text=information + ("\n" + belief if self.show_belief else ""),
                          obs_doc_ids=[b._doc_id(d) for d in shown], info=info)

    def summary(self) -> Dict[str, Any]:
        b, final = self.b, self.last_summary
        return {
            "ua_criteria": [c.to_dict() for c in self.init.criteria],
            "ua_records": self.records,
            "ua_doc_labels": dict(self.registry.doc_of),
            "ua_outcome": {
                "end": self.end, "answer": self.prediction or None,
                "num_turns": self.turns, "num_searches": self.step_no,
                "format_retries": sum(r.get("format_retries", 0) for r in self.records),
                "criteria_mode": self.mode, "criteria_init_error": self.init.error,
                "criteria_errors": list(self.tracker.errors),
                "final_covered": final.num_covered, "final_partial": final.num_partial,
                "final_not_covered": final.num_not_covered, "final_total": final.total,
                "mean_novelty": b._r(b._mean([r["novelty"] for r in self.records
                                              if r.get("novelty") is not None])),
            },
        }


def make_uncertainty_aware_env(cfg: TrainingConfig, tool: ToolEnv) -> EnvFactory:
    """``pipeline.MakeEnv`` for the uncertainty-aware agent: one shared updater, a fresh env per rollout."""
    bcfg = uncertainty_aware_config(cfg.agent)
    criteria_llm = build_criteria_llm(bcfg)
    print(f"[dra_uncertainty_aware_train] env | max_turns={bcfg.max_turns} criteria={bcfg.criteria_mode} "
          f"updater={bcfg.criteria_model} novelty={bcfg.show_novelty} criteria_shown={bcfg.show_criteria}")
    return lambda: UncertaintyAwareEnv(tool, criteria_llm, bcfg, seen_top_k=cfg.rollout.top_k_docs)


# ── Self-test: UncertaintyAwareEnv == UncertaintyAwareAgent ───────────────────────────────────────

class _ScriptedGenerator:
    """Stands in for a reasoner_component generator: replays outputs, records calls."""

    def __init__(self, outputs: List[str]) -> None:
        self.outputs = list(outputs)
        self.calls: List[List[Dict[str, str]]] = []

    def complete(self, messages, **kwargs) -> str:
        self.calls.append(copy.deepcopy(messages))
        return self.outputs[len(self.calls) - 1]


class _RecordingUpdater(MockCriteriaLLM):
    def __init__(self) -> None:
        self.calls: List[List[Dict[str, str]]] = []

    def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        return super().__call__(messages)

    def complete(self, messages, **kwargs) -> str:
        return self(messages)


class _FakeSearchTool:
    """Both a searcher_component tool (``execute``) and a training ToolEnv (``search``)."""

    def execute(self, query, *, original_query=None, reasoning=None, top_k=None):
        h = int(hashlib.md5(query.encode()).hexdigest()[:4], 16)
        return [{"doc_id": f"doc-{(h + j) % 7}", "title": f"Title {(h + j) % 7}",
                 "text": f"Passage {(h + j) % 7} about {query}."} for j in range(8)]

    def search(self, query, *, original_query=None, reasoning=None, top_k=None):
        return self.execute(query, original_query=original_query, reasoning=reasoning, top_k=top_k)


def selftest() -> None:
    """Run the same scripted policy through UncertaintyAwareAgent.inference and UncertaintyAwareEnv;
    every policy call and every updater call must receive identical messages."""
    from training.config import RolloutConfig
    from training.rollout.agent_rollout import rollout_once
    from training.rollout.policy_client import Generation, PolicyClient

    b = _agent_module()
    question = "Which river flows through the capital of the country that borders both A and B?"
    answered = [
        "<think>Start with the countries.</think>\n<search>countries bordering A and B</search>",
        "I think the answer is obvious.",                                   # malformed: re-asked
        "<think>Retry.</think>\n<search>capital of the country bordering A and B</search>trailing",
        "<think>Now the river.</think>\n<search>river through that capital</search>",
        "<think>Found it.</think>\n<answer>The Danube</answer>",
    ]
    format_failure = answered[:1] + ["no action", "<think>still none</think>", "<search></search>"]
    max_turns = [f"<think>Step {i}.</think>\n<search>query number {i}</search>" for i in range(4)]
    settings = dict(max_turns=4, max_format_retries=2, max_passage_chars=200,
                    evidence_top_k=3, evidence_chars=100, criteria_mode="dynamic")
    cases = [(answered, True, True, "answered"), (answered, False, True, "answered"),
             (answered, True, False, "answered"), (answered, False, False, "answered"),
             (format_failure, True, True, "format_failure"), (max_turns, True, True, "max_turns")]

    for script, show_novelty, show_criteria, expected_end in cases:
        flags = dict(show_novelty=show_novelty, show_criteria=show_criteria)

        # -- inference agent --
        gen, upd = _ScriptedGenerator(script), _RecordingUpdater()
        agent = b.UncertaintyAwareAgent(llm_client=gen, retriever=None, seen_top_k=5, verbose=False,
                              criteria_llm_client=upd, criteria_model="scripted", **settings, **flags)
        agent.search_tool = _FakeSearchTool()
        _, prediction, _ = agent.inference(question, generation_temp=1.0)
        outcome = agent._extras["ua_outcome"]

        # -- training env --
        class _ScriptedPolicy(PolicyClient):
            def __init__(self):
                super().__init__()
                self.i = 0

            async def generate(self, messages, *, max_tokens, temperature, stop=None):
                self.i += 1
                return Generation(text=script[self.i - 1])

        env_upd = _RecordingUpdater()
        bcfg = uncertainty_aware_config({**settings, **flags, "criteria_model": "mock"})
        env = UncertaintyAwareEnv(_FakeSearchTool(), env_upd, bcfg, seen_top_k=5)
        traj = asyncio.run(rollout_once(PromptRecord(id="q", question=question), _ScriptedPolicy(), env,
                                        RolloutConfig(temperature=1.0), group_id="g"))
        env_calls = [t.messages for t in traj.turns]
        env_outcome = traj.meta["ua_outcome"]

        checks = {
            "policy calls": gen.calls == env_calls,
            "updater calls": upd.calls == env_upd.calls,
            "end": outcome["end"] == env_outcome["end"] == traj.stop_reason == expected_end,
            "answer": prediction == traj.final_answer,
            "outcome": all(outcome[k] == env_outcome[k] for k in
                           ("end", "answer", "num_turns", "num_searches", "format_retries",
                            "final_covered", "final_partial", "final_total", "mean_novelty")),
            "records": [
                {k: v for k, v in r.items() if k not in ("belief", "search_novelty")}
                for r in agent._extras["ua_records"]] == traj.meta["ua_records"],
        }
        label = f"{expected_end}: novelty={show_novelty} criteria={show_criteria}"
        if not all(checks.values()):
            for i, (a, e) in enumerate(zip(gen.calls, env_calls)):
                if a != e:
                    print(f"first differing policy call: {i}\n--- inference\n{a}\n--- env\n{e}")
                    break
            raise SystemExit(f"[selftest] FAIL ({label}): {checks}")
        print(f"[selftest] ok  {label}: {len(env_calls)} policy calls, {len(env_upd.calls)} updater calls, "
              f"end={env_outcome['end']}")
    print("[selftest] UncertaintyAwareEnv reproduces UncertaintyAwareAgent.inference")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train the uncertainty-aware agent (SFT + GRPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML config with mostly-fixed knobs; CLI flags override it.")
    p.add_argument("--stage", type=str, default=None, choices=["rl", "sft"], help="Training stage (overrides config.stage).")
    p.add_argument("--mode", type=str, default=None, choices=["sync", "async"], help="RL loop mode (overrides config.rl.mode).")
    p.add_argument("--trainer-backend", type=str, default=None, choices=["mock", "grpo"], help="Trainer backend (overrides config.trainer.backend).")
    p.add_argument("--policy", type=str, default=None, choices=["mock", "server"], help="Policy client (overrides config.rollout.policy).")
    p.add_argument("--policy-model", type=str, default=None,
                   help="Policy (student) model to roll out and train; sets rollout.model AND sft.base_model together. Default: {}".format(DEFAULT_POLICY_MODEL))
    p.add_argument("--server-url", type=str, default=None, help="vLLM server of the policy (overrides config.rollout.server_url).")
    p.add_argument("--tool", type=str, default=None, choices=["mock", "retrieval"], help="Tool env (overrides config.rollout.tool).")
    p.add_argument("--dataset", type=str, default=None, help="Dataset (overrides config.data.dataset).")
    p.add_argument("--subset", type=str, default=None, help="Dataset subset, e.g. wiki1 (overrides config.data.subset).")
    p.add_argument("--split", type=str, default=None, help="Split, e.g. validation (overrides config.data.dataset_year).")
    p.add_argument("--criteria-model", type=str, default=None, help="Criteria updater model (overrides config.agent.criteria_model).")
    p.add_argument("--total-steps", type=int, default=None, help="Trainer updates (overrides config.rl.total_steps).")
    p.add_argument("--group-size", type=int, default=None, help="Rollouts per prompt (overrides config.rollout.group_size).")
    p.add_argument("--limit", type=int, default=None, help="Cap number of prompts.")
    p.add_argument("--output-dir", type=str, default=None, help="Run directory (default: run_outputs/training/uncertainty_aware/<stage>_<time>).")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test: mock policy/tool/updater/trainer/reward + synthetic data.")
    p.add_argument("--selftest", action="store_true", help="Check UncertaintyAwareEnv against the inference UncertaintyAwareAgent, then exit.")
    p.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return p.parse_args()


def _overrides_from_args(args) -> dict:
    """Translate CLI flags into a nested-dict override for load_config."""
    ov: dict = {}

    def setpath(section, key, val):
        if val is not None:
            ov.setdefault(section, {})[key] = val

    if args.stage is not None:
        ov["stage"] = args.stage
    if args.output_dir is not None:
        ov["output_dir"] = args.output_dir
    setpath("rl", "mode", args.mode)
    setpath("rl", "total_steps", args.total_steps)
    setpath("trainer", "backend", args.trainer_backend)
    setpath("rollout", "policy", args.policy)
    setpath("rollout", "server_url", args.server_url)
    setpath("rollout", "tool", args.tool)
    # One flag moves both: rolling out one backbone while training another is
    # silent and produces garbage advantages.
    setpath("rollout", "model", args.policy_model)
    setpath("sft", "base_model", args.policy_model)
    setpath("rollout", "group_size", args.group_size)
    setpath("data", "limit", args.limit)
    setpath("data", "dataset", args.dataset)
    setpath("data", "subset", args.subset)
    setpath("data", "dataset_year", args.split)
    setpath("agent", "criteria_model", args.criteria_model)

    if args.smoke:
        ov.setdefault("rollout", {}).update({"policy": "mock", "tool": "mock", "mock_malformed_rate": 0.15})
        ov.setdefault("trainer", {})["backend"] = "mock"
        ov.setdefault("data", {}).update({"source": "synthetic", "limit": args.limit})
        ov.setdefault("reward", {})["outcome_metric"] = "mock"
        ov.setdefault("agent", {})["criteria_model"] = "mock"
        ov.setdefault("rl", {}).setdefault("total_steps", 4)
        ov.setdefault("rl", {})["prompts_per_step"] = 2
        ov.setdefault("rollout", {}).setdefault("group_size", 4)
    return ov


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.selftest:
        selftest()
        return

    cfg = load_config(args.config, overrides=_overrides_from_args(args))
    if cfg.output_dir is None and not args.smoke:
        cfg.output_dir = str(_REPO_ROOT / "run_outputs" / "training" / "uncertainty_aware"
                             / f"{cfg.stage}_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[dra_uncertainty_aware_train] stage={cfg.stage} rl.mode={cfg.rl.mode} "
          f"trainer={cfg.trainer.backend} policy={cfg.rollout.policy} tool={cfg.rollout.tool} "
          f"model={cfg.rollout.model} out={cfg.output_dir}")
    pipeline.run(cfg, make_env=make_uncertainty_aware_env)


if __name__ == "__main__":
    main()
