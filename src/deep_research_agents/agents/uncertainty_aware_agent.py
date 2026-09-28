"""UncertaintyAwareAgent: a SearchR1-style search agent that reads a progress belief.

Loop, per query::

    criteria     one criteria-updater call extracts the query's criteria, all
                 ``not_covered``
    iteration t  policy writes <think> <search>; code retrieves and shows
                 <information>; code computes doc novelty and one updater call
                 updates the criteria statuses; code appends <belief step="t">
    end          policy writes <think> <answer>, or ``max_turns`` runs out
                 (a run that never answers is a failure)

The policy never writes the belief.  The belief carries two things:

    novelty      new / shown passages of the last search (no LLM call)
    criteria     each criterion with its status only: covered, partial or
                 not_covered (the updater's evidence notes are logged, not shown)

The criteria updater lives in
``deep_research_agents.agent_tools.uncertainty_aware_criteria`` (it began as a
copy of the former controller's criteria-coverage signal).  Its mode is ``static`` (criteria copied from the
query, list fixed) or ``dynamic`` (query decomposed, list may change until it
stabilises); ``auto`` picks static for BrowseComp-Plus and dynamic otherwise.

The run is one growing transcript in the user message, as in the SearchR1
family.  ``run_single`` and retrieval come from :class:`BasicAgent`, the
trajectory streams through the standard logger, and the belief data rides on
the result under ``ua_*`` keys, which the trajectory meta line persists
(``utils.config.AGENT_META_KEYS``).  As in every agent, an uncertainty
estimator attached by ``--uncertainty-estimator-mode monitor`` observes each search
iteration; it never changes the run.

Settings come from the ``ua_*`` keys of ``dra_inference.yaml``.  The
pipeline's ``seen_top_k`` (passages per search) and run temperature apply; its
``max_iteration`` does not: this agent's turn cap is ``ua_max_turns``.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from utils.config import InferenceConfig
from utils.text_utils import doc_text, doc_title
from deep_research_agents.prompts.uncertainty_aware import (
    render_format_error,
    render_system,
    render_user,
)
from deep_research_agents.agent_tools.uncertainty_aware_criteria import (
    MODES,
    CriteriaCoverageSummary,
    CriteriaTracker,
    format_summary_for_log,
)

from .base_agent import BasicAgent

logger = logging.getLogger(__name__)


# ── Turn protocol ─────────────────────────────────────────────────────────────
# A turn is ``<think> … </think>`` followed by exactly one ``<search>`` or
# ``<answer>``.  :func:`parse_turn` returns the parsed pieces plus
#
#     errors    format failures, answered by a re-ask: no action, empty action.
#     warnings  repaired and logged: no <think>, a <belief> or <information>
#               written by the policy (dropped).
#
# Anything after the first closing action tag is cut before parsing (see
# :func:`truncate_after_action`), so a second action or a trailing <think>
# never reaches the parser.  Parsing is regex-based rather than XML: model
# output is close to XML but not reliably well-formed.

ACTION_ANSWER = "answer"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_INJECTED_RE = re.compile(r"<(belief|information)\b[^>]*>.*?</\1>", re.DOTALL)


@dataclass
class Turn:
    think: Optional[str]
    action: Optional[str]          # "search" | "answer" | None
    action_text: Optional[str]
    text: str                      # the turn as it goes into the history
    native_reasoning: str = ""     # extra <think> blocks (provider reasoning re-inlined)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def truncate_after_action(text: str) -> str:
    """Cut the turn after its first closing action tag and close an open one.

    Stop sequences end generation at ``</search>`` / ``</answer>`` but are not
    returned by every provider, and not honoured by every provider either.
    """
    text = text or ""
    ends = [text.find(tag) + len(tag) for tag in ("</search>", "</answer>") if tag in text]
    if ends:
        return text[:min(ends)]
    for tag in ("search", "answer"):
        opened = text.rfind(f"<{tag}>")
        if opened != -1 and f"</{tag}>" not in text[opened:]:
            return text.rstrip() + f"</{tag}>"
    return text


def parse_turn(text: str) -> Turn:
    """Parse one policy turn."""
    errors: List[str] = []
    warnings: List[str] = []
    text = truncate_after_action(text)

    if _INJECTED_RE.search(text):
        warnings.append("the turn contains <belief> or <information>; dropped")
        text = _INJECTED_RE.sub("", text)

    am = _ACTION_RE.search(text)
    action = action_text = None
    if am is None:
        errors.append("no <search> or <answer> in the turn")
    else:
        action, action_text = am.group(1), am.group(2).strip()
        if not action_text:
            errors.append(f"empty <{action}>")

    # The protocol's think is the last <think> before the action.  Earlier
    # blocks are provider reasoning the client re-inlined into the content.
    thinks = list(_THINK_RE.finditer(text))
    before = [m for m in thinks if am is None or m.start() < am.start()]
    think = None
    native = ""
    if before:
        think = before[-1].group(1).strip() or None
        native = "\n\n".join(m.group(1).strip() for m in before[:-1])
    if think is None:
        warnings.append("no <think> block")

    if action and action_text:
        parts = [f"<think>{think}</think>"] if think else []
        parts.append(f"<{action}>{action_text}</{action}>")
        history_text = "\n".join(parts)
    else:
        history_text = text.strip()

    return Turn(think=think, action=action, action_text=action_text, text=history_text,
                native_reasoning=native, errors=errors, warnings=warnings)


# ── Passage labels and novelty ────────────────────────────────────────────────
# Every passage shown to the policy gets a label ``d1, d2, …`` that is unique in
# the run and stable: the same ``doc_id`` always gets the same label.  Novelty
# of a search is the fraction of its shown passages never shown before.

def _doc_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("doc_id") or doc.get("id") or "")


@dataclass
class StepNovelty:
    labels: List[str]
    novel_labels: List[str]

    @property
    def shown(self) -> int:
        return len(self.labels)

    @property
    def novel(self) -> int:
        return len(self.novel_labels)

    @property
    def nu(self) -> Optional[float]:
        return self.novel / self.shown if self.shown else None

    def to_dict(self) -> Dict[str, Any]:
        return {"labels": self.labels, "novel_labels": self.novel_labels,
                "shown": self.shown, "novel": self.novel, "nu": self.nu}


@dataclass
class DocRegistry:
    label_of: Dict[str, str] = field(default_factory=dict)   # doc_id -> dN
    doc_of: Dict[str, str] = field(default_factory=dict)     # dN -> doc_id

    def register(self, docs: List[Dict[str, Any]]) -> StepNovelty:
        labels: List[str] = []
        novel: List[str] = []
        for doc in docs:
            did = _doc_id(doc)
            if not did:
                continue
            label = self.label_of.get(did)
            if label is None:
                label = f"d{len(self.label_of) + 1}"
                self.label_of[did] = label
                self.doc_of[label] = did
                novel.append(label)
            if label not in labels:
                labels.append(label)
        return StepNovelty(labels=labels, novel_labels=novel)


def render_information(docs: List[Dict[str, Any]], registry: DocRegistry, max_chars: int) -> str:
    parts = []
    for doc in docs:
        label = registry.label_of.get(_doc_id(doc))
        if label is None:
            continue
        parts.append(f"[{label}] {doc_title(doc)}\n{doc_text(doc, max_length=max_chars)}")
    body = "\n\n".join(parts) if parts else "No documents retrieved."
    return f"<information>\n{body}\n</information>"


# ── Belief block ──────────────────────────────────────────────────────────────

def render_belief(step: int, novelty: Optional[StepNovelty],
                  summary: Optional[CriteriaCoverageSummary]) -> str:
    """The ``<belief>`` block the policy reads; either part may be omitted."""
    lines = [f'<belief step="{step}">']
    if novelty is not None:
        lines.append(f'  <novelty new="{novelty.novel}" shown="{novelty.shown}"/>')
    if summary is not None and summary.criteria:
        lines.append(f'  <criteria covered="{summary.num_covered}" partial="{summary.num_partial}" '
                     f'not_covered="{summary.num_not_covered}" total="{summary.total}">')
        for i, c in enumerate(summary.criteria, 1):
            lines.append(f'    <k{i} status="{c.status}">{escape(c.name, quote=False)}</k{i}>')
        lines.append("  </criteria>")
    lines.append("</belief>")
    return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

STOP_SEQUENCES = ["</search>", "</answer>"]

# Terminal step types.  "answer" is the normal end; the others are failures
# and are deliberately not in the trajectory evaluator's terminal set, so they
# count as runs that never finished.
END_ANSWERED = "answered"
END_MAX_TURNS = "max_turns"
END_FORMAT = "format_failure"
END_ERROR = "llm_error"


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 6)


def resolve_criteria_mode(mode: str, dataset: Optional[str]) -> str:
    """``auto`` -> static for BrowseComp-Plus (enumerated clues), else dynamic."""
    if mode == "auto":
        return "static" if dataset == "browsecomp_plus" else "dynamic"
    if mode not in MODES:
        raise ValueError(f"criteria mode must be auto, static or dynamic, got {mode!r}")
    return mode


@dataclass
class UncertaintyAwareAgentConfig:
    # Generation steps before the run counts as a failure.
    max_turns: int = 8
    # Characters of one passage shown inside <information>.
    max_passage_chars: int = 1500
    # Re-asks after a malformed turn before the run stops.
    max_format_retries: int = 2
    # Output-token cap for one policy turn.
    max_tokens_per_call: int = 4096
    # Turn the backbone's own reasoning mode off, so the only <think> in a
    # turn is the protocol's (also applied to the criteria updater).
    disable_native_thinking: bool = True
    # What the <belief> block shows (ablations).  With both off, no belief
    # is shown and the system prompt drops its belief section.
    show_novelty: bool = True
    show_criteria: bool = True
    # Criteria updater.
    criteria_mode: str = "dynamic"      # resolved: static | dynamic
    criteria_model: str = ""            # "" = the policy backbone
    max_criteria: int = 8
    stabilization_window: int = 15
    criteria_max_tokens: int = 1024
    evidence_top_k: int = 5
    evidence_chars: int = 1500


class UncertaintyAwareAgent(BasicAgent):
    AGENT_NAME = "UncertaintyAware"

    def __init__(self, llm_client, retriever: Optional[Any] = None, max_iteration: int = 100,
                 seen_top_k: int = 5, verbose: bool = True, max_turns: int = 8,
                 max_passage_chars: int = 1500, max_format_retries: int = 2,
                 max_tokens_per_call: int = 4096, disable_native_thinking: bool = True,
                 show_novelty: bool = True, show_criteria: bool = True,
                 criteria_mode: str = "auto", dataset: Optional[str] = None,
                 criteria_llm_client: Optional[Any] = None, criteria_model: str = "",
                 max_criteria: int = 8, stabilization_window: int = 15,
                 criteria_max_tokens: int = 1024, evidence_top_k: int = 5,
                 evidence_chars: int = 1500):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self.cfg = UncertaintyAwareAgentConfig(
            max_turns=max_turns,
            max_passage_chars=max_passage_chars,
            max_format_retries=max_format_retries,
            max_tokens_per_call=max_tokens_per_call,
            disable_native_thinking=disable_native_thinking,
            show_novelty=show_novelty,
            show_criteria=show_criteria,
            criteria_mode=resolve_criteria_mode(criteria_mode, dataset),
            criteria_model=criteria_model or "",
            max_criteria=max_criteria,
            stabilization_window=stabilization_window,
            criteria_max_tokens=criteria_max_tokens,
            evidence_top_k=evidence_top_k,
            evidence_chars=evidence_chars,
        )
        self.criteria_generator = criteria_llm_client
        self.tracker = CriteriaTracker(
            self._criteria_complete,
            mode=self.cfg.criteria_mode,
            max_criteria=max_criteria,
            stabilization_window=stabilization_window,
            evidence_top_k=evidence_top_k,
            evidence_chars=evidence_chars,
        )
        self.inference_config = InferenceConfig(api_type="chat_completion")
        self._extras: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    def _attach_uncertainty_stats(self, result: dict) -> None:
        # run_single's per-result hook, called before the trajectory log is
        # finalised; also used to attach the belief data.
        super()._attach_uncertainty_stats(result)
        result.update(self._extras)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _no_thinking_body(self, generator, backbone: bool) -> Optional[Dict[str, Any]]:
        """``extra_body`` that switches a model's own reasoning off.

        Passing ``extra_body`` per call replaces the configured one, so the
        configured body (OpenRouter provider pin) is copied and extended.  A
        separate criteria model gets the switch only on OpenRouter; other
        APIs may reject the vLLM ``chat_template_kwargs``.
        """
        if not self.cfg.disable_native_thinking:
            return None
        client = getattr(generator, "_client", generator)
        config = getattr(client, "config", None) or {}
        body = dict(config.get("extra_body") or {})
        if str(config.get("model", "")).startswith("openrouter/"):
            body["reasoning"] = {"enabled": False}
        elif backbone:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            return None
        return body

    def _call(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int,
              stop: Optional[List[str]] = None, generator=None, strip_think: bool = False) -> str:
        backbone = generator is None
        generator = generator or self.generator
        kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "strip_think": strip_think,
        }
        if stop:
            kwargs["stop"] = stop
        body = self._no_thinking_body(generator, backbone)
        if body is not None:
            kwargs["extra_body"] = body
        return generator.complete(messages, **kwargs) or ""

    def answer_from_trajectory(self, original_query: str, trajectory: Any, instruction: str) -> str:
        """Intermediate answer with the policy's call settings (native
        thinking off, per-turn token cap), greedy."""
        messages = self._intermediate_answer_messages(trajectory, instruction)
        return self._call(messages, 0.0, self.cfg.max_tokens_per_call)

    def _criteria_complete(self, messages: List[Dict[str, str]]) -> str:
        # The updater answers in JSON: greedy, and any <think> is stripped.
        return self._call(messages, 0.0, self.cfg.criteria_max_tokens,
                          generator=self.criteria_generator, strip_think=True)

    @staticmethod
    def _messages(system: str, transcript: str) -> List[Dict[str, str]]:
        # The whole run so far is one user message, as in the SearchR1 family.
        return [{"role": "system", "content": system},
                {"role": "user", "content": transcript}]

    def _generate_turn(self, messages, t, temperature) -> Tuple[Optional[Turn], int, Optional[str]]:
        """One turn, re-asked on format errors.

        Returns (turn, re-asks, error): turn is None and error the message when
        the LLM call fails.
        """
        extra: List[Dict[str, str]] = []
        turn: Optional[Turn] = None
        for attempt in range(self.cfg.max_format_retries + 1):
            try:
                raw = self._call(messages + extra, temperature, self.cfg.max_tokens_per_call,
                                 stop=STOP_SEQUENCES)
            except Exception as exc:
                logger.warning("Belief turn %d: LLM call failed: %s", t, exc)
                self._log_block(f"LLM call failed: {exc}", title=f"Turn {t}: error")
                return None, attempt, str(exc)
            turn = parse_turn(raw)
            if turn.ok:
                return turn, attempt, None
            self._log_block(
                "\n".join(f"- {e}" for e in turn.errors) + f"\n\n```\n{raw.strip()[:3000]}\n```",
                title=f"Turn {t}: format errors (attempt {attempt + 1})",
            )
            extra = [
                {"role": "assistant", "content": turn.text or raw},
                {"role": "user", "content": render_format_error(turn.errors)},
            ]
        return turn, self.cfg.max_format_retries, None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_criteria(self, summary: CriteriaCoverageSummary, title: str) -> None:
        if not summary.criteria:
            self._log_block(f"**No criteria:** {summary.error or 'empty list'}", title=title)
            return
        lines = ["| k | criterion | status | evidence |", "|---|---|---|---|"]
        changed = set(summary.changed_criteria_this_iter) | set(summary.new_criteria_this_iter)
        for i, c in enumerate(summary.criteria, 1):
            mark = " *" if c.name in changed else ""
            evidence = c.evidence.replace("|", "/").replace("\n", " ")
            lines.append(f"| k{i} | {c.name.replace('|', '/')} | {c.status}{mark} | {evidence} |")
        lines += ["", format_summary_for_log(summary)]
        if summary.removed_criteria_this_iter:
            lines.append("Removed: " + "; ".join(summary.removed_criteria_this_iter))
        self._log_block("\n".join(lines), title=title)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def inference(self, question: str, generation_temp: float = 0.6) -> tuple:
        cfg = self.cfg
        self._extras = {}
        show_belief = cfg.show_novelty or cfg.show_criteria

        # -- criteria (initial list, all not_covered) --
        self._notify_progress("criteria", 0)
        self.tracker.reset()
        init = self.tracker.initialize(question)
        self._log_criteria(init, title=f"Criteria ({cfg.criteria_mode})")
        self._vprint(0, "criteria", format_summary_for_log(init))

        system = render_system(cfg.max_turns, show_belief=show_belief)
        transcript = render_user(question)
        registry = DocRegistry()
        records: List[Dict[str, Any]] = []
        reasoning_path: List[Dict[str, Any]] = []
        last_summary: CriteriaCoverageSummary = init
        prediction = ""
        end = END_MAX_TURNS
        turns = 0
        step = 0

        for t in range(cfg.max_turns):
            self._notify_progress("turn", t)
            turn, attempts, error = self._generate_turn(self._messages(system, transcript), t, generation_temp)
            if turn is None:
                end = END_ERROR
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": END_ERROR, "errors": error,
                    "tokens": self._step_tokens(),
                })
                break
            turns = t + 1
            rec: Dict[str, Any] = {
                "turn": t,
                "valid": turn.ok,
                "errors": turn.errors,
                "warnings": turn.warnings,
                "format_retries": attempts,
                "action": turn.action,
                "action_text": turn.action_text,
                "native_reasoning_chars": len(turn.native_reasoning),
            }
            records.append(rec)
            if turn.warnings:
                self._log_block("; ".join(turn.warnings), title=f"Turn {t}: warnings")
            if turn.think:
                self._vprint(t, "think", turn.think)

            if not turn.ok:
                end = END_FORMAT
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": END_FORMAT, "think": turn.think,
                    "errors": "; ".join(turn.errors), "tokens": self._step_tokens(),
                })
                break

            if turn.action == ACTION_ANSWER:
                prediction = turn.action_text
                end = END_ANSWERED
                self._vprint(t, "answer", prediction)
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": "answer", "think": turn.think,
                    "prediction": prediction, "generation": prediction,
                    "tokens": self._step_tokens(),
                })
                break

            # -- search, then the belief that closes this iteration --
            query = turn.action_text
            self._notify_progress("search", t)
            self._vprint(t, "search", query)
            docs = self.retrieve_documents(query, original_query=question, reasoning=turn.think)
            shown = docs[:self.seen_top_k]
            novelty = registry.register(shown)
            information = render_information(shown, registry, cfg.max_passage_chars)
            self._vprint_docs(t, shown)
            self._record_step(reasoning_path, {
                "iteration": t, "action_type": "search", "think": turn.think,
                "search_query": query, "docs": docs,
                "component_doc_ids": [d.get("doc_id", "") for d in shown],
                "labels": " ".join(novelty.labels),
                "new_labels": " ".join(novelty.novel_labels),
                "novelty": _r(novelty.nu),
                "tokens": self._step_tokens(),
            })

            self._notify_progress("belief", t)
            summary = self.tracker.update(step, shown, [query], question)
            last_summary = summary
            belief = render_belief(step,
                                   novelty if cfg.show_novelty else None,
                                   summary if cfg.show_criteria else None)
            transcript += f"\n\n{turn.text}\n{information}\n"
            if show_belief:
                transcript += f"{belief}\n"

            rec.update({
                "step": step,
                "novelty": _r(novelty.nu),
                "novel_docs": novelty.novel,
                "shown_docs": novelty.shown,
                "search_novelty": novelty.to_dict(),
                "criteria": summary.to_dict(),
                "criteria_error": summary.error,
                "belief": belief if show_belief else None,
            })
            self._log_criteria(summary, title=f"Belief {step} · novelty {novelty.novel}/{novelty.shown}")
            self._vprint(t, "belief", f"novelty {novelty.novel}/{novelty.shown} · "
                                      + format_summary_for_log(summary))
            self._observe_step(
                query, shown, t, question,
                trajectory=self._messages(system, transcript),
            )
            step += 1

        if end == END_MAX_TURNS:
            self._record_step(reasoning_path, {"iteration": turns, "action_type": END_MAX_TURNS})

        final = last_summary
        self._extras = {
            "ua_criteria": [c.to_dict() for c in init.criteria],
            "ua_criteria_raw": init.raw,
            "ua_records": records,
            "ua_doc_labels": dict(registry.doc_of),
            "ua_outcome": {
                "end": end,
                "answer": prediction or None,
                "num_turns": turns,
                "num_searches": step,
                "format_retries": sum(r.get("format_retries", 0) for r in records),
                "criteria_mode": cfg.criteria_mode,
                "criteria_init_error": init.error,
                "criteria_errors": list(self.tracker.errors),
                "criteria_update_failures": sum(1 for r in records if r.get("criteria_error")),
                "final_criteria": [c.to_dict() for c in final.criteria],
                "final_covered": final.num_covered,
                "final_partial": final.num_partial,
                "final_not_covered": final.num_not_covered,
                "final_total": final.total,
                "mean_novelty": _r(_mean([r["novelty"] for r in records if r.get("novelty") is not None])),
                "config": asdict(cfg),
            },
        }
        return reasoning_path, prediction, turns


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None
