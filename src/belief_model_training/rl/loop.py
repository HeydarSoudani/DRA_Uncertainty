"""Stage rl: RL of the student belief model (GRPO-style over belief chains,
the PPO update of Agent-BRACE's BeliefStateLMTrainer; the agent is frozen).

The episodes are stage filter's (``filter/rl_episodes.jsonl``): the train
set from the ``rl`` split, the validation set from the ``val`` split
(every training dataset and agent).  Each iteration, on one GPU
shared by the vLLM engine and the trainer:

    sample   episodes_per_batch episodes (each epoch a new shuffled order)
    rollout  group_size chains per episode, all on the same fixed
             trajectory (``rl.rollout.run_chains``, temperature
             ``rl.temperature``): the inference belief model with the
             student's replies, from the gold beliefs (``rl.beliefs``
             gold) or from its own init call (own); then
             scorer_samples - 1 more replies to each chain's final-score
             call
    reward   per final-score reply r_j = gate_j (1 - (p_j - y)^2); per
             chain R = gate_chain * mean_j r_j (``reward``)
    groups   a group (the chains of one episode) whose R and r_j are all
             equal gives no gradient and is replaced by a new episode, up
             to ``refill_rounds`` more rollouts (dynamic sampling)
    update   advantages normalized within the group: every init and update
             call of chain c gets (R_c - mean R) / std R over the group's
             chains; every final-score reply gets (r_j - mean r) / std r
             over the group's replies -> engine.sleep -> trainer.onload ->
             PPO -> save to sync_dir -> trainer.offload ->
             engine.load_weights(sync_dir)

Validation, before training and every ``val_every`` iterations: one greedy
chain per val episode; s_T against y (AUROC, Brier) and the mean reward.
``final/`` holds the checkpoint with the best validation AUROC (Brier on a
tie; the start model when none beats it), the served
``{student}_rl`` / ``{student}_rl_own_init``.

Outputs (``{student}/rl/`` or ``{student}/rl_own_init/``): ``config.json``
(the settings and the filter's stats), ``metrics.jsonl`` (one line per
iteration), ``validation.jsonl``, ``step_{n}/`` every ``save_every``
iterations (the last ``save_total_limit`` kept), ``final/``.  A run starts
over: its earlier logs and checkpoints are removed.
"""

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..chain import SCORE, CallRecord, ChainResult
from ..config import RL_RUNS, RunPaths, TrainingConfig
from ..reward import chain_gate, mean, score_probability, score_reward
from ..filtering.run import load_selected_episodes, read_filter_stats
from ..runs import DocumentStore, Episode
from .rollout import VLLMRolloutEngine, extra_score_replies, run_chains
from .trainer import BeliefPPOTrainer, Sample

EPS = 1e-6


class Group:
    """The chains of one episode and their rewards."""

    def __init__(self, episode: Episode, chains: List[ChainResult], extras: List[List[CallRecord]]) -> None:
        self.episode = episode
        self.chains = chains
        # Per chain: its final-score replies (every attempt of its own call, then the extras).
        self.score_replies = [[c for c in ch.calls if c.kind == SCORE] + ex for ch, ex in zip(chains, extras)]
        self.score_rewards = [[score_reward(c.reply, episode.y, c.finish_reason) for c in replies]
                              for replies in self.score_replies]
        # The chain's reward over its K replies: its own first attempt and the extras.
        self.gates = [chain_gate(ch) for ch in chains]
        self.rewards = []
        for ch, gate, replies, rs in zip(chains, self.gates, self.score_replies, self.score_rewards):
            k_rewards = [r for c, r in zip(replies, rs) if c.attempt == 1]
            self.rewards.append(float(gate) * (mean(k_rewards) or 0.0))

    @property
    def has_spread(self) -> bool:
        flat = [r for rs in self.score_rewards for r in rs]
        return bool(np.std(self.rewards) > 1e-8 or (len(flat) > 1 and np.std(flat) > 1e-8))

    def samples(self) -> Tuple[List[Sample], List[List[int]]]:
        """The trained calls of the group and the sample indices of each chain."""
        R = np.array(self.rewards)
        chain_adv = (R - R.mean()) / (R.std() + EPS) if R.std() > 1e-8 else np.zeros_like(R)
        flat = np.array([r for rs in self.score_rewards for r in rs])
        score_mean, score_std = (flat.mean(), flat.std()) if len(flat) else (0.0, 0.0)
        samples: List[Sample] = []
        units: List[List[int]] = []
        for ch, adv, replies, rs in zip(self.chains, chain_adv, self.score_replies, self.score_rewards):
            unit = []
            for c in ch.calls:
                if c.kind != SCORE and c.response_ids:
                    unit.append(len(samples))
                    samples.append(Sample(c.prompt_ids, c.response_ids, c.logprobs, float(adv)))
            for c, r in zip(replies, rs):
                if c.response_ids and score_std > 1e-8:
                    unit.append(len(samples))
                    samples.append(Sample(c.prompt_ids, c.response_ids, c.logprobs,
                                          float((r - score_mean) / (score_std + EPS))))
            units.append(unit)
        return samples, units


def rollout_groups(episodes: List[Episode], engine, store: DocumentStore, cfg: TrainingConfig) -> List[Group]:
    rc = cfg.rl
    jobs = [(ep, rc.beliefs) for ep in episodes for _ in range(rc.group_size)]
    chains = run_chains(jobs, engine, store, rc.temperature)
    extras = extra_score_replies(chains, engine, rc.scorer_samples - 1, rc.temperature)
    G = rc.group_size
    return [Group(ep, chains[k * G:(k + 1) * G], extras[k * G:(k + 1) * G]) for k, ep in enumerate(episodes)]


def rollout_stats(groups: List[Group]) -> Dict[str, Any]:
    chains = [ch for g in groups for ch in g.chains]
    calls = [c for ch in chains for c in ch.calls]
    replies = [c for g in groups for rs in g.score_replies for c in rs]
    n = max(len(calls), 1)
    return {
        "groups": len(groups),
        "groups_with_spread": sum(g.has_spread for g in groups),
        "chains": len(chains),
        "chain_errors": sum(ch.error is not None for ch in chains),
        "calls_per_chain": len(calls) / max(len(chains), 1),
        "truncated_rate": sum(c.finish_reason == "length" for c in calls) / n,
        "prompt_too_long": sum(c.finish_reason == "prompt_too_long" for c in calls),
        "mean_prompt_tokens": sum(len(c.prompt_ids) for c in calls) / n,
        "mean_response_tokens": sum(len(c.response_ids) for c in calls) / n,
        "gate_rate": mean(float(x) for g in groups for x in g.gates),
        "score_parse_rate": mean(float(score_probability(c.reply, c.finish_reason) is not None) for c in replies),
        "reward": mean(r for g in groups for r in g.rewards),
        "y_rate": mean(float(g.episode.y) for g in groups),
    }


def validate(val: List[Episode], engine, store: DocumentStore, cfg: TrainingConfig) -> Dict[str, Any]:
    """Greedy chains on the val episodes: s_T against y."""
    from evaluation.measures import auroc, brier

    chains = run_chains([(ep, cfg.rl.beliefs) for ep in val], engine, store, temperature=0.0)
    ys, ss, rewards, gates = [], [], [], []
    for ch in chains:
        gate = chain_gate(ch)
        gates.append(float(gate))
        first = next((c for c in ch.calls if c.kind == SCORE and c.attempt == 1), None)
        r = score_reward(first.reply, ch.episode.y, first.finish_reason) if first else 0.0
        rewards.append(float(gate) * r)
        if ch.s_T is not None:
            ys.append(float(ch.episode.y))
            ss.append(float(ch.s_T))
    out: Dict[str, Any] = {"n": len(chains), "with_s_T": len(ss), "reward": mean(rewards), "gate_rate": mean(gates)}
    if ss:
        y, s = np.array(ys), np.array(ss)
        out.update(brier=float(brier(y, s)), mean_s_T=float(s.mean()),
                   auroc=float(auroc(y, s)) if len(set(ys)) == 2 else None)
    return out


def _better(v: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    def key(m):
        a = m.get("auroc")
        b = m.get("brier")
        return (a if a is not None else -1.0, -(b if b is not None else 1.0))
    return best is None or key(v) > key(best)


class EpisodeSampler:
    """Draws the episodes of a batch without replacement within an epoch
    (a new shuffled order each epoch)."""

    def __init__(self, episodes: List[Episode], seed: int) -> None:
        self.episodes = episodes
        self.rng = random.Random(seed)
        self.order: List[int] = []

    def next(self, n: int) -> List[Episode]:
        out = []
        while len(out) < n:
            if not self.order:
                self.order = list(range(len(self.episodes)))
                self.rng.shuffle(self.order)
            out.append(self.episodes[self.order.pop()])
        return out


def _check_prompt_tokens(engine, processor, episode: Episode, store: DocumentStore) -> None:
    """The trainer's (SFT) tokenization of a prompt equals the engine's."""
    from ..chain import ReplayClient, run_chain
    from ..student import chat_prompt, text_tokenizer

    client = ReplayClient({})
    run_chain(episode, client, store)
    if not client.calls:
        return
    messages = client.calls[0].messages
    ours = text_tokenizer(processor)(chat_prompt(processor, messages), add_special_tokens=False)["input_ids"]
    if ours != engine.prompt_ids(messages):
        raise SystemExit("[stage rl] the SFT tokenization of a prompt differs from the vLLM engine's; "
                         "check the chat template of the student checkpoint")


def run_rl(cfg: TrainingConfig) -> None:
    from ..student import load_processor, quiet_libraries

    rc, sh = cfg.rl, cfg.shared
    paths = RunPaths(sh.student)
    run_name = f"{sh.student}_{RL_RUNS[rc.beliefs]}"
    run_dir = paths.rl_dir(rc.beliefs)
    final_dir = paths.rl_final(rc.beliefs)
    init = Path(rc.init_model) if rc.init_model else paths.sft_final
    if not init.exists():
        raise SystemExit(f"[stage rl] start model {init} not found: run --stage sft first, or pass --init-model")

    filter_stats = read_filter_stats(cfg, paths, "[stage rl]")
    episodes = load_selected_episodes(cfg, paths, "train", tag="[stage rl] rl")
    val = load_selected_episodes(cfg, paths, "val", tag="[stage rl] val")
    if not episodes:
        raise SystemExit("[stage rl] no rl episode: run --stage generate on the rl split, then --stage filter")
    random.Random(sh.seed).shuffle(val)
    if rc.val_max_episodes is not None:
        val = val[:rc.val_max_episodes]

    # A run starts over.
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.glob("step_*"):
        shutil.rmtree(p, ignore_errors=True)
    metrics_file, val_file = run_dir / "metrics.jsonl", run_dir / "validation.jsonl"
    for f in (metrics_file, val_file):
        f.unlink(missing_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"config": cfg.to_dict(), "filter": filter_stats}, indent=2, default=str) + "\n")

    quiet_libraries()
    store = DocumentStore()
    processor = load_processor(str(init))
    engine = VLLMRolloutEngine(str(init), gpu_memory_utilization=rc.vllm_gpu_memory_utilization,
                               max_model_len=sh.max_seq_len, max_num_seqs=rc.vllm_max_num_seqs, seed=sh.seed)
    _check_prompt_tokens(engine, processor, episodes[0], store)
    # The trainer's weights stay on the CPU until its first update (onload).
    trainer = BeliefPPOTrainer(str(init), rc, seed=sh.seed)
    sampler = EpisodeSampler(episodes, sh.seed)
    print(f"[stage rl] run {run_name} ({rc.beliefs} beliefs) from {init}: {len(episodes)} rl episodes, "
          f"{len(val)} val episodes; {rc.episodes_per_batch} episodes x {rc.group_size} chains x "
          f"{rc.scorer_samples} final scores per iteration", flush=True)

    def log(path: Path, row: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def run_validation(iteration: int, best: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not val:
            return best
        v = validate(val, engine, store, cfg)
        v["iteration"] = iteration
        log(val_file, v)
        print(f"[stage rl] validation {iteration}: {json.dumps(v)}", flush=True)
        if _better(v, best):
            shutil.rmtree(final_dir, ignore_errors=True)
            trainer.save(str(final_dir), processor)
            print(f"[stage rl] new best (AUROC {v.get('auroc')}, Brier {v.get('brier')}): saved {final_dir}")
            return v
        return best

    best = run_validation(0, None)
    if best is None:
        shutil.rmtree(final_dir, ignore_errors=True)
        trainer.save(str(final_dir), processor)
    saved_steps: List[Path] = []
    for iteration in range(1, rc.num_iterations + 1):
        t0 = time.time()
        groups = [g for g in rollout_groups(sampler.next(rc.episodes_per_batch), engine, store, cfg)]
        all_groups = list(groups)
        for _ in range(rc.refill_rounds):
            missing = sum(not g.has_spread for g in groups)
            if not missing:
                break
            groups = [g for g in groups if g.has_spread]
            new = rollout_groups(sampler.next(missing), engine, store, cfg)
            all_groups += new
            groups += new
        groups = [g for g in groups if g.has_spread]
        t_rollout = time.time() - t0

        samples: List[Sample] = []
        units: List[List[int]] = []
        for g in groups:
            s, u = g.samples()
            units += [[i + len(samples) for i in unit] for unit in u]
            samples += s
        stats: Dict[str, Any] = {}
        if samples:
            engine.sleep()
            trainer.onload()
            stats = trainer.update(samples, units)
            shutil.rmtree(rc.sync_dir, ignore_errors=True)
            trainer.save(rc.sync_dir)
            trainer.offload()
            engine.load_weights(rc.sync_dir)
        row = {"iteration": iteration, **rollout_stats(all_groups), "trained_groups": len(groups),
               **stats, "rollout_seconds": round(t_rollout, 1), "seconds": round(time.time() - t0, 1)}
        log(metrics_file, row)
        print(f"[stage rl] iteration {iteration}: reward {row.get('reward')}, gate {row.get('gate_rate')}, "
              f"groups with spread {row['groups_with_spread']}/{row['groups']}, samples {row.get('n_samples', 0)}, "
              f"{row['seconds']}s", flush=True)

        if iteration % rc.val_every == 0 or iteration == rc.num_iterations:
            best = run_validation(iteration, best)
        if iteration % rc.save_every == 0:
            step_dir = run_dir / f"step_{iteration}"
            trainer.save(str(step_dir), processor)
            saved_steps.append(step_dir)
            while len(saved_steps) > rc.save_total_limit:
                shutil.rmtree(saved_steps.pop(0), ignore_errors=True)
    print(f"[stage rl] done; best validation {json.dumps(best)}; {final_dir} is served as {run_name}")
