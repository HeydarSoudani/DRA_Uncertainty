"""The inference runs of phase 1, read back as episodes by phases 2 and 3.

Phase 1 runs ``experiments/inference/dra_inference.py`` on one split of the
training pool, with the standard output layout:

    {inference_output}/{dataset}_{split}_{retriever}/{agent dir}/belief-{mode}/

    split  mode      used by
    sft    monitor   phase 2 (SFT samples, teacher scores)
    val    monitor   phase 2 (validation loss, teacher scores), phase 3 (validation)
    rl     off       phase 3 (RL prompts)

``load_episodes`` gives the episodes of a split's run: the trajectories
(``episodes.read_episode``), the gold of the qrels (``annotate_gold``) and
the correctness of the run's ``accuracy.jsonl``; with ``teacher``, also the
teacher's view (``attach_teacher``), after checking that the run's belief
model is the teacher.  Phase 2 replays the teacher (``replay_teacher``).

``check_same_settings`` checks that the runs a phase reads share their agent
and searcher settings (``run_config.json``, the temperature included), so
that their trajectories come from the same agent: phase 2 checks sft and
val, phase 3 rl and val.  The run settings are read from ``run_config.json``
only (the per-query files carry none).
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from indexing_corpus_dataset.dataset_loaders import load_split
from indexing_corpus_dataset.layout import DATASET_SPECS, dataset_root
from utils.config import POLICY_MODEL_TO_LLM
from utils.io_utils import build_belief_config_name, build_dataset_dir_name, run_config_diffs, scan_queries

from .config import Split, TrainingConfig
from .episodes import Episode, annotate_gold, attach_teacher, read_episode

logger = logging.getLogger(__name__)

# The belief mode of each split's run.
SPLIT_MODE = {"sft": "monitor", "val": "monitor", "rl": "off"}


# Sections of run_config.json that must be equal across the runs of a phase.
SAME_SETTINGS = ("agent", "searcher")


def run_config(run_dir: Path) -> Dict:
    return json.loads((run_dir / "run_config.json").read_text())


def _is_agent_run(config: Dict, policy_model: str) -> bool:
    """The run's agent is *policy_model* (the --policy-model of
    dra_inference, e.g. ``oss_20b``, as ``run_config.json`` records it)
    with its LLM."""
    agent = config.get("agent") or {}
    return agent.get("policy_model") == policy_model and agent.get("llm_model") == POLICY_MODEL_TO_LLM.get(policy_model)


# find_run's result per (split root, belief dir, agent): run_config.json is parsed once.
_RUNS: Dict[Tuple[Path, str, str], Tuple[Path, Dict]] = {}


def find_run(cfg: TrainingConfig, split: Split) -> Tuple[Path, Dict]:
    """The run directory of *split* for the agent ``shared.policy_model``
    (``run_config.json``'s agent) in the split's mode, named as
    dra_inference names it, and its ``run_config.json``."""
    dataset_dir = build_dataset_dir_name(
        split.dataset, split.split_id, DATASET_SPECS[split.dataset].query_key, cfg.shared.retriever,
    )
    root = cfg.inference_output / dataset_dir
    belief_dir = build_belief_config_name(belief_mode=SPLIT_MODE[split.subset])
    key = (root, belief_dir, cfg.shared.policy_model)
    if key not in _RUNS:
        configs = [(d, run_config(d)) for d in sorted(root.glob(f"*/{belief_dir}")) if (d / "run_config.json").exists()]
        found = [(d, c) for d, c in configs if _is_agent_run(c, cfg.shared.policy_model)]
        if len(found) != 1:
            raise FileNotFoundError(f"expected one {belief_dir} run of {cfg.shared.policy_model} under {root}, "
                                    f"found {[d for d, _ in found]}; run phase 1 with --split {split.subset}")
        _RUNS[key] = found[0]
    return _RUNS[key]


def check_same_settings(cfg: TrainingConfig, names: List[str]) -> None:
    """The runs of splits *names* have the same agent and searcher
    settings; raises with the keys that differ."""
    configs = {name: find_run(cfg, cfg.pool_split(name))[1] for name in names}
    first = names[0]
    diffs = [f"{first} -> {name}: {d}" for name in names[1:]
             for d in run_config_diffs(configs[first], configs[name], SAME_SETTINGS)]
    if diffs:
        raise SystemExit(f"the runs of {', '.join(names)} differ in their agent / searcher settings "
                         f"(run_config.json): {'; '.join(diffs)}; rerun phase 1 with the same inference config")


def criteria_cap(cfg: TrainingConfig, split: Split) -> Optional[int]:
    """The criteria cap of the init prompt for the episodes of *split*: the
    ``belief_model.max_criteria`` of the teacher's monitor run (the split's
    own, the val run's for rl, which has no belief model), so every split
    uses the same; None (the dataset's default) when no monitor run has it."""
    name = split.subset if SPLIT_MODE[split.subset] == "monitor" else "val"
    try:
        config = find_run(cfg, cfg.pool_split(name))[1]
    except FileNotFoundError:
        return None
    return (config.get("belief_model") or {}).get("max_criteria")


def check_teacher(run_dir: Path, config: Dict, teacher: str) -> None:
    """The run's criteria and updater model is *teacher*."""
    belief = config["belief_model"]
    criteria = belief.get("llm_criteria")
    judge = belief.get("belief_judge_model")
    if criteria != teacher or judge != teacher:
        raise SystemExit(f"{run_dir}: belief model {criteria!r} / updater {judge!r}, "
                         f"expected the teacher {teacher!r} (trajectories.teacher_belief_model)")


def split_gold(split: Split, config: Dict) -> Dict[str, Set[str]]:
    """The queries of *split* that dra_inference runs (those with qrels, in
    file order) and their relevant documents, as dra_inference read them
    with the relevance threshold of its ``run_config.json`` *config*
    (the dataset's default when absent)."""
    dataset = config.get("dataset") or {}
    min_score = (dataset["min_relevance_score"] if "min_relevance_score" in dataset
                 else DATASET_SPECS[split.dataset].min_relevance_score)
    queries, qrels, _ = load_split(
        dataset_root(split.dataset), split.split_id, query_key=DATASET_SPECS[split.dataset].query_key,
        min_relevance_score=min_score, only_with_qrels=True,
    )
    return {qid: {d for d, r in qrels[qid].items() if float(r) > 0} for qid in queries}


def read_correctness(run_dir: Path) -> Dict[str, Optional[bool]]:
    """Per query of the run, its final answer is correct (``correct`` of the
    LLM judge, or ``exact_match``); None when the judge failed.  A query
    missing here was not evaluated (``load_episodes`` warns)."""
    path = run_dir / "accuracy.jsonl"
    if not path.exists():
        logger.warning("%s not found: no correctness", path)
        return {}
    out: Dict[str, Optional[bool]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if "query_id" not in row:
            continue
        label = row.get("correct", row.get("exact_match"))
        out[str(row["query_id"])] = None if row.get("judge_error") or label is None else bool(label)
    return out


def load_episodes(cfg: TrainingConfig, name: str, *, teacher: bool, tag: str = "") -> List[Episode]:
    """The episodes of split *name*'s run, one per finished query.  With
    *teacher* (a monitor run), each also has the teacher's view; an episode
    whose belief file cannot be used is left out."""
    split = cfg.pool_split(name)
    run_dir, config = find_run(cfg, split)
    mode = SPLIT_MODE[split.subset]
    if teacher:
        check_teacher(run_dir, config, cfg.trajectories.teacher_belief_model)
    max_criteria = criteria_cap(cfg, split)
    gold = split_gold(split, config)
    correctness = read_correctness(run_dir)
    # Finished as dra_inference counts them (with the belief file when the belief model is on).
    finished, _ = scan_queries(run_dir, require_belief=mode != "off")
    episodes: List[Episode] = []
    counts: Counter = Counter()
    for qid in gold:
        episode = read_episode(run_dir / "trajectory" / f"{qid}.jsonl", split, run_dir) if qid in finished else None
        if episode is None:
            counts["not_run"] += 1
            continue
        episode.policy_model = cfg.shared.policy_model
        episode.max_criteria = max_criteria
        if teacher:
            reason = attach_teacher(episode, run_dir / "belief" / f"{qid}.jsonl")
            if reason is not None:
                counts["teacher_unusable"] += 1
                logger.warning("%s %s %s: %s", tag, split.name, qid, reason)
                continue
        gold_error = annotate_gold(episode, gold[qid])
        if gold_error is not None:
            counts["gold_mismatch"] += 1
            logger.warning("%s %s %s: gold: %s", tag, split.name, qid, gold_error)
        if correctness and qid not in correctness:
            counts["not_evaluated"] += 1
        episode.correct = correctness.get(qid)
        episodes.append(episode)
    print(f"{tag} {split.name} ({run_dir}): {len(gold)} queries, {dict(counts)}")
    if counts["not_evaluated"]:
        logger.warning("%s %s: %d episodes have no verdict in %s/accuracy.jsonl (the run stopped before its "
                       "evaluation?); rerun dra_inference with --eval-only to judge them",
                       tag, split.name, counts["not_evaluated"], run_dir)
    print_statistics(episodes, f"{tag} {name}")
    return episodes


def print_statistics(episodes: List[Episode], tag: str) -> None:
    """Episodes, the share with a relevant document found, the correctness
    of the final answers, steps (and teacher calls) per episode, sources."""
    if not episodes:
        print(f"{tag}: no episodes")
        return
    n = len(episodes)
    correct = Counter(ep.correct for ep in episodes)
    steps = sum(len(ep.steps) for ep in episodes) / n
    calls = sum(s.teacher_reply is not None for ep in episodes for s in ep.steps) / n
    teacher = f", {calls:.1f} teacher calls" if any(ep.criteria for ep in episodes) else ""
    print(f"{tag}: {n} episodes; with gold {sum(ep.has_gold for ep in episodes) / n:.0%}; "
          f"correct {correct[True]}, wrong {correct[False]}, not judged {correct[None]}; "
          f"{steps:.1f} steps{teacher} per episode; by source {dict(Counter(ep.source for ep in episodes))}")
