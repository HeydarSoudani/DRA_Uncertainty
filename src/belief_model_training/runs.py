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
from typing import Dict, List, Optional, Set

from indexing_corpus_dataset.dataset_loaders import load_split
from indexing_corpus_dataset.layout import DATASET_SPECS, dataset_root
from utils.io_utils import build_belief_config_name, build_dataset_dir_name, run_config_diffs

from .config import Split, TrainingConfig
from .episodes import Episode, annotate_gold, attach_teacher, read_episode

logger = logging.getLogger(__name__)

# The belief mode of each split's run.
SPLIT_MODE = {"sft": "monitor", "val": "monitor", "rl": "off"}


# Sections of run_config.json that must be equal across the runs of a phase.
SAME_SETTINGS = ("agent", "searcher")


def run_config(run_dir: Path) -> Dict:
    return json.loads((run_dir / "run_config.json").read_text())


def _run_agent(config: Dict) -> Optional[str]:
    """The --policy-model of a run."""
    return (config.get("agent") or {}).get("policy_model")


def find_run_dir(cfg: TrainingConfig, split: Split) -> Path:
    """The run directory of *split* for the agent ``shared.policy_model``
    (``run_config.json``'s agent) in the split's mode, named as
    dra_inference names it."""
    dataset_dir = build_dataset_dir_name(
        split.dataset, split.split_id, DATASET_SPECS[split.dataset].query_key, cfg.shared.retriever,
    )
    root = cfg.inference_output / dataset_dir
    belief_dir = build_belief_config_name(belief_mode=SPLIT_MODE[split.subset])
    found = [d for d in sorted(root.glob(f"*/{belief_dir}"))
             if (d / "run_config.json").exists() and _run_agent(run_config(d)) == cfg.shared.policy_model]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {belief_dir} run of {cfg.shared.policy_model} under {root}, "
                                f"found {found}; run phase 1 with --split {split.subset}")
    return found[0]


def check_same_settings(cfg: TrainingConfig, names: List[str]) -> None:
    """The runs of splits *names* have the same agent and searcher
    settings; raises with the keys that differ."""
    runs = {name: find_run_dir(cfg, cfg.pool_split(name)) for name in names}
    configs = {name: run_config(d) for name, d in runs.items()}
    first = names[0]
    diffs = [f"{first} -> {name}: {d}" for name in names[1:]
             for d in run_config_diffs(configs[first], configs[name], SAME_SETTINGS)]
    if diffs:
        raise SystemExit(f"the runs of {', '.join(names)} differ in their agent / searcher settings "
                         f"(run_config.json): {'; '.join(diffs)}; rerun phase 1 with the same inference config")


def check_teacher(run_dir: Path, teacher: str) -> None:
    """The run's criteria and updater model is *teacher*."""
    belief = run_config(run_dir)["belief_model"]
    criteria = belief.get("llm_criteria")
    judge = belief.get("belief_judge_model")
    if criteria != teacher or judge != teacher:
        raise SystemExit(f"{run_dir}: belief model {criteria!r} / updater {judge!r}, "
                         f"expected the teacher {teacher!r} (trajectories.teacher_belief_model)")


def split_gold(split: Split) -> Dict[str, Set[str]]:
    """The queries of *split* that dra_inference runs (those with qrels, in
    file order) and their relevant documents, as dra_inference reads them."""
    queries, qrels, _ = load_split(
        dataset_root(split.dataset), split.split_id, query_key=DATASET_SPECS[split.dataset].query_key,
        min_relevance_score=DATASET_SPECS[split.dataset].min_relevance_score, only_with_qrels=True,
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
    run_dir = find_run_dir(cfg, split)
    if teacher:
        check_teacher(run_dir, cfg.trajectories.teacher_belief_model)
    max_criteria = run_config(run_dir)["belief_model"].get("max_criteria")
    gold = split_gold(split)
    correctness = read_correctness(run_dir)
    episodes: List[Episode] = []
    counts: Counter = Counter()
    for qid in gold:
        path = run_dir / "trajectory" / f"{qid}.jsonl"
        episode = read_episode(path, split, run_dir) if path.exists() else None
        if episode is None:
            counts["not_run"] += 1
            continue
        episode.policy_model = cfg.shared.policy_model
        episode.max_criteria = max_criteria
        if teacher:
            belief = run_dir / "belief" / f"{qid}.jsonl"
            reason = attach_teacher(episode, belief) if belief.exists() else "no belief file"
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
