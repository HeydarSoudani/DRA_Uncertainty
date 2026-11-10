"""The inference runs of phase 1, read back as episodes by phases 2 and 3.

Phase 1 runs ``experiments/inference/dra_inference.py`` on one split of the
training pool, with the standard output layout:

    {inference_output}/{dataset}_{split}_{retriever}/{agent dir}/belief-{mode}[-{source}]/

(``-{source}``: ``shared.criteria_source`` on trqa_validation, where it is a
choice; deep_qa's monitor runs are ``belief-monitor``.)

    split  mode      used by
    sft    monitor   phase 2 (SFT samples, teacher scores)
    val    monitor   phase 2 (validation loss, teacher scores), phase 3 (validation)
    rl     off       phase 3 (RL prompts)

``load_episodes`` gives the episodes of a split's run: the trajectories
(``episodes.read_episode``), the gold of the qrels (``annotate_gold``), the
correctness of the run's ``accuracy.jsonl`` and the pool's gold criteria
(``gold_criteria``; training uses only the queries with the full gold
annotation and criterion texts: the out-of-domain queries of ``val``, whose
criteria the teacher wrote, are for evaluation and left out, as is a query
whose texts failed) with the criteria each agent turn states the gold
answer of (``nugget_gold.annotate_stated``); with ``teacher``, also the
teacher's view (``attach_teacher``), after checking that the run's belief
model is the teacher on the gold criteria.  Phase 2 replays the teacher
(``replay_teacher``).

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
from indexing_corpus_dataset.layout import (
    DATASET_SPECS, GOLD_CRITERIA, criteria_run_tag, dataset_root, effective_criteria_source, gold_criteria_path,
)
from utils.config import POLICY_MODEL_TO_LLM
from utils.io_utils import build_belief_config_name, build_dataset_dir_name, run_config_diffs, scan_queries

from belief_model_component.criteria import GOLD_CRITERIA_LEVEL, GOLD_CRITERIA_MODEL, shape_kinds
from belief_model_component.types import MULTI_ASPECT

from .config import RunPaths, Split, TrainingConfig
from .episodes import Episode, annotate_gold, attach_teacher, read_episode
from .nugget_gold import AnswerJudge, NuggetGold, annotate_stated

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
    # The monitor runs name their criteria source where it is a choice; the off run has none.
    belief_dir = build_belief_config_name(SPLIT_MODE[split.subset],
                                          criteria_run_tag(cfg.shared.criteria_source, split.dataset))
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


def check_teacher(run_dir: Path, config: Dict, teacher: str) -> None:
    """The run's updater model is *teacher*, on the gold criteria."""
    belief = config["belief_model"]
    judge = belief.get("belief_judge_model")
    source = belief.get("criteria_source_effective")
    if judge != teacher or source != GOLD_CRITERIA:
        raise SystemExit(f"{run_dir}: updater {judge!r}, criteria source {source!r}; expected "
                         f"the teacher {teacher!r} (trajectories.teacher_belief_model) on the gold criteria")


def require_gold(cfg: TrainingConfig) -> None:
    """Phases 2 and 3 score the belief against the pool's gold criteria
    (gold answers and documents per criterion id): they read the runs on
    the gold criteria only."""
    source = effective_criteria_source(cfg.shared.criteria_source, cfg.shared.training_data)
    if source != GOLD_CRITERIA:
        raise SystemExit(f"shared.criteria_source {source}: phases 2 and 3 score against the pool's gold "
                         f"criteria and need the runs on them (shared.criteria_source gold); silver runs are "
                         f"for phase 1 and evaluation only")


def read_gold_criteria(split: Split) -> Tuple[Dict[str, Dict], Set[str]]:
    """The gold criteria rows of the queries of *split* with the full
    annotation and criteria with texts (``texts_ok``), by query id
    (``layout.gold_criteria_path``: the pool's rows with that ``split``),
    and the ids of the out-of-domain queries (not full)."""
    path = gold_criteria_path(split.dataset)
    if not path.exists():
        raise SystemExit(f"{path} not found: build the pool (build_training_pool.py: deep_qa step 2, or trqa_validation)")
    out, ood = {}, set()
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if not row or row.get("split") != split.split_id:
            continue
        if row.get("level") != GOLD_CRITERIA_LEVEL:
            ood.add(str(row["id"]))
        elif row.get("texts_ok") and row.get("criteria"):
            out[str(row["id"])] = row
    return out, ood


_JUDGES: Dict[str, AnswerJudge] = {}
_GOLD: Dict[Optional[str], NuggetGold] = {}


def answer_judge(cfg: TrainingConfig) -> Optional[AnswerJudge]:
    """The judge of the gold of the belief (``rl.answer_judge_model``;
    None: matching only), its cache under the training run."""
    model = cfg.rl.answer_judge_model
    if not model:
        return None
    if model not in _JUDGES:
        _JUDGES[model] = AnswerJudge(model, RunPaths(cfg).root / "answer_gold.jsonl")
    return _JUDGES[model]


def nugget_gold(cfg: TrainingConfig) -> NuggetGold:
    """The gold of the belief's values, one per judge model."""
    model = cfg.rl.answer_judge_model or None
    if model not in _GOLD:
        _GOLD[model] = NuggetGold(answer_judge(cfg))
    return _GOLD[model]


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


# Before the shared cut (schema 38) the belief update cut a short-answer
# document at 3000 characters and read a report document whole.
LEGACY_PASSAGE_CHARS = 3000


def passage_chars(config: Dict, query_shape: str) -> Optional[int]:
    """Characters of one document the belief update of a run read (or, a
    run without the belief model, will read in training), from its
    ``run_config.json`` *config*: the belief section's
    ``max_passage_chars``, else the agent's (the shared cut, the same
    value), else the cut of the runs before it was recorded."""
    for section in ("belief_model", "agent"):
        values = config.get(section) or {}
        if "max_passage_chars" in values:
            return values["max_passage_chars"]
    return None if query_shape == MULTI_ASPECT else LEGACY_PASSAGE_CHARS


def load_episodes(cfg: TrainingConfig, name: str, *, teacher: bool, tag: str = "") -> List[Episode]:
    """The episodes of split *name*'s run, one per finished query.  With
    *teacher* (a monitor run), each also has the teacher's view; an episode
    whose belief file cannot be used is left out."""
    require_gold(cfg)
    split = cfg.pool_split(name)
    run_dir, config = find_run(cfg, split)
    mode = SPLIT_MODE[split.subset]
    if teacher:
        check_teacher(run_dir, config, cfg.trajectories.teacher_belief_model)
    gold = split_gold(split, config)
    gold_criteria, ood = read_gold_criteria(split)
    correctness = read_correctness(run_dir)
    # Finished as dra_inference counts them (with the belief file when the belief model is on).
    finished, _ = scan_queries(run_dir, require_belief=mode != "off")
    episodes: List[Episode] = []
    counts: Counter = Counter()
    for qid in gold:
        if qid not in gold_criteria:
            counts["out_of_domain" if qid in ood else "no_gold_criteria"] += 1
            continue
        episode = read_episode(run_dir / "trajectory" / f"{qid}.jsonl", split, run_dir) if qid in finished else None
        if episode is None:
            counts["not_run"] += 1
            continue
        episode.max_passage_chars = passage_chars(config, episode.query_shape)
        # The kinds as the belief reads them (a multi-hop query's hops, ``shape_kinds``).
        rows = gold_criteria[qid]["criteria"]
        kinds = shape_kinds(episode.query_shape, [c["kind"] for c in rows])
        episode.gold_criteria = [{**c, "kind": kind} for c, kind in zip(rows, kinds)]
        episode.source = gold_criteria[qid]["source"]
        episode.gold_names = gold_criteria[qid]["entity_names"]
        if teacher:
            reason = attach_teacher(episode, run_dir / "belief" / f"{qid}.jsonl")
            if reason is None and episode.criteria_model != GOLD_CRITERIA_MODEL:
                reason = f"criteria by {episode.criteria_model!r}, not the gold criteria"
            if reason is None and [c["text"] for c in episode.criteria] != [c["text"] for c in episode.gold_criteria]:
                reason = "the run's criteria differ from the pool's gold criteria"
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
    print(f"{tag} {split.name}: gold answers stated by the agent turns {annotate_stated(episodes, answer_judge(cfg))}")
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
