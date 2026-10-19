"""Phase 1: teacher trajectories.

``experiments/inference/dra_inference.py`` runs the frozen agent with the teacher belief
model in ``monitor`` mode (the agent never sees the belief) on the training
pool ``shared.training_data`` (``experiments/training/build_training_pool.py``:
deep_qa, split ``train``, or trqa_validation, split ``val``): every pool query
with qrels, or the next ``limit`` of them.
The teacher writes the criteria (init, read from the criteria bank when
banked) and the belief updates; the query scorer is the frozen
``query_scorer_model``.  Inference resume applies: queries already run are
skipped, so a call after the pool has grown runs only the new questions.
Then every run's ``belief/{qid}.jsonl`` becomes an episode, with the gold of
the phase 3 rewards (``episodes.annotate_gold``: the new relevant documents
of each step, checked against the saved ``recall_so_far``; the correctness
of the final answer from the run's ``accuracy.jsonl``) and its use, ``sft``,
``rl`` or ``val`` (``episodes.assign_use``).  The ``sft`` episodes give SFT
samples of both calls, the ``val`` episodes the SFT validation samples:

    init     one per episode whose criteria the teacher wrote: the criteria
             extraction prompt, the criteria as the reply
             (``episodes.teacher_init_call``)
    update   the teacher's replies, replayed (``episodes.replay_teacher``):
             each call that parsed (and, with ``require_verified_spans``,
             whose cited spans all occur in their passage)

The teacher is scored on every episode (``scoring.assess``: cited support
against the new gold documents, coverage against ``recall_so_far``, final
coverage against correctness), so its quality is known before SFT.

Outputs (``phase1_trajectories/``):

    runs/...          the inference runs (dra_inference layout)
    episodes.jsonl    one Episode per query (episodes.Episode.to_dict), every use
    sft.jsonl         {qid, dataset, split, call, iteration, messages, completion},
                      sft episodes only
    sft_val.jsonl     the same, val episodes only
    teacher_scores.json  scoring.assess of the teacher, all episodes
"""

import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from indexing_corpus_dataset.dataset_loaders import load_split
from indexing_corpus_dataset.layout import DATASET_SPECS, dataset_root
from utils.io_utils import build_dataset_dir_name

from .config import RunPaths, Split, TrainingConfig
from .scoring import assess
from .episodes import (
    DocumentStore, Episode, annotate_gold, assign_use, read_episode, replay_teacher, teacher_init_call, write_jsonl,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

def inference_command(cfg: TrainingConfig, output: Path) -> List[str]:
    """The ``experiments/inference/dra_inference.py`` command of phase 1 on
    the training pool (all its queries, or the next ``trajectories.limit``
    unprocessed ones), in monitor mode: the teacher belief model writes the
    criteria and the belief updates, the query scorer is the frozen
    ``cfg.query_scorer_model``."""
    tc, split = cfg.trajectories, cfg.training_split
    cmd = [
        sys.executable, str(REPO_ROOT / "experiments" / "inference" / "dra_inference.py"),
        "--config", cfg.shared.inference_config,
        "--dataset", split.dataset,
        "--retriever", cfg.shared.retriever,
        "--policy-model", tc.policy_model,
        "--belief-mode", "monitor",
        "--num-gpus", str(cfg.shared.num_gpus),
        "--quiet",
        "--output", str(output),
        "--llm-criteria", tc.teacher_belief_model,
        "--belief-judge-model", tc.teacher_belief_model,
        "--query-scorer-model", cfg.query_scorer_model,
        "--criteria-bank", "true",
    ]
    if split.subset:
        cmd += ["--subset", split.subset]
    if split.dataset_year:
        cmd += ["--dataset-year", split.dataset_year]
    if tc.limit is not None:
        cmd += ["--limit", str(tc.limit)]
    return cmd


def find_run_dir(split: Split, output: Path, retriever: str) -> Path:
    """The monitor-mode run directory dra_inference wrote for the training
    pool under *output* (one agent per output root)."""
    dataset_dir = build_dataset_dir_name(
        split.dataset, split.split_id, DATASET_SPECS[split.dataset].query_key, retriever,
    )
    found = sorted((output / dataset_dir).glob("*/belief-monitor"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one belief-monitor run under {output / dataset_dir}, found {found}")
    return found[0]


def training_gold(split: Split) -> Dict[str, Set[str]]:
    """The pool queries dra_inference runs (those with qrels, in file order)
    and their relevant documents, as dra_inference reads them."""
    queries, qrels, _ = load_split(
        dataset_root(split.dataset), split.split_id, query_key=DATASET_SPECS[split.dataset].query_key,
        min_relevance_score=DATASET_SPECS[split.dataset].min_relevance_score, only_with_qrels=True,
    )
    return {qid: {d for d, r in qrels[qid].items() if float(r) > 0} for qid in queries}


def read_correctness(run_dir: Path) -> Dict[str, Optional[bool]]:
    """Per query of the run, its final answer is correct (``correct`` of the
    LLM judge, or ``exact_match``); None when the judge failed."""
    path = run_dir / "accuracy.jsonl"
    if not path.exists():
        logger.warning("[phase 1] %s not found: no correctness", path)
        return {}
    out: Dict[str, Optional[bool]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if "query_id" not in row:
            continue
        label = row.get("correct", row.get("exact_match"))
        out[str(row["query_id"])] = None if row.get("judge_error") or label is None else bool(label)
    return out


def run_trajectories(cfg: TrainingConfig, paths: RunPaths) -> None:
    tc = cfg.trajectories
    split = cfg.training_split
    gold = training_gold(split)
    qids = list(gold)
    print(f"[phase 1] {len(qids)} training queries in {split.name}; policy model {tc.policy_model}, "
          f"teacher {tc.teacher_belief_model}, query scorer {cfg.query_scorer_model}")

    cmd = inference_command(cfg, paths.inference_runs)
    print(f"[phase 1] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    run_dir = find_run_dir(split, paths.inference_runs, cfg.shared.retriever)

    collect_episodes(cfg, paths, split, gold, run_dir)


def collect_episodes(cfg: TrainingConfig, paths: RunPaths, split: Split, gold: Dict[str, Set[str]],
                     run_dir: Path) -> None:
    """Write ``episodes.jsonl`` (every use), ``sft.jsonl`` (sft episodes),
    ``sft_val.jsonl`` (val episodes) and ``teacher_scores.json`` from the
    run; *gold* gives the queries and their relevant documents."""
    tc = cfg.trajectories
    teacher = tc.teacher_belief_model
    store = DocumentStore()
    correctness = read_correctness(run_dir)
    episodes: List[Episode] = []
    samples: Dict[str, List[Dict]] = {"sft": [], "val": []}
    counts: Counter = Counter()
    for qid in gold:
        path = run_dir / "belief" / f"{qid}.jsonl"
        if not path.exists():
            counts["not_run"] += 1
            continue
        episode = read_episode(path, split, run_dir)
        if episode is None:
            counts["old_schema_or_no_criteria"] += 1
            continue
        calls, episode.replay_steps, error = replay_teacher(episode, store.docs(episode))
        episode.replay_error = error
        counts["episodes"] += 1
        counts["replay_mismatch"] += error is not None
        if error is not None:
            logger.warning("[phase 1] %s %s: %s", split.name, qid, error)
        gold_error = annotate_gold(episode, gold[qid])
        if gold_error is not None:
            counts["gold_mismatch"] += 1
            logger.warning("[phase 1] %s %s: gold: %s", split.name, qid, gold_error)
        episode.correct = correctness.get(qid)
        episode.use = assign_use(qid, tc.sft_share, tc.val_share)
        episodes.append(episode)
        if episode.use not in samples:
            continue
        out = samples[episode.use]
        sample = {"qid": qid, "dataset": split.dataset, "split": split.split_id}
        if episode.criteria_model != teacher:
            counts["init_not_teacher"] += 1
        else:
            init = teacher_init_call(episode)
            if init["verified"]:
                counts[f"{episode.use}_init_samples"] += 1
                out.append({**sample, "call": "init", "iteration": 0,
                                "messages": init["messages"], "completion": init["output"]})
            else:
                counts["init_not_reproduced"] += 1
        for call in calls:
            counts["teacher_calls"] += 1
            if not call["parsed"]:
                counts["dropped_unparsed"] += 1
                continue
            if tc.require_verified_spans and not call["spans_verified"]:
                counts["dropped_unverified_spans"] += 1
                continue
            counts[f"{episode.use}_update_samples"] += 1
            out.append({**sample, "call": "update", "iteration": call["iteration"],
                            "messages": call["messages"], "completion": call["output"]})
    write_jsonl(paths.episodes_file, (ep.to_dict() for ep in episodes))
    n = write_jsonl(paths.sft_file, iter(samples["sft"]))
    n_val = write_jsonl(paths.sft_val_file, iter(samples["val"]))
    scores = assess(episodes)
    paths.teacher_scores_file.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"[phase 1] {dict(counts)}")
    print_statistics(episodes)
    print(f"[phase 1] teacher: {json.dumps({k: round(v, 3) for k, v in scores.items()})}")
    print(f"[phase 1] wrote {len(episodes)} episodes to {paths.episodes_file}, {n} SFT samples to "
          f"{paths.sft_file}, {n_val} validation samples to {paths.sft_val_file}")


def print_statistics(episodes: List[Episode]) -> None:
    """Per use: episodes, the share with a relevant document found, the
    correctness of the final answers, and steps and teacher calls per episode."""
    for use in ("sft", "rl", "val"):
        eps = [ep for ep in episodes if ep.use == use]
        if not eps:
            print(f"[phase 1] {use}: no episodes")
            continue
        n = len(eps)
        correct = Counter(ep.correct for ep in eps)
        steps = sum(len(ep.steps) for ep in eps) / n
        calls = sum(s.teacher_reply is not None for ep in eps for s in ep.steps) / n
        print(f"[phase 1] {use}: {n} episodes; with gold {sum(ep.has_gold for ep in eps) / n:.0%}; "
              f"correct {correct[True]}, wrong {correct[False]}, not judged {correct[None]}; "
              f"{steps:.1f} steps and {calls:.1f} teacher calls per episode; "
              f"by agent {dict(Counter(ep.policy_model for ep in eps))}, by source {dict(Counter(ep.source for ep in eps))}")

