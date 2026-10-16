"""Phase 1: teacher trajectories.

For each split of the phase 0 queries, ``experiments/dra_inference.py`` runs
the frozen agent with the teacher belief model in ``monitor`` mode (the
agent never sees the belief), restricted to the split's training queries
(``--query-ids-file``).  Inference resume applies: queries already run are
skipped.  Then every run's ``belief/{qid}.jsonl`` becomes an episode, the
teacher's replies are replayed (``episodes.replay_teacher``), and each
teacher call that parsed (and, with ``require_verified_spans``, whose cited
spans all occur in their passage) becomes an SFT sample.

Outputs (``phase1_trajectories/``):

    runs/...          the inference runs (dra_inference layout)
    episodes.jsonl    one Episode per query (episodes.Episode.to_dict)
    sft.jsonl         {qid, dataset, split, iteration, messages, completion}
"""

import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from indexing_corpus_dataset.layout import DATASET_SPECS
from utils.io_utils import build_dataset_dir_name

from .config import RunPaths, Split, TrainingConfig
from .episodes import DocumentStore, read_episode, replay_teacher, write_jsonl
from .queries import read_queries

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def inference_command(cfg: TrainingConfig, split: Split, *, agent: str, belief_mode: str, output: Path,
                      belief_judge_model: str, criteria_model: str,
                      query_ids_file: Optional[Path] = None, gpu_ids: str = "") -> List[str]:
    """The ``experiments/dra_inference.py`` command of one split (all its
    queries without *query_ids_file*)."""
    cmd = [
        sys.executable, str(REPO_ROOT / "experiments" / "dra_inference.py"),
        "--config", cfg.shared.inference_config,
        "--dataset", split.dataset,
        "--retriever", cfg.shared.retriever,
        "--agentic-model", agent,
        "--belief-mode", belief_mode,
        "--num-gpus", str(cfg.shared.num_gpus),
        "--quiet",
        "--output", str(output),
        "--llm-criteria", criteria_model,
        "--belief-judge-model", belief_judge_model,
    ]
    if split.subset:
        cmd += ["--subset", split.subset]
    if split.dataset_year:
        cmd += ["--dataset-year", split.dataset_year]
    if query_ids_file is not None:
        cmd += ["--query-ids-file", str(query_ids_file)]
    if gpu_ids:
        cmd += ["--gpu-ids", gpu_ids]
    return cmd


def find_run_dir(output: Path, split: Split, retriever: str, belief_mode: str) -> Path:
    """The run directory dra_inference wrote for *split* under *output*
    (one agent per output root)."""
    dataset_dir = build_dataset_dir_name(
        split.dataset, split.split_id, DATASET_SPECS[split.dataset].query_key, retriever,
    )
    found = sorted((output / dataset_dir).glob(f"*/belief-{belief_mode}"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one belief-{belief_mode} run under {output / dataset_dir}, found {found}")
    return found[0]


def write_query_ids(path: Path, qids: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{q}\n" for q in qids))
    return path


def run_trajectories(cfg: TrainingConfig, paths: RunPaths) -> None:
    tc = cfg.trajectories
    queries = read_queries(paths.queries_file)
    print(f"[phase 1] {sum(map(len, queries.values()))} training queries in {len(queries)} splits; "
          f"agent {tc.agent}, teacher {tc.teacher_model}")

    run_dirs: Dict[Split, Path] = {}
    for split, qids in queries.items():
        ids_file = write_query_ids(paths.query_ids_dir / f"{split.name}.txt", qids)
        cmd = inference_command(
            cfg, split, agent=tc.agent, belief_mode="monitor", output=paths.inference_runs,
            query_ids_file=ids_file, belief_judge_model=tc.teacher_model, criteria_model=tc.criteria_model,
        )
        print(f"[phase 1] {split.name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        run_dirs[split] = find_run_dir(paths.inference_runs, split, cfg.shared.retriever, "monitor")

    collect_episodes(cfg, paths, queries, run_dirs)


def collect_episodes(cfg: TrainingConfig, paths: RunPaths, queries: Dict[Split, List[str]],
                     run_dirs: Dict[Split, Path]) -> None:
    """Write ``episodes.jsonl`` and ``sft.jsonl`` from the runs."""
    store = DocumentStore()
    episodes, samples = [], []
    counts: Counter = Counter()
    for split, qids in queries.items():
        for qid in qids:
            path = run_dirs[split] / "belief" / f"{qid}.jsonl"
            if not path.exists():
                counts["missing_belief_file"] += 1
                continue
            episode = read_episode(path, split, run_dirs[split])
            if episode is None:
                counts["old_schema_or_no_criteria"] += 1
                continue
            calls, episode.replay_steps, error = replay_teacher(episode, store.docs(episode))
            episode.replay_error = error
            counts["episodes"] += 1
            counts["replay_mismatch"] += error is not None
            if error is not None:
                logger.warning("[phase 1] %s %s: %s", split.name, qid, error)
            episodes.append(episode.to_dict())
            for call in calls:
                counts["teacher_calls"] += 1
                if not call["parsed"]:
                    counts["dropped_unparsed"] += 1
                    continue
                if cfg.trajectories.require_verified_spans and not call["spans_verified"]:
                    counts["dropped_unverified_spans"] += 1
                    continue
                samples.append({
                    "qid": qid, "dataset": split.dataset, "split": split.split_id,
                    "iteration": call["iteration"], "messages": call["messages"],
                    "completion": call["output"],
                })
    write_jsonl(paths.episodes_file, iter(episodes))
    n = write_jsonl(paths.sft_file, iter(samples))
    print(f"[phase 1] {dict(counts)}")
    print(f"[phase 1] wrote {len(episodes)} episodes to {paths.episodes_file} and {n} SFT samples to {paths.sft_file}")

