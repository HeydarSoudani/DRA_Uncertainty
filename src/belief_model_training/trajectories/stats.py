"""Statistics of the generated runs (stage generate), which set the
settings left open (``filter.rl_max_chain_steps``, ``rl.group_size``, the
data size):

* episodes (finished, judged, with gold beliefs) and the agent's accuracy;
* the search iterations with a document per episode (an upper bound on a
  chain's update calls): mean, median, 90th percentile, max.

The teacher's s_T against y (stage teacher) is in the run's ``eval/metrics.md``,
written by the belief stage.

Printed and written to ``{BELIEF_TRAINING_ROOT}/stats/{dataset}_{split}_{agent}.json``.
"""

import json
from typing import Any, Dict, List

import numpy as np

from ..shared.config import RunPaths, TrainingConfig
from ..shared.runs import find_run, load_run_episodes, read_gold_rows


def _quantiles(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {}
    a = np.array(values)
    return {"mean": round(float(a.mean()), 1), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max": int(a.max())}


def run_stats(cfg: TrainingConfig, datasets: List[str], split: str, policy_models: List[str]) -> None:
    paths = RunPaths()
    paths.stats_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.inference_output
    for dataset in datasets:
        gold_rows = read_gold_rows(dataset)
        for policy in policy_models:
            run_dir = find_run(output, dataset, split, cfg.shared.retriever, policy)
            if run_dir is None:
                print(f"[stage generate] {dataset}/{split}/{policy}: no run")
                continue
            dropped: Dict[str, int] = {}
            episodes = load_run_episodes(run_dir, dataset, split, policy, gold_rows, dropped)
            stats = {
                "run_dir": str(run_dir),
                "episodes": len(episodes),
                "left_out": dropped,
                "accuracy": round(sum(e.y for e in episodes) / len(episodes), 4) if episodes else None,
                "doc_steps": _quantiles([e.doc_steps for e in episodes]),
            }
            out = paths.stats_dir / f"{dataset}_{split}_{policy}.json"
            out.write_text(json.dumps(stats, indent=2) + "\n")
            print(f"[stage generate] {dataset}/{split}/{policy}: {json.dumps({k: v for k, v in stats.items() if k != 'run_dir'})}")
