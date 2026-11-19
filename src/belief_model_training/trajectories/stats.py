"""Statistics of the phase 2 runs, which set the RL settings left open
(``rl.max_chain_steps``, ``rl.group_size``, the data size):

* episodes (finished, judged, with gold beliefs) and the agent's accuracy;
* the search iterations with a document per episode (an upper bound on a
  chain's update calls): mean, median, 90th percentile, max;
* on runs with the teacher's beliefs: the teacher's final score s_T
  against y (Brier, AUROC), the baseline of the trained model.

Printed and written to ``{BELIEF_TRAINING_ROOT}/stats/{dataset}_{split}_{agent}.json``.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import RunPaths, TrainingConfig
from ..runs import find_run, load_run_episodes, read_gold_rows


def _quantiles(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {}
    a = np.array(values)
    return {"mean": round(float(a.mean()), 1), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max": int(a.max())}


def teacher_scores(run_dir, teacher: str, episodes) -> Optional[Dict[str, Any]]:
    """s_T of the teacher's belief files against y (None without any)."""
    from evaluation.measures import auroc, brier
    from utils.run_files import belief_dir
    from utils.run_files.belief import read_belief_file

    folder = belief_dir(run_dir, teacher)
    if not folder.is_dir():
        return None
    ys, ss, missing = [], [], 0
    for ep in episodes:
        _, _, final = read_belief_file(folder / f"{ep.qid}.jsonl", light=True)
        s = (final or {}).get("certainty") if (final or {}).get("final_score") else None
        if s is None:
            missing += 1
            continue
        ys.append(float(ep.y))
        ss.append(float(s))
    if not ys:
        return {"n": 0, "without_s_T": missing}
    y, s = np.array(ys), np.array(ss)
    both = len(set(ys)) == 2
    return {"n": len(ys), "without_s_T": missing, "brier": round(float(brier(y, s)), 4),
            "auroc": round(float(auroc(y, s)), 4) if both else None, "mean_s_T": round(float(s.mean()), 4)}


def run_stats(cfg: TrainingConfig, datasets: List[str], split: str, policy_models: List[str]) -> None:
    paths = RunPaths()
    paths.stats_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.inference_output
    for dataset in datasets:
        gold_rows = read_gold_rows(dataset)
        for policy in policy_models:
            run_dir = find_run(output, dataset, split, cfg.shared.retriever, policy)
            if run_dir is None:
                print(f"[phase 2] {dataset}/{split}/{policy}: no run")
                continue
            dropped: Dict[str, int] = {}
            episodes = load_run_episodes(run_dir, dataset, split, policy, gold_rows, dropped)
            stats = {
                "run_dir": str(run_dir),
                "episodes": len(episodes),
                "left_out": dropped,
                "accuracy": round(sum(e.y for e in episodes) / len(episodes), 4) if episodes else None,
                "doc_steps": _quantiles([e.doc_steps for e in episodes]),
                "teacher": teacher_scores(run_dir, cfg.shared.teacher, episodes),
            }
            out = paths.stats_dir / f"{dataset}_{split}_{policy}.json"
            out.write_text(json.dumps(stats, indent=2) + "\n")
            print(f"[phase 2] {dataset}/{split}/{policy}: {json.dumps({k: v for k, v in stats.items() if k != 'run_dir'})}")
