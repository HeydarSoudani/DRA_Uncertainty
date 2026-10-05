"""Helpers shared by the evaluators: result files, statistics, terminal output."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

RULE = "=" * 80


def print_header(title: str) -> None:
    """Print the banner that opens one evaluator's terminal section."""
    print("\n" + RULE)
    print(title)
    print(RULE)


def write_json(path, obj: Any) -> None:
    """Write *obj* as indented JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def write_jsonl(path, records: Iterable[Dict[str, Any]]) -> None:
    """Write one compact JSON record per line, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")


def summary_stats(values: List[float]) -> Dict[str, float]:
    """Mean, std, min and max of *values* (all 0 when empty)."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }
