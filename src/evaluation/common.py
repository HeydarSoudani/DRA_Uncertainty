"""Helpers shared by the evaluators: result files and statistics."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

RULE = "=" * 80


def read_jsonl(path) -> List[Dict[str, Any]]:
    """The records of a JSONL file (none when it does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def mean_or_none(values: List[float], digits: Optional[int] = None) -> Optional[float]:
    """Mean of *values*, rounded to *digits* when given; None when empty."""
    if not values:
        return None
    mean = sum(values) / len(values)
    return round(mean, digits) if digits is not None else mean


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
