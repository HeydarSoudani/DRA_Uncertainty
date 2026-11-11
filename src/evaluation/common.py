"""Helpers shared by the evaluators: result files and statistics (the
measures are in ``evaluation.metrics``; JSONL writing is
``utils.io_utils.write_jsonl``, atomic)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils import io_utils

RULE = "=" * 80


def read_jsonl(path) -> List[Dict[str, Any]]:
    """The records of a JSONL file (none when it does not exist)."""
    path = Path(path)
    return io_utils.read_jsonl(path) if path.exists() else []


def write_json(path, obj: Any) -> None:
    """Write *obj* as indented JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


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
