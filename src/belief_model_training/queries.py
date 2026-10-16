"""Phase 0: the training queries, possibly from several datasets.

Output ``phase0_queries/queries.jsonl``, one query per line::

    {"qid": str, "dataset": str, "subset": str | null, "dataset_year": str | null}

(``dataset``, ``subset`` and ``dataset_year`` as in ``config.Split``), read by
phase 1 with ``read_queries``.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .config import RunPaths, Split, TrainingConfig


def build_training_queries(cfg: TrainingConfig, paths: RunPaths) -> None:
    """Write ``paths.queries_file``.  TODO: the choice of training queries
    (datasets, splits, held-out part) is still to be designed."""
    raise NotImplementedError(
        "Phase 0 (training queries) is TODO. Until then, write "
        f"{paths.queries_file} by hand, one JSON object per line: "
        '{"qid": ..., "dataset": ..., "subset": ..., "dataset_year": ...}'
    )


def read_queries(path: Path) -> Dict[Split, List[str]]:
    """The query ids of *path* (phase 0 output), per split, in file order."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run phase 0 first")
    by_split: Dict[Split, List[str]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = Split.parse(":".join([row["dataset"], row.get("subset") or "", row.get("dataset_year") or ""]))
        if row["qid"] not in by_split[split]:
            by_split[split].append(str(row["qid"]))
    return dict(by_split)
