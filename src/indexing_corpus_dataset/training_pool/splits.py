"""The split of a training pool for belief model training: every query of
split ``all`` goes to ``sft``, ``rl`` or ``val`` by a hash of its id, with
the shares of the ``split`` config section:

    sft   SFT samples (dra_inference in monitor mode, the teacher's replies)
    rl    RL prompts (dra_inference in off mode)
    val   validation of SFT and RL (monitor mode)

A query's split depends on its id only, so a rerun after the pool has grown
(deep_qa step 2) keeps every earlier query in its split.  Each split gets the
queries, the intermediate info (when the pool has it) and the qrels of its
queries, rows copied unchanged.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from indexing_corpus_dataset.layout import POOL_SPLITS

from .config import PoolPaths, SplitConfig


def assign_split(qid: str, shares: Dict[str, float]) -> str:
    """The split of *qid*: a uniform draw in [0, 1) from a hash of the id,
    cut at the cumulative *shares* (in ``POOL_SPLITS`` order)."""
    u = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    total = 0.0
    for name in POOL_SPLITS:
        total += shares[name]
        if u < total:
            return name
    return POOL_SPLITS[-1]


def _lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh if line.strip()]


def write_splits(paths: PoolPaths, cfg: SplitConfig) -> None:
    """Write the splits ``sft``, ``rl`` and ``val`` of the pool's split ``all``."""
    for path in (paths.queries, paths.qrels):
        if not path.exists():
            raise SystemExit(f"{path} not found: build the pool first")
    shares = cfg.shares()
    queries = _lines(paths.queries)
    split_of = {}
    for line in queries:
        row = json.loads(line)
        split_of[str(row["id"])] = (assign_split(str(row["id"]), shares), row.get("source"))
    if len(split_of) != len(queries):
        raise SystemExit(f"{paths.queries}: duplicate query ids")

    info_all = paths.split_info(paths.split)
    files = [(paths.queries, paths.split_queries, lambda line: json.loads(line)["id"]),
             (paths.qrels, paths.split_qrels, lambda line: line.split()[0])]
    if info_all.exists():
        files.append((info_all, paths.split_info, lambda line: json.loads(line)["qid"]))
    for source, target, qid_of in files:
        out: Dict[str, List[str]] = {name: [] for name in POOL_SPLITS}
        for line in _lines(source):
            qid = str(qid_of(line))
            if qid not in split_of:
                raise SystemExit(f"{source}: query {qid} is not in {paths.queries}")
            out[split_of[qid][0]].append(line)
        for name, lines in out.items():
            path = target(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(lines), encoding="utf-8")

    counts = Counter(split for split, _ in split_of.values())
    print(f"[split] {len(split_of)} queries of split {paths.split}, shares {shares} -> {paths.root}")
    for name in POOL_SPLITS:
        by_source = Counter(source for split, source in split_of.values() if split == name and source)
        detail = f" ({', '.join(f'{s} {n}' for s, n in sorted(by_source.items()))})" if by_source else ""
        print(f"[split]   {name}: {counts[name]} queries ({counts[name] / max(1, len(split_of)):.1%}){detail}")
