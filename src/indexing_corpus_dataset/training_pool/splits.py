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

The shares and the per-split counts are recorded in the ``split`` section of
``stats.json``; a split with other shares than the recorded ones (it would
move queries between splits, e.g. from rl into val) is refused.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from indexing_corpus_dataset.layout import POOL_SPLITS
from utils.io_utils import atomic_write_text

from .config import PoolPaths, SplitConfig
from .stats import read_stats, write_stats


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


def read_lines(path: Path) -> List[str]:
    """The non-blank lines of *path*, each ending with a newline: rows to
    copy unchanged."""
    with open(path, encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh if line.strip()]


def check_shares(paths: PoolPaths, cfg: SplitConfig) -> None:
    """Refuse *cfg*'s shares when ``stats.json`` records other ones: other
    shares move queries between splits."""
    recorded = read_stats(paths.stats).get("split", {}).get("shares")
    if recorded is not None and recorded != cfg.shares():
        raise SystemExit(f"{paths.root} was split with the shares {recorded}, the config has {cfg.shares()}: "
                         "other shares move queries between splits (e.g. from rl into val). "
                         f"Restore the shares, or remove the 'split' section of {paths.stats} to re-split")


def write_splits(paths: PoolPaths, cfg: SplitConfig) -> None:
    """Write the splits ``sft``, ``rl`` and ``val`` of the pool's split
    ``all`` and record them in the ``split`` section of ``stats.json``."""
    for path in (paths.queries, paths.qrels):
        if not path.exists():
            raise SystemExit(f"{path} not found: build the pool first")
    check_shares(paths, cfg)
    shares = cfg.shares()
    queries = read_lines(paths.queries)
    rows = [json.loads(line) for line in queries]
    split_of = {str(row["id"]): (assign_split(str(row["id"]), shares), row.get("source")) for row in rows}
    if len(split_of) != len(queries):
        raise SystemExit(f"{paths.queries}: duplicate query ids")

    # (file of split ``all``, its rows, their query ids, the path of a split)
    files = [(paths.queries, queries, [str(row["id"]) for row in rows], paths.split_queries)]
    qrels = read_lines(paths.qrels)
    files.append((paths.qrels, qrels, [line.split()[0] for line in qrels], paths.split_qrels))
    info_all = paths.split_info(paths.split)
    if info_all.exists():
        info = read_lines(info_all)
        files.append((info_all, info, [str(json.loads(line)["qid"]) for line in info], paths.split_info))
    for source, lines, qids, target in files:
        out: Dict[str, List[str]] = {name: [] for name in POOL_SPLITS}
        for line, qid in zip(lines, qids):
            if qid not in split_of:
                raise SystemExit(f"{source}: query {qid} is not in {paths.queries}")
            out[split_of[qid][0]].append(line)
        for name, split_lines in out.items():
            atomic_write_text(target(name), "".join(split_lines))

    counts = Counter(split for split, _ in split_of.values())
    by_source = {name: dict(sorted(Counter(source for split, source in split_of.values()
                                           if split == name and source).items()))
                 for name in POOL_SPLITS}
    write_stats(paths.stats, "split", {"shares": shares, "queries": {name: counts[name] for name in POOL_SPLITS},
                                       "by_source": by_source})
    print(f"[split] {len(split_of)} queries of split {paths.split}, shares {shares} -> {paths.root}")
    for name in POOL_SPLITS:
        detail = f" ({', '.join(f'{s} {n}' for s, n in by_source[name].items())})" if by_source[name] else ""
        print(f"[split]   {name}: {counts[name]} queries ({counts[name] / max(1, len(split_of)):.1%}){detail}")
