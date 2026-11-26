"""The split of a training dataset for belief model training: the
``split`` field of each row of the dataset's queries file
(``annotation/rows.py``).  Every query of every dataset is split into
``sft``, ``rl`` and ``val`` with the shares of the ``split`` config
section.

    sft   SFT samples (dra_inference in monitor mode, the teacher's replies)
    rl    RL prompts (dra_inference in off mode)
    val   validation of SFT and RL (monitor mode, the teacher's beliefs)

A query keeps the split it has in the queries file, so a build after the
dataset has grown (``--limit``) moves no query.  The new queries are taken
in the order of a hash of their id (not the source order) and each goes to
the split furthest below its share, so the shares hold at every dataset
size (20 queries: 6 sft, 12 rl, 2 val).  Training leaves out a query whose
gold beliefs the teacher did not make
(``belief_model_training.runs.read_gold_rows``).  The splits are views of
the one queries file: ``dataset_loaders`` reads split ``sft`` of a dataset
as the rows with that ``split`` (and their qrels), so dra_inference runs
``--subset sft`` as on any dataset.

The shares and the per-split counts are recorded in the ``split`` section of
``stats.json``; a split with other shares than the recorded ones is refused.
To re-split every query (e.g. with other shares), remove the ``split``
section: the next build ignores the splits in the queries file.
"""

import hashlib
from collections import Counter
from typing import Dict, List, Optional

from indexing_corpus_dataset.layout import TRAINING_SPLITS
from utils.io_utils import iter_jsonl

from ..config import DatasetPaths, SplitConfig
from .stats import read_stats, write_stats


def _hash(qid: str) -> str:
    return hashlib.sha256(qid.encode("utf-8")).hexdigest()


def assign_splits(qids: List[str], shares: Dict[str, float], counts: Counter) -> Dict[str, str]:
    """The splits of the new queries *qids*, in the order of the hash of
    their id: each to the split furthest below its share (``shares`` times
    the queries so far, *counts* holding those already split;
    a tie goes to the first in ``TRAINING_SPLITS``).  Updates *counts*."""
    out = {}
    for qid in sorted(qids, key=_hash):
        n = sum(counts[name] for name in TRAINING_SPLITS) + 1
        name = max(TRAINING_SPLITS, key=lambda s: shares[s] * n - counts[s])
        counts[name] += 1
        out[qid] = name
    return out


def check_shares(paths: DatasetPaths, cfg: SplitConfig) -> None:
    """Refuse *cfg*'s shares when ``stats.json`` records other ones: other
    shares would mix two splits in one dataset."""
    recorded = read_stats(paths.stats).get("split", {}).get("shares")
    if recorded is not None and recorded != cfg.shares():
        raise SystemExit(f"{paths.dataset} was split with the shares {recorded}, the config has {cfg.shares()}. "
                         f"Restore the shares, or remove the 'split' section of {paths.stats} to re-split "
                         "every query (queries move between splits)")


def _previous(paths: DatasetPaths) -> Dict[str, Optional[str]]:
    """The split of each query in the dataset's queries file, when the
    ``split`` section of ``stats.json`` records the split (else none: a
    re-split)."""
    if "split" not in read_stats(paths.stats) or not paths.queries.exists():
        return {}
    return {str(row["id"]): row.get("split") for row in iter_jsonl(paths.queries)}


def split_rows(paths: DatasetPaths, cfg: SplitConfig, rows: List[dict]) -> List[dict]:
    """*rows* (every query of the dataset) with their ``split`` set, recorded
    in the ``split`` section of ``stats.json`` (the shares checked first,
    ``check_shares`` in ``relevance.build_relevance``)."""
    shares = cfg.shares()
    previous = _previous(paths)
    split_of: Dict[str, str] = {}
    new: List[str] = []
    for row in rows:
        qid = str(row["id"])
        if previous.get(qid) in TRAINING_SPLITS:
            split_of[qid] = previous[qid]
        else:
            new.append(qid)
    counts = Counter(split_of.values())
    split_of.update(assign_splits(new, shares, counts))
    rows = [{**row, "split": split_of[str(row["id"])]} for row in rows]

    by_source = {name: dict(sorted(Counter(r["source"] for r in rows if r["split"] == name).items()))
                 for name in TRAINING_SPLITS}
    write_stats(paths.stats, "split", {"shares": shares, "queries": {name: counts[name] for name in TRAINING_SPLITS},
                                       "by_source": by_source})
    print(f"[split] {len(rows)} queries split with the shares {shares} ({len(new)} new)")
    for name in TRAINING_SPLITS:
        detail = ", ".join(f"{s} {n}" for s, n in by_source[name].items())
        print(f"[split]   {name}: {counts[name]} queries ({counts[name] / max(1, len(rows)):.1%})"
              f"{f' ({detail})' if detail else ''}")
    return rows
