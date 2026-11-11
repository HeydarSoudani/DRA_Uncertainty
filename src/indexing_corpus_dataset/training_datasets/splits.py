"""The split of a training dataset for belief model training: the
``split`` field of each row of the dataset's queries file
(``gold/annotations.py``).  SFT and RL need the full gold annotation (every
criterion with gold answers and documents: r_ret and r_reas), so the queries
of infoseek and trqa_validation whose annotation ``level`` is ``full`` are
split into ``sft``, ``rl`` and ``val`` with the shares of the ``split``
config section, and every other query goes to ``val`` as the out-of-domain
part (OOD: the belief model writes its own criteria there).  webshaper_frames
is validation only (``val_only``): every query in ``val`` (WebShaper
``partial``, FRAMES ``query``).

    sft   SFT samples (dra_inference in monitor mode, the teacher's replies)
    rl    RL prompts (dra_inference in off mode)
    val   in-domain: validation of SFT and RL (monitor mode, gold criteria);
          OOD: evaluation only (monitor mode, the teacher's own criteria)

A query keeps the split it has in the queries file, so a build after the
dataset has grown (``--limit``) moves no query, even when its criterion texts are written again (removed
``criterion_texts.jsonl``) and fail: training skips a query without texts
(``texts_ok`` not true).  The new full queries are taken in the order of a
hash of their id (not the source order) and each goes to the split
furthest below its share of the full queries, so the shares hold at every
dataset size (20 queries: 6 sft, 12 rl, 2 val).  A new full query without
criterion texts is in no split (``None``): one not written yet (writer
error) gets its split when a later build writes them, one whose texts
failed for good (``texts_ok`` false) never does.  The splits are views of the one queries file:
``dataset_loaders`` reads split ``sft`` of a dataset as the rows with that
``split`` (and their qrels), so dra_inference runs ``--subset sft`` as on
any dataset.

The shares and the per-split counts are recorded in the ``split`` section of
``stats.json``; a split with other shares than the recorded ones is refused.
To re-split every query (e.g. with other shares), remove the ``split``
section: the next build ignores the splits in the queries file.
"""

import hashlib
import json
from collections import Counter
from typing import Dict, List, Optional

from indexing_corpus_dataset.layout import TRAINING_SPLITS

from .gold.annotations import FULL
from .config import DatasetPaths, SplitConfig
from .stats import read_stats, write_stats


# The split of the queries without the full gold annotation (out of domain).
OOD_SPLIT = "val"


def _hash(qid: str) -> str:
    return hashlib.sha256(qid.encode("utf-8")).hexdigest()


def assign_splits(qids: List[str], shares: Dict[str, float], counts: Counter) -> Dict[str, str]:
    """The splits of the new full queries *qids*, in the order of the hash
    of their id: each to the split furthest below its share (``shares``
    times the full queries so far, *counts* holding those already split;
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
        raise SystemExit(f"{paths.root} was split with the shares {recorded}, the config has {cfg.shares()}. "
                         f"Restore the shares, or remove the 'split' section of {paths.stats} to re-split "
                         "every query (queries move between splits)")


def _previous(paths: DatasetPaths) -> Dict[str, Optional[str]]:
    """The split of each query in the dataset's queries file, when the
    ``split`` section of ``stats.json`` records the split (else none: a
    re-split)."""
    if "split" not in read_stats(paths.stats) or not paths.queries.exists():
        return {}
    with open(paths.queries, encoding="utf-8") as fh:
        return {str(row["id"]): row.get("split") for row in map(json.loads, fh) if row}


def split_rows(paths: DatasetPaths, cfg: SplitConfig, rows: List[dict], val_only: bool = False) -> List[dict]:
    """*rows* (every query of the dataset) with their ``split`` set, recorded
    in the ``split`` section of ``stats.json``; *val_only*: every query in
    ``val`` (no shares)."""
    if not val_only:
        check_shares(paths, cfg)
    shares = {name: float(name == OOD_SPLIT) for name in TRAINING_SPLITS} if val_only else cfg.shares()
    previous = _previous(paths)
    split_of: Dict[str, Optional[str]] = {}
    new: List[str] = []
    for row in rows:
        qid = str(row["id"])
        if val_only or row["level"] != FULL:
            split_of[qid] = OOD_SPLIT
        elif previous.get(qid) in TRAINING_SPLITS:
            split_of[qid] = previous[qid]
        elif not row["texts_ok"]:
            split_of[qid] = None
        else:
            new.append(qid)
    counts = Counter(split_of[str(r["id"])] for r in rows if r["level"] == FULL and split_of.get(str(r["id"])))
    split_of.update(assign_splits(new, shares, counts))
    rows = [{**row, "split": split_of[str(row["id"])]} for row in rows]

    def counts_by(key) -> Dict[str, Dict[str, int]]:
        return {name: dict(sorted(Counter(key(r) for r in rows if r["split"] == name).items()))
                for name in TRAINING_SPLITS}

    total = Counter(r["split"] for r in rows)
    by_source = counts_by(lambda r: r["source"])
    by_level = counts_by(lambda r: r["level"])
    n_full = sum(counts.values())
    # Full queries that kept their split without criterion texts (skipped by training).
    no_texts = Counter(r["split"] for r in rows if r["level"] == FULL and r["split"] and not r["texts_ok"])
    write_stats(paths.stats, "split", {"shares": shares, "queries": {name: total[name] for name in TRAINING_SPLITS},
                                       "full": {name: counts[name] for name in TRAINING_SPLITS},
                                       "full_without_texts": {name: no_texts[name] for name in TRAINING_SPLITS},
                                       "no_split": total[None], "by_source": by_source, "by_level": by_level})
    print(f"[split] {len(rows)} queries: {n_full} {FULL} ones split with the shares {shares} ({len(new)} new), "
          f"the others to {OOD_SPLIT} (out of domain); {total[None]} {FULL} ones without criterion texts in no split")
    for name in TRAINING_SPLITS:
        detail = ", ".join(f"{s} {n}" for s, n in by_source[name].items())
        levels = ", ".join(f"{lv} {n}" for lv, n in by_level[name].items())
        print(f"[split]   {name}: {total[name]} queries, {counts[name]} {FULL} "
              f"({counts[name] / max(1, n_full):.1%} of the {FULL} ones"
              f"{f', {no_texts[name]} without texts' if no_texts[name] else ''})"
              f"{f' ({detail})' if detail else ''}; levels {levels or '-'}")
    return rows
