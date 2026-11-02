"""The trqa_validation pool: the TRQA wiki1 and wiki2 validation splits merged
into one split (``all``) in the canonical layout:

    queries/queries_all.jsonl  queries_wiki{1,2}_validation.jsonl and
                               queries_wiki{1,2}_validation_intermediate_info.jsonl
    qrels/qrels_all.txt        qrels_wiki{1,2}_validation.txt

A query and its intermediate info become one row of the pool
(``annotations.py``: id, text, answer, source ``trqa_wiki1`` / ``trqa_wiki2``,
split, the gold annotation; the intermediate info's members, values and
documents as ``members``, its property label and aggregation); the qrels
rows are copied unchanged (the query ids of the two collections do not
collide).  Left out, with their qrels: a query whose
text is also a TRQA test query, and both queries of an id that names two
different questions in one split (14 ids in wiki2: their qrels and
intermediate info cannot be told apart).  No corpus and no index: the pool
searches TRQA's ``wiki_partial`` (``layout._SHARED_CORPUS``).  The build
is incremental, like deep_qa step 2: with a *limit*, it adds the next N
kept queries not yet in the pool (wiki1, then wiki2, in file order) to the
queries already there (0: none, the files are rewritten); without one,
every kept query.  The criterion texts of the earlier builds are reused, and
every query keeps its split.

The gold annotation (``annotations.set_row``) comes from the intermediate info: the set and the rest criterion with the
members and their values and each member's documents
(``entity_values[].passage_ids``), each member named by the page title
(the first line) of its first document in ``wiki_partial`` (``member_titles``,
one pass over the corpus).  Its source is ``trqa_{subset}``;
an LLM writes the criterion texts (``gold_criteria.fill_texts``).
The build sets each row's split (``sft``, ``rl`` or ``val``, ``splits.py``),
records its counts in the ``build`` section of ``stats.json`` and the
annotation stats in its ``annotations`` section.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from indexing_corpus_dataset.layout import (dataset_root, default_corpus_path, qrels_base, queries_base,
                                            resolve_split_id)
from utils.io_utils import atomic_write_text, iter_jsonl

from . import annotations
from .config import GoldCriteriaConfig, PoolPaths, SplitConfig
from .gold_criteria import check_writer, fill_texts
from .splits import check_shares, split_rows
from .stats import format_annotation_table, write_annotation_stats, write_stats

SUBSETS = ("wiki1", "wiki2")


def read_lines(path: Path) -> List[str]:
    """The non-blank lines of *path*, each ending with a newline: rows to
    copy unchanged."""
    with open(path, encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh if line.strip()]


def _test_texts(trqa: Path) -> Set[str]:
    texts = set()
    for subset in SUBSETS:
        path = queries_base(trqa, resolve_split_id("trqa", "test", subset)).with_suffix(".jsonl")
        texts |= {row["text"] for row in iter_jsonl(path)}
    return texts


def member_titles(infos: List[dict]) -> Dict[str, str]:
    """The page title of each member's first document in *infos* (doc id ->
    title): one pass over ``wiki_partial``, a line parsed only when its id is
    one of them."""
    need = {v["passage_ids"][0] for info in infos for v in info.get("entity_values") or [] if v.get("passage_ids")}
    titles: Dict[str, str] = {}
    corpus = default_corpus_path("trqa_validation")
    print(f"[trqa_validation] member titles: {len(need)} documents from {corpus}")
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            start = line.find('"id": "') + 7
            if line[start:line.find('"', start)] in need:
                row = json.loads(line)
                titles[row["id"]] = row["contents"].split("\n", 1)[0].strip()
                if len(titles) == len(need):
                    break
    return titles


def build(paths: PoolPaths, gc: GoldCriteriaConfig, split_cfg: SplitConfig, limit: Optional[int] = None) -> None:
    # Before the writer's calls.
    check_writer(gc, paths)
    check_shares(paths, split_cfg)
    trqa = dataset_root("trqa")
    test_texts = _test_texts(trqa)
    ids, qrels, overlap, ambiguous, pending = [], [], [], [], []
    for subset in SUBSETS:
        split = resolve_split_id("trqa", "validation", subset)
        base = queries_base(trqa, split)
        lines = read_lines(base.with_suffix(".jsonl"))
        rows = [json.loads(line) for line in lines]
        counts = Counter(row["id"] for row in rows)
        dup = {qid for qid, n in counts.items() if n > 1}
        test = {row["id"] for row in rows if row["text"] in test_texts} - dup
        ambiguous += sorted(dup)
        overlap += sorted(test)
        drop = dup | test
        ids += [row["id"] for row in rows if row["id"] not in drop]
        sub_info = [line for line in read_lines(base.with_name(f"{base.name}_intermediate_info.jsonl"))
                    if json.loads(line)["qid"] not in drop]
        sub_qrels = [line for line in read_lines(qrels_base(trqa, split).with_suffix(".txt"))
                     if line.split()[0] not in drop]
        qrels += sub_qrels
        info_of = {str(r["qid"]): r for r in map(json.loads, sub_info)}
        docs_of: Dict[str, List[str]] = {}
        for line in sub_qrels:
            qid, _, doc_id, rel = line.split()
            if float(rel) > 0:
                docs_of.setdefault(qid, []).append(doc_id)
        pending += [(row, subset, info_of.get(str(row["id"]), {}), docs_of.get(str(row["id"]), []))
                    for row in rows if row["id"] not in drop]
    titles = member_titles([info for _, _, info, _ in pending])
    gold = [annotations.set_row(str(row["id"]), f"trqa_{subset}", row["text"], str(row["answer"]), info, docs, titles)
            for row, subset, info, docs in pending]

    if len(ids) != len(set(ids)):
        raise SystemExit("query ids collide between the TRQA validation splits")
    available = len(ids)
    # The queries already in the pool, then the next *limit* others.
    built = {str(r["id"]) for r in iter_jsonl(paths.queries)} if paths.queries.exists() else set()
    new = [str(qid) for qid in ids if str(qid) not in built]
    if limit is not None:
        keep = built | set(new[:limit])
        print(f"[trqa_validation] {len(built & set(map(str, ids)))} queries in the pool, "
              f"{min(limit, len(new))} added (--limit {limit}), {max(0, len(new) - limit)} remaining")
        qrels = [line for line in qrels if line.split()[0] in keep]
        gold = [row for row in gold if row["id"] in keep]

    # The texts first: a stopped run leaves the files of the last full one.
    annotations.check(gold)
    gold = split_rows(paths, split_cfg, fill_texts(gc, paths, gold))
    annotations.write(paths.queries, gold)
    atomic_write_text(paths.qrels, "".join(qrels))

    with_qrels = len({line.split()[0] for line in qrels})
    stats = write_annotation_stats(paths, gc, gold)
    write_stats(paths.stats, "build", {"available": available, "remaining": available - len(gold),
                                       "queries": len(gold), "with_qrels": with_qrels, "qrels": len(qrels),
                                       "left_out": {"test_overlap": overlap, "ambiguous_ids": ambiguous}})
    print(f"[trqa_validation] {len(gold)} of {available} queries ({with_qrels} with qrels), "
          f"{len(qrels)} qrels -> {paths.root}")
    print(f"[trqa_validation] gold annotation\n\n{format_annotation_table(stats)}\n")
