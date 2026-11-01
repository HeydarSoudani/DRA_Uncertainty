"""The trqa_validation pool: the TRQA wiki1 and wiki2 validation splits merged
into one split (``all``) in the canonical layout:

    queries/queries_all.jsonl                    queries_wiki{1,2}_validation.jsonl
    queries/queries_all_intermediate_info.jsonl  queries_wiki{1,2}_validation_intermediate_info.jsonl
    qrels/qrels_all.txt                          qrels_wiki{1,2}_validation.txt

Rows are copied unchanged (the query ids of the two collections do not
collide).  Left out, with their qrels and intermediate info: a query whose
text is also a TRQA test query, and both queries of an id that names two
different questions in one split (14 ids in wiki2: their qrels and
intermediate info cannot be told apart).  No corpus and no index: the pool
searches TRQA's ``wiki_partial`` (``layout._SHARED_CORPUS``).

The gold annotation (``annotations.set_row``, ``gold/annotations_all.jsonl``)
comes from the intermediate info: the set and the rest criterion with the
members and their values and each member's documents
(``entity_values[].passage_ids``).  Its source is ``trqa_{subset}``;
an LLM writes the criterion texts (``gold_criteria.fill_texts``).
The build records its counts and the annotation stats in the ``build``
section of ``stats.json``, then splits ``all`` into ``sft``, ``rl`` and
``val`` (``splits.py``).
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from indexing_corpus_dataset.layout import dataset_root, qrels_base, queries_base, resolve_split_id
from utils.io_utils import atomic_write_text, iter_jsonl

from . import annotations
from .config import GoldCriteriaConfig, PoolPaths
from .gold_criteria import check_writer, fill_texts
from .splits import read_lines
from .stats import annotation_stats, format_annotation_table, write_stats

SUBSETS = ("wiki1", "wiki2")


def _test_texts(trqa: Path) -> Set[str]:
    texts = set()
    for subset in SUBSETS:
        path = queries_base(trqa, resolve_split_id("trqa", "test", subset)).with_suffix(".jsonl")
        texts |= {row["text"] for row in iter_jsonl(path)}
    return texts


def build(paths: PoolPaths, gc: GoldCriteriaConfig) -> None:
    check_writer(gc, paths)
    trqa = dataset_root("trqa")
    test_texts = _test_texts(trqa)
    queries, ids, info, qrels, overlap, ambiguous, gold = [], [], [], [], [], [], []
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
        queries += [line for line, row in zip(lines, rows) if row["id"] not in drop]
        ids += [row["id"] for row in rows if row["id"] not in drop]
        sub_info = [line for line in read_lines(base.with_name(f"{base.name}_intermediate_info.jsonl"))
                    if json.loads(line)["qid"] not in drop]
        sub_qrels = [line for line in read_lines(qrels_base(trqa, split).with_suffix(".txt"))
                     if line.split()[0] not in drop]
        info += sub_info
        qrels += sub_qrels
        info_of = {str(r["qid"]): r for r in map(json.loads, sub_info)}
        docs_of: Dict[str, List[str]] = {}
        for line in sub_qrels:
            qid, _, doc_id, rel = line.split()
            if float(rel) > 0:
                docs_of.setdefault(qid, []).append(doc_id)
        gold += [annotations.set_row(str(row["id"]), f"trqa_{subset}", row["text"], str(row["answer"]),
                         info_of.get(str(row["id"]), {}), docs_of.get(str(row["id"]), []))
                 for row in rows if row["id"] not in drop]

    if len(ids) != len(set(ids)):
        raise SystemExit("query ids collide between the TRQA validation splits")

    # The texts first: a stopped run leaves the files of the last full one.
    annotations.check(gold)
    gold = fill_texts(gc, paths, gold)
    info_path = paths.split_info(paths.split)
    for path, lines in ((paths.queries, queries), (info_path, info), (paths.qrels, qrels)):
        atomic_write_text(path, "".join(lines))
    annotations.write(paths.annotations(paths.split), gold)

    with_qrels = len({line.split()[0] for line in qrels})
    stats = annotation_stats(gold, ids)
    write_stats(paths.stats, "build", {"queries": len(queries), "with_qrels": with_qrels, "qrels": len(qrels),
                                       "left_out": {"test_overlap": overlap, "ambiguous_ids": ambiguous},
                                       "annotations": stats})
    print(f"[trqa_validation] {len(queries)} queries ({with_qrels} with qrels), {len(qrels)} qrels -> {paths.root}")
    print(f"[trqa_validation] gold annotation\n\n{format_annotation_table(stats)}\n")
    if overlap:
        print(f"[trqa_validation] left out, text also a TRQA test query: {', '.join(overlap)}")
    if ambiguous:
        print(f"[trqa_validation] left out, id of two different questions ({2 * len(ambiguous)} queries): {', '.join(ambiguous)}")
