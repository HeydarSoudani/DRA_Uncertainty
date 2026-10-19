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
searches TRQA's ``wiki_partial`` (``layout._SHARED_CORPUS``).  The build
then splits ``all`` into ``sft``, ``rl`` and ``val`` (``splits.py``).
"""

import json
from collections import Counter
from pathlib import Path
from typing import Set

from indexing_corpus_dataset.layout import dataset_root, qrels_base, queries_base, resolve_split_id
from utils.io_utils import atomic_write_text, iter_jsonl

from .config import PoolPaths
from .splits import read_lines

SUBSETS = ("wiki1", "wiki2")


def _test_texts(trqa: Path) -> Set[str]:
    texts = set()
    for subset in SUBSETS:
        path = queries_base(trqa, resolve_split_id("trqa", "test", subset)).with_suffix(".jsonl")
        texts |= {row["text"] for row in iter_jsonl(path)}
    return texts


def build(paths: PoolPaths) -> None:
    trqa = dataset_root("trqa")
    test_texts = _test_texts(trqa)
    queries, ids, info, qrels, overlap, ambiguous = [], [], [], [], [], []
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
        info += [line for line in read_lines(base.with_name(f"{base.name}_intermediate_info.jsonl"))
                 if json.loads(line)["qid"] not in drop]
        qrels += [line for line in read_lines(qrels_base(trqa, split).with_suffix(".txt"))
                  if line.split()[0] not in drop]

    if len(ids) != len(set(ids)):
        raise SystemExit("query ids collide between the TRQA validation splits")

    info_path = paths.split_info(paths.split)
    for path, lines in ((paths.queries, queries), (info_path, info), (paths.qrels, qrels)):
        atomic_write_text(path, "".join(lines))

    with_qrels = len({line.split()[0] for line in qrels})
    print(f"[trqa_validation] {len(queries)} queries ({with_qrels} with qrels), {len(qrels)} qrels -> {paths.root}")
    if overlap:
        print(f"[trqa_validation] left out, text also a TRQA test query: {', '.join(overlap)}")
    if ambiguous:
        print(f"[trqa_validation] left out, id of two different questions ({2 * len(ambiguous)} queries): {', '.join(ambiguous)}")
