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
from typing import List, Set

from indexing_corpus_dataset.layout import dataset_root, qrels_base, queries_base, resolve_split_id

from .config import PoolPaths

SUBSETS = ("wiki1", "wiki2")


def _lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh if line.strip()]


def _test_texts(trqa: Path) -> Set[str]:
    texts = set()
    for subset in SUBSETS:
        path = queries_base(trqa, resolve_split_id("trqa", "test", subset)).with_suffix(".jsonl")
        texts |= {json.loads(line)["text"] for line in _lines(path)}
    return texts


def build(paths: PoolPaths) -> None:
    trqa = dataset_root("trqa")
    test_texts = _test_texts(trqa)
    queries, info, qrels, overlap, ambiguous = [], [], [], [], []
    for subset in SUBSETS:
        split = resolve_split_id("trqa", "validation", subset)
        base = queries_base(trqa, split)
        rows = _lines(base.with_suffix(".jsonl"))
        counts = Counter(json.loads(line)["id"] for line in rows)
        dup = {qid for qid, n in counts.items() if n > 1}
        test = {json.loads(line)["id"] for line in rows if json.loads(line)["text"] in test_texts} - dup
        ambiguous += sorted(dup)
        overlap += sorted(test)
        drop = dup | test
        queries += [line for line in rows if json.loads(line)["id"] not in drop]
        info += [line for line in _lines(base.with_name(f"{base.name}_intermediate_info.jsonl"))
                 if json.loads(line)["qid"] not in drop]
        qrels += [line for line in _lines(qrels_base(trqa, split).with_suffix(".txt"))
                  if line.split()[0] not in drop]

    ids = [json.loads(line)["id"] for line in queries]
    if len(ids) != len(set(ids)):
        raise SystemExit("query ids collide between the TRQA validation splits")

    info_path = paths.split_info(paths.split)
    for path, lines in ((paths.queries, queries), (info_path, info), (paths.qrels, qrels)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")

    with_qrels = len({line.split()[0] for line in qrels})
    print(f"[trqa_validation] {len(queries)} queries ({with_qrels} with qrels), {len(qrels)} qrels -> {paths.root}")
    if overlap:
        print(f"[trqa_validation] left out, text also a TRQA test query: {', '.join(overlap)}")
    if ambiguous:
        print(f"[trqa_validation] left out, id of two different questions ({2 * len(ambiguous)} queries): {', '.join(ambiguous)}")
