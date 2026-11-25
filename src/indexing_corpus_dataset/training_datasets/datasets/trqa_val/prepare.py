"""trqa_val: the checks of stage data_prepare, in order.

    ambiguous_id   an id that names two different questions in one TRQA
                   split (their qrels and intermediate info cannot be told
                   apart): both dropped (``ambiguous_id``)
    test_overlap   a question that is also a TRQA test question
                   (``test_overlap``)
    duplicate      a repeated question (its first row kept)
    answer         empty (``answer_invalid``)
    no_gold        no qrels (``no_gold``)
    pages          a qrels document not in TRQA's corpus (wiki_partial), or
                   without text (``page_not_in_corpus``, ``page_empty``)
"""

from collections import Counter
from typing import List, Optional

from indexing_corpus_dataset.shared.layout import dataset_root, default_corpus_path, queries_base, resolve_split_id
from utils.io import iter_jsonl

from ....shared.doc_lookup import CorpusLookup
from ...pipeline.checks import answer_check, duplicate_check, no_gold_check, question_key
from ...pipeline.funnel import Funnel
from .source import TRQA_SUBSETS


def _test_texts() -> set:
    trqa = dataset_root("trqa")
    return {question_key(row["text"]) for s in TRQA_SUBSETS
            for row in iter_jsonl(queries_base(trqa, resolve_split_id("trqa", "test", s)).with_suffix(".jsonl"))}


def run(cfg, paths, funnel: Funnel) -> dict:
    counts = Counter((r["source"], r["id"]) for r in funnel.kept)
    if len({r["id"] for r in funnel.kept}) != len({(r["source"], r["id"]) for r in funnel.kept}):
        raise SystemExit("query ids collide between the TRQA validation splits")
    funnel.apply("ambiguous_id", lambda r: [{"status": "ambiguous_id"}] if counts[(r["source"], r["id"])] > 1 else None)
    test = _test_texts()
    funnel.apply("test_overlap", lambda r: [{"status": "test_overlap"}] if question_key(r["question"]) in test else None)
    funnel.apply("duplicate", duplicate_check())
    funnel.apply("answer", answer_check)
    funnel.apply("no_gold", no_gold_check)

    corpus = default_corpus_path("trqa_val")
    texts = CorpusLookup(corpus).get({g["id"] for r in funnel.kept for g in r["gold"]})

    def pages(record: dict) -> Optional[List[dict]]:
        reasons = []
        for g in record["gold"]:
            if g["id"] not in texts:
                reasons.append({"status": "page_not_in_corpus", "doc": g["id"]})
            elif not texts[g["id"]]["text"].strip():
                reasons.append({"status": "page_empty", "doc": g["id"]})
        return reasons or None

    funnel.apply("pages", pages)
    for record in funnel.kept:
        record["features"] = {}
    return {"corpus": str(corpus)}
