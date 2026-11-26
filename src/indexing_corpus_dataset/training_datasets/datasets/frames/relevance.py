"""frames, stage data_relevance: the facts, gold passages and binary qrels
of the prepared FRAMES questions (stage data_prepare: ``prepare.py``; each
record holds the chunk ids of its gold pages in corpus_wiki,
``features.page_chunks``).

    facts      one fact per gold page, written by an LLM from the question,
               the answer and the page titles (``facts.py``): the property
               the question needs from the page; its value is unknown
    gold       each fact judged against every chunk of its page, one call
               per (fact, chunk) pair (``gold_passages/select.facts_and_gold``),
               the judge extracting the value; a chunk is relevant when
               the judge finds the value in it; a page without a judged
               relevant chunk gets its first chunk
    rows       binary qrels (the relevant chunks); the row holds the
               question, the answer and the same chunks (split like every
               dataset, ``pipeline/splits.py``)

A question whose facts or judgments failed (a call error) stays
unprocessed and is taken first by the next run.
"""

from typing import List

from ...annotation import rows as annotation_rows
from ...config import BuildConfig, DatasetPaths
from ...gold_passages.select import facts_and_gold, gold_docs, qrels_of
from ...pipeline.relevance import Built
from .facts import SPEC

FACTS = SPEC
JUDGE = True


def build(cfg: BuildConfig, paths: DatasetPaths, processed: List[dict]) -> Built:
    found = [(record, [dict(g) for g in record["gold"]]) for record in processed]
    kept, dropped, pending, section = facts_and_gold(cfg, paths, SPEC, found)
    return Built(rows=[annotation_rows.row(item["record"], gold_docs(item)) for item in kept],
                 qrels=qrels_of(kept), kept=[item["record"] for item in kept], dropped=dropped, pending=pending,
                 pending_reason="with a failed call (facts or judge)", sections={"gold_passages": section})
