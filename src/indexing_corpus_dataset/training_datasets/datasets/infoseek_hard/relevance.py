"""infoseek_hard, stage data_relevance: the gold passages, qrels and rows
of the prepared InfoSeek-Hard questions (stage data_prepare:
``prepare.py``; each record holds the chunk ids of its gold pages in
corpus_wiki, InfoSeek's wiki-25-512, ``features.page_chunks``).

    facts      one per claim of the research tree, written by code
               (``facts.claim_facts``, no LLM): the claim with its node ids
               as the entities' names, the value the target entity
    gold       each claim judged on the first ``CHUNK_WINDOW`` chunks of
               the page of the node it expands, one call per (claim, chunk)
               (``gold_passages/select.gold_and_report``): InfoSeek writes
               every claim from the first 1024 tokens of that page, chunk 1
               or chunk 2.  Every chunk that states the claim (YES-SAME) is
               relevant; a page without one (a page without claims too: a
               leaf node) gets its first chunk.  Claims stay on their own
               page (no spread)
    rows       binary qrels (the relevant chunks); the row's ``doc_ids``
               the same chunks in page order (``annotation/rows.row``)

A question with a claim the judge gave no label yet stays unprocessed and
is taken first by the next run.  The split: ``pipeline/relevance.finish``
(no belief texts: the belief model's beliefs come from the question,
multi-clue).
"""

from typing import List

from ...annotation import rows as annotation_rows
from ...config import BuildConfig, DatasetPaths
from ...corpus.pages import wiki_gold
from ...gold_passages.select import gold_and_report, gold_docs, qrels_of
from ...pipeline.relevance import Built
from .facts import claim_facts

FACTS = None
JUDGE = True
# The chunks of a page each claim is judged on (InfoSeek's first 1024
# tokens: two 512-token chunks).
CHUNK_WINDOW = 2


def build(cfg: BuildConfig, paths: DatasetPaths, processed: List[dict]) -> Built:
    items = [{"record": r, "chunks": r["features"]["page_chunks"],
              "facts": claim_facts(r, lambda href: wiki_gold(href)["key"])} for r in processed]
    kept, dropped, pending, section = gold_and_report(cfg, paths, items, CHUNK_WINDOW, spread=False)
    return Built(rows=[annotation_rows.row(item["record"], gold_docs(item)) for item in kept],
                 qrels=qrels_of(kept), kept=[item["record"] for item in kept], dropped=dropped, pending=pending,
                 pending_reason="with a claim the judge gave no label", sections={"gold_passages": section})
