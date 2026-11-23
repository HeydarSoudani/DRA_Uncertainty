"""webshaper, stage data_relevance: the facts, gold passages, graded qrels
and gold annotation of the prepared WebShaper questions (stage
data_prepare: ``prepare.py``; each record holds its gold pages found in the
corpus, ``features.pages``, and their chunk ids in corpus_wiki or
corpus_web, ``features.page_chunks``).

    facts      one LLM call per question (``facts.py``): the name of each
               variable of the formalization (when the question, answer,
               triples, page titles or URLs show it) and the pages of each
               triple; one fact per triple
    gold       each fact judged against every passage of its pages, one
               call per (fact, passage) pair
               (``gold_passages/select.facts_and_gold``); a question is kept
               when every fact has a gold passage, else dropped
               (``fact_without_gold``)
    rows       graded qrels (2: the passage states a fact; 1: on a fact's
               topic); one belief per variable (``source.skeleton``) with
               the variable's name as its answer and the gold passages of
               its triples as its documents

A question whose facts or judgments are not final stays unprocessed and is
taken first by the next run.
"""

from typing import List

from ...annotation import rows as annotation_rows
from ...config import BuildConfig, DatasetPaths
from ...gold_passages.select import facts_and_gold, gold_docs, qrels_of, ref_docs
from ...pipeline.relevance import Built
from .facts import SPEC

FACTS = SPEC
JUDGE = True


def build(cfg: BuildConfig, paths: DatasetPaths, processed: List[dict]) -> Built:
    found = [(record, record["features"]["pages"]) for record in processed]
    kept, dropped, pending, section = facts_and_gold(cfg, paths, SPEC, found)
    rows = [annotation_rows.row(item["record"], gold_docs(item), ref_docs(item),
                                {v: [name] for v, name in item["names"].items()})
            for item in kept]
    return Built(rows=rows, qrels=qrels_of(kept), kept=[item["record"] for item in kept], dropped=dropped,
                 pending=pending, pending_reason="with facts or judgments not final",
                 sections={"gold_passages": section})
