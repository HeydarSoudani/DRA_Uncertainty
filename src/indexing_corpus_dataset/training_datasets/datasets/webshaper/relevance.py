"""webshaper, stage data_relevance: the facts, gold passages and binary
qrels of the prepared WebShaper questions (stage data_prepare:
``prepare.py``; each record holds its gold pages found in the corpus,
``features.pages``, and their chunk ids in corpus_wiki or corpus_web,
``features.page_chunks``).

    facts      one fact per triple of the formalization, written by code
               (``facts.py``: no name or value the dataset does not state);
               a question without any fact is dropped (``no_facts``, no
               call);
               one LLM call per question maps each fact to the gold pages
               likely to state it, from the pages' titles (a web page by
               its fetched title)
    gold       each fact judged against every passage of its pages, one
               call per (fact, passage) pair
               (``gold_passages/select.facts_and_gold``): a known value
               checked, an unknown one extracted; a fact without a page is
               judged on every page, a page no fact maps to on every fact;
               a passage is relevant when the judge finds the fact in it; a
               page without a judged relevant passage gets its first
               passage
    rows       binary qrels (the relevant passages); the row holds the
               question, the answer and the same passages

A question whose facts or judgments are not final stays unprocessed and is
taken first by the next run.
"""

from collections import Counter
from typing import Dict, List

from utils.io import iter_jsonl

from ...annotation import rows as annotation_rows
from ...config import BuildConfig, DatasetPaths
from ...gold_passages.select import facts_and_gold, gold_docs, qrels_of
from ...pipeline.relevance import Built
from .facts import SPEC, build_facts

FACTS = SPEC
JUDGE = True
# Why a question is dropped when none of its triples makes a fact.
NO_FACTS = "no_facts"


def _web_titles(paths: DatasetPaths) -> Dict[str, str]:
    """The fetched title of each web page by URL (the latest row wins)."""
    if not paths.web_pages.exists():
        return {}
    titles = {}
    for r in iter_jsonl(paths.web_pages):
        if r.get("title"):
            titles[r["url"]] = str(r["title"]).strip()
    return titles


def build(cfg: BuildConfig, paths: DatasetPaths, processed: List[dict]) -> Built:
    titles = _web_titles(paths)
    built = {record["id"]: build_facts(record) for record in processed}
    no_facts = [{"id": r["id"], "source": r["source"], "reasons": [{"status": NO_FACTS}]}
                for r in processed if not built[r["id"]][0]]
    found = [(record, [{**p, "title": titles.get(p["url"])} if p["kind"] == "web" else p
                       for p in record["features"]["pages"]]) for record in processed if built[record["id"]][0]]
    kept, dropped, pending, section = facts_and_gold(cfg, paths, SPEC, found)
    triples = Counter()
    for item in kept:
        triples.update(built[item["record"]["id"]][1])
    section["triples"] = {k: triples[k] for k in ("triples", "duplicate", "identity", "skipped", "no_known_term")}
    section["web_pages_without_title"] = len({p["url"] for item in kept for p in item["record"]["features"]["pages"]
                                              if p["kind"] == "web" and not titles.get(p["url"])})
    rows = [annotation_rows.row(item["record"], gold_docs(item)) for item in kept]
    return Built(rows=rows, qrels=qrels_of(kept), kept=[item["record"] for item in kept], dropped=no_facts + dropped,
                 pending=pending, pending_reason="with facts or judgments not final",
                 sections={"gold_passages": section})
