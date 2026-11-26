"""The relevant passages of the frames and webshaper questions from their
facts (``facts.py``) and the judge (``judge.py``).  The gold pages are given
by the dataset; the facts only help locate the relevant passages in them,
and every gold page ends with at least one relevant passage.

    facts     the dataset's LLM call (``facts.fill_facts``); a question
              whose call failed for good is dropped (``facts_failed``), one
              whose call failed is left for the next run
    routes    the pages each fact is judged on: its pages (frames: its
              own; webshaper: those the LLM maps it to); a fact without a
              page is judged on every page, and a page no fact is judged on
              gets every fact (``routes``)
    judge     one call per (fact, passage of its routed pages)
              (``judge.judge_pairs``); a passage is relevant when the judge
              finds the fact in it (YES-SAME, or a value extracted: VALUE,
              ``judge.relevant``); a pair without a label leaves the
              question unfinished (next run)
    pages     the relevant passages of a gold page (``SOURCES``): its
              judged relevant passages, else its first passage
    qrels     binary: every relevant passage (relevance 1); the row's
              ``doc_ids`` are the same passages in page order

Report: ``gold_passages_report.jsonl`` (``DatasetPaths.gold_report``),
rewritten each run, one row per kept question: each fact with its routed
pages, its relevant passages (label, extracted value) and, for a known
value, the passages that state another value (YES-DIFFERENT); the
``gold_passages`` section of ``stats.json`` counts them (``extracted``:
the distinct values extracted per fact with an unknown value;
``variables``, webshaper: the distinct values found per variable).
"""

import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from utils.io import atomic_write_text

from ..config import BuildConfig, DatasetPaths, GoldPassagesConfig
from .facts import FactSpec, fill_facts
from .judge import DIFFERENT, fact_key, judge_pairs, label_counts, relevant

# Why a question is dropped when its facts failed for good.
FACTS_FAILED = "facts_failed"
# Where the relevant passages of a gold page come from: the judge, else
# the page's first passage.
JUDGED, FIRST = "judged", "chunk_1"
SOURCES = (JUDGED, FIRST)


def routes(facts: List[dict], chunks: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], int, int]:
    """The gold pages each fact is judged on (*chunks*: page key -> passage
    ids, in page order): its own pages, every page for a fact without one;
    a page no fact is judged on is added to every fact.  Returns the routes
    by fact id, the facts sent to every page and the pages added to every
    fact."""
    pages = [key for key, ids in chunks.items() if ids]
    out: Dict[str, List[str]] = {}
    to_all = 0
    for f in facts:
        own = [key for key in dict.fromkeys(f["pages"]) if key in chunks and chunks[key]]
        if not own:
            own = list(pages)
            to_all += 1
        out[f["id"]] = own
    covered = {key for keys in out.values() for key in keys}
    orphans = [key for key in pages if key not in covered]
    if orphans:
        out = {fid: keys + orphans for fid, keys in out.items()}
    return out, to_all, len(orphans) if facts else 0


def route_docs(keys: List[str], chunks: Dict[str, List[str]]) -> List[str]:
    """The passages of the pages *keys*, in page order."""
    return list(dict.fromkeys(d for key in keys for d in chunks.get(key, [])))


def page_sources(chunks: Dict[str, List[str]], docs: Set[str]) -> Dict[str, str]:
    """The source of the relevant passages of each gold page (``SOURCES``)
    with *docs* the judged relevant passages; *docs* gets the first passage
    of each page without a judged relevant one."""
    sources = {}
    for key, ids in chunks.items():
        if not ids:
            continue
        if any(d in docs for d in ids):
            sources[key] = JUDGED
        else:
            sources[key] = FIRST
            docs.add(ids[0])
    return sources


def normalize_value(value: str) -> str:
    """A value for counting distinct ones: lower case, single spaces, no
    surrounding punctuation."""
    return re.sub(r"\s+", " ", value.lower()).strip(" .,;:\"'")


def _distinct_counts(groups: Iterable[Set[str]]) -> Dict[str, int]:
    out = {"0": 0, "1": 0, "2+": 0}
    for values in groups:
        out["0" if not values else "1" if len(values) == 1 else "2+"] += 1
    return out


def judge_questions(gc: GoldPassagesConfig, paths: DatasetPaths, items: List[dict]
                    ) -> Tuple[List[dict], List[str], dict]:
    """*items*: one per question, ``{"record", "chunks": {page key: passage
    ids}, "facts": [...]}``.  Returns the finished items (each with
    ``docs``: its relevant passages, the first passages of the pages
    without a judged one included, ``judged``: fact id -> its judged
    relevant passages, ``sources``: page key -> ``SOURCES``, ``routes``:
    fact id -> its pages, ``report``: the facts as in the report), the
    ids of the unfinished ones and the stats of the judgments."""
    for item in items:
        item["routes"], item["to_every_page"], item["added_pages"] = routes(item["facts"], item["chunks"])
    pairs = [(f, d) for item in items for f in item["facts"]
             for d in route_docs(item["routes"][f["id"]], item["chunks"])]
    values: Dict[Tuple[str, str], str] = {}
    labels = judge_pairs(gc, paths, pairs, values=values) if pairs else {}
    kept, pending = [], []
    per_fact: List[int] = []
    for item in items:
        record, chunks = item["record"], item["chunks"]
        judged: Dict[str, List[str]] = {}
        report = []
        unfinished = False
        for f in item["facts"]:
            key = fact_key(f)
            got: List[Tuple[str, Optional[str]]] = [(d, labels.get((key, d)))
                                                    for d in route_docs(item["routes"][f["id"]], chunks)]
            if any(label is None for _, label in got):
                unfinished = True
                break
            judged[f["id"]] = [d for d, label in got if relevant(label)]
            report.append({"id": f["id"], "statement": f["statement"], "value": f["value"],
                           "value_of": f.get("value_of"), "pages": f["pages"], "routed": item["routes"][f["id"]],
                           "relevant": [{"doc_id": d, "label": label,
                                         **({"value": values[(key, d)]} if (key, d) in values else {})}
                                        for d, label in got if relevant(label)],
                           "different": [d for d, label in got if label == DIFFERENT]})
        if unfinished:
            pending.append(record["id"])
            continue
        per_fact += [len(judged[f["id"]]) for f in item["facts"]]
        docs = {d for ids in judged.values() for d in ids}
        sources = page_sources(chunks, docs)
        kept.append({**item, "docs": docs, "judged": judged, "sources": sources, "report": report})
    unique = {(fact_key(f), d) for f, d in pairs}
    facts = [f for i in kept for f in i["facts"]]
    reported = [f for i in kept for f in i["report"]]
    unknown = [f for f in reported if f["value"] is None]
    variables: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for i in kept:
        for f in i["report"]:
            if f["value_of"]:
                variables[(i["record"]["id"], f["value_of"])] |= {normalize_value(r["value"])
                                                                  for r in f["relevant"] if "value" in r}
    stats = {"pairs": len(unique), "labels": label_counts(labels.get(k) for k in unique),
             "facts": len(facts),
             "facts_known_value": sum(f["value"] is not None for f in facts),
             "facts_unknown_value": sum(f["value"] is None for f in facts),
             "facts_to_every_page": sum(i["to_every_page"] for i in kept),
             "pages_added_to_every_fact": sum(i["added_pages"] for i in kept),
             "facts_with_relevant": sum(n > 0 for n in per_fact),
             "relevant_per_fact": _mean(per_fact),
             "facts_with_different_value": sum(bool(f["different"]) for f in reported if f["value"] is not None),
             "extracted": _distinct_counts({normalize_value(r["value"]) for r in f["relevant"] if "value" in r}
                                           for f in unknown),
             "pages": {s: sum(src == s for i in kept for src in i["sources"].values()) for s in SOURCES},
             "questions_all_chunk_1": sum(bool(i["sources"]) and all(src == FIRST for src in i["sources"].values())
                                          for i in kept),
             "unfinished_questions": len(pending)}
    if variables:
        stats["variables"] = _distinct_counts(variables.values())
    return kept, pending, stats


def _mean(values: List[int]) -> Optional[float]:
    return round(sum(values) / len(values), 2) if values else None


def gold_docs(item: dict) -> List[str]:
    """The relevant passages of a finished item, in page order."""
    return [d for ids in item["chunks"].values() for d in ids if d in item["docs"]]


def qrels_of(kept: List[dict]) -> Dict[str, List[str]]:
    """The binary qrels lines of each finished item, by question id."""
    return {item["record"]["id"]: [f"{item['record']['id']} 0 {d} 1\n" for d in gold_docs(item)] for item in kept}


def write_report(paths: DatasetPaths, kept: List[dict]) -> None:
    """``gold_passages_report.jsonl``: one row per kept question, its facts
    and the source of each gold page's relevant passages."""
    atomic_write_text(paths.gold_report, "".join(
        json.dumps({"id": item["record"]["id"], "facts": item["report"], "pages": item["sources"],
                    "qrels": gold_docs(item)}, ensure_ascii=False) + "\n" for item in kept))


def facts_and_gold(cfg: BuildConfig, paths: DatasetPaths, spec: FactSpec, found: List[Tuple[dict, List[dict]]]
                   ) -> Tuple[List[dict], List[dict], List[str], dict]:
    """The facts and gold passages of the questions of *found* (record, its
    gold pages as the fact LLM sees them; each record holds the passage
    ids of its pages, ``features.page_chunks``).  Returns the kept items
    (``judge_questions``), the dropped questions, the ids of the unfinished
    ones and the ``gold_passages`` section of ``stats.json``; writes the
    report."""
    facts = fill_facts(cfg.facts, paths, spec, found)
    items, pending, dropped = [], [], []
    for record, _ in found:
        got = facts.get(record["id"])
        if got is None:
            pending.append(record["id"])
        elif not got["ok"]:
            dropped.append({"id": record["id"], "source": record["source"],
                            "reasons": [{"status": FACTS_FAILED, "errors": got["errors"]}]})
        else:
            items.append({"record": record, "chunks": record["features"]["page_chunks"], "facts": got["facts"]})
    kept, unjudged, judge_stats = judge_questions(cfg.gold_passages, paths, items)
    write_report(paths, kept)
    section = {"judge": {"model": cfg.gold_passages.model}, "fact_llm": {"model": cfg.facts.model},
               **judge_stats, "qrels": sum(len(item["docs"]) for item in kept)}
    return kept, dropped, pending + unjudged, section
