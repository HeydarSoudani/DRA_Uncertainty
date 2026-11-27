"""The relevant passages of a training dataset's questions from their facts
(``facts.py``) and the judge (``judge.py``): one code path for
infoseek_hard, frames and webshaper.  The gold pages are given by the
dataset; the facts only help locate the relevant passages in them, and
every gold page ends with at least one relevant passage (the one
exception: an empty web page, below).

    facts     frames, webshaper: the dataset's LLM call
              (``facts.fill_facts``, ``facts_and_gold``); a question whose
              call failed for good is dropped (``facts_failed``), one whose
              call failed is left for the next run.  infoseek_hard: the
              claims of its tree, written by code (``gold_and_report``)
    routes    the pages each fact is judged on: its pages (infoseek_hard:
              its claim's page; frames: its own; webshaper: those the LLM
              maps it to).  With ``spread`` (webshaper), a fact without a
              page is judged on every page, and a page no fact is judged on
              gets every fact (``routes``)
    window    the passages judged on a page: its first *window* passages
              (the dataset's ``CHUNK_WINDOW``: infoseek_hard 2, as InfoSeek
              writes its claims from the first 1024 tokens of a page;
              frames and webshaper 10); none on a Wikipedia page whose
              window holds one passage (it is the page's gold passage
              either way, ``window_docs``)
    judge     one call per (fact, passage in the window of its routed
              pages) (``judge.judge_pairs``); a passage is relevant when the
              judge finds the fact in it (YES-SAME, or a value extracted:
              VALUE, ``judge.relevant``); a pair without a label leaves the
              question unfinished (next run)
    pages     the relevant passages of a gold page (``SOURCES``): its
              judged relevant passages, else its first passage, except a
              web page (``web:`` key) whose every judged pair is unrelated
              (NO-UNRELATED, NONE-UNRELATED: a blocked, login or error
              page, or one about nothing the facts state), which gets
              none (``empty_web``, not a gold page); a question left with
              no gold page is dropped (``no_gold_page``)
    qrels     binary: every relevant passage (relevance 1); the row's
              ``doc_ids`` are the same passages in page order

Report: ``gold_passages_report.jsonl`` (``DatasetPaths.gold_report``),
rewritten each run, one row per kept question: each fact with its routed
pages, the number of passages judged (0: only one-passage Wikipedia
pages, ``facts_not_judged`` in the stats), its relevant passages (label,
extracted value) and, for a known value, the passages that state another
value (YES-DIFFERENT); the
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
from .judge import DIFFERENT, UNRELATED_LABELS, fact_key, judge_pairs, label_counts, relevant

# Why a question is dropped: its facts failed for good; none of its gold
# pages is left (every one an empty web page).
FACTS_FAILED = "facts_failed"
NO_GOLD_PAGE = "no_gold_page"
# Where the relevant passages of a gold page come from: the judge, else
# the page's first passage; an empty web page has none.
JUDGED, FIRST, EMPTY_WEB = "judged", "chunk_1", "empty_web"
SOURCES = (JUDGED, FIRST, EMPTY_WEB)
# The page keys of web pages (``corpus/pages.py``).
WEB_KEY_PREFIX = "web:"


def routes(facts: List[dict], chunks: Dict[str, List[str]], spread: bool = True
           ) -> Tuple[Dict[str, List[str]], int, int]:
    """The gold pages each fact is judged on (*chunks*: page key -> passage
    ids, in page order): its own pages; with *spread*, every page for a
    fact without one, and a page no fact is judged on is added to every
    fact.  Returns the routes by fact id, the facts sent to every page and
    the pages added to every fact."""
    pages = [key for key, ids in chunks.items() if ids]
    out: Dict[str, List[str]] = {}
    to_all = 0
    for f in facts:
        own = [key for key in dict.fromkeys(f["pages"]) if key in chunks and chunks[key]]
        if not own and spread:
            own = list(pages)
            to_all += 1
        out[f["id"]] = own
    if not spread or not facts:
        return out, to_all, 0
    covered = {key for keys in out.values() for key in keys}
    orphans = [key for key in pages if key not in covered]
    if orphans:
        out = {fid: keys + orphans for fid, keys in out.items()}
    return out, to_all, len(orphans)


def window_docs(keys: List[str], chunks: Dict[str, List[str]], window: int) -> List[Tuple[str, str]]:
    """``(page key, passage id)`` of the passages judged on the pages
    *keys*, in page order: the first *window* of each, none of a Wikipedia
    page whose window holds one passage (its first passage is its gold
    one, judged or not; a web page is judged for ``empty_web``)."""
    out = []
    for key in keys:
        ids = chunks.get(key, [])[:window]
        if len(ids) == 1 and not key.startswith(WEB_KEY_PREFIX):
            continue
        out += [(key, d) for d in ids]
    return list(dict.fromkeys(out))


def page_sources(chunks: Dict[str, List[str]], docs: Set[str],
                 page_labels: Dict[str, List[str]]) -> Dict[str, str]:
    """The source of the relevant passages of each gold page (``SOURCES``)
    with *docs* the judged relevant passages and *page_labels* the labels
    of the pairs judged on each page; *docs* gets the first passage of each
    page without a judged relevant one, except an empty web page."""
    sources = {}
    for key, ids in chunks.items():
        if not ids:
            continue
        labels = page_labels.get(key) or []
        if any(d in docs for d in ids):
            sources[key] = JUDGED
        elif key.startswith(WEB_KEY_PREFIX) and labels and all(label in UNRELATED_LABELS for label in labels):
            sources[key] = EMPTY_WEB
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


def judge_questions(gc: GoldPassagesConfig, paths: DatasetPaths, items: List[dict], window: int,
                    spread: bool = True) -> Tuple[List[dict], List[dict], List[str], dict]:
    """*items*: one per question, ``{"record", "chunks": {page key: passage
    ids}, "facts": [...]}``.  Returns the finished items (each with
    ``docs``: its relevant passages, the first passages of the pages
    without a judged one included, ``judged``: fact id -> its judged
    relevant passages, ``sources``: page key -> ``SOURCES``, ``routes``:
    fact id -> its pages, ``report``: the facts as in the report), the
    dropped questions (``no_gold_page``), the ids of the unfinished ones
    and the stats of the judgments."""
    for item in items:
        item["routes"], item["to_every_page"], item["added_pages"] = routes(item["facts"], item["chunks"], spread)
    pairs = [(f, d) for item in items for f in item["facts"]
             for _, d in window_docs(item["routes"][f["id"]], item["chunks"], window)]
    values: Dict[Tuple[str, str], str] = {}
    labels = judge_pairs(gc, paths, pairs, values=values) if pairs else {}
    kept, dropped, pending = [], [], []
    # The relevant passages of each judged fact; the facts with no pair (on
    # one-passage Wikipedia pages only).
    per_fact: List[int] = []
    not_judged = 0
    for item in items:
        record, chunks = item["record"], item["chunks"]
        judged: Dict[str, List[str]] = {}
        page_labels: Dict[str, List[str]] = defaultdict(list)
        report = []
        unfinished = False
        for f in item["facts"]:
            key = fact_key(f)
            got = [(page, d, labels.get((key, d)))
                   for page, d in window_docs(item["routes"][f["id"]], chunks, window)]
            if any(label is None for _, _, label in got):
                unfinished = True
                break
            for page, _, label in got:
                page_labels[page].append(label)
            judged[f["id"]] = [d for _, d, label in got if relevant(label)]
            report.append({"id": f["id"], "statement": f["statement"], "value": f["value"],
                           "value_of": f.get("value_of"), "pages": f["pages"], "routed": item["routes"][f["id"]],
                           "judged": len(got),
                           "relevant": [{"doc_id": d, "label": label,
                                         **({"value": values[(key, d)]} if (key, d) in values else {})}
                                        for _, d, label in got if relevant(label)],
                           "different": [d for _, d, label in got if label == DIFFERENT]})
        if unfinished:
            pending.append(record["id"])
            continue
        docs = {d for ids in judged.values() for d in ids}
        sources = page_sources(chunks, docs, page_labels)
        if not docs:
            dropped.append({"id": record["id"], "source": record["source"],
                            "reasons": [{"status": NO_GOLD_PAGE, "pages": len(sources)}]})
            continue
        for f, row in zip(item["facts"], report):
            if row["judged"]:
                per_fact.append(len(judged[f["id"]]))
            else:
                not_judged += 1
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
    stats = {"window": window, "pairs": len(unique), "labels": label_counts(labels.get(k) for k in unique),
             "facts": len(facts),
             "facts_known_value": sum(f["value"] is not None for f in facts),
             "facts_unknown_value": sum(f["value"] is None for f in facts),
             "facts_to_every_page": sum(i["to_every_page"] for i in kept),
             "pages_added_to_every_fact": sum(i["added_pages"] for i in kept),
             "facts_not_judged": not_judged,
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
    return kept, dropped, pending, stats


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


def gold_and_report(cfg: BuildConfig, paths: DatasetPaths, items: List[dict], window: int, spread: bool = True
                    ) -> Tuple[List[dict], List[dict], List[str], dict]:
    """``judge_questions`` on *items*, then the report.  Returns the kept
    items, the dropped questions, the ids of the unfinished ones and the
    ``gold_passages`` section of ``stats.json``."""
    kept, dropped, pending, judge_stats = judge_questions(cfg.gold_passages, paths, items, window, spread)
    write_report(paths, kept)
    section = {"judge": {"model": cfg.gold_passages.model}, **judge_stats,
               "qrels": sum(len(item["docs"]) for item in kept)}
    return kept, dropped, pending, section


def facts_and_gold(cfg: BuildConfig, paths: DatasetPaths, spec: FactSpec, found: List[Tuple[dict, List[dict]]],
                   window: int, spread: bool = True) -> Tuple[List[dict], List[dict], List[str], dict]:
    """The facts (the dataset's LLM) and gold passages of the questions of
    *found* (record, its gold pages as the fact LLM sees them; each record
    holds the passage ids of its pages, ``features.page_chunks``).  Returns
    as ``gold_and_report``, with the fact LLM in the section."""
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
    kept, no_gold, unjudged, section = gold_and_report(cfg, paths, items, window, spread)
    section = {"judge": section.pop("judge"), "fact_llm": {"model": cfg.facts.model}, **section}
    return kept, dropped + no_gold, pending + unjudged, section
