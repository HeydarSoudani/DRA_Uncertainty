"""Counts written to ``stats.json``, one section per step: questions per
source, gold pages per question, drop reasons, and (step 2) page fetch
outcomes and lengths, (step 3) the dataset as written."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def distribution(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered), "mean": round(statistics.mean(ordered), 2), "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "min": ordered[0], "max": ordered[-1],
    }


def question_stats(records: List[dict]) -> dict:
    """Per source (and ``all``): questions, gold pages and facts per question,
    gold pages by kind."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        groups[record["source"]].append(record)
        groups["all"].append(record)
    return {
        source: {
            "questions": len(group),
            "gold_per_question": distribution([len(r["gold"]) for r in group]),
            "facts_per_question": distribution([len(r["facts"]) for r in group]),
            "gold_by_kind": dict(Counter(_gold_kind(g) for r in group for g in r["gold"])),
            "ignored_urls": sum(r["meta"].get("ignored_urls", 0) for r in group),
        }
        for source, group in sorted(groups.items())
    }


def _gold_kind(gold: dict) -> str:
    if gold["kind"] == "wiki":
        return "wiki_en" if gold["lang"] == "en" else "wiki_other_lang"
    return "web"


def drop_stats(dropped: Iterable[dict]) -> dict:
    """Per source: the questions, and per reason the questions with it (a
    question with several failed pages counts once per distinct reason)."""
    out: Dict[str, dict] = defaultdict(lambda: {"questions": 0, "by_reason": Counter()})
    for item in dropped:
        out[item["source"]]["questions"] += 1
        reasons = {r if isinstance(r, str) else r["status"] for r in item["reasons"]}
        out[item["source"]]["by_reason"].update(reasons)
    return {s: {"questions": v["questions"], "by_reason": dict(v["by_reason"].most_common())}
            for s, v in sorted(out.items())}


def page_stats(pages: Iterable[dict]) -> dict:
    """Per page kind: status counts, how pages were fetched, redirects, and the
    word count of the fetched pages."""
    by_kind: Dict[str, List[dict]] = defaultdict(list)
    for page in pages:
        by_kind[page["kind"]].append(page)
    out = {}
    for kind, rows in sorted(by_kind.items()):
        ok = [r for r in rows if r["status"] == "ok"]
        out[kind] = {
            "pages": len(rows),
            "status": dict(Counter(r["status"] for r in rows).most_common()),
            "fetched_via": dict(Counter(r.get("fetched_via") for r in ok)),
            "redirected": sum(bool(r.get("redirected")) for r in rows),
            "unique_pageids": len({r["pageid"] for r in ok if r.get("pageid")}) if kind == "wiki" else None,
            "words": distribution([r.get("words", 0) for r in ok]),
        }
    return out


def write_stats(path: Path, step: int, stats: dict) -> None:
    """Set the ``step{N}`` section of *path*, the other sections kept."""
    sections = json.loads(path.read_text()) if path.exists() else {}
    sections[f"step{step}"] = stats
    path.write_text(json.dumps(dict(sorted(sections.items())), indent=2, ensure_ascii=False) + "\n")


def dataset_stats(candidates: List[dict], queries: List[dict], qrels: Dict[str, List[str]],
                  corpus: Dict[str, dict], dropped: List[dict]) -> dict:
    """Step 3: the dataset as written (queries, qrels, corpus), per source
    (and ``all``), plus consistency checks between the three files.

    *candidates* are the step 1 questions, *qrels* maps a query id to its doc
    ids, *corpus* maps a doc id to its row and *dropped* are the step 2 drops.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for query in queries:
        groups[query["source"]].append(query)
        groups["all"].append(query)
    n_candidates = Counter(r["source"] for r in candidates)
    n_dropped = Counter(r["source"] for r in dropped)
    for counter in (n_candidates, n_dropped):
        counter["all"] = sum(counter.values())

    per_source = {}
    for source in sorted(set(groups) | set(n_candidates)):
        group = groups.get(source, [])
        doc_ids = [d for q in group for d in qrels.get(q["id"], [])]
        uses = Counter(doc_ids)
        per_source[source] = {
            "candidates": n_candidates[source],
            "processed": len(group) + n_dropped[source],
            "kept": len(group),
            "dropped": n_dropped[source],
            "qrels": len(doc_ids),
            "docs_per_query": distribution([len(qrels.get(q["id"], [])) for q in group]),
            "unique_docs": len(uses),
            "docs_by_kind": dict(Counter(d.split("_", 1)[0] for d in uses)),
            "docs_shared": sum(n >= 2 for n in uses.values()),
            "doc_words": distribution([len(corpus[d]["contents"].split()) for d in uses if d in corpus]),
            "query_words": distribution([len(q["text"].split()) for q in group]),
            "answer_words": distribution([len(str(q["answer"]).split()) for q in group]),
        }

    query_ids = {q["id"] for q in queries}
    qrel_docs = {d for docs in qrels.values() for d in docs}
    checks = {
        "duplicate_query_ids": len(queries) - len(query_ids),
        "queries_without_qrels": sum(not qrels.get(q) for q in query_ids),
        "qrels_unknown_query": sum(q not in query_ids for q in qrels),
        "qrels_docs_missing_from_corpus": len(qrel_docs - set(corpus)),
        "corpus_docs_without_qrels": len(set(corpus) - qrel_docs),
    }
    return {"corpus_docs": len(corpus), "per_source": per_source, "checks": checks}


def _fmt(value: float) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _dist(d: dict, keys: Tuple[str, ...]) -> str:
    return " / ".join(_fmt(d[k]) for k in keys) if d["n"] else "-"


def format_dataset_table(stats: dict) -> str:
    """The step 3 stats as a plain-text table: one row per measure, one
    column per source (``all`` last), then the corpus size and the checks."""
    per_source = stats["per_source"]
    sources = [s for s in per_source if s != "all"] + ["all"]
    rows = [
        ("candidates (step 1)", lambda s: s["candidates"]),
        ("processed (step 2)", lambda s: s["processed"]),
        ("dropped (Filter A)", lambda s: s["dropped"]),
        ("queries (kept)", lambda s: s["kept"]),
        ("qrels pairs", lambda s: s["qrels"]),
        ("docs/query mean/med/min/max", lambda s: _dist(s["docs_per_query"], ("mean", "median", "min", "max"))),
        ("unique docs", lambda s: s["unique_docs"]),
        ("  wiki", lambda s: s["docs_by_kind"].get("wiki", 0)),
        ("  web", lambda s: s["docs_by_kind"].get("web", 0)),
        ("docs shared by >=2 queries", lambda s: s["docs_shared"]),
        ("doc words mean/med/p95", lambda s: _dist(s["doc_words"], ("mean", "median", "p95"))),
        ("query words mean/med/max", lambda s: _dist(s["query_words"], ("mean", "median", "max"))),
        ("answer words mean/med", lambda s: _dist(s["answer_words"], ("mean", "median"))),
    ]
    table = [[""] + sources] + [[label] + [_fmt(get(per_source[s])) for s in sources] for label, get in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
             for row in table]
    lines.insert(1, "-" * len(lines[0]))
    lines.append("")
    lines.append(f"corpus docs: {stats['corpus_docs']}")
    lines.append("checks (all 0 = consistent): "
                 + ", ".join(f"{k} {v}" for k, v in stats["checks"].items()))
    return "\n".join(lines)
