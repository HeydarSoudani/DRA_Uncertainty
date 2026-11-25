"""``stats.json`` of a training dataset, one section per stage (``prepare``,
``relevance``, ...), and the counts of the build: questions per source,
gold pages per question, drop reasons; ``dropped.jsonl``.  The
``annotations`` section is written by ``annotation/stats.py``, the
``split`` section by ``splits.py``."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from utils.io import atomic_write_text, iter_jsonl, read_jsonl, write_jsonl


def distribution(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered), "mean": round(statistics.mean(ordered), 2), "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "min": ordered[0], "max": ordered[-1],
    }


def question_stats(records: List[dict]) -> dict:
    """Per source (and ``all``): questions, gold pages per question, gold
    pages by kind."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        groups[record["source"]].append(record)
        groups["all"].append(record)
    return {
        source: {
            "questions": len(group),
            "gold_per_question": distribution([len(r["gold"]) for r in group]),
            "gold_by_kind": dict(Counter(_gold_kind(g) for r in group for g in r["gold"])),
            "ignored_urls": sum(r["meta"].get("ignored_urls", 0) for r in group),
        }
        for source, group in sorted(groups.items())
    }


def _gold_kind(gold: dict) -> str:
    if gold["kind"] == "wiki":
        return "wiki_en" if gold["lang"] == "en" else "wiki_other_lang"
    return gold["kind"]


def drop_stats(dropped: Iterable[dict]) -> dict:
    """Per source: the questions, and per reason the questions with it (a
    question with several failed pages counts once per distinct reason)."""
    out: Dict[str, dict] = defaultdict(lambda: {"questions": 0, "by_reason": Counter()})
    for item in dropped:
        out[item["source"]]["questions"] += 1
        reasons = {r["status"] for r in item["reasons"]}
        out[item["source"]]["by_reason"].update(reasons)
    return {s: {"questions": v["questions"], "by_reason": dict(v["by_reason"].most_common())}
            for s, v in sorted(out.items())}


def read_stats(path: Path) -> dict:
    """The sections of *path* (``{}`` when it does not exist)."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_stats(path: Path, section: str, stats: dict) -> None:
    """Set *section* (a build stage, ``annotations``, ``split``) of *path*,
    the other sections kept."""
    sections = read_stats(path)
    sections[section] = stats
    atomic_write_text(path, json.dumps(dict(sorted(sections.items())), indent=2, ensure_ascii=False) + "\n")


def processed_ids(paths, stage: str) -> set:
    """The questions processed by earlier runs of stage data_relevance on a
    dataset (*paths*, ``DatasetPaths``): kept (the queries file) or dropped
    by *stage* (``dropped.jsonl``)."""
    done = {str(r["id"]) for r in iter_jsonl(paths.queries)} if paths.queries.exists() else set()
    if paths.dropped.exists():
        done |= {str(r["id"]) for r in read_jsonl(paths.dropped) if r["stage"] == stage}
    return done


def write_dropped(path: Path, stage: str, dropped: List[dict]) -> None:
    """Replace the rows of *stage* in the dropped questions of *path*
    (``dropped.jsonl``), the other stages' kept."""
    others = [r for r in read_jsonl(path) if r["stage"] != stage] if path.exists() else []
    write_jsonl(path, others + [{"stage": stage, **d} for d in dropped])
