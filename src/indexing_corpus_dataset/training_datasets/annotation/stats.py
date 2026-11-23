"""The ``annotations`` section of a training dataset's ``stats.json``: the
gold annotation of its rows (``rows.py``) per source and the writer of
their belief texts (``belief_texts.py``), and its plain-text table."""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from ..pipeline.stats import distribution, write_stats
from .belief_texts import PROMPTS, prompt_hash
from .rows import CLOSED, KINDS, LEVELS, has_gold, problems


def annotation_stats(rows: List[dict]) -> dict:
    """The gold annotation (``rows.py``) per source (and ``all``):
    queries per level and per state of the belief texts (``texts_ok``),
    beliefs per query and per kind, the beliefs with answers (all but
    the rest), with documents (the closed ones) and with their gold
    (``rows.has_gold``); the rows that break the row format
    (``rows.problems``) and the duplicate ids."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
        groups["all"].append(row)
    per_source = {}
    for source, group in sorted(groups.items()):
        beliefs = [c for r in group for c in r["beliefs"]]
        levels = Counter(r["level"] for r in group)
        per_source[source] = {
            "queries": len(group),
            "levels": {lv: levels[lv] for lv in LEVELS},
            "texts": {k: sum(r["texts_ok"] is v for r in group)
                      for k, v in (("ok", True), ("failed", False), ("retry", None))},
            "beliefs_per_query": distribution([len(r["beliefs"]) for r in group]),
            "beliefs": len(beliefs),
            "kinds": {k: sum(c["kind"] == k for c in beliefs) for k in KINDS},
            "with_answers": sum(bool(c["answers"]) for c in beliefs),
            "with_docs": sum(bool(c["doc_ids"]) for c in beliefs if c["kind"] == CLOSED),
            "with_gold": sum(has_gold(c) for c in beliefs),
        }
    checks = {"rows_breaking_format": sum(bool(problems(r)) for r in rows),
              "duplicate_ids": len(rows) - len({r["id"] for r in rows})}
    return {"per_source": per_source, "checks": checks}


def format_annotation_table(stats: dict) -> str:
    """The annotation stats as a plain-text table: one row per measure, one
    column per source (``all`` last), then the checks."""
    per_source = stats["per_source"]
    sources = [s for s in per_source if s != "all"] + ["all"] if per_source else []

    def share(n: int, total: int) -> str:
        return f"{n} ({n / total:.0%})" if total else "0"

    rows = [("queries", lambda s: s["queries"])]
    rows += [(f"  level {lv}", lambda s, lv=lv: s["levels"][lv]) for lv in ("full", "partial", "query")]
    rows += [(f"  texts {k}", lambda s, k=k: s["texts"][k]) for k in ("ok", "failed", "retry")]
    rows += [
        ("beliefs/query mean/med/max", lambda s: _dist(s["beliefs_per_query"], ("mean", "median", "max"))),
        ("beliefs", lambda s: s["beliefs"]),
    ]
    rows += [(f"  {k}", lambda s, k=k: s["kinds"][k]) for k in KINDS]
    rows += [
        ("  with answers", lambda s: share(s["with_answers"], s["beliefs"])),
        ("  with docs (closed)", lambda s: share(s["with_docs"], s["kinds"][CLOSED])),
        ("  with their gold", lambda s: share(s["with_gold"], s["beliefs"])),
    ]
    lines = _table(sources, per_source, rows)
    lines.append("")
    lines.append("annotation checks (all 0 = consistent): "
                 + ", ".join(f"{k} {v}" for k, v in stats["checks"].items()))
    return "\n".join(lines)


def _table(sources: List[str], per_source: Dict[str, dict], rows) -> List[str]:
    table = [[""] + sources] + [[label] + [_fmt(get(per_source[s])) for s in sources] for label, get in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
             for row in table]
    lines.insert(1, "-" * len(lines[0]))
    return lines


def _fmt(value: float) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _dist(d: dict, keys: Tuple[str, ...]) -> str:
    return " / ".join(_fmt(d[k]) for k in keys) if d["n"] else "-"


def write_annotation_stats(paths, gc, rows: List[dict]) -> dict:
    """The ``annotations`` section of ``stats.json``, the same in every
    dataset: ``annotation_stats`` of the rows of split ``all`` and the writer of
    their belief texts (``gold_beliefs.model`` and the hash of its
    prompts per query shape)."""
    stats = {"writer": {"model": gc.model, "prompt": {shape: prompt_hash(shape) for shape in PROMPTS}},
             **annotation_stats(rows)}
    write_stats(paths.stats, "annotations", stats)
    return stats
