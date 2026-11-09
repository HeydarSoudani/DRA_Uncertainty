"""Readable views of the belief: the criteria, one iteration's belief and
the final belief.

One renderer feeds the console (``utils.text_utils.verbose_print_*``) and
the trajectory markdown (the md-only blocks the agent mixin writes), so the
two never drift apart.  It reads the saved shapes only: the step and final
records of ``BeliefModel`` and ``BeliefModel.criteria_summary``.

Per criterion, one row with its status, its selected candidate(s) and the
sources its status rests on ("d<n>" a document, "t<step>" an agent turn;
the support part of its statement, ``statement_text.support_labels``).
``*`` marks a criterion whose statement the update rewrote this iteration.
The md also gives the full text of each statement rewritten in the
iteration and the reply's issues.  The documents labelled in the
iteration are listed with their ids, so a label can be traced to its
document.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Candidates are cut to this length in the console; the md shows them in full.
CONSOLE_VALUE_CHARS = 40

Row = Tuple[str, str, str, str, str]  # criterion id, mark, status, selected, sources


def _num(val: Any, fmt: str = ".3f") -> str:
    return f"{val:{fmt}}" if val is not None else "—"


def _value(text: Any, limit: Optional[int]) -> str:
    text = " ".join(str(text).split())
    return text if limit is None or len(text) <= limit else text[:limit - 1] + "…"


def belief_rows(record: Dict[str, Any], criteria: Sequence[Dict[str, Any]],
                value_chars: Optional[int] = None) -> List[Row]:
    """Per criterion its row: status, selected candidate(s), the sources the
    status rests on.  Empty when the record has no belief (``statements``
    null)."""
    statements = record.get("statements")
    if statements is None:
        return []
    changed = set(record.get("changed") or [])
    rows: List[Row] = []
    for i, c in enumerate(criteria):
        cid = c.get("id", f"c{i + 1}")
        s = statements[i] if i < len(statements) else {}
        selected = ", ".join(f'"{_value(v, value_chars)}"' for v in s.get("selected") or []) or "—"
        rows.append((cid, "*" if cid in changed else "", str(s.get("status") or "unknown"), selected,
                     " ".join(s.get("support_cited") or []) or "—"))
    return rows


def _rewritten(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(criterion id, text)`` of the statements the iteration rewrote."""
    changed = set(record.get("changed") or [])
    return [(s["id"], " ".join(str(s.get("text") or "").split())) for s in record.get("statements") or []
            if s.get("id") in changed and s.get("text")]


def _issues(record: Dict[str, Any]) -> List[str]:
    return [f"{i.get('id')}: {i.get('code')} ({i.get('detail')})" for i in record.get("issues") or []]


def gold_summary(record: Dict[str, Any]) -> str:
    """New-item precision, recall so far (+new this step -> seen / all
    relevant) and the intermediate answer."""
    gold = (
        f"new_prec={_num(record.get('new_item_precision'))} "
        f"({_num(record.get('num_new_relevant'), 'd')}/{record.get('num_docs', 0)}) | "
        f"recall={_num(record.get('recall_so_far'))} "
        f"({_num(record.get('num_new_relevant'), '+d')} → "
        f"{_num(record.get('num_relevant_seen'), 'd')}/{_num(record.get('num_relevant'), 'd')})"
    )
    answers = record.get("intermediate_answers")
    if answers is None:
        return gold + f" | answer=null ({record.get('intermediate_answer_status') or 'disabled'})"
    answers = [" ".join(str(a).split()) for a in answers]
    return gold + f" | answer={answers if answers else 'none'}"


def _summary(record: Dict[str, Any], final: bool) -> str:
    out = f"certainty={_num(record.get('certainty'))}"
    if not final:
        out += f" | {record.get('num_new_docs', 0)}/{record.get('num_docs', 0)} new docs"
    return out


def _new_labels(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The documents of the step with a label, first shown in it."""
    return [d for d in record.get("docs") or [] if d.get("new") and d.get("label")]


def _labels(record: Dict[str, Any], md: bool) -> str:
    fmt = "{} `{}`" if md else "{} {}"
    return " · ".join(fmt.format(d.get("label"), d.get("doc_id")) for d in _new_labels(record))


# ---------------------------------------------------------------------------
# Console (plain text): (family, row) pairs, printed under one prefix
# ---------------------------------------------------------------------------

def criteria_header(summary: Dict[str, Any]) -> str:
    """``n criteria · gold · level full · multi_hop · 4 relevant docs``."""
    criteria, info = summary.get("criteria") or [], summary.get("criteria_info") or {}
    parts = [f"{len(criteria)} criteria"]
    kind = info.get("criteria_kind") or info.get("model")
    if kind:
        parts.append(str(kind))
    if info.get("gold_level"):
        parts.append(f"level {info['gold_level']}")
    if info.get("query_shape"):
        parts.append(str(info["query_shape"]))
    if info.get("bank"):
        parts.append(f"bank {info['bank']}")
    if summary.get("num_relevant") is not None:
        parts.append(f"{summary['num_relevant']} relevant docs")
    return " · ".join(parts)


def criteria_lines(summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows = [("criteria", criteria_header(summary))]
    rows += [(c.get("id", ""), f"[{c.get('kind', '')}] {c.get('text', '')}") for c in summary.get("criteria") or []]
    errors = (summary.get("criteria_info") or {}).get("errors")
    if errors:
        rows.append(("errors", "; ".join(map(str, errors))))
    return rows


def belief_lines(record: Dict[str, Any], criteria: Sequence[Dict[str, Any]],
                 final: bool = False) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [("belief", _summary(record, final))]
    if not final:
        rows.append(("gold", gold_summary(record)))
    table = belief_rows(record, criteria, CONSOLE_VALUE_CHARS)
    status_w = max((len(r[2]) for r in table), default=0)
    value_w = max((len(r[3]) for r in table), default=0)
    for cid, mark, status, value, sources in table:
        rows.append((f"{cid} {mark}".rstrip(), f"{status:<{status_w}}  {value:<{value_w}}  ← {sources}"))
    if not final and _new_labels(record):
        rows.append(("labels", _labels(record, md=False)))
    if record.get("issues"):
        rows.append(("issues", "; ".join(_issues(record))))
    if record.get("errors"):
        rows.append(("errors", "; ".join(map(str, record["errors"]))))
    return rows


# ---------------------------------------------------------------------------
# Trajectory md
# ---------------------------------------------------------------------------

def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """A markdown table whose cells are padded to their column's width, so
    the columns line up in the md source as well as when rendered."""
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    return [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, rows)]


def criteria_md(summary: Dict[str, Any]) -> str:
    lines = ["## Criteria", "", criteria_header(summary), ""]
    lines += [f"- **{c.get('id', '')}** [{c.get('kind', '')}] {c.get('text', '')}"
              for c in summary.get("criteria") or []]
    errors = (summary.get("criteria_info") or {}).get("errors")
    if errors:
        lines += ["", f"*errors: {'; '.join(map(str, errors))}*"]
    return "\n".join(lines) + "\n\n---"


def belief_md(record: Dict[str, Any], criteria: Sequence[Dict[str, Any]], final: bool = False) -> str:
    heading = "## Final belief" if final else f"### Belief · iter {record.get('agent_iteration', record.get('iteration'))}"
    lines = [heading, "", _summary(record, final).replace(" | ", " · ")]
    if not final:
        lines += ["", f"gold: {gold_summary(record).replace(' | ', ' · ')}"]
    table = belief_rows(record, criteria)
    if table:
        lines += [""] + _md_table(("criterion", "status", "selected", "sources"),
                                  [(f"{cid} {mark}".strip(), status, _cell(value), sources)
                                   for cid, mark, status, value, sources in table])
    rewritten = _rewritten(record)
    if rewritten:
        lines += [""] + [f"- **{cid}**: {_cell(text)}" for cid, text in rewritten]
    if not final and _new_labels(record):
        lines += ["", f"labels: {_labels(record, md=True)}"]
    if record.get("issues"):
        lines += ["", f"*issues: {_cell('; '.join(_issues(record)))}*"]
    if record.get("errors"):
        lines += ["", f"*errors: {'; '.join(map(str, record['errors']))}*"]
    return "\n".join(lines)
