"""Readable views of the belief: the criteria, one iteration's belief and
the final belief.

One renderer feeds the console (``utils.text_utils.verbose_print_*``) and
the trajectory markdown (the md-only blocks the agent mixin writes), so the
two never drift apart.  It reads the saved shapes only: the step record of
``BeliefModel.observe``, ``BeliefModel.criteria_summary`` and the meta line
(``final_record``).

Per criterion, its nuggets are shown most likely first: the status as a
word, the value, and the evidence labels ("d<n>" a document, "t<step>" an
agent turn; "!" marks a source that contradicts).  ``*`` marks a criterion
the update rewrote this iteration; a criterion without a nugget is
``unknown``.  The documents labelled in the iteration are listed with
their ids, so a label in the evidence can be traced to its document.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .criteria import ISSUE_NOT_IN_REPLY
from .types import CONTRADICTS, STATUS_LEVEL

# Values are cut to this length in the console; the md shows them in full.
CONSOLE_VALUE_CHARS = 40

Row = Tuple[str, str, str, str, str]  # criterion id, mark, status, value, evidence


def _num(val: Any, fmt: str = ".3f") -> str:
    return f"{val:{fmt}}" if val is not None else "—"


def _value(text: Any, limit: Optional[int]) -> str:
    text = " ".join(str(text).split())
    return text if limit is None or len(text) <= limit else text[:limit - 1] + "…"


def _evidence(nugget: Dict[str, Any]) -> str:
    labels = dict.fromkeys(("!" if e.get("stance") == CONTRADICTS else "") + str(e.get("label", "?"))
                           for e in nugget.get("evidence") or [])
    return " ".join(labels) or "—"


def _changed(updates: Optional[List[Dict[str, Any]]]) -> set:
    """Ids of the criteria whose nuggets the update rewrote."""
    return {u["id"] for u in updates or []
            if u.get("after") is not None and ISSUE_NOT_IN_REPLY not in (u.get("issues") or [])}


def belief_rows(record: Dict[str, Any], criteria: Sequence[Dict[str, Any]],
                value_chars: Optional[int] = None) -> List[Row]:
    """One row per nugget, most likely first within each criterion; a
    criterion without a nugget gets one ``unknown`` row.  Empty when the
    record has no belief (``nuggets_after`` null)."""
    nuggets = record.get("nuggets_after")
    if nuggets is None:
        return []
    changed = _changed(record.get("nugget_updates"))
    rows: List[Row] = []
    for i, c in enumerate(criteria):
        cid = c.get("id", f"c{i + 1}")
        mark = "*" if cid in changed else ""
        ranked = sorted(nuggets[i] if i < len(nuggets) else [],
                        key=lambda n: -STATUS_LEVEL.get(n.get("status"), -9))
        if not ranked:
            rows.append((cid, mark, "unknown", "—", "—"))
            continue
        for j, n in enumerate(ranked):
            rows.append((cid if j == 0 else "", mark if j == 0 else "", str(n.get("status")),
                         _value(n.get("value", ""), value_chars), _evidence(n)))
    return rows


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


def _labels(record: Dict[str, Any], md: bool) -> str:
    fmt = "{} `{}`" if md else "{} {}"
    return " · ".join(fmt.format(p.get("label"), p.get("doc_id")) for p in record.get("passages") or [])


def final_record(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The final belief of the meta line in the shape of a step record:
    ``certainty``, ``nuggets_after``, the answer turn's ``nugget_updates``
    and ``errors``."""
    update = meta.get("final_update") or {}
    return {"certainty": meta.get("final_certainty"), "nuggets_after": meta.get("final_nuggets"),
            "nugget_updates": update.get("nugget_updates"), "errors": update.get("errors") or []}


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
    value_w = max((len(r[3]) + 2 for r in table), default=0)
    for cid, mark, status, value, evidence in table:
        family = f"{cid} {mark}".rstrip() if cid else ""
        quoted = f'"{value}"' if value != "—" else value
        rows.append((family, f"{status:<{status_w}}  {quoted:<{value_w}}  ← {evidence}"))
    if not final and record.get("passages"):
        rows.append(("labels", _labels(record, md=False)))
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
        lines += [""] + _md_table(("criterion", "status", "value", "evidence"),
                                  [(f"{cid} {mark}".strip(), status, _cell(value), evidence)
                                   for cid, mark, status, value, evidence in table])
    if not final and record.get("passages"):
        lines += ["", f"labels: {_labels(record, md=True)}"]
    if record.get("errors"):
        lines += ["", f"*errors: {'; '.join(map(str, record['errors']))}*"]
    return "\n".join(lines)
