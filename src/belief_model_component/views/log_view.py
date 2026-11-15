"""Readable views of the belief: the criteria, one iteration's belief and
the final belief.

One renderer feeds the console (``utils.text_utils.verbose_print_*``) and
the trajectory markdown (the md-only blocks the agent mixin writes), so the
two never drift apart.  It reads the saved shapes only: the step and final
records of ``BeliefModel`` and ``BeliefModel.criteria_summary``.

Per criterion, one row with its status, its selected candidate(s) and
the number of its candidates and proposals (a removed candidate leaves the
list).  The row shows at most ``SELECTED_SHOWN`` selected candidates, then
``+N more``; the md lists every selected candidate of such a criterion
(with its status) under the table when the criterion changed in the
iteration, and always in the final belief.  ``*`` marks a criterion whose candidates or proposals changed this
iteration.  A step without a belief update call says so.  The md also
gives every candidate update of the iteration (previous status -> status,
the passages it cites as support (+) or against (-), and its reason).
Both give the reply's issues, one line per effect (``core.issues``:
dropped, corrected, flagged), then its notes; a record of a run before
schema 51 shows its codes under their current names.  The console lists
every document of the iteration with its label ("(again)" when an earlier
iteration showed it), so a label can be traced to its document; in the md
the step's Docs list gives each document its label
(``belief_doc_labels``, ``utils.trajectory_logger``).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.issues import EFFECTS, NOTE, current_code, issue_effect

# Selected candidates shown in a criterion's row; the rest as "+N more".
SELECTED_SHOWN = 2
# Candidates are cut to this length in a row (console, md table); the md
# list under the table shows them in full.
CONSOLE_VALUE_CHARS = 40
MD_VALUE_CHARS = 60

Row = Tuple[str, str, str, str, str]  # criterion id, mark, status, selected, number of candidates


def _num(val: Any, fmt: str = ".3f") -> str:
    return f"{val:{fmt}}" if val is not None else "—"


def _value(text: Any, limit: Optional[int]) -> str:
    text = " ".join(str(text).split())
    return text if limit is None or len(text) <= limit else text[:limit - 1] + "…"


def belief_rows(record: Dict[str, Any], criteria: Sequence[Dict[str, Any]],
                value_chars: Optional[int] = None) -> List[Row]:
    """Per criterion its row: status, selected candidate(s), number of
    candidates.  Empty when the record has no belief (``beliefs`` null)."""
    beliefs = record.get("beliefs")
    if beliefs is None:
        return []
    changed = set(record.get("changed") or [])
    rows: List[Row] = []
    for i, c in enumerate(criteria):
        cid = c.get("id", f"c{i + 1}")
        b = beliefs[i] if i < len(beliefs) else {}
        chosen = b.get("selected") or []
        selected = ", ".join(f'"{_value(v, value_chars)}"' for v in chosen[:SELECTED_SHOWN]) or "—"
        if len(chosen) > SELECTED_SHOWN:
            selected += f" +{len(chosen) - SELECTED_SHOWN} more"
        n = len(b.get("candidates") or [])
        m = len(b.get("proposals") or [])
        count = f"{n} candidate" + ("" if n == 1 else "s") + (f", {m} proposed" if m else "")
        rows.append((cid, "*" if cid in changed else "", str(b.get("status") or "unknown"), selected, count))
    return rows


def selected_lists(record: Dict[str, Any], criteria: Sequence[Dict[str, Any]],
                   final: bool = False) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """``(criterion id, [(candidate, status), ...])`` of each criterion
    whose row cut its selected candidates (more than ``SELECTED_SHOWN``),
    those changed this iteration only unless *final*."""
    beliefs = record.get("beliefs")
    if beliefs is None:
        return []
    changed = set(record.get("changed") or [])
    out = []
    for i, c in enumerate(criteria):
        cid = c.get("id", f"c{i + 1}")
        b = beliefs[i] if i < len(beliefs) else {}
        chosen = b.get("selected") or []
        if len(chosen) <= SELECTED_SHOWN or not (final or cid in changed):
            continue
        status = {x.get("name"): x.get("status") for x in b.get("candidates") or []}
        out.append((cid, [(v, str(status.get(v) or "")) for v in chosen]))
    return out


def _updates(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(criterion id, text)`` of each candidate update of the reply:
    ``"<candidate>": <previous> -> <status> (<reason>)``, "new" for an
    added candidate, "removed" for a removed one, "not taken" when code did
    not take it, and the status written when code corrected it (a removal
    code did not take: ``<status> -> <status> (wrote removed)``)."""
    out = []
    for u in record.get("updates") or []:
        kept, written = u.get("status"), u.get("written")
        if kept is None:
            move = f"not taken (wrote {written})" if written else "not taken"
        else:
            move = f"{u.get('previous') or 'new'} -> {kept}"
            if written and written != kept:
                move += f" (wrote {written})"
        cited = " ".join([f"+{lb}" for lb in u.get("support") or []] + [f"-{lb}" for lb in u.get("against") or []])
        if cited:
            move += f" [{cited}]"
        reason = " ".join(str(u.get("reason") or "").split())
        # An aspect's own status (an item without a candidate): "status: ...".
        head = f'"{u.get("candidate")}"' if u.get("candidate") else "status"
        out.append((str(u.get("id")), f"{head}: {move}" + (f" ({reason})" if reason else "")))
    return out


def _issue_text(issue: Any) -> str:
    """``<id> <code> (<detail>)`` of an issue record (a plain string of an
    old final scorer as it is)."""
    if not isinstance(issue, dict):
        return str(issue)
    detail = f" ({issue['detail']})" if issue.get("detail") else ""
    return f"{issue.get('id')} {current_code(str(issue.get('code')))}{detail}"


def issue_lines(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(effect, text)`` per effect with an issue or note of *record*, in
    the order of ``EFFECTS`` (notes last); repeated identical issues as one
    with ``×n``."""
    groups: Dict[str, Dict[str, int]] = {e: {} for e in EFFECTS}
    for i in [*(record.get("issues") or []), *(record.get("notes") or [])]:
        effect = issue_effect(str(i.get("code"))) if isinstance(i, dict) else EFFECTS[2]
        text = _issue_text(i)
        groups[effect][text] = groups[effect].get(text, 0) + 1
    return [("notes" if e == NOTE else e,
             "; ".join(t + (f" ×{n}" if n > 1 else "") for t, n in groups[e].items()))
            for e in EFFECTS if groups[e]]


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
    if final and record.get("final_score") is not None:
        out += (f" ({record.get('certainty_label')}) | formula={_num(record.get('certainty_formula'))}"
                f" | answer formula={_num(record.get('certainty_answer_formula'))}")
    elif final and record.get("certainty_formula") != record.get("certainty"):
        out += f" | formula={_num(record.get('certainty_formula'))}"
    if not final:
        out += f" | {record.get('num_new_docs', 0)}/{record.get('num_docs', 0)} new docs"
    if record.get("called") is False:
        out += " | no call (belief carried forward)"
    return out


def _final_score(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The final scorer's answer's candidates, reason and issues of a final
    record."""
    score = record.get("final_score")
    if not score:
        return []
    named = "; ".join(f"{c.get('id')} " + (f"\"{c['candidate']}\"" if c.get("candidate") else "none")
                      for c in score.get("candidates") or [])
    rows = [("answer's candidates", named or "none"), ("reason", score.get("reason") or "")]
    if score.get("issues"):
        rows.append(("score issues", "; ".join(_issue_text(i) for i in score["issues"])))
    return rows


def _step_labels(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The documents of the step with a label: every one the call showed,
    new or shown before."""
    return [d for d in record.get("docs") or [] if d.get("label")]


def belief_doc_labels(record: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{doc_id: {label, new}}`` of a step record's labelled documents (the
    trajectory md's Docs list shows them); empty without a record."""
    return {d["doc_id"]: {"label": d["label"], "new": bool(d.get("new"))}
            for d in _step_labels(record or {}) if d.get("doc_id")}


def _labels(record: Dict[str, Any]) -> str:
    """``d1 3942972 (again) · d6 5496567 · ...``: every labelled document of
    the step, "(again)" when an earlier step showed it."""
    return " · ".join(f"{d.get('label')} {d.get('doc_id')}" + ("" if d.get("new") else " (again)")
                      for d in _step_labels(record))


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
    for cid, mark, status, value, count in table:
        rows.append((f"{cid} {mark}".rstrip(), f"{status:<{status_w}}  {value:<{value_w}}  ({count})"))
    rows += _final_score(record)
    if not final and _step_labels(record):
        rows.append(("labels", _labels(record)))
    rows += issue_lines(record)
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
    table = belief_rows(record, criteria, MD_VALUE_CHARS)
    if table:
        lines += [""] + _md_table(("criterion", "status", "selected", "candidates"),
                                  [(f"{cid} {mark}".strip(), status, _cell(value), count)
                                   for cid, mark, status, value, count in table])
    for cid, chosen in selected_lists(record, criteria, final):
        lines += ["", f"- **{cid}** selected ({len(chosen)}):"]
        lines += [f'  - "{_cell(_value(v, None))}"' + (f": {st}" if st else "") for v, st in chosen]
    updates = _updates(record)
    if updates:
        lines += [""] + [f"- **{cid}** {_cell(text)}" for cid, text in updates]
    scored = _final_score(record)
    if scored:
        lines += [""] + [f"- {key}: {_cell(text)}" for key, text in scored]
    issues = issue_lines(record)
    if issues:
        lines += [""] + [f"- *{effect}:* {_cell(text)}" for effect, text in issues]
    if record.get("errors"):
        lines += ["", f"*errors: {'; '.join(map(str, record['errors']))}*"]
    return "\n".join(lines)
