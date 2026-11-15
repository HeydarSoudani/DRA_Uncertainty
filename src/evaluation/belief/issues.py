"""How often the belief update's replies have issues (``core.issues``), over
every belief file of a run (every query, gold criteria or not).

An update call is a step or final record with a call made (``called``),
not the final scorer's; it parsed when its ``observations`` are not null.
Rows (the ``belief`` group of ``metrics.md``), each a share of the update
calls:

    format gate failed   the reply did not parse, or has a ``gate`` code
                         (training's format gate, without the cut-off
                         check: the finish reason is not saved)
    unclean              parsed, with an ``unclean`` code (not an SFT
                         sample)
    with <effect> issue  parsed, with a ``dropped`` / ``corrected`` /
                         ``flagged`` issue

The table below ``metrics.md`` gives every code that occurs (notes and the
final scorer's ``answer`` codes too): its effect, its training role, how
many times it occurs and in how many calls (of its own kind: update calls,
or final scorer calls for an ``answer`` code)."""

from collections import Counter
from typing import Any, Dict, List, Tuple

from belief_model_component import ERROR_EFFECTS, GATE_ISSUES, ISSUES, UNCLEAN_REPLY, issue_effect

#: Columns of the issue table below the main table.
ISSUE_COLUMNS = ("code", "effect", "role", "count", "calls", "share of calls")


def _role(code: str) -> str:
    info = ISSUES.get(code)
    if info is None:
        return ""
    return "+".join(r for r, on in (("gate", info.gate), ("unclean", info.unclean)) if on) or "none"


def issue_summary(files: Dict[str, Tuple[Any, List[Dict], Any]]) -> Dict[str, Any]:
    """``{rows, table}`` of the belief files *files* (``load_belief_files``):
    ``rows`` as ``(metric, value, ci_low, ci_high, n)``, ``table`` the cell
    rows of ``ISSUE_COLUMNS``; ``{}`` without an update call."""
    calls = parsed = gate = unclean = 0
    with_effect: Counter = Counter()
    count: Counter = Counter()
    in_calls: Counter = Counter()
    scorer_calls = 0
    for _meta, steps, final in files.values():
        records = list(steps) + ([final] if final is not None else [])
        for r in records:
            score = r.get("final_score")
            if score is not None:
                scorer_calls += 1
                codes = [i.get("code") for i in score.get("issues") or [] if isinstance(i, dict)]
                count.update(codes)
                in_calls.update(set(codes))
                continue
            if not r.get("called"):
                continue
            calls += 1
            codes = [str(i.get("code")) for i in [*(r.get("issues") or []), *(r.get("notes") or [])]]
            count.update(codes)
            in_calls.update(set(codes))
            if r.get("observations") is None:
                gate += 1
                continue
            parsed += 1
            gate += any(c in GATE_ISSUES for c in codes)
            unclean += any(c in UNCLEAN_REPLY for c in codes)
            with_effect.update({issue_effect(c) for c in codes})
    if not calls:
        return {}
    rows = [("update calls", calls, None, None, calls),
            ("update calls: format gate failed", gate / calls, None, None, calls),
            ("update calls: unclean", unclean / calls, None, None, calls)]
    rows += [(f"update calls: with {e} issue", with_effect[e] / calls, None, None, calls) for e in ERROR_EFFECTS]
    table = []
    for code, n in sorted(count.items(), key=lambda x: (-x[1], x[0])):
        info = ISSUES.get(code)
        own = scorer_calls if info is not None and info.concerns == "answer" else calls
        table.append([code, issue_effect(code), _role(code), str(n), str(in_calls[code]),
                      f"{in_calls[code] / own:.2f}" if own else ""])
    return {"rows": rows, "table": table}
