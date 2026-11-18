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

The table below ``metrics.md`` gives every code that occurs (notes too):
its effect, its training role, how many times it occurs and in how many
update calls."""

from collections import Counter
from typing import Any, Dict, List, Tuple

from belief_model_component import ERROR_EFFECTS, GATE_ISSUES, ISSUES, UNCLEAN_REPLY, issue_effect

from ..metrics_file import GroupResult

#: Columns of the issue table below the main table.
ISSUE_COLUMNS = ("code", "effect", "role", "count", "calls", "share of calls")


def _role(code: str) -> str:
    info = ISSUES.get(code)
    if info is None:
        return ""
    return "+".join(r for r, on in (("gate", info.gate), ("unclean", info.unclean)) if on) or "none"


def issue_summary(files: Dict[str, Tuple[Any, List[Dict], Any]]) -> GroupResult:
    """The ``belief`` rows of the belief files *files* (``load_belief_files``)
    and the issue table (``ISSUE_COLUMNS``); empty without an update call."""
    calls = parsed = gate = unclean = 0
    with_effect: Counter = Counter()
    count: Counter = Counter()
    in_calls: Counter = Counter()
    for _meta, steps, final in files.values():
        records = list(steps) + ([final] if final is not None else [])
        for r in records:
            if r.get("final_score") is not None:
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
        return GroupResult()
    res = GroupResult()
    res.add("belief", "update calls", calls, n=calls)
    res.add("belief", "update calls: format gate failed", gate / calls, n=calls)
    res.add("belief", "update calls: unclean", unclean / calls, n=calls)
    for e in ERROR_EFFECTS:
        res.add("belief", f"update calls: with {e} issue", with_effect[e] / calls, n=calls)
    table = []
    for code, n in sorted(count.items(), key=lambda x: (-x[1], x[0])):
        table.append([code, issue_effect(code), _role(code), str(n), str(in_calls[code]),
                      f"{in_calls[code] / calls:.2f}"])
    if table:
        res.sections.append(("Belief update issues (every call)", list(ISSUE_COLUMNS), table))
    return res
