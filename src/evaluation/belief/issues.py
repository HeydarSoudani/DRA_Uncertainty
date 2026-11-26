"""How often the belief update's replies have issues (``core.issues``), over
every belief file of a run (every query, gold beliefs or not).

An update call is a step record with a call made (``called``), the final
update's included (the final record is the scorer's, never an update); it
parsed when its ``observations`` are not null.

Rows (``belief`` group):
    update calls                       how many update calls
    update calls: format gate failed   the share whose reply did not parse,
                                       or has a ``gate`` code (training's
                                       format gate, without the cut-off
                                       check: the finish reason is not saved)

Section: the issue table (:data:`ISSUE_COLUMNS`), every code that occurs
(notes too) with its effect, training role, count and the update calls it
occurs in.  No LLM call.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from belief_model_component import GATE_ISSUES, ISSUES, issue_effect

from ..shared.context import BELIEF_MODEL, EvalContext, EvalSetup
from ..shared.result import GroupResult
from .metrics import GATE_FAILED, GROUP, ISSUES_SECTION, UPDATE_CALLS

#: Columns of the issue table.
ISSUE_COLUMNS = ("code", "effect", "role", "count", "calls", "share of calls")


def _role(code: str) -> str:
    info = ISSUES.get(code)
    if info is None:
        return ""
    return "+".join(r for r, on in (("gate", info.gate), ("unclean", info.unclean)) if on) or "none"


def issue_summary(files: Dict[str, Tuple[Any, List[Dict], Any]]) -> GroupResult:
    """The rows and the issue table of the belief *files* (``load_belief_files``);
    empty without an update call."""
    calls = gate = 0
    count: Counter = Counter()
    in_calls: Counter = Counter()
    for _meta, steps, _final in files.values():
        for r in steps:
            if not r.get("called"):
                continue
            calls += 1
            codes = [str(i.get("code")) for i in [*(r.get("issues") or []), *(r.get("notes") or [])]]
            count.update(codes)
            in_calls.update(set(codes))
            gate += r.get("observations") is None or any(c in GATE_ISSUES for c in codes)
    if not calls:
        return GroupResult()
    res = GroupResult()
    res.add(GROUP, UPDATE_CALLS, calls, n=calls)
    res.add(GROUP, GATE_FAILED, gate / calls, n=calls)
    table = []
    for code, k in sorted(count.items(), key=lambda x: (-x[1], x[0])):
        table.append([code, issue_effect(code), _role(code), str(k), str(in_calls[code]),
                      f"{in_calls[code] / calls:.2f}"])
    if table:
        res.sections.append((ISSUES_SECTION, list(ISSUE_COLUMNS), table))
    return res


class IssueEvaluator:
    """:func:`issue_summary` of each belief model; every dataset."""

    name = "belief issues"
    scope = BELIEF_MODEL
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["IssueEvaluator"]:
        return cls()

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        return issue_summary(ctx.belief_files)
