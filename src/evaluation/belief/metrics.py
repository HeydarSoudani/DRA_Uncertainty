"""The ``belief`` group: its metric names and its log line (per belief model)."""

from typing import List, Optional

from ..shared.log import ci, get, n, val
from ..shared.result import Row
from ..trajectory import metrics as trajectory

GROUP = "belief"

EVIDENCE_UPTAKE = "evidence uptake"
FALSE_UPTAKE = "false uptake"
UPDATE_CALLS = "update calls"
GATE_FAILED = "update calls: format gate failed"
#: Title of the issue table (one per belief model, suffixed `` ({model})``).
ISSUES_SECTION = "Belief update issues (every call)"


def model_log_line(rows: List[Row]) -> Optional[str]:
    """``belief       evidence uptake 0.570 [0.47,0.67] (n 180) | false uptake ...
    | update calls 1840 | format gate failed 0.020 | belief tok 18230``; the
    belief tokens are the ``trajectory`` group's.  None without any of them."""
    parts = []
    if get(rows, GROUP, EVIDENCE_UPTAKE):
        parts.append(f"evidence uptake {ci(rows, GROUP, EVIDENCE_UPTAKE)}"
                     f" (n {n(rows, GROUP, EVIDENCE_UPTAKE)})")
    if get(rows, GROUP, FALSE_UPTAKE):
        parts.append(f"false uptake {ci(rows, GROUP, FALSE_UPTAKE)}"
                     f" (n {n(rows, GROUP, FALSE_UPTAKE)})")
    if get(rows, GROUP, UPDATE_CALLS):
        parts.append(f"update calls {val(rows, GROUP, UPDATE_CALLS)}"
                     f" | format gate failed {val(rows, GROUP, GATE_FAILED)}")
    if get(rows, trajectory.GROUP, trajectory.BELIEF_TOKENS):
        parts.append(f"belief tok {val(rows, trajectory.GROUP, trajectory.BELIEF_TOKENS, 0)}")
    return "belief       " + " | ".join(parts) if parts else None
