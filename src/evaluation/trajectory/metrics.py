"""The ``trajectory`` group: its metric names and its log line."""

from typing import List, Optional

from ..shared.log import val
from ..shared.result import Row

GROUP = "trajectory"

SEARCHES = "searches"
AGENT_TOKENS = "agent tokens"
FORCED = "forced answers"
MAX_ITER = "max iterations reached"
#: Per belief model (``belief_tokens.py``); shown on the belief model's ``belief`` log line.
BELIEF_TOKENS = "belief tokens"


def log_line(rows: List[Row]) -> Optional[str]:
    """``trajectory   searches 9.2 | agent tok 41234 | forced 4``, always shown."""
    return (f"trajectory   searches {val(rows, GROUP, SEARCHES, 1)}"
            f" | agent tok {val(rows, GROUP, AGENT_TOKENS, 0)}"
            f" | forced {val(rows, GROUP, FORCED)}")
