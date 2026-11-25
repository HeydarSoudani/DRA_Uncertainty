"""The ``generation`` group: its metric names and its log line."""

from typing import List, Optional

from ..shared.log import get, val
from ..shared.result import Row

GROUP = "generation"

# Answer accuracy (``accuracy.py``: the LLM judge and numeric match).
ACCURACY = "accuracy"
CORRECT = "correct"
EVALUATED = "evaluated"
JUDGE_ERRORS = "judge errors"

# Reports (``argue.py``); EVALUATED is shared.
NUGGET_COVERAGE = "nugget coverage"
NUGGET_F1 = "nugget F1"
NUGGET_COVERAGE_WEIGHTED = "nugget coverage (weighted)"
NUGGET_F1_WEIGHTED = "nugget F1 (weighted)"
WITHOUT_NUGGETS = "without nuggets"
EMPTY_REPORTS = "empty reports"
UNCITED_REPORTS = "reports without citations"
UNMAPPED_REPORTS = "reports with unmapped citations"
JUDGE_FAILURES = "judge failures"
JUDGE_MALFORMED = "judge malformed replies"


def log_line(rows: List[Row]) -> Optional[str]:
    """``generation   accuracy 0.310 (62/200)`` (QA) or ``generation   nugget
    coverage 0.512 | F1 0.401 (50 reports)`` (reports); None without either."""
    if get(rows, GROUP, ACCURACY):
        return (f"generation   accuracy {val(rows, GROUP, ACCURACY)}"
                f" ({val(rows, GROUP, CORRECT)}/{val(rows, GROUP, EVALUATED)})")
    if get(rows, GROUP, NUGGET_COVERAGE):
        return (f"generation   nugget coverage {val(rows, GROUP, NUGGET_COVERAGE)}"
                f" | F1 {val(rows, GROUP, NUGGET_F1)}"
                f" ({val(rows, GROUP, EVALUATED)} reports)")
    return None
