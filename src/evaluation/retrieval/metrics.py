"""The ``retrieval`` group: its metric names and its log line."""

from typing import List, Optional

from ..shared.log import val
from ..shared.result import Row

GROUP = "retrieval"

RECALL = "Recall@N"
PRECISION = "Precision@N"
N_SEEN = "N (docs seen)"


def log_line(rows: List[Row]) -> Optional[str]:
    """``retrieval    Recall@N 0.412 | Precision@N 0.081 (N 38.5)``, always shown."""
    return (f"retrieval    Recall@N {val(rows, GROUP, RECALL)}"
            f" | Precision@N {val(rows, GROUP, PRECISION)}"
            f" (N {val(rows, GROUP, N_SEEN, 1)})")
