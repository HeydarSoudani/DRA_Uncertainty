"""What an evaluator returns: rows of ``metrics.md``, per-query values and
extra sections (:class:`GroupResult`)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: The ``belief_model`` cell of a measure of the generation (no belief model).
NO_MODEL = "-"


@dataclass
class Row:
    """One row of the ``metrics.md`` table."""
    group: str
    metric: str
    value: Any
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None
    belief_model: str = NO_MODEL


#: An extra section of ``metrics.md``: ``(title, columns, cell rows)``.
Section = Tuple[str, List[str], List[List[str]]]


@dataclass
class GroupResult:
    """What one evaluator adds to a run's evaluation.

    Attributes:
        rows:      Its rows of ``metrics.md``.
        per_query: ``{query_id: {column: value}}``, its columns of ``per_query.csv``.
        sections:  Its extra titled tables of ``metrics.md``.
        raw:       Its own output that a later evaluator reads (the answer
                   verdicts for the uncertainty group).
    """
    rows: List[Row] = field(default_factory=list)
    per_query: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)
    raw: Any = None

    def add(self, group: str, metric: str, value: Any, ci_low: Optional[float] = None,
            ci_high: Optional[float] = None, n: Optional[int] = None) -> None:
        self.rows.append(Row(group, metric, value, ci_low, ci_high, n))
