"""The terminal block of an evaluation.

Each group writes its own lines (``log_line`` / ``model_log_line`` in its
``metrics.py``) with the helpers below; :func:`log_lines` puts them
together: the generation's lines, then per belief model a ``[model]``
header and its lines.
"""

from typing import Callable, List, Optional, Sequence

from .files import fmt
from .result import NO_MODEL, Row

#: The rule around the log block.
RULE = "=" * 80

#: A group's log line from rows of one scope (the generation, or one belief
#: model); None when the group has nothing to show.
LineFn = Callable[[List[Row]], Optional[str]]


def get(rows: List[Row], group: str, metric: str) -> Optional[Row]:
    """The row of *group* and *metric* (*rows* are of one scope)."""
    return next((r for r in rows if r.group == group and r.metric == metric), None)


def val(rows: List[Row], group: str, metric: str, digits: int = 3) -> str:
    """The row's value, "n/a" without the row."""
    r = get(rows, group, metric)
    return "n/a" if r is None else fmt(r.value, digits)


def n(rows: List[Row], group: str, metric: str) -> str:
    """The row's n, "n/a" without the row or its n."""
    r = get(rows, group, metric)
    return "n/a" if r is None or r.n is None else str(r.n)


def ci(rows: List[Row], group: str, metric: str) -> str:
    """The row's value with its CI (``0.610 [0.53,0.69]``), "n/a" without the row."""
    r = get(rows, group, metric)
    if r is None:
        return "n/a"
    interval = f" [{fmt(r.ci_low, 2)},{fmt(r.ci_high, 2)}]" if r.ci_low is not None else ""
    return fmt(r.value, 3) + interval


def log_lines(title: str, rows: List[Row], generation_lines: Sequence[LineFn],
              model_lines: Sequence[LineFn]) -> List[str]:
    """The terminal block: *title*, the *generation_lines* of the generation's
    rows, then per belief model in *rows* ``[model]`` and its *model_lines*."""
    shared = [r for r in rows if r.belief_model == NO_MODEL]
    out = [title] + [line for fn in generation_lines if (line := fn(shared)) is not None]
    models = list(dict.fromkeys(r.belief_model for r in rows if r.belief_model != NO_MODEL))
    for model in models:
        own = [r for r in rows if r.belief_model == model]
        out.append(f"[{model}]")
        out += [line for fn in model_lines if (line := fn(own)) is not None]
    return out
