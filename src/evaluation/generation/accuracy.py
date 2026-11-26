"""The ``generation`` rows of answer accuracy, shared by the two accuracy
evaluators (``llm_judge.py``, ``numeric_match.py``).

Both ``score`` methods return the same raw dict: ``accuracy``,
``num_correct``, ``num_evaluated``, ``num_judge_errors`` and ``per_query``
(one record per query with ``query_id``, ``correct``, ``judge_error`` and
``verbal_confidence``); ``{}`` when no query has a ground-truth answer.

Rows (``generation`` group):
    accuracy       correct / evaluated (n = evaluated)
    correct
    evaluated      queries with a ground-truth answer
    judge errors   queries the judge could not score (counted incorrect)

Per query: ``correct`` (blank on a judge error), ``verbal_confidence``.
"""

from pathlib import Path
from typing import Any, Callable, Dict

from ..shared.files import ACCURACY_FILE
from ..shared.result import GroupResult
from .metrics import ACCURACY, CORRECT, EVALUATED, GROUP, JUDGE_ERRORS


def accuracy_result(metrics: Dict[str, Any], run_dir, save: Callable[[Dict[str, Any], Path], None]) -> GroupResult:
    """The rows and per-query values of an accuracy evaluator's *metrics*
    (its ``score``; kept as *raw*), saved at once with *save* to
    ``eval/generation/accuracy.jsonl`` when there is a *run_dir*: the
    verdicts are paid for, and a later evaluator may fail.  Empty without
    any evaluated query."""
    if not metrics:
        return GroupResult(raw=metrics)
    if run_dir is not None:
        save(metrics, Path(run_dir) / ACCURACY_FILE)
    res = GroupResult(raw=metrics, per_query={
        r["query_id"]: {"correct": None if r.get("judge_error") else bool(r["correct"]),
                        "verbal_confidence": r.get("verbal_confidence")} for r in metrics["per_query"]})
    res.add(GROUP, ACCURACY, metrics["accuracy"], n=metrics["num_evaluated"])
    res.add(GROUP, CORRECT, metrics["num_correct"])
    res.add(GROUP, EVALUATED, metrics["num_evaluated"])
    res.add(GROUP, JUDGE_ERRORS, metrics.get("num_judge_errors", 0))
    return res
