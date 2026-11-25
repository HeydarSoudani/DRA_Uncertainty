"""The agent's verbal confidence against the correctness of its answer.

QA datasets.  Per judged query: y is the correctness of the final answer
(the accuracy verdicts, ``eval/generation/accuracy.jsonl``: the judge's
``correct``, TRQA's exact match; a judge error leaves the query out), the
score the agent's own confidence (``generation.confidence``, glm and oss).
A query without a stated confidence (no confidence line, a tool call or
narration as the final text) is scored :data:`~.metrics.MISSING_CONFIDENCE`,
so every judged query counts; ``per_query.csv`` keeps the raw value (blank
when missing).

A measure of the generation, scored once per run (the generate stage
reports it): the baseline of the belief scores.

Rows (``uncertainty`` group):
    verbal: missing (scored 0.5)   judged queries without a confidence
    verbal AUROC                   bootstrap CI over queries
    verbal Brier

No LLM call.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..shared.context import GENERATION, EvalContext, EvalSetup
from ..shared.measures import auroc, bootstrap_ci, brier
from ..shared.result import GroupResult
from . import metrics as m

logger = logging.getLogger(__name__)


def verbal_confidence_scores(accuracy_per_query: List[Dict[str, Any]]) -> GroupResult:
    """The rows from the accuracy records (``correct``, ``judge_error``,
    ``verbal_confidence``); empty when no query has a verbal confidence (an
    agent that states none)."""
    if not any(r.get("verbal_confidence") is not None for r in accuracy_per_query):
        return GroupResult()
    scored = [r for r in accuracy_per_query if not r.get("judge_error")]
    missing = sum(1 for r in scored if r.get("verbal_confidence") is None)
    if len(scored) < len(accuracy_per_query):
        logger.info("verbal confidence evaluation: %d judge error(s) left out",
                    len(accuracy_per_query) - len(scored))

    def score(r):
        c = r.get("verbal_confidence")
        return m.MISSING_CONFIDENCE if c is None else c

    def stat(fn):
        return lambda rs: fn(np.array([int(bool(r.get("correct"))) for r in rs], dtype=float),
                             np.array([score(r) for r in rs], dtype=float)) if rs else math.nan

    res = GroupResult()
    res.add(m.GROUP, m.VERBAL_MISSING, missing, n=len(scored))
    res.add(m.GROUP, m.auroc_row(m.VERBAL), stat(auroc)(scored), *bootstrap_ci(scored, stat(auroc)), len(scored))
    res.add(m.GROUP, m.brier_row(m.VERBAL), stat(brier)(scored), n=len(scored))
    return res


class VerbalConfidenceEvaluator:
    """:func:`verbal_confidence_scores` of the run's answer verdicts; QA
    datasets whose answers are scored."""

    name = "verbal"
    scope = GENERATION
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["VerbalConfidenceEvaluator"]:
        return cls() if setup.task == "qa" and setup.has_accuracy else None

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        if not ctx.accuracy:
            return GroupResult()
        return verbal_confidence_scores(ctx.accuracy["per_query"])
