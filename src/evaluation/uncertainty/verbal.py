"""The agent's verbal confidence against the correctness of its answer.

Per query of a short-answer dataset (``task == "qa"``), y is the
correctness of the agent's final answer (``eval/generation/accuracy.jsonl``:
the judge's ``correct``, TRQA's exact match; a judge error leaves the
query out) and the score is the agent's own confidence in its answer
(``generation.confidence``, glm and oss).  A measure of the generation (no
belief model), so the generate stage reports it.  Rows: the queries scored,
the AUROC with a bootstrap CI over queries and the Brier score, over the
judged queries with a verbal confidence.  No LLM call.
"""

import logging
import math
from typing import Any, Dict, List

import numpy as np

from ..measures import auroc, bootstrap_ci, brier
from ..metrics_file import GroupResult

logger = logging.getLogger(__name__)


class VerbalConfidenceEvaluator:
    """The verbal confidence measures of a run (module docstring)."""

    def evaluate(self, accuracy_per_query: List[Dict[str, Any]]) -> GroupResult:
        """The generation's ``uncertainty`` rows from the accuracy records
        (``correct``, ``judge_error``, ``verbal_confidence``); empty when no
        query has a verbal confidence (an agent that states none)."""
        if not any(r.get("verbal_confidence") is not None for r in accuracy_per_query):
            return GroupResult()
        judged = [r for r in accuracy_per_query if not r.get("judge_error")]
        scored = [r for r in judged if r.get("verbal_confidence") is not None]
        left_out = {"judge error": len(accuracy_per_query) - len(judged),
                    "no verbal confidence": len(judged) - len(scored)}
        if any(left_out.values()):
            logger.info("verbal confidence evaluation: queries left out %s", left_out)

        def stat(fn):
            return lambda rs: fn(np.array([int(bool(r.get("correct"))) for r in rs], dtype=float),
                                 np.array([r["verbal_confidence"] for r in rs], dtype=float)) if rs else math.nan

        res = GroupResult()
        res.add("uncertainty", "verbal: queries scored", len(scored), n=len(accuracy_per_query))
        res.add("uncertainty", "verbal AUROC", stat(auroc)(scored), *bootstrap_ci(scored, stat(auroc)), len(scored))
        res.add("uncertainty", "verbal Brier", stat(brier)(scored), n=len(scored))
        return res
