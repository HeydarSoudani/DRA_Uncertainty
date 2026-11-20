"""The agent's verbal confidence against the correctness of its answer.

Per query of a short-answer dataset (``task == "qa"``), y is the
correctness of the agent's final answer (``eval/generation/accuracy.jsonl``:
the judge's ``correct``, TRQA's exact match; a judge error leaves the
query out) and the score is the agent's own confidence in its answer
(``generation.confidence``, glm and oss).  A measure of the generation (no
belief model), so the generate stage reports it.  A judged query without a
stated confidence (no confidence line, a tool call or narration as the final
text) is scored :data:`MISSING_CONFIDENCE` (0.5: unknown, neither confident
nor unconfident), so every judged query counts; per_query.csv keeps the raw
value (blank when missing).  Rows: how many of the queries scored were
missing (scored 0.5), the AUROC with a bootstrap CI over queries and the
Brier score.  No LLM call.
"""

import logging
import math
from typing import Any, Dict, List

import numpy as np

from ..measures import auroc, bootstrap_ci, brier
from ..metrics_file import GroupResult

logger = logging.getLogger(__name__)

# The score of a judged query without a stated confidence (module docstring).
MISSING_CONFIDENCE = 0.5


class VerbalConfidenceEvaluator:
    """The verbal confidence measures of a run (module docstring)."""

    def evaluate(self, accuracy_per_query: List[Dict[str, Any]]) -> GroupResult:
        """The generation's ``uncertainty`` rows from the accuracy records
        (``correct``, ``judge_error``, ``verbal_confidence``); empty when no
        query has a verbal confidence (an agent that states none)."""
        if not any(r.get("verbal_confidence") is not None for r in accuracy_per_query):
            return GroupResult()
        scored = [r for r in accuracy_per_query if not r.get("judge_error")]
        missing = sum(1 for r in scored if r.get("verbal_confidence") is None)
        if len(scored) < len(accuracy_per_query):
            logger.info("verbal confidence evaluation: %d judge error(s) left out",
                        len(accuracy_per_query) - len(scored))

        def score(r):
            c = r.get("verbal_confidence")
            return MISSING_CONFIDENCE if c is None else c

        def stat(fn):
            return lambda rs: fn(np.array([int(bool(r.get("correct"))) for r in rs], dtype=float),
                                 np.array([score(r) for r in rs], dtype=float)) if rs else math.nan

        res = GroupResult()
        res.add("uncertainty", f"verbal: missing (scored {MISSING_CONFIDENCE:g})", missing, n=len(scored))
        res.add("uncertainty", "verbal AUROC", stat(auroc)(scored), *bootstrap_ci(scored, stat(auroc)), len(scored))
        res.add("uncertainty", "verbal Brier", stat(brier)(scored), n=len(scored))
        return res
