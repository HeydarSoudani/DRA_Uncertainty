"""A belief model's certainty against the correctness of the agent's answer:
does it rank correct answers above wrong ones, and is it calibrated?

QA datasets, once per belief model.  Per query: y is the correctness of the
final answer (the accuracy verdicts, ``eval/generation/accuracy.jsonl``; a
judge error leaves the query out); the scores:

    belief rule-based   the fixed function of the final beliefs' statuses
                        (``certainty_formula``: the mean status probability)
    belief LLM score    the belief model's final scorer, s_T (``certainty``
                        of the ``final`` line), the headline; only when the
                        scorer ran (``final_score`` set)

Both are measured on the queries with y and the LLM score; the rule-based
score on those of them where it has a value, with its own n.  The queries
left out are logged.

Rows (``uncertainty`` group), per score: ``{score} AUROC`` (bootstrap CI over
queries) and ``{score} Brier``.  Per query: ``belief_llm_score``,
``belief_llm_score_label``, ``belief_rule_based``.  No LLM call.  The
baseline, the agent's verbal confidence, is ``verbal.py``.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..shared.context import BELIEF_MODEL, EvalContext, EvalSetup
from ..shared.measures import auroc, bootstrap_ci, brier
from ..shared.result import GroupResult
from . import metrics as m

logger = logging.getLogger(__name__)

#: The scores, in report order: (row name, per-query column, the final line's field).
SCORES = ((m.RULE_BASED, "belief_rule_based", "certainty_formula"),
          (m.LLM_SCORE, "belief_llm_score", "certainty"))
#: The headline score (the queries scored are those with it).
HEADLINE = "belief_llm_score"


def belief_scores(finals: Dict[str, Optional[Dict[str, Any]]],
                  accuracy_per_query: List[Dict[str, Any]]) -> GroupResult:
    """The rows and per-query values from the accuracy records (``correct``,
    ``judge_error``) and each query's ``final`` belief line (*finals*, by
    query id); empty without any LLM score."""
    per_query: Dict[str, Dict[str, Any]] = {}
    for rec in accuracy_per_query:
        qid = rec["query_id"]
        final = finals.get(qid) or {}
        row = {
            "y": None if rec.get("judge_error") else int(bool(rec.get("correct"))),
            "belief_llm_score_label": final.get("certainty_label"),
        }
        has_scorer = final.get("final_score") is not None
        for _, column, fld in SCORES:
            row[column] = final.get(fld) if (has_scorer or column != HEADLINE) else None
        per_query[qid] = row
    if not any(r[HEADLINE] is not None for r in per_query.values()):
        return GroupResult()

    judged = [r for r in per_query.values() if r["y"] is not None]
    common = [r for r in judged if r[HEADLINE] is not None]
    left_out = {"judge error": len(per_query) - len(judged), "no LLM score": len(judged) - len(common)}
    if any(left_out.values()):
        logger.info("uncertainty evaluation: queries left out %s", left_out)

    def stat(fn, name):
        return lambda rs: fn(np.array([r["y"] for r in rs], dtype=float),
                             np.array([r[name] for r in rs], dtype=float)) if rs else math.nan

    columns = ("belief_llm_score", "belief_llm_score_label", "belief_rule_based")
    res = GroupResult(per_query={q: {c: r[c] for c in columns} for q, r in per_query.items()})
    for name, column, _ in SCORES:
        scored = [r for r in common if r[column] is not None]
        res.add(m.GROUP, m.auroc_row(name), stat(auroc, column)(scored),
                *bootstrap_ci(scored, stat(auroc, column)), len(scored))
        res.add(m.GROUP, m.brier_row(name), stat(brier, column)(scored), n=len(scored))
    return res


class UncertaintyEvaluator:
    """:func:`belief_scores` of each belief model; QA datasets whose answers are scored."""

    name = "uncertainty"
    scope = BELIEF_MODEL
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["UncertaintyEvaluator"]:
        return cls() if setup.task == "qa" and setup.has_accuracy else None

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        if not ctx.accuracy:
            return GroupResult()
        return belief_scores(ctx.finals, ctx.accuracy["per_query"])
