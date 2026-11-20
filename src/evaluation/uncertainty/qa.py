"""Uncertainty evaluation of a belief model: does its certainty score rank
the agent's correct answers above its wrong ones, and is it calibrated?

Per query of a short-answer dataset (``task == "qa"``), y is the
correctness of the agent's final answer (``eval/generation/accuracy.jsonl``: the judge's
``correct``, TRQA's exact match; a judge error leaves the query out), and
the scores are:

- ``belief rule-based``: the fixed function of the final beliefs'
  statuses (``certainty_formula``: the mean status probability);
- ``belief LLM score``: the belief model's final scorer, s_T (the
  ``final`` line of ``belief/{model}/{qid}.jsonl``, ``certainty``), the
  headline.

Both are measured on the queries with y and the LLM score; the rule-based
score on those of them where it has a value, with its own n.  Rows per
score: AUROC with a bootstrap CI over queries and the Brier score.  The
queries left out are logged.  No LLM call.  The baseline, the agent's
verbal confidence, is a measure of the generation (:mod:`.verbal`), scored
once per run.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..measures import auroc, bootstrap_ci, brier
from ..metrics_file import GroupResult

logger = logging.getLogger(__name__)

#: The scores, in report order: (row name, per-query column, the final line's field).
SCORES = (("belief rule-based", "belief_rule_based", "certainty_formula"),
          ("belief LLM score", "belief_llm_score", "certainty"))
#: The headline score (the queries scored are those with it).
HEADLINE = "belief_llm_score"


class UncertaintyEvaluator:
    """The uncertainty measures of a run (module docstring)."""

    def evaluate(self, finals: Dict[str, Optional[Dict[str, Any]]],
                 accuracy_per_query: List[Dict[str, Any]]) -> GroupResult:
        """The ``uncertainty`` group from the accuracy records (``correct``,
        ``judge_error``) and each query's ``final``
        belief line (*finals*, by query id); empty without any LLM score."""
        per_query: Dict[str, Dict[str, Any]] = {}
        for rec in accuracy_per_query:
            qid = rec["query_id"]
            final = finals.get(qid) or {}
            row = {
                "y": None if rec.get("judge_error") else int(bool(rec.get("correct"))),
                "belief_llm_score_label": final.get("certainty_label"),
            }
            # The LLM score is the scorer's only when it ran (a short-answer query).
            has_scorer = final.get("final_score") is not None
            for _, column, field in SCORES:
                row[column] = final.get(field) if (has_scorer or column != HEADLINE) else None
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
            res.add("uncertainty", f"{name} AUROC", stat(auroc, column)(scored),
                    *bootstrap_ci(scored, stat(auroc, column)), len(scored))
            res.add("uncertainty", f"{name} Brier", stat(brier, column)(scored), n=len(scored))
        return res
