"""Uncertainty evaluation of a belief model: does its certainty score rank
the agent's correct answers above its wrong ones, and is it calibrated?

Per query of a short-answer dataset (``task == "qa"``), y is the
correctness of the agent's final answer (``eval/generation/accuracy.jsonl``: the judge's
``correct``, TRQA's exact match; a judge error leaves the query out), and
the scores are:

- ``s_T``: the belief model's final scorer (the ``final`` line of
  ``belief/{model}/{qid}.jsonl``, ``certainty``), the headline;
- ``formula``: the fixed function of the final beliefs' statuses
  (``certainty_formula``), an ablation.

Every score is measured on the queries with y and s_T; an ablation on
those of them where it has a value, with its own n.  Rows: the queries
scored; AUROC with a bootstrap CI over queries for every score; the Brier
score for s_T.  The queries left out are logged.  No LLM call.  The
baseline, the agent's verbal confidence, is a measure of the generation
(:mod:`.verbal`), scored once per run.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..measures import auroc, bootstrap_ci, brier
from ..metrics_file import GroupResult

logger = logging.getLogger(__name__)

#: The scores, in report order: (name, the final line's field).
SCORES = (("s_T", "certainty"), ("formula", "certainty_formula"))
#: The scores with a Brier row; the others are ablations (AUROC only).
HEADLINE = ("s_T",)


class UncertaintyEvaluator:
    """The uncertainty measures of a run (module docstring)."""

    def evaluate(self, finals: Dict[str, Optional[Dict[str, Any]]],
                 accuracy_per_query: List[Dict[str, Any]]) -> GroupResult:
        """The ``uncertainty`` group from the accuracy records (``correct``,
        ``judge_error``) and each query's ``final``
        belief line (*finals*, by query id); empty without any s_T."""
        per_query: Dict[str, Dict[str, Any]] = {}
        for rec in accuracy_per_query:
            qid = rec["query_id"]
            final = finals.get(qid) or {}
            row = {
                "y": None if rec.get("judge_error") else int(bool(rec.get("correct"))),
                "s_T_label": final.get("certainty_label"),
            }
            # s_T is the scorer's only when it ran (a short-answer query).
            has_scorer = final.get("final_score") is not None
            for name, field in SCORES:
                row[name] = final.get(field) if (has_scorer or name != "s_T") else None
            per_query[qid] = row
        if not any(r["s_T"] is not None for r in per_query.values()):
            return GroupResult()

        judged = [r for r in per_query.values() if r["y"] is not None]
        common = [r for r in judged if r["s_T"] is not None]
        left_out = {"judge error": len(per_query) - len(judged), "no s_T": len(judged) - len(common)}
        if any(left_out.values()):
            logger.info("uncertainty evaluation: queries left out %s", left_out)

        def stat(fn, name):
            return lambda rs: fn(np.array([r["y"] for r in rs], dtype=float),
                                 np.array([r[name] for r in rs], dtype=float)) if rs else math.nan

        res = GroupResult(per_query={q: {"s_T": r["s_T"], "s_T_label": r["s_T_label"],
                                         "certainty_formula": r["formula"]}
                                     for q, r in per_query.items()})
        res.add("uncertainty", "queries scored", len(common), n=len(per_query))
        for name, _ in SCORES:
            scored = [r for r in common if r[name] is not None]
            res.add("uncertainty", f"{name} AUROC", stat(auroc, name)(scored),
                    *bootstrap_ci(scored, stat(auroc, name)), len(scored))
            if name in HEADLINE:
                res.add("uncertainty", f"{name} Brier", stat(brier, name)(scored), n=len(scored))
        return res
