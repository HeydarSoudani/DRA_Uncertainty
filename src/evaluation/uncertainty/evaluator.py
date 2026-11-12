"""Uncertainty evaluation: does a certainty score rank the agent's correct
answers above its wrong ones, and is it calibrated?

Per query of a short-answer dataset (``task == "qa"``), y is the
correctness of the agent's final answer (``accuracy.jsonl``: the judge's
``correct``, TRQA's exact match; a judge error leaves the query out), and
the scores are:

- ``s_T``: the belief model's final scorer (the ``final`` line of
  ``belief/{qid}.jsonl``, ``certainty``), the headline;
- ``verbal``: the agent's own confidence in its answer
  (``answer.confidence``, glm and oss), the baseline;
- ``formula`` and ``answer formula``: the fixed function of the final
  beliefs' statuses (``certainty_formula``) and its form on the scorer's
  answer candidates (``certainty_answer_formula``), ablations.

Every score is measured on the same queries: those with y, s_T and (when
any query of the run has one) a verbal confidence; an ablation on those of
them where it has a value (the answer formula has none without
criteria), with its own n.  Rows: the
queries scored; AUROC with a bootstrap CI over queries for every score;
the Brier score for s_T and the verbal confidence.  The queries left out
are logged.  No LLM call.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..metrics import auroc, bootstrap_ci, brier

logger = logging.getLogger(__name__)

#: The scores, in report order: (name, the final line's field or None for
#: the verbal confidence).
SCORES = (("s_T", "certainty"), ("verbal", None), ("formula", "certainty_formula"),
          ("answer formula", "certainty_answer_formula"))
#: The scores that select the queries and have a Brier row; the others are
#: ablations (AUROC only).
HEADLINE = ("s_T", "verbal")


class UncertaintyEvaluator:
    """The uncertainty measures of a run (module docstring)."""

    def evaluate(self, finals: Dict[str, Optional[Dict[str, Any]]],
                 accuracy_per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
        """``{per_query, rows}`` from the accuracy records (``correct``,
        ``judge_error``, ``verbal_confidence``) and each query's ``final``
        belief line (*finals*, by query id); ``{}`` without any s_T."""
        per_query: Dict[str, Dict[str, Any]] = {}
        for rec in accuracy_per_query:
            qid = rec["query_id"]
            final = finals.get(qid) or {}
            row = {
                "y": None if rec.get("judge_error") else int(bool(rec.get("correct"))),
                "verbal": rec.get("verbal_confidence"),
                "s_T_label": final.get("certainty_label"),
            }
            # s_T is the scorer's only when it ran (a short-answer query).
            has_scorer = final.get("final_score") is not None
            for name, field in SCORES:
                if field is not None:
                    row[name] = final.get(field) if (has_scorer or name != "s_T") else None
            per_query[qid] = row
        if not any(r["s_T"] is not None for r in per_query.values()):
            return {}

        use_verbal = any(r["verbal"] is not None for r in per_query.values())
        names = [n for n, _ in SCORES if n != "verbal" or use_verbal]
        judged = [r for r in per_query.values() if r["y"] is not None]
        common = [r for r in judged if all(r[n] is not None for n in names if n in HEADLINE)]
        left_out = {"judge error": len(per_query) - len(judged),
                    "no s_T": sum(r["s_T"] is None for r in judged)}
        if use_verbal:
            left_out["no verbal confidence"] = sum(r["verbal"] is None for r in judged)
        if any(left_out.values()):
            logger.info("uncertainty evaluation: queries left out (a query may count twice) %s", left_out)

        def stat(fn, name):
            return lambda rs: fn(np.array([r["y"] for r in rs], dtype=float),
                                 np.array([r[name] for r in rs], dtype=float)) if rs else math.nan

        rows: List[Tuple[str, Any, Any, Any, Any]] = [("queries scored", len(common), None, None, len(per_query))]
        for name in names:
            scored = [r for r in common if r[name] is not None]
            rows.append((f"{name} AUROC", stat(auroc, name)(scored), *bootstrap_ci(scored, stat(auroc, name)),
                         len(scored)))
            if name in HEADLINE:
                rows.append((f"{name} Brier", stat(brier, name)(scored), None, None, len(scored)))
        out_rows = {q: {"y": r["y"], "s_T": r["s_T"], "s_T_label": r["s_T_label"],
                        "certainty_formula": r["formula"], "certainty_answer_formula": r["answer formula"]}
                    for q, r in per_query.items()}
        return {"per_query": out_rows, "rows": rows}
