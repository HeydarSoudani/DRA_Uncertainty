"""Uncertainty evaluation: does a certainty score rank the agent's correct
answers above its wrong ones, and is it calibrated?

Per query of a short-answer dataset (``task == "qa"``), y is the
correctness of the agent's final answer (``accuracy.jsonl``: the judge's
``correct``, TRQA's exact match; a judge error leaves the query out), and
the scores are:

- ``s_T``: the belief model's final scorer (the ``final`` line of
  ``belief/{qid}.jsonl``, ``certainty``), the headline;
- ``formula`` and ``answer formula``: the fixed function of the final
  beliefs' statuses (``certainty_formula``) and its form on the scorer's
  answer candidates (``certainty_answer_formula``), ablations;
- ``verbal``: the agent's own confidence in its answer
  (``answer.confidence``, glm and oss), the baseline.

The headline scores (s_T and the verbal confidence) are measured on the
same queries: those with y, s_T and (when any query of the run has one) a
verbal confidence.  An ablation is measured on those of them where it has a
value (the answer formula has none when every criterion is a rest or an
aggregate one), with its own count.  Measures per score:
AUROC with a bootstrap CI over queries, Brier score and ECE; and the
paired AUROC difference of s_T and the verbal confidence with its CI.  The
queries left out are counted.  No LLM call.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..metrics import auroc, bootstrap_ci, brier, ece, paired_diff_ci

#: The scores, in report order: (name, the final line's field or None for
#: the verbal confidence).
SCORES = (("s_T", "certainty"), ("verbal", None), ("formula", "certainty_formula"),
          ("answer formula", "certainty_answer_formula"))
#: The scores that select the queries; the others are ablations.
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
        judged = {q: r for q, r in per_query.items() if r["y"] is not None}
        common = [r for r in judged.values() if all(r[n] is not None for n in names if n in HEADLINE)]
        rows: List[Tuple[str, Any, Any, Any, Any]] = [
            ("queries judged", len(judged), None, None, len(per_query)),
            ("queries scored", len(common), None, None, len(judged)),
            ("excluded: no s_T", sum(r["s_T"] is None for r in judged.values()), None, None, None),
        ]
        if use_verbal:
            rows.append(("excluded: no verbal confidence", sum(r["verbal"] is None for r in judged.values()),
                         None, None, None))
        rows.append(("correct rate", float(np.mean([r["y"] for r in common])) if common else math.nan,
                     None, None, len(common)))

        def stat(fn, name):
            return lambda rs: fn(np.array([r["y"] for r in rs], dtype=float),
                                 np.array([r[name] for r in rs], dtype=float)) if rs else math.nan

        for name in names:
            scored = [r for r in common if r[name] is not None]
            rows.append((f"{name} AUROC", stat(auroc, name)(scored), *bootstrap_ci(scored, stat(auroc, name)),
                         len(scored)))
            rows.append((f"{name} Brier", stat(brier, name)(scored), None, None, len(scored)))
            rows.append((f"{name} ECE", stat(ece, name)(scored), None, None, len(scored)))
        if use_verbal:
            rows.append(("s_T - verbal AUROC", *paired_diff_ci(common, stat(auroc, "s_T"), stat(auroc, "verbal")),
                         len(common)))
        out_rows = {q: {"y": r["y"], "s_T": r["s_T"], "s_T_label": r["s_T_label"],
                        "certainty_formula": r["formula"], "certainty_answer_formula": r["answer formula"]}
                    for q, r in per_query.items()}
        return {"per_query": out_rows, "rows": rows}
