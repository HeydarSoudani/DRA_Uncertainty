"""Uncertainty of a report: does the belief's certainty rank the agent's
reports by how many nuggets they cover?

Per query of a report dataset (``task == "report"``), s_T is the certainty
of the final belief (the ``final`` line of ``belief/{qid}.jsonl``,
``certainty``: the geometric mean of the aspects' coverage statuses,
``belief_model_component.core.certainty``; a report has no final scorer),
and y is the report's nugget coverage from Auto-ARGUE
(``evaluation.answer.argue``), weighted too when the dataset has nugget
importance.  Measure: Kendall's tau-b of s_T and y over the queries, with a
bootstrap CI over queries.  Queries without s_T or without a scored report
(an empty report, a judge failure) are left out and counted.  No LLM call.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..metrics import bootstrap_ci, tau_b

#: The report scores s_T is compared with: (row name, ARGUE per-query key).
TARGETS = (("nugget coverage", "nugget_coverage"), ("nugget coverage (weighted)", "nugget_coverage_weighted"))


class ReportUncertaintyEvaluator:
    """The report uncertainty measures of a run (module docstring)."""

    def evaluate(self, finals: Dict[str, Optional[Dict[str, Any]]],
                 report_per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
        """``{per_query, rows}`` from the ARGUE records (``nugget_coverage``,
        ``empty_report``) and each query's ``final`` belief line (*finals*,
        by query id); ``{}`` without any s_T."""
        per_query: Dict[str, Dict[str, Any]] = {}
        for rec in report_per_query:
            final = finals.get(rec["query_id"]) or {}
            per_query[rec["query_id"]] = {"s_T": final.get("certainty"), "empty": bool(rec.get("empty_report")),
                                          **{key: rec.get(key) for _, key in TARGETS}}
        if not any(r["s_T"] is not None for r in per_query.values()):
            return {}
        scored = [r for r in per_query.values() if r["s_T"] is not None and not r["empty"]]
        rows: List[Tuple[str, Any, Any, Any, Any]] = [
            ("queries scored", len(scored), None, None, len(per_query)),
            ("excluded: no s_T", sum(r["s_T"] is None for r in per_query.values()), None, None, None),
            ("s_T mean", float(np.mean([r["s_T"] for r in scored])) if scored else math.nan, None, None,
             len(scored)),
        ]
        for name, key in TARGETS:
            items = [r for r in scored if r.get(key) is not None]
            if not items:
                continue

            def stat(rs, key=key):
                return tau_b(np.array([r["s_T"] for r in rs], dtype=float),
                             np.array([r[key] for r in rs], dtype=float)) if rs else math.nan

            rows.append((f"s_T tau-b vs {name}", stat(items), *bootstrap_ci(items, stat), len(items)))
        return {"per_query": {q: {"s_T": r["s_T"]} for q, r in per_query.items()}, "rows": rows}
