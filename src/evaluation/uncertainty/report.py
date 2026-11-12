"""Uncertainty of a report: does the belief's certainty rank the agent's
reports by how many nuggets they cover?

Per query of a report dataset (``task == "report"``), s_T is the certainty
of the final belief (the ``final`` line of ``belief/{qid}.jsonl``,
``certainty``: the mean of the aspects' status probabilities,
``belief_model_component.core.certainty``; a report has no final scorer),
and y is the report's nugget coverage from Auto-ARGUE
(``evaluation.answer.argue``), weighted too when the dataset has nugget
importance.  Measures: Kendall's tau-b of s_T and y over the queries, with a
bootstrap CI over queries (the ranking), and the Brier score of s_T against
y (the level: s_T as the expected coverage).  Queries without s_T or
without a scored report (an empty report, a judge failure) are left out and
logged.  No LLM call.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..metrics import bootstrap_ci, brier, tau_b

logger = logging.getLogger(__name__)

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
        if len(scored) < len(per_query):
            logger.info("report uncertainty evaluation: %d queries left out (no s_T or an empty report)",
                        len(per_query) - len(scored))
        rows: List[Tuple[str, Any, Any, Any, Any]] = [("queries scored", len(scored), None, None, len(per_query))]
        for name, key in TARGETS:
            items = [r for r in scored if r.get(key) is not None]
            if not items:
                continue

            def stat(rs, key=key):
                return tau_b(np.array([r["s_T"] for r in rs], dtype=float),
                             np.array([r[key] for r in rs], dtype=float)) if rs else math.nan

            rows.append((f"s_T tau-b vs {name}", stat(items), *bootstrap_ci(items, stat), len(items)))
            rows.append((f"s_T Brier vs {name}", brier(np.array([r[key] for r in items], dtype=float),
                                                       np.array([r["s_T"] for r in items], dtype=float)),
                         None, None, len(items)))
        return {"per_query": {q: {"s_T": r["s_T"]} for q, r in per_query.items()}, "rows": rows}
