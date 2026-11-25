"""A report's certainty against its nugget coverage: does the belief rank the
agent's reports by how many nuggets they cover?

Report datasets, once per belief model.  Per query: s_T is the certainty of
the final belief (``certainty`` of the ``final`` line: the mean of the
aspects' status probabilities, ``belief_model_component.core.certainty``; a
report has no final scorer); y is the report's Auto-ARGUE nugget coverage
(``generation.argue``), weighted too when the nuggets have an importance.
Queries without s_T or with an empty report are left out and logged; a
report whose judging failed has no ARGUE record, so it is not counted here
(the generation group's ``judge failures``).

Rows (``uncertainty`` group):
    queries scored                         n = the reports with an ARGUE record
    s_T tau-b vs nugget coverage           Kendall's tau-b (the ranking),
                                           bootstrap CI over queries
    s_T Brier vs nugget coverage           s_T as the expected coverage (the level)
    ... and the same vs nugget coverage (weighted)

Per query: ``s_T``.  No LLM call.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..shared.context import BELIEF_MODEL, EvalContext, EvalSetup
from ..shared.measures import bootstrap_ci, brier, tau_b
from ..shared.result import GroupResult
from . import metrics as m

logger = logging.getLogger(__name__)

#: The report scores s_T is compared with: (row name, ARGUE per-query key).
TARGETS = ((m.COVERAGE, "nugget_coverage"), (m.COVERAGE_WEIGHTED, "nugget_coverage_weighted"))


def report_scores(finals: Dict[str, Optional[Dict[str, Any]]],
                  report_per_query: List[Dict[str, Any]]) -> GroupResult:
    """The rows and per-query values from the ARGUE records
    (``nugget_coverage``, ``empty_report``) and each query's ``final``
    belief line (*finals*, by query id); empty without any s_T."""
    per_query: Dict[str, Dict[str, Any]] = {}
    for rec in report_per_query:
        final = finals.get(rec["query_id"]) or {}
        per_query[rec["query_id"]] = {"s_T": final.get("certainty"), "empty": bool(rec.get("empty_report")),
                                      **{key: rec.get(key) for _, key in TARGETS}}
    if not any(r["s_T"] is not None for r in per_query.values()):
        return GroupResult()
    scored = [r for r in per_query.values() if r["s_T"] is not None and not r["empty"]]
    if len(scored) < len(per_query):
        logger.info("report uncertainty evaluation: %d queries left out (no s_T or an empty report)",
                    len(per_query) - len(scored))
    res = GroupResult(per_query={q: {"s_T": r["s_T"]} for q, r in per_query.items()})
    res.add(m.GROUP, m.QUERIES_SCORED, len(scored), n=len(per_query))
    for name, key in TARGETS:
        items = [r for r in scored if r.get(key) is not None]
        if not items:
            continue

        def stat(rs, key=key):
            return tau_b(np.array([r["s_T"] for r in rs], dtype=float),
                         np.array([r[key] for r in rs], dtype=float)) if rs else math.nan

        res.add(m.GROUP, m.tau_vs(name), stat(items), *bootstrap_ci(items, stat), len(items))
        res.add(m.GROUP, m.brier_vs(name), brier(np.array([r[key] for r in items], dtype=float),
                                                 np.array([r["s_T"] for r in items], dtype=float)),
                n=len(items))
    return res


class ReportUncertaintyEvaluator:
    """:func:`report_scores` of each belief model; report datasets whose reports are scored."""

    name = "uncertainty"
    scope = BELIEF_MODEL
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["ReportUncertaintyEvaluator"]:
        return cls() if setup.task == "report" and setup.has_report_eval else None

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        if not ctx.report:
            return GroupResult()
        return report_scores(ctx.finals, ctx.report["per_query"])
