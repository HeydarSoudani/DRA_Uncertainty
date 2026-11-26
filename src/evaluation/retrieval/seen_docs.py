"""Retrieval quality of the documents the agent *saw*.

Per query with a relevant document: the distinct documents shown to the
agent over the run (``utils.run_files.retrieval``, seen level; the surfaced
and cited levels are saved only), N of them, against the relevant
documents of the qrels (at the run's ``min_relevance_score``).

Rows (``retrieval`` group), means over those queries:
    Recall@N        relevant seen / relevant
    Precision@N     relevant seen / N (0 when nothing was seen)
    N (docs seen)   N

Per query: ``n_seen``, ``recall_at_n``, ``precision_at_n``.  No LLM call.
"""

from typing import Any, Dict, Optional

from utils.run_files.retrieval import doc_ids, seen_iterations

from ..shared.context import GENERATION, EvalContext, EvalSetup
from ..shared.measures import mean
from ..shared.result import GroupResult
from .metrics import GROUP, N_SEEN, PRECISION, RECALL


def seen_doc_scores(results: Dict[str, Dict[str, Any]], qrels: Dict[str, Dict[str, Any]]) -> GroupResult:
    """The rows and per-query values of the module docstring."""
    out: Dict[str, Dict[str, Any]] = {}
    for qid, result in results.items():
        relevant = {d for d, rel in (qrels.get(qid) or {}).items() if rel > 0}
        if not relevant:
            continue
        seen = set(doc_ids(seen_iterations(result)))
        hits = len(seen & relevant)
        out[qid] = {
            "n_seen": len(seen),
            "recall_at_n": hits / len(relevant),
            "precision_at_n": hits / len(seen) if seen else 0.0,
        }
    res = GroupResult(per_query=out)
    for metric, key in ((RECALL, "recall_at_n"), (PRECISION, "precision_at_n"), (N_SEEN, "n_seen")):
        res.add(GROUP, metric, mean([r[key] for r in out.values()]), n=len(out))
    return res


class SeenDocEvaluator:
    """:func:`seen_doc_scores` of a run; every dataset."""

    name = "retrieval"
    scope = GENERATION
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["SeenDocEvaluator"]:
        return cls()

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        return seen_doc_scores(ctx.results, ctx.qrels)
