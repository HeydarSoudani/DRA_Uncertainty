"""Seen-doc retrieval evaluator.

"Seen" documents are the subset (capped at ``seen_top_k``) that was formatted
with citations and injected into the LLM prompt during planning/writing steps.
They are recorded in each trajectory search step as ``step["output"]["doc_ids"]``
(AgentCPM) or ``step["component_doc_ids"]`` (reasoning agents such as WebWeaver,
GLM, OSS, Tongyi).
"""

from typing import Any, Dict, List

from .base import BaseDocRetrievalEvaluator

_NON_SEARCH_ACTIONS = frozenset({
    "write", "init-plan", "extend-plan", "nop",
    "analyst-init_plan", "analyst-extend_plan",
})


def _extract_seen_iterations(
    trajectory: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Return one doc list per retrieval step using only the *seen* documents.

    Only steps that represent actual search/retrieval actions are included.
    Write, plan, and other non-search steps are skipped even if they carry
    ``component_doc_ids`` (e.g. CPM-Report write steps store cited doc IDs).

    Returns a list-of-lists of ``{"doc_id": str}`` dicts, one inner list per
    search step.
    """
    iterations: List[List[Dict[str, Any]]] = []
    for step in trajectory:
        action = step.get("action_type") or step.get("action") or step.get("_cpm_state") or ""
        if action in _NON_SEARCH_ACTIONS:
            continue

        # AgentCPM stores seen doc IDs in step["output"]["doc_ids"]
        output = step.get("output", {})
        doc_ids = output.get("doc_ids") if isinstance(output, dict) else None
        # Reasoning agents store them in step["component_doc_ids"] at inference
        # time; trajectories reloaded from the JSONL expose them as "seen_docs".
        if not doc_ids:
            doc_ids = step.get("component_doc_ids") or step.get("seen_docs")
        if doc_ids:
            iterations.append([{"doc_id": did} for did in doc_ids if did])
    return iterations


class SeenDocEvaluator(BaseDocRetrievalEvaluator):
    """Retrieval quality of the *seen* documents: per query the set of
    distinct docs shown to the agent over the run, N of them, against the
    relevant docs of the qrels (at the run's ``min_relevance_score``).

    Usage::

        evaluator = SeenDocEvaluator(qrels=qrels)
        out = evaluator.evaluate(results)
    """

    def __init__(self, qrels: Dict[str, Dict[str, int]]) -> None:
        self.qrels = qrels

    @classmethod
    def doc_iterations(cls, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        iterations = _extract_seen_iterations(result.get("trajectory", []))
        if not iterations:
            iterations = result.get("seen_docs_iterations", [])
        return iterations

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Per query with a relevant doc: ``{n_seen, recall_at_n,
        precision_at_n}`` (Recall@N = relevant seen / relevant; Precision@N =
        relevant seen / N, 0 when nothing was seen)."""
        out: Dict[str, Dict[str, Any]] = {}
        for qid, result in results.items():
            relevant = {d for d, rel in (self.qrels.get(qid) or {}).items() if rel > 0}
            if not relevant:
                continue
            seen = set(self.doc_ids(result))
            hits = len(seen & relevant)
            out[qid] = {
                "n_seen": len(seen),
                "recall_at_n": hits / len(relevant),
                "precision_at_n": hits / len(seen) if seen else 0.0,
            }
        return out
