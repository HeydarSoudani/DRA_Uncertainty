"""Retrieval evaluation (the ``retrieval`` group of the run's ``metrics.md``).

Only the *seen* documents are scored (``run_files.retrieval``: the docs
passed to the agent/LLM); the surfaced and cited levels are saved only.
"""

from typing import Any, Dict

from run_files.retrieval import doc_ids, seen_iterations


class SeenDocEvaluator:
    """Retrieval quality of the *seen* documents: per query the set of
    distinct docs shown to the agent over the run, N of them, against the
    relevant docs of the qrels (at the run's ``min_relevance_score``).

    Usage::

        evaluator = SeenDocEvaluator(qrels=qrels)
        out = evaluator.evaluate(results)
    """

    def __init__(self, qrels: Dict[str, Dict[str, int]]) -> None:
        self.qrels = qrels

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Per query with a relevant doc: ``{n_seen, recall_at_n,
        precision_at_n}`` (Recall@N = relevant seen / relevant; Precision@N =
        relevant seen / N, 0 when nothing was seen)."""
        out: Dict[str, Dict[str, Any]] = {}
        for qid, result in results.items():
            relevant = {d for d, rel in (self.qrels.get(qid) or {}).items() if rel > 0}
            if not relevant:
                continue
            seen = set(doc_ids(seen_iterations(result)))
            hits = len(seen & relevant)
            out[qid] = {
                "n_seen": len(seen),
                "recall_at_n": hits / len(relevant),
                "precision_at_n": hits / len(seen) if seen else 0.0,
            }
        return out
