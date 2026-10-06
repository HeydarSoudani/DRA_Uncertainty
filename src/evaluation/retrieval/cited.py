"""Cited-doc retrieval evaluator.

"Cited" documents are those referenced by the final report, resolved by
:func:`evaluation.retrieval.citations.resolve_cited_docs`.  They form a single
ranked list (no per-step fusion).  Queries with no cited docs are skipped.
"""

from pathlib import Path
from typing import Any, Dict, List

from .base import BaseDocRetrievalEvaluator
from .citations import resolve_cited_docs


class CitedDocEvaluator(BaseDocRetrievalEvaluator):
    """Retrieval quality restricted to the documents cited by the report.

    Usage::

        evaluator = CitedDocEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
    """

    emit_metrics_at_n = True

    @classmethod
    def doc_iterations(cls, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        docs = resolve_cited_docs(result)
        return [docs] if docs else []

    @classmethod
    def save_item(cls, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save the cited docs of one query as a TREC file.

        The run_tag column keeps each doc's originating step (``iter_N``)
        rather than the single list's ``iter_1``.  Nothing is written when
        the report cites nothing.
        """
        docs = resolve_cited_docs(result)
        if not docs:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        lines: List[str] = [
            f"{query_id} Q0 {doc['doc_id']} {rank} {1.0 / rank:.6f} iter_{doc['iter']}"
            for rank, doc in enumerate(docs, 1)
        ]
        with open(Path(output_dir) / f"{query_id}.trec", "w") as f:
            f.write("\n".join(lines) + "\n")
