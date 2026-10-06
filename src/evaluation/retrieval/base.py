"""Shared base class for the three document-level retrieval evaluators.

All three evaluators answer the same question — *how good is this set of
documents against the qrels?* — but differ only in **which** documents they
look at:

* :class:`~evaluation.retrieval.surfaced.SurfacedDocEvaluator`: every doc the
  retriever returned in each step (top-k surfaced).
* :class:`~evaluation.retrieval.seen.SeenDocEvaluator`: the subset actually
  passed to the agent/LLM (seen-top-k).
* :class:`~evaluation.retrieval.cited.CitedDocEvaluator`: the docs cited by the
  final report.

Subclasses implement a single hook, :meth:`doc_iterations`, returning one
ranked doc list per retrieval step.  Everything else (fusing those lists into
a final ranking, TREC evaluation, Metrics@N, per-query TREC
saving) lives here.  :meth:`doc_iterations` and :meth:`save_item` are
classmethods, so the per-query files are written without an evaluator.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.ranking_results import RankingResults, RankingResult
from utils.text_utils import doc_id as _doc_id
from searcher_component.fusion import fuse_retrieval_results

from .metrics import DEFAULT_K_VALUES, evaluate_results, metrics_at_n

logger = logging.getLogger(__name__)


class BaseDocRetrievalEvaluator:
    """Evaluate retrieval quality for one document level (surfaced/seen/cited).

    Subclasses override :meth:`doc_iterations` (required) and optionally
    :meth:`_score` (per-doc rank score) and :attr:`emit_metrics_at_n`.
    """

    #: Whether :meth:`evaluate` adds a ``Metrics@N`` block (N = #docs per query).
    emit_metrics_at_n: bool = True

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
        fusion_method: str = "interleaving",
        interleaving_window: Optional[int] = 3,
        rrf_k: int = 60,
        graded_qrels: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """Initialise the evaluator.

        Args:
            qrels:               Ground-truth relevance judgements {query_id: {doc_id: score}}.
            k_values:            Cut-off values for TREC metrics. Defaults to DEFAULT_K_VALUES.
            fusion_method:       Method used to consolidate per-step ranked lists
                                 ("interleaving" or "rrf").
            interleaving_window: Block size for interleaving fusion. Ignored for rrf.
            rrf_k:               K constant for reciprocal rank fusion. Ignored otherwise.
            graded_qrels:        {query_id: {doc_id: gain}} with the official gains,
                                 for GradedRecall@N and as the NDCG gains.
                                 None: GradedRecall@N is not reported and
                                 NDCG uses the ``qrels`` grades.
        """
        self.qrels = qrels
        self.graded_qrels = graded_qrels
        self.k_values = list(k_values or DEFAULT_K_VALUES)
        self.fusion_method = fusion_method
        self.interleaving_window = interleaving_window
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def doc_iterations(cls, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Return one ranked doc list per retrieval step of *result*.

        Each inner list is a list of doc dicts (or doc-id strings).  Return an
        empty list to skip the query.  Subclasses must implement this.
        """
        raise NotImplementedError

    @staticmethod
    def _score(doc: Dict[str, Any], rank: int) -> float:
        """Return the rank score written for *doc* at 1-based *rank*."""
        return 1.0 / rank

    # ------------------------------------------------------------------
    # Shared ranking construction
    # ------------------------------------------------------------------

    def build_ranking_results(self, results: Dict[str, Dict[str, Any]]) -> RankingResults:
        """Convert per-step doc lists into a fused :class:`RankingResults`.

        Queries without qrels are skipped.
        """
        ranking_results = RankingResults(results=[])

        for query_id, result in results.items():
            if query_id not in self.qrels:
                continue
            iterations = self.doc_iterations(result)
            if not iterations:
                continue

            # Track the first step each doc_id appeared in (for the run_tag).
            doc_first_iter: Dict[str, int] = {}
            for iter_idx, iteration_docs in enumerate(iterations, 1):
                for doc in iteration_docs:
                    did = _doc_id(doc)
                    if did and did not in doc_first_iter:
                        doc_first_iter[did] = iter_idx

            if len(iterations) == 1:
                docs = iterations[0]
            else:
                docs = fuse_retrieval_results(
                    iterations,
                    fusion_method=self.fusion_method,
                    rrf_k=self.rrf_k,
                    interleaving_window=self.interleaving_window,
                )

            for rank, doc in enumerate(docs, 1):
                did = _doc_id(doc)
                if did:
                    iter_idx = doc_first_iter.get(did, 1)
                    ranking_results.add_result(RankingResult(
                        query_id=query_id,
                        doc_id=did,
                        rank=rank,
                        rank_score=self._score(doc, rank),
                        metadata={"run_tag": f"iter_{iter_idx}"},
                    ))

        return ranking_results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval quality and return a metrics dict.

        Returns a flat dict with NDCG/MAP/Recall/Precision/F1/Success (each
        ``metric@k -> value``) plus ``num_queries`` and ``avg_docs_per_query``,
        and, when :attr:`emit_metrics_at_n`, a ``Metrics@N`` block.
        Returns ``{}`` when there are no results or no qrels.
        """
        ranking_results = self.build_ranking_results(results)
        if len(ranking_results.results) == 0 or len(self.qrels) == 0:
            logger.warning("No retrieval results or qrels available; skipping evaluation.")
            return {}

        num_queries = len(ranking_results.get_unique_queries())
        metrics = evaluate_results(
            results=ranking_results,
            qrels=self.qrels,
            k_values=self.k_values,
            gain_qrels=self.graded_qrels,
        )
        metrics["num_queries"] = num_queries
        metrics["avg_docs_per_query"] = (
            len(ranking_results.results) / num_queries if num_queries > 0 else 0.0
        )

        if self.emit_metrics_at_n:
            metrics["Metrics@N"] = metrics_at_n(self.qrels, ranking_results, self.graded_qrels)

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def save_item(cls, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query retrieval results as a TREC file.

        Docs are written per retrieval step; the 6th column records the step
        number (``iter_1``, ``iter_2``, ...).

        Standard TREC columns: ``qid Q0 doc_id rank score run_tag``
        """
        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        iterations = cls.doc_iterations(result)
        lines: List[str] = []
        for iter_idx, iteration_docs in enumerate(iterations, 1):
            for rank, doc in enumerate(iteration_docs, 1):
                did = _doc_id(doc)
                if not did:
                    continue
                score = cls._score(doc, rank)
                lines.append(f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_idx}")

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with open(trec_path, "w") as f:
            f.write(content)
