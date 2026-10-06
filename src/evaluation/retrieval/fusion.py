"""Fusion evaluation of the surfaced docs: one fused ranking per fusion method.

For every method in the run's ``fusion_methods``, the per-step surfaced lists
of each query (:meth:`SurfacedDocEvaluator.doc_iterations`) are fused into one
ranking, scored against the qrels, and written to
``retrieval/fusion_{method}.trec``.  The metrics are returned to the caller
(``evaluate_and_save`` folds them into ``summary.json`` under
``retrieval.fusion``).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from searcher_component.fusion_methods.concatenation import SimpleConcatenation
from searcher_component.fusion_methods.epsilon_greedy import EpsilonGreedyFusion
from searcher_component.fusion_methods.interleaving import InterleavingFusion
from searcher_component.fusion_methods.random_fusion import RandomFusion
from searcher_component.fusion_methods.rrf import ReciprocalRankFusion
from searcher_component.fusion_methods.thompson_bernoulli import ThompsonBernoulliFusion
from searcher_component.fusion_methods.thompson_gaussian import ThompsonGaussianFusion
from utils.ranking_results import RankingResult, RankingResults, save_ranking_results
from utils.text_utils import doc_id as _doc_id

from .metrics import DEFAULT_K_VALUES, evaluate_results
from .surfaced import SurfacedDocEvaluator

#: Fusion method name -> factory taking ``(rrf_k, interleaving_window)``.
FUSION_METHODS: Dict[str, Callable[[int, int], Any]] = {
    "interleaving": lambda rrf_k, window: InterleavingFusion(window=window),
    "rrf": lambda rrf_k, window: ReciprocalRankFusion(k=rrf_k),
    "concatenation": lambda rrf_k, window: SimpleConcatenation(),
    "random": lambda rrf_k, window: RandomFusion(),
    "epsilon_greedy": lambda rrf_k, window: EpsilonGreedyFusion(),
    "thompson_bernoulli": lambda rrf_k, window: ThompsonBernoulliFusion(),
    "bernoulli_rank_aware": lambda rrf_k, window: ThompsonBernoulliFusion(rank_aware=True),
    "bernoulli_ucb": lambda rrf_k, window: ThompsonBernoulliFusion(ucb=True),
    "bernoulli_topk": lambda rrf_k, window: ThompsonBernoulliFusion(k=5),
    "thompson_gaussian": lambda rrf_k, window: ThompsonGaussianFusion(),
    "thompson_gaussian_nig": lambda rrf_k, window: ThompsonGaussianFusion(nig=True),
}


def build_ranking_with_fusion(
    results: Dict[str, Any],
    fusion_method: str,
    rrf_k: int = 60,
    interleaving_window: int = 3,
    fusion_k: Optional[int] = None,
) -> RankingResults:
    """Fuse each query's per-step surfaced lists with *fusion_method*.

    With *fusion_k*, only the first *fusion_k* docs of each list go through
    the fusion method; the rest are appended by interleaving.  The fused list
    keeps the first occurrence of each doc; its run_tag is the first step
    that surfaced the doc.  Queries without surfaced docs are skipped.
    """
    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"Unknown fusion method: {fusion_method!r}")
    fuser = FUSION_METHODS[fusion_method](rrf_k, interleaving_window)
    tail_fuser = InterleavingFusion(window=interleaving_window)
    ranking_results = RankingResults(results=[])

    for query_id, result in tqdm(results.items(), total=len(results), desc=fusion_method,
                                 unit="query", leave=False):
        lists = [it for it in SurfacedDocEvaluator.doc_iterations(result) if it]
        if not lists:
            continue

        doc_first_iter: Dict[str, int] = {}
        for iter_idx, docs in enumerate(lists, 1):
            for doc in docs:
                did = _doc_id(doc)
                if did and did not in doc_first_iter:
                    doc_first_iter[did] = iter_idx

        if len(lists) == 1:
            fused = lists[0]
        elif fusion_k is not None:
            fused = fuser.fuse([lst[:fusion_k] for lst in lists])
            tails = [lst[fusion_k:] for lst in lists if lst[fusion_k:]]
            if tails:
                fused = fused + (tail_fuser.fuse(tails) if len(tails) > 1 else tails[0])
        else:
            fused = fuser.fuse(lists)

        seen: set = set()
        rank = 0
        for doc in fused:
            did = _doc_id(doc)
            if not did or did in seen:
                continue
            seen.add(did)
            rank += 1
            ranking_results.add_result(RankingResult(
                query_id=query_id,
                doc_id=did,
                rank=rank,
                rank_score=doc.get("score", doc.get("rank_score", 0.0)),
                metadata={"run_tag": f"iter_{doc_first_iter.get(did, 1)}"},
            ))

    # Scores are normalized by normalize_scores_by_rank() at evaluation /
    # TREC-write time, so the original scores are kept here.
    return ranking_results


def run_fusion_eval(
    results: Dict[str, Any],
    qrels: Dict[str, Dict[str, int]],
    kwargs: Dict[str, Any],
    run_dir: Optional[Path],
    gain_qrels: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Score every fusion method of the run and write its TREC file.

    Args:
        results:    Unified results dict keyed by query_id.
        qrels:      Ground-truth relevance judgements.
        kwargs:     Pipeline kwargs (read: ``k_values``, ``fusion_methods``,
                    ``rrf_k``, ``interleaving_window``, ``fusion_k``).
        run_dir:    Run directory for ``retrieval/fusion_{method}.trec``;
                    None writes nothing.
        gain_qrels: ``{query_id: {doc_id: gain}}`` (official gains) for NDCG.

    Returns:
        ``{method: metrics}`` for every method that could be scored.
    """
    k_values = kwargs.get("k_values") or list(DEFAULT_K_VALUES)
    fusion_methods: List[str] = kwargs.get("fusion_methods") or ["interleaving"]
    fusion_k = kwargs.get("fusion_k")
    all_metrics: Dict[str, Any] = {}
    for method in fusion_methods:
        try:
            ranking = build_ranking_with_fusion(
                results, method,
                rrf_k=kwargs.get("rrf_k", 60),
                interleaving_window=kwargs.get("interleaving_window", 3),
                fusion_k=fusion_k,
            )
            if len(ranking.results) == 0 or len(qrels) == 0:
                print(f"  ⚠  [{method}] No results or qrels; skipping evaluation")
                continue

            metrics = evaluate_results(results=ranking, qrels=qrels, k_values=k_values,
                                       gain_qrels=gain_qrels)
            num_queries = len(ranking.get_unique_queries())
            metrics["num_queries"] = num_queries
            metrics["avg_docs_per_query"] = len(ranking.results) / num_queries

            if run_dir:
                save_ranking_results(ranking, f"{str(run_dir).rstrip('/')}/retrieval/fusion_{method}.trec",
                                     format_type="trec")
            all_metrics[method] = metrics
        except Exception as exc:
            print(f"  ✗ [{method}] error: {exc}")
    return all_metrics
