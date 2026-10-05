"""Shared retrieval metric computation.

TREC metric computation (NDCG, MAP, Recall, Precision, F1, Success) via
pytrec_eval, and the ``Metrics@N`` helper (where N = number of retrieved docs
per query).

Core evaluation functions modified from:
https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py
"""

import logging
from typing import Any, Dict, List, Optional

import pytrec_eval

from utils.ranking_results import RankingResults

logger = logging.getLogger(__name__)

#: Metric cut-offs when the run sets none (the inference config's default).
DEFAULT_K_VALUES = (1, 3, 5, 10, 25, 50, 75, 100, 500, 1000)


def compute_trec_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
    ignore_identical_ids: bool = True,
    include_all_metrics: bool = True,
    gain_qrels: Optional[Dict[str, Dict[str, int]]] = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Compute TREC evaluation metrics using pytrec_eval (low-level function).

    Returns a 6-tuple: (NDCG, MAP, Recall, Precision, F1, Success).

    Args:
        qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}
        results: Search results {query_id: {doc_id: score}}
        k_values: List of k values for metrics (default: DEFAULT_K_VALUES)
        ignore_identical_ids: Whether to ignore query-doc pairs with identical IDs
        include_all_metrics: Also compute uncut @all metrics.
        gain_qrels: ``{query_id: {doc_id: gain}}`` with the official gains,
            used as the NDCG gains instead of the ``qrels`` grades.  None:
            NDCG uses the ``qrels`` grades.

    Returns:
        Tuple of (NDCG, MAP, Recall, Precision, F1, Success) dictionaries
    """
    k_values = sorted(set(k_values or DEFAULT_K_VALUES))

    if ignore_identical_ids:
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    f1 = {}
    success = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
        success[f"Success@{k}"] = 0.0

    if include_all_metrics:
        # Add @all metrics (computed across all documents without cutoff)
        ndcg["NDCG@all"] = 0.0
        _map["MAP@all"] = 0.0
        recall["Recall@all"] = 0.0
        precision["P@all"] = 0.0
        success["Success@all"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    success_string = "success." + ",".join([str(k) for k in k_values])
    # NOTE: recall must be evaluated in a separate pytrec_eval call from map_cut.
    # When map_cut.k and recall.k share the same k in one evaluator, pytrec_eval
    # overwrites the recall_k key with MAP-related intermediate values (e.g. 34.5),
    # producing recall > 1. Using a dedicated recall-only evaluator avoids this collision.
    measures = {map_string, ndcg_string, precision_string, success_string}
    if include_all_metrics:
        measures |= {"ndcg", "map"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scores = evaluator.evaluate(results)

    recall_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {recall_string})
    recall_scores = recall_evaluator.evaluate(results)

    # NDCG on the official gains when given; a query without a positive gain
    # has NDCG 0.
    ndcg_scores = scores
    if gain_qrels:
        ndcg_measures = {ndcg_string, "ndcg"} if include_all_metrics else {ndcg_string}
        ndcg_scores = pytrec_eval.RelevanceEvaluator(gain_qrels, ndcg_measures).evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += ndcg_scores.get(query_id, {}).get("ndcg_cut_" + str(k), 0.0)
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += recall_scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
            success[f"Success@{k}"] += scores[query_id]["success_" + str(k)]

        if include_all_metrics:
            # Add @all metrics (without cutoff)
            ndcg["NDCG@all"] += ndcg_scores.get(query_id, {}).get("ndcg", 0.0)
            _map["MAP@all"] += scores[query_id].get("map", 0.0)

            # For Recall@all, Precision@all, Success@all: compute across all retrieved docs
            if query_id in qrels and query_id in results:
                relevant_docs = set(doc_id for doc_id, rel in qrels[query_id].items() if rel > 0)
                retrieved_docs = results[query_id].keys()
                num_retrieved = len(retrieved_docs)

                if num_retrieved > 0:
                    relevant_retrieved = relevant_docs & set(retrieved_docs)
                    recall["Recall@all"] += len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
                    precision["P@all"] += len(relevant_retrieved) / num_retrieved
                    success["Success@all"] += 1.0 if len(relevant_retrieved) > 0 else 0.0

    # Guard against division by zero if no scores
    if len(scores) == 0:
        return ndcg, _map, recall, precision, f1, success

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
        success[f"Success@{k}"] = round(success[f"Success@{k}"] / len(scores), 5)
        f1[f"F1@{k}"] = round(
            (
                2
                * (precision[f"P@{k}"] * recall[f"Recall@{k}"])
                / (precision[f"P@{k}"] + recall[f"Recall@{k}"])
                if (precision[f"P@{k}"] + recall[f"Recall@{k}"]) > 0
                else 0
            ),
            5,
        )

    if include_all_metrics:
        # Compute averaged @all metrics
        ndcg["NDCG@all"] = round(ndcg["NDCG@all"] / len(scores), 5)
        _map["MAP@all"] = round(_map["MAP@all"] / len(scores), 5)
        recall["Recall@all"] = round(recall["Recall@all"] / len(scores), 5)
        precision["P@all"] = round(precision["P@all"] / len(scores), 5)
        success["Success@all"] = round(success["Success@all"] / len(scores), 5)
        f1["F1@all"] = round(
            (
                2
                * (precision["P@all"] * recall["Recall@all"])
                / (precision["P@all"] + recall["Recall@all"])
                if (precision["P@all"] + recall["Recall@all"]) > 0
                else 0
            ),
            5,
        )

    return ndcg, _map, recall, precision, f1, success


def evaluate_results(
    results: RankingResults,
    qrels: Dict[str, Dict[str, int]],
    k_values: list,
    ignore_identical_ids: bool = True,
    gain_qrels: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Evaluate RankingResults with logging (high-level function).

    ``gain_qrels`` (official gains) replaces the ``qrels`` grades as the NDCG
    gains; see :func:`compute_trec_metrics`.

    Returns:
        Dict with keys NDCG, MAP, Recall, Precision, F1, Success, each mapping
        ``metric@k -> value``.
    """
    logger.info("Evaluating retrieval results...")

    # Normalize scores so that pytrec_eval (which re-sorts by score) sees the
    # same ordering as our rank column.
    results.normalize_scores_by_rank()

    search_results = {
        query_id: {result.doc_id: result.rank_score for result in query_results}
        for query_id, query_results in results.by_query().items()
    }

    ndcg, map_scores, recall, precision, f1, success = compute_trec_metrics(
        qrels=qrels,
        results=search_results,
        k_values=k_values,
        ignore_identical_ids=ignore_identical_ids,
        gain_qrels=gain_qrels,
    )

    evaluation_results = {
        "NDCG": ndcg,
        "MAP": map_scores,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Success": success,
    }

    logger.info("Evaluation Results:")
    for k in [1, 5, 10]:
        if f"NDCG@{k}" in ndcg:
            logger.info(f"  NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
        if f"Recall@{k}" in recall:
            logger.info(f"  Recall@{k}: {recall[f'Recall@{k}']:.4f}")

    return evaluation_results


def metrics_at_n(
    qrels: Dict[str, Dict[str, int]],
    ranking_results: RankingResults,
    graded_qrels: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Compute Metrics@N where N = number of retrieved docs for each query.

    For each query with at least one positive gold doc, N_q is the number of
    docs retrieved for that query.  Recall/Precision/F1 are computed at cutoff
    N_q, then averaged across queries.

    With ``graded_qrels`` (``{query_id: {doc_id: gain}}``, official gains) it
    also reports GradedRecall@N: summed gain of the retrieved docs / summed
    gain of all the query's docs, averaged over the queries that have a
    positive gain (``num_queries_graded``; the same queries as Recall@N
    unless ``min_relevance_score`` admits a grade with gain 0).  It equals
    Recall@N when every relevant doc has the same gain.

    Returns ``{}`` when no query can be evaluated.
    """
    recall_vals: List[float] = []
    precision_vals: List[float] = []
    f1_vals: List[float] = []
    graded_vals: List[float] = []
    n_vals: List[int] = []

    for query_id, query_results in ranking_results.by_query().items():
        gold = qrels.get(query_id, {})
        gold_ids = {doc_id for doc_id, rel in gold.items() if rel > 0}
        if not gold_ids:
            continue
        retrieved_ids = {r.doc_id for r in query_results}
        n_q = len(retrieved_ids)
        n_vals.append(n_q)
        hits = len(gold_ids & retrieved_ids)
        r = hits / len(gold_ids)
        p = hits / n_q if n_q > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        recall_vals.append(r)
        precision_vals.append(p)
        f1_vals.append(f)
        gains = (graded_qrels or {}).get(query_id, {})
        if gains:
            graded_vals.append(
                sum(g for doc_id, g in gains.items() if doc_id in retrieved_ids) / sum(gains.values())
            )

    if not recall_vals:
        return {}

    num_q = len(recall_vals)
    metrics = {
        "Recall@N": sum(recall_vals) / num_q,
        "Precision@N": sum(precision_vals) / num_q,
        "F1@N": sum(f1_vals) / num_q,
    }
    if graded_vals:
        metrics["GradedRecall@N"] = sum(graded_vals) / len(graded_vals)
    metrics["avg_N"] = sum(n_vals) / num_q
    metrics["num_queries"] = num_q
    if graded_vals:
        metrics["num_queries_graded"] = len(graded_vals)
    return metrics
