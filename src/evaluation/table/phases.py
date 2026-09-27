"""Per-phase observations extracted from a GridFill run.

GridFill runs one schedule per query -- ``init_grid``, then
``(fill sweep, modify barrier)*``, then ``report`` -- and the shared
evaluators score the whole of it, exactly as they score every other agent.
Total recall and final-answer accuracy therefore stay directly comparable
across agents, and nothing in this module changes them.

What the shared evaluators deliberately do not record is *which phase* a
search belonged to, or which documents a filled cell ended up citing.  That
is what this module extracts, and it goes into GridFill's own
``tables/{qid}.jsonl`` sidecar rather than into any shared file: no shared
format grows a phase column, and ``--eval-only`` reloads the phase view from
GridFill's own record.

The extraction runs once, at save time, off the live result -- which is the
only moment the full per-step document lists and the ``phase`` tags exist in
the same place.

Sidecar record (one line in ``tables/{qid}.jsonl``)::

    {"record": "phases",
     "phases": {"init":   {"searches": 6,  "surfaced": [[docid, ...], ...],
                           "seen": [[docid, ...], ...]},
                "fill":   {...}, "modify": {...}},
     "cells":  [{"row_id": "r1", "col_id": "c1", "entity": "Michigan",
                 "status": "filled", "value": 10077331,
                 "doc_ids": ["msmarco_doc_17_..."]}, ...]}
"""

from typing import Any, Dict, List, Optional, Sequence

from utils.ranking_results import RankingResult, RankingResults
from utils.text_utils import _doc_id
from searcher_component.fusion import fuse_retrieval_results

from ..retrieval.metrics import evaluate_results
from .vocabulary import DEFAULT_VOCAB, TableVocabulary

#: The phases of the GridFill schedule, in the order they run.  ``report``
#: issues no searches, so it is not a retrieval phase and is left out.
#: Kept as a module constant for callers that predate :mod:`.vocabulary`; the
#: authoritative copy now lives on :data:`~.vocabulary.GRIDFILL_VOCAB`.
RETRIEVAL_PHASES: tuple = DEFAULT_VOCAB.retrieval_phases

#: The running prefixes of that schedule, built by :func:`cumulative_phases`.
#: ``after_modify`` is every search the run made, so it reproduces the whole-run
#: seen/surfaced numbers the shared evaluators report -- which is the point: it
#: puts the run's own recall on the same axis as the phase it was earned in.
CUMULATIVE_PHASES: tuple = DEFAULT_VOCAB.cumulative_names


def _step_surfaced(step: Dict[str, Any]) -> List[str]:
    """Doc ids surfaced by one step, mirroring ``surfaced._extract_trajectory_iterations``.

    Kept deliberately identical to the shared extractor (``all_docs`` first,
    flattened when it is a list of lists, ``docs`` as the fallback) so a
    phase's numbers are a strict slice of the shared surfaced numbers rather
    than a differently-derived quantity.
    """
    docs = None
    if step.get("all_docs"):
        all_docs = step["all_docs"]
        docs = all_docs if isinstance(all_docs[0], dict) else [
            d for retrieve_docs in all_docs for d in retrieve_docs
        ]
    elif step.get("docs"):
        docs = step["docs"]
    if not docs:
        return []
    return [did for did in (_doc_id(d) for d in docs) if did]


def _step_seen(step: Dict[str, Any]) -> List[str]:
    """Doc ids shown to the LLM by one step, mirroring ``seen._extract_seen_iterations``."""
    output = step.get("output", {})
    doc_ids = output.get("doc_ids") if isinstance(output, dict) else None
    if not doc_ids:
        doc_ids = step.get("component_doc_ids") or step.get("seen_docs")
    return [did for did in (doc_ids or []) if did]


def extract_phase_docs(trajectory: Sequence[Dict[str, Any]],
                       vocab: TableVocabulary = DEFAULT_VOCAB) -> Dict[str, Dict[str, Any]]:
    """Group a trajectory's per-step document lists by the phase that issued them.

    Only steps that actually retrieved are kept: a grid-mutating step
    (``init_grid``, ``fill_cell``, ``modify_grid``) carries no documents and
    would otherwise contribute an empty list that inflates the search count.

    Args:
        trajectory: The run's per-step dicts.
        vocab: Names the phases this agent tags its steps with. A step tagged
            with anything else is dropped -- which is how a non-retrieval phase
            like ``report`` stays out, and why passing the wrong agent's
            vocabulary yields an empty result rather than a wrong one.
    """
    phases: Dict[str, Dict[str, Any]] = {}
    for step in trajectory:
        phase = step.get("phase")
        if phase not in vocab.retrieval_phases:
            continue
        surfaced = _step_surfaced(step)
        if not surfaced:
            continue
        bucket = phases.setdefault(phase, {"surfaced": [], "seen": []})
        bucket["surfaced"].append(surfaced)
        bucket["seen"].append(_step_seen(step))

    for bucket in phases.values():
        bucket["searches"] = len(bucket["surfaced"])
    return phases


def cumulative_phases(
    per_query_phases: Dict[str, Dict[str, Dict[str, Any]]],
    vocab: TableVocabulary = DEFAULT_VOCAB,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Rewrite per-phase buckets as running prefixes of the schedule.

    The per-phase view is disjoint: ``fill``'s recall counts only what ``fill``
    itself retrieved, which is the right question for "how well did this phase
    search" and the wrong one for "what did the run have to work with by the
    end of this phase".  Documents do not expire at a phase boundary -- the
    evidence store is global and any phase may cite anything already in it --
    so the second question needs the prefix, not the slice.

    The steps are concatenated in schedule order, so the fusion downstream sees
    them in the order they actually ran.

    Args:
        per_query_phases: ``{qid: {phase: {"surfaced": [[id]], "seen": [[id]]}}}``.
        vocab: Supplies the schedule order and the prefix names.

    Returns:
        The same shape keyed by ``vocab.cumulative_names``.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for query_id, phases in (per_query_phases or {}).items():
        surfaced: List[List[str]] = []
        seen: List[List[str]] = []
        searches = 0
        buckets: Dict[str, Dict[str, Any]] = {}
        for phase, name in zip(vocab.retrieval_phases, vocab.cumulative_names):
            bucket = (phases or {}).get(phase) or {}
            surfaced = surfaced + list(bucket.get("surfaced") or [])
            seen = seen + list(bucket.get("seen") or [])
            searches += len(bucket.get("surfaced") or [])
            # A phase that ran no searches still gets a prefix entry, carrying
            # what the earlier phases retrieved: the run had those documents
            # whether or not this phase added to them, and dropping the entry
            # would silently shrink the query set of the later prefixes.
            if surfaced:
                buckets[name] = {
                    "surfaced": list(surfaced), "seen": list(seen), "searches": searches,
                }
        if buckets:
            out[query_id] = buckets
    return out


def extract_cell_evidence(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One record per resolved cell of the final grid, with the docs it cites.

    ``evidence_ids`` on a cell are store-local ids (``e7``); the run's
    ``evidence_store`` maps those to corpus doc ids.  Resolving them here means
    the sidecar carries doc ids, which is what a qrels lookup needs and what
    survives without the store.
    """
    table = result.get("table") or {}
    store = result.get("evidence_store") or {}
    columns = {c.get("col_id"): c.get("name", "") for c in (table.get("columns") or [])}

    cells: List[Dict[str, Any]] = []
    for row in table.get("rows") or []:
        for col_id, cell in (row.get("cells") or {}).items():
            status = cell.get("status")
            if status in (None, "empty"):
                continue
            doc_ids = [
                (store.get(eid) or {}).get("doc_id", "")
                for eid in (cell.get("evidence_ids") or [])
            ]
            cells.append({
                "row_id": row.get("row_id"),
                "col_id": col_id,
                "column": columns.get(col_id, ""),
                "entity": row.get("entity", ""),
                "status": status,
                "value": cell.get("value"),
                "doc_ids": [d for d in doc_ids if d],
            })
    return cells


def build_phase_record(result: Dict[str, Any],
                       vocab: TableVocabulary = DEFAULT_VOCAB) -> Optional[Dict[str, Any]]:
    """Build the sidecar's ``phases`` line, or None when there is nothing to record.

    Returns None for a result with neither retrieval steps nor resolved cells,
    so an agent without a grid (or a query that died before laying one out) adds
    no line rather than an empty one.
    """
    phases = extract_phase_docs(result.get("trajectory") or [], vocab)
    cells = extract_cell_evidence(result)
    if not phases and not cells:
        return None
    return {"record": "phases", "phases": phases, "cells": cells}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _ranking_for_phase(
    per_query: Dict[str, List[List[str]]],
    fusion_method: str,
    rrf_k: int,
    interleaving_window: Optional[int],
) -> RankingResults:
    """Fuse each query's per-step id lists into one ranking, as the shared base does.

    The steps are the same ones ``BaseDocRetrievalEvaluator.build_ranking_results``
    takes -- fuse the per-step lists, rank the fused list, score by ``1/rank``.
    Scores are then overwritten by ``normalize_scores_by_rank`` inside
    ``evaluate_results``, so rank order is the only thing that reaches the
    metrics and storing bare ids in the sidecar loses nothing.
    """
    ranking = RankingResults(results=[])
    for query_id, iterations in per_query.items():
        iterations = [it for it in iterations if it]
        if not iterations:
            continue
        if len(iterations) == 1:
            docs: List[Any] = list(iterations[0])
        else:
            docs = fuse_retrieval_results(
                [[{"doc_id": did} for did in it] for it in iterations],
                fusion_method=fusion_method,
                rrf_k=rrf_k,
                interleaving_window=interleaving_window,
            )
        for rank, doc in enumerate(docs, 1):
            did = _doc_id(doc)
            if did:
                ranking.add_result(RankingResult(
                    query_id=query_id, doc_id=did, rank=rank, rank_score=1.0 / rank,
                ))
    return ranking


def score_phase_retrieval(
    per_query_phases: Dict[str, Dict[str, Dict[str, Any]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Sequence[int],
    *,
    level: str = "surfaced",
    fusion_method: str = "interleaving",
    rrf_k: int = 60,
    interleaving_window: Optional[int] = 3,
    phase_order: Sequence[str] = RETRIEVAL_PHASES,
) -> Dict[str, Any]:
    """Retrieval metrics per phase, over the same qrels the shared evaluators use.

    A phase's recall is not comparable to another phase's without its search
    count beside it: ``fill`` issues roughly two searches per cell while
    ``init`` is capped at ``init_max_steps``, so ``fill`` surfaces far more
    documents and would win on recall alone regardless of how well it searched.
    ``num_searches`` and ``searches_per_query`` are reported alongside every
    block for exactly that reason.

    Args:
        per_query_phases: ``{qid: {phase: {"surfaced": [[id]], "seen": [[id]]}}}``.
        qrels:            The run's qrels, unchanged.
        k_values:         Cut-offs, the run's own.
        level:            ``"surfaced"`` (all retrieved) or ``"seen"`` (shown to the LLM).
        phase_order:      Which buckets to score and in what order.  The
                          disjoint phases by default; :data:`CUMULATIVE_PHASES`
                          for the prefix view built by :func:`cumulative_phases`.

    Returns:
        ``{phase: {NDCG/MAP/Recall/... , num_queries, num_searches,
        searches_per_query}}``, empty when nothing was scorable.
    """
    if not per_query_phases or not qrels:
        return {}

    by_phase: Dict[str, Dict[str, List[List[str]]]] = {}
    searches: Dict[str, int] = {}
    for query_id, phases in per_query_phases.items():
        for phase, bucket in (phases or {}).items():
            iterations = bucket.get(level) or []
            if not any(iterations):
                continue
            by_phase.setdefault(phase, {})[query_id] = iterations
            searches[phase] = searches.get(phase, 0) + len(iterations)

    out: Dict[str, Any] = {}
    for phase in phase_order:
        per_query = by_phase.get(phase)
        if not per_query:
            continue
        ranking = _ranking_for_phase(per_query, fusion_method, rrf_k, interleaving_window)
        if not ranking.results:
            continue
        metrics = evaluate_results(
            results=ranking, qrels=qrels, k_values=list(k_values),
        )
        num_queries = len(ranking.get_unique_queries())
        metrics["num_queries"] = num_queries
        metrics["num_searches"] = searches.get(phase, 0)
        metrics["searches_per_query"] = (
            searches.get(phase, 0) / num_queries if num_queries else 0.0
        )
        out[phase] = metrics
    return out


def score_cell_grounding(
    per_query_cells: Dict[str, List[Dict[str, Any]]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """How often a filled cell cites a document the qrels call relevant.

    The most direct read of whether ``fill_cell`` retrieved the right document,
    and the only one that is per cell rather than per phase: a phase-level
    recall says the run surfaced a relevant document somewhere, not that the
    cell which needed it actually cited it.

    ``not_found`` cells are counted separately rather than pooled in -- a cell
    the corpus cannot answer has no document to cite, and scoring it as
    ungrounded would punish an honest abstention.
    """
    if not per_query_cells or not qrels:
        return {}

    filled = grounded = not_found = conflicting = uncited = 0
    for query_id, cells in per_query_cells.items():
        relevant = {d for d, s in (qrels.get(query_id) or {}).items() if s > 0}
        for cell in cells:
            status = cell.get("status")
            if status == "not_found":
                not_found += 1
                continue
            if status == "conflicting":
                conflicting += 1
            filled += 1
            doc_ids = cell.get("doc_ids") or []
            if not doc_ids:
                uncited += 1
            elif relevant.intersection(doc_ids):
                grounded += 1

    if not filled:
        return {}
    return {
        "num_cells_scored": filled,
        "grounded_rate": grounded / filled,
        "uncited_rate": uncited / filled,
        "num_not_found": not_found,
        "num_conflicting": conflicting,
    }
