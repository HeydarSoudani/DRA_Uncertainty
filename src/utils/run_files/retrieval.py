"""The three document levels of a run and their per-query TREC files.

* surfaced: every doc the retriever returned in each step (top-k surfaced,
  before the agent down-selects what it reads), ``retrieval/surfaced/{qid}.trec``;
* seen: the subset passed to the agent/LLM (seen-top-k),
  ``retrieval/seen/{qid}.trec``; the only level the evaluation scores
  (``evaluation.retrieval``);
* cited: the docs cited by the final report, ``retrieval/cited/{qid}.trec``.

Each level gives one ranked doc list per retrieval step of a result
(``*_iterations``), written by :func:`save_retrieval` at the end of a
query.  The TREC files are never read back: a result read from the saved
files (``utils.run_files.load_result_from_saved_files``) has its seen doc
ids in the trajectory steps, the only level the evaluation needs.

TREC columns: ``qid Q0 doc_id rank score run_tag``, with the step number as
run tag (``iter_1``, ``iter_2``, ...).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from utils.documents import doc_id as _doc_id

from .trajectory_logger import seen_doc_ids

#: The per-level TREC directories under the run directory.
SURFACED_DIR = "retrieval/surfaced"
SEEN_DIR = "retrieval/seen"
CITED_DIR = "retrieval/cited"

_NON_SEARCH_ACTIONS = frozenset({
    "write", "init-plan", "extend-plan", "nop",
    "analyst-init_plan", "analyst-extend_plan",
})


# ── Surfaced ─────────────────────────────────────────────────────────────

def _rank_score(doc: Dict[str, Any], fallback_rank: int) -> float:
    """The doc's retrieval score (``rank_score``, ``rerank_score`` or ``score``),
    else ``1 / rank``."""
    for key in ("rank_score", "rerank_score", "score"):
        val = doc.get(key)
        if val is not None:
            return val
    return 1.0 / fallback_rank


def surfaced_iterations(result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """One surfaced doc list per retrieval step of *result*: a step's
    ``all_docs`` (flat list of dicts, or list-of-lists) or ``docs`` (a live
    result: the trajectory file keeps no surfaced docs)."""
    iterations = []
    for step in result.get("trajectory", []):
        all_docs = step.get("all_docs")
        if all_docs:
            if isinstance(all_docs[0], dict):
                turn_docs = all_docs
            else:
                turn_docs = [doc for retrieve_docs in all_docs for doc in retrieve_docs]
            if turn_docs:
                iterations.append(turn_docs)
        elif step.get("docs"):
            iterations.append(step["docs"])
    return iterations


# ── Seen ─────────────────────────────────────────────────────────────────

def seen_iterations(result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """One ``[{"doc_id"}]`` list per search step of *result*, the *seen*
    documents only (a live result, or one read back from the trajectory).

    A step's seen ids are read by ``trajectory_logger.seen_doc_ids``, the
    reader the trajectory file is written with.  Write, plan and other
    non-search steps are skipped even when they carry ``component_doc_ids``
    (CPM-Report write steps store cited doc ids).
    """
    iterations: List[List[Dict[str, Any]]] = []
    for step in result.get("trajectory", []):
        action = step.get("action_type") or step.get("action") or step.get("_cpm_state") or ""
        if action in _NON_SEARCH_ACTIONS:
            continue
        doc_ids = seen_doc_ids(step)
        if doc_ids:
            iterations.append([{"doc_id": did} for did in doc_ids])
    return iterations


# ── Cited ────────────────────────────────────────────────────────────────

def cited_docs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The docs cited by a result's report, in citation order, without repeats.

    Each doc is ``{"doc_id", "iter"}`` (the step that surfaced it, 1 when
    unknown).  Sources, in priority order:

    1. ``cited_docs_ranked_list``: the cited list built by the pipeline
       (``utils.citations.ranked_cited_docs``).
    2. ``memory_bank``: doc-id keys in insertion order (WebWeaver).
    3. ``[N]`` markers parsed from ``generation`` / ``final_report`` and
       mapped through ``citation_to_doc_id``, so any report agent works.
    """
    cited_ranked = result.get("cited_docs_ranked_list")
    if cited_ranked:
        docs = [{"doc_id": _doc_id(d), "iter": d.get("iter", 1) if isinstance(d, dict) else 1}
                for d in cited_ranked]
    elif result.get("memory_bank"):
        docs = [{"doc_id": did, "iter": 1} for did in result["memory_bank"]]
    else:
        report = result.get("generation") or result.get("final_report") or ""
        citation_to_doc_id = result.get("citation_to_doc_id") or {}
        docs = []
        if report and citation_to_doc_id:
            for cit in re.findall(r"\[(\d+)\]", report):  # in order; repeats dropped below
                did = citation_to_doc_id.get(int(cit)) or citation_to_doc_id.get(cit)
                docs.append({"doc_id": did, "iter": 1})

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for doc in docs:
        if doc["doc_id"] and doc["doc_id"] not in seen:
            seen.add(doc["doc_id"])
            out.append(doc)
    return out


# ── Files ────────────────────────────────────────────────────────────────

def doc_ids(iterations: List[List[Any]]) -> List[str]:
    """The distinct doc ids of *iterations*, in first-seen order."""
    return list(dict.fromkeys(did for docs in iterations for doc in docs if (did := _doc_id(doc))))


def _write_iterations(path: Path, query_id: str, iterations: List[List[Any]], scored: bool) -> None:
    """One TREC line per doc of each step (``iter_N``); *scored*: the doc's
    retrieval score, else ``1 / rank``."""
    lines: List[str] = []
    for iter_idx, iteration_docs in enumerate(iterations, 1):
        for rank, doc in enumerate(iteration_docs, 1):
            did = _doc_id(doc)
            if not did:
                continue
            score = _rank_score(doc, rank) if scored else 1.0 / rank
            lines.append(f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_idx}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_cited(path: Path, query_id: str, docs: List[Dict[str, Any]]) -> None:
    """The cited docs as one ranked list, each run tag the doc's originating
    step; nothing is written when the report cites nothing, and the file of
    an earlier attempt of the query is removed."""
    if not docs:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{query_id} Q0 {doc['doc_id']} {rank} {1.0 / rank:.6f} iter_{doc['iter']}"
                              for rank, doc in enumerate(docs, 1)) + "\n")


def save_retrieval(run_dir: Union[str, Path], query_id: str, result: Dict[str, Any]) -> None:
    """Write the surfaced, seen and cited TREC files of one query."""
    run_dir = Path(run_dir)
    _write_iterations(run_dir / SURFACED_DIR / f"{query_id}.trec", query_id, surfaced_iterations(result), True)
    _write_iterations(run_dir / SEEN_DIR / f"{query_id}.trec", query_id, seen_iterations(result), False)
    _write_cited(run_dir / CITED_DIR / f"{query_id}.trec", query_id, cited_docs(result))
