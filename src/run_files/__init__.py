"""The files of a run directory: every writer and reader.

The generate stage writes, per query (:func:`save_query_outputs`):

    retrieval/{surfaced,seen,cited}/{qid}.trec   the three document levels (``retrieval``)
    generation/{qid}.md                          the final generation (``generation``)
    trajectory/{qid}.jsonl                       the steps, then the meta line (``trajectory``);
                                                 written last: it marks the query as finished

The belief stage writes ``belief/{qid}.jsonl`` (``belief``); both write
``run_config.json`` (``run_config``).  The evaluation only reads these files
(:func:`load_result_from_saved_files`, ``belief.load_belief_files``) and
writes its own under ``eval/`` (``evaluation.metrics_file``).
"""

from pathlib import Path
from typing import Any, Dict, Union

from utils.config import AGENT_META_KEYS

from .belief import (BELIEF_DIR, SCHEMA_VERSION, BeliefFile, believed_queries, clean_record, load_belief_files,
                     read_belief_file, save_belief)
from .generation import GENERATION_DIR, read_generation, save_generation
from .retrieval import (CITED_DIR, SEEN_DIR, SURFACED_DIR, cited_docs, doc_ids, read_cited_ids, read_trec_iterations,
                        save_retrieval, seen_iterations, surfaced_iterations)
from .run_config import read_run_config
from .trajectory import TRAJECTORY_DIR, is_finished, read_trajectory, save_trajectory, scan_queries

#: Per-query output directories of the generate stage (``belief/`` is the
#: belief stage's, created by its writer).
QUERY_OUTPUT_DIRS = (SURFACED_DIR, SEEN_DIR, CITED_DIR, GENERATION_DIR, TRAJECTORY_DIR)


def save_query_outputs(run_dir: Union[str, Path], query_id: str, question: str, result: Dict[str, Any]) -> None:
    """Write every per-query file of *result* under *run_dir*: the TREC
    files and the generation, then the trajectory, whose meta line marks the
    query as finished (:func:`scan_queries`), so it is written last."""
    save_retrieval(run_dir, query_id, result)
    save_generation(run_dir, query_id, result)
    save_trajectory(run_dir, query_id, question, result)


def load_result_from_saved_files(run_dir: Union[str, Path], query_id: str) -> Dict[str, Any]:
    """Rebuild a result dict from the saved per-query files.

    The evaluation reads every run back this way, whatever the stage
    (``--stage``), so every stage scores the same result.  Sources:

    * ``trajectory/{qid}.jsonl``: the steps (seen doc ids, no doc text) and the
      meta line (counts, :data:`utils.config.AGENT_META_KEYS`);
    * ``generation/{qid}.md``: the final generation;
    * the cited, seen and surfaced TRECs: the cited list and the per-step
      seen and surfaced lists (the trajectory keeps only the seen doc ids).

    ``{}`` when the query is not finished (no meta line, :func:`is_finished`).
    """
    meta, trajectory = read_trajectory(run_dir, query_id)
    if not meta:
        return {}
    result: Dict[str, Any] = {
        "trajectory":     trajectory,
        "num_steps":      meta.get("num_steps", 0),
        "num_searches":   meta.get("num_searches", 0),
        "num_iterations": meta.get("num_iterations"),
    }
    for key in AGENT_META_KEYS:
        if meta.get(key):
            result[key] = meta[key]
    result["generation"] = read_generation(run_dir, query_id)

    run_dir = Path(run_dir)
    cited = read_cited_ids(run_dir / CITED_DIR / f"{query_id}.trec")
    if cited:
        result["cited_docs_ranked_list"] = cited
    # Seen lists from the seen TREC only when the trajectory has none.
    if not any(step.get("seen_docs") for step in trajectory):
        seen = [[{"doc_id": d["doc_id"]} for d in docs if d.get("doc_id")]
                for docs in read_trec_iterations(run_dir / SEEN_DIR / f"{query_id}.trec")]
        if seen:
            result["seen_docs_iterations"] = seen
    surfaced = [docs for docs in read_trec_iterations(run_dir / SURFACED_DIR / f"{query_id}.trec") if docs]
    if surfaced:
        result["surfaced_docs_iterations"] = surfaced
    return result


__all__ = [
    "QUERY_OUTPUT_DIRS", "save_query_outputs", "load_result_from_saved_files",
    "BELIEF_DIR", "SCHEMA_VERSION", "BeliefFile", "believed_queries", "clean_record", "load_belief_files",
    "read_belief_file", "save_belief",
    "GENERATION_DIR", "read_generation", "save_generation",
    "CITED_DIR", "SEEN_DIR", "SURFACED_DIR", "cited_docs", "doc_ids", "read_cited_ids", "read_trec_iterations",
    "save_retrieval", "seen_iterations", "surfaced_iterations",
    "read_run_config",
    "TRAJECTORY_DIR", "is_finished", "read_trajectory", "save_trajectory", "scan_queries",
]
