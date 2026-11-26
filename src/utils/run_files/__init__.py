"""The files of a run directory: every writer and reader.

The generate stage writes, per query (:func:`save_query_outputs`):

    retrieval/{surfaced,seen,cited}/{qid}.trec   the three document levels (``retrieval``)
    generation/{qid}.md                          the final generation (``generation``)
    trajectory/{qid}.jsonl                       the steps, then the meta line (``trajectory``);
                                                 written last: it marks the query as finished

The belief stage reads a query's saved run back
(``belief_model_component.read_saved_query``) and writes
``belief/{model}/{qid}.jsonl``, one folder per belief model (``belief``);
both write ``run_config.json`` (``pipeline.run_config``; read by
``run_config``).  The evaluation only reads these files
(:func:`load_result_from_saved_files`, ``belief.load_belief_files``) and
writes its own under ``eval/`` (``evaluation.shared.files``).
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from .belief import (SCHEMA_VERSION, belief_dir, belief_models, belief_schemas, believed_queries, load_belief_files,
                     save_belief)
from .generation import GENERATION_DIR, read_generation, save_generation
from .retrieval import CITED_DIR, SEEN_DIR, SURFACED_DIR, save_retrieval
from .trajectory import TRAJECTORY_DIR, read_trajectory, save_trajectory, scan_queries
from .trajectory_logger import AGENT_META_KEYS

#: Per-query output directories of the generate stage (``belief/`` is the
#: belief stage's, created by its writer).
QUERY_OUTPUT_DIRS = (SURFACED_DIR, SEEN_DIR, CITED_DIR, GENERATION_DIR, TRAJECTORY_DIR)


def setup_output_dirs(
    run_dir: Union[str, Path],
    subdirs: List[str],
) -> Dict[str, str]:
    """Create output sub-directories under *run_dir* and return their paths.

    Args:
        run_dir: Root run directory.
        subdirs: List of sub-directory names,
                 e.g. ``["retrieval", "generation", "trajectory"]``.

    Returns:
        Dict mapping each sub-directory name to its full path string.
    """
    rd = Path(str(run_dir).rstrip("/"))
    rd.mkdir(parents=True, exist_ok=True)
    result: Dict[str, str] = {}
    for name in subdirs:
        d = rd / name
        d.mkdir(parents=True, exist_ok=True)
        result[name] = str(d)
    return result


def save_query_outputs(run_dir: Union[str, Path], query_id: str, question: str, result: Dict[str, Any]) -> None:
    """Write every per-query file of *result* under *run_dir*: the TREC
    files and the generation, then the trajectory, whose meta line marks the
    query as finished (:func:`scan_queries`), so it is written last."""
    save_retrieval(run_dir, query_id, result)
    save_generation(run_dir, query_id, result)
    save_trajectory(run_dir, query_id, question, result)


def load_result_from_saved_files(run_dir: Union[str, Path], query_id: str) -> Dict[str, Any]:
    """Rebuild a result dict from the saved per-query files: what the
    evaluation scores.

    The evaluation reads every run back this way, whatever the stage
    (``--stage``), so every stage scores the same result.  Sources:

    * ``trajectory/{qid}.jsonl``: the steps (with their seen doc ids) and the
      meta line (counts, :data:`AGENT_META_KEYS`);
    * ``generation/{qid}.md``: the final generation.

    The TREC files are not read: the surfaced and cited levels are saved
    only, and the seen level is in the trajectory.  ``{}`` when the query is
    not finished (no meta line, :func:`is_finished`).
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
    return result


#: What the other packages import from here; the rest is read from the
#: modules (``belief``, ``retrieval``, ``run_config``, ``trajectory``, ...).
__all__ = [
    "QUERY_OUTPUT_DIRS", "setup_output_dirs", "save_query_outputs", "load_result_from_saved_files", "scan_queries",
    "SCHEMA_VERSION", "belief_dir", "belief_models", "belief_schemas", "believed_queries", "load_belief_files",
    "save_belief",
]
