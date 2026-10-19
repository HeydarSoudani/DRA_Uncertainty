"""Per-query trajectory file ``trajectory/{query_id}.jsonl``.

:class:`utils.trajectory_logger.TrajectoryLogger` streams the step lines
during the run; :func:`save_trajectory` rewrites the file from the final
result at the end of the query, with the meta line last.  Both build the
lines with the step-shaping helpers of ``utils.trajectory_logger``, which
keeps the two identical in schema.
"""

import os
from pathlib import Path
from typing import Any, Dict

from utils.trajectory_logger import build_meta_line, dump_line, iter_label, step_to_line


def save_trajectory(query_id: str, question: str, result: Dict[str, Any], output_dir) -> None:
    """Write ``{output_dir}/{query_id}.jsonl``: one line per step, then a
    ``{"record": "meta", ...}`` line (step counts, agent-specific data).

    Search steps carry ``iter`` / ``action_type`` / ``think`` /
    ``search_query`` / ``seen_docs`` (``action_type`` and ``think`` only on
    the first subquery of an iteration); terminal steps carry ``iter`` /
    ``action_type`` / ``generation``.

    The meta line marks the query as finished
    (``utils.io_utils.get_processed_queries``), so this is the last file of
    the query to be written, and it is written atomically.

    Not stored here, since it lives in its own file: the final generation
    (``generation/{qid}.md``), the surfaced ranking
    (``retrieval/surfaced/{qid}.trec``), the belief-model signals
    (``belief/{qid}.jsonl``) and the cited docs
    (``retrieval/cited/{qid}.trec``).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f".{query_id}.jsonl.tmp"
    with open(tmp_path, "w") as f:
        last_iter = 0
        for step in result.get("trajectory", []):
            it = step.get("iteration")
            last_iter = int(it) if it is not None else last_iter + 1
            f.write(dump_line(step_to_line(step, iter_label(step, last_iter))) + "\n")
        f.write(dump_line(build_meta_line(query_id, question, result)) + "\n")
    os.replace(tmp_path, out_dir / f"{query_id}.jsonl")
