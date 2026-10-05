"""Per-query trajectory file ``trajectory/{query_id}.jsonl``.

:class:`utils.trajectory_logger.TrajectoryLogger` streams the same file step
by step during the run; :func:`save_trajectory` rewrites it from the final
result at the end of the query, so an interrupted stream is repaired.  Both
build the lines with the step-shaping helpers of ``utils.trajectory_logger``,
which keeps the two files identical in schema.
"""

from pathlib import Path
from typing import Any, Dict

from utils.trajectory_logger import build_meta_line, dump_line, iter_label, step_to_line


def save_trajectory(query_id: str, question: str, result: Dict[str, Any], output_dir) -> None:
    """Write ``{output_dir}/{query_id}.jsonl``: one line per step, then a
    ``{"record": "meta", ...}`` line with what a resumed run needs to rebuild
    the result (generation, step counts, agent-specific resume data).

    Search steps carry ``iter`` / ``action_type`` / ``think`` /
    ``search_query`` / ``seen_docs`` (``action_type`` and ``think`` only on
    the first subquery of an iteration); terminal steps carry ``iter`` /
    ``action_type`` / ``generation``.  Readers dispatch on ``record`` per
    line, so the meta line may come last.

    Not stored here, since it lives in its own file: the surfaced ranking
    (``retrieval/surfaced/{qid}.trec``), the uncertainty signals
    (``uncertainty/{qid}.jsonl``) and the cited docs
    (``retrieval/cited/{qid}.trec``).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"{query_id}.jsonl", "w") as f:
        last_iter = 0
        for step in result.get("trajectory", []):
            it = step.get("iteration")
            last_iter = int(it) if it is not None else last_iter + 1
            f.write(dump_line(step_to_line(step, iter_label(step, last_iter))) + "\n")
        f.write(dump_line(build_meta_line(query_id, question, result)) + "\n")
