"""Per-query trajectory file ``trajectory/{query_id}.jsonl``.

:class:`utils.trajectory_logger.TrajectoryLogger` streams the step lines
during the run; :func:`save_trajectory` rewrites the file from the final
result at the end of the query, with the meta line last.  Both build the
lines with the step-shaping helpers of ``utils.trajectory_logger``, which
keeps the two identical in schema.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from utils.trajectory_logger import build_meta_line, dump_line, iter_label, step_to_line

try:
    from orjson import loads as _json_loads
except ImportError:
    from json import loads as _json_loads

TRAJECTORY_DIR = "trajectory"


def save_trajectory(run_dir: Union[str, Path], query_id: str, question: str, result: Dict[str, Any]) -> None:
    """Write ``{run_dir}/trajectory/{query_id}.jsonl``: one line per step, then a
    ``{"record": "meta", ...}`` line (step counts, agent-specific data).

    Search steps carry ``iter`` / ``action_type`` / ``think`` /
    ``search_query`` / ``seen_docs`` (``action_type`` and ``think`` only on
    the first subquery of an iteration); terminal steps carry ``iter`` /
    ``action_type`` / ``generation``.

    The meta line marks the query as finished (:func:`scan_queries`), so this is the last file of
    the query to be written, and it is written atomically.

    Not stored here, since it lives in its own file: the final generation
    (``generation/{qid}.md``), the surfaced ranking
    (``retrieval/surfaced/{qid}.trec``) and the cited docs
    (``retrieval/cited/{qid}.trec``).
    """
    out_dir = Path(run_dir) / TRAJECTORY_DIR
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


def _last_line(path: Path) -> bytes:
    """The last non-empty line of *path*, read from the end of the file."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        size = 4096
        while True:
            start = max(0, end - size)
            f.seek(start)
            chunk = f.read(end - start).rstrip(b"\n")
            cut = chunk.rfind(b"\n")
            if cut >= 0 or start == 0:
                return chunk[cut + 1:]
            size *= 2


def is_finished(trajectory_file: Union[str, Path]) -> bool:
    """The query's trajectory file ends with its meta line, which
    :func:`save_trajectory` writes after every other file of the query."""
    try:
        return _last_line(Path(trajectory_file)).startswith(b'{"record":"meta"')
    except OSError:
        return False


def scan_queries(run_dir: Union[str, Path]) -> Tuple[set, set]:
    """``(finished, unfinished)`` query ids of the run's generate stage.

    A query is finished when ``trajectory/{qid}.jsonl`` ends with its meta
    line; a trajectory file without it is a query whose run stopped, and it
    is run again on resume.
    """
    traj_dir = Path(run_dir) / TRAJECTORY_DIR
    if not traj_dir.is_dir():
        return set(), set()
    finished, unfinished = set(), set()
    for path in traj_dir.glob("*.jsonl"):
        (finished if is_finished(path) else unfinished).add(path.stem)
    return finished, unfinished


def read_trajectory(run_dir: Union[str, Path], query_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``(meta, steps)`` of a saved trajectory file; ``({}, [])`` when it is
    missing (a file without its meta line gives an empty meta)."""
    try:
        content = (Path(run_dir) / TRAJECTORY_DIR / f"{query_id}.jsonl").read_text(encoding="utf-8")
    except OSError:
        return {}, []
    meta: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _json_loads(line)
        if obj.get("record") == "meta":
            meta = obj
        else:
            steps.append(obj)
    return meta, steps
