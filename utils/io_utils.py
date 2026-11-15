"""Generic file helpers: JSONL reading and writing, atomic writes.

The files of a run directory are written and read by ``utils.run_files``.
"""

import json as _json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

_log = logging.getLogger(__name__)

try:
    from orjson import loads as _json_loads
except ImportError:
    _json_loads = _json.loads


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write *text* to *path* through a temp file and ``os.replace``, so a
    crash never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """The rows of a JSONL file; blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield _json_loads(line)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """All rows of a JSONL file (:func:`iter_jsonl`)."""
    return list(iter_jsonl(path))


def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    """Write *rows* as JSONL, atomically (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def drop_partial_line(path: Union[str, Path]) -> None:
    """Cut a last line left unfinished by a killed run, so that appended
    rows start on a line of their own (the file is read from its end)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as fh:
        end = fh.seek(0, 2)
        fh.seek(end - 1)
        if fh.read(1) == b"\n":
            return
        pos, size = end, 1 << 16
        while pos > 0:
            start = max(0, pos - size)
            fh.seek(start)
            cut = fh.read(pos - start).rfind(b"\n")
            if cut >= 0:
                fh.truncate(start + cut + 1)
                break
            pos = start
        else:
            fh.truncate(0)
    _log.warning("%s: removed an unfinished last line", path)


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
