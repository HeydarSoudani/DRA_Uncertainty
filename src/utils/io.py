"""Generic file helpers: JSONL reading and writing, atomic writes, a file lock.

The files of a run directory are written and read by ``utils.run_files``.
"""

import fcntl
import json as _json
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Dict, Iterable, Iterator, List, Union

_log = logging.getLogger(__name__)

try:
    from orjson import loads as json_loads
except ImportError:
    json_loads = _json.loads


@contextmanager
def atomic_open(path: Union[str, Path], **open_kwargs) -> Iterator[IO[str]]:
    """A text file to write *path* through: a temp file next to it, moved
    onto *path* (``os.replace``) when the block ends without an error, so a
    crash never leaves a truncated file; on an error *path* is left as it
    was.  The temp name is unique per call, so two writers of one path never
    share it.  *open_kwargs* go to ``open`` (default ``encoding="utf-8"``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    open_kwargs.setdefault("encoding", "utf-8")
    try:
        with open(tmp, "w", **open_kwargs) as f:
            yield f
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write *text* to *path* atomically (:func:`atomic_open`)."""
    with atomic_open(path) as f:
        f.write(text)


@contextmanager
def file_lock(path: Union[str, Path]) -> Iterator[None]:
    """Hold an exclusive lock on *path* (through the lock file
    ``.{name}.lock`` next to it) for the block: a read-modify-write of a file
    several processes update (``run_config.json``).  Advisory: only callers
    that take the lock are kept out."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """The rows of a JSONL file; blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json_loads(line)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """All rows of a JSONL file (:func:`iter_jsonl`)."""
    return list(iter_jsonl(path))


def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    """Write *rows* as JSONL, atomically (:func:`atomic_open`)."""
    with atomic_open(path) as f:
        for row in rows:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")


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
