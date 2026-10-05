"""Look up corpus documents by id without loading the corpus.

A corpus JSONL (``{"id", "contents"}`` per line) can be tens of GB, so
:class:`CorpusLookup` keeps a byte-offset index next to it,
``{corpus}.offsets.npz``: the 64-bit hash of every doc id, sorted, with the
byte offset of its line.  The index is built by one pass over the corpus the
first time it is needed (minutes for the largest corpus) and memory-mapped
afterwards, so fetching a few hundred documents is a few hundred seeks.  The
index is rebuilt when the corpus file is newer than it.
"""

import hashlib
import json
import os
import re
from array import array
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm

# The id field at the start of a corpus line, read without parsing the line.
_ID_RE = re.compile(rb'^\{\s*"id"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _id_hash(doc_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest(), "little")


def _line_id(line: bytes) -> str:
    m = _ID_RE.match(line)
    if m:
        return json.loads(b'"' + m.group(1) + b'"')
    return str(json.loads(line)["id"])


def split_contents(contents: str) -> Tuple[str, str]:
    """``(title, text)`` of a corpus ``contents`` field: the first line is
    the title when the field has more than one line."""
    first, sep, rest = contents.strip().partition("\n")
    return (first.strip(), rest.strip()) if sep else ("", first.strip())


class CorpusLookup:
    """Fetch documents of one corpus JSONL by id.

    Args:
        corpus_path: The corpus JSONL file.
    """

    def __init__(self, corpus_path: Path | str) -> None:
        self.corpus_path = Path(corpus_path)
        self.index_path = self.corpus_path.with_name(self.corpus_path.name + ".offsets.npz")
        self._hashes: np.ndarray | None = None
        self._offsets: np.ndarray | None = None

    def _build_index(self) -> None:
        # Typed arrays: tens of millions of Python ints would take gigabytes.
        hashes, offsets = array("Q"), array("Q")
        size = self.corpus_path.stat().st_size
        with open(self.corpus_path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True,
                                                     desc=f"Indexing {self.corpus_path.name}") as bar:
            offset = 0
            for line in f:
                if line.strip():
                    hashes.append(_id_hash(_line_id(line)))
                    offsets.append(offset)
                offset += len(line)
                bar.update(len(line))
        hashes_arr = np.frombuffer(hashes, dtype=np.uint64)
        order = np.argsort(hashes_arr, kind="stable")
        tmp = self.index_path.with_name(self.index_path.name + ".tmp.npz")
        np.savez(tmp, hashes=hashes_arr[order], offsets=np.frombuffer(offsets, dtype=np.uint64)[order])
        os.replace(tmp, self.index_path)

    def _load_index(self) -> None:
        if self._hashes is not None:
            return
        if (not self.index_path.exists()
                or self.index_path.stat().st_mtime < self.corpus_path.stat().st_mtime):
            self._build_index()
        with np.load(self.index_path) as index:
            self._hashes, self._offsets = index["hashes"], index["offsets"]

    def get(self, doc_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        """``{doc_id: {"title", "text"}}`` for the ids found in the corpus."""
        wanted = sorted(set(doc_ids))
        if not wanted:
            return {}
        self._load_index()
        found: Dict[str, Dict[str, str]] = {}
        with open(self.corpus_path, "rb") as f:
            for doc_id in wanted:
                h = np.uint64(_id_hash(doc_id))
                i = int(np.searchsorted(self._hashes, h))
                # Equal hashes sit side by side; the line's own id settles a collision.
                while i < len(self._hashes) and self._hashes[i] == h:
                    f.seek(int(self._offsets[i]))
                    record = json.loads(f.readline())
                    if str(record.get("id")) == doc_id:
                        title, text = split_contents(record.get("contents", ""))
                        found[doc_id] = {"title": title, "text": text}
                        break
                    i += 1
        return found
