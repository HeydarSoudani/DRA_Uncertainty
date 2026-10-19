"""The corpus of deep_qa, written by step 2 (after new gold pages) and step 3
(after new hard negatives) from the same files, so neither drops the
other's documents:

    corpus/corpus.jsonl           every document with its text: the gold
                                  pages of the qrels, then the hard
                                  negatives of those of them step 3 processed
    corpus/ids_gold.txt           the ids of the gold pages, one per line
    corpus/ids_hard_negative.txt  the ids of the hard negatives (a page that
                                  is gold for any query is listed as gold)

The text of a document is its last row in ``pages/text.jsonl``; the hard
negatives are the rows of ``negatives/index.jsonl`` (the memory of step 3).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils.io_utils import atomic_write_text, iter_jsonl

from .config import PoolPaths

log = logging.getLogger(__name__)

_ID_PREFIX = b'{"id": "'


def drop_partial_line(path: Path) -> None:
    """Cut a last line left unfinished by a killed run, so that appended
    rows start on a line of their own (the file is read from its end)."""
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
    log.warning("%s: removed an unfinished last line", path)


def row_id(line: bytes) -> Optional[str]:
    """The id of a ``pages/text.jsonl`` row, read from its prefix as
    ``PageStore.add_text`` writes it, without parsing the text (the full row
    otherwise)."""
    if line.startswith(_ID_PREFIX):
        end = line.find(b'"', len(_ID_PREFIX))
        if end > 0 and b"\\" not in line[len(_ID_PREFIX):end]:
            return line[len(_ID_PREFIX):end].decode("utf-8")
    return json.loads(line)["id"] if line.strip() else None


def read_negatives(paths: PoolPaths) -> Dict[str, dict]:
    """The memory of step 3: gold page id -> its row ``{"gold", "title",
    "negatives": [{"id", "title", "url", "source", "rank", "words"}]}`` (the
    last row of a gold page wins)."""
    if not paths.negatives.exists():
        return {}
    return {row["gold"]: row for row in iter_jsonl(paths.negatives)}


def write_corpus(paths: PoolPaths, gold_urls: Dict[str, str]) -> Tuple[int, int]:
    """The corpus and the two id files from *gold_urls* (gold page id -> url,
    in qrels order) and the hard negatives of those gold pages in the memory
    of step 3, each file replaced atomically; returns the numbers of gold
    pages and hard negatives written."""
    negative_urls: Dict[str, str] = {}
    for gold, row in read_negatives(paths).items():
        if gold in gold_urls:
            for neg in row["negatives"]:
                if neg["id"] not in gold_urls:
                    negative_urls.setdefault(neg["id"], neg["url"])
    urls = {**gold_urls, **negative_urls}

    # Byte offset of the last text row of each document, then those rows only.
    offsets: Dict[str, int] = {}
    if paths.page_text.exists():
        with open(paths.page_text, "rb") as fh:
            offset = 0
            for line in fh:
                doc_id = row_id(line)
                if doc_id in urls:
                    offsets[doc_id] = offset
                offset += len(line)
    missing = len(urls) - len(offsets)
    if missing:
        log.warning("%d documents have no text in %s; left out of the corpus", missing, paths.page_text)
    paths.corpus.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.corpus.with_name(f".{paths.corpus.name}.tmp")
    order = {doc_id: i for i, doc_id in enumerate(urls)}
    with open(tmp, "w", encoding="utf-8", errors="replace") as out:
        if offsets:
            with open(paths.page_text, "rb") as fh:
                for doc_id in sorted(offsets, key=order.__getitem__):
                    fh.seek(offsets[doc_id])
                    item = json.loads(fh.readline())
                    # {id, contents} as the other corpora (layout.py), plus the
                    # page url; a wiki page's title is the first line of its text.
                    out.write(json.dumps({"id": doc_id, "contents": item["text"], "url": urls[doc_id]},
                                         ensure_ascii=False) + "\n")
    os.replace(tmp, paths.corpus)
    golds = [d for d in gold_urls if d in offsets]
    negatives = [d for d in negative_urls if d in offsets]
    atomic_write_text(paths.ids_gold, "".join(f"{d}\n" for d in golds))
    atomic_write_text(paths.ids_hard_negative, "".join(f"{d}\n" for d in negatives))
    return len(golds), len(negatives)
