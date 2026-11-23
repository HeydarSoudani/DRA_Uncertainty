"""corpus_wiki, the corpus every training dataset but trqa_val searches:
``wiki-25-512.jsonl`` of ``Lk123/wiki-25-512`` (pinned, ``CORPUS``),
InfoSeek's release: Wikipedia 2025 in chunks of 512 tokens, the page title
on the first line of each chunk, the doc id the chunk's line (as a
string).  Downloaded once into ``DATA_ROOT/infoseek/corpus``
(``ensure_corpus``), used unchanged.  The web gold pages of webshaper are
in corpus_web (``web.py``); the user merges the two files for the index.

A gold page of a question is found in corpus_wiki by its title
(``title_key`` on both sides: the page title of the source, the first line
of every chunk); ``lookup_titles`` keeps the chunk ids of each title
looked up in ``corpus_titles.jsonl`` of the dataset's raw folder, so stage
data_prepare reads the corpus only for new titles.
"""

import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from indexing_corpus_dataset.layout import default_corpus_path
from utils.io_utils import drop_partial_line, iter_jsonl, write_jsonl

from ..config import DatasetPaths
from ..pipeline.stats import read_stats

# The corpus of InfoSeek's release (Hugging Face dataset repo), pinned.
CORPUS = ("Lk123/wiki-25-512", "wiki-25-512.jsonl", "fc1e312568b14385c04f41bc09157d8fa4c20658")
# The version of ``title_key`` in the rows of ``corpus_titles.jsonl``: a row
# of another version (a key made another way, its miss maybe a match now)
# is looked up again.  1: no version field (no HTML entities decoded).
TITLE_KEY_VERSION = 2

# The id and the title (the first line of ``contents``, up to its escaped
# newline) at the start of a corpus line, read without parsing the line.
_LINE = re.compile(rb'^\{\s*"id"\s*:\s*"?(-?\d+|(?:[^"\\]|\\.)*)"?\s*,\s*"contents"\s*:\s*"((?:[^"\\]|\\[^n])*)\\n')


def wiki_corpus_path() -> Path:
    """corpus_wiki: ``DATA_ROOT/infoseek/corpus/wiki-25-512.jsonl``."""
    return default_corpus_path("infoseek_hard")


def title_key(title: str) -> str:
    """A page title as both sides are matched: HTML entities decoded (the
    corpus writes "&amp;"), Unicode NFC, spaces for underscores, whitespace
    collapsed, first letter upper case (as MediaWiki displays it,
    ``pages.normalize_title``)."""
    title = html.unescape(title)
    title = re.sub(r"\s+", " ", unicodedata.normalize("NFC", title).replace("_", " ")).strip()
    return title[:1].upper() + title[1:]


def _line_title(line: bytes) -> Optional[Tuple[str, str]]:
    """``(doc id, title key)`` of a corpus line; None for a chunk without a
    title line."""
    m = _LINE.match(line)
    if m:
        doc_id = m.group(1).decode("utf-8")
        title = json.loads(b'"' + m.group(2) + b'"')
    else:
        row = json.loads(line)
        first, sep, _ = str(row.get("contents") or "").partition("\n")
        if not sep:
            return None
        doc_id, title = str(row["id"]), first
    return doc_id, title_key(title)


def ensure_corpus() -> None:
    """Download corpus_wiki (``CORPUS``) into its place under ``DATA_ROOT``
    when it is not there."""
    path = wiki_corpus_path()
    if path.exists():
        return
    repo, filename, revision = CORPUS
    print(f"[corpus] downloading {repo}/{filename} at {revision[:12]} (18.5 GB) -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision,
                          local_dir=str(path.parent))
    if Path(got).resolve() != path.resolve():
        raise SystemExit(f"downloaded to {got}, expected {path}")


def check_corpus(paths: DatasetPaths) -> None:
    """Refuse a dataset built from another corpus revision: its titles and
    qrels would mix chunk ids of two corpora."""
    recorded = read_stats(paths.stats).get("corpus", {}).get("revision")
    if recorded is not None and recorded != CORPUS[2]:
        raise SystemExit(f"{paths.dataset} was built from {CORPUS[0]} at {recorded}, the code pins {CORPUS[2]}; "
                         f"remove {paths.queries}, {paths.qrels} and the build files in {paths.raw} but the "
                         f"source (questions.jsonl, source.json), then run --stage data_prepare and "
                         f"data_relevance again (keep the silver beliefs in belief_bank/)")


def corpus_stats(paths: DatasetPaths, looked_up: Set[str], titles: Dict[str, List[str]]) -> dict:
    """The ``corpus`` section of ``stats.json``: corpus_wiki and its title
    lookups."""
    return {"repo": CORPUS[0], "file": CORPUS[1], "revision": CORPUS[2], "path": str(wiki_corpus_path()),
            "titles": {"looked_up": len(looked_up), "found": sum(bool(titles.get(t)) for t in looked_up)}}


def read_titles(paths: DatasetPaths) -> Dict[str, List[str]]:
    """The titles looked up by earlier builds with this ``title_key``
    (``TITLE_KEY_VERSION``): title key -> chunk ids.  Rows of another
    version are removed from the file."""
    if not paths.titles.exists():
        return {}
    drop_partial_line(paths.titles)
    rows = list(iter_jsonl(paths.titles))
    current = [row for row in rows if row.get("key_version", 1) == TITLE_KEY_VERSION]
    if len(current) < len(rows):
        write_jsonl(paths.titles, current)
        print(f"[gold] {len(rows) - len(current)} titles of another title_key version removed from {paths.titles}")
    return {row["title"]: row["ids"] for row in current}


def lookup_titles(paths: DatasetPaths, wanted: Set[str], known: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """The chunk ids of every title key of *wanted* (in corpus order): *known*
    ones from earlier builds, the others from one pass over corpus_wiki,
    appended to ``corpus_titles.jsonl`` (``ids`` empty: not in the corpus)."""
    new = wanted - set(known)
    if not new:
        return known
    corpus = wiki_corpus_path()
    found: Dict[str, List[str]] = {t: [] for t in new}
    print(f"[gold] {len(new)} new titles: one pass over {corpus}")
    with open(corpus, "rb") as fh, tqdm(total=corpus.stat().st_size, unit="B", unit_scale=True,
                                        desc="[gold] corpus", mininterval=30) as bar:
        for line in fh:
            bar.update(len(line))
            got = _line_title(line)
            if got is not None and got[1] in found:
                found[got[1]].append(got[0])
    paths.titles.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.titles, "a", encoding="utf-8") as out:
        for title in sorted(found):
            out.write(json.dumps({"title": title, "ids": found[title], "key_version": TITLE_KEY_VERSION},
                                 ensure_ascii=False) + "\n")
    return {**known, **found}


def page_chunks(titles: Dict[str, List[str]], key: str) -> List[str]:
    """The chunk ids of the page of title key *key*, in page order (the
    chunks of a page are consecutive lines)."""
    return sorted(titles.get(key) or [], key=int)
