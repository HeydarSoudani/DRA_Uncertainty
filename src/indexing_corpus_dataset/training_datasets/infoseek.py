"""The infoseek build: InfoSeek-Hard questions over the corpus of the
original release, as the InfoSeek code searches it (VectorSpaceLab/InfoSeek,
``research/InfoSeek/rl/InfoSeek/Retrieve``: ``wiki-25-512.jsonl``,
Wikipedia 2025 in chunks of 512 tokens, the page title on the first line of
each chunk).  One call (``build``):

    questions  InfoSeek-Hard-18K (``sources.load_infoseek``, pinned), a
               repeated question dropped (its first row kept)
    corpus     ``wiki-25-512.jsonl`` of ``Lk123/wiki-25-512`` (pinned),
               downloaded once into ``DATA_ROOT/infoseek/corpus``
               (``layout.default_corpus_path``) and used unchanged: a doc id
               is the chunk's ``id`` (its line, the row of the released
               index), as a string
    gold       the first chunk of each gold page: a research tree node's
               ``href`` (``sources.wiki_gold``) matched exactly against the
               first line of every chunk (``title_key`` on both sides), the
               page's chunk with the lowest id; the titles looked up are kept
               with all their chunks in ``corpus_titles.jsonl``, so a build
               reads the corpus only for new titles.  InfoSeek writes every
               claim from the first 1024 tokens of a page and picks links
               from its front part: in the corpus each claimed link is in the
               page's first chunk, the later chunks hold no evidence
    rows       a question is kept when every gold page is in the corpus,
               else dropped (``title_not_in_corpus``, with its titles); its
               row (``annotations.row``): each hop's documents are the first
               chunks of its page and of the pages linking to it
               (``annotations.infoseek_skeleton``); the qrels (binary) the
               first chunk of every gold page
    texts      the criterion texts (``criterion_texts.fill_texts``) and the
               split (``splits.split_rows``: sft / rl / val with the shares)

The build is incremental, like ``dra_inference.py``: with a *limit* it
processes the next N questions not processed yet (kept or dropped), in file
order (0: none, the dataset files are rewritten); without one, every
question.  A question keeps its split as the dataset grows.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from indexing_corpus_dataset.layout import default_corpus_path
from utils.io_utils import atomic_write_text, drop_partial_line, iter_jsonl, read_jsonl

from .gold import annotations
from .config import BuildConfig, DatasetPaths
from .gold.criterion_texts import check_writer, fill_texts, require_api_key
from .sources import load_infoseek
from .splits import check_shares, split_rows
from .stats import (drop_stats, format_annotation_table, question_stats, read_stats, write_annotation_stats,
                    write_dropped, write_stats)

log = logging.getLogger(__name__)

# The corpus of the original release (Hugging Face dataset repo), pinned.
CORPUS = ("Lk123/wiki-25-512", "wiki-25-512.jsonl", "fc1e312568b14385c04f41bc09157d8fa4c20658")

# The stage of the dropped questions in ``dropped.jsonl``.
GOLD_STAGE = "gold"

# The id and the title (the first line of ``contents``, up to its escaped
# newline) at the start of a corpus line, read without parsing the line.
_LINE = re.compile(rb'^\{\s*"id"\s*:\s*"?(-?\d+|(?:[^"\\]|\\.)*)"?\s*,\s*"contents"\s*:\s*"((?:[^"\\]|\\[^n])*)\\n')


def title_key(title: str) -> str:
    """A page title as both sides are matched: Unicode NFC, spaces for
    underscores, whitespace collapsed, first letter upper case (as
    MediaWiki displays it, ``sources.normalize_title``)."""
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


# ── Corpus ──────────────────────────────────────────────────────────────────

def ensure_corpus() -> None:
    """Download the corpus (``CORPUS``) into its place under ``DATA_ROOT``
    when it is not there."""
    path = default_corpus_path("infoseek")
    if path.exists():
        return
    repo, filename, revision = CORPUS
    print(f"[corpus] downloading {repo}/{filename} at {revision[:12]} (18.5 GB) -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision,
                          local_dir=str(path.parent))
    if Path(got).resolve() != path.resolve():
        raise SystemExit(f"downloaded to {got}, expected {path}")


def _read_titles(paths: DatasetPaths) -> Dict[str, List[str]]:
    """The titles looked up by earlier builds (title key -> chunk ids)."""
    if not paths.titles.exists():
        return {}
    drop_partial_line(paths.titles)
    return {row["title"]: row["ids"] for row in iter_jsonl(paths.titles)}


def _check_corpus(paths: DatasetPaths) -> None:
    """Refuse a dataset built from another corpus revision: its titles and
    qrels would mix chunk ids of two corpora."""
    recorded = read_stats(paths.stats).get("corpus", {}).get("revision")
    if recorded is not None and recorded != CORPUS[2]:
        raise SystemExit(f"{paths.root} was built from {CORPUS[0]} at {recorded}, the code pins {CORPUS[2]}; "
                         f"remove the built files in {paths.root} (keep criteria_bank/, the silver criteria) "
                         "and build the dataset again")


def lookup_titles(paths: DatasetPaths, wanted: Set[str], known: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """The chunk ids of every title key of *wanted*: *known* ones from
    earlier builds, the others from one pass over the corpus, appended to
    ``corpus_titles.jsonl`` (``ids`` empty: not in the corpus)."""
    new = wanted - set(known)
    if not new:
        return known
    corpus = default_corpus_path("infoseek")
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
            out.write(json.dumps({"title": title, "ids": found[title]}, ensure_ascii=False) + "\n")
    return {**known, **found}


# ── Build ───────────────────────────────────────────────────────────────────

def _questions() -> Tuple[List[dict], List[dict]]:
    """The InfoSeek-Hard questions, a repeated one dropped."""
    records, dropped, seen = [], [], {}
    for record in load_infoseek():
        key = re.sub(r"\W+", " ", record["question"].lower()).strip()
        if key in seen:
            dropped.append({"id": record["id"], "source": record["source"], "reasons": ["duplicate"],
                            "duplicate_of": seen[key]})
            continue
        seen[key] = record["id"]
        records.append(record)
    return records, dropped


def _gold_titles(record: dict) -> Dict[str, str]:
    """Gold page key -> title key of a question's gold pages."""
    return {g["key"]: title_key(g["title"]) for g in record["gold"]}


def _processed(paths: DatasetPaths) -> Set[str]:
    """The questions processed by earlier builds: kept (the queries file)
    or dropped (``dropped.jsonl``)."""
    done = {str(r["id"]) for r in iter_jsonl(paths.queries)} if paths.queries.exists() else set()
    if paths.dropped.exists():
        done |= {str(r["id"]) for r in read_jsonl(paths.dropped)}
    return done


def build(cfg: BuildConfig, paths: DatasetPaths, limit: Optional[int] = None) -> None:
    check_writer(cfg.gold_criteria, paths)
    check_shares(paths, cfg.split)
    _check_corpus(paths)
    records, duplicates = _questions()
    done = _processed(paths)
    remaining = [r for r in records if r["id"] not in done]
    todo = remaining if limit is None else remaining[:limit]
    print(f"[questions] {len(records)} questions ({len(duplicates)} repeated ones dropped): "
          f"{len(records) - len(remaining)} already processed, {len(todo)} to process"
          + (f" (--limit {limit})" if limit is not None else ""))
    if todo:
        require_api_key(cfg.gold_criteria, "the criterion texts of the new questions")
    ensure_corpus()

    done |= {r["id"] for r in todo}
    processed = [r for r in records if r["id"] in done]
    titles = lookup_titles(paths, {t for r in processed for t in _gold_titles(r).values()}, _read_titles(paths))

    kept, dropped = [], []
    for record in processed:
        missing = sorted({t for t in _gold_titles(record).values() if not titles.get(t)})
        if not record["gold"] or missing:
            reasons = [{"status": "title_not_in_corpus", "title": t} for t in missing] or [{"status": "no_gold"}]
            dropped.append({"id": record["id"], "source": record["source"], "reasons": reasons})
        else:
            kept.append(record)
    docs_of = {key: [min(titles[t], key=int)] for r in kept for key, t in _gold_titles(r).items()}
    rows = [annotations.row(r, docs_of) for r in kept]
    annotations.check(rows)

    # The texts first: a stopped build leaves the files of the last full one.
    rows = split_rows(paths, cfg.split, fill_texts(cfg.gold_criteria, paths, rows))
    annotations.write(paths.queries, rows)
    atomic_write_text(paths.qrels, "".join(f"{row['id']} 0 {doc_id} 1\n" for row in rows for doc_id in row["doc_ids"]))
    write_dropped(paths.dropped, GOLD_STAGE, dropped)
    write_dropped(paths.dropped, "questions", duplicates)

    stats = write_annotation_stats(paths, cfg.gold_criteria, rows)
    looked_up = {t for r in processed for t in _gold_titles(r).values()}
    write_stats(paths.stats, "questions", {"available": len(records), "processed": len(processed),
                                           "remaining": len(records) - len(processed), "kept": question_stats(kept),
                                           "dropped": drop_stats(dropped + duplicates)})
    write_stats(paths.stats, "corpus", {"repo": CORPUS[0], "file": CORPUS[1], "revision": CORPUS[2],
                                        "path": str(default_corpus_path("infoseek")),
                                        "titles": {"looked_up": len(looked_up),
                                                   "found": sum(bool(titles.get(t)) for t in looked_up)},
                                        "qrels": sum(len(r["doc_ids"]) for r in rows)})
    print(f"[infoseek] {len(kept)} kept, {len(dropped)} dropped of {len(processed)} processed questions "
          f"({len(records) - len(processed)} remaining); {sum(len(r['doc_ids']) for r in rows)} qrels -> {paths.root}")
    print(f"[infoseek] gold annotation\n\n{format_annotation_table(stats)}\n")

