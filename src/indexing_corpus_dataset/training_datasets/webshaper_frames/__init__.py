"""The webshaper_frames build: WebShaper and FRAMES questions with their gold
pages fetched from the source, one call (``build``) running its stages in
order:

    questions  merge the sources into one record format, drop duplicate
               questions and the fixed question types
               (``filters.DROP_FRAMES_REASONING_TYPES``): questions.jsonl
               (``questions.py``)
    pages      fetch every gold page of the next ``limit`` questions from its
               source (Wikipedia API, live web or the Wayback Machine), then
               Filter A (``filters.py``): keep a question when every gold page
               was fetched or, with web gold pages, at least
               ``pages.min_page_share`` of them; write the kept questions, one
               row each with its gold annotation (``gold/annotations.py``: the
               skeletons with the fetched documents, the criterion texts by
               an LLM, ``gold/criterion_texts.py``) and its split (every query in
               ``val``: the dataset is validation only, ``splits.py``), the
               qrels of their fetched gold pages in the canonical layout, and
               the corpus (``corpus.py``: those pages and their hard negatives)
    negatives  hard negatives of the Wikipedia gold pages without them
               (``negatives.py``), then the corpus again
    report     stats of the dataset as written (queries, qrels, corpus) per
               source, the hard negatives, the gold annotation (levels,
               criteria with answers and documents), consistency checks
               between the files, and the duplicate documents of the corpus

The build is incremental, like ``dra_inference.py``: a question is processed
once Filter A has a final verdict (``filters.verdict``) from the rows of its
gold pages in ``pages/index.jsonl``; a build skips the processed questions and
takes the next ``limit`` of the others (of ``sources`` only, when given;
otherwise in the questions' order; 0: none, the dataset files are rewritten
and no negatives fetched).  Transient failures (transport errors, HTTP 429 /
5xx) leave a question unprocessed, so a later build retries it, up to
``filters.MAX_ATTEMPTS`` fetches of the page.  The dataset files are
rewritten by every build from all the processed questions.  The page
settings that decide a question's verdict (``PAGE_SETTINGS``) must stay those
of the earlier builds.

Modules:
    questions   the questions stage
    pages       the pages stage (``PageStore``, progress, Filter A, dataset files)
    negatives   the negatives stage (memory, resume)
    report      the report stage
    corpus      the corpus and its id files
    duplicates  the corpus duplicates (negatives filter, report)
    filters     type filter, Filter A
    fetch/      HTTP sessions (``http_client``), Wikipedia (``wikipedia_api``),
                web pages (``web_pages``)
"""

from typing import List, Optional

from ..config import BuildConfig, DatasetPaths, require_contact
from .negatives import run_negatives
from .pages import run_pages
from .questions import run_questions
from .report import run_report


def build(cfg: BuildConfig, paths: DatasetPaths, limit: Optional[int] = None,
          sources: Optional[List[str]] = None) -> None:
    """The stages in order: the questions, the gold pages of the next
    *limit* of them (of *sources*), the hard negatives of the new Wikipedia
    gold pages (none with ``limit`` 0), the report."""
    require_contact(cfg, "the webshaper_frames build")
    run_questions(paths)
    run_pages(cfg, paths, limit, sources)
    if limit != 0:
        run_negatives(cfg, paths)
    run_report(paths, cfg.negatives.dedup_threshold)
