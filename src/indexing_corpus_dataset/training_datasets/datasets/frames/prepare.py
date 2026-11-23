"""frames: the checks of stage data_prepare, in order.

    duplicate       a repeated question (its first row kept)
    reasoning_type  a question of a reasoning type in
                    ``DROP_REASONING_TYPES`` (``reasoning_type``)
    answer          empty, or stated by the question (``answer_invalid``)
    links           no gold link (``no_gold``), a link that does not parse
                    or is not an English Wikipedia page (``link_invalid``)
    pages           a gold page not in corpus_wiki, or whose first chunk
                    has no text (``page_not_in_corpus``, ``page_empty``)
"""

from typing import List, Optional

from ...corpus.wiki import corpus_stats
from ...corpus.wiki_pages import WikiPages
from ...pipeline.checks import answer_check, duplicate_check, no_gold_check
from ...pipeline.funnel import Funnel
from ...pipeline.stats import write_stats

# FRAMES reasoning types whose questions are dropped.  Tabular: the answer
# sits in a table, which corpus_wiki strips.  Post processing: a
# transformation of the found facts (e.g. "in binary"), whose failures are
# not research failures.
DROP_REASONING_TYPES = ("Tabular reasoning", "Post processing")
WIKI_LANG = "en"


def _pages(record: dict) -> List[dict]:
    return record["gold"]


def reasoning_type_check(record: dict) -> Optional[List[dict]]:
    types = [t for t in record["meta"].get("reasoning_types", []) if t in DROP_REASONING_TYPES]
    return [{"status": "reasoning_type", "type": t} for t in types] or None


def link_check(record: dict) -> Optional[List[dict]]:
    reasons = []
    if record["meta"].get("ignored_urls"):
        reasons.append({"status": "link_invalid", "why": "not a URL", "count": record["meta"]["ignored_urls"]})
    reasons += [{"status": "link_invalid", "why": "not an English Wikipedia page", "page": g["key"]}
                for g in record["gold"] if g["kind"] != "wiki" or g["lang"] != WIKI_LANG]
    return reasons or None


def run(cfg, paths, funnel: Funnel) -> dict:
    funnel.apply("duplicate", duplicate_check())
    funnel.apply("reasoning_type", reasoning_type_check)
    funnel.apply("answer", answer_check)
    funnel.apply("no_gold", no_gold_check)
    funnel.apply("links", link_check)
    pages = WikiPages(paths, funnel.kept, _pages)
    funnel.apply("pages", pages.check(_pages))
    for record in funnel.kept:
        record["features"] = {"page_chunks": pages.features(record, _pages)}
    write_stats(paths.stats, "corpus", corpus_stats(paths, pages.looked_up, pages.titles))
    return {}
