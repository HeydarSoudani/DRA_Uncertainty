"""The text of passages of corpus_wiki and corpus_web, by id."""

from typing import Dict, Iterable

from ...doc_lookup import CorpusLookup
from .web import WEB_ID_PREFIX, web_corpus_path
from .wiki import wiki_corpus_path


def passage_texts(doc_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """``{doc_id: {"title", "text"}}`` from corpus_wiki and corpus_web (a
    web passage by its id prefix)."""
    ids = set(doc_ids)
    web = {d for d in ids if d.startswith(WEB_ID_PREFIX)}
    found: Dict[str, Dict[str, str]] = {}
    if ids - web:
        found.update(CorpusLookup(wiki_corpus_path()).get(ids - web))
    if web:
        found.update(CorpusLookup(web_corpus_path()).get(web))
    return found
