"""The text of passages of corpus_wiki and corpus_web, by id."""

from typing import Dict, Iterable

from ...shared.doc_lookup import CorpusLookup
from ..pipeline.progress import CallBar, batches
from .web import WEB_ID_PREFIX, web_corpus_path
from .wiki import wiki_corpus_path

# Passages read per lookup (one progress step).
_BATCH = 2000


def passage_texts(doc_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """``{doc_id: {"title", "text"}}`` from corpus_wiki and corpus_web (a
    web passage by its id prefix), with a progress bar."""
    ids = set(doc_ids)
    web = {d for d in ids if d.startswith(WEB_ID_PREFIX)}
    found: Dict[str, Dict[str, str]] = {}
    with CallBar(len(ids), "[passages]", unit="passage") as bar:
        for corpus, wanted in ((wiki_corpus_path, ids - web), (web_corpus_path, web)):
            if not wanted:
                continue
            lookup = CorpusLookup(corpus())
            for batch in batches(sorted(wanted), _BATCH):
                got = lookup.get(batch)
                found.update(got)
                bar.update("found", len(got))
                if len(got) < len(batch):
                    bar.update("missing", len(batch) - len(got))
    return found
