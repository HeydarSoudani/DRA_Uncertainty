"""The English Wikipedia gold pages of a dataset's records in corpus_wiki
(stage data_prepare)."""

from typing import Callable, Dict, List, Optional

from ..pipeline.funnel import Check
from .texts import passage_texts
from .wiki import check_corpus, ensure_corpus, lookup_titles, page_chunks, read_titles, title_key


class WikiPages:
    """The English Wikipedia gold pages of the kept records, found in
    corpus_wiki by title (``wiki.lookup_titles``, cached in the raw
    folder) with the text of their first two chunks (where InfoSeek writes
    its claims from; chunk 1 is checked for text)."""

    def __init__(self, paths, records: List[dict], pages_of: Callable[[dict], List[dict]]):
        check_corpus(paths)
        ensure_corpus()
        keys = {g["key"]: title_key(g["title"]) for r in records for g in pages_of(r)}
        self.titles = lookup_titles(paths, set(keys.values()), read_titles(paths))
        self.chunks = {key: page_chunks(self.titles, t) for key, t in keys.items()}
        heads = {d for ids in self.chunks.values() for d in ids[:2]}
        print(f"[prepare] reading the first two chunks of {sum(bool(c) for c in self.chunks.values())} gold pages")
        self.texts = passage_texts(heads)
        self.looked_up = set(keys.values())

    def head(self, key: str) -> str:
        """The title and text of the first two chunks of a page, joined."""
        parts = []
        for doc_id in self.chunks.get(key, [])[:2]:
            passage = self.texts.get(doc_id) or {}
            parts += [passage.get("title", ""), passage.get("text", "")]
        return "\n".join(p for p in parts if p)

    def check(self, pages_of: Callable[[dict], List[dict]]) -> Check:
        """A gold page not in corpus_wiki (``page_not_in_corpus``) or whose
        first chunk has no text (``page_empty``)."""
        def check(record: dict) -> Optional[List[dict]]:
            reasons = []
            for g in pages_of(record):
                ids = self.chunks.get(g["key"]) or []
                if not ids:
                    reasons.append({"status": "page_not_in_corpus", "title": g["title"]})
                elif not (self.texts.get(ids[0]) or {}).get("text", "").strip():
                    reasons.append({"status": "page_empty", "title": g["title"]})
            return reasons or None
        return check

    def features(self, record: dict, pages_of: Callable[[dict], List[dict]]) -> Dict[str, List[str]]:
        """Gold page key -> its chunk ids, in page order."""
        return {g["key"]: self.chunks[g["key"]] for g in pages_of(record)}
