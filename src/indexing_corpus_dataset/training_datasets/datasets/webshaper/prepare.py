"""webshaper: the checks of stage data_prepare, in order.

    duplicate   a repeated question (its first row kept)
    answer      empty, or stated by the question (``answer_invalid``)
    no_gold     no gold link (``no_gold``)
    fetch       a question with a web gold page whose fetch may still
                succeed is pending (neither kept nor dropped; the next run
                fetches it again)
    pages       no gold page in the corpus (``page_not_in_corpus``)

The English Wikipedia gold pages are found in corpus_wiki by title; the
other gold pages (web pages, other-language Wikipedia pages) are fetched,
live or from the Wayback Machine, and cut into corpus_web
(``corpus/web.py``; the fetches are kept in the raw folder).  As
in WebShaper, where a link of a question often no longer serves its page,
a question keeps the gold pages found (a page not found, or whose first
chunk has no text, is left out) and is dropped when none is found; its
facts are later judged on the pages kept (stage data_relevance).
"""

from typing import Dict, List, Optional
from urllib.parse import quote

from ...config import BuildConfig, require_contact
from ...corpus.web import PageStore, fetch_pages, is_final, web_corpus_path, write_web_corpus
from ...corpus.wiki import corpus_stats
from ...corpus.wiki_pages import WikiPages
from ...pipeline.checks import answer_check, duplicate_check, no_gold_check
from ...pipeline.funnel import Funnel
from ...pipeline.stats import write_stats

WIKI_LANG = "en"


def _wiki_pages(record: dict) -> List[dict]:
    return [g for g in record["gold"] if g["kind"] == "wiki" and g["lang"] == WIKI_LANG]


def pages(record: dict) -> List[dict]:
    """The gold pages of a question as the fact writer sees them: an
    English Wikipedia page by its title, any other page by its URL
    (fetched into corpus_web)."""
    out = []
    for g in record["gold"]:
        if g["kind"] == "wiki" and g["lang"] == WIKI_LANG:
            out.append({"key": g["key"], "kind": "wiki", "title": g["title"]})
        elif g["kind"] == "wiki":
            url = f"https://{g['lang']}.wikipedia.org/wiki/{quote(g['title'].replace(' ', '_'))}"
            out.append({"key": g["key"], "kind": "web", "url": url})
        else:
            out.append({"key": g["key"], "kind": "web", "url": g["url"]})
    return out


def run(cfg: BuildConfig, paths, funnel: Funnel) -> dict:
    funnel.apply("duplicate", duplicate_check())
    funnel.apply("answer", answer_check)
    funnel.apply("no_gold", no_gold_check)

    store = PageStore(paths)
    urls = list(dict.fromkeys(p["url"] for r in funnel.kept for p in pages(r) if p["kind"] == "web"))
    if any(not is_final(store.rows.get(u)) for u in urls):
        require_contact(cfg, "fetching the web gold pages")
    fetch_pages(cfg.web, store, urls)
    web_chunks = write_web_corpus(store)
    funnel.hold("fetch", {r["id"] for r in funnel.kept
                          if any(p["kind"] == "web" and not is_final(store.rows.get(p["url"])) for p in pages(r))})

    wiki = WikiPages(paths, funnel.kept, _wiki_pages)
    wiki_ok = wiki.check(_wiki_pages)

    def found(record: dict) -> Dict[str, List[str]]:
        """Page key -> chunk ids of the question's pages in the corpus."""
        bad = {r["title"] for r in wiki_ok(record) or []}
        out = {}
        for p in pages(record):
            chunks = (wiki.chunks.get(p["key"], []) if p["kind"] == "wiki" and p["title"] not in bad
                      else web_chunks.get(p["url"], []) if p["kind"] == "web" else [])
            if chunks:
                out[p["key"]] = chunks
        return out

    found_of = {r["id"]: found(r) for r in funnel.kept}

    def check(record: dict) -> Optional[List[dict]]:
        if found_of[record["id"]]:
            return None
        return [{"status": "page_not_in_corpus", "page": p.get("title") or p.get("url")} for p in pages(record)]

    funnel.apply("pages", check)
    for record in funnel.kept:
        chunks = found_of[record["id"]]
        record["features"] = {"pages": [p for p in pages(record) if p["key"] in chunks], "page_chunks": chunks}

    write_stats(paths.stats, "corpus", corpus_stats(paths, wiki.looked_up, wiki.titles))
    status: Dict[str, int] = {}
    for u in urls:
        s = store.rows[u]["status"] if u in store.rows else "not_fetched"
        status[s] = status.get(s, 0) + 1
    write_stats(paths.stats, "web", {"path": str(web_corpus_path()), "pages": len(urls),
                                     "status": dict(sorted(status.items())),
                                     "passages": sum(len(web_chunks.get(u, [])) for u in urls)})
    return {}

