"""Wikipedia pages through the MediaWiki APIs.

* ``resolve_batch``: titles to canonical pages, up to 50 titles per Action
  API call (``action=query&redirects&prop=info|pageprops``): normalization,
  redirects, page id, latest revision id, and which titles are missing or
  disambiguation pages.
* ``more_like``, ``categories``, ``category_members``: the candidate hard
  negatives of a page (step 3): the articles of the most similar text
  (CirrusSearch ``morelike:``), the page's visible categories with their
  sizes, and the articles of one category; one Action API call each.
* ``fetch_html``: the Parsoid HTML of one revision (core REST API
  ``/w/rest.php/v1/revision/{id}/html``), which keeps infoboxes and tables.
* ``html_to_text``: plain text of that HTML, one line per block: the
  page text of the corpus.
"""

import re
from typing import Dict, List

import lxml.html

from .http_client import HttpClient

_DROP_XPATH = (
    "//script | //style | //link | //meta"
    " | //sup[contains(@class, 'reference')]"
    " | //*[contains(@class, 'mw-references-wrap')] | //ol[contains(@class, 'references')]"
    " | //*[contains(@class, 'navbox')] | //*[contains(@class, 'mw-editsection')]"
)
# The page title (<title>) is a block: the first line of the text.
_BLOCK_TAGS = {"title", "p", "li", "tr", "caption", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6",
               "section", "table", "ul", "ol", "dl", "figcaption", "blockquote", "pre"}
_CELL_TAGS = {"td", "th"}
# A <div> is a block outside table cells (e.g. the short description after
# the title); inside a cell it stays inline, so a cell keeps one line.
_DIV_IN_CELL_XPATH = "ancestor::td | ancestor::th"


class WikipediaClient:
    def __init__(self, http: HttpClient, lang: str = "en"):
        self.http = http
        self.lang = lang
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.rest = f"https://{lang}.wikipedia.org/w/rest.php/v1"

    def page_url(self, title: str) -> str:
        return f"https://{self.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"

    def resolve_batch(self, batch: List[str]) -> Dict[str, dict]:
        """Per title of *batch* (at most 50): ``{"status": "ok" | "missing" |
        "invalid" | "not_article" | "disambiguation", "canonical_title",
        "pageid", "revid", "redirected"}``."""
        response = self.http.post(self.api, data={
            "action": "query", "titles": "|".join(batch), "redirects": 1,
            "prop": "info|pageprops", "ppprop": "disambiguation",
            "format": "json", "formatversion": 2, "maxlag": 5,
        })
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        query = data.get("query", {})
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        pages = {p["title"]: p for p in query.get("pages", [])}
        out = {}
        for title in batch:
            name = normalized.get(title, title)
            target = redirects.get(name, name)
            page = pages.get(target)
            if page is None or page.get("invalid"):
                out[title] = {"status": "invalid"}
            elif page.get("special") or page.get("ns", 0) != 0:
                # Special:, Category:, Module: ... pages are not articles
                # (special pages have no page id).
                out[title] = {"status": "not_article"}
            elif page.get("missing"):
                out[title] = {"status": "missing"}
            elif "disambiguation" in page.get("pageprops", {}):
                out[title] = {"status": "disambiguation"}
            else:
                out[title] = {"status": "ok", "canonical_title": page["title"], "pageid": page["pageid"],
                              "revid": page["lastrevid"], "redirected": name in redirects}
        return out

    def _query(self, params: dict) -> dict:
        """The ``query`` part of an Action API GET reply (``{}`` when empty)."""
        response = self.http.get(self.api, params={**params, "action": "query", "format": "json",
                                                   "formatversion": 2, "maxlag": 5})
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        return data.get("query", {})

    @staticmethod
    def _articles(pages: List[dict]) -> List[dict]:
        """``{"pageid", "title", "revid", "disambiguation"}`` per article of
        a generator's *pages*."""
        return [{"pageid": p["pageid"], "title": p["title"], "revid": p["lastrevid"],
                 "disambiguation": "disambiguation" in p.get("pageprops", {})}
                for p in pages if p.get("ns", 0) == 0 and not p.get("missing") and "lastrevid" in p]

    def more_like(self, title: str, limit: int) -> List[dict]:
        """The articles whose text is most similar to page *title*'s
        (CirrusSearch ``morelike:``), most similar first, at most *limit*."""
        query = self._query({"generator": "search", "gsrsearch": f"morelike:{title}", "gsrnamespace": 0,
                             "gsrlimit": min(limit, 500), "prop": "info|pageprops", "ppprop": "disambiguation"})
        pages = sorted(query.get("pages", []), key=lambda p: p.get("index", 0))
        return self._articles(pages)

    def categories(self, pageid: int) -> List[dict]:
        """The visible (not hidden) categories of page *pageid*:
        ``{"title", "pages"}``, ``pages`` the number of articles in it."""
        query = self._query({"generator": "categories", "pageids": pageid, "gclshow": "!hidden",
                             "gcllimit": "max", "prop": "categoryinfo"})
        return [{"title": p["title"], "pages": p.get("categoryinfo", {}).get("pages", 0)}
                for p in query.get("pages", [])]

    def category_members(self, category: str, limit: int) -> List[dict]:
        """The articles of *category* (at most *limit*), by title."""
        query = self._query({"generator": "categorymembers", "gcmtitle": category, "gcmnamespace": 0,
                             "gcmtype": "page", "gcmlimit": min(limit, 500), "prop": "info|pageprops",
                             "ppprop": "disambiguation"})
        return sorted(self._articles(query.get("pages", [])), key=lambda a: a["title"])

    def fetch_html(self, revid: int) -> str:
        response = self.http.get(f"{self.rest}/revision/{revid}/html")
        response.raise_for_status()
        return response.text


def html_to_text(html: str) -> str:
    """Plain text of a Parsoid page: references, navboxes and edit links
    dropped; the title on the first line, then one line per block element,
    table cells joined by `` | ``."""
    root = lxml.html.fromstring(html)
    for node in root.xpath(_DROP_XPATH):
        node.drop_tree()
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        if node.tag in _BLOCK_TAGS or (node.tag == "div" and not node.xpath(_DIV_IN_CELL_XPATH)):
            node.tail = "\n" + (node.tail or "")
        elif node.tag in _CELL_TAGS:
            node.tail = " | " + (node.tail or "")
        elif node.tag == "br":
            node.tail = "\n" + (node.tail or "")
    lines = (re.sub(r"[ \t ]+", " ", line).strip(" |") for line in root.text_content().splitlines())
    return "\n".join(line for line in lines if line)
