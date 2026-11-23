"""frames' fact writer (``gold_passages/facts.py``): one fact per gold page,
written by an LLM (``prompts/facts_*.txt``) from the question, the answer
and the page titles; the entity is the page title, the LLM writes the
property the question needs from the page and its value when the question
or the answer states it."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...gold_passages.facts import FactSpec, fact, property_statement, read_prompts


def parse(record: dict, pages: List[dict], data: Dict[str, Any]) -> Tuple[List[dict], Dict[str, str], List[str]]:
    """The facts of a reply (one per page), or the problems."""
    items = [i for i in data.get("facts") or [] if isinstance(i, dict)]
    by_page: Dict[int, dict] = {}
    problems = []
    for item in items:
        try:
            k = int(item.get("page"))
        except (TypeError, ValueError):
            problems.append(f"a fact without a page number: {item}")
            continue
        if not 1 <= k <= len(pages) or k in by_page:
            problems.append(f"page {k} is not a page number of the list or has two facts")
            continue
        by_page[k] = item
    missing = [k for k in range(1, len(pages) + 1) if k not in by_page]
    if missing:
        problems.append(f"no fact for page(s) {missing}")
    for k, item in sorted(by_page.items()):
        if not str(item.get("property") or "").strip():
            problems.append(f"the fact of page {k} has no property")
    if problems:
        return [], {}, problems
    facts = []
    for k in range(1, len(pages) + 1):
        item, page = by_page[k], pages[k - 1]
        value = item.get("value")
        value = str(value).strip() if value is not None and str(value).strip() else None
        prop = str(item["property"]).strip()
        facts.append(fact(k, page["title"], prop, value, property_statement(page["title"], prop, value),
                          [page["key"]], []))
    return facts, {}, []


SPEC = FactSpec("frames", *read_prompts(Path(__file__).resolve().parent / "prompts"), parse=parse)
