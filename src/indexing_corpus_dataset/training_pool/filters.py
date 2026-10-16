"""Question filters: the type filter of step 1 and Filter A of step 2.
Each returns the kept records and the dropped ones, the latter with their
reasons (``{"id", "source", "reasons": [...]}``).

Filter A keeps a question with only Wikipedia gold pages when every page was
fetched (each is a hop of the question: InfoSeek, FRAMES), and a question
with web gold pages when at least ``min_page_share`` of all its gold pages
were (any page is one source among others: WebShaper); the failed pages of a
kept question get no qrels.
"""

from typing import Dict, Iterable, List, Tuple

# FRAMES reasoning types whose questions are dropped.  Tabular: the answer
# sits in a table, the part of a page most likely changed since the gold
# (current revisions are fetched).  Post processing: a transformation of the
# found facts (e.g. "in binary"), whose failures are not research failures.
DROP_FRAMES_REASONING_TYPES = ("Tabular reasoning", "Post processing")

# A page that failed transiently is fetched again by the next runs, this many
# fetches in all; then its failure is final.
MAX_ATTEMPTS = 3


def type_filter(records: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    """Drop the FRAMES questions with a reasoning type of
    ``DROP_FRAMES_REASONING_TYPES``."""
    kept, dropped = [], []
    for record in records:
        reasons = []
        if record["source"] == "frames":
            reasons = [f"reasoning_type:{t}" for t in record["meta"].get("reasoning_types", [])
                       if t in DROP_FRAMES_REASONING_TYPES]
        if reasons:
            dropped.append({"id": record["id"], "source": record["source"], "reasons": reasons})
        else:
            kept.append(record)
    return kept, dropped


def failed_transiently(row: dict) -> bool:
    """A page row of a failure a retry may fix: transport errors, HTTP 429 /
    5xx, or a Wayback fallback that could not query the archive."""
    status = row["status"]
    return (status in ("error", "fetch_error", "http_429") or status.startswith("http_5")
            or row.get("wayback_status") == "wayback_error")


def is_transient(row: dict) -> bool:
    """A failure a later run retries: transient, fetched fewer than
    ``MAX_ATTEMPTS`` times (rows without ``attempts`` count once)."""
    return failed_transiently(row) and row.get("attempts", 1) < MAX_ATTEMPTS


def verdict(record: dict, rows: Dict[str, dict], min_page_share: float) -> str:
    """``dropped`` once Filter A cannot keep the question (no gold page, or
    too many pages failed for good: one without web gold pages, else more
    than ``1 - min_page_share`` of them), ``kept`` once its pages are fetched
    and pass it, ``retry`` when a page failed transiently, ``pending`` while
    a page has no row."""
    if not record["gold"]:
        return "dropped"
    states = []
    for gold in record["gold"]:
        row = rows.get(gold["key"])
        if row is None:
            states.append("pending")
        elif row["status"] == "ok":
            states.append("ok")
        else:
            states.append("retry" if is_transient(row) else "failed")
    share = min_page_share if any(g["kind"] == "web" for g in record["gold"]) else 1.0
    if (len(states) - states.count("failed")) / len(states) < share:
        return "dropped"
    if "pending" in states:
        return "pending"
    return "retry" if "retry" in states else "kept"


def failed_pages(record: dict, rows: Dict[str, dict]) -> List[dict]:
    """The gold pages of *record* that have a row and were not fetched."""
    return [{"key": g["key"], "status": rows[g["key"]]["status"]} for g in record["gold"]
            if g["key"] in rows and rows[g["key"]]["status"] != "ok"]


def filter_a(records: Iterable[dict], rows: Dict[str, dict], min_page_share: float) -> Tuple[List[dict], List[dict]]:
    """The questions with a final verdict, kept or dropped; a dropped question
    lists each failed page with its status (``no_gold`` without gold pages)."""
    kept, dropped = [], []
    for record in records:
        outcome = verdict(record, rows, min_page_share)
        if outcome == "kept":
            kept.append(record)
        elif outcome == "dropped":
            reasons = failed_pages(record, rows) or [{"key": None, "status": "no_gold"}]
            dropped.append({"id": record["id"], "source": record["source"], "reasons": reasons})
    return kept, dropped
