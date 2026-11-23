"""frames' source: FRAMES (``google/frames-benchmark``), questions with
their Wikipedia links and reasoning types; no skeleton (no beliefs)."""

import ast
import csv
import re
import sys
from typing import Iterator, List

from ...corpus.pages import dedup_gold, url_gold
from ...pipeline.source import hf_file, hf_info

# Pinned to a revision: the question ids are row numbers.
FRAMES = ("google/frames-benchmark", "test.tsv", "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef")


def info() -> dict:
    return hf_info(FRAMES)


def _links(raw: str) -> List[str]:
    """The links of a FRAMES row: a list literal, some of whose items pack
    several links ("https://...a, https://...b") or a note after the link
    ("https://... (NOT REQUIRED, BUT HELPFUL)")."""
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, (list, tuple)):
            return [part.split()[0] for item in items
                    for part in re.split(r",\s*(?=https?://)", str(item).strip()) if part.split()]
    except (ValueError, SyntaxError):
        pass
    return re.findall(r"https?://[^\s'\",\]]+", raw or "")


def load() -> Iterator[dict]:
    csv.field_size_limit(sys.maxsize)
    with open(hf_file(FRAMES), encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            entries = [url_gold(u) for u in _links(row.get("wiki_links", ""))]
            yield {
                "id": f"frames_{i:04d}", "source": "frames",
                "question": row["Prompt"].strip(), "answer": row["Answer"].strip(),
                "gold": dedup_gold(entries),
                "annotation": {"query_shape": "multi_hop", "beliefs": [], "entity_names": []},
                "meta": {"ignored_urls": entries.count(None),
                         "reasoning_types": [t.strip() for t in (row.get("reasoning_types") or "").split("|")
                                             if t.strip()]},
            }
