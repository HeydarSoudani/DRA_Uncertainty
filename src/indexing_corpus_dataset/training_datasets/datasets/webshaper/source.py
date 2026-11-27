"""webshaper's source: WebShaper (``Alibaba-NLP/WebShaper``), questions with
their source URLs and formalization triples (``meta``; the facts of the
gold passages are made from them, ``facts.py``)."""

from typing import Iterator

from utils.io import iter_jsonl

from ...corpus.pages import dedup_gold, url_gold
from ...pipeline.source import hf_file, hf_info

WEBSHAPER = ("Alibaba-NLP/WebShaper", "data/main-00000-of-00001.jsonl", "5dd17c63c2d0f86d245c262bfd7bf444200e126f")


def info() -> dict:
    return hf_info(WEBSHAPER)


def load() -> Iterator[dict]:
    for row in iter_jsonl(hf_file(WEBSHAPER)):
        triples = row.get("formalization") or []
        entries = [url_gold(u) for u in row.get("urls") or []]
        answer = str(row["answer"]).strip()
        yield {
            "id": f"webshaper_{row['id']}", "source": "webshaper",
            "question": row["question"].strip().strip('"').strip(), "answer": answer,
            "gold": dedup_gold(entries),
            "annotation": {"entity_names": []},
            "meta": {"ignored_urls": entries.count(None), "formalization": triples},
        }

