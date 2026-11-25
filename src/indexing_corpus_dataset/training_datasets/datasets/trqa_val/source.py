"""trqa_val's source: the TRQA wiki1 and wiki2 validation queries (local
files under ``DATA_ROOT/trqa``), each with its intermediate info and its
qrels rows (``meta``, copied unchanged); the gold pages are the relevant
qrels documents (grade above 0).  The id is TRQA's (the two collections'
ids do not collide)."""

from typing import Dict, Iterator, List

from indexing_corpus_dataset.shared.layout import dataset_root, qrels_base, queries_base, resolve_split_id
from utils.io import iter_jsonl

from .trqa import docs_of

# The TRQA collections whose validation files make trqa_val.
TRQA_SUBSETS = ("wiki1", "wiki2")


def info() -> dict:
    return {"files": [str(queries_base(dataset_root("trqa"), resolve_split_id("trqa", "validation", s)))
                      for s in TRQA_SUBSETS]}


def _lines(path) -> List[str]:
    """The non-blank lines of *path*, each ending with a newline."""
    with open(path, encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh if line.strip()]


def load() -> Iterator[dict]:
    trqa = dataset_root("trqa")
    for subset in TRQA_SUBSETS:
        split = resolve_split_id("trqa", "validation", subset)
        base = queries_base(trqa, split)
        info = {str(r["qid"]): r for r in iter_jsonl(base.with_name(f"{base.name}_intermediate_info.jsonl"))}
        qrels: Dict[str, List[str]] = {}
        for line in _lines(qrels_base(trqa, split).with_suffix(".txt")):
            qrels.setdefault(line.split()[0], []).append(line)
        for row in iter_jsonl(base.with_suffix(".jsonl")):
            qid = str(row["id"])
            docs = list(dict.fromkeys(docs_of(qrels.get(qid, [])).get(qid, [])))
            yield {
                "id": qid, "source": f"trqa_{subset}",
                "question": str(row["text"]).strip(), "answer": str(row["answer"]).strip(),
                "gold": [{"key": f"doc:{d}", "kind": "doc", "id": d} for d in docs],
                "annotation": None,
                "meta": {"info": info.get(qid, {}), "qrels": qrels.get(qid, [])},
            }
