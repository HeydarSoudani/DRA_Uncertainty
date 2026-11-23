"""The record format of the source questions and the pinned Hugging Face
files they come from.  Each dataset loads its source in this format
(``datasets/{dataset}/source.py``: ``load`` and ``info``); stage
data_prepare writes every record of a source, nothing removed, to the
dataset's raw folder (``DatasetPaths.raw_questions``).

Record (one JSON line per question)::

    {"id": "infoseek_00123", "source": "infoseek" | "webshaper" | "frames"
                                       | "trqa_wiki1" | "trqa_wiki2",
     "question": str, "answer": str,
     "gold": [{"key": str, "kind": "wiki", "lang": str, "title": str}
              | {"key": str, "kind": "web", "url": str}
              | {"key": str, "kind": "doc", "id": str}],
     "annotation": {"query_shape": str, "beliefs": [...], "entity_names": [...]},
     "meta": {"ignored_urls": int, "reasoning_types": [...],
              "root": {...}, "formalization": [...],
              "info": {...}, "qrels": [str]}}   (what the source has)

``gold`` holds the source's gold pages (``corpus/pages.py``; InfoSeek: the
Wikipedia page of every tree node; WebShaper: its URLs; FRAMES: its
Wikipedia links; TRQA: the documents of its qrels, by corpus id).
``annotation`` is the skeleton of the gold annotation
(``annotation/rows.py``): the beliefs the source states (InfoSeek: the hops
of its research tree; WebShaper: the variables of its formalization;
FRAMES: none) with their relations, answers and source id (``ref``); the
build gives each ref its documents.  ``meta`` keeps what the facts of the
gold passages are made from (``datasets/{dataset}/facts.py``): InfoSeek's
research tree (``root``), WebShaper's triples (``formalization``).
"""

from typing import Dict, Tuple

from huggingface_hub import hf_hub_download

# A source file: (Hugging Face dataset repo, file, revision).
Pinned = Tuple[str, str, str]


def hf_file(pinned: Pinned) -> str:
    """The local path of a pinned source file, downloaded when missing."""
    repo, filename, revision = pinned
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision)


def hf_info(pinned: Pinned) -> Dict[str, str]:
    """Where the questions come from (``source.json``)."""
    repo, filename, revision = pinned
    return {"repo": repo, "file": filename, "revision": revision}
