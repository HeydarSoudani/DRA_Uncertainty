"""The gold beliefs of a TRQA query (``trqa``: wiki1, wiki2, ecommerce; test
or validation), made from its intermediate info when the belief model
first needs them, and banked (``TRQAGoldBeliefs`` is the producer of a
``BankedBeliefSource`` over the gold bank,
``layout.belief_bank_path(..., "gold")``).

Per query, the row of the training datasets' format
(``indexing_corpus_dataset.training_datasets.annotation.rows``,
``multi_member_row``), as the trqa_val dataset builds its own: one belief per member of the set
("<member>: <property>", its value and documents,
``entity_values[].passage_ids``), then the aggregate.  A
member is named by the page title of its first document, looked up in the
corpus by id (``CorpusLookup``; wiki1 and wiki2: ``wiki_partial``); without
a corpus (ecommerce), by TRQA's label.  The writer (the run's
``belief_model``, native reasoning off) writes the property phrase and the
aggregation (``belief_texts.multi_member_texts``, the prompts of the training datasets), from
which the belief texts are built.

Results:

- texts written: the beliefs (id, text, kind) and, in the bank only
  (``BANK_ONLY``), the whole row (gold answers, documents, members);
- texts failed for good (a leak after the retry): no beliefs, banked as
  final (``FINAL``), so the writer is not asked again; the query gets
  silver beliefs (``FallbackBeliefSource``);
- no intermediate info, or the writer call failed: no beliefs, not banked
  (asked again next run).
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import GOLD_BELIEFS, dataset_root, default_corpus_path, qrels_base, queries_base
from indexing_corpus_dataset.training_datasets.annotation.belief_texts import BeliefWriter, belief_texts, prompt_hash
from indexing_corpus_dataset.training_datasets.annotation.rows import multi_member_row
from indexing_corpus_dataset.training_datasets.datasets.trqa_val.trqa import docs_of, member_titles
from utils.io_utils import iter_jsonl

from ..core.types import Belief
from .sources import BANK_ONLY, FINAL, GOLD_BELIEF_MODEL, BeliefSource

DATASET = "trqa"
# The TRQA subsets whose member documents are in a local corpus (the
# page title is the first line of a document).
TITLED_SUBSETS = ("wiki1", "wiki2")
# Where the gold beliefs come from, in the bank key.
SOURCE = "trqa_intermediate_info"


def subset_of(split: str) -> str:
    """The TRQA subset of split id *split* (``wiki2_test`` -> ``wiki2``)."""
    return split.split("_", 1)[0]


def _docs_of(path: Path) -> Dict[str, List[str]]:
    """The relevant documents of each query in the qrels file *path*."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return docs_of(fh)


def corpus_of(split: str) -> Optional[Path]:
    """The corpus of the split's member documents, or None (ecommerce, or
    a corpus not on disk)."""
    subset = subset_of(split)
    if subset not in TITLED_SUBSETS:
        return None
    path = default_corpus_path(DATASET, subset)
    return path if path.exists() else None


class TRQAGoldBeliefs(BeliefSource):
    """Makes the gold beliefs of a query of TRQA split *split* (e.g.
    ``wiki2_test``) from its intermediate info, with the texts of *writer*
    (an LLM name, the run's ``belief_model``); the producer of the gold
    bank.  The split's files are read once; the writer is created at its
    first call.

    Args:
        split: TRQA split id.
        writer: LLM that writes the property phrase and the aggregation;
            in the bank key (``init.factory.bank_model_id``: a served
            checkpoint's path included).
        llm_client: A client of *writer* (``complete(messages, **kwargs)``);
            None: an API client is created at the first call.
        max_tokens: Max tokens of a writer call.
        data_path: TRQA's dataset folder (None = ``layout.dataset_root``).
    """

    def __init__(self, split: str, writer: str, llm_client: Any = None, max_tokens: int = 4096,
                 data_path: Optional[Path] = None) -> None:
        self.split = split
        self.subset = subset_of(split)
        self.writer_model = writer
        self._max_tokens = max_tokens
        root = Path(data_path) if data_path else dataset_root(DATASET)
        base = queries_base(root, split)
        info_path = base.with_name(f"{base.name}_intermediate_info.jsonl")
        self._infos = {str(r["qid"]): r for r in iter_jsonl(info_path)} if info_path.exists() else {}
        queries = base.with_suffix(".jsonl")
        self._answers = {str(r["id"]): str(r.get("answer", "")) for r in iter_jsonl(queries)} if queries.exists() else {}
        self._docs = _docs_of(qrels_base(root, split).with_suffix(".txt"))
        corpus = corpus_of(split)
        self._lookup = CorpusLookup(corpus) if corpus is not None else None
        self._llm_client = llm_client
        self._writer: Optional[BeliefWriter] = None
        self._writer_lock = threading.Lock()

    @property
    def bank_fields(self) -> Dict[str, Any]:
        """What, besides the query, decides the gold beliefs: the bank key
        of ``BankedBeliefSource``."""
        return {"model": GOLD_BELIEF_MODEL, "source": SOURCE, "writer": self.writer_model,
                "prompt": prompt_hash("multi_member")}

    def prepare(self) -> None:
        """Build the corpus index of the member titles now, once, before
        threads or workers would each build it."""
        if self._lookup is not None:
            self._lookup.ensure_index()

    def _writer_client(self) -> BeliefWriter:
        with self._writer_lock:
            if self._writer is None:
                self._writer = BeliefWriter(self.writer_model, self._max_tokens, self._llm_client)
            return self._writer

    def _titles(self, info: dict) -> Dict[str, str]:
        """The page title of each member's first document (doc id -> title)."""
        return member_titles(self._lookup, [info]) if self._lookup is not None else {}

    def row(self, query_id: str, query: str) -> dict:
        """The query's annotation row, without belief texts."""
        info = self._infos[query_id]
        return multi_member_row(query_id, f"trqa_{self.subset}", query, self._answers.get(query_id, ""), info,
                                self._docs.get(query_id, []), self._titles(info))

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Belief], Dict[str, Any]]:
        qid = str(query_id)
        base: Dict[str, Any] = {"model": GOLD_BELIEF_MODEL, "query_shape": "multi_member", "belief_source": GOLD_BELIEFS,
                                "writer": self.writer_model}
        if qid not in self._infos:
            return [], {**base, "errors": [f"gold beliefs: no intermediate info for query {qid} of trqa {self.split}"]}
        row = self.row(qid, query)
        if not row["beliefs"]:
            return [], {**base, "errors": ["gold beliefs: no members in the intermediate info"], FINAL: True,
                        BANK_ONLY: {"row": row}}
        try:
            done = belief_texts(row, self._writer_client())
        except Exception as e:  # noqa: BLE001 - not banked; asked again next run
            return [], {**base, "errors": [f"gold beliefs: writer: {e}"]}
        if not done["texts_ok"]:
            return [], {**base, "errors": [f"gold beliefs: {err}" for err in done["errors"]] or
                        ["gold beliefs: no belief texts"], FINAL: True, BANK_ONLY: {"row": done}}
        beliefs = [Belief(id=c["id"], text=c["text"], kind=c["kind"]) for c in done["beliefs"]]
        return beliefs, {**base, "gold_level": done["level"], "errors": [], BANK_ONLY: {"row": done}}
