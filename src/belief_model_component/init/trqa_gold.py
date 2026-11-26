"""The gold beliefs of a TRQA query, made from its gold annotation when the
belief model first needs them, and banked (the producer of a
``BankedBeliefSource`` over the gold bank,
``layout.belief_bank_path(..., "gold")``):

    TRQAGoldBeliefs     a query of ``trqa`` (wiki1, wiki2, ecommerce; test
                        or validation), its row made from the split's
                        intermediate info
    DatasetGoldBeliefs  a query of a training dataset whose gold beliefs
                        are its annotation (trqa_val; frames has none):
                        its row as the dataset build wrote it
                        (``layout.gold_beliefs_path``)

Per query, the row of the training datasets' format
(``indexing_corpus_dataset.training_datasets.annotation.rows``,
``member_row``) with its beliefs (``with_member_beliefs``): one belief per
member of the set ("<member>: <property>", its value and documents,
``entity_values[].passage_ids``), then the aggregate.  A member is named by the page title of its first
document, looked up in the corpus by id (``CorpusLookup``; wiki1 and
wiki2: ``wiki_partial``); without a corpus (ecommerce), by TRQA's label.
The writer (the run's ``belief_model``, native reasoning off) writes the
property phrase and the aggregation (``member_texts.py``), from which the
belief texts are built.

Results:

- texts written: the beliefs (id, text, kind) and, in the bank only
  (``BANK_ONLY``), the row with its texts (gold answers, documents,
  members);
- texts failed for good (a leak after the retry, no member): no beliefs,
  banked as final (``FINAL``), so the writer is not asked again; the
  query gets silver beliefs (``FallbackBeliefSource``);
- no gold annotation (no intermediate info, no row, a row without
  members such as FRAMES'), or the writer call failed: no beliefs, not
  banked (asked again next run).
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.shared.doc_lookup import CorpusLookup
from indexing_corpus_dataset.shared.layout import GOLD_BELIEFS, dataset_root, default_corpus_path, qrels_base, queries_base
from indexing_corpus_dataset.training_datasets.annotation.rows import AGGREGATE, CLOSED, member_row, value_texts
from indexing_corpus_dataset.training_datasets.datasets.trqa_val.trqa import docs_of, member_titles
from indexing_corpus_dataset.training_datasets.llm import JsonWriter
from utils.io import iter_jsonl

from ..core.types import Belief
from .member_texts import PROMPT_HASH, member_texts
from .sources import BANK_ONLY, FINAL, GOLD_BELIEF_MODEL, BeliefSource

DATASET = "trqa"
# The TRQA subsets whose member documents are in a local corpus (the
# page title is the first line of a document).
TITLED_SUBSETS = ("wiki1", "wiki2")


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


def _number(text: str) -> Any:
    """*text* as a number when it is one (TRQA's answer "26832.0" as
    26832.0), else *text*."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def with_member_beliefs(row: dict, answer_value: Any) -> dict:
    """*row* (a multi-member row, ``member_row``) with its beliefs: one per
    member (``closed``; answers = its value, doc_ids = its documents), then
    the aggregate one (answers = the final answer as the query states it
    and *answer_value*, TRQA's answer value, as text; no doc_ids: it is
    computed).  Belief: ``{"id", "answers", "doc_ids", "kind",
    "relations"}``."""
    def belief(k: int, kind: str, answers: List[str], doc_ids: List[str]) -> dict:
        return {"id": f"B{k}", "answers": answers, "doc_ids": doc_ids, "kind": kind, "relations": []}

    beliefs = [belief(k, CLOSED, value_texts(m["value"]), m["doc_ids"]) for k, m in enumerate(row["members"], 1)]
    answer = row["answer"]
    answers = list(dict.fromkeys([answer, *value_texts(answer_value)])) if answer else value_texts(answer_value)
    beliefs.append(belief(len(beliefs) + 1, AGGREGATE, answers, []))
    return {**row, "beliefs": beliefs}


class MemberGoldBeliefs(BeliefSource):
    """The gold beliefs of a multi-member query from its row with its
    beliefs (``row``, by the subclass), with the texts of *writer* (an LLM name,
    the run's ``belief_model``; in the bank key: ``init.factory.bank_model_id``,
    a served checkpoint's path included).  *llm_client*: a client of
    *writer* (``complete(messages, **kwargs)``); None: an API client is
    created at the first call."""

    #: Where the rows come from, in the bank key.
    SOURCE = ""

    def __init__(self, writer: str, llm_client: Any = None, max_tokens: int = 4096) -> None:
        self.writer_model = writer
        self._max_tokens = max_tokens
        self._llm_client = llm_client
        self._writer: Optional[JsonWriter] = None
        self._writer_lock = threading.Lock()

    @property
    def bank_fields(self) -> Dict[str, Any]:
        """What, besides the query, decides the gold beliefs: the bank key
        of ``BankedBeliefSource``."""
        return {"model": GOLD_BELIEF_MODEL, "source": self.SOURCE, "writer": self.writer_model,
                "prompt": PROMPT_HASH}

    def _writer_client(self) -> JsonWriter:
        with self._writer_lock:
            if self._writer is None:
                self._writer = JsonWriter(self.writer_model, self._max_tokens, self._llm_client)
            return self._writer

    def row(self, query_id: str, query: str) -> Tuple[Optional[dict], Optional[str]]:
        """``(row, None)``: the query's multi-member row with its beliefs
        (``with_member_beliefs``), without belief texts; ``(None, why)``
        when it has none."""
        raise NotImplementedError

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Belief], Dict[str, Any]]:
        base: Dict[str, Any] = {"model": GOLD_BELIEF_MODEL, "query_shape": "multi_member", "belief_source": GOLD_BELIEFS,
                                "writer": self.writer_model}
        row, why = self.row(str(query_id), query)
        if row is None:
            return [], {**base, "errors": [f"gold beliefs: {why}"]}
        try:
            texts, errors = member_texts(row, self._writer_client())
        except Exception as e:  # noqa: BLE001 - not banked; asked again next run
            return [], {**base, "errors": [f"gold beliefs: writer: {e}"]}   # sources.INIT_CALL_ERRORS
        if texts is None:
            return [], {**base, "errors": [f"gold beliefs: {err}" for err in errors], FINAL: True,
                        BANK_ONLY: {"row": row}}
        done = {**row, "beliefs": [{**c, "text": t} for c, t in zip(row["beliefs"], texts)]}
        beliefs = [Belief(id=c["id"], text=c["text"], kind=c["kind"]) for c in done["beliefs"]]
        return beliefs, {**base, "errors": [], BANK_ONLY: {"row": done}}


class TRQAGoldBeliefs(MemberGoldBeliefs):
    """The gold beliefs of a query of TRQA split *split* (e.g.
    ``wiki2_test``) from its intermediate info; the split's files are read
    once.  *data_path*: TRQA's dataset folder (None = ``layout.dataset_root``)."""

    SOURCE = "trqa_intermediate_info"

    def __init__(self, split: str, writer: str, llm_client: Any = None, max_tokens: int = 4096,
                 data_path: Optional[Path] = None) -> None:
        super().__init__(writer, llm_client, max_tokens)
        self.split = split
        self.subset = subset_of(split)
        root = Path(data_path) if data_path else dataset_root(DATASET)
        base = queries_base(root, split)
        info_path = base.with_name(f"{base.name}_intermediate_info.jsonl")
        self._infos = {str(r["qid"]): r for r in iter_jsonl(info_path)} if info_path.exists() else {}
        queries = base.with_suffix(".jsonl")
        self._answers = {str(r["id"]): str(r.get("answer", "")) for r in iter_jsonl(queries)} if queries.exists() else {}
        self._docs = _docs_of(qrels_base(root, split).with_suffix(".txt"))
        corpus = corpus_of(split)
        self._lookup = CorpusLookup(corpus) if corpus is not None else None

    def prepare(self) -> None:
        """Build the corpus index of the member titles now, once, before
        threads or workers would each build it."""
        if self._lookup is not None:
            self._lookup.ensure_index()

    def _titles(self, info: dict) -> Dict[str, str]:
        """The page title of each member's first document (doc id -> title)."""
        return member_titles(self._lookup, [info]) if self._lookup is not None else {}

    def row(self, query_id: str, query: str) -> Tuple[Optional[dict], Optional[str]]:
        info = self._infos.get(query_id)
        if info is None:
            return None, f"no intermediate info for query {query_id} of trqa {self.split}"
        row = member_row(query_id, f"trqa_{self.subset}", query, self._answers.get(query_id, ""), info,
                         self._docs.get(query_id, []), self._titles(info))
        if not row["members"]:
            return None, "no members in the intermediate info"
        return with_member_beliefs(row, info.get("answer")), None


class DatasetGoldBeliefs(MemberGoldBeliefs):
    """The gold beliefs of a query of a training dataset whose gold beliefs
    are its annotation: its row of the dataset's queries file *path*
    (``layout.gold_beliefs_path``), read once.  A row without members
    (FRAMES') or with another query text has none."""

    SOURCE = "training_dataset_rows"

    def __init__(self, path: Path | str, writer: str, llm_client: Any = None, max_tokens: int = 4096) -> None:
        super().__init__(writer, llm_client, max_tokens)
        self.path = Path(path)
        self._rows = {str(r["id"]): r for r in iter_jsonl(self.path)} if self.path.exists() else {}

    def row(self, query_id: str, query: str) -> Tuple[Optional[dict], Optional[str]]:
        row = self._rows.get(query_id)
        if row is None:
            return None, f"no row for query {query_id} in {self.path}"
        if not row["members"]:
            return None, "no members (the source states no set)"
        if row["text"] != query:
            return None, f"the query of {query_id} in {self.path} differs"
        return with_member_beliefs(row, _number(row["answer"])), None
