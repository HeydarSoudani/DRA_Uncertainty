"""The criteria sources of the belief model: the fixed per-query criteria
list C = {c_1, ..., c_K}, produced once at the start of each sample.

``LLMCriteriaSource`` derives the criteria from the query;
``BankedCriteriaSource`` reads them from a bank file and has its producer
(the extractor, or TRQA's gold criteria built from the intermediate info,
``indexing_corpus_dataset.training_datasets.gold.trqa``) make and add only the
missing ones, so runs and analyses share one list per query;
``GoldCriteriaSource`` reads a training dataset's gold criteria (the dataset's
queries file is their bank, written by the dataset build only);
``FallbackCriteriaSource`` gives a query without gold criteria silver
ones.  Which source a dataset and ``criteria_source`` get is decided in
``criteria.factory``.

Criteria are gold or silver (``layout.CRITERIA_SOURCES``): gold when they
come from the question (the extractor's criteria of a BrowseComp-Plus
question or a report request, ``layout.extraction_kind``) or from the
dataset's annotation; silver when an LLM extracts structure the question
does not state (TRQA's members, FRAMES' hops).  Each query's criteria info
records the kind as ``criteria_kind``.

Each criterion is closed (one fact with one answer: a clue, a hop, a
member's value), an aggregate (the target of a multi-clue query, the
entity that meets the clues before it, or the computation of a
multi-member query), or, in a report request, an aspect of the topic.  A criterion refers to criteria before it by their
ids ("the town of c1", "the composer that meets c1 to c3"), never to a
later one (``parsing.forward_refs``).  A capped list keeps the answer and
the criteria closest to it (``parsing.cap_chain``); a multi-clue list is
checked against its layout (``parsing.clue_layout``).  A multi-member query lists its members as
closed criteria, then an aggregate criterion, how the answer follows from
the members' values, as in the gold criteria of the training datasets.  The kind follows from the dataset's query shape
and the criterion's position (``parsing.criterion_kinds``); the extractor
only lists the criteria.
"""

import fcntl
import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.layout import CRITERIA_SOURCES, GOLD_CRITERIA, SILVER_CRITERIA
from utils.text_utils import parse_json_object

from ..core.types import CRITERION_KINDS, MULTI_ASPECT, MULTI_CLUE, MULTI_MEMBER, QUERY_SHAPES, Criterion
from ..prompts import CRITERIA_INIT_SYSTEMS, criteria_init_user
from .parsing import cap_chain, clue_layout, criterion_kinds, forward_refs, member_layout, salvage_items

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Criteria source
# ---------------------------------------------------------------------------

class CriteriaSource(ABC):
    """Produces the fixed criteria list of one query."""

    @abstractmethod
    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        """Return ``(criteria, info)``; *info* is saved in the meta line.

        *info* carries at least ``errors`` (a list of strings); an empty
        criteria list with a non-empty ``errors`` means the source failed.
        """

    def prepare(self) -> None:
        """Set up what ``get`` needs before it is called from several
        threads (e.g. a corpus index); nothing by default."""

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Criterion]]:
        """The query's stored criteria (banked or gold) without extracting
        any, or None when none are stored; None by default."""
        return None


class LLMCriteriaSource(CriteriaSource):
    """Derives the criteria of the query with one LLM call.

    The dataset's query shape (multi-clue, multi-hop, multi-member or
    multi-aspect) picks the prompt and the kind of each criterion; the
    model only lists the criteria, as strings.  The shape is saved in
    *info* as ``query_shape``.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_criteria: Cap on the number of criteria kept (the dataset's
            ``max_criteria`` in layout.DATASET_SPECS), or None (uncapped:
            gold criteria); at least 2 for a multi-member query (a member and
            the aggregate).  A multi-aspect query lists every
            criterion the request states and is never capped (must be None).
        model_name: Saved in the meta line and in the bank key
            (``criteria.factory.bank_model_id``: a served checkpoint's path
            included).
        max_tokens: Max tokens for the LLM call.
        temperature: LLM temperature.
        query_shape: ``"multi_clue"``, ``"multi_hop"``, ``"multi_member"`` or
            ``"multi_aspect"`` (the dataset's ``query_shape`` in
            layout.DATASET_SPECS).
        kind: ``"gold"`` or ``"silver"``, the kind of the criteria it lists
            (``layout.extraction_kind``), saved in *info* as
            ``criteria_kind``.
    """

    def __init__(
        self,
        llm_client: Any,
        max_criteria: Optional[int],
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        query_shape: str = "multi_clue",
        kind: str = SILVER_CRITERIA,
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        if kind not in CRITERIA_SOURCES:
            raise ValueError(f"unknown criteria kind {kind!r}; expected one of {CRITERIA_SOURCES}")
        if query_shape == MULTI_ASPECT and max_criteria is not None:
            raise ValueError(f"max_criteria={max_criteria} for a report request; its criteria are every "
                             "criterion the request states, uncapped (max_criteria null)")
        if query_shape == MULTI_MEMBER and max_criteria is not None and max_criteria < 2:
            raise ValueError(f"max_criteria={max_criteria} for a multi-member query; it needs at least 2 "
                             "(a member and the aggregate)")
        self._shape = query_shape
        self.kind = kind
        self._system = CRITERIA_INIT_SYSTEMS[query_shape]
        self._llm = llm_client
        self.model_name = model_name
        self._max_criteria = max_criteria
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def bank_fields(self) -> Dict[str, Any]:
        """What, besides the query, decides the criteria: the bank key of
        ``BankedCriteriaSource``.  ``prompt`` hashes the system prompt and
        the user message without the query."""
        prompt = self._system + "\n" + criteria_init_user(self._shape, "", self._max_criteria)
        return {
            "model": self.model_name,
            "query_shape": self._shape,
            "max_criteria": self._max_criteria,
            "prompt": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
        }

    def messages(self, query: str) -> List[Dict[str, str]]:
        """The extractor's chat messages for *query*."""
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": criteria_init_user(self._shape, query, self._max_criteria)},
        ]

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        try:
            raw = self._llm.complete(self.messages(query), max_tokens=self._max_tokens,
                                     temperature=self._temperature)
        except Exception as e:
            logger.warning("LLMCriteriaSource: LLM call failed", exc_info=True)
            return [], {"model": self.model_name, "query_shape": self._shape, "criteria_kind": self.kind,
                        "errors": [f"llm_call: {e}"]}
        return self.parse(raw)

    def parse(self, raw: Optional[str]) -> Tuple[List[Criterion], Dict[str, Any]]:
        """The criteria of the extractor's reply *raw* and the extraction info."""
        info: Dict[str, Any] = {"model": self.model_name, "query_shape": self._shape, "criteria_kind": self.kind,
                                "errors": []}
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("criteria"), list):
            # Keep the complete items of malformed or truncated output.
            body = (raw or "").partition('"criteria"')[2].partition("[")[2]
            salvaged = salvage_items(body)
            if not salvaged:
                info["errors"].append("parse: no JSON object with a 'criteria' list")
                info["raw"] = raw
                return [], info
            info["warnings"] = [f"parse: invalid JSON, kept {len(salvaged)} complete items"]
            info["salvaged"] = True
            data = {"criteria": salvaged}

        items: List[str] = []
        for item in data["criteria"]:
            if isinstance(item, dict):
                item = item.get("text") or ""
            item = str(item).strip()
            if item and item not in items:
                items.append(item)
        if not items:
            info["errors"].append("parse: empty criteria list")
            info["raw"] = raw

        if self._shape == MULTI_MEMBER:
            items = member_layout(items, self._max_criteria, info)
        elif self._max_criteria is not None:
            items = cap_chain(items, self._max_criteria, info)
        if self._shape == MULTI_CLUE and items:
            layout = clue_layout(items)
            if layout:
                # Asked once more (``BankedCriteriaSource``), then kept.
                info["warnings"] = info.get("warnings", []) + [f"layout: {'; '.join(layout)}"]
                info["bad_layout"] = True
        criteria = [
            Criterion(id=f"c{k + 1}", text=t, kind=kind)
            for k, (t, kind) in enumerate(zip(items, criterion_kinds(self._shape, items)))
        ]
        bad = forward_refs(items)
        if bad:
            # A criterion refers only to criteria before it; the extraction
            # is asked once more (``BankedCriteriaSource``), then kept.
            info["warnings"] = info.get("warnings", []) + [f"order: {'; '.join(bad)}"]
            info["bad_order"] = True
        return criteria, info


# Key of the info a producer returns that goes to the bank only (its
# record's ``bank_only``), never to a run's criteria info: e.g. the gold
# annotation row of a TRQA query.
BANK_ONLY = "bank_only"
# Key of the info of a producer's empty result that is final (banked, so
# not asked again): e.g. gold criterion texts that failed for good.
FINAL = "final"


def without_bank_only(info: Dict[str, Any]) -> Dict[str, Any]:
    """*info* without its bank-only part (``BANK_ONLY``)."""
    return {k: v for k, v in info.items() if k != BANK_ONLY}


class BankedCriteriaSource(CriteriaSource):
    """Reads the criteria of a query from a bank file; a query missing from
    it gets them from *inner*, the producer, and they are added to the bank.

    *inner* is any ``CriteriaSource`` with ``bank_fields`` (what, besides
    the query, decides its criteria): the extractor
    (``LLMCriteriaSource``: model, query shape, cap, prompt hash) or TRQA's
    gold producer (``indexing_corpus_dataset.training_datasets.gold.trqa``: the
    writer of the texts and its prompt hash).

    The bank (``layout.criteria_bank_path``) is a JSONL file, one line per
    query and setting: the query id and text, ``inner.bank_fields``, the
    criteria, the info and the producer's bank-only data (``BANK_ONLY`` of
    its info, e.g. a gold annotation row; never returned).  A line is used
    only when all of these match, so a changed prompt or cap makes the
    criteria again.  A salvaged extraction (the complete items of a
    truncated reply) is asked once more and fails when salvaged again, so
    every criteria list used is banked.  A failed result (no criteria) is
    never added, unless the producer marks it final (``FINAL``: texts that
    failed for good), so it is not asked again: it is banked with no
    criteria.  Lines are appended under a file lock, after reading the lines
    added since the last read (by another worker or process): when one of
    them is the query's, it is used and none is added, so one query has one
    banked list.

    *info* is the stored info with ``bank``: ``"hit"`` (read from the
    bank), ``"added"`` (made and added) or ``"failed"`` (no criteria: not
    added, or banked as a final failure).
    """

    def __init__(self, inner: CriteriaSource, path: Path | str) -> None:
        self._inner = inner
        self.path = Path(path)
        self._fields = inner.bank_fields
        self._lock = threading.Lock()
        self._bank: Dict[str, Dict[str, Any]] = {}
        # Bytes of the file read so far, up to the end of its last whole line.
        self._offset = 0
        if self.path.exists():
            with open(self.path, "rb") as f:
                self._read_new(f)

    def prepare(self) -> None:
        self._inner.prepare()

    def _read_new(self, f) -> None:
        """Add the whole lines of *f* (binary) after ``_offset`` to the
        in-memory bank.  A partial last line (being written, or cut by a
        killed process) is left for the next read; a line that is not a
        record is skipped with a warning."""
        f.seek(self._offset)
        data = f.read()
        end = data.rfind(b"\n") + 1
        for line in data[:end].decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                self._bank[rec["key"]] = rec
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("BankedCriteriaSource: a line of %s is not a bank record; skipped", self.path)
        self._offset += end

    def _key(self, query_id: Optional[str], query: str) -> str:
        fields = {"query_id": query_id, "query": query, **self._fields}
        return hashlib.sha1(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _banked(self, rec: Dict[str, Any]) -> Tuple[List[Criterion], Dict[str, Any]]:
        criteria = [Criterion(**c) for c in rec["criteria"]]
        old = sorted({c.kind for c in criteria} - set(CRITERION_KINDS))
        if old:
            # A bank made before schema 45 (kinds hop, target, rest).
            raise ValueError(f"{self.path}: banked criteria of kinds {old}, which the belief no longer takes "
                             f"(kinds {CRITERION_KINDS}); delete the bank and rebuild it")
        return criteria, {**rec["info"], "bank": "hit" if criteria else "failed"}

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Criterion]]:
        with self._lock:
            rec = self._bank.get(self._key(query_id, query))
        return None if rec is None else self._banked(rec)[0]

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        key = self._key(query_id, query)
        with self._lock:
            rec = self._bank.get(key)
        if rec is not None:
            return self._banked(rec)

        # A salvaged reply is asked once more; salvaged again, the extraction
        # fails.  A reply whose criteria refer to a later one, exceed the cap
        # or (multi-clue) miss the layout is asked once more too, then kept
        # (with its warning).
        for _ in range(2):
            criteria, info = self._inner.get(query_id, query)
            if not any(info.get(flag) for flag in ("salvaged", "bad_order", "over_cap", "bad_layout")):
                break
        if info.get("salvaged"):
            info = {**info, "errors": info.get("errors", []) + ["parse: truncated reply twice, not banked"]}
            criteria = []
        bank_only = info.get(BANK_ONLY)
        info = without_bank_only(info)
        if not criteria and not info.get(FINAL):
            return criteria, {**info, "bank": "failed"}
        rec = {
            "key": key, "query_id": query_id, "query": query, **self._fields,
            "criteria": [c.to_dict() for c in criteria], "info": info,
            **({BANK_ONLY: bank_only} if bank_only is not None else {}),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a+b") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # Lines another worker or process added since the last read.
                    self._read_new(f)
                    banked = self._bank.get(key)
                    if banked is None:
                        # A partial last line (a killed writer) must not run
                        # into the new record.
                        size = os.fstat(f.fileno()).st_size
                        line = json.dumps(rec, ensure_ascii=False) + "\n"
                        f.write((("\n" if size > self._offset else "") + line).encode("utf-8"))
                        f.flush()
                        self._offset = f.tell()
                        self._bank[key] = rec
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        if banked is not None:
            return self._banked(banked)
        return criteria, {**info, "bank": "added" if criteria else "failed"}


# Model name in the criteria info of gold criteria (``GoldCriteriaSource``).
GOLD_CRITERIA_MODEL = "gold"


# The annotation level of a query whose gold criteria have every gold
# answer and document (training_datasets/gold/annotations.py); belief model training
# scores only such queries.  The belief model uses the gold criteria of any
# level.
GOLD_CRITERIA_LEVEL = "full"


class GoldCriteriaSource(CriteriaSource):
    """Reads each query's gold criteria from a training dataset's queries file
    (``layout.gold_criteria_path``, one row per query in the format of
    ``training_datasets/gold/annotations.py``, written by the dataset build): per query,
    its criteria with their gold answers and documents.  Only the criteria
    (id, text, kind; the kind as ``parsing.criterion_kinds`` gives it for
    the shape) reach the belief model, of a query whose row has
    criteria with their texts (``texts_ok``), at any annotation ``level``
    (WebShaper's ``partial`` rows included); the info records the row's
    ``level`` as ``gold_level``.

    Any other query (no row, no criteria such as FRAMES', failed texts, or
    another query text: ``missing``) gets no criteria, the reason in its
    ``errors``; ``FallbackCriteriaSource`` gives it silver ones."""

    def __init__(self, path: Path | str, query_shape: str) -> None:
        self.path = Path(path)
        self._shape = query_shape
        # Per query, only what the belief model reads of its row (the rows
        # also hold the gold answers and documents).
        self._rows: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        criteria = row.get("criteria") or []
                        if any(c.get("kind") == "rest" for c in criteria):
                            raise ValueError(f"{self.path}: query {row['id']} has a rest criterion "
                                             "(\"any other member\"), which the belief no longer takes; "
                                             "rebuild the dataset's gold criteria")
                        kinds = criterion_kinds(query_shape, [c["text"] for c in criteria])
                        self._rows[str(row["id"])] = {
                            "text": row.get("text"), "texts_ok": row.get("texts_ok"), "level": row.get("level"),
                            "criteria": [Criterion(id=c["id"], text=c["text"], kind=kind)
                                         for c, kind in zip(criteria, kinds)],
                        }

    def missing(self, query_id: Optional[str], query: str) -> Optional[str]:
        """Why the query has no gold criteria to use, or None when it has."""
        row = self._rows.get(str(query_id))
        if row is None:
            return f"no row for query {query_id} in {self.path}"
        if not row.get("criteria"):
            return "no gold criteria (the source states no structure)"
        if not row.get("texts_ok"):
            return "no criterion texts"
        if row.get("text") is not None and row["text"] != query:
            return f"the query of {query_id} in {self.path} differs"
        return None

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Criterion]]:
        if self.missing(query_id, query) is not None:
            return None
        return list(self._rows[str(query_id)]["criteria"])

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        info: Dict[str, Any] = {"model": GOLD_CRITERIA_MODEL, "query_shape": self._shape, "criteria_kind": GOLD_CRITERIA,
                                "errors": []}
        reason = self.missing(query_id, query)
        if reason is not None:
            return [], {**info, "errors": [f"gold criteria: {reason}"]}
        row = self._rows[str(query_id)]
        return list(row["criteria"]), {**info, "gold_level": row.get("level")}


class FallbackCriteriaSource(CriteriaSource):
    """The gold criteria of a query from *gold* (a training dataset's rows, or
    TRQA's banked gold), and for a query without them (no criteria) the
    silver ones of *fallback*, with the reason, the gold source's
    ``errors``, in its info as ``gold_missing``.  The bank-only part of the
    info (``BANK_ONLY``) is never returned."""

    def __init__(self, gold: CriteriaSource, fallback: CriteriaSource) -> None:
        self.gold = gold
        self.fallback = fallback

    def prepare(self) -> None:
        self.gold.prepare()
        self.fallback.prepare()

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Criterion]]:
        return self.gold.lookup(query_id, query) or self.fallback.lookup(query_id, query)

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        criteria, info = self.gold.get(query_id, query)
        if criteria:
            return criteria, without_bank_only(info)
        reason = "; ".join(info.get("errors") or []) or "no gold criteria"
        criteria, info = self.fallback.get(query_id, query)
        return criteria, {**without_bank_only(info), "gold_missing": reason}
