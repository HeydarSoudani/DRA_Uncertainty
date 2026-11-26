"""The belief sources of the belief model: the fixed per-query beliefs
list C = {c_1, ..., c_K}, produced once at the start of each sample.

``LLMBeliefSource`` derives the beliefs from the query;
``BankedBeliefSource`` reads them from a bank file and has its producer
(the extractor, or TRQA's gold beliefs built from the intermediate info or
from trqa_val's annotation rows, ``trqa_gold.py``) make and add only the
missing ones, so runs and analyses share one list per query;
``FallbackBeliefSource`` gives a query without gold beliefs silver
ones.  Which source a dataset and ``belief_source`` get is decided in
``init.factory``.

Beliefs are gold or silver (``layout.BELIEF_SOURCES``): gold when they
come from the question (the extractor's beliefs of a BrowseComp-Plus
question or a report request, ``layout.extraction_kind``) or from the
dataset's annotation; silver when an LLM extracts structure the question
does not state (TRQA's members, FRAMES' hops).  Each query's belief info
records the kind as ``belief_source``.

Each belief is closed (one fact with one answer: a clue, a hop, a
member's value), an aggregate (the target of a multi-clue query, the
entity that meets the clues before it, or the computation of a
multi-member query), or, in a report request, an aspect of the topic.  A belief refers to beliefs before it by their
ids ("born in B1", "the composer that meets B1 to B3"), never to a
later one (``parsing.forward_refs``).  A capped list keeps the answer and
the beliefs closest to it (``parsing.cap_chain``); a multi-clue list is
checked against its layout (``parsing.clue_layout``).  A multi-member query lists its members as
closed beliefs, then an aggregate belief, how the answer follows from
the members' values, as in the gold beliefs of the training datasets.  The kind follows from the dataset's query shape
and the belief's position (``parsing.belief_kinds``); the extractor
only lists the beliefs.
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

from indexing_corpus_dataset.layout import BELIEF_SOURCES, SILVER_BELIEFS
from utils.text_utils import parse_json_object

from ..core.types import BELIEF_KINDS, MULTI_ASPECT, MULTI_CLUE, MULTI_MEMBER, QUERY_SHAPES, Belief
from ..prompts import BELIEF_INIT_SYSTEMS, belief_init_user
from .parsing import cap_chain, clue_layout, belief_kinds, forward_refs, member_layout, salvage_items

logger = logging.getLogger(__name__)

# The list of the initializer's reply (``prompts/belief_init/system.txt``).
_REPLY_KEY = "beliefs"


# ---------------------------------------------------------------------------
# Belief source
# ---------------------------------------------------------------------------

class BeliefSource(ABC):
    """Produces the fixed belief list of one query."""

    @abstractmethod
    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Belief], Dict[str, Any]]:
        """Return ``(beliefs, info)``; *info* is saved in the meta line.

        *info* carries at least ``errors`` (a list of strings); an empty
        belief list with a non-empty ``errors`` means the source failed.
        """

    def prepare(self) -> None:
        """Set up what ``get`` needs before it is called from several
        threads (e.g. a corpus index); nothing by default."""

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Belief]]:
        """The query's stored beliefs (banked or gold) without extracting
        any, or None when none are stored; None by default."""
        return None


class LLMBeliefSource(BeliefSource):
    """Derives the beliefs of the query with one LLM call.

    The dataset's query shape (multi-clue, multi-hop, multi-member or
    multi-aspect) picks the prompt and the kind of each belief; the
    model only lists the beliefs (``{id, text}`` each; the ids are code's).  The shape is saved in
    *info* as ``query_shape``.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_beliefs: Cap on the number of beliefs kept (the dataset's
            ``max_beliefs`` in layout.DATASET_SPECS), or None (uncapped:
            gold beliefs); at least 2 for a multi-member query (a member and
            the aggregate).  A multi-aspect query lists every
            belief the request states and is never capped (must be None).
        model_name: Saved in the meta line and in the bank key
            (``init.factory.bank_model_id``: a served checkpoint's path
            included).
        max_tokens: Max tokens for the LLM call.
        temperature: LLM temperature.
        query_shape: ``"multi_clue"``, ``"multi_hop"``, ``"multi_member"`` or
            ``"multi_aspect"`` (the dataset's ``query_shape`` in
            layout.DATASET_SPECS).
        kind: ``"gold"`` or ``"silver"``, the kind of the beliefs it lists
            (``layout.extraction_kind``), saved in *info* as
            ``belief_source``.
    """

    def __init__(
        self,
        llm_client: Any,
        max_beliefs: Optional[int],
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        query_shape: str = MULTI_CLUE,
        kind: str = SILVER_BELIEFS,
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        if kind not in BELIEF_SOURCES:
            raise ValueError(f"unknown belief kind {kind!r}; expected one of {BELIEF_SOURCES}")
        if query_shape == MULTI_ASPECT and max_beliefs is not None:
            raise ValueError(f"max_beliefs={max_beliefs} for a report request; its beliefs are every "
                             "belief the request states, uncapped (max_beliefs null)")
        if query_shape == MULTI_MEMBER and max_beliefs is not None and max_beliefs < 2:
            raise ValueError(f"max_beliefs={max_beliefs} for a multi-member query; it needs at least 2 "
                             "(a member and the aggregate)")
        self._shape = query_shape
        self.kind = kind
        self._system = BELIEF_INIT_SYSTEMS[query_shape]
        self._llm = llm_client
        self.model_name = model_name
        self._max_beliefs = max_beliefs
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def bank_fields(self) -> Dict[str, Any]:
        """What, besides the query, decides the beliefs: the bank key of
        ``BankedBeliefSource``.  ``prompt`` hashes the system prompt and
        the user message without the query."""
        prompt = self._system + "\n" + belief_init_user(self._shape, "", self._max_beliefs)
        return {
            "model": self.model_name,
            "query_shape": self._shape,
            "max_beliefs": self._max_beliefs,
            "prompt": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
        }

    def messages(self, query: str) -> List[Dict[str, str]]:
        """The extractor's chat messages for *query*."""
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": belief_init_user(self._shape, query, self._max_beliefs)},
        ]

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Belief], Dict[str, Any]]:
        try:
            raw = self._llm.complete(self.messages(query), max_tokens=self._max_tokens,
                                     temperature=self._temperature)
        except Exception as e:
            logger.warning("LLMBeliefSource: LLM call failed", exc_info=True)
            return [], {"model": self.model_name, "query_shape": self._shape, "belief_source": self.kind,
                        "errors": [f"llm_call: {e}"]}
        return self.parse(raw)

    def parse(self, raw: Optional[str]) -> Tuple[List[Belief], Dict[str, Any]]:
        """The beliefs of the extractor's reply *raw* and the extraction info."""
        info: Dict[str, Any] = {"model": self.model_name, "query_shape": self._shape, "belief_source": self.kind,
                                "errors": []}
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get(_REPLY_KEY), list):
            # Keep the complete items of malformed or truncated output.
            body = (raw or "").partition(f'"{_REPLY_KEY}"')[2].partition("[")[2]
            salvaged = salvage_items(body)
            if not salvaged:
                info["errors"].append(f"parse: no JSON object with a '{_REPLY_KEY}' list")
                info["raw"] = raw
                return [], info
            info["warnings"] = [f"parse: invalid JSON, kept {len(salvaged)} complete items"]
            info["salvaged"] = True
            data = {_REPLY_KEY: salvaged}

        items: List[str] = []
        for item in data[_REPLY_KEY]:
            if isinstance(item, dict):
                item = item.get("text") or ""
            item = str(item).strip()
            if item and item not in items:
                items.append(item)
        if not items:
            info["errors"].append("parse: empty belief list")
            info["raw"] = raw

        if self._shape == MULTI_MEMBER:
            items = member_layout(items, self._max_beliefs, info)
        elif self._max_beliefs is not None:
            items = cap_chain(items, self._max_beliefs, info)
        if self._shape == MULTI_CLUE and items:
            layout = clue_layout(items)
            if layout:
                # Asked once more (``BankedBeliefSource``), then kept.
                info["warnings"] = info.get("warnings", []) + [f"layout: {'; '.join(layout)}"]
                info["bad_layout"] = True
        beliefs = [
            Belief(id=f"B{k + 1}", text=t, kind=kind)
            for k, (t, kind) in enumerate(zip(items, belief_kinds(self._shape, items)))
        ]
        bad = forward_refs(items)
        if bad:
            # A belief refers only to beliefs before it; the extraction
            # is asked once more (``BankedBeliefSource``), then kept.
            info["warnings"] = info.get("warnings", []) + [f"order: {'; '.join(bad)}"]
            info["bad_order"] = True
        return beliefs, info


# Key of the info a producer returns that goes to the bank only (its
# record's ``bank_only``), never to a run's belief info: e.g. the gold
# annotation row of a TRQA query.
BANK_ONLY = "bank_only"
# Key of the info of a producer's empty result that is final (banked, so
# not asked again): e.g. gold belief texts that failed for good.
FINAL = "final"


def without_bank_only(info: Dict[str, Any]) -> Dict[str, Any]:
    """*info* without its bank-only part (``BANK_ONLY``)."""
    return {k: v for k, v in info.items() if k != BANK_ONLY}


class BankedBeliefSource(BeliefSource):
    """Reads the beliefs of a query from a bank file; a query missing from
    it gets them from *inner*, the producer, and they are added to the bank.

    *inner* is any ``BeliefSource`` with ``bank_fields`` (what, besides
    the query, decides its beliefs): the extractor
    (``LLMBeliefSource``: model, query shape, cap, prompt hash) or TRQA's
    gold producer (``trqa_gold.py``: the
    writer of the texts and its prompt hash).

    The bank (``layout.belief_bank_path``) is a JSONL file, one line per
    query and setting: the query id and text, ``inner.bank_fields``, the
    beliefs, the info and the producer's bank-only data (``BANK_ONLY`` of
    its info, e.g. a gold annotation row; never returned).  A line is used
    only when all of these match, so a changed prompt or cap makes the
    beliefs again.  A salvaged extraction (the complete items of a
    truncated reply) is asked once more and fails when salvaged again, so
    every belief list used is banked.  A failed result (no beliefs) is
    never added, unless the producer marks it final (``FINAL``: texts that
    failed for good), so it is not asked again: it is banked with no
    beliefs.  Lines are appended under a file lock, after reading the lines
    added since the last read (by another worker or process): when one of
    them is the query's, it is used and none is added, so one query has one
    banked list.

    *info* is the stored info with ``bank``: ``"hit"`` (read from the
    bank), ``"added"`` (made and added) or ``"failed"`` (no beliefs: not
    added, or banked as a final failure).
    """

    def __init__(self, inner: BeliefSource, path: Path | str) -> None:
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
                logger.warning("BankedBeliefSource: a line of %s is not a bank record; skipped", self.path)
        self._offset += end

    def _key(self, query_id: Optional[str], query: str) -> str:
        fields = {"query_id": query_id, "query": query, **self._fields}
        return hashlib.sha1(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _banked(self, rec: Dict[str, Any]) -> Tuple[List[Belief], Dict[str, Any]]:
        beliefs = [Belief(**c) for c in rec["beliefs"]]
        old = sorted({c.kind for c in beliefs} - set(BELIEF_KINDS))
        if old:
            # A bank made before schema 45 (kinds hop, target, rest).
            raise ValueError(f"{self.path}: banked beliefs of kinds {old}, which the belief no longer takes "
                             f"(kinds {BELIEF_KINDS}); delete the bank and rebuild it")
        return beliefs, {**rec["info"], "bank": "hit" if beliefs else "failed"}

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Belief]]:
        with self._lock:
            rec = self._bank.get(self._key(query_id, query))
        return None if rec is None else self._banked(rec)[0]

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Belief], Dict[str, Any]]:
        key = self._key(query_id, query)
        with self._lock:
            rec = self._bank.get(key)
        if rec is not None:
            return self._banked(rec)

        # A salvaged reply is asked once more; salvaged again, the extraction
        # fails.  A reply whose beliefs refer to a later one, exceed the cap
        # or (multi-clue) miss the layout is asked once more too, then kept
        # (with its warning).
        for _ in range(2):
            beliefs, info = self._inner.get(query_id, query)
            if not any(info.get(flag) for flag in ("salvaged", "bad_order", "over_cap", "bad_layout")):
                break
        if info.get("salvaged"):
            info = {**info, "errors": info.get("errors", []) + ["parse: truncated reply twice, not banked"]}
            beliefs = []
        bank_only = info.get(BANK_ONLY)
        info = without_bank_only(info)
        if not beliefs and not info.get(FINAL):
            return beliefs, {**info, "bank": "failed"}
        rec = {
            "key": key, "query_id": query_id, "query": query, **self._fields,
            "beliefs": [c.to_dict() for c in beliefs], "info": info,
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
        return beliefs, {**info, "bank": "added" if beliefs else "failed"}


# Model name in the belief info of gold beliefs (``trqa_gold.py``).
GOLD_BELIEF_MODEL = "gold"


class FallbackBeliefSource(BeliefSource):
    """The gold beliefs of a query from *gold* (a training dataset's rows, or
    TRQA's banked gold), and for a query without them (no beliefs) the
    silver ones of *fallback*, with the reason, the gold source's
    ``errors``, in its info as ``gold_missing``.  The bank-only part of the
    info (``BANK_ONLY``) is never returned."""

    def __init__(self, gold: BeliefSource, fallback: BeliefSource) -> None:
        self.gold = gold
        self.fallback = fallback

    def prepare(self) -> None:
        self.gold.prepare()
        self.fallback.prepare()

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Belief]]:
        return self.gold.lookup(query_id, query) or self.fallback.lookup(query_id, query)

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Belief], Dict[str, Any]]:
        beliefs, info = self.gold.get(query_id, query)
        if beliefs:
            return beliefs, without_bank_only(info)
        reason = "; ".join(info.get("errors") or []) or "no gold beliefs"
        beliefs, info = self.fallback.get(query_id, query)
        return beliefs, {**without_bank_only(info), "gold_missing": reason}
