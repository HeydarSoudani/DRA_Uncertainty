"""Criteria and belief state of the belief model.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K}; the belief state b_t gives each
criterion its nuggets: the answers the documents and the agent's turns
give for it, each with a status from Agent-BRACE's seven-level belief
scale (``types.BELIEF_STATUSES``) and the sources it rests on.

- ``CriteriaSource``: produces C once, at the start of each sample.
  ``LLMCriteriaSource`` derives the criteria from the query;
  ``BankedCriteriaSource`` reads them from a bank file and has its
  producer (the extractor, or TRQA's gold criteria built from the
  intermediate info, ``indexing_corpus_dataset.trqa_gold_criteria``) make
  and add only the missing ones, so runs and analyses share one list per
  query; ``GoldCriteriaSource`` reads a training pool's gold criteria (the
  pool's queries file is their bank, written by the pool build only);
  ``FallbackCriteriaSource`` gives a query without gold criteria silver
  ones.  Which source a dataset and ``criteria_source`` get is decided in
  ``criteria_init``.

Criteria are gold or silver (``layout.CRITERIA_SOURCES``): gold when they
come from the question (the extractor's criteria of a BrowseComp-Plus
question or a report request, ``layout.extraction_kind``) or from the
dataset's annotation; silver when an LLM extracts structure the question
does not state (TRQA's members, FRAMES' hops).  Each query's criteria info
records the kind as ``criteria_kind``.
- ``BeliefState``: b_t, per criterion its nuggets with their evidence.

Each criterion is closed (one fact) or, in a report request, an aspect of
the topic.  A set query lists its members as closed criteria, then a rest
criterion, "any other member: <property>", for the members the extractor
did not list, then an aggregate criterion, how the answer follows from the
members' values, as in the gold criteria of the training pools.  The kind
follows from the dataset's query shape and the criterion's position
(``_kinds``); the extractor only lists the criteria.

The belief update call is in ``updater``.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from indexing_corpus_dataset.layout import CRITERIA_SOURCES, GOLD_CRITERIA, SILVER_CRITERIA
from utils.text_utils import parse_json_object
from .prompts import CRITERIA_INIT_SYSTEMS, criteria_init_user
from .types import (
    AGGREGATE, ASPECT, CLOSED, DOC, MAX_NUGGETS, MULTI_ASPECT, NO_DOC_KINDS, QUERY_SHAPES, REST, REST_PAIR, SET,
    SUPPORTS, Criterion, Evidence, Nugget, NuggetUpdate, Observation, alnum, alnum_index, same_value,
)

logger = logging.getLogger(__name__)

# One complete string item of the criteria list, for output that is not
# valid JSON (cut at the token limit, or a stray character between items).
_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*(?=[,\]])')
# One complete object item (``{"text": ...}``), likewise.
_OBJECT_RE = re.compile(r'\{[^{}]*\}')


def _salvage(body: str) -> List[Any]:
    """The complete items of a malformed or truncated criteria list: its
    objects when it has any, else its strings."""
    objects = []
    for chunk in _OBJECT_RE.findall(body):
        try:
            objects.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return objects or [json.loads(f'"{t}"') for t in _ITEM_RE.findall(body)]


# Subject of the rest criterion of a set query ("any other member: <property>").
REST_SUBJECT = "any other member"


def _is_rest(text: str) -> bool:
    subject = text.rsplit(":", 1)[0] if ":" in text else text
    return " ".join(re.findall(r"[a-z]+", subject.lower())) in (REST_SUBJECT, REST_SUBJECT + "s")


def _kinds(shape: str, texts: List[str]) -> List[str]:
    """Kind of each criterion of a *shape* query: every clue of a single
    target and every hop of a multi-hop query is closed; a set query's list
    (``_set_layout``) is its members (closed), the rest criterion and the
    aggregate; every report criterion is an aspect."""
    if shape == MULTI_ASPECT:
        return [ASPECT] * len(texts)
    if shape == SET and len(texts) >= 2:
        return [CLOSED] * (len(texts) - 2) + [REST, AGGREGATE]
    return [CLOSED] * len(texts)


def _set_layout(items: List[str], cap: Optional[int], info: Dict[str, Any]) -> List[str]:
    """A set query's criteria as the prompt asks: the members, one rest
    criterion, then the aggregate, within *cap* (at least 3; None:
    uncapped) by leaving out members.  The aggregate is the first item after the rest criterion or,
    without one, the last item when it is not "<member>: <property>".  A
    missing rest criterion is added with the members' property, a missing
    aggregate with a generic text.  Every change is noted in
    ``info["warnings"]``."""
    if not items:
        return items
    warnings = info.setdefault("warnings", [])
    rest_at = next((k for k, t in enumerate(items) if _is_rest(t)), None)
    if rest_at is not None:
        members, after = items[:rest_at], items[rest_at + 1:]
        rest_item = items[rest_at]
        extra_rest = [t for t in after if _is_rest(t)]
        if extra_rest:
            warnings.append(f"rest: kept the first of {len(extra_rest) + 1} rest criteria, dropped {extra_rest}")
        after = [t for t in after if not _is_rest(t)]
        aggregate = after[0] if after else None
        if len(after) > 1:
            warnings.append(f"aggregate: kept the first item after the rest criterion, dropped {after[1:]}")
    else:
        aggregate = items[-1] if ":" not in items[-1] else None
        members = items[:-1] if aggregate is not None else list(items)
        rest_item = None
    prop = next((t.rsplit(":", 1)[1].strip() for t in members if ":" in t), "")
    if rest_item is None:
        rest_item = f"{REST_SUBJECT}: {prop or 'the property the query asks about'}"
        warnings.append("rest: added the missing rest criterion")
    if aggregate is None:
        aggregate = f"the answer the query computes from every member's {prop or 'value'}"
        warnings.append("aggregate: added the missing aggregate criterion")
    keep = len(members) if cap is None else cap - 2
    if len(members) > keep:
        warnings.append(f"cap: kept {keep} of {len(members)} members, dropped {members[keep:]}")
    if not warnings:
        del info["warnings"]
    return members[:keep] + [rest_item, aggregate]


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


class LLMCriteriaSource(CriteriaSource):
    """Derives the criteria of the query with one LLM call.

    The dataset's query shape (single target, multi-hop, set or
    multi-aspect) picks the prompt and the kind of each criterion; the
    model only lists the criteria, as strings.  The shape is saved in
    *info* as ``query_shape``.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_criteria: Cap on the number of criteria kept (the dataset's
            ``max_criteria`` in layout.DATASET_SPECS), or None (uncapped:
            gold criteria); at least 3 for a set query (a member, the rest
            criterion and the aggregate).  A multi-aspect query lists every
            criterion the request states and is never capped (must be None).
        model_name: Saved in the meta line and in the bank key
            (``criteria_init.bank_model_id``: a served checkpoint's path
            included).
        max_tokens: Max tokens for the LLM call.
        temperature: LLM temperature.
        query_shape: ``"single_target"``, ``"multi_hop"``, ``"set"`` or
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
        query_shape: str = "single_target",
        kind: str = SILVER_CRITERIA,
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        if kind not in CRITERIA_SOURCES:
            raise ValueError(f"unknown criteria kind {kind!r}; expected one of {CRITERIA_SOURCES}")
        if query_shape == MULTI_ASPECT and max_criteria is not None:
            raise ValueError(f"max_criteria={max_criteria} for a report request; its criteria are every "
                             "criterion the request states, uncapped (max_criteria null)")
        if query_shape == SET and max_criteria is not None and max_criteria < 3:
            raise ValueError(f"max_criteria={max_criteria} for a set query; it needs at least 3 "
                             "(a member, the rest criterion and the aggregate)")
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
            salvaged = _salvage(body)
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

        if self._shape == SET:
            items = _set_layout(items, self._max_criteria, info)
        elif self._max_criteria is not None:
            items = items[:self._max_criteria]
        criteria = [
            Criterion(id=f"c{k + 1}", text=t, kind=kind)
            for k, (t, kind) in enumerate(zip(items, _kinds(self._shape, items)))
        ]
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
    gold producer (``indexing_corpus_dataset.trqa_gold_criteria``: the
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

    @staticmethod
    def _banked(rec: Dict[str, Any]) -> Tuple[List[Criterion], Dict[str, Any]]:
        criteria = [Criterion(**c) for c in rec["criteria"]]
        return criteria, {**rec["info"], "bank": "hit" if criteria else "failed"}

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Criterion]]:
        """The banked criteria of the query (empty: a final failure), or
        None when it is not banked; never makes them."""
        with self._lock:
            rec = self._bank.get(self._key(query_id, query))
        return None if rec is None else [Criterion(**c) for c in rec["criteria"]]

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        key = self._key(query_id, query)
        with self._lock:
            rec = self._bank.get(key)
        if rec is not None:
            return self._banked(rec)

        # A salvaged reply is asked once more; salvaged again, the extraction fails.
        for _ in range(2):
            criteria, info = self._inner.get(query_id, query)
            if not info.get("salvaged"):
                break
        else:
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
# answer and document (training_pool/annotations.py); belief model training
# scores only such queries.  The belief model uses the gold criteria of any
# level.
GOLD_CRITERIA_LEVEL = "full"


class GoldCriteriaSource(CriteriaSource):
    """Reads each query's gold criteria from a training pool's queries file
    (``layout.gold_criteria_path``, one row per query in the format of
    ``training_pool/annotations.py``, written by the pool build): per query,
    its criteria with their gold answers and documents.  Only the criteria
    (id, text, kind) reach the belief model, of a query whose row has
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
                        self._rows[str(row["id"])] = {
                            "text": row.get("text"), "texts_ok": row.get("texts_ok"), "level": row.get("level"),
                            "criteria": [Criterion(id=c["id"], text=c["text"], kind=c["kind"])
                                         for c in row.get("criteria") or []],
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
        """The gold criteria of the query, or None."""
        if self.missing(query_id, query) is not None:
            return None
        return list(self._rows[str(query_id)]["criteria"])

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        info: Dict[str, Any] = {"model": GOLD_CRITERIA_MODEL, "query_shape": self._shape, "criteria_kind": GOLD_CRITERIA,
                                "errors": []}
        reason = self.missing(query_id, query)
        if reason is not None:
            return [], {**info, "errors": [f"gold criteria: {reason}"]}
        return self.lookup(query_id, query), {**info, "gold_level": self._rows[str(query_id)].get("level")}


class FallbackCriteriaSource(CriteriaSource):
    """The gold criteria of a query from *gold* (a training pool's rows, or
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

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        criteria, info = self.gold.get(query_id, query)
        if criteria:
            return criteria, without_bank_only(info)
        reason = "; ".join(info.get("errors") or []) or "no gold criteria"
        criteria, info = self.fallback.get(query_id, query)
        return criteria, {**without_bank_only(info), "gold_missing": reason}


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

# The letters-and-digits index of the agent's turns, kept for the few turns
# a call anchors against (every think observation of a call reads one turn).
_turn_index = lru_cache(maxsize=8)(alnum_index)


# A value whose letters and digits are this short, or only digits, must
# match whole words of the turn: "3" must not match inside "13".
SHORT_VALUE_CHARS = 3


def _whole_word(turn: str, start: int, end: int) -> bool:
    """*turn[start:end]* is not part of a longer word or number."""
    return ((start == 0 or not turn[start - 1].isalnum())
            and (end == len(turn) or not turn[end].isalnum()))


def anchor(value: str, turns: List[str]) -> Optional[str]:
    """The text of *turns* matching *value* (case, whitespace and
    punctuation ignored, so "12,388" matches "12388"), from the latest turn
    that has it, or None.  A short or numeric value (``SHORT_VALUE_CHARS``)
    matches only whole words or numbers."""
    target = alnum(value)
    if not target:
        return None
    whole = len(target) <= SHORT_VALUE_CHARS or target.isdigit()
    for turn in reversed(turns):
        joined, pos = _turn_index(turn)
        j = joined.find(target)
        while j >= 0:
            start, end = pos[j], pos[j + len(target) - 1] + 1
            if not whole or _whole_word(turn, start, end):
                return turn[start:end]
            j = joined.find(target, j + 1)
    return None


def anchor_value(value: str, kind: str, turns: List[str]) -> Optional[str]:
    """*value* as the agent's *turns* write it (``anchor``), or None.  A
    rest value "<member>: <value>" (``REST_PAIR``) is anchored part by
    part, since the agent need not write the two together."""
    if kind == REST and REST_PAIR in value:
        member, val = (anchor(part.strip(), turns) for part in value.split(REST_PAIR, 1))
        return f"{member}{REST_PAIR}{val}" if member is not None and val is not None else None
    return anchor(value, turns)


# Notes of an observation record (``BeliefState.apply``); an empty note:
# the observation entered the belief.
NOTE_UNKNOWN_CRITERION = "unknown criterion id"
NOTE_NOT_NEW_PASSAGE = "no new passage of the call has this label"
NOTE_NO_DOC_KIND = "the criterion is computed: no document observation"
NOTE_NO_SPAN = "no verified span"
NOTE_NOT_IN_TURN = "value not in the agent's turn"
NOTE_REPEATED = "repeated observation"

# Issues of a nugget update record (``BeliefState.apply``).
ISSUE_UNKNOWN_CRITERION = "unknown_criterion"
ISSUE_NO_NEW_OBSERVATION = "no_new_observation"
ISSUE_DUPLICATE_UPDATE = "duplicate_update"
ISSUE_NOT_IN_REPLY = "not_in_reply"
ISSUE_UNKNOWN_VALUE = "unknown_value"
ISSUE_NO_STATUS = "no_status"
ISSUE_REPEATED_VALUE = "repeated_value"
ISSUE_OVER_CAP = "over_cap"
ISSUE_UNKNOWN_EVIDENCE = "unknown_evidence"
ISSUE_UNOBSERVED_VALUE = "unobserved_value"
ISSUE_DROPPED_VALUE = "dropped_value"
ISSUE_NOTES = {
    ISSUE_UNKNOWN_CRITERION: "unknown criterion id",
    ISSUE_NO_NEW_OBSERVATION: "criterion without a new observation; ignored",
    ISSUE_DUPLICATE_UPDATE: "duplicate update",
    ISSUE_NOT_IN_REPLY: "not in the reply; nuggets kept, new observations left out",
}


class BeliefState:
    """Belief state b_t: per criterion, its nuggets (the answers, each with
    a status and its evidence, ``types.Nugget``).  It is self-contained: the
    nuggets and their evidence (each source's own observation, with a
    document's verified spans) are all that the next update reads of it.

    Each step, the belief update writes the observations of the agent's
    turn and of the step's new documents, then the new nuggets of the
    criteria with a new observation; ``apply`` keeps what is valid:

    - an observation enters only when it is grounded (a document
      observation with a verified span, a think observation whose value
      occurs in its turn), names a known criterion and a stance, is not a
      document observation on a criterion no document states
      (``NO_DOC_KINDS``), and is not a repeat (same source, stance and
      value as an observation already in the belief or in this step);
    - the criteria with a new observation take their new nuggets; the
      other criteria keep theirs.  A nugget is kept only when its value is
      the value of one of the criterion's current nuggets or new
      observations, its status is valid and its value is not repeated, at
      most ``MAX_NUGGETS[kind]`` per criterion.  Its evidence is, per label
      it cites, that source's observations on the nugget's value (else all
      of that source's observations on the criterion: a source that points
      against it), plus every source with an observation on its value.  A
      current nugget the reply leaves out (or writes invalidly) is carried
      over unchanged, with this step's sources on its value, so a faulty
      reply never erases an answer (noted as ``dropped_value``).

    Documents get their label ("d<n>") when they are shown
    (``label_doc``).  Criteria are independent: the nuggets of a criterion
    never refer to another criterion.
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self._index = {c.id: k for k, c in enumerate(self.criteria)}
        self.nuggets: List[List[Nugget]] = [[] for _ in self.criteria]
        self._doc_labels: Dict[str, str] = {}
        self._titles: Dict[str, str] = {}

    def index(self, criterion_id: str) -> Optional[int]:
        return self._index.get(criterion_id)

    def label_doc(self, key: str, title: str) -> str:
        """The label of the document *key* (its id, or a key of its text),
        new ("d<n>", next number) when it was not shown before."""
        label = self._doc_labels.get(key)
        if label is None:
            label = f"d{len(self._doc_labels) + 1}"
            self._doc_labels[key] = label
            self._titles[label] = title
        return label

    def doc_titles(self) -> Dict[str, str]:
        """Per document label, the title line of the document."""
        return dict(self._titles)

    def doc_keys(self) -> Dict[str, str]:
        """Per document key (its id, or a key of its text), its label."""
        return dict(self._doc_labels)

    def entries(self, k: int) -> List[Evidence]:
        """The distinct evidence of criterion *k*'s nuggets."""
        out: List[Evidence] = []
        seen: Set[Tuple[str, str, str]] = set()
        for n in self.nuggets[k]:
            for e in n.evidence:
                key = _entry_key(e)
                if key not in seen:
                    seen.add(key)
                    out.append(e)
        return out

    def apply(self, think: List[Observation], docs: List[Observation],
              updates: List[NuggetUpdate]) -> Dict[str, List[Dict[str, Any]]]:
        """Apply one belief update: the observations of the turn (*think*)
        and of the documents (*docs*), then the nugget lists (*updates*).
        Returns ``{think_observations, doc_observations, nugget_updates}``:
        one record per observation as written (``Observation.to_dict`` with
        ``grounded`` and ``note``, empty when it entered the belief; the
        notes are the ``NOTE_*`` constants), then
        one ``{id, written, after, reason, note, issues, [carried]}`` per criterion with a new
        observation (``written`` the list as written, ``after`` as kept,
        ``carried`` the values of the current nuggets carried over unchanged;
        such a criterion the reply left out keeps its nuggets, and its new
        observations stay out of the belief), then one per update of an
        unknown or unchanged criterion, or a repeated update.  ``issues``
        lists what was wrong with the criterion's update (``ISSUE_*``
        codes; ``note`` says it in words): also a value of the current
        belief or a new observation left without a nugget."""
        new: List[List[Evidence]] = [[] for _ in self.criteria]
        # Per criterion, the keys of its evidence and of this step's new
        # entries (built when first needed), for the repeat check.
        seen: Dict[int, Set[Tuple[str, str, str]]] = {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for key, observations in (("think_observations", think), ("doc_observations", docs)):
            records = []
            for o in observations:
                k = self._index.get(o.criterion)
                note = ""
                if k is None:
                    note = NOTE_UNKNOWN_CRITERION
                elif o.source == DOC and o.doc_id is None:
                    note = NOTE_NOT_NEW_PASSAGE
                elif o.source == DOC and self.criteria[k].kind in NO_DOC_KINDS:
                    note = NOTE_NO_DOC_KIND
                elif not o.grounded:
                    note = NOTE_NO_SPAN if o.source == DOC else NOTE_NOT_IN_TURN
                else:
                    e = o.evidence()
                    if k not in seen:
                        seen[k] = {_entry_key(f) for f in self.entries(k)}
                    if _entry_key(e) in seen[k]:
                        note = NOTE_REPEATED
                    else:
                        seen[k].add(_entry_key(e))
                        new[k].append(e)
                records.append({**o.to_dict(), "grounded": o.grounded, "note": note})
            out[key] = records
        changed = [k for k in range(len(self.criteria)) if new[k]]

        first: Dict[int, NuggetUpdate] = {}
        extra: List[Dict[str, Any]] = []
        for u in updates:
            k = self._index.get(u.id)
            if k is None or k not in changed or k in first:
                issue = (ISSUE_UNKNOWN_CRITERION if k is None else
                         ISSUE_NO_NEW_OBSERVATION if k not in changed else ISSUE_DUPLICATE_UPDATE)
                extra.append({"id": u.id, "written": [n.to_dict() for n in u.nuggets], "after": None,
                              "reason": u.reason, "note": ISSUE_NOTES[issue], "issues": [issue]})
                continue
            first[k] = u

        records: List[Dict[str, Any]] = []
        for k in changed:
            c = self.criteria[k]
            u = first.get(k)
            if u is None:
                records.append({"id": c.id, "written": None, "after": [n.to_dict() for n in self.nuggets[k]],
                                "reason": "", "note": ISSUE_NOTES[ISSUE_NOT_IN_REPLY],
                                "issues": [ISSUE_NOT_IN_REPLY]})
                continue
            pool = self.entries(k) + new[k]
            known = [n.value for n in self.nuggets[k]] + [e.value for e in new[k]]
            labels = {e.label for e in pool}
            cap = MAX_NUGGETS.get(c.kind, 0)
            kept: List[Nugget] = []
            notes: List[str] = []
            issues: List[str] = []
            for n in u.nuggets:
                if not any(same_value(v, n.value) for v in known):
                    notes.append(f"{n.value!r} is no current or observed value; dropped")
                    issues.append(ISSUE_UNKNOWN_VALUE)
                    continue
                if n.status is None:
                    notes.append(f"{n.value!r} has no valid status; dropped")
                    issues.append(ISSUE_NO_STATUS)
                    continue
                if any(same_value(n.value, m.value) for m in kept):
                    notes.append(f"{n.value!r} repeated; dropped")
                    issues.append(ISSUE_REPEATED_VALUE)
                    continue
                if len(kept) >= cap:
                    notes.append(f"{n.value!r} over the cap of {cap}; dropped")
                    issues.append(ISSUE_OVER_CAP)
                    continue
                unknown = [e for e in n.evidence if e not in labels]
                if unknown:
                    notes.append(f"{n.value!r}: unknown evidence {unknown} dropped")
                    issues.append(ISSUE_UNKNOWN_EVIDENCE)
                chosen = set()
                for label in dict.fromkeys(e for e in n.evidence if e in labels):
                    own = [i for i, e in enumerate(pool) if e.label == label and same_value(e.value, n.value)]
                    chosen.update(own or (i for i, e in enumerate(pool) if e.label == label))
                chosen.update(i for i, e in enumerate(pool) if same_value(e.value, n.value))
                kept.append(Nugget(value=n.value, status=n.status, evidence=[pool[i] for i in sorted(chosen)]))
            # A value is kept when a nugget has it, or cites a source on it
            # (values that name one answer in other words share a nugget).
            # A full list (at the cap) may leave the least likely values out.
            kept_keys = {alnum(v) for n in kept for v in [n.value] + [e.value for e in n.evidence]}
            full = len(kept) >= cap
            missing = {} if full else {alnum(e.value): e.value for e in new[k] if alnum(e.value) not in kept_keys}
            if missing:
                notes.append(f"observed values without a nugget: {list(missing.values())}")
                issues.append(ISSUE_UNOBSERVED_VALUE)
            dropped = [] if full else [n for n in self.nuggets[k] if alnum(n.value) not in kept_keys]
            carried: List[str] = []
            if dropped:
                notes.append(f"values of the current belief without a nugget: {[n.value for n in dropped]}; "
                             "carried over unchanged")
                issues.append(ISSUE_DROPPED_VALUE)
                # The reply never removes an answer: a nugget it left out keeps
                # its status and evidence (with this step's sources on its value).
                for n in dropped[:cap - len(kept)]:
                    seen = {_entry_key(e) for e in n.evidence}
                    extra_evidence = [e for e in new[k] if same_value(e.value, n.value) and _entry_key(e) not in seen]
                    kept.append(Nugget(value=n.value, status=n.status, evidence=list(n.evidence) + extra_evidence))
                    carried.append(n.value)
            self.nuggets[k] = kept
            record = {"id": c.id, "written": [n.to_dict() for n in u.nuggets],
                      "after": [n.to_dict() for n in kept], "reason": u.reason, "note": "; ".join(notes),
                      "issues": list(dict.fromkeys(issues))}
            if carried:
                record["carried"] = carried
            records.append(record)
        out["nugget_updates"] = records + extra
        return out

    def nugget_lists(self) -> List[List[Dict[str, Any]]]:
        """Per criterion, its nuggets (``{value, status, evidence}``)."""
        return [[n.to_dict() for n in nuggets] for nuggets in self.nuggets]

    def sources(self) -> List[int]:
        """Per criterion, its distinct documents that support one of its nuggets' values."""
        return [len({e.label for n in nuggets for e in n.evidence
                     if e.label.startswith("d") and e.stance == SUPPORTS})
                for nuggets in self.nuggets]


def _entry_key(e: Evidence) -> Tuple[str, str, str]:
    """What makes an evidence entry one source's one observation: its label,
    stance and value (letters and digits)."""
    return e.label, e.stance, alnum(e.value)
