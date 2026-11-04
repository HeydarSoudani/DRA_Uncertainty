"""Criteria and belief state of the belief model.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K}; the belief state b_t gives each
criterion a status from Agent-BRACE's seven-level belief scale
(``types.BELIEF_STATUSES``).  Every step the belief updater updates the
criteria that the step's new documents support, contradict or relate to (a
status can move up or down); the other criteria keep their status.

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
- ``BeliefState``: b_t and the evidence attached to each criterion.

Each criterion is closed (one fact) or, in a report request, an aspect of
the topic.  A set query lists its members as closed criteria, then a rest
criterion, "any other member: <property>", for the members the extractor
did not list, then an aggregate criterion, how the answer follows from the
members' values, as in the gold criteria of the training pools.  The kind
follows from the dataset's query shape and the criterion's position
(``_kinds``); the extractor only lists the criteria.

The belief updater and the query scorer are in ``judges``.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.layout import CRITERIA_SOURCES, GOLD_CRITERIA, SILVER_CRITERIA
from utils.text_utils import parse_json_object
from .prompts import CRITERIA_INIT_SYSTEMS, criteria_init_user
from .types import (
    AGGREGATE, ASPECT, CLOSED, MAX_CANDIDATES, MULTI_ASPECT, QUERY_SHAPES, REASONING_KINDS_EXCLUDED, REST,
    REST_PAIR, SET, STATUSLESS_KINDS, Candidate, CandidateUpdate, Criterion, CriterionUpdate, Evidence,
    candidate_probability, initial_status,
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

    @staticmethod
    def target(criteria: List[Criterion]) -> str:
        """The reply that parses to *criteria*, in the format the prompt asks
        for (the SFT target of belief model training)."""
        return json.dumps({"criteria": [c.text for c in criteria]}, ensure_ascii=False)

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

def _alnum(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


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
    target = _alnum(value)
    if not target:
        return None
    whole = len(target) <= SHORT_VALUE_CHARS or target.isdigit()
    for turn in reversed(turns):
        keys, pos = [], []
        for i, ch in enumerate(turn):
            for k in ch.lower():
                if k.isalnum():
                    keys.append(k)
                    pos.append(i)
        joined = "".join(keys)
        j = joined.find(target)
        while j >= 0:
            start, end = pos[j], pos[j + len(target) - 1] + 1
            if not whole or _whole_word(turn, start, end):
                return turn[start:end]
            j = joined.find(target, j + 1)
    return None


def anchor_candidate(value: str, kind: str, turns: List[str]) -> Optional[str]:
    """The candidate *value* as the agent's turns write it (``anchor``), or
    None.  A rest candidate "<member>: <value>" (``REST_PAIR``) is anchored
    part by part, since the agent need not write the two together."""
    if kind == REST and REST_PAIR in value:
        member, val = (anchor(part.strip(), turns) for part in value.split(REST_PAIR, 1))
        return f"{member}{REST_PAIR}{val}" if member is not None and val is not None else None
    return anchor(value, turns)


class BeliefState:
    """Belief state b_t: per criterion, its belief status and its evidence
    (retrieval), and its answer candidates (reasoner).

    b_0 gives every criterion ``INITIAL_STATUS`` (an aggregate criterion,
    ``STATUSLESS_KINDS``, has no status: None) and no candidate.  Each
    step the belief updater updates the criteria that the step's new
    passages support, contradict or relate to, from the current belief and
    those passages (Agent-BRACE style), and ``update`` takes the statuses as
    written:

    - the updater lists only the criteria the step's passages support,
      contradict or relate to; each listed criterion takes the status it gave;
    - a criterion the updater left out, or gave no valid status, keeps its
      status (``note`` says why);
    - an update of a criterion without a status is ignored, its cited
      passages too (``note``);
    - a cited passage is attached only with at least one span that occurs
      in it verbatim (``Evidence.verified_spans``); the others are dropped.
      This guards the evidence only, never the status;
    - a passage already attached to the criterion with the same role is not
      attached again.

    The updater sees the attached evidence in later steps.

    Answer candidates: the same call reads the agent's turn of the step
    (``read_turn``) and rewrites the candidate list of each criterion whose
    candidates the turn changes (``revise``): the answers the agent
    currently considers for it, from its memory or from the passages, each
    with a status for how likely it is correct.  A candidate is kept only
    when its value occurs in an agent turn read so far (``anchor_candidate``)
    and it
    has a valid status, at most ``MAX_CANDIDATES[kind]`` per criterion; the
    other criteria keep their list.
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self.statuses: List[Optional[str]] = [initial_status(c.kind) for c in self.criteria]
        self._evidence: List[List[Evidence]] = [[] for _ in self.criteria]
        self._index = {c.id: k for k, c in enumerate(self.criteria)}
        self.candidates: List[List[Candidate]] = [[] for _ in self.criteria]
        # The agent turns read so far, in order (the candidates' anchors).
        self.turns: List[str] = []

    def attached(self, k: int, limit: Optional[int] = None) -> List[Evidence]:
        """Evidence of criterion *k*, oldest first; the last *limit* when set."""
        items = self._evidence[k]
        return items[-limit:] if limit else list(items)

    def update(self, updates: List[CriterionUpdate]) -> List[Dict[str, Any]]:
        """Apply the statuses the updater wrote for one step.

        Returns one record per criterion, in criteria order, then one per
        update with an unknown or repeated id: ``{id, from, to, written,
        support, contradict, reason, note}``.  ``written`` is the status as
        the updater wrote it (null when left out); ``from`` and ``to`` are
        null for a criterion without a status; ``support`` and
        ``contradict`` are the passages as cited, with ``span_verified``.
        Only the first update of a criterion is used.
        """
        first: Dict[int, CriterionUpdate] = {}
        extra: List[Dict[str, Any]] = []
        for u in updates:
            k = self._index.get(u.id)
            if k is None or k in first:
                extra.append({
                    "id": u.id, "from": None, "to": None, "written": u.status,
                    "support": [e.to_dict() for e in u.support],
                    "contradict": [e.to_dict() for e in u.contradict],
                    "reason": u.reason,
                    "note": "unknown criterion id" if k is None else "duplicate update",
                })
                continue
            first[k] = u

        records: List[Dict[str, Any]] = []
        for k, c in enumerate(self.criteria):
            u = first.get(k)
            record = {
                "id": c.id, "from": self.statuses[k], "to": self.statuses[k], "written": None,
                "support": [], "contradict": [], "reason": "", "note": "",
            }
            records.append(record)
            if u is None:
                record["note"] = "not updated; status kept"
                continue
            if c.kind in STATUSLESS_KINDS:
                record.update(written=u.status, reason=u.reason,
                              note=f"{c.kind} criterion has no status; update ignored")
                continue
            record.update(
                written=u.status, reason=u.reason,
                support=[e.to_dict() for e in u.support],
                contradict=[e.to_dict() for e in u.contradict],
            )
            notes = []
            cited = u.support + u.contradict
            verified = [e for e in cited if e.verified_spans]
            if len(cited) > len(verified):
                notes.append(f"{len(cited) - len(verified)} cited passages without a verified span dropped")
            known = {(e.doc_id, e.role) for e in self._evidence[k]}
            for e in verified:
                if (e.doc_id, e.role) not in known:
                    self._evidence[k].append(e)
                    known.add((e.doc_id, e.role))
            if u.status is None:
                notes.append("no valid status; status kept")
            else:
                self.statuses[k] = u.status
                record["to"] = u.status
            record["note"] = "; ".join(notes)
        return records + extra

    def read_turn(self, turn: Optional[str]) -> None:
        """Add the agent's turn of the step to the candidates' anchors."""
        if turn and turn.strip():
            self.turns.append(turn)

    def revise(self, updates: List[CandidateUpdate]) -> List[Dict[str, Any]]:
        """Apply the candidate lists the updater wrote for one step.

        Returns one record per listed criterion, then one per update with
        an unknown or repeated id: ``{id, before, after, written, reason,
        note}``; ``before`` / ``after`` are the lists as kept, ``written``
        the list as written, each candidate with ``anchored``.  A written
        candidate is dropped (``note``) when its value occurs in no agent
        turn read so far, when it has no valid status, when it repeats an
        earlier value of the list, or beyond the cap of the criterion's kind.
        Only the first update of a criterion is used.
        """
        records: List[Dict[str, Any]] = []
        seen = set()
        for u in updates:
            k = self._index.get(u.id)
            kind = self.criteria[k].kind if k is not None else CLOSED
            written = []
            for c in u.candidates:
                matched = anchor_candidate(c.value, kind, self.turns)
                c.anchored = matched is not None
                c.value = matched or c.value
                written.append({**c.to_dict(), "anchored": c.anchored})
            record = {"id": u.id, "before": None, "after": None, "written": written,
                      "reason": u.reason, "note": ""}
            records.append(record)
            if k is None or k in seen:
                record["note"] = "unknown criterion id" if k is None else "duplicate update"
                continue
            seen.add(k)
            cap = MAX_CANDIDATES.get(self.criteria[k].kind, 0)
            kept: List[Candidate] = []
            notes = []
            for c in u.candidates:
                if not c.anchored:
                    notes.append(f"{c.value!r} not in the agent's turns; dropped")
                elif c.status is None:
                    notes.append(f"{c.value!r} has no valid status; dropped")
                elif any(_alnum(c.value) == _alnum(o.value) for o in kept):
                    notes.append(f"{c.value!r} repeated; dropped")
                elif len(kept) >= cap:
                    notes.append(f"{c.value!r} over the cap of {cap}; dropped")
                else:
                    kept.append(c)
            record["before"] = [c.to_dict() for c in self.candidates[k]]
            record["after"] = [c.to_dict() for c in kept]
            record["note"] = "; ".join(notes)
            self.candidates[k] = kept
        return records

    def candidate_lists(self) -> List[List[Dict[str, Any]]]:
        """Per criterion, its candidates (``{value, status}``)."""
        return [[c.to_dict() for c in cands] for cands in self.candidates]

    def reasoning_confidence(self) -> float:
        """R: the mean over the criteria of the probability of their most
        likely candidate (``reasoning_confidence``)."""
        return reasoning_confidence(self.candidate_lists(), [c.kind for c in self.criteria])

    def sources(self) -> List[int]:
        """Per criterion, its distinct supporting documents."""
        return [self.num_sources(k) for k in range(len(self.criteria))]

    def num_sources(self, k: int) -> int:
        """Distinct documents among the supporting evidence of criterion *k*."""
        return len({e.doc_id for e in self._evidence[k] if e.role == "support"})

    def evidence(self) -> List[Dict[str, Any]]:
        """Per criterion, its kind, status and its attached evidence (without
        text)."""
        return [
            {"id": c.id, "kind": c.kind, "status": status, "evidence": [e.to_dict() for e in ev]}
            for c, status, ev in zip(self.criteria, self.statuses, self._evidence)
        ]


def reasoning_confidence(candidates: List[List[Dict[str, Any]]], kinds: List[str]) -> float:
    """R of per-criterion candidate lists (``BeliefState.candidate_lists``)
    of criteria of *kinds*: the mean over the criteria whose kind is not in
    ``REASONING_KINDS_EXCLUDED`` of their top candidate probability, 0 for a
    criterion without candidates and for no such criterion."""
    tops = [max((candidate_probability(c["status"]) for c in cands), default=0.0)
            for cands, kind in zip(candidates, kinds) if kind not in REASONING_KINDS_EXCLUDED]
    return sum(tops) / len(tops) if tops else 0.0
