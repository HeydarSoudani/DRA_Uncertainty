"""Criteria and belief state of the belief model.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K}; the belief state b_t gives each
criterion its candidates (possible answers), each with a status from
Agent-BRACE's belief scale (``types.BELIEF_STATUSES``) for how likely it is
an answer, and its basis (passage or reasoning).

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
- ``BeliefState``: b_t, per criterion its candidates (shown), and the trace,
  references and candidate updates behind them (saved).

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.layout import CRITERIA_SOURCES, GOLD_CRITERIA, SILVER_CRITERIA
from utils.text_utils import parse_json_object
from .prompts import CRITERIA_INIT_SYSTEMS, criteria_init_user
from .reason_text import parse_reason, reason_labels
from .types import (
    AGGREGATE, ASPECT, CLOSED, DOC, MULTI_ASPECT, NO_DOC_KINDS, PASSAGE, QUERY_SHAPES, REASONING, REST, SET,
    REMOVED, STATUS_LEVEL, THINK, TURN_ONLY_MAX_STATUS, TURN_ONLY_MIN_STATUS, UNKNOWN, Candidate, CandidateUpdate,
    Criterion, Observation, RemovedCandidate, alnum, contains_words, exclusive, is_removed, parse_status,
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

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        info: Dict[str, Any] = {"model": GOLD_CRITERIA_MODEL, "query_shape": self._shape, "criteria_kind": GOLD_CRITERIA,
                                "errors": []}
        reason = self.missing(query_id, query)
        if reason is not None:
            return [], {**info, "errors": [f"gold criteria: {reason}"]}
        row = self._rows[str(query_id)]
        return list(row["criteria"]), {**info, "gold_level": row.get("level")}


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

# Issues of one belief update (``BeliefState.apply``): one record
# ``{id, code, detail, [label]}`` per thing the reply got wrong.
# An observation that did not enter the belief:
ISSUE_UNKNOWN_CRITERION = "unknown_criterion"      # also a candidate update of an unknown criterion
ISSUE_NOT_NEW_PASSAGE = "not_new_passage"          # its label is no new passage of the call
ISSUE_COMPUTED = "computed_criterion"              # a document observation on a computed criterion
ISSUE_NO_SPAN = "no_span"                          # a document observation without a verified span
ISSUE_NOT_IN_TURN = "not_in_turn"                  # a think observation without a span of the turn
ISSUE_REPEATED = "repeated_observation"            # a source already observed on the criterion
# A candidate update that is not taken:
ISSUE_EMPTY_CANDIDATE = "empty_candidate"          # no candidate name
ISSUE_BAD_STATUS = "bad_status"                    # no status of the scale nor "removed", or "unknown"
ISSUE_DUPLICATE_CANDIDATE = "duplicate_candidate"  # a second update of one candidate in the reply
ISSUE_UNEXPECTED_CANDIDATE = "unexpected_candidate"  # a criterion without a new observation
ISSUE_REMOVE_UNLISTED = "remove_unlisted"          # "removed" on a candidate that is not listed
ISSUE_BAD_REMOVAL = "bad_removal"                  # "removed" without the observation that can remove it: kept
# A candidate update that is taken, with a flaw in its reason or name:
ISSUE_NO_REASON = "no_reason_opening"              # the reason opens with neither "Was <status>:" nor "New:"
ISSUE_WRONG_PREVIOUS = "wrong_previous"            # "Was <X>:" with X not the candidate's current status
ISSUE_NOT_LISTED = "not_listed"                    # "Was <X>:" on a candidate that is not listed (added)
ISSUE_LISTED_AS_NEW = "listed_as_new"              # "New:" on a listed candidate (its status updated)
ISSUE_BAD_REF = "bad_ref"                          # a cited source with no observation on the criterion
ISSUE_UNKNOWN_CANDIDATE = "unknown_candidate"      # a new candidate in no span or title of the criterion
ISSUE_NEAR_DUPLICATE = "near_duplicate"            # a new candidate whose name contains a listed one, or the reverse
ISSUE_UNCHANGED = "unchanged"                      # the status written is the current one
ISSUE_READDED = "readded"                          # a new candidate the update removed before (information only)
# A status code corrected:
ISSUE_TURN_ONLY = "turn_only"                      # a reasoning-only candidate, or one new in a step with no new passage, outside "unlikely".."possible": moved into it
ISSUE_TURN_ONLY_CHANGE = "turn_only_change"        # a listed passage candidate changed with no new passage (nor, one answer, a passage rival change): kept
# The belief after the reply:
ISSUE_COMPETING = "competing_candidates"           # a one-answer criterion with more than one candidate above "possible"

# Observations that did not enter for a reason of their own (the reply
# should not have written them); a repeat is harmless.
REJECTED_OBSERVATION = (ISSUE_COMPUTED, ISSUE_NO_SPAN, ISSUE_NOT_IN_TURN)
# A candidate update whose status code corrected (the reply's status is
# not the one kept).
CORRECTED_STATUS = (ISSUE_TURN_ONLY, ISSUE_TURN_ONLY_CHANGE)
# A reply with one of these is not used as an SFT sample (with
# ``require_verified_spans``): an observation left out for a reason of its
# own, a status code corrected, a reason that misreads the belief or cites
# a source with no observation on the criterion, a removal code did not
# take, or a one-answer criterion left with competing candidates.
# ``unknown_candidate`` is not among them: a candidate the reply writes in
# other words than its spans (an initial, a computed value) raises it too;
# nor are ``near_duplicate``, ``listed_as_new``, ``unchanged`` and
# ``readded`` (harmless for the belief).
UNCLEAN_REPLY = (*REJECTED_OBSERVATION, *CORRECTED_STATUS, ISSUE_NO_REASON, ISSUE_WRONG_PREVIOUS,
                 ISSUE_NOT_LISTED, ISSUE_BAD_REF, ISSUE_REMOVE_UNLISTED, ISSUE_BAD_REMOVAL, ISSUE_COMPETING)
# A reply with one of these fails the format gate of training: it names
# what the call did not show, or writes an update code cannot take.
GATE_ISSUES = (ISSUE_UNKNOWN_CRITERION, ISSUE_NOT_NEW_PASSAGE, ISSUE_EMPTY_CANDIDATE, ISSUE_BAD_STATUS,
               ISSUE_DUPLICATE_CANDIDATE)


def _issue(cid: str, code: str, detail: str, o: Optional[Observation] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": cid, "code": code, "detail": detail}
    if o is not None:
        out["label"] = o.label
    return out


def _level(status: str) -> int:
    return STATUS_LEVEL[status]


class BeliefState:
    """Belief state b_t, kept at two levels.

    Shown: per criterion its candidates, each with its status and basis
    (``types.Candidate``; ``belief_state.format_belief``), the whole belief
    the next update and the agent read of it.

    Saved: per criterion the trace of the observations that entered it
    (``trace``) and its references (``reference_lists``): every source ever
    assigned to it, with its document, title, the step it came in and its
    verified spans; and per step the candidate updates with their reasons
    (``apply``), never shown again; and per criterion the candidates the
    update removed (``removed``, ``types.RemovedCandidate``).

    Each step, the belief update writes the observations of the agent's
    turn and of the step's new documents, then the candidates that are new,
    removed or whose status changes, each with its reason; ``apply`` keeps
    what is valid:

    - an observation enters only when it is grounded (a verified span of
      its passage or turn), names a known criterion, is not a document
      observation on a criterion no document states (``NO_DOC_KINDS``),
      and its source is not observed on the criterion yet;
    - a candidate update is taken only for a criterion with a new
      observation, once per candidate, with a status of the scale other
      than "unknown", or "removed"; a name not listed (letters and digits
      compared) adds the candidate, a listed one changes its status.
      Candidates the reply does not name keep their status.  The reason is
      checked (its opening against the current status, its labels against
      the trace) and flagged, never used to reject;
    - "removed" takes a listed candidate out of the list (into ``removed``)
      when the step has the observation that can rule it out: a new passage
      observation on the criterion that the reason cites, or, for a
      candidate only reasoning names ("reasoning" basis, or a computed
      criterion), a new reasoning observation on it.  Otherwise it is
      flagged and the candidate kept;
    - the basis of a candidate is "passage" when a passage observed on the
      criterion names it (a span or the title, as whole words) or the
      reason cites such a passage, else "reasoning"; once "passage" it
      stays.  Passages set the status, reasoning only names or rejects
      candidates (a computed criterion is exempt): a "reasoning" candidate,
      and a new one in a step with no new passage observation on the
      criterion, is moved into "unlikely" .. "possible"
      (``TURN_ONLY_MIN_STATUS`` .. ``TURN_ONLY_MAX_STATUS``) when the reply
      wrote a status outside it; in a step with no new passage observation
      on the criterion, a listed "passage" candidate keeps its status, up
      or down (a passage of an earlier step does not move a status in a
      later one), unless the criterion has one answer and another of its
      "passage" candidates is added, removed or changed in the same step
      (its status depends on the other candidates; a "reasoning" one meets
      no part of the criterion, so it moves none).
    - the candidates of a one-answer criterion compete: at most one is above
      "possible" (only one can meet more parts than every other); more are
      flagged (``competing_candidates``), not corrected: code cannot tell
      which one the passages favour.

    Derived per criterion (``belief_lists``): its status, the highest of
    its candidates ("unknown" without candidates); its selected
    candidate(s), among those at "possible" or above: for a one-answer
    criterion the first one shown (``ordered``), for a several-answers
    criterion every one.

    Documents get their label ("d<n>") when they are shown
    (``label_doc``).
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self._index = {c.id: k for k, c in enumerate(self.criteria)}
        # Per criterion its candidates, in the order they were added.
        self.candidates: List[List[Candidate]] = [[] for _ in self.criteria]
        # The trace: per criterion, the observations that entered it.
        self.trace: List[List[Observation]] = [[] for _ in self.criteria]
        # Per criterion the candidates the update removed, in order.
        self.removed: List[List[RemovedCandidate]] = [[] for _ in self.criteria]
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

    def doc_label(self, key: str) -> Optional[str]:
        """The label of the document *key*, None when it was never shown."""
        return self._doc_labels.get(key)

    def is_doc_label(self, label: str) -> bool:
        """Whether *label* names a document shown to the update ("d<n>")."""
        return label in self._titles

    # ------------------------------------------------------------------
    # Derived per criterion
    # ------------------------------------------------------------------

    def ordered(self, k: int) -> List[Candidate]:
        """The candidates of criterion *k* as shown: highest status first,
        then "passage" before "reasoning", then in the order they were
        added (so a one-answer criterion's selected candidate is first)."""
        return sorted(self.candidates[k], key=lambda c: (-_level(c.status), c.basis != PASSAGE))

    def status(self, k: int) -> str:
        """The status of criterion *k*: the highest of its candidates,
        "unknown" without candidates."""
        return max((c.status for c in self.candidates[k]), key=_level, default=UNKNOWN)

    def selected(self, k: int) -> List[str]:
        """The selected candidate(s) of criterion *k*, among those at
        "possible" or above (a ruled-out candidate is never selected): for
        a one-answer criterion the first one shown (``ordered``: highest
        status, then "passage" before "reasoning", then the earliest
        added); for a several-answers criterion every one."""
        kept = [c.name for c in self.ordered(k) if _level(c.status) >= _level(TURN_ONLY_MAX_STATUS)]
        return kept[:1] if exclusive(self.criteria[k].kind) else kept

    # ------------------------------------------------------------------
    # The update
    # ------------------------------------------------------------------

    def apply(self, think: List[Observation], docs: List[Observation],
              updates: List[CandidateUpdate], step: int) -> Dict[str, Any]:
        """Apply one belief update of step *step*: the observations of the
        turn (*think*) and of the documents (*docs*), then the candidate
        *updates*.

        Returns ``{observations, updates, changed, issues}``: the
        observations that entered the belief (``Observation.to_dict``), one
        record per candidate update of the reply (``{id, candidate, reason,
        written, previous, status}``: ``written`` the status the reply
        wrote ("removed" for a removal), ``previous`` the candidate's status
        before (null for a new one), ``status`` the one kept ("removed" when
        the candidate left the list), null when the update was not taken),
        the ids of the criteria with a candidate added, removed or whose
        status changed, and one issue record per thing the reply got wrong
        (the ``ISSUE_*`` codes)."""
        issues: List[Dict[str, Any]] = []
        new: List[List[Observation]] = [[] for _ in self.criteria]
        seen = [{o.label for o in obs} for obs in self.trace]
        entered: List[Observation] = []
        for o in [*think, *docs]:
            k = self._index.get(o.criterion)
            if k is None:
                issues.append(_issue(o.criterion, ISSUE_UNKNOWN_CRITERION, "unknown criterion id", o))
            elif o.source == DOC and o.doc_id is None:
                issues.append(_issue(o.criterion, ISSUE_NOT_NEW_PASSAGE, "no new passage of the call has this label", o))
            elif o.source == DOC and self.criteria[k].kind in NO_DOC_KINDS:
                issues.append(_issue(o.criterion, ISSUE_COMPUTED, "the criterion is computed: no document observation", o))
            elif not o.grounded:
                issues.append(_issue(o.criterion, ISSUE_NO_SPAN, "no verified span", o) if o.source == DOC else
                              _issue(o.criterion, ISSUE_NOT_IN_TURN, "no span of the agent's turn", o))
            elif o.label in seen[k]:
                issues.append(_issue(o.criterion, ISSUE_REPEATED, "the source is already observed on the criterion", o))
            else:
                seen[k].add(o.label)
                new[k].append(o)
                entered.append(o)
        for k, obs in enumerate(new):
            self.trace[k].extend(obs)

        records: List[Dict[str, Any]] = []
        done = set()
        changed = set()
        # Criteria with a "passage" candidate added, removed or changed: only
        # such a rival moves the status of the others (a reasoning-only one
        # meets no part of the criterion).
        passage_moved = set()
        # Status changes of listed passage candidates of one-answer criteria
        # in a step with no new passage on the criterion: taken only when
        # another passage candidate of the criterion changes in the step.
        deferred: List[Tuple[int, Candidate, str, Dict[str, Any]]] = []
        for u in updates:
            name = " ".join(str(u.candidate or "").split())
            rec: Dict[str, Any] = {"id": u.id, "candidate": name, "reason": u.reason, "written": u.status,
                                   "previous": None, "status": None}
            records.append(rec)
            k = self._index.get(u.id)
            if k is None:
                issues.append(_issue(u.id, ISSUE_UNKNOWN_CRITERION, f"candidate {name!r} of an unknown criterion id"))
                continue
            if not alnum(name):
                issues.append(_issue(u.id, ISSUE_EMPTY_CANDIDATE, "an update without a candidate name"))
                continue
            status = REMOVED if is_removed(u.status) else parse_status(u.status)
            if status is None or status == UNKNOWN:
                issues.append(_issue(u.id, ISSUE_BAD_STATUS, f"candidate {name!r}: status {u.status!r} is not a candidate status"))
                continue
            if (k, alnum(name)) in done:
                issues.append(_issue(u.id, ISSUE_DUPLICATE_CANDIDATE, f"a second update of {name!r}; ignored"))
                continue
            done.add((k, alnum(name)))
            if not new[k]:
                issues.append(_issue(u.id, ISSUE_UNEXPECTED_CANDIDATE, f"{name!r}: no new observation on the criterion; ignored"))
                continue
            rec["written"] = status
            if status == REMOVED:
                basis = self._remove_candidate(k, name, u.reason, step, new[k], rec, issues)
            else:
                basis = self._update_candidate(k, name, status, u.reason, step, new[k], rec, issues, deferred)
            if basis is not None:
                changed.add(k)
                if basis == PASSAGE:
                    passage_moved.add(k)

        for k, existing, status, rec in deferred:
            if k in passage_moved:
                existing.status = status
                rec["status"] = status
            else:
                issues.append(_issue(self.criteria[k].id, ISSUE_TURN_ONLY_CHANGE,
                                     f"{existing.name!r}: {status!r} from {existing.status!r} with no new passage on "
                                     f"the criterion and no other passage candidate changed; kept"))
                rec["status"] = existing.status

        for k in sorted(changed):
            if exclusive(self.criteria[k].kind):
                # Only one candidate can meet more parts than every other.
                above = [c.name for c in self.candidates[k] if _level(c.status) > _level(TURN_ONLY_MAX_STATUS)]
                if len(above) > 1:
                    issues.append(_issue(self.criteria[k].id, ISSUE_COMPETING,
                                         f"a one-answer criterion with {above} above {TURN_ONLY_MAX_STATUS!r}"))
        return {"observations": [o.to_dict() for o in entered], "updates": records,
                "changed": [self.criteria[k].id for k in sorted(changed)], "issues": issues}

    def _listed(self, k: int, name: str) -> Optional[Candidate]:
        """The listed candidate of criterion *k* named *name* (letters and
        digits compared), or None."""
        key = alnum(name)
        return next((x for x in self.candidates[k] if alnum(x.name) == key), None)

    def _check_reason(self, k: int, name: str, reason: str, existing: Optional[Candidate],
                      issues: List[Dict[str, Any]]) -> List[str]:
        """Flag what the *reason* of an update of candidate *name* of
        criterion *k* gets wrong (its opening against the current status, its
        labels against the trace); returns the labels it cites."""
        c = self.criteria[k]
        opening = parse_reason(reason)
        if opening is None:
            issues.append(_issue(c.id, ISSUE_NO_REASON, f"{name!r}: the reason opens with neither 'Was <status>:' nor 'New:'"))
        elif opening.new and existing is not None:
            issues.append(_issue(c.id, ISSUE_LISTED_AS_NEW, f"{name!r} is listed ({existing.status}); its status is updated"))
        elif not opening.new and existing is None:
            issues.append(_issue(c.id, ISSUE_NOT_LISTED, f"'Was' on {name!r}, which is not listed; it is added"))
        elif not opening.new and opening.previous != existing.status:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS,
                                 f"{name!r}: the reason says it was {opening.previous!r}, it is {existing.status!r}"))
        observed = {o.label for o in self.trace[k]}
        cited = reason_labels(reason)
        bad = [lb for lb in cited if lb not in observed]
        if bad:
            issues.append(_issue(c.id, ISSUE_BAD_REF, f"{name!r}: cited sources {bad} have no observation on the criterion"))
        return cited

    def _remove_candidate(self, k: int, name: str, reason: str, step: int, new_obs: List[Observation],
                          rec: Dict[str, Any], issues: List[Dict[str, Any]]) -> Optional[str]:
        """Remove candidate *name* of criterion *k* when the step has the
        observation that can rule it out (``apply``); fills *rec* and
        returns the removed candidate's basis, None when it is kept.  A
        computed criterion's candidates are all "reasoning"."""
        c = self.criteria[k]
        existing = self._listed(k, name)
        if existing is None:
            issues.append(_issue(c.id, ISSUE_REMOVE_UNLISTED, f"'removed' on {name!r}, which is not listed; ignored"))
            return None
        rec["previous"], rec["candidate"] = existing.status, existing.name
        cited = self._check_reason(k, existing.name, reason, existing, issues)
        new_passages = {o.label for o in new_obs if o.source == DOC}
        by_passage = any(lb in new_passages for lb in cited)
        by_reasoning = existing.basis != PASSAGE and any(o.source == THINK for o in new_obs)
        if not (by_passage or by_reasoning):
            why = ("cites no new passage on the criterion" if existing.basis == PASSAGE
                   else "cites no new passage and the turn has no observation on the criterion")
            issues.append(_issue(c.id, ISSUE_BAD_REMOVAL, f"{existing.name!r}: removed, but the reason {why}; kept"))
            rec["status"] = existing.status
            return None
        self.candidates[k].remove(existing)
        self.removed[k].append(RemovedCandidate(name=existing.name, status=existing.status, basis=existing.basis,
                                                added_step=existing.added_step, removed_step=step, reason=reason))
        rec["status"] = REMOVED
        return existing.basis

    def _update_candidate(self, k: int, name: str, status: str, reason: str, step: int,
                          new_obs: List[Observation], rec: Dict[str, Any], issues: List[Dict[str, Any]],
                          deferred: List[Tuple[int, Candidate, str, Dict[str, Any]]]) -> Optional[str]:
        """Add candidate *name* of criterion *k* with *status*, or change its
        status, checked and bounded (``apply``); fills *rec* and returns
        the candidate's basis when the belief changed, else None.  A change
        *apply* decides after the reply's other updates goes to
        *deferred*."""
        c = self.criteria[k]
        key = alnum(name)
        existing = self._listed(k, name)
        rec["previous"] = existing.status if existing is not None else None
        if existing is not None:
            rec["candidate"] = existing.name
        cited = self._check_reason(k, name, reason, existing, issues)
        observed = {o.label for o in self.trace[k]}
        if existing is None:
            if any(alnum(x.name) == key for x in self.removed[k]):
                issues.append(_issue(c.id, ISSUE_READDED, f"{name!r} was removed before; it is added again"))
            if self._unknown_candidate(k, name):
                issues.append(_issue(c.id, ISSUE_UNKNOWN_CANDIDATE, f"{name!r} in no span or title of the criterion"))
            near = [x.name for x in self.candidates[k] if contains_words(x.name, name) or contains_words(name, x.name)]
            if near:
                issues.append(_issue(c.id, ISSUE_NEAR_DUPLICATE, f"{name!r} is close to listed {near}"))
        elif status == existing.status:
            issues.append(_issue(c.id, ISSUE_UNCHANGED, f"{name!r}: the status written is its current one"))

        computed = c.kind in NO_DOC_KINDS
        if computed:
            basis = REASONING
        elif (existing is not None and existing.basis == PASSAGE) or self._named_by_passage(k, name) \
                or any(self.is_doc_label(lb) and lb in observed for lb in cited):
            basis = PASSAGE
        else:
            basis = REASONING
        if not computed:
            # Passages set the status; reasoning only names or rejects
            # candidates.  A step with no new passage on the criterion moves
            # no status of a passage candidate, whatever earlier passage the
            # reason cites.
            new_passage = any(o.source == DOC for o in new_obs)
            if existing is not None and basis == PASSAGE and not new_passage:
                if status != existing.status:
                    if exclusive(c.kind):
                        # Its status depends on the other candidates: taken
                        # when another passage candidate changes in the step.
                        deferred.append((k, existing, status, rec))
                        return None
                    issues.append(_issue(c.id, ISSUE_TURN_ONLY_CHANGE,
                                         f"{name!r}: {status!r} from {existing.status!r} with no new passage on "
                                         f"the criterion; kept"))
                    status = existing.status
            elif basis == REASONING or not new_passage:
                bounded = min(max(status, TURN_ONLY_MIN_STATUS, key=_level), TURN_ONLY_MAX_STATUS, key=_level)
                if bounded != status:
                    why = "no passage on it" if basis == REASONING else "no new passage on the criterion"
                    issues.append(_issue(c.id, ISSUE_TURN_ONLY, f"{name!r}: {status!r} with {why}; "
                                                                f"set to {bounded!r}"))
                    status = bounded

        rec["status"] = status
        if existing is None:
            self.candidates[k].append(Candidate(name=name, status=status, basis=basis, added_step=step))
            return basis
        moved = status != existing.status or basis != existing.basis
        existing.status, existing.basis = status, basis
        return basis if moved else None

    def _texts(self, k: int, passages_only: bool) -> List[str]:
        """The verified spans of the observations on criterion *k* (of its
        passages only, with *passages_only*), and the titles of its
        passages (a passage may name its subject in its title only)."""
        out: List[str] = []
        for o in self.trace[k]:
            if o.source == DOC:
                out += [*o.verified_spans, self._titles.get(o.label) or ""]
            elif not passages_only:
                out += o.verified_spans
        return out

    def _parts(self, k: int, name: str) -> List[str]:
        """The parts of a candidate a source writes out: a rest criterion's
        "<member>: <value>" the member and the value, else the name whole."""
        return name.split(":", 1) if self.criteria[k].kind == REST and ":" in name else [name]

    def _named_by_passage(self, k: int, name: str) -> bool:
        """Whether a passage observed on criterion *k* names candidate
        *name* as whole words (a part of it, for a rest criterion's
        "<member>: <value>")."""
        texts = self._texts(k, passages_only=True)
        return any(contains_words(t, p) for p in self._parts(k, name) for t in texts)

    def _unknown_candidate(self, k: int, name: str) -> bool:
        """Whether candidate *name* occurs in none of the verified spans of
        criterion *k* and in no title of a passage observed on it (a passage
        may name its subject in its title only).  Checked for the kinds
        whose answer is a name or figure a source writes out: a closed or
        aggregate criterion (the candidate whole), the rest criterion
        ("<member>: <value>": the member and the value each).  An aspect's
        answers are statements the update words itself, so they are not
        checked."""
        if self.criteria[k].kind == ASPECT:
            return False
        texts = self._texts(k, passages_only=False)
        return not all(any(contains_words(t, p) for t in texts) for p in self._parts(k, name) if alnum(p))

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    def belief_lists(self) -> List[Dict[str, Any]]:
        """Per criterion ``{id, status, selected, candidates}``: its derived
        status and selection, and its candidates as shown
        (``Candidate.to_dict``, highest status first)."""
        return [{"id": c.id, "status": self.status(k), "selected": self.selected(k),
                 "candidates": [x.to_dict() for x in self.ordered(k)]}
                for k, c in enumerate(self.criteria)]

    def reference_lists(self) -> List[Dict[str, Any]]:
        """Per criterion ``{id, references, removed}``: every source assigned
        to it so far, in the order it came in, as ``{label, step, spans}``
        plus ``doc_id`` and ``title`` for a document; and the candidates the
        update removed (``RemovedCandidate.to_dict``)."""
        out = []
        for k, (c, obs) in enumerate(zip(self.criteria, self.trace)):
            refs = []
            for o in obs:
                r: Dict[str, Any] = {"label": o.label, "step": o.step}
                if o.source == DOC:
                    r.update(doc_id=o.doc_id, title=self._titles.get(o.label))
                r["spans"] = o.verified_spans
                refs.append(r)
            out.append({"id": c.id, "references": refs, "removed": [x.to_dict() for x in self.removed[k]]})
        return out
