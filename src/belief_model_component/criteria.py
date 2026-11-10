"""Criteria and belief state of the belief model.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K}; the belief state b_t gives each
criterion its candidates (possible answers), each with a status from
Agent-BRACE's belief scale (``types.BELIEF_STATUSES``) for how likely it is
an answer and its support (the passage observations on it), and the names
the agent proposed that no passage supports yet.

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
- ``BeliefState``: b_t, per criterion its candidates with their support
  and its proposals (shown), and the trace, references and candidate
  updates behind them (saved).

Each criterion is closed (one fact), a hop of a multi-hop query before its
answer (which may have several answers), or, in a report request, an
aspect of the topic.  A multi-hop criterion names only its own hop and
refers to hops before it by their ids ("the town of c1"), never to a
later one (``forward_refs``).  A set
query lists its members as closed criteria, then a rest criterion, "any other member: <property>", for the members the extractor
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
    AGGREGATE, ASPECT, CLOSED, CRITERION_REF_RE, DOC, HOP, MULTI_ASPECT, MULTI_HOP, NAMED_KINDS, NO_DOC_KINDS,
    NO_SUPPORT_MAX_STATUS,
    QUERY_SHAPES, REASONING, REJECTED, REMOVED, REST, SET, STATUS_LEVEL, THINK, UNKNOWN, Candidate, CandidateUpdate,
    Criterion, Observation, Proposal, RemovedCandidate, Support, alnum, contains_words, exclusive, is_removed,
    parse_status,
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
    target is closed; a multi-hop query's hops before its answer are hops
    (several answers), its answer closed (``shape_kinds``); a set query's
    list (``_set_layout``) is its members (closed), the rest criterion and
    the aggregate; every report criterion is an aspect."""
    if shape == MULTI_ASPECT:
        return [ASPECT] * len(texts)
    if shape == SET and len(texts) >= 2:
        return [CLOSED] * (len(texts) - 2) + [REST, AGGREGATE]
    return shape_kinds(shape, [CLOSED] * len(texts))


def shape_kinds(shape: str, kinds: List[str]) -> List[str]:
    """The *kinds* of a *shape* query's criteria as the belief reads them: in
    a multi-hop query every closed criterion but the last (the answer) is a
    hop, which may have several answers; any other shape is unchanged.
    Gold rows record every hop as closed, so their kinds are mapped too."""
    if shape != MULTI_HOP:
        return list(kinds)
    return [HOP if kind == CLOSED and k < len(kinds) - 1 else kind for k, kind in enumerate(kinds)]


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

        # A salvaged reply is asked once more; salvaged again, the extraction
        # fails.  A reply whose criteria refer to a later one is asked once
        # more too, then kept (with its warning).
        for _ in range(2):
            criteria, info = self._inner.get(query_id, query)
            if not info.get("salvaged") and not info.get("bad_order"):
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
# answer and document (training_pool/annotations.py); belief model training
# scores only such queries.  The belief model uses the gold criteria of any
# level.
GOLD_CRITERIA_LEVEL = "full"


class GoldCriteriaSource(CriteriaSource):
    """Reads each query's gold criteria from a training pool's queries file
    (``layout.gold_criteria_path``, one row per query in the format of
    ``training_pool/annotations.py``, written by the pool build): per query,
    its criteria with their gold answers and documents.  Only the criteria
    (id, text, kind; a multi-hop query's hops as ``shape_kinds`` reads
    them) reach the belief model, of a query whose row has
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
                        kinds = shape_kinds(query_shape, [c["kind"] for c in criteria])
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
ISSUE_NOT_NEW_PASSAGE = "not_new_passage"          # its label is no passage of the call
ISSUE_COMPUTED = "computed_criterion"              # a passage observation on a computed criterion
ISSUE_NO_SPAN = "no_span"                          # a passage observation without a verified span
ISSUE_NOT_IN_TURN = "not_in_turn"                  # a reasoning observation without a span of its turn
ISSUE_BAD_CLAIM = "bad_claim"                      # a reasoning observation without a candidate or a valid claim
# A candidate update that is not taken:
ISSUE_EMPTY_CANDIDATE = "empty_candidate"          # no candidate name
ISSUE_BAD_STATUS = "bad_status"                    # no status of the scale nor "removed", or "unknown"
ISSUE_DUPLICATE_CANDIDATE = "duplicate_candidate"  # a second update of one candidate in the reply
ISSUE_UNEXPECTED_CANDIDATE = "unexpected_candidate"  # a criterion without a new observation
ISSUE_REMOVE_UNLISTED = "remove_unlisted"          # "removed" on a name that is neither listed nor proposed
ISSUE_BAD_REMOVAL = "bad_removal"                  # "removed" without a passage that speaks against it: kept
ISSUE_AGAINST_ONLY = "against_only"                # a status for a name only passages against: not added (a proposal stays one)
# A candidate update that is taken, with a flaw in its reason, its labels or its name:
ISSUE_NO_REASON = "no_reason_opening"              # the reason opens with none of "Was <status>:", "Was proposed:", "New:"
ISSUE_WRONG_PREVIOUS = "wrong_previous"            # "Was <X>:" with X not the candidate's current status (or not proposed)
ISSUE_NOT_LISTED = "not_listed"                    # "Was <X>:" on a candidate that is not listed (added)
ISSUE_LISTED_AS_NEW = "listed_as_new"              # "New:" on a listed candidate (its status updated)
ISSUE_BAD_REF = "bad_ref"                          # a support label, or a source of the reason, with no observation on the criterion
ISSUE_UNKNOWN_CANDIDATE = "unknown_candidate"      # a new candidate in no span or title of the criterion's passages
ISSUE_NEAR_DUPLICATE = "near_duplicate"            # a new candidate whose name contains a listed one, or the reverse
ISSUE_UNCHANGED = "unchanged"                      # no new support and the status written is the current one
ISSUE_READDED = "readded"                          # a name the belief dropped before (information only)
# A status code corrected (the hard limits):
ISSUE_NO_SUPPORT = "no_support"                    # no passage supports the candidate: at most "possible", or a new name kept as proposed
ISSUE_NO_NEW_SUPPORT = "no_new_support"            # a status change without new support (nor, one answer, another candidate's support changed): kept
# The belief after the reply (flags; the statuses are kept):
ISSUE_CERTAIN_WITH_RIVAL = "certain_with_rival"    # a one-answer candidate "certain" while another candidate has support
ISSUE_UNEQUAL_TIE = "unequal_tie"                  # one-answer candidates with the same support but different statuses

# Observations that did not enter for a reason of their own (the reply
# should not have written them).
REJECTED_OBSERVATION = (ISSUE_COMPUTED, ISSUE_NO_SPAN, ISSUE_NOT_IN_TURN, ISSUE_BAD_CLAIM)
# A candidate update whose status code corrected (the reply's status is
# not the one kept).
CORRECTED_STATUS = (ISSUE_NO_SUPPORT, ISSUE_NO_NEW_SUPPORT)
# A reply with one of these is not used as an SFT sample (with
# ``require_verified_spans``): an observation left out for a reason of its
# own, a status code corrected, a reason or label that misreads the belief
# or cites a source with no observation on the criterion, a removal code
# did not take, a candidate update code ignored (no observation on its
# criterion in the call, or only passages against a name not listed), or
# a one-answer criterion whose statuses do not follow its support
# (``certain_with_rival``, ``unequal_tie``).  ``unknown_candidate`` is not
# among them: a candidate the reply writes in other words than its spans
# (an initial, a computed value) raises it too; nor are
# ``near_duplicate``, ``listed_as_new``, ``unchanged`` and ``readded``
# (harmless for the belief).
UNCLEAN_REPLY = (*REJECTED_OBSERVATION, *CORRECTED_STATUS, ISSUE_NO_REASON, ISSUE_WRONG_PREVIOUS,
                 ISSUE_NOT_LISTED, ISSUE_BAD_REF, ISSUE_REMOVE_UNLISTED, ISSUE_BAD_REMOVAL,
                 ISSUE_UNEXPECTED_CANDIDATE, ISSUE_AGAINST_ONLY, ISSUE_CERTAIN_WITH_RIVAL, ISSUE_UNEQUAL_TIE)
# A reply with one of these fails the format gate of training: it names
# what the call did not show, or writes an update code cannot take.
GATE_ISSUES = (ISSUE_UNKNOWN_CRITERION, ISSUE_NOT_NEW_PASSAGE, ISSUE_EMPTY_CANDIDATE, ISSUE_BAD_STATUS,
               ISSUE_DUPLICATE_CANDIDATE, ISSUE_BAD_CLAIM)

# Proposals kept per criterion (beyond it the least recently claimed is dropped).
MAX_PROPOSALS = 8


def _issue(cid: str, code: str, detail: str, o: Optional[Observation] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": cid, "code": code, "detail": detail}
    if o is not None:
        out["label"] = o.label
    return out


def _level(status: str) -> int:
    return STATUS_LEVEL[status]


def _cap(status: str) -> str:
    """*status* at most ``NO_SUPPORT_MAX_STATUS``."""
    return min(status, NO_SUPPORT_MAX_STATUS, key=_level)



def forward_refs(texts: List[str]) -> List[str]:
    """The references of criterion *texts* (ids "c1", "c2", ... in order)
    to the criterion itself or to one after it, as "c<k> refers to c<m>":
    a criterion refers only to criteria before it."""
    return [f"c{k} refers to c{m}" for k, t in enumerate(texts, 1)
            for m in dict.fromkeys(int(x) for x in CRITERION_REF_RE.findall(t)) if k <= m <= len(texts)]


def _label_number(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


class BeliefState:
    """Belief state b_t, kept at two levels.

    Shown (``belief_state.format_belief``): per criterion its candidates,
    each with its status and its support (the passage observations that
    support it or speak against it, with their spans and passage titles),
    and its proposals (names the agent's reasoning proposed that no passage
    supports yet, without a status).  The agent's claims are kept but never
    shown to the update.

    Saved: per criterion the trace of the observations that entered it
    (``trace``) and its references (``reference_lists``): every source ever
    assigned to it, with its document, title, the step it came in and its
    verified spans; per step the candidate updates with their reasons
    (``apply``), never shown again; and per criterion the candidates and
    proposals it dropped (``removed``, ``types.RemovedCandidate``).

    Each call, the belief update writes the claims of the agent's turns
    (reasoning observations), the observations of the call's passages,
    then the candidates whose support or status changes, each with the
    passages that support it or speak against it, its reason and its new
    status; ``apply`` keeps what is valid:

    - an observation enters only when it is grounded (a verified span of
      its passage or turn), names a known criterion, and is not a passage
      observation on a criterion no document states (``NO_DOC_KINDS``); a
      reasoning observation also needs a candidate and a claim.  A second
      observation of a passage on the same criterion adds its new spans to
      the first one;
    - a claim sets the agent's claim on a listed candidate, or adds, updates
      or (``REJECTED``) drops a proposal; it never moves a status;
    - a candidate update is taken only for a criterion with an observation
      in the call, once per candidate, with a status of the scale other
      than "unknown", or "removed".  Its support labels must be passages of
      the call observed on the criterion (``bad_ref`` otherwise, the label
      dropped).  A name neither listed nor proposed, or a proposed one,
      becomes a candidate only with a passage that supports it (none at
      all: it stays or becomes a proposal, ``no_support``; only passages
      against it: a new name is not added, a proposal stays one,
      ``against_only``); a listed one gets the new support and status.  Candidates the reply
      does not name keep their status.  The reason is checked and flagged,
      never used to reject;
    - "removed" takes a listed candidate (or a proposal) out of the belief
      (into ``removed``) only when a passage of the call speaks against it
      (``against``); on a computed criterion, when the call has a claim on
      it.  Otherwise it is flagged and kept;
    - the hard limits (a computed criterion is exempt): a candidate no
      passage supports is at most "possible" (``no_support``); a listed
      candidate whose update adds no support keeps its status, unless in
      the same call another of its candidates gained support, was added or
      was removed (one answer: its status depends on the others; a
      proposal added or dropped does not count); ``no_new_support``
      otherwise.  A criterion that refers to another one ("the town of
      c1") is judged on its own parts, whatever the statuses of the
      candidates it links to, so a change there moves nothing here.
    - flags of a one-answer criterion, statuses kept: a "certain" candidate
      while another has support (``certain_with_rival``); two candidates
      with the same support and different statuses (``unequal_tie``).

    Derived per criterion (``belief_lists``): its status, the highest of
    its candidates ("unknown" without candidates); its selected
    candidate(s), among those at "possible" or above: for a one-answer
    criterion the first one shown (``ordered``), for a several-answers
    criterion every one.  A proposal is never selected.

    Documents get their label ("d<n>") when they are first shown
    (``label_doc``).
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self._index = {c.id: k for k, c in enumerate(self.criteria)}
        # Per criterion its candidates, in the order they were added, and its proposals.
        self.candidates: List[List[Candidate]] = [[] for _ in self.criteria]
        self.proposals: List[List[Proposal]] = [[] for _ in self.criteria]
        # The trace: per criterion, the observations that entered it.
        self.trace: List[List[Observation]] = [[] for _ in self.criteria]
        # Per criterion the candidates and proposals dropped, in order.
        self.removed: List[List[RemovedCandidate]] = [[] for _ in self.criteria]
        self._doc_labels: Dict[str, str] = {}
        self._titles: Dict[str, str] = {}
        self._call_dropped: set = set()

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

    def title(self, label: str) -> str:
        """The title of the document labelled *label* ("" when unknown)."""
        return self._titles.get(label) or ""

    # ------------------------------------------------------------------
    # Derived per criterion
    # ------------------------------------------------------------------

    def ordered(self, k: int) -> List[Candidate]:
        """The candidates of criterion *k* as shown: highest status first,
        then "passage" before "reasoning", then in the order they were
        added (so a one-answer criterion's selected candidate is first)."""
        return sorted(self.candidates[k], key=lambda c: (-_level(c.status), c.basis == REASONING))

    def status(self, k: int) -> str:
        """The status of criterion *k*: the highest of its candidates,
        "unknown" without candidates."""
        return max((c.status for c in self.candidates[k]), key=_level, default=UNKNOWN)

    def selected(self, k: int) -> List[str]:
        """The selected candidate(s) of criterion *k*, among those at
        "possible" or above: for a one-answer criterion the first one shown
        (``ordered``), for a several-answers criterion every one."""
        kept = [c.name for c in self.ordered(k) if _level(c.status) >= _level(NO_SUPPORT_MAX_STATUS)]
        return kept[:1] if exclusive(self.criteria[k].kind) else kept

    def ruled_out(self, k: int) -> List[str]:
        """The names criterion *k* dropped because a passage spoke against
        them (``RemovedCandidate.against``), each once, those listed or
        proposed again left out (the final scorer reads them as removed)."""
        names: Dict[str, str] = {}
        for x in self.removed[k]:
            if x.against and self._listed(k, x.name) is None and self._proposed(k, x.name) is None:
                names.setdefault(alnum(x.name), x.name)
        return list(names.values())

    def ruled_out_lists(self) -> Dict[str, List[str]]:
        """``ruled_out`` of every criterion, by id."""
        return {c.id: self.ruled_out(k) for k, c in enumerate(self.criteria)}

    # ------------------------------------------------------------------
    # The update
    # ------------------------------------------------------------------

    def apply(self, claims: List[Observation], docs: List[Observation],
              updates: List[CandidateUpdate], step: int) -> Dict[str, Any]:
        """Apply one belief update of the call of step *step*: the claims of
        the turns (*claims*, reasoning observations), the observations of
        the passages (*docs*), then the candidate *updates*.

        Returns ``{observations, updates, changed, issues}``: the
        observations that entered the belief (``Observation.to_dict``), one
        record per candidate update of the reply (``{id, candidate, reason,
        support, against, written, previous, status}``: ``support`` and
        ``against`` the labels kept, ``written`` the status the reply wrote
        ("removed" for a removal), ``previous`` the candidate's status
        before (null for a new one, "proposed" for a proposal), ``status``
        the one kept ("removed" when it left the belief, "proposed" when it
        stays a proposal), null when the update was not taken), the ids of
        the criteria whose candidates or proposals changed, and one issue
        record per thing the reply got wrong (the ``ISSUE_*`` codes)."""
        issues: List[Dict[str, Any]] = []
        # The proposals a claim of this call rejected.
        self._call_dropped = set()
        # Per criterion: the passages of the call observed on it (label ->
        # its observation in the trace), and whether the call has a claim on it.
        call_docs: List[Dict[str, Observation]] = [{} for _ in self.criteria]
        claimed = [False] * len(self.criteria)
        entered: List[Observation] = []
        changed = set()
        for o in [*claims, *docs]:
            k = self._index.get(o.criterion)
            if k is None:
                issues.append(_issue(o.criterion, ISSUE_UNKNOWN_CRITERION, "unknown criterion id", o))
            elif o.source == DOC and o.doc_id is None:
                issues.append(_issue(o.criterion, ISSUE_NOT_NEW_PASSAGE, "no passage of the call has this label", o))
            elif o.source == DOC and self.criteria[k].kind in NO_DOC_KINDS:
                issues.append(_issue(o.criterion, ISSUE_COMPUTED, "the criterion is computed: no passage observation", o))
            elif not o.grounded:
                issues.append(_issue(o.criterion, ISSUE_NO_SPAN, "no verified span", o) if o.source == DOC else
                              _issue(o.criterion, ISSUE_NOT_IN_TURN, "no span of the agent's turn", o))
            elif o.source == THINK and (not alnum(o.candidate or "") or o.claim is None):
                issues.append(_issue(o.criterion, ISSUE_BAD_CLAIM, "a claim needs a candidate and a status or "
                                                                   f"'{REJECTED}'", o))
            elif o.source == DOC:
                known = next((x for x in self.trace[k] if x.source == DOC and x.label == o.label), None)
                if known is None:
                    self.trace[k].append(o)
                    known = o
                else:
                    # A second observation of the passage: its new spans join the first.
                    for span in o.verified_spans:
                        if span not in known.spans:
                            known.spans.append(span)
                            known.span_verified.append(True)
                call_docs[k][o.label] = known
                entered.append(o)
            else:
                self.trace[k].append(o)
                claimed[k] = True
                entered.append(o)

        # Claims, in the order of their turns: the latest one stands.
        for o in sorted((o for o in entered if o.source == THINK), key=lambda o: _label_number(o.label)):
            if self._claim(self._index[o.criterion], o, issues):
                changed.add(self._index[o.criterion])

        records: List[Dict[str, Any]] = []
        done = set()
        # The support of the one-answer criteria before the updates (only
        # they defer a status change, below).
        before = {k: self._support_key(k) for k, c in enumerate(self.criteria) if exclusive(c.kind)}
        deferred: List[Tuple[int, Candidate, str, Dict[str, Any]]] = []
        for u in updates:
            name = " ".join(str(u.candidate or "").split())
            rec: Dict[str, Any] = {"id": u.id, "candidate": name, "reason": u.reason, "support": [], "against": [],
                                   "written": u.status, "previous": None, "status": None}
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
            if not call_docs[k] and not claimed[k]:
                issues.append(_issue(u.id, ISSUE_UNEXPECTED_CANDIDATE, f"{name!r}: no observation on the criterion "
                                                                        "in the call; ignored"))
                continue
            rec["written"] = status
            support = self._labels(k, name, u.support, call_docs[k], issues)
            against = [lb for lb in self._labels(k, name, u.against, call_docs[k], issues) if lb not in support]
            rec["support"], rec["against"] = support, against
            entries = [self._support(call_docs[k][lb], step, False) for lb in support] + \
                      [self._support(call_docs[k][lb], step, True) for lb in against]
            if status == REMOVED:
                moved = self._remove(k, name, u.reason, step, entries, claimed[k], rec, issues)
            else:
                moved = self._update(k, name, status, u.reason, step, entries, rec, issues, deferred)
            if moved:
                changed.add(k)

        # Criteria with a candidate added or removed, or whose support
        # changed (a proposal added or dropped is not): only then a
        # one-answer candidate without new support of its own may change
        # (its status depends on the others).
        support_moved = {k for k in before if self._support_key(k) != before[k]} if deferred else set()
        for k, existing, status, rec in deferred:
            if exclusive(self.criteria[k].kind) and k in support_moved:
                if not existing.supports() and _cap(status) != status:
                    issues.append(_issue(self.criteria[k].id, ISSUE_NO_SUPPORT,
                                         f"{existing.name!r}: {status!r} with no passage supporting it; "
                                         f"set to {_cap(status)!r}"))
                    status = _cap(status)
                existing.status = status
                rec["status"] = existing.status
                changed.add(k)
            else:
                issues.append(_issue(self.criteria[k].id, ISSUE_NO_NEW_SUPPORT,
                                     f"{existing.name!r}: {status!r} from {existing.status!r} with no new passage on "
                                     f"it, and no other candidate's support changed; kept"))
                rec["status"] = existing.status

        for k in sorted(changed):
            if exclusive(self.criteria[k].kind) and self.criteria[k].kind not in NO_DOC_KINDS:
                self._flag_rivals(k, issues)
        return {"observations": [o.to_dict() for o in entered], "updates": records,
                "changed": [self.criteria[k].id for k in sorted(changed)], "issues": issues}

    def _support_key(self, k: int) -> frozenset:
        """The candidates of criterion *k* with their support (passage
        labels and sides), statuses left out: what ``apply`` compares to
        tell whether the support of the criterion moved in a call."""
        return frozenset((alnum(x.name), frozenset((s.label, s.against) for s in x.support))
                         for x in self.candidates[k])

    def _listed(self, k: int, name: str) -> Optional[Candidate]:
        """The listed candidate of criterion *k* named *name* (letters and
        digits compared), or None."""
        key = alnum(name)
        return next((x for x in self.candidates[k] if alnum(x.name) == key), None)

    def _proposed(self, k: int, name: str) -> Optional[Proposal]:
        key = alnum(name)
        return next((x for x in self.proposals[k] if alnum(x.name) == key), None)

    def _dropped(self, k: int, name: str) -> bool:
        key = alnum(name)
        return any(alnum(x.name) == key for x in self.removed[k])

    def _claim(self, k: int, o: Observation, issues: List[Dict[str, Any]]) -> bool:
        """Apply the claim *o* on criterion *k*; returns whether the
        proposals changed."""
        step = o.step
        existing = self._listed(k, o.candidate or "")
        if existing is not None:
            existing.claim, existing.claim_step = o.claim, step
            return False
        proposal = self._proposed(k, o.candidate or "")
        if proposal is not None:
            if o.claim == REJECTED:
                self.proposals[k].remove(proposal)
                self._call_dropped.add((k, alnum(proposal.name)))
                self.removed[k].append(RemovedCandidate(
                    name=proposal.name, status=None, basis=REASONING, added_step=proposal.added_step, removed_step=step,
                    reason=f"rejected in {o.label}: {' ... '.join(o.verified_spans)}"))
                return True
            proposal.claim, proposal.step = o.claim, step
            return False
        if o.claim == REJECTED:
            return False
        self._add_proposal(k, " ".join((o.candidate or "").split()), o.claim, step, issues, o)
        return True

    def _add_proposal(self, k: int, name: str, claim: Optional[str], step: int, issues: List[Dict[str, Any]],
                      o: Optional[Observation] = None) -> None:
        """Propose *name* for criterion *k* (from a claim *o*, or an update
        without a passage): flagged when it was dropped before; past
        ``MAX_PROPOSALS`` the least recently claimed proposal is dropped
        (the agent's current names stay)."""
        if self._dropped(k, name):
            issues.append(_issue(self.criteria[k].id, ISSUE_READDED, f"{name!r} was dropped before; proposed again", o))
        self.proposals[k].append(Proposal(name=name, claim=claim, step=step, added_step=step))
        if len(self.proposals[k]) > MAX_PROPOSALS:
            old = min(self.proposals[k], key=lambda x: x.step)
            self.proposals[k].remove(old)
            self.removed[k].append(RemovedCandidate(name=old.name, status=None, basis=REASONING,
                                                    added_step=old.added_step, removed_step=step,
                                                    reason=f"more than {MAX_PROPOSALS} proposals: the least recently claimed dropped"))

    def _labels(self, k: int, name: str, labels: List[str], call_docs: Dict[str, Observation],
                issues: List[Dict[str, Any]]) -> List[str]:
        """The passage *labels* of an update of candidate *name* that are
        passages of the call observed on criterion *k*; the others are
        flagged and dropped."""
        kept = [lb for lb in dict.fromkeys(labels) if lb in call_docs]
        bad = [lb for lb in dict.fromkeys(labels) if lb not in call_docs]
        if bad:
            issues.append(_issue(self.criteria[k].id, ISSUE_BAD_REF,
                                 f"{name!r}: passages {bad} have no observation on the criterion in the call"))
        return kept

    def _support(self, o: Observation, step: int, against: bool) -> Support:
        return Support(label=o.label, doc_id=o.doc_id, title=self.title(o.label), step=step,
                       spans=o.verified_spans, against=against)

    def _check_reason(self, k: int, name: str, reason: str, existing: Optional[Candidate],
                      proposal: Optional[Proposal], issues: List[Dict[str, Any]]) -> None:
        """Flag what the *reason* of an update of *name* of criterion *k*
        gets wrong: its opening against the candidate's current status (or
        its proposal), its labels against the trace."""
        c = self.criteria[k]
        opening = parse_reason(reason)
        if opening is None:
            issues.append(_issue(c.id, ISSUE_NO_REASON, f"{name!r}: the reason opens with none of 'Was <status>:', "
                                                        "'Was proposed:', 'New:'"))
        elif opening.new and existing is not None:
            issues.append(_issue(c.id, ISSUE_LISTED_AS_NEW, f"{name!r} is listed ({existing.status}); its status is updated"))
        elif opening.new and proposal is not None:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS, f"{name!r}: the reason says it is new, it was proposed"))
        elif opening.proposed and proposal is None:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS, f"{name!r}: the reason says it was proposed, it is "
                                                             f"{existing.status if existing else 'not listed'}"))
        elif not opening.new and not opening.proposed and existing is None:
            issues.append(_issue(c.id, ISSUE_NOT_LISTED, f"'Was' on {name!r}, which is not listed"))
        elif not opening.new and not opening.proposed and opening.previous != existing.status:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS,
                                 f"{name!r}: the reason says it was {opening.previous!r}, it is {existing.status!r}"))
        observed = {o.label for o in self.trace[k]}
        bad = [lb for lb in reason_labels(reason) if lb not in observed]
        if bad:
            issues.append(_issue(c.id, ISSUE_BAD_REF, f"{name!r}: the reason cites sources {bad} with no "
                                                      "observation on the criterion"))

    def _remove(self, k: int, name: str, reason: str, step: int, entries: List[Support], claimed: bool,
                rec: Dict[str, Any], issues: List[Dict[str, Any]]) -> bool:
        """Remove candidate or proposal *name* of criterion *k* when a
        passage of the call speaks against it (``apply``); fills *rec* and
        returns whether it was removed."""
        c = self.criteria[k]
        existing, proposal = self._listed(k, name), self._proposed(k, name)
        if existing is None and proposal is None and (k, alnum(name)) in self._call_dropped:
            # A proposal a claim of the call rejected: already dropped.
            rec["previous"], rec["status"] = "proposed", REMOVED
            return False
        if existing is None and proposal is None:
            issues.append(_issue(c.id, ISSUE_REMOVE_UNLISTED, f"'removed' on {name!r}, which is neither listed nor "
                                                              "proposed; ignored"))
            return False
        rec["previous"] = existing.status if existing is not None else "proposed"
        rec["candidate"] = existing.name if existing is not None else proposal.name
        self._check_reason(k, rec["candidate"], reason, existing, proposal, issues)
        against = [x for x in entries if x.against]
        if not (against or (c.kind in NO_DOC_KINDS and claimed)):
            issues.append(_issue(c.id, ISSUE_BAD_REMOVAL, f"{rec['candidate']!r}: removed, but no passage of the "
                                                          "call speaks against it; kept"))
            rec["status"] = rec["previous"]
            return False
        if existing is not None:
            self.candidates[k].remove(existing)
            self.removed[k].append(RemovedCandidate(name=existing.name, status=existing.status, basis=existing.basis,
                                                    added_step=existing.added_step, removed_step=step, reason=reason,
                                                    against=bool(against)))
        else:
            self.proposals[k].remove(proposal)
            self.removed[k].append(RemovedCandidate(name=proposal.name, status=None, basis=REASONING,
                                                    added_step=proposal.added_step, removed_step=step, reason=reason,
                                                    against=bool(against)))
        rec["status"] = REMOVED
        return True

    def _update(self, k: int, name: str, status: str, reason: str, step: int, entries: List[Support],
                rec: Dict[str, Any], issues: List[Dict[str, Any]],
                deferred: List[Tuple[int, Candidate, str, Dict[str, Any]]]) -> bool:
        """Add candidate *name* of criterion *k* with *status* and its
        support *entries*, promote its proposal, or change a listed one,
        within the hard limits (``apply``); fills *rec* and returns whether
        the belief changed.  A status change ``apply`` decides after the
        reply's other updates goes to *deferred*."""
        c = self.criteria[k]
        existing, proposal = self._listed(k, name), self._proposed(k, name)
        if existing is not None:
            rec["previous"], rec["candidate"] = existing.status, existing.name
        elif proposal is not None:
            rec["previous"], rec["candidate"] = "proposed", proposal.name
        self._check_reason(k, rec["candidate"], reason, existing, proposal, issues)
        computed = c.kind in NO_DOC_KINDS

        if existing is None:
            if not entries and not computed:
                issues.append(_issue(c.id, ISSUE_NO_SUPPORT, f"{name!r}: no passage of the call supports it or speaks "
                                                             "against it; kept as proposed"))
                rec["status"] = "proposed"
                if proposal is None:
                    self._add_proposal(k, name, None, step, issues)
                    return True
                return False
            if not computed and not any(not x.against for x in entries):
                # Only passages against it: no possible answer (a passage
                # that states it fails a part removes a proposal instead).
                issues.append(_issue(c.id, ISSUE_AGAINST_ONLY, f"{name!r}: only passages against it; " +
                                     ("kept as proposed" if proposal is not None else "not added")))
                rec["status"] = "proposed" if proposal is not None else None
                return False
            if not computed and c.kind in NAMED_KINDS and self._unknown_candidate(k, name):
                issues.append(_issue(c.id, ISSUE_UNKNOWN_CANDIDATE, f"{name!r} in no span or title of the "
                                                                    "criterion's passages"))
            near = [x.name for x in self.candidates[k] if contains_words(x.name, name) or contains_words(name, x.name)]
            if near:
                issues.append(_issue(c.id, ISSUE_NEAR_DUPLICATE, f"{name!r} is close to listed {near}"))
            if proposal is None and self._dropped(k, name):
                issues.append(_issue(c.id, ISSUE_READDED, f"{name!r} was dropped before; it is added again"))
            cand = Candidate(name=proposal.name if proposal is not None else name, status=status,
                             added_step=step, support=list(entries))
            if proposal is not None:
                cand.claim, cand.claim_step = proposal.claim, proposal.step
                self.proposals[k].remove(proposal)
            if not computed and not cand.supports() and _cap(status) != status:
                issues.append(_issue(c.id, ISSUE_NO_SUPPORT, f"{name!r}: {status!r} with no passage supporting it; "
                                                             f"set to {_cap(status)!r}"))
                cand.status = _cap(status)
            self.candidates[k].append(cand)
            rec["status"] = cand.status
            return True

        known = {(x.label, x.against) for x in existing.support}
        added = [x for x in entries if (x.label, x.against) not in known]
        if not added and not computed:
            if status == existing.status:
                issues.append(_issue(c.id, ISSUE_UNCHANGED, f"{existing.name!r}: no new support and the status "
                                                            "written is its current one"))
                rec["status"] = existing.status
                return False
            deferred.append((k, existing, status, rec))
            return False
        existing.support.extend(added)
        if not computed and not existing.supports() and _cap(status) != status:
            issues.append(_issue(c.id, ISSUE_NO_SUPPORT, f"{existing.name!r}: {status!r} with no passage supporting "
                                                         f"it; set to {_cap(status)!r}"))
            status = _cap(status)
        moved = bool(added) or status != existing.status
        existing.status = status
        rec["status"] = status
        return moved

    def _flag_rivals(self, k: int, issues: List[Dict[str, Any]]) -> None:
        """The flags of a one-answer criterion *k* (``apply``)."""
        c = self.criteria[k]
        cands = self.candidates[k]
        for x in cands:
            if x.status == "certain":
                rivals = [y.name for y in cands if y is not x and y.supports()]
                if rivals:
                    issues.append(_issue(c.id, ISSUE_CERTAIN_WITH_RIVAL,
                                         f"{x.name!r} is certain while {rivals} have support"))
        groups: Dict[Tuple, List[Candidate]] = {}
        for x in cands:
            if x.supports():
                key = (frozenset(s.label for s in x.supports()), frozenset(s.label for s in x.supports(True)))
                groups.setdefault(key, []).append(x)
        for same in groups.values():
            if len({x.status for x in same}) > 1:
                issues.append(_issue(c.id, ISSUE_UNEQUAL_TIE, "the same support, different statuses: " +
                                     ", ".join(f"{x.name!r} {x.status}" for x in same)))

    def _passage_texts(self, k: int) -> List[str]:
        """The verified spans of the passage observations on criterion *k*
        and the titles of their passages (a passage may name its subject in
        its title only)."""
        out: List[str] = []
        for o in self.trace[k]:
            if o.source == DOC:
                out += [*o.verified_spans, self.title(o.label)]
        return out

    def _parts(self, k: int, name: str) -> List[str]:
        """The parts of a candidate a source writes out: a rest criterion's
        "<member>: <value>" the member and the value, else the name whole."""
        return name.split(":", 1) if self.criteria[k].kind == REST and ":" in name else [name]

    def _unknown_candidate(self, k: int, name: str) -> bool:
        """Whether candidate *name* occurs in none of the spans of the
        passage observations on criterion *k* (its support among them) nor
        in their passages' titles (a candidate a passage names in its title
        only, or a hop's candidate named by the passage of another span).
        Checked for the kinds whose answer is a name or figure a source
        writes out (``NAMED_KINDS``; a rest criterion's "<member>:
        <value>", the member and the value each)."""
        texts = self._passage_texts(k)
        return not all(any(contains_words(t, p) for t in texts) for p in self._parts(k, name) if alnum(p))

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    def belief_lists(self) -> List[Dict[str, Any]]:
        """Per criterion ``{id, status, selected, candidates, proposals}``:
        its derived status and selection, its candidates as shown
        (``Candidate.to_dict``, highest status first) and its proposals
        (``Proposal.to_dict``)."""
        return [{"id": c.id, "status": self.status(k), "selected": self.selected(k),
                 "candidates": [x.to_dict() for x in self.ordered(k)],
                 "proposals": [x.to_dict() for x in self.proposals[k]]}
                for k, c in enumerate(self.criteria)]

    def reference_lists(self) -> List[Dict[str, Any]]:
        """Per criterion ``{id, references, removed}``: every source assigned
        to it so far, in the order it came in, as ``{label, step, spans}``
        plus ``doc_id`` and ``title`` for a document, ``candidate`` and
        ``claim`` for a claim; and the candidates and proposals dropped
        (``RemovedCandidate.to_dict``)."""
        out = []
        for k, (c, obs) in enumerate(zip(self.criteria, self.trace)):
            refs = []
            for o in obs:
                r: Dict[str, Any] = {"label": o.label, "step": o.step}
                if o.source == DOC:
                    r.update(doc_id=o.doc_id, title=self.title(o.label))
                else:
                    r.update(candidate=o.candidate, claim=o.claim)
                r["spans"] = o.verified_spans
                refs.append(r)
            out.append({"id": c.id, "references": refs, "removed": [x.to_dict() for x in self.removed[k]]})
        return out
