"""Criteria and belief state of the belief model.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K}; the belief state b_t gives each
criterion a label from the seven ICD 203 words of estimative probability
(``types.WEP_LABELS``), rewritten every step by the belief updater from the
current belief and the step's documents (it can move up or down).

- ``CriteriaSource``: produces C once, at the start of each sample.
  ``LLMCriteriaSource`` derives the criteria from the query;
  ``BankedCriteriaSource`` reads them from a bank file and derives (and
  adds) only the missing ones, so runs and analyses share one list per query.
- ``BeliefState``: b_t and the evidence attached to each criterion.

Each criterion is closed (one fact), open (several parts or answers that
documents establish only together) or, in a report request, an aspect of the
topic.  A set query ends with a rest criterion, "any other member:
<property>", for the members the extractor did not list.  The kind follows
from the dataset's query shape and the criterion's position and text
(``_kinds``); the extractor only lists the criteria.

The belief updater and the query scorer are in ``judges``.
"""

import fcntl
import hashlib
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import parse_json_object
from .prompts import CRITERIA_INIT_SYSTEMS, criteria_init_user
from .types import (
    ASPECT, CLOSED, INITIAL_LABEL, MULTI_ASPECT, OPEN, QUERY_SHAPES, REST, SET,
    Criterion, CriterionUpdate, Evidence,
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
    target is closed; the first criterion of a set query (the complete set)
    is open, its rest criterion is rest and its members are closed; every
    report criterion is an aspect."""
    if shape == MULTI_ASPECT:
        return [ASPECT] * len(texts)
    if shape == SET:
        return [OPEN if k == 0 else REST if _is_rest(t) else CLOSED for k, t in enumerate(texts)]
    return [CLOSED] * len(texts)


def _set_with_rest(items: List[str], cap: int, info: Dict[str, Any]) -> List[str]:
    """A set query's criteria as the prompt asks: the set criterion, the
    members, then one rest criterion (added with the members' property when
    the model left it out), within *cap* by leaving out members."""
    if not items:
        return items
    first, rest = items[0], [t for t in items[1:] if _is_rest(t)]
    members = [t for t in items[1:] if not _is_rest(t)]
    if rest:
        rest_item = rest[0]
    else:
        prop = next((t.rsplit(":", 1)[1].strip() for t in members if ":" in t), "")
        rest_item = f"{REST_SUBJECT}: {prop or 'the property the query asks about'}"
        info.setdefault("warnings", []).append("rest: added the missing rest criterion")
    keep = max(cap - 2, 0)
    if len(members) > keep:
        info.setdefault("warnings", []).append(f"cap: kept {keep} of {len(members)} members")
    return [first] + members[:keep] + [rest_item]


# ---------------------------------------------------------------------------
# Criteria source
# ---------------------------------------------------------------------------

class CriteriaSource(ABC):
    """Produces the fixed criteria list of one query."""

    name: str = ""

    @abstractmethod
    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        """Return ``(criteria, info)``; *info* is saved in the meta line.

        *info* carries at least ``errors`` (a list of strings); an empty
        criteria list with a non-empty ``errors`` means the source failed.
        """


class LLMCriteriaSource(CriteriaSource):
    """Derives the criteria of the query with one LLM call.

    The dataset's query shape (single target, set or multi-aspect) picks the
    prompt and the kind of each criterion; the model only lists the
    criteria, as strings.  The shape is saved in *info* as ``query_shape``.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_criteria: Cap on the number of criteria kept (the dataset's
            ``max_criteria`` in layout.DATASET_SPECS).  For a multi-aspect
            query it is the number of criteria the model may add to the
            request's own, asked in the prompt; the list is kept whole, since
            a cut could drop a criterion the request states.
        model_name: Saved in the meta line.
        max_tokens: Max tokens for the LLM call.
        temperature: LLM temperature.
        query_shape: ``"single_target"``, ``"set"`` or ``"multi_aspect"``
            (the dataset's ``query_shape`` in layout.DATASET_SPECS).
    """

    name = "llm"

    def __init__(
        self,
        llm_client: Any,
        max_criteria: int,
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        query_shape: str = "single_target",
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        self._shape = query_shape
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

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        info: Dict[str, Any] = {"model": self.model_name, "query_shape": self._shape, "errors": []}
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": criteria_init_user(self._shape, query, self._max_criteria)},
        ]
        try:
            raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        except Exception as e:
            logger.warning("LLMCriteriaSource: LLM call failed", exc_info=True)
            info["errors"].append(f"llm_call: {e}")
            return [], info

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
            items = _set_with_rest(items, self._max_criteria, info)
        elif self._shape != MULTI_ASPECT:
            items = items[:self._max_criteria]
        criteria = [
            Criterion(id=f"c{k + 1}", text=t, kind=kind)
            for k, (t, kind) in enumerate(zip(items, _kinds(self._shape, items)))
        ]
        return criteria, info


class BankedCriteriaSource(CriteriaSource):
    """Reads the criteria of a query from a bank file; a query missing from
    it gets them from *inner*, and they are added to the bank.

    The bank (``layout.criteria_bank_path``) is a JSONL file, one line per
    query and setting: the query id and text, ``inner.bank_fields`` (model,
    query shape, cap, prompt hash), the criteria and the extraction info.
    A line is used only when all of these match, so a changed prompt or cap
    extracts the criteria again.  A salvaged extraction (the complete items
    of a truncated reply) is asked once more and fails when salvaged again,
    so every criteria list used is banked; a failed extraction (no criteria)
    is never added.  Lines are appended under a file lock, after the file is
    read again: when another worker or process added the query meanwhile,
    its line is used and none is added, so one query has one banked list.

    *info* is the stored extraction info with ``bank``: ``"hit"`` (read from
    the bank), ``"added"`` (extracted and added) or ``"failed"`` (no
    criteria, not added).
    """

    name = "bank"

    def __init__(self, inner: LLMCriteriaSource, path: Path | str) -> None:
        self._inner = inner
        self.path = Path(path)
        self.model_name = inner.model_name
        self._fields = inner.bank_fields
        self._lock = threading.Lock()
        self._bank: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._bank[rec["key"]] = rec

    def _key(self, query_id: Optional[str], query: str) -> str:
        fields = {"query_id": query_id, "query": query, **self._fields}
        return hashlib.sha1(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def lookup(self, query_id: Optional[str], query: str) -> Optional[List[Criterion]]:
        """The banked criteria of the query, or None; never extracts."""
        with self._lock:
            rec = self._bank.get(self._key(query_id, query))
        return None if rec is None else [Criterion(**c) for c in rec["criteria"]]

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        key = self._key(query_id, query)
        with self._lock:
            rec = self._bank.get(key)
        if rec is not None:
            return [Criterion(**c) for c in rec["criteria"]], {**rec["info"], "bank": "hit"}

        # A salvaged reply is asked once more; salvaged again, the extraction fails.
        for _ in range(2):
            criteria, info = self._inner.get(query_id, query)
            if not info.get("salvaged"):
                break
        else:
            info = {**info, "errors": info.get("errors", []) + ["parse: truncated reply twice, not banked"]}
            criteria = []
        if not criteria:
            return criteria, {**info, "bank": "failed"}
        rec = {
            "key": key, "query_id": query_id, "query": query, **self._fields,
            "criteria": [c.to_dict() for c in criteria], "info": info,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # Lines another worker or process added since the bank was read.
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            other = json.loads(line)
                            self._bank[other["key"]] = other
                    banked = self._bank.get(key)
                    if banked is None:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        self._bank[key] = rec
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        if banked is not None:
            return [Criterion(**c) for c in banked["criteria"]], {**banked["info"], "bank": "hit"}
        return criteria, {**info, "bank": "added"}


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

class BeliefState:
    """Belief state b_t: per criterion, its WEP label and its evidence.

    b_0 labels every criterion ``INITIAL_LABEL``.  Each step the belief
    updater rewrites the belief from the current one and the step's passages
    (Agent-BRACE style) and ``update`` takes it as written:

    - the label of each criterion is the one the updater gave; a label may
      stay, rise or fall by any number of levels;
    - a criterion the updater left out, or gave no WEP label, keeps its
      label (``note`` says why);
    - a cited passage is attached only with at least one span that occurs
      in it verbatim (``Evidence.verified_spans``); the others are dropped.
      This guards the evidence only, never the label;
    - a passage already attached to the criterion with the same role is not
      attached again (every passage of a step is shown, also ones seen
      before).

    The updater sees the attached evidence in later steps.
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self.labels: List[str] = [INITIAL_LABEL] * len(self.criteria)
        self._evidence: List[List[Evidence]] = [[] for _ in self.criteria]
        self._index = {c.id: k for k, c in enumerate(self.criteria)}

    def attached(self, k: int, limit: Optional[int] = None) -> List[Evidence]:
        """Evidence of criterion *k*, oldest first; the last *limit* when set."""
        items = self._evidence[k]
        return items[-limit:] if limit else list(items)

    def update(self, updates: List[CriterionUpdate]) -> List[Dict[str, Any]]:
        """Apply the belief the updater wrote for one step.

        Returns one record per criterion, in criteria order, then one per
        update with an unknown or repeated id: ``{id, from, to, written,
        support, contradict, reason, note}``.  ``written`` is the label as
        the updater wrote it (null when left out); ``support`` and
        ``contradict`` are the passages as cited, with ``span_verified``.
        Only the first update of a criterion is used.
        """
        first: Dict[int, CriterionUpdate] = {}
        extra: List[Dict[str, Any]] = []
        for u in updates:
            k = self._index.get(u.id)
            if k is None or k in first:
                extra.append({
                    "id": u.id, "from": None, "to": None, "written": u.label,
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
                "id": c.id, "from": self.labels[k], "to": self.labels[k], "written": None,
                "support": [], "contradict": [], "reason": "", "note": "",
            }
            records.append(record)
            if u is None:
                record["note"] = "left out by the updater; label kept"
                continue
            record.update(
                written=u.label, reason=u.reason,
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
            if u.label is None:
                notes.append("no WEP label; label kept")
            else:
                self.labels[k] = u.label
                record["to"] = u.label
            record["note"] = "; ".join(notes)
        return records + extra

    def sources(self) -> List[int]:
        """Per criterion, its distinct supporting documents."""
        return [self.num_sources(k) for k in range(len(self.criteria))]

    def num_sources(self, k: int) -> int:
        """Distinct documents among the supporting evidence of criterion *k*."""
        return len({e.doc_id for e in self._evidence[k] if e.role == "support"})

    def evidence(self) -> List[Dict[str, Any]]:
        """Per criterion, its kind, label and its attached evidence (without
        text)."""
        return [
            {"id": c.id, "kind": c.kind, "label": label, "evidence": [e.to_dict() for e in ev]}
            for c, label, ev in zip(self.criteria, self.labels, self._evidence)
        ]
