"""LLM judges of the belief model.

- ``LLMBeliefUpdater``: the belief update b_t = f(query, b_{t-1}, o_t),
  Agent-BRACE style.  One call per step sees the current belief (each
  criterion's WEP label and evidence) plus every passage of the step, and
  rewrites the belief: per criterion, the passages that support or
  contradict it and its new label.  ``BeliefState.update`` takes the labels
  as written.
- ``LLMQueryScorer``: one call per step scores how strongly each query
  targets each criterion (0, 0.5 or 1), given a short summary of the
  belief; used for the targeting state a.

``build_belief_judges`` builds both on one LLM client.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.text_utils import doc_id as _doc_id, doc_text, parse_json_object

from .criteria import BeliefState
from .prompts import (
    BELIEF_UPDATE_PASSAGE_TEMPLATE,
    BELIEF_UPDATE_SYSTEMS,
    BELIEF_UPDATE_USER_TEMPLATE,
    QUERY_TARGETING_SYSTEM,
    QUERY_TARGETING_USER_TEMPLATE,
)
from .types import MULTI_ASPECT, QUERY_SHAPES, SINGLE_TARGET, CriterionUpdate, Evidence, parse_label

logger = logging.getLogger(__name__)


def _quote(spans: List[str]) -> str:
    return " | ".join(f'"{s}"' for s in spans)


def format_state_summary(state: BeliefState) -> str:
    """Per criterion, its label and the verified spans of its latest
    evidence passage that has any."""
    blocks = []
    for k, c in enumerate(state.criteria):
        lines = [f"{c.id} [{state.labels[k]}]: {c.text}"]
        latest = next((e for e in reversed(state.attached(k)) if e.verified_spans), None)
        if latest is not None:
            lines.append(f"  evidence ({latest.role}): {_quote(latest.verified_spans)}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _keys(text: str) -> Tuple[str, List[int]]:
    """*text* lowercased with everything but letters and digits removed, and
    the position in *text* of each character kept, for span matching."""
    keys, pos = [], []
    for i, ch in enumerate(text):
        for k in ch.lower():
            if k.isalnum():
                keys.append(k)
                pos.append(i)
    return "".join(keys), pos


def span_in_passage(text: str, span: str) -> Optional[str]:
    """The passage's own text matching *span* (case, whitespace and
    punctuation ignored, so words the passage runs together still match),
    or None when *span* does not occur in *text*."""
    target, _ = _keys(span)
    if not target:
        return None
    keys, pos = _keys(text)
    j = keys.find(target)
    if j < 0:
        return None
    return text[pos[j]:pos[j + len(target) - 1] + 1]


def split_span(span: str) -> List[str]:
    """The pieces of a cited span that joins sentences with an ellipsis."""
    return [p.strip() for p in re.split(r"\.\.\.|\u2026", span) if _keys(p)[0]]


def passage_title(text: str, max_chars: int = 120) -> str:
    """The first line of *text* with a letter or digit (a document's title
    or headline), cut at *max_chars* with ``...``."""
    line = next((ln.strip() for ln in text.splitlines() if _keys(ln)[0]), "")
    return line[:max_chars] + "..." if len(line) > max_chars else line


# ---------------------------------------------------------------------------
# Belief updater (b_t, Delta^D)
# ---------------------------------------------------------------------------

def _parse_belief(raw: str) -> Optional[List[Any]]:
    """The updater's list of criteria, or None when *raw* has none.  Besides
    ``{"belief": [...]}``, a bare list (the criteria without their key) is
    accepted."""
    data = parse_json_object(raw)
    if data is not None:
        return data["belief"] if isinstance(data.get("belief"), list) else None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return None
    return items if isinstance(items, list) else None


class LLMBeliefUpdater:
    """Stateful LLM belief updater: one call per step rewrites the belief.

    The prompt (one per query shape) shows each criterion with its label and
    its last ``max_evidence_per_criterion`` evidence passages (each as its
    verified spans and its title line), then every passage of the step
    (``max_passage_chars`` each), also the ones seen in earlier steps.  The
    updater gives every criterion its label, with the step's passages that
    support or contradict it.  It sees neither the agent's search queries
    nor the signals.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        model_name: Saved in the meta line.
        query_shape: ``"single_target"``, ``"set"`` or ``"multi_aspect"``
            (the dataset's ``query_shape`` in layout.DATASET_SPECS); picks
            the prompt.
        max_passage_chars: Text of one passage shown to the updater; None
            shows the whole passage.
        max_evidence_per_criterion: Attached evidence passages shown per
            criterion (the most recent).
        max_tokens / temperature: LLM call settings.  The reply has one
            entry per criterion, so it grows with the criteria list.
        parse_retries: Extra calls when the output cannot be parsed; each
            failed attempt is recorded in the step's errors.
    """

    def __init__(
        self,
        llm_client: Any,
        model_name: Optional[str] = None,
        query_shape: str = SINGLE_TARGET,
        max_passage_chars: Optional[int] = 3000,
        max_evidence_per_criterion: int = 4,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        parse_retries: int = 1,
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        self._llm = llm_client
        self.model_name = model_name
        self.name = f"llm:{model_name}"
        self._system = BELIEF_UPDATE_SYSTEMS[query_shape]
        self._parse_retries = parse_retries
        self._max_passage_chars = max_passage_chars
        self._max_evidence = max_evidence_per_criterion
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _format_belief(self, state: BeliefState) -> str:
        lines = []
        for k, c in enumerate(state.criteria):
            lines.append(f"{c.id} [{state.labels[k]}]: {c.text}")
            for e in state.attached(k, self._max_evidence):
                title = f" ({e.title})" if e.title else ""
                lines.append(f"  - {e.role} (step {e.step}): {_quote(e.verified_spans)}{title}")
            hidden = len(state.attached(k)) - self._max_evidence
            if hidden > 0:
                lines.append(f"  - and {hidden} earlier evidence passages")
        return "\n".join(lines)

    def _evidence(self, items: Any, docs: List[Dict[str, Any]], step: int, role: str) -> List[Evidence]:
        """One Evidence per cited passage; the spans of a passage cited twice
        are merged.  Each span is checked on its own, and a span joined with
        an ellipsis is split first."""
        out: Dict[str, Evidence] = {}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("passage")) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= i < len(docs):
                continue
            cited = item.get("spans", [])
            cited = cited if isinstance(cited, list) else [cited]
            text = doc_text(docs[i], max_length=None)
            ev = out.setdefault(_doc_id(docs[i]), Evidence(
                doc_id=_doc_id(docs[i]), step=step, role=role, spans=[], span_verified=[],
                title=passage_title(text),
            ))
            for piece in (p for c in cited for p in split_span(str(c))):
                matched = span_in_passage(text, piece)
                span = matched or piece
                if span not in ev.spans:
                    ev.spans.append(span)
                    ev.span_verified.append(matched is not None)
        return list(out.values())

    def update(
        self, query: str, state: BeliefState, docs: List[Dict[str, Any]], step: int,
    ) -> Tuple[Optional[List[CriterionUpdate]], Optional[str], List[str]]:
        """Return ``(updates, raw, errors)``; *updates* is None when the call
        or its parsing failed, *raw* is None when the updater was not called
        (no document with text).  Documents without text are left out."""
        has_text = [bool(doc_text(doc, max_length=None)) for doc in docs]
        kept = [doc for doc, ok in zip(docs, has_text) if ok]
        errors = [f"belief_updater: empty text for doc {_doc_id(d)}" for d, ok in zip(docs, has_text) if not ok]
        if not kept:
            return [], None, errors
        passages = "\n\n".join(
            BELIEF_UPDATE_PASSAGE_TEMPLATE.format(n=i + 1, text=doc_text(doc, max_length=self._max_passage_chars))
            for i, doc in enumerate(kept)
        )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": BELIEF_UPDATE_USER_TEMPLATE.format(
                query=query, belief=self._format_belief(state), passages=passages,
            )},
        ]
        raw, data = "", None
        for attempt in range(1 + self._parse_retries):
            try:
                raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature) or ""
            except Exception as e:
                logger.warning("LLMBeliefUpdater: step call failed: %s", e)
                return None, "", errors + [f"belief_updater: {e}"]
            data = _parse_belief(raw)
            if data is not None:
                break
            errors.append(f"belief_updater: parse (attempt {attempt + 1}): no JSON object with a 'belief' list")
        else:
            logger.warning("LLMBeliefUpdater: step %d dropped, updater output not parseable", step)
            return None, raw, errors

        updates: List[CriterionUpdate] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            updates.append(CriterionUpdate(
                id=str(item.get("id", "")).strip(),
                label=parse_label(item.get("label")),
                support=self._evidence(item.get("support"), kept, step, "support"),
                contradict=self._evidence(item.get("contradict"), kept, step, "contradict"),
                reason=str(item.get("reason", "")),
            ))
        return updates, raw, errors


# ---------------------------------------------------------------------------
# Query scorers (targeting state a)
# ---------------------------------------------------------------------------

class LLMQueryScorer:
    """LLM as judge: one call per step scores every (query, criterion) pair
    as 0, 0.5 or 1.  The prompt shows the belief (label, latest verified
    evidence), so a query naming an entity the evidence already linked to a
    criterion counts as targeting it."""

    def __init__(
        self,
        llm_client: Any,
        model_name: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm_client
        self.name = f"llm:{model_name}"
        self._max_tokens = max_tokens
        self._temperature = temperature

    def score(self, query: str, subqueries: List[str], state: BeliefState) -> List[List[float]]:
        """Return ``scores[j][k]`` in [0, 1] for ``subqueries[j]`` and the
        k-th criterion of *state*; raises on failure."""
        criteria = state.criteria
        messages = [
            {"role": "system", "content": QUERY_TARGETING_SYSTEM},
            {"role": "user", "content": QUERY_TARGETING_USER_TEMPLATE.format(
                query=query,
                criteria=format_state_summary(state),
                subqueries="\n".join(f"{j + 1}. {q}" for j, q in enumerate(subqueries)),
            )},
        ]
        raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("queries"), list):
            raise ValueError("parse: no JSON object with a 'queries' list")

        col = {c.id: k for k, c in enumerate(criteria)}
        scores = [[0.0] * len(criteria) for _ in subqueries]
        for item in data["queries"]:
            if not isinstance(item, dict):
                continue
            try:
                j = int(item.get("index")) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= j < len(subqueries):
                continue
            for target in item.get("targets") or []:
                if not isinstance(target, dict) or str(target.get("id", "")).strip() not in col:
                    continue
                try:
                    value = float(target.get("score", 0))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                scores[j][col[str(target["id"]).strip()]] = min(max(value, 0.0), 1.0)
        return scores


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_belief_judges(
    llm_client: Any, model: str = "", query_shape: str = SINGLE_TARGET,
) -> Tuple[LLMBeliefUpdater, LLMQueryScorer]:
    """Build ``(belief_updater, query_scorer)`` on one LLM client (named
    *model*) for the queries of one shape.  A report document is shown
    whole, as in the criteria reachability analysis; a short-answer document
    (often a long web page) is cut."""
    if llm_client is None:
        raise ValueError("the criteria judges need an LLM client")
    max_passage_chars = None if query_shape == MULTI_ASPECT else 3000
    return (
        LLMBeliefUpdater(llm_client, model_name=model, query_shape=query_shape, max_passage_chars=max_passage_chars),
        LLMQueryScorer(llm_client, model_name=model),
    )
