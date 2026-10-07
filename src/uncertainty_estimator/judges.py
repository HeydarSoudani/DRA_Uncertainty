"""LLM judges behind the criteria-based signals of the uncertainty estimator.

- ``LLMCoverageJudge``: stateful coverage judge, used for Delta^D.  One
  call per step sees each criterion's status and attached evidence plus the
  step's novel passages, and proposes status updates (raise, keep with new
  evidence, or lower on a contradiction); ``CriteriaState.apply`` enforces
  the rules.
- ``LLMQueryScorer``: one call per step scores how strongly each query
  targets each criterion (0, 0.5 or 1), given a short summary of the
  criteria state; used for the targeting state a.

``build_criteria_judges`` builds both on one LLM client.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.text_utils import doc_id as _doc_id, doc_text, parse_json_object

from .criteria import CriteriaState
from .prompts import (
    CRITERIA_MATCH_QUERY_SYSTEM,
    CRITERIA_MATCH_QUERY_USER_TEMPLATE,
    CRITERIA_UPDATE_DOC_SYSTEMS,
    CRITERIA_UPDATE_DOC_USER_TEMPLATE,
)
from .types import QUERY_SHAPES, SINGLE_TARGET, CriterionUpdate, Evidence

logger = logging.getLogger(__name__)


def _quote(spans: List[str]) -> str:
    return " | ".join(f'"{s}"' for s in spans)


def format_state_summary(state: CriteriaState) -> str:
    """Per criterion, its status, the verified spans of its latest evidence
    passage that has any, and what it still lacks."""
    blocks = []
    for k, c in enumerate(state.criteria):
        lines = [f"{c.id} [{state.statuses[k]}]: {c.text}"]
        latest = next((e for e in reversed(state.attached(k)) if e.verified_spans), None)
        if latest is not None:
            lines.append(f"  evidence ({latest.role}): {_quote(latest.verified_spans)}")
        if state.missing[k]:
            lines.append(f"  missing: {state.missing[k]}")
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


def passage_head(text: str, num_words: int) -> str:
    """The first *num_words* words of *text*, with ``...`` when cut."""
    words = text.split()
    return " ".join(words[:num_words]) + (" ..." if len(words) > num_words else "")


# ---------------------------------------------------------------------------
# Coverage judge (Delta^D)
# ---------------------------------------------------------------------------

class LLMCoverageJudge:
    """Stateful LLM judge: one call per step updates the criteria state.

    The prompt (one per query shape) shows each criterion with its status
    and its last ``max_evidence_per_criterion`` evidence passages (each as
    the cited spans that occur verbatim and the first ``head_words`` words of
    the passage; a passage attached to several criteria is shown under each) and,
    when partially covered, what it still lacks; then the step's novel
    passages (``max_passage_chars`` each).  The judge lists only the criteria
    a new passage, alone or with the attached evidence, supports or
    contradicts.  It does not see the agent's search queries.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        model_name: Saved in the meta line.
        query_shape: ``"single_target"``, ``"set"`` or ``"multi_aspect"``
            (the dataset's ``query_shape`` in layout.DATASET_SPECS); picks
            the prompt.
        max_passage_chars: Text of one new passage shown to the judge.
        head_words: Words shown from the start of an attached passage.
        max_evidence_per_criterion: Attached evidence passages shown per
            criterion (the most recent).
        max_tokens / temperature: LLM call settings.
        parse_retries: Extra calls when the output cannot be parsed; each
            failed attempt is recorded in the step's errors.
    """

    def __init__(
        self,
        llm_client: Any,
        model_name: Optional[str] = None,
        query_shape: str = SINGLE_TARGET,
        max_passage_chars: int = 3000,
        head_words: int = 100,
        max_evidence_per_criterion: int = 4,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        parse_retries: int = 1,
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        self._llm = llm_client
        self.model_name = model_name
        self.name = f"llm:{model_name}"
        self._system = CRITERIA_UPDATE_DOC_SYSTEMS[query_shape]
        self._parse_retries = parse_retries
        self._max_passage_chars = max_passage_chars
        self._head_words = head_words
        self._max_evidence = max_evidence_per_criterion
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _format_state(self, state: CriteriaState) -> str:
        blocks = []
        for k, c in enumerate(state.criteria):
            lines = [f"{c.id} [{state.statuses[k]}]: {c.text}"]
            attached = state.attached(k, self._max_evidence)
            if not attached:
                lines.append("  evidence: none")
            for e in attached:
                spans = f"spans: {_quote(e.verified_spans)} | " if e.verified_spans else ""
                lines.append(f"  - {e.role} (step {e.step}): {spans}text: {e.text}")
            if state.missing[k]:
                lines.append(f"  missing: {state.missing[k]}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _evidence(self, items: Any, docs: List[Dict[str, Any]], step: int, role: str) -> List[Evidence]:
        """One Evidence per cited passage; the spans of a passage cited twice
        are merged.  Each span (``spans``, or the older single ``span``) is
        checked on its own, and a span joined with an ellipsis is split
        first."""
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
            cited = item.get("spans", item.get("span", []))
            cited = cited if isinstance(cited, list) else [cited]
            text = doc_text(docs[i], max_length=None)
            ev = out.setdefault(_doc_id(docs[i]), Evidence(
                doc_id=_doc_id(docs[i]), step=step, role=role, spans=[], span_verified=[],
                text=passage_head(text, self._head_words),
            ))
            for piece in (p for c in cited for p in split_span(str(c))):
                matched = span_in_passage(text, piece)
                span = matched or piece
                if span not in ev.spans:
                    ev.spans.append(span)
                    ev.span_verified.append(matched is not None)
        return list(out.values())

    def judge(
        self, query: str, state: CriteriaState, docs: List[Dict[str, Any]], step: int,
    ) -> Tuple[Optional[List[CriterionUpdate]], str, List[str]]:
        """Return ``(updates, raw, errors)``; *updates* is None when the call
        or its parsing failed.  Documents without text are left out."""
        has_text = [bool(doc_text(doc, max_length=None)) for doc in docs]
        kept = [doc for doc, ok in zip(docs, has_text) if ok]
        errors = [f"coverage_judge: empty text for doc {_doc_id(d)}" for d, ok in zip(docs, has_text) if not ok]
        if not kept:
            return [], "", errors
        passages = "\n\n".join(
            f"[{i + 1}] {doc_text(doc, max_length=self._max_passage_chars)}" for i, doc in enumerate(kept)
        )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": CRITERIA_UPDATE_DOC_USER_TEMPLATE.format(
                query=query, criteria=self._format_state(state), passages=passages,
            )},
        ]
        raw, data = "", None
        for attempt in range(1 + self._parse_retries):
            try:
                raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature) or ""
            except Exception as e:
                logger.warning("LLMCoverageJudge: step call failed: %s", e)
                return None, "", errors + [f"coverage_judge: {e}"]
            data = parse_json_object(raw)
            if data is not None and isinstance(data.get("updates"), list):
                break
            errors.append(f"coverage_judge: parse (attempt {attempt + 1}): no JSON object with an 'updates' list")
        else:
            logger.warning("LLMCoverageJudge: step %d dropped, judge output not parseable", step)
            return None, raw, errors

        updates: List[CriterionUpdate] = []
        for item in data["updates"]:
            if not isinstance(item, dict):
                continue
            updates.append(CriterionUpdate(
                id=str(item.get("id", "")).strip(),
                status=str(item.get("status", "")).strip().lower(),
                support=self._evidence(item.get("support"), kept, step, "support"),
                contradict=self._evidence(item.get("contradict"), kept, step, "contradict"),
                reason=str(item.get("reason", "")),
                missing=str(item.get("missing", "")).strip(),
            ))
        return updates, raw, errors


# ---------------------------------------------------------------------------
# Query scorers (targeting state a)
# ---------------------------------------------------------------------------

class LLMQueryScorer:
    """LLM as judge: one call per step scores every (query, criterion) pair
    as 0, 0.5 or 1.  The prompt shows the criteria state (status, latest
    verified evidence, what is missing), so a query naming an entity the
    evidence already linked to a criterion counts as targeting it."""

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

    def score(self, query: str, subqueries: List[str], state: CriteriaState) -> List[List[float]]:
        """Return ``scores[j][k]`` in [0, 1] for ``subqueries[j]`` and the
        k-th criterion of *state*; raises on failure.  Masking by the
        criteria state is done by the caller."""
        criteria = state.criteria
        messages = [
            {"role": "system", "content": CRITERIA_MATCH_QUERY_SYSTEM},
            {"role": "user", "content": CRITERIA_MATCH_QUERY_USER_TEMPLATE.format(
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

def build_criteria_judges(
    llm_client: Any, model: str = "", query_shape: str = SINGLE_TARGET,
) -> Tuple[LLMCoverageJudge, LLMQueryScorer]:
    """Build ``(coverage_judge, query_scorer)`` on one LLM client (named
    *model*) for the queries of one shape."""
    if llm_client is None:
        raise ValueError("the criteria judges need an LLM client")
    return (
        LLMCoverageJudge(llm_client, model_name=model, query_shape=query_shape),
        LLMQueryScorer(llm_client, model_name=model),
    )
