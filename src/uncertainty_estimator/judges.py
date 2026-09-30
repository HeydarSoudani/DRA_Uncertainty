"""Judges behind the criteria-based signals of the uncertainty estimator.

- ``DocCriteriaJudge``: coverage of each criterion by each passage the
  agent saw in a step, one of ``STATUSES``; used for Delta^D.
  ``NLIDocJudge``  entailment probability of every (passage, criterion)
                   pair, k * t pairs run in batches, mapped to a status by
                   two thresholds.
  ``LLMDocJudge``  one LLM call per step that labels every (passage,
                   criterion) pair.
- ``QueryCriteriaScorer``: how strongly each query targets each criterion,
  in [0, 1]; used for tau^q.
  ``EmbeddingQueryScorer``  cosine similarity with the retriever encoder.
  ``LLMQueryScorer``        one LLM call per step that scores all pairs.

``build_criteria_judges`` pairs them: ``nli`` = NLIDocJudge +
EmbeddingQueryScorer (local models only), ``llm`` = LLMDocJudge +
LLMQueryScorer.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.text_utils import doc_id as _doc_id, doc_text

from ._helpers import parse_json_object
from .prompts import (
    CRITERIA_JUDGE_DOC_SYSTEM,
    CRITERIA_JUDGE_DOC_USER_TEMPLATE,
    CRITERIA_JUDGE_QUERY_SYSTEM,
    CRITERIA_JUDGE_QUERY_USER_TEMPLATE,
)
from .types import FULLY_COVERED, PARTIALLY_COVERED, STATUS_VALUE, STATUSES, UNCOVERED, Criterion, DocJudgment

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Per document: its judgment, or None with the reason when it failed.
DocJudgeResult = Tuple[List[Optional[DocJudgment]], List[str]]


def _format_criteria(criteria: List[Criterion]) -> str:
    return "\n".join(f"{c.id}: {c.text}" for c in criteria)


# ---------------------------------------------------------------------------
# Document judges (Delta^D)
# ---------------------------------------------------------------------------

class DocCriteriaJudge(ABC):
    """Judges how well each document covers each criterion."""

    name: str = ""

    @abstractmethod
    def judge(self, query: str, docs: List[Dict[str, Any]], criteria: List[Criterion]) -> DocJudgeResult:
        """Return ``(judgments, errors)``.

        ``judgments[i]`` is the ``DocJudgment`` of ``docs[i]`` (one status per
        criterion), or None when that document could not be judged; *errors*
        gives the reasons.
        """


class NLIDocJudge(DocCriteriaJudge):
    """NLI cross-encoder: passage as premise, criterion as hypothesis.

    All k * t (passage, criterion) pairs of a step are scored in batches.
    The entailment probability ``>= full_threshold`` is fully covered,
    ``>= partial_threshold`` partially covered, else uncovered.  The scores
    are saved, so the thresholds can be re-tuned offline.  A passage longer
    than the model's 512 tokens is truncated.

    Args:
        model_name: HF sequence-classification NLI model.
        device: Torch device; None = cuda when available.
        full_threshold / partial_threshold: Entailment thresholds.
        batch_size: Premise-hypothesis pairs per forward pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None,
        full_threshold: float = 0.9,
        partial_threshold: float = 0.5,
        batch_size: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not 0.0 < partial_threshold <= full_threshold <= 1.0:
            raise ValueError("expected 0 < partial_threshold <= full_threshold <= 1")
        # DeBERTa's TorchScript helpers get fused into NVRTC-compiled GPU kernels
        # after warm-up, which fails when libnvrtc-builtins is not loadable
        # (cu13 venv); run them unfused instead.
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        self._torch = torch
        self.model_name = model_name
        self.name = f"nli:{model_name}"
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self._device).eval()
        label2id = {k.lower(): v for k, v in self._model.config.label2id.items()}
        entail = [v for k, v in label2id.items() if k.startswith("entail")]
        if not entail:
            raise ValueError(f"{model_name} has no entailment label: {self._model.config.label2id}")
        self._entail_idx = entail[0]
        self._full_threshold = full_threshold
        self._partial_threshold = partial_threshold
        self._batch_size = batch_size

    def _entailment(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        probs: List[np.ndarray] = []
        for start in range(0, len(premises), self._batch_size):
            batch = self._tokenizer(
                premises[start:start + self._batch_size],
                hypotheses[start:start + self._batch_size],
                truncation="only_first", max_length=512, padding=True, return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                logits = self._model(**batch).logits
            probs.append(logits.softmax(dim=-1)[:, self._entail_idx].float().cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)

    def _status(self, score: float) -> str:
        if score >= self._full_threshold:
            return FULLY_COVERED
        if score >= self._partial_threshold:
            return PARTIALLY_COVERED
        return UNCOVERED

    def judge(self, query: str, docs: List[Dict[str, Any]], criteria: List[Criterion]) -> DocJudgeResult:
        judgments: List[Optional[DocJudgment]] = [None] * len(docs)
        errors: List[str] = []
        texts = [doc_text(doc, max_length=None) for doc in docs]
        kept = [i for i, t in enumerate(texts) if t]
        for i in set(range(len(docs))) - set(kept):
            errors.append(f"nli_judge: empty text for doc {_doc_id(docs[i])}")
        premises = [texts[i] for i in kept for _ in criteria]
        hypotheses = [c.text for _ in kept for c in criteria]
        probs = self._entailment(premises, hypotheses).reshape(len(kept), len(criteria))
        for row, i in zip(probs, kept):
            scores = [float(p) for p in row]
            judgments[i] = DocJudgment(statuses=[self._status(p) for p in scores], scores=scores)
        return judgments, errors


class LLMDocJudge(DocCriteriaJudge):
    """LLM as judge: one call per step labels every (passage, criterion) pair.

    The passages are numbered in the prompt; the judge lists only the pairs
    that are partially or fully covered, the rest are uncovered.  The judge
    does not see the criteria state: it judges only what the passages say,
    and ``CriteriaState`` does the update.  When the call fails, no passage
    of the step is judged.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        model_name: Saved in the meta line.
        max_passage_chars: Text of one passage shown to the judge.
        max_tokens / temperature: LLM call settings.
    """

    def __init__(
        self,
        llm_client: Any,
        model_name: Optional[str] = None,
        max_passage_chars: int = 3000,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm_client
        self.model_name = model_name
        self.name = f"llm:{model_name}"
        self._max_passage_chars = max_passage_chars
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _call(self, query: str, docs: List[Dict[str, Any]], criteria: List[Criterion]) -> List[DocJudgment]:
        passages = "\n\n".join(
            f"[{i + 1}] {doc_text(doc, max_length=self._max_passage_chars)}" for i, doc in enumerate(docs)
        )
        messages = [
            {"role": "system", "content": CRITERIA_JUDGE_DOC_SYSTEM},
            {"role": "user", "content": CRITERIA_JUDGE_DOC_USER_TEMPLATE.format(
                query=query, criteria=_format_criteria(criteria), passages=passages,
            )},
        ]
        raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("judgments"), list):
            raise ValueError("parse: no JSON object with a 'judgments' list")

        col = {c.id: k for k, c in enumerate(criteria)}
        statuses = [[UNCOVERED] * len(criteria) for _ in docs]
        evidence = [[""] * len(criteria) for _ in docs]
        for item in data["judgments"]:
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("passage")) - 1
            except (TypeError, ValueError):
                continue
            k = col.get(str(item.get("id", "")).strip())
            status = str(item.get("status", "")).strip().lower()
            if not 0 <= i < len(docs) or k is None or status not in STATUSES:
                continue
            if STATUS_VALUE[status] > STATUS_VALUE[statuses[i][k]]:
                statuses[i][k] = status
                evidence[i][k] = str(item.get("evidence", ""))
        return [DocJudgment(statuses=s, evidence=e) for s, e in zip(statuses, evidence)]

    def judge(self, query: str, docs: List[Dict[str, Any]], criteria: List[Criterion]) -> DocJudgeResult:
        judgments: List[Optional[DocJudgment]] = [None] * len(docs)
        kept = [i for i, doc in enumerate(docs) if doc_text(doc, max_length=None)]
        errors = [f"llm_judge: empty text for doc {_doc_id(docs[i])}" for i in range(len(docs)) if i not in kept]
        if not kept:
            return judgments, errors
        try:
            for i, judgment in zip(kept, self._call(query, [docs[i] for i in kept], criteria)):
                judgments[i] = judgment
        except Exception as e:
            logger.warning("LLMDocJudge: step call failed: %s", e)
            errors.append(f"llm_judge: {e}")
        return judgments, errors


# ---------------------------------------------------------------------------
# Query scorers (tau^q)
# ---------------------------------------------------------------------------

class QueryCriteriaScorer(ABC):
    """Scores how strongly each query targets each criterion."""

    name: str = ""

    @abstractmethod
    def score(self, query: str, subqueries: List[str], criteria: List[Criterion]) -> List[List[float]]:
        """Return ``scores[j][k]`` in [0, 1] for ``subqueries[j]`` and
        ``criteria[k]``; raises on failure.  Masking by the criteria state is
        done by the caller."""


class EmbeddingQueryScorer(QueryCriteriaScorer):
    """Cosine similarity of query and criterion embeddings, clipped to [0, 1].

    Uses the retriever encoder with ``is_query=True`` for both sides.  Dense
    encoders rarely give a similarity near 0, so the scale is compressed;
    compare tau^q across steps, not to a fixed threshold.
    """

    def __init__(self, encode_fn, encoder_name: Optional[str] = None) -> None:
        self._encode_fn = encode_fn
        self.name = f"embedding:{encoder_name}"
        self._criteria_key: Optional[Tuple[str, ...]] = None
        self._criteria_embs: Optional[np.ndarray] = None

    def _unit(self, texts: List[str]) -> np.ndarray:
        x = np.asarray(self._encode_fn(texts, is_query=True), dtype=np.float32).reshape(len(texts), -1)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    def score(self, query: str, subqueries: List[str], criteria: List[Criterion]) -> List[List[float]]:
        key = tuple(c.text for c in criteria)
        if key != self._criteria_key:
            self._criteria_embs = self._unit(list(key))
            self._criteria_key = key
        sims = self._unit(subqueries) @ self._criteria_embs.T
        # Round in float64: float32.round(4).tolist() keeps the float32 noise.
        return np.clip(sims.astype(np.float64), 0.0, 1.0).round(4).tolist()


class LLMQueryScorer(QueryCriteriaScorer):
    """LLM as judge: one call per step scores every (query, criterion) pair
    as 0, 0.5 or 1."""

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

    def score(self, query: str, subqueries: List[str], criteria: List[Criterion]) -> List[List[float]]:
        messages = [
            {"role": "system", "content": CRITERIA_JUDGE_QUERY_SYSTEM},
            {"role": "user", "content": CRITERIA_JUDGE_QUERY_USER_TEMPLATE.format(
                query=query,
                criteria=_format_criteria(criteria),
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
    kind: str,
    model: str = "",
    llm_client: Any = None,
    encode_fn=None,
    encoder_name: Optional[str] = None,
) -> Tuple[Optional[DocCriteriaJudge], Optional[QueryCriteriaScorer]]:
    """Build ``(doc_judge, query_scorer)`` for *kind* in {none, nli, llm}.

    ``nli``: *model* is the NLI model ("" = ``DEFAULT_NLI_MODEL``); the query
    scorer needs *encode_fn* and is None without it.  ``llm``: *llm_client*
    (named *model*) serves both.
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return None, None
    if kind == "nli":
        doc_judge = NLIDocJudge(model_name=model or DEFAULT_NLI_MODEL)
        if encode_fn is None:
            logger.warning("criteria_judge=nli without a retriever encoder; criteria_targeting is null")
            return doc_judge, None
        return doc_judge, EmbeddingQueryScorer(encode_fn, encoder_name)
    if kind == "llm":
        if llm_client is None:
            raise ValueError("criteria_judge=llm needs an LLM client")
        return LLMDocJudge(llm_client, model_name=model), LLMQueryScorer(llm_client, model_name=model)
    raise ValueError(f"unknown criteria_judge {kind!r}; expected 'none', 'nli' or 'llm'")
