"""Per-step signals of the uncertainty estimator.

Report (Section "Instantiation"), x_t = (nu^D_t, Delta^D_t, nu^q_t, a_t):

- ``DocNoveltySignal``   nu^D: fraction of the step's documents whose id was
  not seen in an earlier step.
- ``CriteriaCoverageSignal`` Delta^D: change of the criteria state caused
  by the step's novel documents; negative when coverage was lost.
- ``QueryNoveltySignal`` nu^q: novelty of the step's queries w.r.t. the
  queries of earlier steps.
- ``CriteriaTargetingSignal`` a: per criterion, the number of steps whose
  queries directly targeted it.

Extra, not part of x_t:

- ``RetrievalGainSignal``: supervised retrieval gain against qrels
  (binary new-item precision, new-item graded recall).
- ``IntermediateAnswerSignal``: the agent's own answer(s) after each turn,
  not evaluated.

Query embeddings come from the retriever's encoder
(``encode_fn_from_retriever``).  Without one (BM25, SPLADE, endpoint
retrievers) nu^q is null.  The criteria signals take an LLM judge from ``judges``.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from deep_research_agents.prompts.answer_prompts import extract_answer_candidates, extract_bare_answer
from reasoner_component import REASONING_FALLBACK_PREFIX
from utils.text_utils import doc_id as _doc_id
from .criteria import CriteriaState
from .judges import LLMCoverageJudge, LLMQueryScorer
from .prompts import intermediate_answer_instruction
from .types import STATUS_VALUE, Criterion

logger = logging.getLogger(__name__)

EncodeFn = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_fn_from_retriever(retriever) -> Tuple[Optional[EncodeFn], Optional[str]]:
    """Return ``(encode_fn, encoder_name)`` from a retriever.

    ``encode_fn(texts, is_query=True) -> np.ndarray`` of shape (N, D).  Works
    with ``DenseRetriever`` (has ``.encoder.encode``); ``(None, None)`` for
    retrievers without a local encoder.
    """
    encoder = getattr(retriever, "encoder", None)
    if encoder is None or not callable(getattr(encoder, "encode", None)):
        return None, None

    def _encode(texts: List[str], is_query: bool = True) -> np.ndarray:
        return np.asarray(encoder.encode(texts, is_query=is_query), dtype=np.float32)

    return _encode, getattr(encoder, "model_name", None)


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def _max_sim(emb: np.ndarray, bank: List[np.ndarray]) -> Optional[float]:
    """Max cosine similarity of a unit vector to a list of unit vectors."""
    if not bank:
        return None
    return float(np.max(np.stack(bank) @ emb))


def _novelty(max_sim: Optional[float]) -> float:
    """1 - max similarity, clipped to [0, 1]; 1 when there is nothing to compare."""
    if max_sim is None:
        return 1.0
    return float(1.0 - min(max(max_sim, 0.0), 1.0))


def _mean(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# nu^D: document novelty
# ---------------------------------------------------------------------------

class DocNoveltySignal:
    """Fraction of the step's documents whose id was not seen in an earlier step.

    Per document: 1 for a new id, 0 for an id seen in an earlier step.  The
    step value is the mean, so with k unique documents it is a multiple of 1/k.
    """

    def __init__(self) -> None:
        self._seen_ids: Set[str] = set()

    @property
    def seen_ids(self) -> Set[str]:
        return self._seen_ids

    def reset(self) -> None:
        self._seen_ids.clear()

    def score(self, docs: List[Dict[str, Any]]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """Return ``(step_novelty, per_doc)``; ``step_novelty`` is None without docs."""
        ids: List[str] = []
        for doc in docs:
            doc_id = _doc_id(doc)
            if doc_id and doc_id not in ids:
                ids.append(doc_id)
        if not ids:
            return None, []

        per_doc = [
            {"doc_id": d, "seen_before": d in self._seen_ids, "novelty": 0.0 if d in self._seen_ids else 1.0}
            for d in ids
        ]
        # Update state after scoring: novelty is w.r.t. earlier steps only.
        self._seen_ids.update(ids)
        return _mean([d["novelty"] for d in per_doc]), per_doc


# ---------------------------------------------------------------------------
# nu^q: query novelty
# ---------------------------------------------------------------------------

class QueryNoveltySignal:
    """Novelty of the step's queries w.r.t. queries of earlier steps.

    Per query: ``1 - max cosine similarity`` to the queries of earlier steps
    (1 for the first step).  The step value is the mean over the step's
    queries.  None without an encoder.
    """

    def __init__(self, encode_fn: Optional[EncodeFn] = None) -> None:
        self._encode_fn = encode_fn
        self._earlier_embs: List[np.ndarray] = []

    def reset(self) -> None:
        self._earlier_embs.clear()

    def close(self) -> None:
        """Drop the encoder reference; nu^q is null from then on."""
        self._encode_fn = None

    def score(self, subqueries: List[str]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """Return ``(step_novelty, per_query)``."""
        if not subqueries:
            return None, []
        if self._encode_fn is None:
            return None, [{"text": q, "max_sim_to_earlier": None, "novelty": None} for q in subqueries]

        embs = _normalise_rows(self._encode_fn(subqueries, is_query=True))
        per_query: List[Dict[str, Any]] = []
        for q, emb in zip(subqueries, embs):
            max_sim = _max_sim(emb, self._earlier_embs)
            per_query.append({
                "text": q,
                "max_sim_to_earlier": round(max_sim, 4) if max_sim is not None else None,
                "novelty": round(_novelty(max_sim), 4),
            })
        self._earlier_embs.extend(embs)
        return _mean([p["novelty"] for p in per_query]), per_query


# ---------------------------------------------------------------------------
# Delta^D: criteria coverage
# ---------------------------------------------------------------------------

class CriteriaCoverageSignal:
    """Change of the criteria state caused by the step's novel documents.

    Delta^D = sum_k (sigma_t(k) - sigma_{t-1}(k)) with uncovered = 0,
    partially = 1, fully covered = 2.  The stateful judge sees the current
    state with its attached evidence and the step's novel documents, so
    sigma can go up (new or combined evidence) or down (a contradiction);
    Delta^D is negative when coverage was lost.  With no novel document
    Delta^D is 0 and the judge is not called; when the judge fails the
    state is unchanged and Delta^D is null.
    """

    def __init__(self, judge: LLMCoverageJudge) -> None:
        self.judge = judge
        self._query = ""
        self._state: Optional[CriteriaState] = None

    @property
    def active(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> Optional[CriteriaState]:
        return self._state

    def statuses(self) -> Optional[List[str]]:
        return self._state.snapshot() if self._state is not None else None

    def reset(self, query: str, criteria: List[Criterion]) -> None:
        self._query = query
        self._state = CriteriaState(criteria) if criteria else None

    def score(
        self, new_docs: List[Dict[str, Any]], step: int,
    ) -> Tuple[Optional[int], List[Dict[str, Any]], Optional[str], List[str]]:
        """Return ``(delta, updates, raw, errors)``; *updates* has one record
        per update the judge proposed (``CriteriaState.apply``), *raw* is the
        judge's reply (None when it was not called)."""
        if self._state is None:
            return None, [], None, []
        if not new_docs:
            return 0, [], None, []
        before = self._state.snapshot()
        updates, raw, errors = self.judge.judge(self._query, self._state, new_docs, step)
        if updates is None:
            return None, [], raw, errors
        records = self._state.apply(updates)
        after = self._state.snapshot()
        delta = sum(STATUS_VALUE[a] - STATUS_VALUE[b] for b, a in zip(before, after))
        return delta, records, raw, errors


# ---------------------------------------------------------------------------
# a^q: criteria targeting state
# ---------------------------------------------------------------------------

class CriteriaTargetingSignal:
    """Targeting state: how many steps have aimed a query at each criterion.

    The scorer scores every (query, criterion) pair as 0, 0.5 or 1; one
    query may target several criteria.  A criterion is targeted in a step
    when at least one of the step's queries scores 1 on it (0.5 never
    counts), and its attempt count a_t(k) then grows by 1, once per step
    however many queries target it.  Covered criteria are counted too.
    """

    def __init__(self, scorer: LLMQueryScorer) -> None:
        self.scorer = scorer
        self._query = ""
        self._criteria: List[Criterion] = []
        self._attempts: List[int] = []

    @property
    def attempts(self) -> List[int]:
        return list(self._attempts)

    def reset(self, query: str, criteria: List[Criterion]) -> None:
        self._query = query
        self._criteria = list(criteria)
        self._attempts = [0] * len(self._criteria)

    def score(
        self, subqueries: List[str], state: Optional[CriteriaState],
    ) -> Tuple[Optional[List[str]], List[Dict[str, Any]]]:
        """Return ``(targeted_ids, per_query)``; per query ``target_scores``
        (one per criterion).  *state* is sigma_{t-1}, shown to the scorer.
        ``targeted_ids`` is None, and the attempts are unchanged, without
        queries, criteria or state."""
        if not subqueries or not self._criteria or state is None:
            return None, []
        scores = self.scorer.score(self._query, subqueries, state)
        targeted = [k for k in range(len(self._criteria)) if any(row[k] >= 1.0 for row in scores)]
        for k in targeted:
            self._attempts[k] += 1
        return [self._criteria[k].id for k in targeted], [{"target_scores": row} for row in scores]


# ---------------------------------------------------------------------------
# Supervised retrieval gain (extra)
# ---------------------------------------------------------------------------

class RetrievalGainSignal:
    """Supervised retrieval gain of each step against the qrels.

    Binary (``qrels``, relevant = grade >= ``min_relevance_score``), over the
    step's unique doc ids:

    - ``new_item_precision``: newly seen relevant docs / the step's docs
      (0, 0.2, ..., 1 for 5 docs).
    - ``num_new_relevant``, ``num_repeated_relevant``, ``num_irrelevant``.
    - ``recall_so_far``: ``num_relevant_seen / num_relevant``, the relevant
      docs seen up to and including this step.

    Graded (``graded_qrels``, ``{query_id: {doc_id: gain}}`` with the official
    gains):

    - ``new_item_graded_recall``: ``new_gain / total_gain``.  Summed over the
      steps it is the trajectory's GradedRecall@N.
    - ``new_gain``: summed gain of the relevant docs first seen in this step.
    - ``total_gain``: summed gain of all relevant docs of the query.
    - ``graded_recall_so_far``: ``gain_seen / total_gain``, the gain seen up
      to and including this step.

    The binary fields are null when the query has no relevant doc in the
    qrels, the graded one when it has none in the graded qrels; all are null
    when the step has no docs.  Call :meth:`reset` at the start of each query.
    """

    _NULL = {
        "new_item_precision": None,
        "num_new_relevant": None,
        "num_repeated_relevant": None,
        "num_irrelevant": None,
        "recall_so_far": None,
        "num_relevant_seen": None,
        "num_relevant": None,
        "new_item_graded_recall": None,
        "new_gain": None,
        "total_gain": None,
        "graded_recall_so_far": None,
        "gain_seen": None,
    }

    def __init__(
        self,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        graded_qrels: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._qrels = qrels or {}
        self._graded_qrels = graded_qrels or {}
        self._relevant_ids: Set[str] = set()
        self._relevant_seen: Set[str] = set()
        self._gains: Dict[str, int] = {}
        self._gain_seen: Set[str] = set()

    @property
    def num_relevant(self) -> Optional[int]:
        """Relevant docs of the current query; None without qrels for it."""
        return len(self._relevant_ids) or None

    @property
    def total_gain(self) -> Optional[int]:
        """Summed gain of the current query's relevant docs; None without graded qrels."""
        return sum(self._gains.values()) or None

    def reset(self, query_id: Optional[str] = None) -> None:
        """Reset per-query state and load the relevant ids and gains of ``query_id``."""
        self._relevant_seen.clear()
        self._gain_seen.clear()
        judged = self._qrels.get(query_id, {}) if query_id else {}
        self._relevant_ids = {d for d, r in judged.items() if float(r) > 0}
        graded = self._graded_qrels.get(query_id, {}) if query_id else {}
        self._gains = {d: int(g) for d, g in graded.items() if int(g) > 0}

    def score(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the step's gain statistics (see the class docstring)."""
        ids = {_doc_id(doc) for doc in docs}
        ids.discard("")
        result = dict(self._NULL)
        if not ids:
            return result
        if self._relevant_ids:
            relevant = ids & self._relevant_ids
            new = relevant - self._relevant_seen
            self._relevant_seen.update(new)
            result.update({
                "new_item_precision": round(len(new) / len(ids), 4),
                "num_new_relevant": len(new),
                "num_repeated_relevant": len(relevant) - len(new),
                "num_irrelevant": len(ids) - len(relevant),
                "recall_so_far": round(len(self._relevant_seen) / self.num_relevant, 4),
                "num_relevant_seen": len(self._relevant_seen),
                "num_relevant": self.num_relevant,
            })
        if self._gains:
            new_graded = (ids & self._gains.keys()) - self._gain_seen
            self._gain_seen.update(new_graded)
            gain = sum(self._gains[d] for d in new_graded)
            gain_seen = sum(self._gains[d] for d in self._gain_seen)
            result.update({
                "new_item_graded_recall": round(gain / self.total_gain, 4),
                "new_gain": gain,
                "total_gain": self.total_gain,
                "graded_recall_so_far": round(gain_seen / self.total_gain, 4),
                "gain_seen": gain_seen,
            })
        return result


# ---------------------------------------------------------------------------
# Intermediate answer (extra)
# ---------------------------------------------------------------------------

# (original_query, trajectory, instruction) -> raw model output; the agent's
# ``answer_from_trajectory``.
AnswerFn = Callable[[str, Any, str], str]


class IntermediateAnswerSignal:
    """The agent's own answer(s) at the end of each turn, not evaluated.

    Asks the agent's model, through ``answer_fn``, for its most likely
    answer given the trajectory so far, in the agent's answer format.  The
    model may give one answer, several (a list) or none ("no candidate").
    A reply that is only the answer value (one short line, no tags, e.g. a
    bare ``no candidate``) is accepted as is.

    ``score`` returns ``{"answers": [...], "reasoning": str}``; ``answers``
    is empty when the model gave no answer.  A stated confidence is ignored.
    It returns ``None`` when the reply has no answer format, and raises
    when the call fails.

    Args:
        answer_fn: The agent's ``answer_from_trajectory``.
        agentic_model: Agent name; picks the answer format and the parser.
    """

    def __init__(self, answer_fn: AnswerFn, agentic_model: str) -> None:
        self._answer_fn = answer_fn
        self._agentic_model = agentic_model

    def score(self, original_query: str, trajectory: Any) -> Optional[Dict[str, Any]]:
        instruction = intermediate_answer_instruction(self._agentic_model)
        raw = self._answer_fn(original_query, trajectory, instruction) or ""
        if raw.startswith(REASONING_FALLBACK_PREFIX):
            raw = raw[len(REASONING_FALLBACK_PREFIX):]
        outputs, format_matched = extract_answer_candidates(raw, expected_format=self._agentic_model)
        if not format_matched:
            outputs = extract_bare_answer(raw)
            if outputs is None:
                logger.warning("Intermediate answer without answer format: %r", raw.strip()[:200])
                return None
        answers = [o.candidate for o in outputs if o.candidate.lower() != "no candidate"]
        first = outputs[0] if outputs else None
        return {
            "answers": answers,
            "reasoning": first.reasoning if first else "",
        }
