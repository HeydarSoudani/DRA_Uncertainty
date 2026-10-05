"""UncertaintyEstimator: per-step uncertainty signals for deep research agents.

With ``inform=False`` (``monitor``) it never changes the trajectory; with
``inform=True`` (``inform``) each record also carries ``certainty_tag``, the
``<certainty>`` tag the agent appends to its context (``certainty``).  Per
sample:

    reset(query_id, query)   the criteria list C is created once (fixed for
                             the sample) and sigma_0 is all uncovered
    observe(...)             called once at the end of every search
                             iteration with the iteration's queries and seen
                             documents; returns and stores the step record

Order inside ``observe`` (the targeting scorer sees sigma_{t-1}; the
documents then move the state to sigma_t):

    a (targeting) -> nu^q -> nu^D (picks the novel documents) -> Delta^D (the
    stateful judge updates sigma with the novel documents) -> retrieval
    gain -> intermediate answer

Step record, one per iteration (saved as one line of
``uncertainty/{query_id}.jsonl`` by ``UncertaintyEvaluator``); flat scalars
first, nested detail last::

    iteration, agent_iteration,
    num_subqueries, num_docs, num_new_docs,
    doc_novelty, criteria_delta, query_novelty     # x_t, with criteria_attempts_after
    new_item_precision,
    num_new_relevant, num_repeated_relevant, num_irrelevant,
    new_item_graded_recall, new_gain, total_gain,
    intermediate_answers, intermediate_answer_status,
    subqueries[], queries[], docs[],
    criteria_state_before, criteria_state_after,
    criteria_targeted, criteria_attempts_after, criteria_updates[],
    criteria_judge_output, intermediate_answer_reasoning, errors[],
    certainty_tag                                  # inform only; null when empty

``iteration`` counts the observed search iterations from 1, the same for
every agent; ``agent_iteration`` is the agent's own counter, whose base and
meaning differ per agent.

A signal that could not be computed is null, never 0; the reason is in
``errors`` (or a component is not configured: no encoder, no criteria,
no qrels).  ``criteria_delta`` is negative when coverage was lost.
``criteria_targeted`` lists the ids of the criteria that a query of the
step targeted directly (score 1); ``criteria_attempts_after`` is, per
criterion, the number of steps that targeted it so far.  When the targeting
scorer fails, ``criteria_targeted`` is null and the attempts are unchanged;
both are null when it is not configured.
``criteria_updates`` has one ``{id, from, to, proposed, support[],
contradict[], reason, missing, applied, note}`` per update the coverage judge
proposed (``CriteriaState.apply``); ``criteria_judge_output`` is the
judge's raw reply (null when it was not called).
``intermediate_answers`` is null when not configured or failed and ``[]``
when the model gave no answer; ``intermediate_answer_status`` says which:
``ok``, ``no_candidate`` (``[]``), ``unparsed`` (reply without answer
format), ``failed`` (call raised; see ``errors``) or ``disabled`` (not
configured, e.g. ``--add-intermediate-answer false``).
"""

import logging
from typing import Any, Dict, List, Optional, Set

from utils.text_utils import doc_id

from .certainty import render_certainty
from .criteria import CriteriaSource
from .judges import LLMCoverageJudge, LLMQueryScorer
from .signals import (
    AnswerFn,
    CriteriaCoverageSignal,
    CriteriaTargetingSignal,
    DocNoveltySignal,
    EncodeFn,
    IntermediateAnswerSignal,
    RetrievalGainSignal,
    QueryNoveltySignal,
)
from .types import Criterion

logger = logging.getLogger(__name__)


def _round(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


class UncertaintyEstimator:
    """Computes and stores the per-step signals of one sample at a time.

    Args:
        criteria_source: Produces the fixed criteria list at ``reset``.  None
            disables the criteria-based signals.
        coverage_judge: Stateful coverage judge (``judges``).  None:
            sigma, Delta^D and the attempts are null.
        query_scorer: Targeting scorer of queries vs criteria.  None:
            the attempts are null.
        encode_fn: ``(texts, is_query) -> np.ndarray`` from the retriever
            (``encode_fn_from_retriever``).  None: nu^q is null.
        encoder_name: Saved in the meta line.
        qrels: ``{query_id: {doc_id: relevance}}``, thresholded, for the
            binary new-item precision.
        graded_qrels: ``{query_id: {doc_id: gain}}`` with the official
            gains, for the new-item graded recall.
        intermediate_answer_fn: The agent's ``answer_from_trajectory``;
            asked every step, its answers are saved (extra, not in x_t).
            None disables the intermediate answer.
        agentic_model: Agent name; picks the intermediate answer format.
        run_info: Run settings saved in every meta line (agent, dataset,
            estimator configuration), so a file can be read on its own.
        inform: Render each step's ``<certainty>`` tag for the agent to
            read (``--uncertainty-estimator-mode inform``).
    """

    def __init__(
        self,
        criteria_source: Optional[CriteriaSource] = None,
        coverage_judge: Optional[LLMCoverageJudge] = None,
        query_scorer: Optional[LLMQueryScorer] = None,
        encode_fn: Optional[EncodeFn] = None,
        encoder_name: Optional[str] = None,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        graded_qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        intermediate_answer_fn: Optional[AnswerFn] = None,
        agentic_model: str = "",
        run_info: Optional[Dict[str, Any]] = None,
        inform: bool = False,
    ) -> None:
        self.inform = inform
        self._run_info = dict(run_info or {})
        self._criteria_source = criteria_source
        self._encoder_name = encoder_name

        self._doc_novelty = DocNoveltySignal()
        self._query_novelty = QueryNoveltySignal(encode_fn)
        self._retrieval_gain = RetrievalGainSignal(qrels, graded_qrels)
        self._coverage = CriteriaCoverageSignal(coverage_judge) if coverage_judge is not None else None
        self._targeting = CriteriaTargetingSignal(query_scorer) if query_scorer is not None else None
        self._intermediate_answer = (
            IntermediateAnswerSignal(intermediate_answer_fn, agentic_model)
            if intermediate_answer_fn is not None else None
        )

        self._criteria: List[Criterion] = []
        self._criteria_info: Dict[str, Any] = {}
        self._step = 0
        self.steps: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-sample lifecycle
    # ------------------------------------------------------------------

    @property
    def unique_doc_ids(self) -> Set[str]:
        """All unique document ids observed in the current sample."""
        return self._doc_novelty.seen_ids

    @property
    def _tracks_state(self) -> bool:
        return self._coverage is not None and self._coverage.active

    @property
    def _tracks_targeting(self) -> bool:
        return self._tracks_state and self._targeting is not None

    def reset(self, query_id: Optional[str], query: str) -> None:
        """Start a new sample: clear all state and create its criteria list."""
        self._doc_novelty.reset()
        self._query_novelty.reset()
        self._retrieval_gain.reset(query_id)
        self._step = 0
        self.steps = []

        self._criteria, self._criteria_info = [], {}
        if self._criteria_source is not None:
            try:
                self._criteria, self._criteria_info = self._criteria_source.get(query_id, query)
            except Exception as e:
                logger.warning("UncertaintyEstimator: criteria source failed", exc_info=True)
                self._criteria_info = {"errors": [f"criteria_source: {e}"]}
            logger.info("UncertaintyEstimator: %d criteria: %s",
                        len(self._criteria), [c.text for c in self._criteria])
        if self._coverage is not None:
            self._coverage.reset(query, self._criteria)
        if self._targeting is not None:
            self._targeting.reset(query, self._criteria)

    def meta(self) -> Dict[str, Any]:
        """Per-sample information for the meta line of the saved file."""
        return {
            **self._run_info,
            "criteria_source": self._criteria_source.name if self._criteria_source else None,
            "criteria_judge": self._coverage.judge.name if self._coverage else None,
            "query_scorer": self._targeting.scorer.name if self._targeting else None,
            "encoder": self._encoder_name,
            "num_criteria": len(self._criteria),
            "num_iterations": len(self.steps),
            "num_unique_docs": len(self.unique_doc_ids),
            "num_relevant": self._retrieval_gain.num_relevant,
            "total_gain": self._retrieval_gain.total_gain,
            "criteria": [c.to_dict() for c in self._criteria],
            "criteria_info": self._criteria_info,
            "final_criteria_state": self._coverage.statuses() if self._tracks_state else None,
            "final_criteria_attempts": self._targeting.attempts if self._tracks_targeting else None,
            "criteria_evidence": self._coverage.state.evidence() if self._tracks_state else None,
        }

    def close(self) -> None:
        """Drop every model and callback reference (encoder, judges, the
        agent's answer hook) so their GPU memory can be freed."""
        self._query_novelty.close()
        self._coverage = None
        self._targeting = None
        self._intermediate_answer = None

    # ------------------------------------------------------------------
    # Per-step observation
    # ------------------------------------------------------------------

    def observe(
        self,
        subqueries: List[str],
        docs: List[Dict[str, Any]],
        iter_num: int,
        original_query: str,
        trajectory: Any = None,
    ) -> Dict[str, Any]:
        """Compute the signals of one search iteration and store its record.

        Args:
            subqueries: The iteration's search queries.
            docs: The documents the agent was shown in this iteration.
            iter_num: The agent's own iteration number (saved as
                ``agent_iteration``).
            original_query: The user query.
            trajectory: The agent's conversation so far, for the
                intermediate answer.
        """
        self._step += 1
        subqueries = [q.strip() for q in subqueries if q and q.strip()]
        errors: List[str] = []

        tracks_state = self._tracks_state
        state_before = self._coverage.statuses() if tracks_state else None

        # --- a: targeting state (the scorer sees sigma_{t-1}) -------------------
        criteria_targeted, attempts_after, per_target = None, None, []
        if self._tracks_targeting:
            try:
                criteria_targeted, per_target = self._targeting.score(subqueries, self._coverage.state)
            except Exception as e:
                logger.warning("UncertaintyEstimator: criteria targeting failed", exc_info=True)
                errors.append(f"criteria_targeting: {e}")
            attempts_after = self._targeting.attempts

        # --- nu^q -------------------------------------------------------------
        query_novelty, per_query = None, [{"text": q} for q in subqueries]
        try:
            query_novelty, per_query = self._query_novelty.score(subqueries)
        except Exception as e:
            logger.warning("UncertaintyEstimator: query novelty failed", exc_info=True)
            errors.append(f"query_novelty: {e}")
        for j, entry in enumerate(per_query):
            entry.update(per_target[j] if j < len(per_target) else {"target_scores": None})

        # --- nu^D -------------------------------------------------------------
        doc_novelty, per_doc = None, []
        try:
            doc_novelty, per_doc = self._doc_novelty.score(docs)
        except Exception as e:
            logger.warning("UncertaintyEstimator: doc novelty failed", exc_info=True)
            errors.append(f"doc_novelty: {e}")
        new_ids = [d["doc_id"] for d in per_doc if not d["seen_before"]]

        # --- Delta^D and sigma_t (novel documents only) -------------------------
        # Null, not 0, when the step had no docs or nu^D failed: then the novel
        # documents are unknown.
        criteria_delta, updates, judge_output, state_after = None, None, None, state_before
        if tracks_state and doc_novelty is not None:
            try:
                by_id = {doc_id(d): d for d in docs}
                criteria_delta, updates, judge_output, judge_errors = self._coverage.score(
                    [by_id[i] for i in new_ids], self._step,
                )
                errors.extend(f"criteria_judgment: {e}" for e in judge_errors)
            except Exception as e:
                logger.warning("UncertaintyEstimator: criteria judgment failed", exc_info=True)
                errors.append(f"criteria_judgment: {e}")
            state_after = self._coverage.statuses()

        # --- Retrieval gain (extra) ---------------------------------------------
        try:
            gain = self._retrieval_gain.score(docs)
        except Exception as e:
            logger.warning("UncertaintyEstimator: retrieval gain failed", exc_info=True)
            errors.append(f"retrieval_gain: {e}")
            gain = dict(RetrievalGainSignal._NULL)

        # --- Intermediate answer (extra) ----------------------------------------
        intermediate_answer, intermediate_answer_status = None, "disabled"
        if self._intermediate_answer is not None:
            try:
                intermediate_answer = self._intermediate_answer.score(original_query, trajectory)
                if intermediate_answer is None:
                    intermediate_answer_status = "unparsed"
                else:
                    intermediate_answer_status = "ok" if intermediate_answer["answers"] else "no_candidate"
            except Exception as e:
                logger.warning("UncertaintyEstimator: intermediate answer failed: %s", e)
                errors.append(f"intermediate_answer: {e}")
                intermediate_answer_status = "failed"

        record: Dict[str, Any] = {
            "iteration": self._step,
            "agent_iteration": iter_num,
            "num_subqueries": len(subqueries),
            "num_docs": len(per_doc),
            "num_new_docs": len(new_ids),
            "doc_novelty": _round(doc_novelty),
            "criteria_delta": criteria_delta,
            "query_novelty": _round(query_novelty),
            **gain,
            "intermediate_answers": intermediate_answer["answers"] if intermediate_answer else None,
            "intermediate_answer_status": intermediate_answer_status,
            "subqueries": subqueries,
            "queries": per_query,
            "docs": per_doc,
            "criteria_state_before": state_before,
            "criteria_state_after": state_after,
            "criteria_targeted": criteria_targeted,
            "criteria_attempts_after": attempts_after,
            "criteria_updates": updates,
            "criteria_judge_output": judge_output,
            "intermediate_answer_reasoning": intermediate_answer["reasoning"] if intermediate_answer else None,
            "errors": errors,
        }
        if self.inform:
            record["certainty_tag"] = render_certainty(record, self._criteria)
        self.steps.append(record)
        return record
