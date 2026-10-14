"""BeliefModel: the per-step belief state and signals for deep research agents.

With ``inform=False`` (``monitor``) it never changes the trajectory; with
``inform=True`` (``inform``) each record also carries ``belief_state_tag``, the
``<belief_state>`` tag the agent appends to its context (``belief_state``).  Per
sample:

    reset(query_id, query)   the criteria list C is created once (fixed for
                             the sample) and b_0 labels every criterion
                             "almost no chance"
    observe(...)             called once at the end of every search
                             iteration with the iteration's queries and seen
                             documents; returns and stores the step record

Order inside ``observe`` (the targeting scorer sees b_{t-1}; the documents
then move the belief to b_t):

    a (targeting) -> nu^q -> nu^D -> b_t and Delta^D (the belief updater
    rewrites the belief from every document of the step) -> retrieval gain
    -> intermediate answer

Step record, one per iteration (saved as one line of
``belief/{query_id}.jsonl`` by ``evaluation.belief.save_belief``); flat scalars
first, nested detail last::

    iteration, agent_iteration,
    num_subqueries, num_docs, num_new_docs,
    doc_novelty, belief_delta, query_novelty          # x_t, with criteria_attempts_after
    new_item_precision,
    num_new_relevant, num_repeated_relevant, num_irrelevant,
    recall_so_far, num_relevant_seen, num_relevant,
    intermediate_answers, intermediate_answer_status,
    subqueries[], queries[], docs[],
    belief_before, belief_after, criteria_sources_after,
    criteria_targeted, criteria_attempts_after, criteria_updates[],
    belief_updater_output, intermediate_answer_reasoning, errors[],
    belief_state_tag                                  # inform only; null when empty

``iteration`` counts the observed search iterations from 1, the same for
every agent; ``agent_iteration`` is the agent's own counter, whose base and
meaning differ per agent.

A signal that could not be computed is null, never 0; the reason is in
``errors`` (or a component is not configured: no encoder, no criteria,
no qrels).  ``belief_delta`` is negative when the belief fell.
``belief_before`` / ``belief_after`` hold one WEP label per
criterion (``types.WEP_LABELS``), in ``criteria`` order.
``criteria_targeted`` lists the ids of the criteria that a query of the
step targeted directly (score 1); ``criteria_attempts_after`` is, per
criterion, the number of steps that targeted it so far.  When the targeting
scorer fails, ``criteria_targeted`` is null and the attempts are unchanged;
both are null when it is not configured.
``criteria_sources_after`` is, per criterion, its distinct supporting
documents so far (null without a criteria state).
``criteria_updates`` has one ``{id, from, to, written, support[],
contradict[], reason, note}`` per criterion (``BeliefState.update``);
``belief_updater_output`` is the updater's raw reply (null when it was not
called).
``intermediate_answers`` is null when not configured or failed and ``[]``
when the model gave no answer; ``intermediate_answer_status`` says which:
``ok``, ``no_candidate`` (``[]``), ``unparsed`` (reply without answer
format), ``failed`` (call raised; see ``errors``) or ``disabled`` (not
configured, e.g. ``--add-intermediate-answer false``).
"""

import logging
from typing import Any, Dict, List, Optional, Set

from utils.text_utils import doc_id

from .belief_state import render_belief_state
from .criteria import CriteriaSource
from .judges import LLMBeliefUpdater, LLMQueryScorer
from .signals import (
    AnswerFn,
    BeliefUpdateSignal,
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


def _unique(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """*docs* without repeated ids, first occurrence kept (docs without an
    id are all kept)."""
    seen: Set[str] = set()
    out = []
    for d in docs:
        key = doc_id(d)
        if key and key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


class BeliefModel:
    """Computes and stores the per-step signals of one sample at a time.

    Args:
        criteria_source: Produces the fixed criteria list at ``reset``.  None
            disables the belief and the criteria-based signals.
        belief_updater: Stateful belief updater (``judges``).  None:
            b_t, Delta^D and the attempts are null.
        query_scorer: Targeting scorer of queries vs criteria.  None:
            the attempts are null.
        encode_fn: ``(texts, is_query) -> np.ndarray`` from the retriever
            (``encode_fn_from_retriever``).  None: nu^q is null.
        encoder_name: Saved in the meta line.
        qrels: ``{query_id: {doc_id: relevance}}``, thresholded, for the
            new-item precision and the recall so far.
        intermediate_answer_fn: The agent's ``answer_from_trajectory``;
            asked every step, its answers are saved (extra, not in x_t).
            None disables the intermediate answer.
        agentic_model: Agent name; picks the intermediate answer format.
        run_info: Run settings saved in every meta line (agent, dataset,
            belief model configuration), so a file can be read on its own.
        inform: Render each step's ``<belief_state>`` tag for the agent to
            read (``--belief-mode inform``).
    """

    def __init__(
        self,
        criteria_source: Optional[CriteriaSource] = None,
        belief_updater: Optional[LLMBeliefUpdater] = None,
        query_scorer: Optional[LLMQueryScorer] = None,
        encode_fn: Optional[EncodeFn] = None,
        encoder_name: Optional[str] = None,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
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
        self._retrieval_gain = RetrievalGainSignal(qrels)
        self._belief = BeliefUpdateSignal(belief_updater) if belief_updater is not None else None
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
        return self._belief is not None and self._belief.active

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
                logger.warning("BeliefModel: criteria source failed", exc_info=True)
                self._criteria_info = {"errors": [f"criteria_source: {e}"]}
            logger.info("BeliefModel: %d criteria: %s",
                        len(self._criteria), [c.text for c in self._criteria])
        if self._belief is not None:
            self._belief.reset(query, self._criteria)
        if self._targeting is not None:
            self._targeting.reset(query, self._criteria)

    def meta(self) -> Dict[str, Any]:
        """Per-sample information for the meta line of the saved file."""
        return {
            **self._run_info,
            "criteria_source": self._criteria_source.name if self._criteria_source else None,
            "belief_updater": self._belief.updater.name if self._belief else None,
            "query_scorer": self._targeting.scorer.name if self._targeting else None,
            "encoder": self._encoder_name,
            "num_criteria": len(self._criteria),
            "num_iterations": len(self.steps),
            "num_unique_docs": len(self.unique_doc_ids),
            "num_relevant": self._retrieval_gain.num_relevant,
            "criteria": [c.to_dict() for c in self._criteria],
            "criteria_info": self._criteria_info,
            "final_belief": self._belief.labels() if self._tracks_state else None,
            "final_criteria_attempts": self._targeting.attempts if self._tracks_targeting else None,
            "criteria_evidence": self._belief.state.evidence() if self._tracks_state else None,
        }

    def close(self) -> None:
        """Drop every model and callback reference (encoder, judges, the
        agent's answer hook) so their GPU memory can be freed."""
        self._query_novelty.close()
        self._belief = None
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
        state_before = self._belief.labels() if tracks_state else None

        # --- a: targeting state (the scorer sees b_{t-1}) -----------------------
        criteria_targeted, attempts_after, per_target = None, None, []
        if self._tracks_targeting:
            try:
                criteria_targeted, per_target = self._targeting.score(subqueries, self._belief.state)
            except Exception as e:
                logger.warning("BeliefModel: criteria targeting failed", exc_info=True)
                errors.append(f"criteria_targeting: {e}")
            attempts_after = self._targeting.attempts

        # --- nu^q -------------------------------------------------------------
        query_novelty, per_query = None, [{"text": q} for q in subqueries]
        try:
            query_novelty, per_query = self._query_novelty.score(subqueries)
        except Exception as e:
            logger.warning("BeliefModel: query novelty failed", exc_info=True)
            errors.append(f"query_novelty: {e}")
        for j, entry in enumerate(per_query):
            entry.update(per_target[j] if j < len(per_target) else {"target_scores": None})

        # --- nu^D -------------------------------------------------------------
        doc_novelty, per_doc = None, []
        try:
            doc_novelty, per_doc = self._doc_novelty.score(docs)
        except Exception as e:
            logger.warning("BeliefModel: doc novelty failed", exc_info=True)
            errors.append(f"doc_novelty: {e}")
        new_ids = [d["doc_id"] for d in per_doc if not d["seen_before"]]

        # --- b_t and Delta^D (every document of the step) -----------------------
        belief_delta, updates, updater_output, state_after = None, None, None, state_before
        if tracks_state:
            try:
                belief_delta, updates, updater_output, updater_errors = self._belief.score(
                    _unique(docs), self._step,
                )
                errors.extend(f"belief_update: {e}" for e in updater_errors)
            except Exception as e:
                logger.warning("BeliefModel: belief update failed", exc_info=True)
                errors.append(f"belief_update: {e}")
            state_after = self._belief.labels()

        # --- Retrieval gain (extra) ---------------------------------------------
        try:
            gain = self._retrieval_gain.score(docs)
        except Exception as e:
            logger.warning("BeliefModel: retrieval gain failed", exc_info=True)
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
                logger.warning("BeliefModel: intermediate answer failed: %s", e)
                errors.append(f"intermediate_answer: {e}")
                intermediate_answer_status = "failed"

        record: Dict[str, Any] = {
            "iteration": self._step,
            "agent_iteration": iter_num,
            "num_subqueries": len(subqueries),
            "num_docs": len(per_doc),
            "num_new_docs": len(new_ids),
            "doc_novelty": _round(doc_novelty),
            "belief_delta": belief_delta,
            "query_novelty": _round(query_novelty),
            **gain,
            "intermediate_answers": intermediate_answer["answers"] if intermediate_answer else None,
            "intermediate_answer_status": intermediate_answer_status,
            "subqueries": subqueries,
            "queries": per_query,
            "docs": per_doc,
            "belief_before": state_before,
            "belief_after": state_after,
            "criteria_sources_after": self._belief.state.sources() if tracks_state else None,
            "criteria_targeted": criteria_targeted,
            "criteria_attempts_after": attempts_after,
            "criteria_updates": updates,
            "belief_updater_output": updater_output,
            "intermediate_answer_reasoning": intermediate_answer["reasoning"] if intermediate_answer else None,
            "errors": errors,
        }
        if self.inform:
            record["belief_state_tag"] = render_belief_state(record, self._criteria)
        self.steps.append(record)
        return record
