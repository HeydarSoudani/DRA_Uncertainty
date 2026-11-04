"""BeliefModel: the per-step belief state for deep research agents.

With ``inform=False`` (``monitor``) it never changes the trajectory; with
``inform=True`` (``inform``) each record also carries ``belief_state_tag``, the
``<belief_state>`` tag the agent appends to its context (``belief_state``).  Per
sample:

    reset(query_id, query)   the criteria list C is created once (fixed for
                             the sample); b_0 has no observation
                             and no nugget
    observe(...)             called once at the end of every search
                             iteration with the iteration's queries and seen
                             documents; returns and stores the step record

Order inside ``observe`` (the step's documents and the agent's turn move
the belief from b_{t-1} to b_t):

    b_t (one belief update call reads the agent's turn and the documents
    not judged in earlier steps, records their observations and rewrites
    the nuggets of the criteria with new observations) ->
    retrieval gain -> intermediate answer (both for evaluation only)

Step record, one per iteration (saved as one line of
``belief/{query_id}.jsonl`` by ``evaluation.belief.save_belief``); flat scalars
first, nested detail last::

    iteration, agent_iteration,
    num_subqueries, num_docs, num_new_docs,
    new_item_precision,
    num_new_relevant, num_repeated_relevant, num_irrelevant,
    recall_so_far, num_relevant_seen, num_relevant,
    certainty,
    intermediate_answers, intermediate_answer_status,
    docs[],
    nuggets_after,
    passages[], think_observations[], doc_observations[], nugget_updates[],
    agent_turn, belief_output,
    intermediate_answer_reasoning, errors[],
    belief_state_tag                                  # inform only, not saved (the
                                                      # trajectory step has it)

``iteration`` counts the observed search iterations from 1, the same for
every agent; ``agent_iteration`` is the agent's own counter, whose base and
meaning differ per agent.

A value that could not be computed is null, never 0; the reason is in
``errors`` (or a component is not configured: no criteria, no qrels).
``docs`` is, per document of the step, ``{doc_id, seen_before}`` (its id
was shown in an earlier step); ``num_new_docs`` counts the others.
``nuggets_after`` holds, per criterion in ``criteria`` order, its nuggets
after the step (``[{value, status, evidence}]``: an answer, how likely it
is correct on the seven-word scale ``types.BELIEF_STATUSES``, and its
sources, each ``{label, stance, value}``, "d<n>" a document, "t<step>"
the agent's turn); the nuggets before the step are the previous step's.
A source's spans are saved once, in the ``doc_observations`` of the step
that observed it.
``passages`` are the documents shown to the update call (``{label,
doc_id}``; a document is labelled when first shown).
``think_observations`` / ``doc_observations`` are the observations the
call wrote this step (``{id, label, step, stance, value, [doc_id, spans,
span_verified], grounded, note}``; only grounded ones with an empty note
entered the belief), null when the call was not made or failed;
``nugget_updates`` has one ``{id, after, reason, note, issues,
[carried]}`` per criterion with a new observation; the raw reply is ``belief_output``.
``agent_turn`` is the agent's reasoning and queries of the iteration
(``agent_turn.agent_turn_text``).
``certainty`` is the certainty score s_t of the belief after the step
(``certainty.certainty``: the geometric mean over the criteria of their
most likely nugget's probability, rest and aggregate criteria left out);
null without criteria.  The answer turn is read in the meta line's
``final_update``; the meta line's ``final_certainty`` is s_T after it,
the answer criterion of a multi-hop query scored by the nugget of the
agent's answer.
``intermediate_answers`` is null when not configured or failed and ``[]``
when the model gave no answer; ``intermediate_answer_status`` says which:
``ok``, ``no_candidate`` (``[]``), ``unparsed`` (reply without answer
format), ``failed`` (call raised; see ``errors``) or ``disabled`` (not
configured, e.g. ``--add-intermediate-answer false``).
"""

import logging
from typing import Any, Dict, List, Optional, Set

# unique_docs is also imported from here by belief model training.
from utils.text_utils import doc_id as _doc_id, unique_docs

from .belief_state import render_belief_state
from .criteria import CriteriaSource
from .updater import LLMBeliefUpdater
from .signals import AnswerFn, BeliefUpdateSignal, IntermediateAnswerSignal, RetrievalGainSignal
from .types import Criterion

logger = logging.getLogger(__name__)


def _round(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


class BeliefModel:
    """Computes and stores the per-step belief of one sample at a time.

    Args:
        criteria_source: Produces the fixed criteria list at ``reset``.  None
            disables the belief.
        belief_updater: The belief update call
            (``updater.LLMBeliefUpdater``).  None: b_t is null.
        qrels: ``{query_id: {doc_id: relevance}}``, thresholded, for the
            new-item precision and the recall so far.
        intermediate_answer_fn: The agent's ``answer_from_trajectory``;
            asked every step, its answers are saved (evaluation only).
            None disables the intermediate answer.
        policy_model: Agent name; picks the intermediate answer format.
        inform: Render each step's ``<belief_state>`` tag for the agent to
            read (``--belief-mode inform``).
    """

    def __init__(
        self,
        criteria_source: Optional[CriteriaSource] = None,
        belief_updater: Optional[LLMBeliefUpdater] = None,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        intermediate_answer_fn: Optional[AnswerFn] = None,
        policy_model: str = "",
        inform: bool = False,
    ) -> None:
        self.inform = inform
        self._criteria_source = criteria_source

        self._seen_ids: Set[str] = set()
        self._retrieval_gain = RetrievalGainSignal(qrels)
        self._belief = BeliefUpdateSignal(belief_updater) if belief_updater is not None else None
        self._intermediate_answer = (
            IntermediateAnswerSignal(intermediate_answer_fn, policy_model)
            if intermediate_answer_fn is not None else None
        )

        self._criteria: List[Criterion] = []
        self._criteria_info: Dict[str, Any] = {}
        self._step = 0
        self.steps: List[Dict[str, Any]] = []
        self._final_update: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Per-sample lifecycle
    # ------------------------------------------------------------------

    @property
    def unique_doc_ids(self) -> Set[str]:
        """All unique document ids observed in the current sample."""
        return set(self._seen_ids)

    @property
    def _tracks_state(self) -> bool:
        return self._belief is not None and self._belief.active

    def reset(self, query_id: Optional[str], query: str) -> None:
        """Start a new sample: clear all state and create its criteria list."""
        self._seen_ids.clear()
        self._retrieval_gain.reset(query_id)
        self._step = 0
        self.steps = []
        self._final_update = None

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

    def criteria_summary(self) -> Dict[str, Any]:
        """The sample's criteria for the readable logs (``belief_view``):
        ``{criteria, criteria_info, num_relevant}``."""
        return {
            "criteria": [c.to_dict() for c in self._criteria],
            "criteria_info": self._criteria_info,
            "num_relevant": self._retrieval_gain.num_relevant,
        }

    def meta(self) -> Dict[str, Any]:
        """Per-sample information for the meta line of the saved file.  The
        run settings (agent, models, mode, criteria cap) are in the run's
        ``run_config.json``; ``criteria_info`` gives the model of this
        query's criteria (and ``bank: hit`` when read from the bank)."""
        return {
            "num_criteria": len(self._criteria),
            "num_iterations": len(self.steps),
            "num_unique_docs": len(self.unique_doc_ids),
            "num_relevant": self._retrieval_gain.num_relevant,
            "criteria": [c.to_dict() for c in self._criteria],
            "criteria_info": self._criteria_info,
            "final_nuggets": self._belief.nuggets() if self._tracks_state else None,
            "final_certainty": _round(self._belief.certainty()) if self._tracks_state else None,
            "final_update": self._final_update,
        }

    def finish(self, answer_turn: Optional[str]) -> None:
        """Read the agent's answer turn (``agent_turn.agent_turn_text`` with
        the answer; its label is "t<last step + 1>"); the outcome is the
        meta line's ``final_update``: ``{agent_turn, think_observations,
        nugget_updates, belief_output, errors}``, or null
        when there is no answer turn.  Called once, after the last step."""
        if not self._tracks_state or answer_turn is None:
            return
        try:
            out = self._belief.finish(self._step + 1, answer_turn)
        except Exception as e:
            logger.warning("BeliefModel: final belief update failed", exc_info=True)
            out = {"think_observations": None, "nugget_updates": [], "belief_output": None,
                   "errors": [f"belief_update: {e}"]}
        out = {k: out.get(k) for k in ("think_observations", "nugget_updates", "belief_output", "errors")}
        self._final_update = {"agent_turn": answer_turn, **out}

    def close(self) -> None:
        """Drop every model and callback reference (the updater, the agent's
        answer hook) so their GPU memory can be freed."""
        self._belief = None
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
        agent_turn: Optional[str] = None,
        agent_doc_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Update the belief for one search iteration and store its record.

        Args:
            subqueries: The iteration's search queries.
            docs: The documents the agent was shown in this iteration.
            iter_num: The agent's own iteration number (saved as
                ``agent_iteration``).
            original_query: The user query.
            trajectory: The agent's conversation so far, for the
                intermediate answer.
            agent_turn: The agent's turn of the iteration
                (``agent_turn.agent_turn_text``); the turn observations are
                read from it.  None: no turn to read.
            agent_doc_labels: Document id -> the label the agent shows the
                passage with (an agent that labels passages itself); the
                inform tag cites documents with these labels.
        """
        self._step += 1
        subqueries = [q.strip() for q in subqueries if q and q.strip()]
        errors: List[str] = []

        tracks_state = self._tracks_state

        per_doc = []
        for d in unique_docs(docs):
            i = _doc_id(d)
            per_doc.append({"doc_id": i, "seen_before": i in self._seen_ids})
        self._seen_ids.update(d["doc_id"] for d in per_doc if d["doc_id"])
        new_ids = [d["doc_id"] for d in per_doc if not d["seen_before"]]

        # --- b_t (the documents not judged in earlier steps, the turn) ----------
        update: Dict[str, Any] = {}
        if tracks_state:
            try:
                update = self._belief.score(unique_docs(docs), self._step, agent_turn)
            except Exception as e:
                logger.warning("BeliefModel: belief update failed", exc_info=True)
                errors.append(f"belief_update: {e}")
            errors.extend(update.pop("errors", []))

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

        state = self._belief.state if tracks_state else None
        record: Dict[str, Any] = {
            "iteration": self._step,
            "agent_iteration": iter_num,
            "num_subqueries": len(subqueries),
            "num_docs": len(per_doc),
            "num_new_docs": len(new_ids),
            **gain,
            "certainty": _round(self._belief.certainty()) if state is not None else None,
            "intermediate_answers": intermediate_answer["answers"] if intermediate_answer else None,
            "intermediate_answer_status": intermediate_answer_status,
            "docs": per_doc,
            "nuggets_after": state.nugget_lists() if state is not None else None,
            "passages": update.get("passages") if tracks_state else None,
            "think_observations": update.get("think_observations"),
            "doc_observations": update.get("doc_observations"),
            "nugget_updates": update.get("nugget_updates") if tracks_state else None,
            "agent_turn": agent_turn,
            "belief_output": update.get("belief_output"),
            "intermediate_answer_reasoning": intermediate_answer["reasoning"] if intermediate_answer else None,
            "errors": errors,
        }
        if self.inform:
            record["belief_state_tag"] = render_belief_state(record, state, agent_doc_labels)
        self.steps.append(record)
        return record
