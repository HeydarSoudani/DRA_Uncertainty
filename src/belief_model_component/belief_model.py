"""BeliefModel: the per-step belief state for deep research agents.

With ``inform=False`` (``monitor``) it never changes the trajectory; with
``inform=True`` (``inform``) each record also carries ``belief_state_tag``, the
``<belief_state>`` tag the agent appends to its context (``belief_state``).  Per
sample:

    reset(query_id, query)   the criteria list C is created once (fixed for
                             the sample); b_0 has every statement "unknown"
    observe(...)             called once at the end of every search
                             iteration with the iteration's queries and seen
                             documents; returns and stores the step record
    finish(answer_turn)      once, after the last step: the update on the
                             agent's answer turn; stores the final record

Order inside ``observe`` (the step's documents and the agent's turn move
the belief from b_{t-1} to b_t):

    b_t (one belief update call reads the agent's turn and the documents
    not judged in earlier steps, records their observations and rewrites
    the statements of the criteria with a new observation) ->
    retrieval gain -> intermediate answer (both for evaluation only)

Step record, one per iteration (saved as one ``step`` line of
``belief/{query_id}.jsonl`` by ``evaluation.belief.save_belief``); flat scalars
first, nested detail last::

    iteration, agent_iteration,
    num_subqueries, num_docs, num_new_docs,
    new_item_precision,
    num_new_relevant, num_repeated_relevant, num_irrelevant,
    recall_so_far, num_relevant_seen, num_relevant,
    certainty,
    intermediate_answers, intermediate_answer_status,
    docs[], statements[], changed[], observations[], issues[],
    agent_turn, reply, intermediate_answer_reasoning, errors[],
    belief_state_tag                                  # inform only, not saved (the
                                                      # trajectory step has it)

``iteration`` counts the observed search iterations from 1, the same for
every agent; ``agent_iteration`` is the agent's own counter, whose base and
meaning differ per agent.

A value that could not be computed is null, never 0; the reason is in
``errors`` (or a component is not configured: no criteria, no qrels).
``docs`` is, per document of the step, ``{doc_id, new, label}``: ``new``
when its id was not shown in an earlier step (``num_new_docs`` counts
them), ``label`` the belief's label of it ("d<n>", given when the document
is first shown to the belief update; null for a document without text).
``statements`` holds, per criterion in ``criteria`` order, its statement
after the step (``{id, status, selected, text, cited, support_cited}``:
``text`` the statement as the belief update wrote it, the only part shown
to the next update and to the agent; parsed from it, its status on the
seven-word scale ``types.BELIEF_STATUSES``, its selected candidate(s), the
source labels it cites and those its status rests on, "d<n>" a document,
"t<step>" the agent's turn); ``changed`` the ids of the criteria with a
new observation in the step (their statement is the reply's).
``observations`` are the observations of the turn and of the new documents
that entered the belief in the step (``{id, label, step, [doc_id],
spans}``, verified spans only), null when the call was not made or failed;
together over the steps they are every criterion's references (the final
record gathers them).  ``issues`` one ``{id, code, detail, [label]}`` per
thing the reply got wrong (``criteria.ISSUE_*``: an observation left out,
a statement not taken or flagged).
``agent_turn`` is the agent's reasoning of the iteration as the update read
it (``agent_turn.agent_turn_text``; the queries are in the trajectory),
``reply`` the raw reply of the call.
``certainty`` is the certainty score s_t of the belief after the step
(``certainty.certainty``: the geometric mean over the criteria of their
status's probability, the rest criterion left out); null without criteria.

The final record (one ``final`` line) is the update on the answer turn
(``agent_turn.answer_turn``, labelled "t<last step + 1>"), read as any
other turn: ``{iteration, certainty, statements, changed, observations,
issues, references, agent_turn, reply, errors}``; its ``certainty`` is
s_T; ``references`` per criterion every source assigned to it over the
sample (``BeliefState.reference_lists``: ``{id, references: [{label,
step, [doc_id, title], spans}]}``), whether or not its statement cites it.
It is null without an answer turn.
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
        self.final: Optional[Dict[str, Any]] = None

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
        self.final = None

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
        }

    def finish(self, answer_turn: Optional[str]) -> None:
        """Read the agent's answer turn (``agent_turn.agent_turn_text`` with
        the answer; its label is "t<last step + 1>") as one more update; its
        record is ``final`` (null without an answer turn).  Called once,
        after the last step."""
        if not self._tracks_state or answer_turn is None:
            return
        step = self._step + 1
        errors: List[str] = []
        update: Dict[str, Any] = {}
        try:
            update = self._belief.finish(step, answer_turn)
        except Exception as e:
            logger.warning("BeliefModel: final belief update failed", exc_info=True)
            errors.append(f"belief_update: {e}")
        errors.extend(update.get("errors") or [])
        self.final = {
            "iteration": step,
            "certainty": _round(self._belief.certainty()),
            "statements": self._belief.statements(),
            "changed": update.get("changed") or [],
            "observations": update.get("observations"),
            "issues": update.get("issues") or [],
            "references": self._belief.state.reference_lists() if self._belief.state is not None else None,
            "agent_turn": answer_turn,
            "reply": update.get("reply"),
            "errors": errors,
        }

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

        docs = unique_docs(docs)
        per_doc = []
        for d in docs:
            i = _doc_id(d)
            per_doc.append({"doc_id": i, "new": i not in self._seen_ids, "label": None})
        self._seen_ids.update(d["doc_id"] for d in per_doc if d["doc_id"])
        new_ids = [d["doc_id"] for d in per_doc if d["new"]]

        # --- b_t (the documents not judged in earlier steps, the turn) ----------
        update: Dict[str, Any] = {}
        if tracks_state:
            try:
                update = self._belief.score(docs, self._step, agent_turn)
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
        if state is not None:
            for d, doc in zip(per_doc, docs):
                d["label"] = self._belief.label_of(doc)
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
            "statements": state.statement_lists() if state is not None else None,
            "changed": update.get("changed") or [],
            "observations": update.get("observations"),
            "issues": update.get("issues") or [],
            "agent_turn": agent_turn,
            "reply": update.get("reply"),
            "intermediate_answer_reasoning": intermediate_answer["reasoning"] if intermediate_answer else None,
            "errors": errors,
        }
        if self.inform:
            record["belief_state_tag"] = render_belief_state(record, state, agent_doc_labels)
        self.steps.append(record)
        return record
