"""BeliefModel: the per-sample belief of a deep research agent.

``reset`` makes the criteria of a sample, ``observe`` updates the belief
once per search iteration and stores the step record, ``finish`` reads the
answer turn and stores the final record.  With ``inform=True`` each record
also carries the ``<belief_state>`` tag for the agent.  The flow and every
field of the step and final records are described in ``README.md``."""

import logging
from typing import Any, Dict, List, Optional, Set

from utils.text_utils import doc_id as _doc_id, unique_docs

from .core.certainty import answer_certainty
from .core.types import Criterion, is_computed
from .criteria.sources import CriteriaSource
from .eval_signals.intermediate_answer import AnswerFn, IntermediateAnswerSignal
from .eval_signals.retrieval_gain import RetrievalGainSignal
from .update.final_score import LLMFinalScorer, check_issues
from .update.signal import BeliefUpdateSignal
from .update.updater import LLMBeliefUpdater
from .views.tag import render_belief_state

logger = logging.getLogger(__name__)


def _round(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


class BeliefModel:
    """Computes and stores the per-step belief of one sample at a time.

    Args:
        criteria_source: Produces the fixed criteria list at ``reset``.  None
            disables the belief.
        belief_updater: The belief update call
            (``update.updater.LLMBeliefUpdater``).  None: b_t is null.
        final_scorer: The final scorer call (``update.final_score.LLMFinalScorer``)
            that gives s_T; None: the answer turn is one more update and
            s_T the certainty of the belief after it (a report).
        qrels: ``{query_id: {doc_id: relevance}}``, thresholded, for the
            new-item precision and the recall so far.
        intermediate_answer_fn: The agent's ``answer_from_trajectory``;
            asked every step, its answers are saved (evaluation only).
            None disables the intermediate answer.
        policy_model: Agent name; picks the intermediate answer format.
        inform: Render each step's ``<belief_state>`` tag for the agent to
            read (``--belief-mode inform``).
        token_meter: The ``TokenMeter`` of the belief update and final
            scorer's client; the tokens of each sample are saved in its meta
            line (``tokens``).  None: not counted.
    """

    def __init__(
        self,
        criteria_source: Optional[CriteriaSource] = None,
        belief_updater: Optional[LLMBeliefUpdater] = None,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        intermediate_answer_fn: Optional[AnswerFn] = None,
        policy_model: str = "",
        inform: bool = False,
        final_scorer: Optional[LLMFinalScorer] = None,
        token_meter: Any = None,
    ) -> None:
        self.inform = inform
        self._token_meter = token_meter
        self._tokens_at_reset: Optional[Dict[str, int]] = None
        self._criteria_source = criteria_source
        self._final_scorer = final_scorer

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
        self._tokens_at_reset = self._token_meter.snapshot() if self._token_meter is not None else None
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
        """The sample's criteria for the readable logs (``views.log_view``):
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
        query's criteria (and ``bank: hit`` when read from the bank), and
        ``tokens`` the belief model's tokens on the sample (``tokens``)."""
        return {
            "num_criteria": len(self._criteria),
            "num_iterations": len(self.steps),
            "num_unique_docs": len(self.unique_doc_ids),
            "num_relevant": self._retrieval_gain.num_relevant,
            "criteria": [c.to_dict() for c in self._criteria],
            "criteria_info": self._criteria_info,
            "tokens": self.tokens(),
        }

    def tokens(self) -> Optional[Dict[str, int]]:
        """The tokens of the belief update and final scorer calls since
        ``reset`` (``{input_tokens, output_tokens, total_tokens,
        num_calls}``), None when not counted.  The criteria (banked, or
        made by their own client) and the intermediate answers (the
        agent's client) are not in it."""
        if self._token_meter is None or self._tokens_at_reset is None:
            return None
        now = self._token_meter.snapshot()
        return {k: now[k] - self._tokens_at_reset[k] for k in now}

    def finish(self, answer_turn: Optional[str]) -> None:
        """Read the agent's answer turn (``text.agent_turn.agent_turn_text`` with
        the answer): the final scorer's call, or (no final scorer) one more
        update, its label "t<last step + 1>"; its record is ``final`` (null
        without an answer turn).  Called once, after the last step."""
        if not self._tracks_state or answer_turn is None:
            return
        step = self._step + 1
        if self._final_scorer is not None:
            self.final = self._score_final(step, answer_turn)
            return
        errors: List[str] = []
        update: Dict[str, Any] = {}
        try:
            update = self._belief.finish(step, answer_turn)
        except Exception as e:
            logger.warning("BeliefModel: final belief update failed", exc_info=True)
            errors.append(f"belief_update: {e}")
        errors.extend(update.get("errors") or [])
        s_t = _round(self._belief.certainty())
        self.final = self._final_record(
            step, answer_turn, errors, certainty=s_t, certainty_formula=s_t, called=bool(update.get("called")),
            beliefs=update.get("beliefs") or self._belief.beliefs(), updates=update.get("updates") or [],
            changed=update.get("changed") or [], observations=update.get("observations"),
            issues=update.get("issues") or [], reply=update.get("reply"))

    def _score_final(self, step: int, answer_turn: str) -> Dict[str, Any]:
        """The final record of the final scorer's call on b_{T-1} and the
        answer turn (see the module docstring)."""
        state = self._belief.state
        errors: List[str] = []
        score, raw = None, None
        try:
            score, raw, call_errors = self._final_scorer.score(self._belief.query, state, answer_turn)
            errors.extend(f"final_score: {e}" for e in call_errors)
        except Exception as e:
            logger.warning("BeliefModel: final scorer failed", exc_info=True)
            errors.append(f"final_score: {e}")
        beliefs = self._belief.beliefs()
        kinds = [c.kind for c in state.criteria]
        computed = [is_computed(c.kind, c.text) for c in state.criteria]
        s_answer = (answer_certainty(beliefs, kinds, score.candidates, state.ruled_out_lists(), computed=computed)
                    if score else None)
        return self._final_record(
            step, answer_turn, errors,
            certainty=_round(score.probability) if score else None,
            certainty_label=score.status if score else None,
            certainty_formula=_round(self._belief.certainty()),
            certainty_answer_formula=_round(s_answer),
            called=True,
            final_score={**score.to_dict(), "issues": check_issues(score, state)} if score else None,
            beliefs=beliefs, reply=raw)

    def _final_record(self, step: int, answer_turn: str, errors: List[str], **fields: Any) -> Dict[str, Any]:
        """The final record: every key (README), *fields* over the defaults
        of a final line without a scorer value or an update."""
        state = self._belief.state
        return {
            "iteration": step,
            "certainty": None,
            "certainty_label": None,
            "certainty_formula": None,
            "certainty_answer_formula": None,
            "called": False,
            "final_score": None,
            "beliefs": None,
            "updates": [],
            "changed": [],
            "observations": None,
            "issues": [],
            "references": state.reference_lists() if state is not None else None,
            "agent_turn": answer_turn,
            "reply": None,
            **fields,
            "errors": errors,
        }

    def close(self) -> None:
        """Drop every model and callback reference (the updater, the agent's
        answer hook) so their GPU memory can be freed."""
        self._belief = None
        self._final_scorer = None
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
                (``text.agent_turn.agent_turn_text``); the turn observations are
                read from it.  None: no turn to read.
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
                update = self._belief.step(docs, self._step, agent_turn)
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
            gain = dict(RetrievalGainSignal.NULL)

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
            "called": bool(update.get("called")),
            "intermediate_answers": intermediate_answer["answers"] if intermediate_answer else None,
            "intermediate_answer_status": intermediate_answer_status,
            "docs": per_doc,
            "beliefs": update.get("beliefs") or (state.belief_lists() if state is not None else None),
            "updates": update.get("updates") or [],
            "changed": update.get("changed") or [],
            "observations": update.get("observations"),
            "issues": update.get("issues") or [],
            "agent_turn": agent_turn,
            "reply": update.get("reply"),
            "intermediate_answer_reasoning": intermediate_answer["reasoning"] if intermediate_answer else None,
            "errors": errors,
        }
        if self.inform:
            record["belief_state_tag"] = render_belief_state(record, state)
        self.steps.append(record)
        return record
