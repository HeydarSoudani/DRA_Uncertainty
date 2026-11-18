"""BeliefModel: the per-sample belief of a deep research agent.

``reset`` makes the criteria of a sample, ``observe`` updates the belief
once per search iteration and stores the step record, ``finish`` reads the
answer turn and stores the final record.  The flow and every field of the
step and final records are described in ``README.md``."""

import logging
from typing import Any, Dict, List, Optional, Set

from utils.text_utils import doc_id as _doc_id, unique_docs

from .core.types import Criterion
from .criteria.sources import CriteriaSource
from .eval_signals.retrieval_gain import RetrievalGainSignal
from .update.final_score import LLMFinalScorer
from .update.signal import BeliefUpdateSignal
from .update.updater import LLMBeliefUpdater

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
        token_meter: The ``TokenMeter`` of the belief update and final
            scorer's client; the tokens of each sample are saved in its meta
            line (``tokens``).  None: not counted.
    """

    def __init__(
        self,
        criteria_source: Optional[CriteriaSource] = None,
        belief_updater: Optional[LLMBeliefUpdater] = None,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        final_scorer: Optional[LLMFinalScorer] = None,
        token_meter: Any = None,
    ) -> None:
        self._token_meter = token_meter
        self._tokens_at_reset: Optional[Dict[str, int]] = None
        self._criteria_source = criteria_source
        self._final_scorer = final_scorer

        self._seen_ids: Set[str] = set()
        self._retrieval_gain = RetrievalGainSignal(qrels)
        self._belief = BeliefUpdateSignal(belief_updater) if belief_updater is not None else None

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
        made by their own client) are not in it."""
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
            issues=update.get("issues") or [], notes=update.get("notes") or [], reply=update.get("reply"))

    def _score_final(self, step: int, answer_turn: str) -> Dict[str, Any]:
        """The final record of the final scorer's call on b_{T-1} and the
        answer turn (see the module docstring)."""
        state = self._belief.state
        errors: List[str] = []
        score, raw = None, None
        try:
            score, raw, call_errors = self._final_scorer.score(self._belief.query, state, answer_turn, step)
            errors.extend(f"final_score: {e}" for e in call_errors)
        except Exception as e:
            logger.warning("BeliefModel: final scorer failed", exc_info=True)
            errors.append(f"final_score: {e}")
        return self._final_record(
            step, answer_turn, errors,
            certainty=_round(score.probability) if score else None,
            certainty_label=score.status if score else None,
            certainty_formula=_round(self._belief.certainty()),
            called=True,
            final_score=score.to_dict() if score else None,
            beliefs=self._belief.beliefs(), reply=raw)

    def _final_record(self, step: int, answer_turn: str, errors: List[str], **fields: Any) -> Dict[str, Any]:
        """The final record: every key (README), *fields* over the defaults
        of a final line without a scorer value or an update."""
        state = self._belief.state
        return {
            "iteration": step,
            "certainty": None,
            "certainty_label": None,
            "certainty_formula": None,
            "called": False,
            "final_score": None,
            "beliefs": None,
            "updates": [],
            "changed": [],
            "observations": None,
            "issues": [],
            "notes": [],
            "references": state.reference_lists() if state is not None else None,
            "agent_turn": answer_turn,
            "reply": None,
            **fields,
            "errors": errors,
        }

    def close(self) -> None:
        """Drop every model reference (the updater, the final scorer) so
        their GPU memory can be freed."""
        self._belief = None
        self._final_scorer = None

    # ------------------------------------------------------------------
    # Per-step observation
    # ------------------------------------------------------------------

    def observe(
        self,
        subqueries: List[str],
        docs: List[Dict[str, Any]],
        iter_num: int,
        agent_turn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update the belief for one search iteration and store its record.

        Args:
            subqueries: The iteration's search queries.
            docs: The documents the agent was shown in this iteration.
            iter_num: The agent's own iteration number (saved as
                ``agent_iteration``).
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

        # --- b_t (the documents of the step, the turn) ---------------------------
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
            "docs": per_doc,
            "beliefs": update.get("beliefs") or (state.belief_lists() if state is not None else None),
            "updates": update.get("updates") or [],
            "changed": update.get("changed") or [],
            "observations": update.get("observations"),
            "issues": update.get("issues") or [],
            "notes": update.get("notes") or [],
            "agent_turn": agent_turn,
            "reply": update.get("reply"),
            "errors": errors,
        }
        self.steps.append(record)
        return record
