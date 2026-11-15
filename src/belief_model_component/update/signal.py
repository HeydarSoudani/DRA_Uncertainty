"""``BeliefUpdateSignal``, b_t of each step: when to make a belief update
call (a step with a new passage, and the answer turn), what it reads
(``CallPlan``) and applying its reply.  The evaluation-only signals are in
``eval_signals``."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import doc_id as _doc_id, doc_text, doc_title

from ..core.certainty import certainty
from ..core.state import BeliefState
from ..core.types import Criterion
from ..text.passages import Passage, Turn, passage_title
from .updater import LLMBeliefUpdater

logger = logging.getLogger(__name__)


def empty_update(errors: Optional[List[str]] = None) -> Dict[str, Any]:
    """The outcome of a step without a parsed call (``BeliefUpdateSignal``)."""
    return {"called": False, "observations": None, "updates": [], "changed": [], "issues": [], "notes": [],
            "beliefs": None, "reply": None, "errors": list(errors or [])}


def _doc_key(doc: Dict[str, Any], text: Optional[str] = None) -> Optional[str]:
    """A document's key for its label: its id, else a hash of its text
    (*text*, read from *doc* when not given); None without either."""
    i = _doc_id(doc)
    if i:
        return i
    text = doc_text(doc, max_tokens=None) if text is None else text
    return "text:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else None


# Turns kept for the next call while no call is made (the most recent).
MAX_PENDING_TURNS = 6


@dataclass
class CallPlan:
    """What one belief update call reads: the agent's turns since the last
    call (``(step, text)``, labelled "t<step>") and its passages, those
    its step's search returned (new or shown before)."""
    turns: List[Turn]
    passages: List[Passage]


class BeliefUpdateSignal:
    """Belief update of the search: one call (``updater``) per step that
    has passages to read, then one on the answer turn (``finish``).

    A step makes a call when its search returned a document with text;
    the final turn always makes one.  The call shows every document of
    the step, also one shown in an earlier step: a passage gets its label
    when first shown (``BeliefState.label_doc``) and keeps it, so the
    update sees which candidates it already supports, and a passage shown
    again raises no status on its own (``BeliefState.apply``: a status
    change needs support the candidate did not have).  A step without a
    call carries the belief forward, and its turn waits for the next
    call, which reads every turn since the last one (the last
    ``MAX_PENDING_TURNS``).

    When the call fails the belief stays as it was and its turns' claims
    are lost.  A document is known by its id, else by a hash of its text
    (``_doc_key``).  ``step`` and ``finish`` return the step's outcome::

        {called, observations, updates, changed, issues, notes, beliefs, reply, errors}

    (``called`` whether a call was made; ``observations``, ``updates``, ``changed``,
    ``issues`` and ``notes`` as ``BeliefState.apply`` gives them, null / empty when the
    call was not made or failed; ``beliefs`` every criterion's belief after
    the step (``BeliefState.belief_lists``); ``reply`` the raw reply, null
    when not called or the call raised).

    The call is made in two halves, ``plan`` (label the documents, gather
    the turns) and ``apply`` (the parsed reply), so belief model training
    can make the LLM calls itself
    (``belief_model_training.episodes.BeliefTracker``).  ``certainty`` is
    the certainty score s_t of the belief (``core.certainty.certainty``).
    """

    def __init__(self, updater: LLMBeliefUpdater) -> None:
        self.updater = updater
        self._query = ""
        self._state: Optional[BeliefState] = None
        self._pending: List[Turn] = []

    @property
    def active(self) -> bool:
        return self._state is not None

    @property
    def query(self) -> str:
        return self._query

    @property
    def state(self) -> Optional[BeliefState]:
        return self._state

    def beliefs(self) -> Optional[List[Dict[str, Any]]]:
        return self._state.belief_lists() if self._state is not None else None

    def certainty(self) -> Optional[float]:
        """s_t of the current belief; None without criteria."""
        if self._state is None:
            return None
        return certainty([{"status": self._state.status(k)} for k in range(len(self._state.criteria))],
                         [c.kind for c in self._state.criteria])

    def reset(self, query: str, criteria: List[Criterion]) -> None:
        self._query = query
        self._state = BeliefState(criteria) if criteria else None
        self._pending = []

    def step(self, docs: List[Dict[str, Any]], step: int, agent_turn: Optional[str] = None) -> Dict[str, Any]:
        """The update of one step from its unique *docs* and the agent's
        turn (*agent_turn*; None: no turn to read)."""
        return self._update(self.plan(docs, step, agent_turn), step)

    def finish(self, step: int, agent_turn: str) -> Dict[str, Any]:
        """The final update: the agent's answer turn (labelled ``t<step>``)
        with the turns still waiting."""
        return self._update(self.plan([], step, agent_turn, final=True), step)

    def passages(self, docs: List[Dict[str, Any]]) -> Tuple[List[Passage], List[str]]:
        """The passages of *docs* (each labelled by its document's key,
        ``label_doc``) and the errors of documents without text."""
        passages: List[Passage] = []
        errors: List[str] = []
        for d in docs:
            text = doc_text(d, max_tokens=None)
            if not text:
                errors.append(f"empty text for doc {_doc_id(d)}")
                continue
            key = _doc_key(d, text)
            passages.append(Passage(self._state.label_doc(key, passage_title(text, doc_title(d, default=""))),
                                    _doc_id(d), text))
        return passages, errors

    def plan(self, docs: List[Dict[str, Any]], step: int, agent_turn: Optional[str],
             final: bool = False) -> Tuple[Optional[CallPlan], List[str]]:
        """The call of step *step*, or None when it makes none (``(plan,
        errors)``): its turn joins the waiting ones; the call has the
        *docs* with text, and is made when it has a passage or is the
        *final* one (with a turn).  A call takes the waiting turns."""
        if self._state is None:
            return None, []
        turn = agent_turn.strip() if agent_turn and agent_turn.strip() else None
        if turn is not None:
            self._pending.append((step, turn))
            if len(self._pending) > MAX_PENDING_TURNS:
                dropped = [t for t, _ in self._pending[:-MAX_PENDING_TURNS]]
                logger.info("BeliefUpdateSignal: %d waiting turn(s) dropped (steps %s), over %d",
                            len(dropped), dropped, MAX_PENDING_TURNS)
                self._pending = self._pending[-MAX_PENDING_TURNS:]
        passages, errors = self.passages(docs)
        if not passages and not (final and self._pending):
            return None, errors
        plan = CallPlan(turns=list(self._pending), passages=passages)
        self._pending = []
        return plan, errors

    def label_of(self, doc: Dict[str, Any]) -> Optional[str]:
        """The label of a document once it was shown, else None."""
        key = _doc_key(doc) if self._state is not None else None
        return self._state.label_for_key(key) if key else None

    def apply(self, parsed: Any, step: int) -> Dict[str, Any]:
        """Apply a parsed reply (``LLMBeliefUpdater.parse``) of the call of
        step *step*; returns the records of ``BeliefState.apply``."""
        claims, doc_obs, updates = parsed
        return self._state.apply(claims, doc_obs, updates, step)

    def _update(self, planned: Tuple[Optional[CallPlan], List[str]], step: int) -> Dict[str, Any]:
        plan, errors = planned
        out = empty_update(errors)
        if self._state is None:
            return out
        if plan is not None:
            out["called"] = True
            parsed, raw, call_errors = self.updater.update(self._query, self._state, plan.turns, plan.passages, step)
            out["reply"] = raw
            out["errors"].extend(f"belief_update: {e}" for e in call_errors)
            if parsed is not None:
                out.update(self.apply(parsed, step))
        out["beliefs"] = self.beliefs()
        return out
