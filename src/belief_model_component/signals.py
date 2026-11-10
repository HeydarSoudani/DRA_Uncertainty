"""Per-step parts of the belief model:

- ``BeliefUpdateSignal`` b_t: the belief update of the step (one call when
  the step has a new passage: the claims of the
  agent's turns and the observations of the passages, then the candidate
  updates).

Extra, for evaluation only (never shown to the agent):

- ``RetrievalGainSignal``: supervised retrieval gain against the qrels
  (new-item precision, recall so far).
- ``IntermediateAnswerSignal``: the agent's own answer(s) after each turn,
  not evaluated.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from deep_research_agents.prompts.answer_prompts import extract_answer_candidates, extract_bare_answer
from reasoner_component import REASONING_FALLBACK_PREFIX
from utils.text_utils import doc_id as _doc_id, doc_text, doc_title
from .certainty import certainty
from .criteria import BeliefState
from .updater import LLMBeliefUpdater, Passage, Turn, passage_title
from .prompts import intermediate_answer_instruction
from .types import Criterion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# b_t: belief update
# ---------------------------------------------------------------------------

def _doc_key(doc: Dict[str, Any], text: Optional[str] = None) -> Optional[str]:
    """A document's key for its label: its id, else a hash of its text
    (*text*, read from *doc* when not given); None without either."""
    i = _doc_id(doc)
    if i:
        return i
    text = doc_text(doc, max_length=None) if text is None else text
    return "text:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else None


# Turns kept for the next call while no call is made (the most recent).
MAX_PENDING_TURNS = 6


@dataclass
class CallPlan:
    """What one belief update call reads: the agent's turns since the last
    call (``(step, text)``, labelled "t<step>") and its passages, those
    new to the belief."""
    turns: List[Turn]
    passages: List[Passage]


class BeliefUpdateSignal:
    """Belief update of the search: one call (``updater``) per step that
    has something new to read, then one on the answer turn (``finish``).

    A step makes a call when it has a document not judged in an earlier
    step (with text); the final turn always makes one.  A judged passage
    is never shown again: the reasoning about it reaches the next call
    through the turns.  A step without a call carries the belief
    forward, and its turn waits for the next call, which reads every turn
    since the last one (the last ``MAX_PENDING_TURNS``).  Each document
    gets its label when first shown (``BeliefState.label_doc``), and the
    reply is applied by ``BeliefState.apply``.

    When the call fails the belief stays as it was, the call's new
    documents stay unjudged (a later step that retrieves them shows them
    again, with the same labels) and its turns' claims are lost.  ``score``
    and ``finish`` return the step's outcome::

        {called, passages, observations, updates, changed, issues, beliefs, reply, errors}

    (``called`` whether a call was made; ``passages`` ``[{label, doc_id}]``
    as shown; ``observations``, ``updates``, ``changed`` and
    ``issues`` as ``BeliefState.apply`` gives them, null / empty when the
    call was not made or failed; ``beliefs`` every criterion's belief after
    the step (``BeliefState.belief_lists``); ``reply`` the raw reply, null
    when not called or the call raised).

    The call is made in two halves, ``plan`` (label the documents, gather
    the turns) and ``apply`` (the parsed reply), so belief model training
    can make the LLM calls itself
    (``belief_model_training.episodes.BeliefTracker``).  ``certainty`` is
    the certainty score s_t of the belief (``certainty.certainty``).
    """

    def __init__(self, updater: LLMBeliefUpdater) -> None:
        self.updater = updater
        self._query = ""
        self._state: Optional[BeliefState] = None
        self._judged: Set[str] = set()
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
        self._judged.clear()
        self._pending = []

    def unjudged(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The documents of *docs* not judged yet (documents without an id
        are shown every step)."""
        return [d for d in docs if not _doc_id(d) or _doc_id(d) not in self._judged]

    def score(self, docs: List[Dict[str, Any]], step: int, agent_turn: Optional[str] = None) -> Dict[str, Any]:
        """The update of one step from its unique *docs* and the agent's
        turn (*agent_turn*; None: no turn to read)."""
        return self._update(self.plan(docs, step, agent_turn), step)

    def finish(self, step: int, agent_turn: str) -> Dict[str, Any]:
        """The final update: the agent's answer turn (labelled ``t<step>``)
        with the turns still waiting."""
        return self._update(self.plan([], step, agent_turn, final=True), step)

    def passages(self, docs: List[Dict[str, Any]]) -> Tuple[List[Passage], List[str]]:
        """The passages of *docs* (each labelled, ``label_doc``) and the
        errors of documents without text."""
        passages: List[Passage] = []
        errors: List[str] = []
        for d in docs:
            text = doc_text(d, max_length=None)
            if not text:
                errors.append(f"empty text for doc {_doc_id(d)}")
                continue
            passages.append(Passage(self._state.label_doc(_doc_key(d, text), passage_title(text, doc_title(d, default=""))), _doc_id(d), text))
        return passages, errors

    def plan(self, docs: List[Dict[str, Any]], step: int, agent_turn: Optional[str],
             final: bool = False) -> Tuple[Optional[CallPlan], List[str]]:
        """The call of step *step*, or None when it makes none (``(plan,
        errors)``): its turn joins the waiting ones; the call has the
        unjudged *docs* with text, and is made when it has a passage or is
        the *final* one (with a turn).  A call takes the waiting turns."""
        if self._state is None:
            return None, []
        turn = agent_turn.strip() if agent_turn and agent_turn.strip() else None
        if turn is not None:
            self._pending.append((step, turn))
            self._pending = self._pending[-MAX_PENDING_TURNS:]
        passages, errors = self.passages(self.unjudged(docs))
        if not passages and not (final and self._pending):
            return None, errors
        plan = CallPlan(turns=list(self._pending), passages=passages)
        self._pending = []
        return plan, errors

    def label_of(self, doc: Dict[str, Any]) -> Optional[str]:
        """The label of a document once it was shown, else None."""
        key = _doc_key(doc) if self._state is not None else None
        return self._state.doc_label(key) if key else None

    def apply(self, parsed: Any, plan: CallPlan, step: int) -> Dict[str, Any]:
        """Apply a parsed reply (``LLMBeliefUpdater.parse``) of the call of
        step *step* (*plan*): its passages are judged once it is applied
        (a reply that raises leaves them unjudged); returns the records
        of ``BeliefState.apply``."""
        claims, doc_obs, updates = parsed
        out = self._state.apply(claims, doc_obs, updates, step)
        self._judged.update(p.doc_id for p in plan.passages if p.doc_id)
        return out

    def _update(self, planned: Tuple[Optional[CallPlan], List[str]], step: int) -> Dict[str, Any]:
        plan, errors = planned
        out: Dict[str, Any] = {"called": False, "passages": [], "observations": None, "updates": [], "changed": [],
                               "issues": [], "beliefs": None, "reply": None, "errors": list(errors)}
        if self._state is None:
            return out
        if plan is not None:
            out["called"] = True
            out["passages"] = [{"label": p.label, "doc_id": p.doc_id} for p in plan.passages]
            parsed, raw, call_errors = self.updater.update(self._query, self._state, plan.turns, plan.passages, step)
            out["reply"] = raw
            out["errors"].extend(f"belief_update: {e}" for e in call_errors)
            if parsed is not None:
                out.update(self.apply(parsed, plan, step))
        out["beliefs"] = self.beliefs()
        return out


# ---------------------------------------------------------------------------
# Supervised retrieval gain (extra)
# ---------------------------------------------------------------------------

class RetrievalGainSignal:
    """Supervised retrieval gain of each step against the qrels.

    Binary (``qrels``, thresholded by the caller: relevant = grade > 0),
    over the step's unique doc ids:

    - ``new_item_precision``: newly seen relevant docs / the step's docs
      (0, 0.2, ..., 1 for 5 docs).
    - ``num_new_relevant``, ``num_repeated_relevant``, ``num_irrelevant``.
    - ``recall_so_far``: ``num_relevant_seen / num_relevant``, the relevant
      docs seen up to and including this step.

    All fields are null when the query has no relevant doc in the qrels or
    the step has no docs.  Call :meth:`reset` at the start of each query.
    """

    _NULL = {
        "new_item_precision": None,
        "num_new_relevant": None,
        "num_repeated_relevant": None,
        "num_irrelevant": None,
        "recall_so_far": None,
        "num_relevant_seen": None,
        "num_relevant": None,
    }

    def __init__(self, qrels: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._qrels = qrels or {}
        self._relevant_ids: Set[str] = set()
        self._relevant_seen: Set[str] = set()

    @property
    def num_relevant(self) -> Optional[int]:
        """Relevant docs of the current query; None without qrels for it."""
        return len(self._relevant_ids) or None

    def reset(self, query_id: Optional[str] = None) -> None:
        """Reset per-query state and load the relevant ids of ``query_id``."""
        self._relevant_seen.clear()
        judged = self._qrels.get(query_id, {}) if query_id else {}
        self._relevant_ids = {d for d, r in judged.items() if float(r) > 0}

    def score(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the step's gain statistics (see the class docstring)."""
        ids = {_doc_id(doc) for doc in docs}
        ids.discard("")
        result = dict(self._NULL)
        if not ids or not self._relevant_ids:
            return result
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
        policy_model: Agent name; picks the answer format and the parser.
    """

    def __init__(self, answer_fn: AnswerFn, policy_model: str) -> None:
        self._answer_fn = answer_fn
        self._policy_model = policy_model

    def score(self, original_query: str, trajectory: Any) -> Optional[Dict[str, Any]]:
        instruction = intermediate_answer_instruction(self._policy_model)
        raw = self._answer_fn(original_query, trajectory, instruction) or ""
        if raw.startswith(REASONING_FALLBACK_PREFIX):
            raw = raw[len(REASONING_FALLBACK_PREFIX):]
        outputs, format_matched = extract_answer_candidates(raw, expected_format=self._policy_model)
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
