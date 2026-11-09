"""Per-step parts of the belief model:

- ``BeliefUpdateSignal`` b_t: the belief update of the step (one call:
  observations of the agent's turn and of the documents, then the
  statements).

Extra, for evaluation only (never shown to the agent):

- ``RetrievalGainSignal``: supervised retrieval gain against the qrels
  (new-item precision, recall so far).
- ``IntermediateAnswerSignal``: the agent's own answer(s) after each turn,
  not evaluated.
"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from deep_research_agents.prompts.answer_prompts import extract_answer_candidates, extract_bare_answer
from reasoner_component import REASONING_FALLBACK_PREFIX
from utils.text_utils import doc_id as _doc_id, doc_text
from .certainty import certainty
from .criteria import BeliefState
from .updater import LLMBeliefUpdater, Passage, passage_title
from .prompts import intermediate_answer_instruction
from .types import Criterion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# b_t: belief update
# ---------------------------------------------------------------------------

def _doc_key(doc: Dict[str, Any], text: str) -> str:
    """A document's key for its label: its id, else a hash of its text."""
    return _doc_id(doc) or "text:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class BeliefUpdateSignal:
    """Belief update of each step: one call (``updater``) on the agent's
    turn and the step's documents not judged in an earlier step; each
    document gets its label when shown (``BeliefState.label_doc``), and the
    reply is applied by ``BeliefState.apply``.  The answer turn is one more
    call (``finish``), with no document.

    A call with nothing to read (no turn, no new document with text) is not
    made.  When the call fails the belief stays as it was, the step's
    documents stay unjudged (a later step that retrieves them shows them
    again, with the same labels) and the turn's observations are lost.
    ``score`` and ``finish`` return the step's outcome::

        {passages, observations, changed, issues, statements, reply, errors}

    (``passages`` ``[{label, doc_id}]`` as shown; ``observations``,
    ``changed`` and ``issues`` as ``BeliefState.apply`` gives them, null /
    empty when the call was not made or failed; ``statements`` every
    criterion's statement after the call; ``reply`` the raw reply, null
    when not called or the call raised).

    The call is made in two halves, ``passages`` (label the documents) and
    ``apply`` (the parsed reply), so belief model training can make the
    LLM calls itself (``belief_model_training.episodes.BeliefTracker``).
    ``certainty`` is the certainty score s_t of the belief
    (``certainty.certainty``).
    """

    def __init__(self, updater: LLMBeliefUpdater) -> None:
        self.updater = updater
        self._query = ""
        self._state: Optional[BeliefState] = None
        self._judged: Set[str] = set()

    @property
    def active(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> Optional[BeliefState]:
        return self._state

    def statements(self) -> Optional[List[Dict[str, Any]]]:
        return self._state.statement_lists() if self._state is not None else None

    def certainty(self) -> Optional[float]:
        """s_t of the current belief; None without criteria."""
        if self._state is None:
            return None
        return certainty(self._state.statement_lists(), [c.kind for c in self._state.criteria])

    def reset(self, query: str, criteria: List[Criterion]) -> None:
        self._query = query
        self._state = BeliefState(criteria) if criteria else None
        self._judged.clear()

    def unjudged(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The documents of *docs* not judged yet (documents without an id
        are shown every step)."""
        return [d for d in docs if not _doc_id(d) or _doc_id(d) not in self._judged]

    def score(self, docs: List[Dict[str, Any]], step: int, agent_turn: Optional[str] = None) -> Dict[str, Any]:
        """The update of one step from its unique *docs* and the agent's
        turn (*agent_turn*; None: no turn to read)."""
        return self._update(self.unjudged(docs), step, agent_turn)

    def finish(self, step: int, agent_turn: str) -> Dict[str, Any]:
        """The final update: the agent's answer turn (labelled ``t<step>``),
        no document."""
        return self._update([], step, agent_turn)

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
            passages.append((self._state.label_doc(_doc_key(d, text), passage_title(text)), _doc_id(d), text))
        return passages, errors

    def label_of(self, doc: Dict[str, Any]) -> Optional[str]:
        """The label of a document once it was shown, else None."""
        text = doc_text(doc, max_length=None)
        return self._state.doc_label(_doc_key(doc, text)) if self._state is not None and text else None

    def apply(self, parsed: Any, passages: List[Passage]) -> Dict[str, Any]:
        """Apply a parsed reply (``LLMBeliefUpdater.parse``) of a call on
        *passages*: they are judged now; returns the records of
        ``BeliefState.apply``."""
        self._judged.update(doc for _, doc, _ in passages if doc)
        think, doc_obs, statements = parsed
        return self._state.apply(think, doc_obs, statements)

    def _update(self, docs: List[Dict[str, Any]], step: int, agent_turn: Optional[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"passages": [], "observations": None, "changed": [], "issues": [],
                               "statements": self.statements(), "reply": None, "errors": []}
        if self._state is None:
            return out
        passages, out["errors"] = self.passages(docs)
        out["passages"] = [{"label": label, "doc_id": doc} for label, doc, _ in passages]
        turn = agent_turn if agent_turn and agent_turn.strip() else None
        if not passages and turn is None:
            return out

        parsed, raw, errors = self.updater.update(self._query, self._state, step, turn, passages)
        out["reply"] = raw
        out["errors"].extend(f"belief_update: {e}" for e in errors)
        if parsed is None:
            return out
        out.update(self.apply(parsed, passages))
        out["statements"] = self.statements()
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
