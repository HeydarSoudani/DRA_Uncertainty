"""The final scorer of the belief model: s_T, how likely the agent's final
answer is correct, judged by the LLM from the belief.

One call after the last step, in place of a belief update on the answer
turn: it reads the query, the belief b_{T-1} (every criterion with its
candidates, their statuses and support, the proposals and the names a
passage ruled out: ``format_belief(state, removed=True)``) and the agent's
answer turn (``text.agent_turn.answer_turn``), and replies with one JSON object:

- ``candidates``: per criterion, the answer's candidate (the one the
  answer relies on, or null);
- ``reason``: one sentence;
- ``final_status``: the status of the final answer, on the belief's scale
  (``FINAL_STATUSES``: the five candidate statuses, no "unknown", no
  "removed"), read as a probability as a candidate's status is
  (``core.types.status_probability``).

The prompt shares the belief update's definitions and status definitions;
the final answer is judged as a candidate of the query.

The scorer weighs the criteria by how much each identifies the answer and
anchors s_T on the agent's answer, which a fixed function of the statuses
cannot (``core.certainty.certainty`` scores the belief, whatever the answer).
Short-answer queries only (``prompts.FINAL_SCORE_SYSTEMS``): a report has
no single answer to judge.

The same prompt serves the prompted teacher and the trained student (the
belief model's second task).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import parse_json_object

from ..core.state import BeliefState
from ..core.types import BELIEF_STATUSES, UNKNOWN, criterion_id, parse_status, status_probability
from ..prompts import FINAL_SCORE_SYSTEMS, FINAL_SCORE_USER_TEMPLATE
from ..text.agent_turn import FINAL_ANSWER_HEADER
from ..text.matching import alnum, squash
from ..views.tag import format_belief
from .repair import complete_with_repair

# The status of the final answer, lowest first: the candidate statuses
# (unlikely 1/6, possible 1/2, probable 2/3, almost certain 5/6, certain 1).
FINAL_STATUSES = tuple(s for s in BELIEF_STATUSES if s != UNKNOWN)

# Output tokens of one call (a set query has up to ~50 criteria).
MAX_TOKENS = 4096

# Characters of the answer turn's reasoning the scorer reads (its last
# ones, the conclusion); the final answer is always shown whole.
ANSWER_REASONING_CHARS = 2000


def scorer_answer_text(answer_turn: str, max_reasoning_chars: int = ANSWER_REASONING_CHARS) -> str:
    """The answer turn as the scorer reads it: the last
    *max_reasoning_chars* of its reasoning, then the final answer."""
    text = answer_turn.strip()
    cut = text.rfind(FINAL_ANSWER_HEADER)
    reasoning, answer = (text[:cut].strip(), text[cut:]) if cut >= 0 else ("", text)
    if len(reasoning) > max_reasoning_chars:
        reasoning = "..." + reasoning[-max_reasoning_chars:].lstrip(".")
    return "\n\n".join(p for p in (reasoning, answer) if p)


def parse_final_status(value: Any) -> Optional[str]:
    """The final status a reply writes (as ``core.types.parse_status``, a
    trailing period and quotes ignored), None when it is none of
    ``FINAL_STATUSES``."""
    status = parse_status(str(value or "").strip(" .\"'"))
    return status if status in FINAL_STATUSES else None


@dataclass
class FinalScore:
    """One parsed reply: the status of the final answer and its
    probability, the answer's candidates (``{id, candidate}``) and the
    reason."""
    status: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @property
    def probability(self) -> float:
        return status_probability(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "candidates": self.candidates, "reason": self.reason}


def parse_final_score(raw: str) -> Optional[FinalScore]:
    """The reply *raw* as a ``FinalScore``, None unless it is a JSON object
    with a valid status."""
    data = parse_json_object(raw or "")
    if data is None:
        return None
    status = parse_final_status(data.get("final_status"))
    if status is None:
        return None
    candidates = []
    for item in data.get("candidates") if isinstance(data.get("candidates"), list) else []:
        if not isinstance(item, dict):
            continue
        name = squash(item.get("candidate"))
        candidates.append({"id": criterion_id(item.get("id")), "candidate": name or None})
    return FinalScore(status=status, candidates=candidates, reason=squash(data.get("reason")))


def check_issues(score: FinalScore, state: BeliefState) -> List[str]:
    """What the answer's candidates of *score* get wrong against the belief
    *state*: a criterion missing or unknown, a candidate the criterion does
    not list (as a candidate, proposal or removed name).  Information only:
    the status is kept."""
    issues: List[str] = []
    ids = [c.id for c in state.criteria]
    seen = {c["id"] for c in score.candidates}
    missing = [i for i in ids if i not in seen]
    if missing:
        issues.append(f"no answer's candidate for {missing}")
    for c in score.candidates:
        if c["id"] not in ids:
            issues.append(f"answer's candidate of unknown criterion {c['id']!r}")
            continue
        k = ids.index(c["id"])
        names = ([x.name for x in state.candidates[k]] + [x.name for x in state.proposals[k]]
                 + state.ruled_out(k))
        if c["candidate"] and alnum(c["candidate"]) not in {alnum(n) for n in names}:
            issues.append(f"{c['id']}: candidate {c['candidate']!r} is not listed")
    return issues


class LLMFinalScorer:
    """The final scorer call: one JSON-replying LLM call, its retries and
    its errors."""

    def __init__(self, llm_client: Any, model_name: Optional[str] = None, query_shape: str = "single_target",
                 max_tokens: int = MAX_TOKENS, temperature: float = 0.0, parse_retries: int = 1) -> None:
        if query_shape not in FINAL_SCORE_SYSTEMS:
            raise ValueError(f"no final scorer for query shape {query_shape!r}; "
                             f"expected one of {list(FINAL_SCORE_SYSTEMS)}")
        self._llm = llm_client
        self.name = f"llm:{model_name}"
        self.query_shape = query_shape
        self._system = FINAL_SCORE_SYSTEMS[query_shape]
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._parse_retries = parse_retries

    def prompt(self, query: str, state: BeliefState, answer_turn: str) -> List[Dict[str, str]]:
        """The messages of the call: *state* is b_{T-1}, *answer_turn* the
        agent's answer turn."""
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": FINAL_SCORE_USER_TEMPLATE.format(
                query=query, criteria=format_belief(state, removed=True), answer=scorer_answer_text(answer_turn))},
        ]

    def score(self, query: str, state: BeliefState, answer_turn: str
              ) -> Tuple[Optional[FinalScore], Optional[str], List[str]]:
        """``(score, raw, errors)``: *score* None when no reply parsed
        (*raw* the last reply, None when the first call raised).  A retry
        tells the model what was wrong with its reply
        (``repair.complete_with_repair``)."""
        return complete_with_repair(
            "LLMFinalScorer", self._llm, self.prompt(query, state, answer_turn), parse_final_score,
            FINAL_SCORE_REPAIR, f"no JSON object with a final_status of {list(FINAL_STATUSES)}", self._parse_retries,
            max_tokens=self.max_tokens, temperature=self._temperature)


# The retry message after a reply that did not parse.
FINAL_SCORE_REPAIR = ("Your reply could not be read. Respond with ONLY a raw JSON object with the keys "
                      "\"candidates\", \"reason\" and \"final_status\", the final status one of: "
                      + ", ".join(f'"{t}"' for t in FINAL_STATUSES) + ".")


def build_final_scorer(llm_client: Any, model: str = "", query_shape: str = "single_target"
                       ) -> Optional[LLMFinalScorer]:
    """The final scorer on one LLM client (named *model*) for the queries
    of one shape; None for a shape without one (a report)."""
    if llm_client is None:
        raise ValueError("the final scorer needs an LLM client")
    if query_shape not in FINAL_SCORE_SYSTEMS:
        return None
    return LLMFinalScorer(llm_client, model_name=model, query_shape=query_shape)
