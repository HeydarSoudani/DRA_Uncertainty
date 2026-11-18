"""The final scorer of the belief model: s_T, how likely the agent's final
answer is correct, judged by the LLM from the belief.

One call after the last step, in place of a belief update on the answer
turn: it reads the query, the belief b_{T-1} (every belief with its
statement, the passages on its candidates and the proposals:
``format_belief``, as the update reads it) and the agent's
final turn (``scorer_answer_text``: "[t<N>]", the end of its reasoning and
its final answer under "Final answer:", with no citations and no
confidence line), and replies with one JSON object:

- ``reason``: one or two sentences naming the beliefs that set the word;
- ``wep``: the word of estimative probability that the final answer is
  correct (``core.types.BELIEF_STATUSES``, "unknown" included),
  read as a probability as a belief's WEP is
  (``core.types.status_probability``).

The scorer weighs each belief by how much the answer's correctness depends
on it; it is not told which candidate the answer is.  It anchors s_T on
the agent's answer, which ``core.certainty.certainty`` cannot (it scores
the belief, whatever the answer).  Short-answer queries only
(``prompts.BELIEF_SCORE_SYSTEMS``): a report has no single answer to judge.

The same prompt serves the prompted teacher and the trained student (the
belief model's second task).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import parse_json_object

from ..core.state import BeliefState
from ..core.types import BELIEF_STATUSES, MULTI_CLUE, parse_status, status_probability
from ..prompts import BELIEF_SCORE_SYSTEMS, BELIEF_SCORE_USER, TURN_TEMPLATE
from ..text.agent_turn import FINAL_ANSWER_HEADER
from ..text.matching import squash
from ..views.belief_text import format_belief
from .repair import complete_with_repair

# Output tokens of one call (a reason and a word, after any thinking).
MAX_TOKENS = 4096

# Characters of the answer turn's reasoning the scorer reads (its last
# ones, the conclusion).
ANSWER_REASONING_CHARS = 2000

# A citation in the agent's response: a bracketed document id (one with a
# digit), or a list of them ("[5412]", "【68484】", "[15641‑0005]", "[12, 34]").
_ID = r"[\w\-\u2010-\u2011.:/]+"
_CITATION_RE = re.compile(r"[ \t\u202f\u00a0]*[\[【](?=[^\]】]*\d)" + _ID + r"(?:\s*[,;]\s*" + _ID + r")*[\]】]")
# The agent's confidence line ("Confidence: 75%", "**Confidence:** 75%").
_CONFIDENCE_RE = re.compile(r"^[ \t>*#_-]*confidence[*_ \t]*:.*(?:\n|$)", re.IGNORECASE | re.MULTILINE)


def clean_response(response: str) -> str:
    """The agent's final response as the scorer reads it: its citations
    and its confidence line removed (the scorer judges from the belief,
    and s_T is compared with the agent's verbal confidence)."""
    text = _CONFIDENCE_RE.sub("", _CITATION_RE.sub("", response))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def scorer_answer_text(answer_turn: str, step: int, max_reasoning_chars: int = ANSWER_REASONING_CHARS) -> str:
    """The answer turn as the scorer reads it: "[t<step>]", the last
    *max_reasoning_chars* of its reasoning (no citations) and the agent's
    final response (``clean_response``) under ``FINAL_ANSWER_HEADER``."""
    text = answer_turn.strip()
    cut = text.rfind(FINAL_ANSWER_HEADER)
    reasoning, response = (text[:cut].strip(), text[cut + len(FINAL_ANSWER_HEADER):]) if cut >= 0 else ("", text)
    response = clean_response(response)
    if len(reasoning) > max_reasoning_chars:
        reasoning = "..." + reasoning[-max_reasoning_chars:].lstrip(".")
    reasoning = _CITATION_RE.sub("", reasoning).strip()
    body = "\n\n".join(p for p in (reasoning, f"{FINAL_ANSWER_HEADER}\n{response}") if p)
    return TURN_TEMPLATE.format(label=f"t{step}", text=body)


def parse_final_status(value: Any) -> Optional[str]:
    """The WEP a reply writes (as ``core.types.parse_status``, a trailing
    period and quotes ignored), None when it is no word of the scale."""
    return parse_status(str(value or "").strip(" .\"'"))


@dataclass
class FinalScore:
    """One parsed reply: the WEP of the final answer (``status``) and its
    probability, and the reason."""
    status: str
    reason: str = ""

    @property
    def probability(self) -> float:
        return status_probability(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason}


def parse_final_score(raw: str) -> Optional[FinalScore]:
    """The reply *raw* as a ``FinalScore``, None unless it is a JSON object
    with a valid ``wep``."""
    data = parse_json_object(raw or "")
    if data is None:
        return None
    status = parse_final_status(data.get("wep"))
    if status is None:
        return None
    return FinalScore(status=status, reason=squash(data.get("reason")))


class LLMFinalScorer:
    """The final scorer call: one JSON-replying LLM call, its retries and
    its errors."""

    def __init__(self, llm_client: Any, model_name: Optional[str] = None, query_shape: str = MULTI_CLUE,
                 max_tokens: int = MAX_TOKENS, temperature: float = 0.0, parse_retries: int = 1) -> None:
        if query_shape not in BELIEF_SCORE_SYSTEMS:
            raise ValueError(f"no final scorer for query shape {query_shape!r}; "
                             f"expected one of {list(BELIEF_SCORE_SYSTEMS)}")
        self._llm = llm_client
        self.name = f"llm:{model_name}"
        self.query_shape = query_shape
        self._system = BELIEF_SCORE_SYSTEMS[query_shape]
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._parse_retries = parse_retries

    def prompt(self, query: str, state: BeliefState, answer_turn: str, step: int) -> List[Dict[str, str]]:
        """The messages of the call: *state* is b_{T-1}, *answer_turn* the
        agent's answer turn, labelled "t<step>"."""
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": BELIEF_SCORE_USER.format(
                query=query, criteria=format_belief(state),
                answer=scorer_answer_text(answer_turn, step))},
        ]

    def score(self, query: str, state: BeliefState, answer_turn: str, step: int
              ) -> Tuple[Optional[FinalScore], Optional[str], List[str]]:
        """``(score, raw, errors)``: *score* None when no reply parsed
        (*raw* the last reply, None when the first call raised).  A retry
        tells the model what was wrong with its reply
        (``repair.complete_with_repair``)."""
        return complete_with_repair(
            "LLMFinalScorer", self._llm, self.prompt(query, state, answer_turn, step), parse_final_score,
            FINAL_SCORE_REPAIR, f"no JSON object with a wep of {list(BELIEF_STATUSES)}", self._parse_retries,
            max_tokens=self.max_tokens, temperature=self._temperature)


# The retry message after a reply that did not parse.
FINAL_SCORE_REPAIR = ("Your reply could not be read. Respond with ONLY a raw JSON object with the keys "
                      "\"reason\" and \"wep\", the wep one of: "
                      + ", ".join(f'"{t}"' for t in BELIEF_STATUSES) + ".")


def build_final_scorer(llm_client: Any, model: str = "", query_shape: str = MULTI_CLUE
                       ) -> Optional[LLMFinalScorer]:
    """The final scorer on one LLM client (named *model*) for the queries
    of one shape; None for a shape without one (a report)."""
    if llm_client is None:
        raise ValueError("the final scorer needs an LLM client")
    if query_shape not in BELIEF_SCORE_SYSTEMS:
        return None
    return LLMFinalScorer(llm_client, model_name=model, query_shape=query_shape)
