"""The LLM belief update of the belief model.

One call per step implements b_t = f(query, b_{t-1}, o_t) (Agent-BRACE
style), the observation o_t being the agent's turn of the step (its
reasoning; at the end, its reasoning and final answer) and the new
passages its search returned.  The call reads every criterion with its
current statement (text only) and replies with three lists:

- ``think_observations``: per criterion, the spans of the turn that give
  information about it; a span counts when it occurs in the turn;
- ``passage_observations``: per new passage and criterion, the spans of
  the passage that give information about it; a span counts when it
  occurs in the passage as the call showed it (``span_in_passage``);
- ``statements``: the new statement text of every criterion with a new
  observation (``statement_text`` reads its opening).

The answer turn is read as any other turn (its reasoning and final answer,
no passage).

The same prompt serves the prompted teacher and the trained student, so a
training sample is one call.  ``BeliefState.apply`` keeps the valid parts.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import parse_json_object

from .belief_state import format_belief
from .criteria import BeliefState
from .prompts import BELIEF_UPDATE_SYSTEMS, BELIEF_UPDATE_USER_TEMPLATE, NO_TEXT, PASSAGE_TEMPLATE
from .types import (
    DOC, MULTI_ASPECT, QUERY_SHAPES, SET, SINGLE_TARGET, THINK, Observation, WrittenStatement,
    alnum, alnum_index,
)

logger = logging.getLogger(__name__)

# Text of one short-answer passage shown to the belief update; a report
# document (multi-aspect query) is shown whole.
SHORT_ANSWER_PASSAGE_CHARS = 3000

# Output tokens of one call; a set query (up to ~50 member criteria, each
# with its statement) gets more, since a reply cut at the limit cannot be
# parsed and the step's update is lost.
DEFAULT_MAX_TOKENS = 12288
SET_MAX_TOKENS = 16384

# The three lists of a reply, in order.
REPLY_KEYS = ("think_observations", "passage_observations", "statements")


def cut_passage(text: str, max_chars: Optional[int]) -> str:
    """*text* cut to *max_chars* as ``doc_text`` cuts it (None keeps it whole)."""
    return text[:max_chars] + "..." if max_chars is not None and len(text) > max_chars else text


def span_in_passage(text: str, span: str, index: Optional[Tuple[str, List[int]]] = None) -> Optional[str]:
    """The passage's own text matching *span* (case, whitespace and
    punctuation ignored, so words the passage runs together still match),
    or None when *span* does not occur in *text*.  *index* is
    ``alnum_index(text)`` when the caller has it (several spans of one
    passage)."""
    target = alnum(span)
    if not target:
        return None
    keys, pos = index if index is not None else alnum_index(text)
    j = keys.find(target)
    if j < 0:
        return None
    return text[pos[j]:pos[j + len(target) - 1] + 1]


def split_span(span: str) -> List[str]:
    """The pieces of a cited span that joins sentences with an ellipsis."""
    return [p.strip() for p in re.split(r"\.\.\.|…", span) if alnum(p)]


def passage_title(text: str, max_chars: int = 120) -> str:
    """The first line of *text* with a letter or digit (a document's title
    or headline), cut at *max_chars* with ``...``."""
    line = next((ln.strip() for ln in text.splitlines() if alnum(ln)), "")
    return line[:max_chars] + "..." if len(line) > max_chars else line


def doc_label(value: Any) -> str:
    """A passage label as the reply writes it ("d4", "[d4]", 4) as "d4"."""
    text = str(value if value is not None else "").strip().strip("[]").strip().lower()
    return f"d{text}" if text.isdigit() else text


def _add_spans(o: Observation, cited: Any, text: str, index: Optional[Tuple[str, List[int]]]) -> None:
    """Add the *cited* spans to *o*, each checked against *text* (its
    ``alnum_index`` *index*): a verified span is the text's own words."""
    cited = cited if isinstance(cited, list) else [cited]
    for piece in (p for c in cited if c is not None for p in split_span(str(c))):
        matched = span_in_passage(text, piece, index) if text else None
        span = matched or piece
        if span not in o.spans:
            o.spans.append(span)
            o.span_verified.append(matched is not None)
    o.grounded = any(o.span_verified)


# One passage of a call: its label, document id and text.
Passage = Tuple[str, str, str]


class LLMBeliefUpdater:
    """The belief update call: one JSON-replying LLM call per step, its
    retries and its errors."""

    def __init__(self, llm_client: Any, model_name: Optional[str] = None, query_shape: str = SINGLE_TARGET,
                 max_passage_chars: Optional[int] = SHORT_ANSWER_PASSAGE_CHARS,
                 max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.0, parse_retries: int = 1) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        self._llm = llm_client
        self.name = f"llm:{model_name}"
        self.query_shape = query_shape
        self._system = BELIEF_UPDATE_SYSTEMS[query_shape]
        self._max_passage_chars = max_passage_chars
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._parse_retries = parse_retries

    def shown_text(self, text: str) -> str:
        """A passage's text as the call shows it (cut for a short-answer
        query); its spans are checked against this text only."""
        return cut_passage(text, self._max_passage_chars)

    def prompt(self, query: str, state: BeliefState, step: int, agent_turn: Optional[str],
               passages: List[Passage]) -> List[Dict[str, str]]:
        """The messages of one call: *state* is b_{t-1}, *agent_turn* the
        turn labelled "t<step>" (None: no turn), *passages* the new ones."""
        shown = "\n\n".join(PASSAGE_TEMPLATE.format(label=label, text=self.shown_text(text).strip())
                            for label, _, text in passages)
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": BELIEF_UPDATE_USER_TEMPLATE.format(
                query=query, criteria=format_belief(state), turn_label=f"t{step}",
                agent_turn=(agent_turn or "").strip() or NO_TEXT, passages=shown or NO_TEXT)},
        ]

    def parse(self, raw: str, step: int, agent_turn: Optional[str], passages: List[Passage],
              ) -> Optional[Tuple[List[Observation], List[Observation], List[WrittenStatement]]]:
        """``(think, docs, statements)`` of the reply *raw*: the spans of a
        think observation checked against *agent_turn*, those of a
        document observation against its passage as the call showed it
        (``shown_text``: a span from past the cut is not verified), the
        items of one criterion and source merged; None unless the reply is
        a JSON object with the three lists."""
        data = parse_json_object(raw or "")
        if data is None or not all(isinstance(data.get(key), list) for key in REPLY_KEYS):
            return None

        turn_index = alnum_index(agent_turn) if agent_turn else None
        think: Dict[str, Observation] = {}
        for item in data["think_observations"]:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id", "")).strip()
            o = think.setdefault(cid, Observation(criterion=cid, source=THINK, label=f"t{step}", step=step))
            _add_spans(o, item.get("spans"), agent_turn or "", turn_index)

        # Per passage label: its document, the text the call showed and that
        # text's index (built when the reply first cites the passage).
        by_label = {label: (doc, self.shown_text(text)) for label, doc, text in passages}
        indexes: Dict[str, Tuple[str, List[int]]] = {}
        docs: Dict[Tuple[str, str], Observation] = {}
        for item in data["passage_observations"]:
            if not isinstance(item, dict):
                continue
            label = doc_label(item.get("passage"))
            cid = str(item.get("id", "")).strip()
            if not label:
                continue
            # A label that is no new passage of the call: kept with no
            # document (doc_id None), so ``BeliefState.apply`` notes it.
            doc, text = by_label.get(label, (None, ""))
            o = docs.setdefault((cid, label), Observation(criterion=cid, source=DOC, label=label, step=step,
                                                          doc_id=doc))
            if doc is not None and label not in indexes:
                indexes[label] = alnum_index(text)
            _add_spans(o, item.get("spans"), text if doc is not None else "", indexes.get(label))

        statements = [WrittenStatement(id=str(item.get("id", "")).strip(), text=" ".join(str(item.get("statement") or "").split()))
                      for item in data["statements"] if isinstance(item, dict)]
        return list(think.values()), list(docs.values()), statements

    def update(self, query: str, state: BeliefState, step: int, agent_turn: Optional[str],
               passages: List[Passage]) -> Tuple[Any, Optional[str], List[str]]:
        """``(parsed, raw, errors)``: *parsed* as ``parse`` gives it, None
        when the call raised (*raw* None) or no reply parsed (*raw* the
        last reply)."""
        messages = self.prompt(query, state, step, agent_turn, passages)
        errors: List[str] = []
        raw = None
        for attempt in range(1 + self._parse_retries):
            try:
                raw = self._llm.complete(messages, max_tokens=self.max_tokens, temperature=self._temperature) or ""
            except Exception as e:
                logger.warning("LLMBeliefUpdater: call failed: %s", e)
                return None, None, errors + [f"{e}"]
            parsed = self.parse(raw, step, agent_turn, passages)
            if parsed is not None:
                return parsed, raw, errors
            errors.append(f"parse (attempt {attempt + 1}): no JSON object with the lists {list(REPLY_KEYS)}")
        return None, raw, errors


def build_belief_updater(llm_client: Any, model: str = "", query_shape: str = SINGLE_TARGET) -> LLMBeliefUpdater:
    """Build the belief updater on one LLM client (named *model*) for the
    queries of one shape.  A report document is shown whole, as in the
    criteria reachability analysis; a short-answer document (often a long
    web page) is cut."""
    if llm_client is None:
        raise ValueError("the belief updater needs an LLM client")
    return LLMBeliefUpdater(
        llm_client, model_name=model, query_shape=query_shape,
        max_passage_chars=None if query_shape == MULTI_ASPECT else SHORT_ANSWER_PASSAGE_CHARS,
        max_tokens=SET_MAX_TOKENS if query_shape == SET else DEFAULT_MAX_TOKENS,
    )
