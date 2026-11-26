"""The LLM belief update of the belief model.

One call implements b_t = f(query, b_{t-1}, o_t) (Agent-BRACE style), the
observation o_t being the agent's turns since the last call (its
reasoning; at the end, its reasoning and final answer) and the passages
of the call: those its last search returned, new or shown before (a
passage keeps its label; ``BeliefUpdateSignal`` decides when a call is
made).  The call reads every belief (id "B<k>") with its statement, the
passages assigned to its candidates and its proposals, and replies with
three lists:

- ``reasoning_observations``: per candidate the reasoning gives as an
  answer or drops, its belief, the span of the reasoning and the agent
  claim (``agent_claim``: "keep" or "drop"); its turn is the latest
  turn the span occurs in, and a span counts when it occurs in one;
- ``passage_observations``: per passage, belief and candidate, the
  stance (+ the candidate meets a part, - it fails one) and the spans of
  the passage that state it; a span counts when it occurs in the passage
  as the call showed it (``span_in_passage``);
- ``beliefs``: per belief with a new passage observation, a ``reason``,
  its ``most_probable`` candidate (null on an aspect) and its ``wep``.
  A belief computed from others (an aggregate) takes none of the three.
  The statement shown next time is built by code
  (``views.belief_text.belief_statement``).

The answer turn is read as any other turn (its reasoning and final answer).

The same prompt serves the prompted teacher and the trained student, so a
training sample is one call.  ``BeliefState.apply`` keeps the valid parts.
"""

from typing import Any, Dict, List, Optional, Tuple

from utils.documents import MAX_PASSAGE_TOKENS, cut_passage
from utils.json_parse import parse_json_object

from ..core.state import BeliefState
from ..core.types import (
    DOC, MULTI_CLUE, MULTI_MEMBER, QUERY_SHAPES, THINK, BeliefUpdate, Observation, belief_id, parse_claim,
)
from ..prompts import BELIEF_UPDATE_SYSTEMS, BELIEF_UPDATE_USER, NO_TEXT, PASSAGE_TEMPLATE, TURN_TEMPLATE
from ..text.matching import alnum, alnum_index, squash
from ..text.passages import Passage, Turn, doc_label, span_in_passage, split_span
from ..views.belief_text import format_belief
from .repair import complete_with_repair

# Output tokens of one call; a multi-member query (up to ~50 member beliefs, each
# with its candidates) gets more, since a reply cut at the limit cannot be
# parsed and the step's update is lost.
DEFAULT_MAX_TOKENS = 12288
MULTI_MEMBER_MAX_TOKENS = 16384

# The three lists of a reply, in order (every query shape).
REPLY_KEYS = ("reasoning_observations", "passage_observations", "beliefs")


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


class LLMBeliefUpdater:
    """The belief update call: one JSON-replying LLM call per step, its
    retries and its errors."""

    def __init__(self, llm_client: Any, model_name: Optional[str] = None, query_shape: str = MULTI_CLUE,
                 max_passage_tokens: Optional[int] = MAX_PASSAGE_TOKENS,
                 max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.0, parse_retries: int = 1) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        self._llm = llm_client
        self.name = f"llm:{model_name}"
        self.query_shape = query_shape
        self._system = BELIEF_UPDATE_SYSTEMS[query_shape]
        self._max_passage_tokens = max_passage_tokens
        self.max_tokens = max_tokens
        self._temperature = temperature
        self._parse_retries = parse_retries

    def shown_text(self, text: str) -> str:
        """A passage's text as the call shows it (cut as the agent's view
        of it); its spans are checked against this text only."""
        return cut_passage(text, self._max_passage_tokens)

    def prompt(self, query: str, state: BeliefState, turns: List[Turn],
               passages: List[Passage]) -> List[Dict[str, str]]:
        """The messages of one call: *state* is b_{t-1}, *turns* the agent's
        turns since the last call (each labelled "t<step>"), *passages* the
        call's passages."""
        shown = "\n\n".join(PASSAGE_TEMPLATE.format(label=p.label, text=self.shown_text(p.text).strip())
                             for p in passages)
        turn_text = "\n\n".join(TURN_TEMPLATE.format(label=f"t{step}", text=text.strip()) for step, text in turns)
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": BELIEF_UPDATE_USER.format(
                query=query, beliefs=format_belief(state), turns=turn_text or NO_TEXT, passages=shown or NO_TEXT)},
        ]

    def parse(self, raw: str, turns: List[Turn], passages: List[Passage], step: int,
              ) -> Optional[Tuple[List[Observation], List[Observation], List[BeliefUpdate]]]:
        """``(claims, docs, updates)`` of the reply *raw* of the call of
        step *step*: the span of a claim checked against the call's turns
        (its turn, and so its step, is the latest one the span occurs in),
        those of a document observation against its
        passage as the call showed it (``shown_text``: a span from past the
        cut is not verified; its step is *step*), the items of one
        belief, passage, candidate and stance merged; None unless the
        reply is a JSON object with the three lists (``REPLY_KEYS``)."""
        data = parse_json_object(raw or "")
        if data is None or not all(isinstance(data.get(key), list) for key in REPLY_KEYS):
            return None

        # The turn of a claim is the latest turn its span occurs in (the
        # last turn, its span unverified, when it occurs in none).
        turn_indexes = [(n, text, alnum_index(text)) for n, text in reversed(turns)]
        claims: List[Observation] = []
        for item in data["reasoning_observations"]:
            if not isinstance(item, dict):
                continue
            span = item.get("span")
            pieces = [p for c in (span if isinstance(span, list) else [span]) if c is not None
                      for p in split_span(str(c))]
            turn_step, text, index = next(
                ((n, t, ix) for n, t, ix in turn_indexes if any(span_in_passage(t, p, ix) for p in pieces)),
                turn_indexes[0] if turn_indexes else (step, "", None))
            o = Observation(belief=belief_id(item.get("id")), source=THINK, label=f"t{turn_step}",
                            step=turn_step, candidate=squash(item.get("candidate")) or None,
                            claim=parse_claim(item.get("agent_claim")),
                            claim_written=squash(item.get("agent_claim")) or None)
            _add_spans(o, span, text, index)
            claims.append(o)

        # Per passage label: its document, the text the call showed and that
        # text's index (built when the reply first cites the passage).
        by_label = {p.label: (p.doc_id, self.shown_text(p.text)) for p in passages}
        indexes: Dict[str, Tuple[str, List[int]]] = {}
        docs: Dict[Tuple[str, str, str, str], Observation] = {}
        for item in data["passage_observations"]:
            if not isinstance(item, dict):
                continue
            label = doc_label(item.get("passage"))
            cid = belief_id(item.get("id"))
            # A label that is no passage of the call (or none): kept with no
            # document (doc_id None), so ``BeliefState.apply`` notes it.
            doc, text = by_label.get(label, (None, ""))
            candidate = squash(item.get("candidate")) or None
            stance = squash(item.get("stance"))
            key = (cid, label, alnum(candidate or ""), stance)
            o = docs.get(key)
            if o is None:
                o = docs[key] = Observation(belief=cid, source=DOC, label=label, step=step, doc_id=doc,
                                            candidate=candidate, against=stance == "-",
                                            bad_stance=None if stance in ("+", "-") else stance)
            if doc is not None and label not in indexes:
                indexes[label] = alnum_index(text)
            _add_spans(o, item.get("spans"), text if doc is not None else "", indexes.get(label))

        updates = [_belief_update(item) for item in data["beliefs"] if isinstance(item, dict)]
        return claims, list(docs.values()), updates

    def update(self, query: str, state: BeliefState, turns: List[Turn],
               passages: List[Passage], step: int) -> Tuple[Any, Optional[str], List[str]]:
        """``(parsed, raw, errors)`` of the call of step *step*: *parsed* as
        ``parse`` gives it, None when no reply parsed (*raw* the last
        reply, None when the first call raised; ``complete_with_repair``)."""
        return complete_with_repair(
            "LLMBeliefUpdater", self._llm, self.prompt(query, state, turns, passages),
            lambda raw: self.parse(raw, turns, passages, step), self.repair,
            f"no JSON object with the lists {list(REPLY_KEYS)}", self._parse_retries,
            max_tokens=self.max_tokens, temperature=self._temperature)

    @property
    def repair(self) -> str:
        """The retry message after a reply that did not parse."""
        return update_repair(REPLY_KEYS)


def _belief_update(item: Dict[str, Any]) -> BeliefUpdate:
    """One item of the reply's ``beliefs`` list as written."""
    return BeliefUpdate(id=belief_id(item.get("id")), reason=squash(item.get("reason")),
                        most_probable=squash(item.get("most_probable")) or None, wep=squash(item.get("wep")))


def update_repair(keys: tuple) -> str:
    """The retry message after a reply without the lists *keys*."""
    return ("Your reply could not be read. Respond with ONLY a raw JSON object with the lists "
            + ", ".join(f'"{k}"' for k in keys) + ", in this order.")


def build_belief_updater(llm_client: Any, model: str = "", query_shape: str = MULTI_CLUE,
                         max_passage_tokens: Optional[int] = MAX_PASSAGE_TOKENS) -> LLMBeliefUpdater:
    """Build the belief updater on one LLM client (named *model*) for the
    queries of one shape.  A document is cut at *max_passage_tokens*, the
    run's shared setting, so the update reads the text the agent read."""
    if llm_client is None:
        raise ValueError("the belief updater needs an LLM client")
    return LLMBeliefUpdater(
        llm_client, model_name=model, query_shape=query_shape,
        max_passage_tokens=max_passage_tokens,
        max_tokens=MULTI_MEMBER_MAX_TOKENS if query_shape == MULTI_MEMBER else DEFAULT_MAX_TOKENS,
    )
