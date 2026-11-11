"""The LLM belief update of the belief model.

One call implements b_t = f(query, b_{t-1}, o_t) (Agent-BRACE style), the
observation o_t being the agent's turns since the last call (its
reasoning; at the end, its reasoning and final answer) and the passages
of the call: those its search returned for the first time
(``BeliefUpdateSignal`` decides when a call is made).  The call reads every criterion with its
current candidates (name, status, support) and proposals, and replies
with three lists:

- ``reasoning_observations``: per candidate the reasoning names, assesses
  or rejects, its criterion, turn label, the span of the turn and the
  agent claim (``agent_claim``: a status, or "rejected"); a span counts
  when it occurs in that turn;
- ``passage_observations``: per passage and criterion, the spans of the
  passage that state a part of the criterion; a span counts when it
  occurs in the passage as the call showed it (``span_in_passage``);
- ``candidates``: per candidate whose support or status changes, the
  labels of the passages that support it or speak against it, its reason
  ("Was <status>: ...", "Was proposed: ..." or "New: ...", ``core.reason_text``)
  and its new status (``belief_status``), or "removed";
- ``coverage`` (a report's aspects only, ``MULTI_ASPECT``): per aspect
  whose coverage changes, its reason ("Was <coverage>: ...") and its new
  coverage status (``coverage_status``): how completely the passages cover
  what the aspect asks for.

The answer turn is read as any other turn (its reasoning and final answer).

The same prompt serves the prompted teacher and the trained student, so a
training sample is one call.  ``BeliefState.apply`` keeps the valid parts.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import MAX_PASSAGE_TOKENS, cut_passage, parse_json_object

from ..core.state import BeliefState
from ..core.types import (
    DOC, MULTI_ASPECT, MULTI_CLUE, MULTI_MEMBER, QUERY_SHAPES, THINK, CandidateUpdate, CoverageUpdate, Observation,
    criterion_id, parse_claim,
)
from ..prompts import BELIEF_UPDATE_SYSTEMS, BELIEF_UPDATE_USER_TEMPLATE, NO_TEXT, PASSAGE_TEMPLATE, TURN_TEMPLATE
from ..text.matching import alnum_index, squash
from ..text.passages import Passage, Turn, doc_label, reply_labels, span_in_passage, split_span, turn_label
from ..views.tag import format_belief
from .repair import complete_with_repair

logger = logging.getLogger(__name__)

# Output tokens of one call; a multi-member query (up to ~50 member criteria, each
# with its candidates) gets more, since a reply cut at the limit cannot be
# parsed and the step's update is lost.
DEFAULT_MAX_TOKENS = 12288
MULTI_MEMBER_MAX_TOKENS = 16384

# The three lists of a reply, in order; a report's reply ends with a fourth,
# the aspects' coverage (``reply_keys``).
REPLY_KEYS = ("reasoning_observations", "passage_observations", "candidates")
COVERAGE_KEY = "coverage"


def reply_keys(query_shape: str) -> tuple:
    """The lists of a reply for a *query_shape* query, in order."""
    return REPLY_KEYS + ((COVERAGE_KEY,) if query_shape == MULTI_ASPECT else ())


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
        self.reply_keys = reply_keys(query_shape)
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
            {"role": "user", "content": BELIEF_UPDATE_USER_TEMPLATE.format(
                query=query, criteria=format_belief(state), turns=turn_text or NO_TEXT, passages=shown or NO_TEXT)},
        ]

    def parse(self, raw: str, turns: List[Turn], passages: List[Passage], step: int,
              ) -> Optional[Tuple[List[Observation], List[Observation], List[CandidateUpdate],
                                  List[CoverageUpdate]]]:
        """``(claims, docs, updates, coverage)`` of the reply *raw* of the call of
        step *step*: the span of a claim checked against its turn (the only
        turn when the call has one and the label is unknown; the claim's
        step is its turn's), those of a document observation against its
        passage as the call showed it (``shown_text``: a span from past the
        cut is not verified; its step is *step*), the items of one
        criterion and passage merged; *coverage* the aspects' coverage
        updates (empty unless a report); None unless the reply is a JSON
        object with the shape's lists (``reply_keys``)."""
        data = parse_json_object(raw or "")
        if data is None or not all(isinstance(data.get(key), list) for key in self.reply_keys):
            return None

        by_turn = {f"t{n}": (n, text) for n, text in turns}
        turn_indexes: Dict[str, Tuple[str, List[int]]] = {}
        claims: List[Observation] = []
        for item in data["reasoning_observations"]:
            if not isinstance(item, dict):
                continue
            label = turn_label(item.get("turn"))
            if label not in by_turn and len(turns) == 1:
                label = f"t{turns[0][0]}"
            turn_step, text = by_turn.get(label, (turns[-1][0] if turns else step, ""))
            if text and label not in turn_indexes:
                turn_indexes[label] = alnum_index(text)
            o = Observation(criterion=criterion_id(item.get("id")), source=THINK, label=label or f"t{turn_step}",
                            step=turn_step, candidate=squash(item.get("candidate")) or None,
                            claim=parse_claim(item.get("agent_claim")))
            _add_spans(o, item.get("span"), text, turn_indexes.get(label))
            claims.append(o)

        # Per passage label: its document, the text the call showed and that
        # text's index (built when the reply first cites the passage).
        by_label = {p.label: (p.doc_id, self.shown_text(p.text)) for p in passages}
        indexes: Dict[str, Tuple[str, List[int]]] = {}
        docs: Dict[Tuple[str, str], Observation] = {}
        for item in data["passage_observations"]:
            if not isinstance(item, dict):
                continue
            label = doc_label(item.get("passage"))
            cid = criterion_id(item.get("id"))
            if not label:
                continue
            # A label that is no passage of the call: kept with no document
            # (doc_id None), so ``BeliefState.apply`` notes it.
            doc, text = by_label.get(label, (None, ""))
            o = docs.setdefault((cid, label), Observation(criterion=cid, source=DOC, label=label, step=step,
                                                          doc_id=doc))
            if doc is not None and label not in indexes:
                indexes[label] = alnum_index(text)
            _add_spans(o, item.get("spans"), text if doc is not None else "", indexes.get(label))

        updates = [CandidateUpdate(id=criterion_id(item.get("id")),
                                   candidate=squash(item.get("candidate")),
                                   reason=squash(item.get("reason")),
                                   status=str(item.get("belief_status") or "").strip(),
                                   support=reply_labels(item.get("support")), against=reply_labels(item.get("against")))
                   for item in data["candidates"] if isinstance(item, dict)]
        coverage = [CoverageUpdate(id=criterion_id(item.get("id")), reason=squash(item.get("reason")),
                                   status=str(item.get("coverage_status") or "").strip())
                    for item in data.get(COVERAGE_KEY) or [] if isinstance(item, dict)]
        return claims, list(docs.values()), updates, coverage

    def update(self, query: str, state: BeliefState, turns: List[Turn],
               passages: List[Passage], step: int) -> Tuple[Any, Optional[str], List[str]]:
        """``(parsed, raw, errors)`` of the call of step *step*: *parsed* as
        ``parse`` gives it, None when no reply parsed (*raw* the last
        reply, None when the first call raised; ``complete_with_repair``)."""
        return complete_with_repair(
            "LLMBeliefUpdater", self._llm, self.prompt(query, state, turns, passages),
            lambda raw: self.parse(raw, turns, passages, step), self.repair,
            f"no JSON object with the lists {list(self.reply_keys)}", self._parse_retries,
            max_tokens=self.max_tokens, temperature=self._temperature)

    @property
    def repair(self) -> str:
        """The retry message after a reply that did not parse."""
        return update_repair(self.reply_keys)


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
