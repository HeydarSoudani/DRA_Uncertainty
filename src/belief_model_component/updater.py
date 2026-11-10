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
  ("Was <status>: ...", "Was proposed: ..." or "New: ...", ``reason_text``)
  and its new status (``belief_status``), or "removed".

The answer turn is read as any other turn (its reasoning and final answer).

The same prompt serves the prompted teacher and the trained student, so a
training sample is one call.  ``BeliefState.apply`` keeps the valid parts.
"""

import logging
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from utils.text_utils import MAX_PASSAGE_CHARS, cut_passage, parse_json_object

from .belief_state import format_belief
from .criteria import BeliefState
from .prompts import (
    BELIEF_UPDATE_SYSTEMS, BELIEF_UPDATE_USER_TEMPLATE, NO_TEXT, PASSAGE_TEMPLATE,
    TURN_TEMPLATE,
)
from .types import (
    DOC, QUERY_SHAPES, SET, SINGLE_TARGET, THINK, CandidateUpdate, Observation,
    alnum, alnum_index, criterion_id, parse_claim,
)

logger = logging.getLogger(__name__)

# Output tokens of one call; a set query (up to ~50 member criteria, each
# with its candidates) gets more, since a reply cut at the limit cannot be
# parsed and the step's update is lost.
DEFAULT_MAX_TOKENS = 12288
SET_MAX_TOKENS = 16384

# The three lists of a reply, in order.
REPLY_KEYS = ("reasoning_observations", "passage_observations", "candidates")


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


# A document's front matter (``---\ntitle: ...\n---``, BrowseComp-Plus).
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_FRONT_TITLE = re.compile(r"^title:\s*(.+)$", re.M)


def passage_title(text: str, title: str = "", max_chars: int = 120) -> str:
    """A document's *title* (its metadata), else the title of its front
    matter, else the first line of *text* with a letter or digit (a wiki
    page's title or a headline); cut at *max_chars* with ``...``."""
    line = " ".join(str(title or "").split())
    if not line:
        front = _FRONT_MATTER.match(text)
        found = _FRONT_TITLE.search(front.group(1)) if front else None
        body = text[front.end():] if front else text
        line = (found.group(1).strip().strip("'\"") if found else
                next((ln.strip() for ln in body.splitlines() if alnum(ln)), ""))
    return cut_passage(line, max_chars)


def _label(value: Any, prefix: str) -> str:
    """A label as the reply writes it ("d4", "[d4]", 4) with its *prefix*
    ("d4")."""
    text = str(value if value is not None else "").strip().strip("[]").strip().lower()
    return f"{prefix}{text}" if text.isdigit() else text


def doc_label(value: Any) -> str:
    """A passage label as the reply writes it ("d4", "[d4]", 4) as "d4"."""
    return _label(value, "d")


def turn_label(value: Any) -> str:
    """A turn label as the reply writes it ("t4", "[t4]", 4) as "t4"."""
    return _label(value, "t")


_LABEL_SEP_RE = re.compile(r"[,;\s]+")


def _labels(value: Any) -> List[str]:
    """The passage labels of a reply's ``support`` / ``against`` list (a
    single label read as a list of one, a string of several, "d1, d2",
    split)."""
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    pieces = [p for v in items for p in (_LABEL_SEP_RE.split(v) if isinstance(v, str) else [v])]
    return [lb for lb in (doc_label(v) for v in pieces) if lb]


def complete_with_repair(name: str, llm: Any, messages: List[Dict[str, str]], parse: Callable[[str], Any],
                         repair: str, parse_error: str, retries: int,
                         **kwargs: Any) -> Tuple[Any, Optional[str], List[str]]:
    """``(parsed, raw, errors)`` of one JSON-replying call (the belief
    update's, the final scorer's; *name* for the log): *parse* of the reply, else up to
    *retries* more attempts.  A reply that did not parse is shown back with
    the *repair* note; an empty reply is asked again as it was (no empty
    assistant turn).  *parsed* None when no reply parsed (*raw* the last
    reply, None when the first call raised); a call that raises after a
    reply keeps that reply."""
    errors: List[str] = []
    raw = None
    asked = messages
    for attempt in range(1 + retries):
        try:
            reply = llm.complete(asked, **kwargs) or ""
        except Exception as e:
            logger.warning("%s: call failed: %s", name, e)
            return None, raw, errors + [f"{e}"]
        raw = reply if reply or raw is None else raw
        parsed = parse(reply)
        if parsed is not None:
            return parsed, reply, errors
        errors.append(f"parse (attempt {attempt + 1}): {parse_error}" if reply else
                      f"parse (attempt {attempt + 1}): empty reply")
        asked = ([*messages, {"role": "assistant", "content": reply}, {"role": "user", "content": repair}]
                 if reply else messages)
    return None, raw, errors


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


class Passage(NamedTuple):
    """One passage of a call: its label, document id and text."""
    label: str
    doc_id: str
    text: str


# One turn of a call: its step (its label is "t<step>") and its text.
Turn = Tuple[int, str]


class LLMBeliefUpdater:
    """The belief update call: one JSON-replying LLM call per step, its
    retries and its errors."""

    def __init__(self, llm_client: Any, model_name: Optional[str] = None, query_shape: str = SINGLE_TARGET,
                 max_passage_chars: Optional[int] = MAX_PASSAGE_CHARS,
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
        """A passage's text as the call shows it (cut as the agent's view
        of it); its spans are checked against this text only."""
        return cut_passage(text, self._max_passage_chars)

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
              ) -> Optional[Tuple[List[Observation], List[Observation], List[CandidateUpdate]]]:
        """``(claims, docs, updates)`` of the reply *raw* of the call of
        step *step*: the span of a claim checked against its turn (the only
        turn when the call has one and the label is unknown; the claim's
        step is its turn's), those of a document observation against its
        passage as the call showed it (``shown_text``: a span from past the
        cut is not verified; its step is *step*), the items of one
        criterion and passage merged; None unless the reply is a JSON
        object with the three lists."""
        data = parse_json_object(raw or "")
        if data is None or not all(isinstance(data.get(key), list) for key in REPLY_KEYS):
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
                            step=turn_step, candidate=" ".join(str(item.get("candidate") or "").split()) or None,
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
                                   candidate=" ".join(str(item.get("candidate") or "").split()),
                                   reason=" ".join(str(item.get("reason") or "").split()),
                                   status=str(item.get("belief_status") or "").strip(),
                                   support=_labels(item.get("support")), against=_labels(item.get("against")))
                   for item in data["candidates"] if isinstance(item, dict)]
        return claims, list(docs.values()), updates

    def update(self, query: str, state: BeliefState, turns: List[Turn],
               passages: List[Passage], step: int) -> Tuple[Any, Optional[str], List[str]]:
        """``(parsed, raw, errors)`` of the call of step *step*: *parsed* as
        ``parse`` gives it, None when no reply parsed (*raw* the last
        reply, None when the first call raised; ``complete_with_repair``)."""
        return complete_with_repair(
            "LLMBeliefUpdater", self._llm, self.prompt(query, state, turns, passages),
            lambda raw: self.parse(raw, turns, passages, step), UPDATE_REPAIR,
            f"no JSON object with the lists {list(REPLY_KEYS)}", self._parse_retries,
            max_tokens=self.max_tokens, temperature=self._temperature)


# The retry message after a reply that did not parse.
UPDATE_REPAIR = ("Your reply could not be read. Respond with ONLY a raw JSON object with the lists "
                 + ", ".join(f'"{k}"' for k in REPLY_KEYS) + ", in this order.")


def build_belief_updater(llm_client: Any, model: str = "", query_shape: str = SINGLE_TARGET,
                         max_passage_chars: Optional[int] = MAX_PASSAGE_CHARS) -> LLMBeliefUpdater:
    """Build the belief updater on one LLM client (named *model*) for the
    queries of one shape.  A document is cut at *max_passage_chars*, the
    run's shared setting, so the update reads the text the agent read."""
    if llm_client is None:
        raise ValueError("the belief updater needs an LLM client")
    return LLMBeliefUpdater(
        llm_client, model_name=model, query_shape=query_shape,
        max_passage_chars=max_passage_chars,
        max_tokens=SET_MAX_TOKENS if query_shape == SET else DEFAULT_MAX_TOKENS,
    )
