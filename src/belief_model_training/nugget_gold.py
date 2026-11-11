"""Gold of the belief: whether a value is a gold answer of a criterion,
and which criteria each agent turn states the gold answer of.

The gold answers are the pool's (``gold_criteria``: per criterion the
entity of the hop, a set member's value, or the final answer for the
aggregate criterion).  A text matches a criterion's gold when one of its
answers occurs in it as whole words (case, accents and punctuation
ignored), outside a longer entity name of the query's gold structure that
contains it, or, for a numeric answer, when a number of the text equals it
(relative tolerance 1e-3, "12,388" = 12388).  Without a match an LLM judge
decides (``AnswerJudge``, the correctness judge's model; null: the match
alone), its verdicts cached on disk by text, so a pair is asked once.  The
match and the judge live in ``evaluation.belief.gold``, shared with the
belief update evaluation of inference runs.

Only criteria with gold answers have gold (``gold_criterion``); the rest
criterion of a multi-member query has none until members are masked into it (the
gold lists every member as its own criterion), so it is left out.

- ``NuggetGold``: a selected candidate of a criterion is correct when it
  matches the criterion's gold (judge question: is the candidate the gold
  answer?, ``correct``); an observation's spans state the gold when they
  match it (judge question: does the text state the gold answer?,
  ``stated``).  The texts of any belief chain are resolved in batches
  (``resolve``), then read.
- ``annotate_stated``: per agent turn, the criteria whose gold answer it
  states (``Episode.steps[t].stated``, ``Episode.answer_stated``): the
  think observations a call should record.  The judge is asked only when
  the criterion's gold documents were shown before the turn (a turn that
  states an answer from memory is found by the match only).
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from belief_model_component import REST
# The match and the judge are shared with the evaluation (evaluation.belief.gold).
from evaluation.belief.gold import CANDIDATE, SPANS, AnswerJudge, states

from .episodes import Episode

logger = logging.getLogger(__name__)


def gold_criterion(episode: Episode, cid: str) -> Optional[Dict[str, Any]]:
    """The gold criterion *cid* of *episode* when it has gold answers (not
    the rest criterion), else None."""
    c = episode.gold_by_id.get(cid)
    return c if c is not None and c["answers"] and c["kind"] != REST else None


def _criterion(episode: Episode, cid: str, answers: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """The gold criterion *cid* (``gold_criterion``), with only *answers*
    as its gold answers when given (None when *answers* is empty)."""
    c = gold_criterion(episode, cid)
    if c is None or answers is None:
        return c
    return {**c, "answers": list(answers)} if answers else None


def matches(episode: Episode, criterion: Dict[str, Any], text: str) -> bool:
    """*text* (an agent turn or a value) matches the gold of *criterion*."""
    return states(text, criterion["answers"], episode.gold_names)


class NuggetGold:
    """Whether a text of a criterion has its gold: the match, else the
    judge's verdict (``resolve`` asks it in one batch).  A candidate
    (``CANDIDATE``) is asked whether it is the gold answer, an
    observation's spans (``SPANS``, the judge's "turn" question) whether
    they state it."""

    def __init__(self, judge: Optional[AnswerJudge] = None) -> None:
        self.judge = judge
        self._verdicts: Dict[str, bool] = {}

    def resolve(self, items: Iterable[Tuple]) -> Dict[str, int]:
        """Ask the judge about every ``(episode, criterion id, kind, text[,
        answers])`` the match does not decide (*answers*: judge against
        these gold answers of the criterion only, ``_criterion``); returns
        counts (items, matched, asked: pairs not cached yet)."""
        counts = {"items": 0, "matched": 0, "asked": 0}
        todo = []
        for episode, cid, kind, text, *rest in items:
            c = _criterion(episode, cid, rest[0] if rest else None)
            if c is None:
                continue
            counts["items"] += 1
            if matches(episode, c, text):
                counts["matched"] += 1
            elif self.judge is not None:
                todo.append((kind, c, text))
        if todo:
            counts["asked"] = self.judge.uncached(todo)
            self._verdicts.update(self.judge.resolve(todo))
        return counts

    def _has_gold(self, episode: Episode, cid: str, kind: str, text: str,
                  answers: Optional[List[str]] = None) -> Optional[bool]:
        c = _criterion(episode, cid, answers)
        if c is None:
            return None
        if matches(episode, c, text):
            return True
        if self.judge is None:
            return False
        return self._verdicts.get(self.judge.key(kind, c, text), False)

    def correct(self, episode: Episode, cid: str, value: str,
                answers: Optional[List[str]] = None) -> Optional[bool]:
        """The candidate *value* of criterion *cid* is a gold answer (one of
        *answers* when given); None when the criterion has no gold (or
        *answers* is empty).  A candidate the judge was not asked about
        (``resolve``, with the same *answers*) counts as wrong unless it
        matches."""
        return self._has_gold(episode, cid, CANDIDATE, value, answers)

    def stated(self, episode: Episode, cid: str, text: str) -> Optional[bool]:
        """The spans *text* of an observation on *cid* state its gold answer
        (likewise)."""
        return self._has_gold(episode, cid, SPANS, text)


def annotate_stated(episodes: List[Episode], judge: Optional[AnswerJudge] = None) -> Dict[str, int]:
    """Set ``stated`` of every step and ``answer_stated`` of every episode
    with gold criteria; returns counts (pairs matched, asked, stated)."""
    pending: List[Tuple[str, Dict[str, Any], str]] = []
    where: List[Tuple[Set[str], str, str]] = []
    counts = {"pairs": 0, "matched": 0, "asked": 0, "stated_by_judge": 0}
    for ep in episodes:
        scored = [c for c in ep.gold_criteria or [] if gold_criterion(ep, c["id"]) is not None]
        if not scored:
            continue
        # The step each criterion's gold documents were first shown in.
        first_shown: Dict[str, int] = {}
        for t, step in enumerate(ep.steps):
            for c in scored:
                if c["id"] not in first_shown and set(c["doc_ids"]) & set(step.doc_ids):
                    first_shown[c["id"]] = t
        turns = [(t, step.agent_turn) for t, step in enumerate(ep.steps)] + [(len(ep.steps), ep.answer_turn)]
        for t, turn in turns:
            stated: Set[str] = set()
            if t < len(ep.steps):
                ep.steps[t].stated = stated
            else:
                ep.answer_stated = stated
            if turn is None:
                continue
            for c in scored:
                counts["pairs"] += 1
                if matches(ep, c, turn):
                    stated.add(c["id"])
                    counts["matched"] += 1
                elif judge is not None and first_shown.get(c["id"], len(ep.steps) + 1) < t:
                    pending.append(("turn", c, turn))
                    where.append((stated, c["id"], judge.key("turn", c, turn)))
    if judge is not None and pending:
        counts["asked"] = judge.uncached(pending)
        verdicts = judge.resolve(pending)
        for stated, cid, key in where:
            if verdicts[key]:
                stated.add(cid)
                counts["stated_by_judge"] += 1
    return counts
