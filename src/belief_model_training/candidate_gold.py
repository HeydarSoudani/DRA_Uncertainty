"""Gold of the reasoner side: whether an answer candidate is a gold
answer, and which criteria each agent turn states the gold answer of.

The gold intermediate answers are the pool's (``gold_criteria``: per
criterion the entity of the hop, or for a set query the members or their
values).  A text matches a criterion's gold when one of its answers occurs
in it as whole words (case, accents and punctuation ignored), outside a
longer entity name of the query's gold structure that contains it, or, for
a numeric answer, when a number of the text equals it (relative tolerance
1e-3, "12,388" = 12388).  Without a match an LLM judge decides
(``AnswerJudge``, the correctness judge's model; null: the match alone),
its verdicts cached on disk by text, so a pair is asked once.  A criterion
without gold answers (the source states none: ``answers`` empty) has no
gold.

The rest criterion of a set query ("any other member: <property>") is
matched member by member: a text matches when it names a member and states
that member's own value (``gold_answers``: "<member>: <values>" per member),
so a value is never credited to the wrong member.  A rest candidate
"<member>: <value>" (``REST_PAIR``) is matched part by part.

- ``CandidateGold``: a candidate is correct when its value matches the
  criterion's gold (judge question: is the candidate the gold answer?).
  The candidates of any belief chain are resolved in batches
  (``resolve``), then read with ``correct``.
- ``annotate_stated``: per agent turn, the criteria whose gold answer it
  states (``Episode.steps[t].stated``, ``Episode.answer_stated``), for the
  missed answers of r_reas: a turn that states a gold answer no candidate
  of the criterion matches.  The judge is asked only when the criterion's
  gold documents were shown before the turn (a turn that states an answer
  from memory is found by the match only).
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from belief_model_component.types import REST, REST_PAIR
from indexing_corpus_dataset.training_pool.annotations import entity_names, mentions, value_texts

from .episodes import Episode

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# The judge's question per kind of text: an agent turn (does it state the
# answer?) or an answer candidate (is it the answer?).
JUDGE_PROMPTS = {
    "turn": """\
Criterion: {criterion}
Gold answer: {answers}

Text:
{text}

Does the text state the gold answer of the criterion, in any wording (a name, a number in another format, or an unambiguous reference to it)? A text that only asks for it, searches for it without stating it, or states a different answer does not. Answer YES or NO.""",
    "candidate": """\
Criterion: {criterion}
Gold answer: {answers}

Candidate answer: {text}

Is the candidate answer the gold answer of the criterion (or, when several gold answers are listed, one of them), in any wording (a name, a number in another format, or an unambiguous reference to it)? A candidate that names a different entity or value, or only part of a compound answer, is not. Answer YES or NO.""",
}

# Gold answers shown to the judge per criterion (a set has up to 50 members).
MAX_JUDGE_ANSWERS = 50


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _as_number(answer: str) -> Optional[float]:
    try:
        return float(answer.replace(",", ""))
    except ValueError:
        return None


def states(turn: str, answers: List[str], others: List[str] = ()) -> bool:
    """*turn* states one of *answers* (whole words outside the longer names
    of *others*, or an equal number)."""
    numbers = None
    for answer in answers:
        value = _as_number(answer)
        if value is None:
            if mentions(turn, answer, list(others)):
                return True
            continue
        if numbers is None:
            numbers = _numbers(turn)
        if any(abs(n - value) <= 1e-3 * max(abs(value), 1.0) for n in numbers):
            return True
    return False


def _rest_pairs(episode: Episode, criterion: Dict[str, Any]) -> bool:
    """The criterion is matched member by member (the rest criterion of a
    set query with its members)."""
    return criterion["kind"] == REST and bool(episode.gold_members)


def gold_answers(episode: Episode, criterion: Dict[str, Any]) -> List[str]:
    """The gold answers of *criterion* as the judge sees them: per member
    "<member>: <values>" for the rest criterion of a set query, else its
    ``answers``."""
    if _rest_pairs(episode, criterion):
        return [f"{m['entity']}{REST_PAIR}{', '.join(value_texts(m['value']))}" for m in episode.gold_members]
    return criterion["answers"]


def matches(episode: Episode, criterion: Dict[str, Any], text: str) -> bool:
    """*text* (an agent turn or a candidate) matches the gold of *criterion*
    (``states``; member by member for the rest criterion of a set query)."""
    if not _rest_pairs(episode, criterion):
        return states(text, criterion["answers"], episode.gold_names)
    member_part, value_part = (text.split(REST_PAIR, 1) if REST_PAIR in text else (text, text))
    return any(any(mentions(member_part, n, episode.gold_names) for n in entity_names(m["entity"]))
               and states(value_part, value_texts(m["value"]))
               for m in episode.gold_members)


class AnswerJudge:
    """The LLM fallback of ``states``, with its verdicts cached in *cache*
    (JSONL, one ``{key, stated}`` per asked pair)."""

    def __init__(self, model: str, cache: Path, workers: int = 16) -> None:
        from evaluation.judge import YesNoJudge

        self._judge = YesNoJudge(model)
        self.model = model
        self.cache_path = cache
        self.workers = workers
        self._cache: Dict[str, bool] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["stated"]

    def key(self, kind: str, criterion: Dict[str, Any], text: str) -> str:
        blob = json.dumps([self.model, kind, criterion["text"], criterion["answers"], text], ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def resolve(self, items: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, bool]:
        """The verdict of each ``(kind, criterion, text)``, by its ``key``;
        uncached ones are asked (a reply without YES counts as no)."""
        from evaluation.judge import judge_all, run_sync

        keyed = {self.key(*item): item for item in items}
        todo = [(k, item) for k, item in keyed.items() if k not in self._cache]
        if todo:
            def one(pair):
                key, (kind, criterion, text) = pair
                prompt = JUDGE_PROMPTS[kind].format(
                    criterion=criterion["text"], answers=" | ".join(criterion["answers"][:MAX_JUDGE_ANSWERS]),
                    text=text)
                reply = run_sync(self._judge.ask([{"role": "user", "content": prompt}]))
                return {"key": key, "stated": "YES" in reply.upper()}

            rows = judge_all(todo, one, lambda pair, e: {"key": pair[0], "stated": None}, self.workers,
                             "[answer gold]")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as fh:
                for row in rows:
                    if row["stated"] is None:
                        continue
                    self._cache[row["key"]] = row["stated"]
                    fh.write(json.dumps(row) + "\n")
        return {k: self._cache.get(k, False) for k in keyed}


def _criterion(episode: Episode, cid: str) -> Optional[Dict[str, Any]]:
    return next((c for c in episode.gold_criteria or [] if c["id"] == cid), None)


class CandidateGold:
    """Whether an answer candidate of a criterion is a gold answer: the
    match, else the judge's verdict (``resolve`` asks it in one batch)."""

    def __init__(self, judge: Optional[AnswerJudge] = None) -> None:
        self.judge = judge
        self._verdicts: Dict[str, bool] = {}

    def resolve(self, items: Iterable[Tuple[Episode, str, str]]) -> Dict[str, int]:
        """Ask the judge about every ``(episode, criterion id, value)`` the
        match does not decide; returns counts (items, matched, asked)."""
        counts = {"items": 0, "matched": 0, "asked": 0}
        todo = []
        for episode, cid, value in items:
            c = _criterion(episode, cid)
            if c is None or not c["answers"]:
                continue
            counts["items"] += 1
            if matches(episode, c, value):
                counts["matched"] += 1
            elif self.judge is not None:
                todo.append(("candidate", {**c, "answers": gold_answers(episode, c)}, value))
        if todo:
            verdicts = self.judge.resolve(todo)
            self._verdicts.update(verdicts)
            counts["asked"] = len(verdicts)
        return counts

    def correct(self, episode: Episode, cid: str, value: str) -> Optional[bool]:
        """The candidate *value* of criterion *cid* is a gold answer; None
        when the criterion has no gold answers.  A value the judge was not
        asked about (``resolve``) counts as wrong unless it matches."""
        c = _criterion(episode, cid)
        if c is None or not c["answers"]:
            return None
        if matches(episode, c, value):
            return True
        if self.judge is None:
            return False
        return self._verdicts.get(self.judge.key("candidate", {**c, "answers": gold_answers(episode, c)}, value),
                                  False)


def annotate_stated(episodes: List[Episode], judge: Optional[AnswerJudge] = None) -> Dict[str, int]:
    """Set ``stated`` of every step and ``answer_stated`` of every episode
    with gold criteria; returns counts (pairs matched, asked, stated)."""
    pending: List[Tuple[str, Dict[str, Any], str]] = []
    where: List[Tuple[Set[str], str, str]] = []
    counts = {"pairs": 0, "matched": 0, "asked": 0, "stated_by_judge": 0}
    for ep in episodes:
        if not ep.gold_criteria:
            continue
        # The step each criterion's gold documents were first shown in.
        first_shown: Dict[str, int] = {}
        for t, step in enumerate(ep.steps):
            for c in ep.gold_criteria:
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
            for c in ep.gold_criteria:
                counts["pairs"] += 1
                if matches(ep, c, turn):
                    stated.add(c["id"])
                    counts["matched"] += 1
                elif judge is not None and c["answers"] and first_shown.get(c["id"], len(ep.steps) + 1) < t:
                    gold = {**c, "answers": gold_answers(ep, c)}
                    pending.append(("turn", gold, turn))
                    where.append((stated, c["id"], judge.key("turn", gold, turn)))
    if judge is not None and pending:
        verdicts = judge.resolve(pending)
        counts["asked"] = len(verdicts)
        for stated, cid, key in where:
            if verdicts[key]:
                stated.add(cid)
                counts["stated_by_judge"] += 1
    return counts
