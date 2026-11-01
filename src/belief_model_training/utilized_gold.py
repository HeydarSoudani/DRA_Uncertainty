"""Gold of the reasoner side: which criteria each agent turn states the
gold answer of.

A rise of criterion k at step t-1 is utilized when the agent's turn of step
t (its reasoning and queries; the answer turn after the last step) states
k's gold intermediate answer (``gold_criteria``: the entity of the hop, or
for a set query a member or a value).  ``Episode.steps[t].stated`` and
``Episode.answer_stated`` hold, per turn, the ids of the criteria whose gold
answer it states, so a rise judged on any turn of any belief chain has its
gold.

A turn states an answer when the answer occurs in it as whole words
(case, accents and punctuation ignored), outside a longer entity name of
the query's gold structure that contains it, or, for a numeric answer, when a
number of the turn equals it (relative tolerance 1e-3, "12,388" = 12388).
A (turn, criterion) pair without such a match is asked of an LLM judge
(``YesNoJudge``, the correctness judge's model) when the criterion's gold
documents were shown before the turn (otherwise the turn cannot have taken
the answer from them); the verdicts are cached on disk by turn text, so a
pair is asked once.  Without a judge model the match alone decides.  A
criterion without gold answers (the source states none: ``answers`` empty)
is never stated.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from indexing_corpus_dataset.training_pool.annotations import mentions

from .episodes import Episode

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

JUDGE_PROMPT = """\
Criterion: {criterion}
Gold answer: {answers}

Text:
{turn}

Does the text state the gold answer of the criterion, in any wording (a name, a number in another format, or an unambiguous reference to it)? A text that only asks for it, searches for it without stating it, or states a different answer does not. Answer YES or NO."""

# Gold answers shown to the judge per criterion (a set has many members).
MAX_JUDGE_ANSWERS = 30


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


class UtilizedJudge:
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

    def key(self, criterion: Dict[str, Any], turn: str) -> str:
        blob = json.dumps([self.model, criterion["text"], criterion["answers"], turn], ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def resolve(self, pairs: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, bool]:
        """The verdict of each ``(key, criterion, turn)``; uncached ones are
        asked (a reply without YES counts as not stated)."""
        from evaluation.judge import judge_all, run_sync

        todo = [p for p in pairs if p[0] not in self._cache]
        if todo:
            def one(pair):
                key, criterion, turn = pair
                prompt = JUDGE_PROMPT.format(criterion=criterion["text"],
                                             answers=" | ".join(criterion["answers"][:MAX_JUDGE_ANSWERS]), turn=turn)
                reply = run_sync(self._judge.ask([{"role": "user", "content": prompt}]))
                return {"key": key, "stated": "YES" in reply.upper()}

            rows = judge_all(todo, one, lambda pair, e: {"key": pair[0], "stated": None}, self.workers,
                             "[utilized gold]")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as fh:
                for row in rows:
                    if row["stated"] is None:
                        continue
                    self._cache[row["key"]] = row["stated"]
                    fh.write(json.dumps(row) + "\n")
        return {key: self._cache.get(key, False) for key, _, _ in pairs}


def annotate_utilized(episodes: List[Episode], judge: Optional[UtilizedJudge] = None) -> Dict[str, int]:
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
                if states(turn, c["answers"], ep.gold_names):
                    stated.add(c["id"])
                    counts["matched"] += 1
                elif judge is not None and c["answers"] and first_shown.get(c["id"], len(ep.steps) + 1) < t:
                    pending.append((judge.key(c, turn), c, turn))
                    where.append((stated, c["id"], pending[-1][0]))
    if judge is not None and pending:
        verdicts = judge.resolve(pending)
        counts["asked"] = len(pending)
        for stated, cid, key in where:
            if verdicts[key]:
                stated.add(cid)
                counts["stated_by_judge"] += 1
    return counts
