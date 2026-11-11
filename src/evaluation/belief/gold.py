"""Whether a text has the gold answer of a criterion.

Shared by the belief update evaluation (``evaluation.belief.evaluator``)
and belief model training (``belief_model_training.nugget_gold``).  A
criterion is a plain dict with at least ``text`` and ``answers``.

- ``states``: a text states one of the gold answers: an answer occurs in it
  as whole words (case, accents and punctuation ignored), outside a longer
  entity name of the query's gold structure that contains it (*others*),
  or, for a numeric answer, a number of the text equals it (relative
  tolerance 1e-3, "12,388" = 12388).
- ``AnswerJudge``: the LLM fallback when the match does not decide, its
  verdicts cached on disk by text, so a pair is asked once.  Two questions:
  a value (``CANDIDATE``: is it the gold answer?) and a text (``SPANS``:
  does it state the gold answer?).
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.training_pool.annotations import mentions

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# The judge's question per kind of text: an agent turn or spans (does it
# state the answer?) or a value (is it the answer?).
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

# The kinds of text the judge is asked about: a candidate (is it the gold
# answer?) and spans or a turn (do they state it?).
CANDIDATE = "candidate"
SPANS = "turn"

# Gold answers shown to the judge per criterion.
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


def states(text: str, answers: List[str], others: List[str] = ()) -> bool:
    """*text* states one of *answers* (whole words outside the longer names
    of *others*, or an equal number)."""
    numbers = None
    for answer in answers:
        value = _as_number(answer)
        if value is None:
            if mentions(text, answer, list(others)):
                return True
            continue
        if numbers is None:
            numbers = _numbers(text)
        if any(abs(n - value) <= 1e-3 * max(abs(value), 1.0) for n in numbers):
            return True
    return False


class AnswerJudge:
    """The LLM fallback of ``states``, with its verdicts cached in *cache*
    (JSONL, one ``{key, stated}`` per asked pair)."""

    def __init__(self, model: str, cache: Path, workers: int = 16) -> None:
        from evaluation.judge import YesNoJudge

        self._judge = YesNoJudge(model)
        self.model = model
        self.cache_path = Path(cache)
        self.workers = workers
        self._cache: Dict[str, bool] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["stated"]

    def key(self, kind: str, criterion: Dict[str, Any], text: str) -> str:
        blob = json.dumps([self.model, kind, criterion["text"], criterion["answers"], text], ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def uncached(self, items: List[Tuple[str, Dict[str, Any], str]]) -> int:
        """How many distinct pairs of *items* are not cached yet (the ones
        ``resolve`` sends to the judge)."""
        return len({self.key(*item) for item in items} - self._cache.keys())

    def resolve(self, items: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, bool]:
        """The verdict of each ``(kind, criterion, text)``, by its ``key``;
        uncached ones are asked (a reply without YES counts as no; one
        without YES/NO is not cached)."""
        from evaluation.judge import has_yes_no, judge_all, run_sync

        keyed = {self.key(*item): item for item in items}
        todo = [(k, item) for k, item in keyed.items() if k not in self._cache]
        if todo:
            def one(pair):
                key, (kind, criterion, text) = pair
                prompt = JUDGE_PROMPTS[kind].format(
                    criterion=criterion["text"], answers=" | ".join(criterion["answers"][:MAX_JUDGE_ANSWERS]),
                    text=text)
                reply = run_sync(self._judge.ask([{"role": "user", "content": prompt}]))
                # A reply without YES/NO (malformed, or a timeout) is no
                # verdict: it counts as no here and is asked again next time.
                return {"key": key, "stated": ("YES" in reply.upper()) if has_yes_no(reply) else None}

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
