"""Match nugget question-answer pairs against criteria, as Auto-ARGUE does for reports.

The criteria evaluation of a report dataset mirrors the report evaluation
(``evaluation.answer.argue``) so the two can be compared nugget by nugget:
the same nuggets (the Auto-ARGUE banks, ``gold.loaders``), the same judge and
settings (Qwen3-32B, temperature 0, reasoning off, a YES/NO reply asked again
twice, then counted as NO), and the same rule for a covered nugget.

Auto-ARGUE asks, for every (report sentence, nugget question, gold answer),
whether the sentence agrees with the question-answer pair.  A criterion holds
no answer, so the check here is whether the criterion *asks for* the pair:
a document that establishes the criterion would state that answer to that
question.  It runs for every (criterion, question, answer); like the report's
sentences, the criteria together answer a nugget when their matched answers
meet its aggregator (``answer.argue.nugget_covered``).  Every answer is
checked (a report sentence is checked only against the answers of the
documents it cites; criteria cite none), so a list costs (criteria x gold
answers) judge calls; the judgments are cached one by one
(:class:`NuggetAskMatcher`).
"""

import asyncio
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..common import read_jsonl
from ..judge import YesNoJudge, has_yes_no, is_yes, run_sync

logger = logging.getLogger(__name__)

# Judge requests in flight per scored criteria list.
MAX_CONCURRENT = 8

NUGGET_ASK_SYSTEM = """\
You are an expert at determining what information a research criterion asks for.
A research agent will search for documents that establish each criterion. Your task is to determine if a criterion asks for the information in a provided question-answer pair, that is, whether a document that establishes the criterion would state this answer to the question.
Consider the following aspects when making your decision:
1. Core meaning of the criterion and of the question
2. Scope of the criterion, including what it says to leave out ("..., not ...")
3. Specificity: a criterion that only names the general topic does not ask for a specific detail of it
4. Semantic equivalence
5. Direct vs. indirect relation

Respond with ONLY 'YES' or 'NO' in English."""

NUGGET_ASK_USER_TEMPLATE = """\
Does the criterion ask for the question-answer pair?

Examples:
1. Criterion: measured concentrations of microplastics in rivers and lakes
Question: How much microplastic does a typical river carry per cubic meter?
Answer: About 5 particles
Does the criterion ask for the question-answer pair (YES/NO): YES

2. Criterion: microplastic pollution in rivers
Question: What share of river microplastics comes from tire wear?
Answer: About 30%
Does the criterion ask for the question-answer pair (YES/NO): NO

3. Criterion: the share of each source of river microplastics, such as tire wear and textiles
Question: What share of river microplastics comes from tire wear?
Answer: About 30%
Does the criterion ask for the question-answer pair (YES/NO): YES

4. Criterion: how microplastics enter freshwater ecosystems, not marine ecosystems
Question: How do microplastics reach the open ocean?
Answer: Carried by rivers
Does the criterion ask for the question-answer pair (YES/NO): NO

5. Criterion: the effects of microplastics on the health of freshwater fish
Question: Which organ of freshwater fish do microplastics damage most?
Answer: The gut
Does the criterion ask for the question-answer pair (YES/NO): YES

6. Criterion: regulations proposed to limit microplastic pollution
Question: What did the 2019 proposal to limit microplastics require?
Answer: A ban on microbeads in cosmetics
Does the criterion ask for the question-answer pair (YES/NO): YES

Criterion: {criterion}
Question: {question}
Answer: {answer}
Does the criterion ask for the question-answer pair (YES/NO):"""

# The check's answer when the judge gives no YES/NO (NO, as Auto-ARGUE's
# sentence-answers-question check).
DEFAULT_YES = False

# Hash of the prompts, part of every judgment's cache key.
_PROMPT_HASH = hashlib.sha256((NUGGET_ASK_SYSTEM + NUGGET_ASK_USER_TEMPLATE).encode()).hexdigest()[:16]


def judgment_key(judge_model: str, criterion: str, question: str, answer: str) -> str:
    """Cache key of one YES/NO judgment: the judge, the prompts and the
    (criterion, question, answer) triple."""
    payload = [judge_model, _PROMPT_HASH, criterion, question, answer]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class NuggetAskMatcher:
    """Which nugget answers each criterion asks for, judged YES/NO per
    (criterion, question, answer).

    Every YES/NO judgment is cached by :func:`judgment_key`, so a list that
    shares criteria with one judged before (an edited list, a list judged
    again after a failed call) asks only the new triples.  The cache lives in
    memory and, after :meth:`use_cache_file`, in a JSONL file as well.  A
    reply without YES/NO is not cached.  Thread-safe.

    Args:
        judge_model: Judge model, resolved by ``reasoner_component.api``
            (Auto-ARGUE's judge client and settings).
    """

    def __init__(self, judge_model: str) -> None:
        self.judge_model = judge_model
        self._cache: Dict[str, bool] = {}
        self._cache_path: Optional[Path] = None
        self._lock = threading.Lock()

    def use_cache_file(self, path: Union[str, Path]) -> None:
        """Load the judgments saved in *path* and append new ones to it."""
        path = Path(path)
        with self._lock:
            if path == self._cache_path:
                return
            for rec in read_jsonl(path):
                self._cache[rec["key"]] = rec["yes"]
            self._cache_path = path

    def match(
        self, units: List[Any], criteria: List[str],
    ) -> Tuple[List[List[Set[str]]], Dict[str, int]]:
        """``(matched, stats)``: ``matched[i][j]`` is the set of gold answers
        of unit *i* (a ``GoldUnit`` with ``answers``) that criterion *j* asks
        for; *stats* has the judge's ``calls``, its ``malformed`` replies and
        the judgments taken from the cache (``cached``).  A call that fails
        raises, after the other calls have finished and been cached."""
        triples = [(i, j, a) for i, u in enumerate(units) for j in range(len(criteria)) for a in u.answers]
        keys = [judgment_key(self.judge_model, criteria[j], units[i].text, a) for i, j, a in triples]
        with self._lock:
            verdicts = [self._cache.get(k) for k in keys]
        todo = [n for n, v in enumerate(verdicts) if v is None]

        # A new client per list: each call runs its own event loop.
        chat = YesNoJudge(self.judge_model)

        async def judge_todo():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)

            async def one(n):
                i, j, answer = triples[n]
                messages = [
                    {"role": "system", "content": NUGGET_ASK_SYSTEM},
                    {"role": "user", "content": NUGGET_ASK_USER_TEMPLATE.format(
                        criterion=criteria[j], question=units[i].text, answer=answer)},
                ]
                async with semaphore:
                    return await chat.ask(messages)

            return await asyncio.gather(*(one(n) for n in todo), return_exceptions=True)

        replies = run_sync(judge_todo()) if todo else []
        fresh, failure = [], None
        for n, reply in zip(todo, replies):
            if isinstance(reply, BaseException):
                failure = failure or reply
                continue
            verdicts[n] = is_yes(reply, DEFAULT_YES)
            if has_yes_no(reply):
                fresh.append({"key": keys[n], "yes": verdicts[n]})
        self._save(fresh)
        if failure is not None:
            raise failure

        matched: List[List[Set[str]]] = [[set() for _ in criteria] for _ in units]
        for (i, j, answer), yes in zip(triples, verdicts):
            if yes:
                matched[i][j].add(answer)
        return matched, {"calls": chat.calls, "malformed": chat.malformed, "cached": len(triples) - len(todo)}

    def _save(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        with self._lock:
            self._cache.update((r["key"], r["yes"]) for r in records)
            if self._cache_path is not None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "a", encoding="utf-8") as f:
                    f.writelines(json.dumps(r) + "\n" for r in records)
