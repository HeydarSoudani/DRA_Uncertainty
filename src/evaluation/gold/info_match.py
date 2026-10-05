"""Match gold units of information against criteria with an LLM judge.

A gold unit is a nugget question (NeuCLIR, RAGTIME) or a reference clue
(BrowseComp-Plus).  The judge is answer-blind: it decides whether a
criterion *asks for* the unit's information, not whether a text contains its
answer (that is the generation evaluation's job).  One call per query scores
every (unit, criterion) pair:

* 1: the criterion names or clearly implies every qualifier of the unit
  (the details that set it apart from the query's topic: group, measure,
  time, place, event, cause), possibly as one item of a short list.
* 0.5: the unit falls within the criterion's topic, but the criterion misses
  a qualifier.

The judge lists each unit's qualifiers before scoring it; they are returned
for inspection.  Run the judge with its reasoning on: without it, qwen3-32b
lists the qualifiers but still scores a criterion that restates the query's
topic 1 for every unit.
* 0: otherwise (pairs left out of the output).

A criterion and a unit with the same normalized text score 1 without the
judge.
"""

import logging
import re
from typing import Any, List, Optional, Tuple

from utils.text_utils import parse_json_object

logger = logging.getLogger(__name__)


INFO_MATCH_SYSTEM = """\
You compare two lists written for the same query: gold units of information (questions or conditions a complete response must cover) and criteria (the pieces of information a research agent will look for). For each gold unit, decide which criteria ask for its information.

For each gold unit, first list its qualifiers: the specific details that set it apart from the query's general topic, such as the group, quantity or measure, time, place, event or cause it is about (for example "women", "in their 20s", "2020 compared with 2019"). The query's general topic itself is not a qualifier.

Scores:
- 1: The criterion names or clearly implies every qualifier of the gold unit, possibly as one item of a short list, so a document that answers the criterion states the gold unit's information.
- 0.5: The gold unit falls within the criterion's topic, but the criterion misses at least one of its qualifiers.
- 0: The criterion does not ask for the gold unit's information.

Examples, for a query about microplastics in rivers:
- Gold "How much microplastic does a typical river carry per cubic meter?" (qualifiers: concentration) and criterion "measured concentrations of microplastics in rivers": 1.
- Gold "What share of river microplastics comes from tire wear?" (qualifiers: share; tire wear) and criterion "the share of each source of river microplastics, such as tire wear and textiles": 1.
- The same gold and criterion "the main sources of microplastics in rivers": 0.5 (it misses the share).
- The same gold and criterion "microplastic pollution in rivers": 0.5 (it misses both qualifiers).

Rules:
1. Judge what the criterion asks for, not whether its answer is known; ignore differences in wording.
2. A criterion that only restates the query's general topic scores 0.5 for every gold unit that has qualifiers.
3. A gold unit may match several criteria and a criterion may match several gold units.

Respond with ONLY a raw JSON object, no markdown, no code blocks, with one entry per gold unit, in order:
{"units": [{"gold": gold unit number, "qualifiers": ["..."], "matches": [{"criterion": criterion number, "score": 1 or 0.5}]}]}
List in "matches" only the criteria with a score above 0."""

INFO_MATCH_USER_TEMPLATE = """\
Query: {query}

Gold units:
{gold}

Criteria:
{criteria}"""


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


class LLMInfoMatcher:
    """Scores every (gold unit, criterion) pair of one query.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``;
            None keeps only the exact-text scores.
        max_tokens / temperature: LLM call settings.
    """

    def __init__(self, llm_client: Any, max_tokens: int = 16000, temperature: float = 0.0) -> None:
        self._llm = llm_client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def match(
        self, query: str, gold: List[str], criteria: List[str],
    ) -> Tuple[List[List[float]], List[List[str]], Optional[str]]:
        """``(scores, qualifiers, error)``: ``scores[i][j]`` for gold unit *i*
        and criterion *j*, the qualifiers the judge listed per unit, and an
        error that is None on success (the exact-text scores stand when the
        call fails)."""
        scores = [[0.0] * len(criteria) for _ in gold]
        qualifiers: List[List[str]] = [[] for _ in gold]
        gold_norm, crit_norm = [_norm(g) for g in gold], [_norm(c) for c in criteria]
        for i, g in enumerate(gold_norm):
            for j, c in enumerate(crit_norm):
                if g and g == c:
                    scores[i][j] = 1.0
        if not gold or not criteria or self._llm is None:
            return scores, qualifiers, None

        messages = [
            {"role": "system", "content": INFO_MATCH_SYSTEM},
            {"role": "user", "content": INFO_MATCH_USER_TEMPLATE.format(
                query=query,
                gold="\n".join(f"{i + 1}. {g}" for i, g in enumerate(gold)),
                criteria="\n".join(f"{j + 1}. {c}" for j, c in enumerate(criteria)),
            )},
        ]
        try:
            raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        except Exception as e:
            logger.warning("LLMInfoMatcher: LLM call failed", exc_info=True)
            return scores, qualifiers, f"llm_call: {e}"
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("units"), list):
            return scores, qualifiers, "parse: no JSON object with a 'units' list"
        for unit in data["units"]:
            try:
                i = int(unit["gold"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= i < len(gold):
                continue
            qualifiers[i] = [str(q) for q in unit.get("qualifiers") or []]
            for m in unit.get("matches") or []:
                try:
                    j, s = int(m["criterion"]) - 1, float(m["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= j < len(criteria) and s in (0.5, 1.0):
                    scores[i][j] = max(scores[i][j], s)
        return scores, qualifiers, None
