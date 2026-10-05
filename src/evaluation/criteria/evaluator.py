"""Criteria evaluator: score per-query criteria lists against gold units.

Two modes, by the gold (``evaluation.gold.loaders``):

* ``entities`` (TRQA): the closed criteria are matched to the query's gold
  entities by name (:func:`~evaluation.gold.match_entities`, then
  :class:`~evaluation.gold.LLMEntityMatcher` for the rest).

  - ``recall``: matched entities / gold entities.
  - ``precision``: closed criteria that match an entity / closed criteria.

* ``info`` (nugget questions, reference clues): every (unit, criterion)
  pair is scored 0 / 0.5 / 1 by :class:`~evaluation.gold.LLMInfoMatcher`
  (1: the criterion names every qualifier of the unit; 0.5: the unit is
  only within the criterion's topic).

  - ``recall_strict`` / ``recall_lenient``: units whose best score is 1 /
    at least 0.5, over all units.
  - ``precision_strict`` / ``precision_lenient``: criteria whose best score
    is 1 / at least 0.5, over all criteria.
  - ``recall_strict_vital``: ``recall_strict`` over the ``vital`` units
    (NeuCLIR); ``recall_strict_rare``: over the units with at most
    ``RARE_MAX_DOCS`` support documents (where support documents exist).

Every query also gets ``num_criteria``, ``num_open``, ``num_gold`` and the
match details.  The summary macro-averages each metric over the queries where
it is defined.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..gold import GoldUnit, LLMEntityMatcher, LLMInfoMatcher, match_entities

logger = logging.getLogger(__name__)

# A gold unit with at most this many support documents counts as rare.
RARE_MAX_DOCS = 5

GOLD_MODES = ("entities", "info")


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


class CriteriaEvaluator:
    """Scores criteria lists against gold units.

    Args:
        gold: ``{query_id: [GoldUnit, ...]}``.
        mode: ``"entities"`` or ``"info"`` (see the module docstring).
        llm_client: Object with ``complete(messages, **kwargs) -> str`` for
            the matchers; None keeps only the string passes (exact names for
            ``entities``, exact texts for ``info``).
        max_workers: Queries scored in parallel.
    """

    def __init__(
        self,
        gold: Dict[str, List[GoldUnit]],
        mode: str,
        llm_client: Any = None,
        max_workers: int = 16,
    ) -> None:
        if mode not in GOLD_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {GOLD_MODES}")
        self.gold = gold
        self.mode = mode
        self._entity_matcher = LLMEntityMatcher(llm_client) if llm_client is not None else None
        self._info_matcher = LLMInfoMatcher(llm_client)
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # One query
    # ------------------------------------------------------------------

    def _entities(self, units: List[GoldUnit], criteria: List[Dict]) -> Dict[str, Any]:
        members = [c for c in criteria if c.get("kind", "closed") != "open"]
        names, texts = [u.text for u in units], [c["text"] for c in members]
        matches = {i: (j, "string") for i, j in match_entities(names, texts).items()}
        errors = []
        left_e = [i for i in range(len(names)) if i not in matches]
        used = {j for j, _ in matches.values()}
        left_t = [j for j in range(len(texts)) if j not in used]
        if self._entity_matcher is not None and left_e and left_t:
            extra, error = self._entity_matcher.match([names[i] for i in left_e], [texts[j] for j in left_t])
            if error:
                errors.append(f"entity_matcher: {error}")
            for a, b in extra.items():
                matches[left_e[a]] = (left_t[b], "llm")
        return {
            "recall": round(len(matches) / len(names), 4) if names else None,
            "precision": round(len(matches) / len(members), 4) if members else None,
            "num_member_criteria": len(members),
            "matches": [
                {"gold": units[i].text, "criterion": members[j]["id"], "by": by}
                for i, (j, by) in sorted(matches.items())
            ],
            "unmatched_gold": [units[i].text for i in range(len(names)) if i not in matches],
            "errors": errors,
        }

    def _info(self, query: str, units: List[GoldUnit], criteria: List[Dict]) -> Dict[str, Any]:
        gold, texts = [u.text for u in units], [c["text"] for c in criteria]
        scores, qualifiers, error = self._info_matcher.match(query, gold, texts)
        best_gold = [max(row) if row else 0.0 for row in scores]
        best_crit = [max((scores[i][j] for i in range(len(gold))), default=0.0) for j in range(len(texts))]

        def recall(idx: List[int], level: float) -> Optional[float]:
            return round(sum(best_gold[i] >= level for i in idx) / len(idx), 4) if idx else None

        everything = list(range(len(units)))
        vital = [i for i, u in enumerate(units) if u.importance == "vital"]
        rare = [i for i, u in enumerate(units)
                if u.num_support_docs is not None and u.num_support_docs <= RARE_MAX_DOCS]
        return {
            "recall_strict": recall(everything, 1.0),
            "recall_lenient": recall(everything, 0.5),
            "precision_strict": round(sum(s >= 1.0 for s in best_crit) / len(texts), 4) if texts else None,
            "precision_lenient": round(sum(s >= 0.5 for s in best_crit) / len(texts), 4) if texts else None,
            "recall_strict_vital": recall(vital, 1.0),
            "recall_strict_rare": recall(rare, 1.0),
            "matches": [
                {"gold": u.text, "importance": u.importance, "num_support_docs": u.num_support_docs,
                 "qualifiers": qualifiers[i], "score": best_gold[i],
                 "criteria": [criteria[j]["id"] for j in range(len(texts)) if scores[i][j] == best_gold[i] > 0]}
                for i, u in enumerate(units)
            ],
            "errors": [f"info_matcher: {error}"] if error else [],
        }

    def evaluate_query(self, query_id: str, query: str, criteria: List[Dict]) -> Dict[str, Any]:
        units = self.gold.get(query_id, [])
        record: Dict[str, Any] = {
            "query_id": query_id,
            "num_gold": len(units),
            "num_criteria": len(criteria),
            "num_open": sum(c.get("kind") == "open" for c in criteria),
        }
        if self.mode == "entities":
            record.update(self._entities(units, criteria))
        else:
            record.update(self._info(query, units, criteria))
        return record

    # ------------------------------------------------------------------
    # All queries
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: Dict[str, str],
        criteria: Dict[str, List[Dict]],
        infos: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """Score every query that has gold units and a criteria list.

        *infos* (``{query_id: criteria_info}``) adds the extraction errors
        to the summary.  Returns ``{"summary",
        "per_query"}``.
        """
        qids = [q for q in queries if q in self.gold and q in criteria]
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            per_query = list(tqdm(
                pool.map(lambda q: self.evaluate_query(q, queries[q], criteria[q]), qids),
                total=len(qids), desc=f"criteria eval ({self.mode})",
            ))
        return {"summary": self.summarize(per_query, infos), "per_query": per_query}

    def summarize(self, per_query: List[Dict], infos: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        metrics = (
            ("recall", "precision") if self.mode == "entities" else
            ("recall_strict", "recall_lenient", "precision_strict", "precision_lenient",
             "recall_strict_vital", "recall_strict_rare")
        )
        summary: Dict[str, Any] = {"mode": self.mode, "num_queries": len(per_query)}
        for m in metrics:
            summary[m] = _mean([r[m] for r in per_query if r.get(m) is not None])
        for m in ("num_criteria", "num_open", "num_gold"):
            summary[f"mean_{m}"] = _mean([r[m] for r in per_query])
        summary["num_empty_criteria"] = sum(r["num_criteria"] == 0 for r in per_query)
        summary["num_matcher_errors"] = sum(bool(r["errors"]) for r in per_query)
        if infos:
            summary["num_extraction_errors"] = sum(
                bool((infos.get(r["query_id"]) or {}).get("errors")) for r in per_query
            )
        return summary
