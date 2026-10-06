"""Criteria evaluator: score per-query criteria lists against gold units.

Three modes, by the gold (``evaluation.gold.loaders``):

* ``entities`` (TRQA): the closed criteria are matched to the query's gold
  entities by name (:func:`~evaluation.gold.match_entities`, then
  :class:`~evaluation.gold.LLMEntityMatcher` for the rest).

  - ``recall``: matched entities / gold entities.
  - ``precision``: closed criteria that match an entity / closed criteria.

* ``nuggets`` (NeuCLIR, RAGTIME): scored as Auto-ARGUE scores the reports
  (``evaluation.answer.argue``), so the two compare nugget by nugget: the
  same nuggets and queries (the Auto-ARGUE banks), the same judge and
  settings, and the same rule for a covered nugget.  The judge
  (:class:`~evaluation.gold.NuggetAskMatcher`) says YES or NO for every
  (criterion, nugget question, gold answer): does the criterion ask for that
  question-answer pair?  A nugget is covered when the answers matched by
  all the criteria meet its aggregator (one answer for OR, all for AND).

  - ``nugget_coverage``: covered nuggets / nuggets.
  - ``nugget_coverage_weighted``: the same with vital nuggets weighted 2 and
    okay ones 1 (all 1 without importance labels), as Auto-ARGUE.
  - ``num_unmatched_criteria``: criteria that ask for no nugget answer (a
    diagnostic: the nuggets are not every piece of useful information).

  The criteria are also checked against the request itself
  (:class:`.request_coverage.RequestCoverageJudge`, report requests):
  ``constraint_recall`` (the request's constraints carried by a criterion),
  ``exclusion_recall`` (the same over what the
  request leaves out) and ``num_violations`` (criteria that ask for what the
  request leaves out).

  :func:`compare_with_report` crosses each nugget's criteria label with the
  report's (Auto-ARGUE's answered nuggets).

* ``info`` (a reference criteria list, ``--reference``): every (unit,
  criterion) pair is scored 0 / 0.5 / 1 by :class:`~evaluation.gold.LLMInfoMatcher`
  (1: the criterion names every qualifier of the unit; 0.5: the unit is
  only within the criterion's topic).

  - ``recall_strict`` / ``recall_lenient``: units whose best score is 1 /
    at least 0.5, over all units.
  - ``precision_strict`` / ``precision_lenient``: criteria whose best score
    is 1 / at least 0.5, over all criteria.

A judge call that fails is retried once; a query whose judge still fails
keeps its record (with ``errors``) but is left out of the averages.

Every query also gets ``num_criteria``, ``num_open``, ``num_gold``, the
match details, the ``method`` and ``judge_model`` and a ``criteria_hash`` of
the scored list.  The summary (:meth:`CriteriaEvaluator.summarize`) is laid
out as the report's (``generation.nuggets``): the method and counts, then
``metrics`` (the scores, macro-averaged over the queries where they are
defined and the judges did not fail) and ``stats`` (list sizes and
diagnostics).

In a run (uncertainty estimator ``monitor`` or ``inform``, dataset with
``criteria_gold``) the estimator scores each query's criteria in the
background while the agent runs (:meth:`CriteriaEvaluator.score`) and keeps
the record in the meta line of ``uncertainty/{qid}.jsonl`` as
``criteria_eval``.  :meth:`CriteriaEvaluator.evaluate_run` reuses it while
the criteria, the method and the judge are unchanged, and writes a record it
judges again back into that meta line.  The nugget judgments of the queries
judged at evaluation are cached one by one in ``criteria_eval_judgments.jsonl``
under the run directory.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm

from indexing_corpus_dataset.layout import DATASET_SPECS
from uncertainty_estimator.types import MULTI_ASPECT, MULTI_SOURCE_KINDS

from ..answer.argue import nugget_covered
from ..common import mean_or_none
from ..gold import GoldUnit, LLMEntityMatcher, LLMInfoMatcher, NuggetAskMatcher, match_entities
from ..uncertainty import load_uncertainty_meta, update_uncertainty_meta
from .request_coverage import RequestCoverageJudge

logger = logging.getLogger(__name__)

GOLD_MODES = ("entities", "nuggets", "info")

#: Scoring method of each mode, saved with every record; a record of another
#: method is judged again.
METHODS = {"entities": "entity_match", "nuggets": "argue_nugget_ask", "info": "info_match"}

#: Metrics of each mode, macro-averaged in the summary's ``metrics``.
MODE_METRICS = {
    "entities": ("recall", "precision"),
    "nuggets": ("nugget_coverage", "nugget_coverage_weighted"),
    "info": ("recall_strict", "recall_lenient", "precision_strict", "precision_lenient"),
}

# Auto-ARGUE's nugget weights (``auto_argue.score``); without any, all 1.
_IMPORTANCE_WEIGHTS = {"vital": 2.0, "okay": 1.0}

#: Cached YES/NO nugget judgments (``nuggets`` mode), under the run directory.
CRITERIA_JUDGMENTS_FILE = "criteria_eval_judgments.jsonl"


def _mean(values: List[float]) -> Optional[float]:
    return mean_or_none(values, 4)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _nugget_weights(importances: List[Optional[str]]) -> List[float]:
    """Auto-ARGUE's weight of each nugget: vital 2, okay 1; all 1 without
    importance labels."""
    weights = [_IMPORTANCE_WEIGHTS.get(imp, 0.0) for imp in importances]
    return weights if any(weights) else [1.0] * len(importances)


def _retry_once(fn: Callable, *args):
    """``fn(*args)``, called once more when its last return value (the
    error) is set."""
    out = fn(*args)
    return fn(*args) if out[-1] else out


def criteria_hash(criteria: List[Dict]) -> str:
    """Hash of a criteria list (texts and kinds), the key a score is reused by."""
    payload = [[c.get("text", ""), c.get("kind", "closed")] for c in criteria]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:16]


def build_criteria_evaluator(
    dataset: str,
    gold: Dict[str, List[GoldUnit]],
    judge_model: str,
    max_workers: int = 16,
    mode: Optional[str] = None,
    use_llm: bool = True,
) -> Optional["CriteriaEvaluator"]:
    """The evaluator of *dataset*'s criteria gold (or of *mode*, e.g.
    ``"info"`` for a reference list) with *judge_model* as the matchers'
    LLM; None for a dataset without gold and no *mode*.  The entity matcher
    only compares names, so it runs without reasoning; the nugget matcher
    runs as Auto-ARGUE's judge (``gold.nugget_ask``); the information
    matcher and the request check of a report request reason.  Without
    *use_llm* only the string passes run."""
    mode = mode or DATASET_SPECS[dataset].criteria_gold
    if mode is None:
        return None
    judge = None
    if use_llm:
        from reasoner_component.factory import create_generator, disable_native_thinking

        judge = create_generator(judge_model, backend="api")
        if mode == "entities":
            judge = disable_native_thinking(judge)
    return CriteriaEvaluator(gold, mode, llm_client=judge, max_workers=max_workers, judge_model=judge_model,
                             request_coverage=DATASET_SPECS[dataset].query_shape == MULTI_ASPECT)


class CriteriaEvaluator:
    """Scores criteria lists against gold units.

    Args:
        gold: ``{query_id: [GoldUnit, ...]}``.
        mode: ``"entities"``, ``"nuggets"`` or ``"info"`` (see the module
            docstring).
        llm_client: Object with ``complete(messages, **kwargs) -> str`` for
            the matchers (the nugget matcher builds its own client from
            *judge_model*); None keeps only the string passes (exact names
            for ``entities``, exact texts for ``info``; no match for
            ``nuggets``).
        max_workers: Queries scored in parallel.
        judge_model: Name of the model behind *llm_client*, saved with every
            record (None without one).
        request_coverage: Also check each list against its request
            (``nuggets`` mode with an *llm_client*; report requests).
    """

    def __init__(
        self,
        gold: Dict[str, List[GoldUnit]],
        mode: str,
        llm_client: Any = None,
        max_workers: int = 16,
        judge_model: Optional[str] = None,
        request_coverage: bool = False,
    ) -> None:
        if mode not in GOLD_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {GOLD_MODES}")
        self.gold = gold
        self.mode = mode
        self.method = METHODS[mode]
        with_llm = llm_client is not None
        self.judge_model = judge_model if with_llm else None
        self._entity_matcher = LLMEntityMatcher(llm_client) if mode == "entities" and with_llm else None
        self._info_matcher = LLMInfoMatcher(llm_client) if mode == "info" else None
        self._nugget_matcher = (NuggetAskMatcher(judge_model)
                                if mode == "nuggets" and with_llm and judge_model else None)
        self._request_judge = (RequestCoverageJudge(llm_client)
                               if request_coverage and mode == "nuggets" and with_llm else None)
        self._max_workers = max_workers

    def use_judgment_cache(self, path: Union[str, Path]) -> None:
        """Keep the nugget judgments in *path* (``nuggets`` mode; else no-op)."""
        if self._nugget_matcher is not None:
            self._nugget_matcher.use_cache_file(path)

    # ------------------------------------------------------------------
    # One query
    # ------------------------------------------------------------------

    def _entities(self, units: List[GoldUnit], criteria: List[Dict]) -> Dict[str, Any]:
        # Members only: not the set criterion (open) nor the rest criterion.
        members = [c for c in criteria if c.get("kind", "closed") == "closed"]
        names, texts = [u.text for u in units], [c["text"] for c in members]
        matches = {i: (j, "string") for i, j in match_entities(names, texts).items()}
        errors = []
        left_e = [i for i in range(len(names)) if i not in matches]
        used = {j for j, _ in matches.values()}
        left_t = [j for j in range(len(texts)) if j not in used]
        if self._entity_matcher is not None and left_e and left_t:
            extra, error = _retry_once(self._entity_matcher.match,
                                       [names[i] for i in left_e], [texts[j] for j in left_t])
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
        scores, qualifiers, error = _retry_once(self._info_matcher.match, query, gold, texts)
        best_gold = [max(row) if row else 0.0 for row in scores]
        best_crit = [max((scores[i][j] for i in range(len(gold))), default=0.0) for j in range(len(texts))]

        def share(flags: List[bool]) -> Optional[float]:
            return round(sum(flags) / len(flags), 4) if flags else None

        return {
            "recall_strict": share([s >= 1.0 for s in best_gold]),
            "recall_lenient": share([s >= 0.5 for s in best_gold]),
            "precision_strict": share([s >= 1.0 for s in best_crit]),
            "precision_lenient": share([s >= 0.5 for s in best_crit]),
            "matches": [
                {"gold": u.text, "qualifiers": qualifiers[i], "score": best_gold[i],
                 "criteria": [criteria[j]["id"] for j in range(len(texts)) if scores[i][j] == best_gold[i] > 0]}
                for i, u in enumerate(units)
            ],
            "errors": [f"info_matcher: {error}"] if error else [],
        }

    def _nuggets(self, units: List[GoldUnit], criteria: List[Dict]) -> Dict[str, Any]:
        texts = [c["text"] for c in criteria]
        matched = [[set() for _ in texts] for _ in units]
        stats, errors = {"calls": 0, "malformed": 0, "cached": 0}, []
        if self._nugget_matcher is not None and units and texts:
            # A second attempt asks only what the first left unjudged (the
            # matcher caches every judgment).
            for _ in range(2):
                try:
                    matched, stats = self._nugget_matcher.match(units, texts)
                    errors = []
                    break
                except Exception as e:
                    logger.warning("NuggetAskMatcher: judge failed", exc_info=True)
                    errors = [f"nugget_matcher: {e}"]

        weights = _nugget_weights([u.importance for u in units])
        covered = [nugget_covered(set().union(*matched[i]), set(u.answers), u.aggregator)
                   for i, u in enumerate(units)]
        return {
            "nugget_coverage": round(sum(covered) / max(len(units), 1), 4),
            "nugget_coverage_weighted": round(sum(w for w, c in zip(weights, covered) if c) / max(sum(weights), 1), 4),
            "nuggets": len(units),
            "correct_nuggets": sum(covered),
            "num_unmatched_criteria": sum(not any(matched[i][j] for i in range(len(units)))
                                          for j in range(len(texts))),
            "matches": [
                {"gold": u.text, "importance": u.importance, "aggregator": u.aggregator,
                 "covered": covered[i], "matched_answers": sorted(set().union(*matched[i])),
                 "criteria": [criteria[j]["id"] for j in range(len(texts)) if matched[i][j]]}
                for i, u in enumerate(units)
            ],
            "judge_calls": stats["calls"],
            "judge_cached": stats["cached"],
            "judge_malformed": stats["malformed"],
            "errors": errors,
        }

    def evaluate_query(self, query_id: str, query: str, criteria: List[Dict]) -> Dict[str, Any]:
        units = self.gold.get(query_id, [])
        record: Dict[str, Any] = {
            "query_id": query_id,
            "num_gold": len(units),
            "num_criteria": len(criteria),
            "num_open": sum(c.get("kind") in MULTI_SOURCE_KINDS for c in criteria),
        }
        if self.mode == "entities":
            record.update(self._entities(units, criteria))
        elif self.mode == "nuggets":
            record.update(self._nuggets(units, criteria))
        else:
            record.update(self._info(query, units, criteria))
        if self._request_judge is not None:
            checked, error = _retry_once(self._request_judge.check, query, criteria)
            record["request_coverage"] = checked or None
            if error:
                record["errors"].append(f"request_coverage: {error}")
        record["method"] = self.method
        record["judge_model"] = self.judge_model
        record["criteria_hash"] = criteria_hash(criteria)
        return record

    def score(self, query_id: str, query: str, criteria: List[Dict]) -> Optional[Dict[str, Any]]:
        """:meth:`evaluate_query` for a query with gold units, else None
        (the estimator's per-query hook)."""
        if not self.gold.get(query_id):
            return None
        return self.evaluate_query(query_id, query, criteria)

    def reusable(self, record: Optional[Dict], criteria: List[Dict]) -> bool:
        """Whether *record* scored this criteria list with this method and
        judge, without errors."""
        return bool(record) and not record.get("errors") \
            and record.get("criteria_hash") == criteria_hash(criteria) \
            and record.get("method") == self.method \
            and record.get("judge_model") == self.judge_model

    def format_record(self, record: Dict[str, Any]) -> str:
        """One log line for a query's record."""
        def f(value):
            return _fmt(value, 2)
        head = f"criteria eval ({self.mode}, {record['num_criteria']} criteria, {record['num_gold']} gold)"
        if self.mode == "entities":
            line = f"{head}: recall {f(record['recall'])}, precision {f(record['precision'])}"
        elif self.mode == "nuggets":
            line = (f"{head}: nugget coverage {f(record['nugget_coverage'])}"
                    f" (weighted {f(record['nugget_coverage_weighted'])})")
        else:
            line = (f"{head}: recall lenient {f(record['recall_lenient'])} / strict {f(record['recall_strict'])},"
                    f" precision lenient {f(record['precision_lenient'])} / strict {f(record['precision_strict'])}")
        checked = record.get("request_coverage")
        if checked:
            line += (f"; request constraints {f(checked['constraint_recall'])},"
                     f" exclusions {f(checked['exclusion_recall'])}, violations {checked['num_violations']}")
        if record.get("errors"):
            line += f" [judge errors: {'; '.join(record['errors'])}]"
        return line

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
        to the summary.  Returns ``{"summary", "per_query"}``.
        """
        qids = [q for q in queries if q in self.gold and q in criteria]
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            per_query = list(tqdm(
                pool.map(lambda q: self.evaluate_query(q, queries[q], criteria[q]), qids),
                total=len(qids), desc=f"criteria eval ({self.mode})",
            ))
        return {"summary": self.summarize(per_query, infos), "per_query": per_query}

    def evaluate_run(self, query_ids, run_dir: Optional[Union[str, Path]],
                     results: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """Score the criteria of a run's queries (*query_ids* with gold units).

        The criteria come from the meta line of ``uncertainty/{qid}.jsonl``
        (or ``result["uncertainty_meta"]`` without a run directory).  A score
        is reused from the meta line's ``criteria_eval`` when
        :meth:`reusable`; the rest are judged, and their records written
        into the meta line.  Returns ``{"summary", "per_query"}``, plus
        ``num_judged`` (the queries judged now); ``{}`` when no query has
        criteria.
        """
        metas = load_uncertainty_meta(run_dir) if run_dir else {
            q: r["uncertainty_meta"] for q, r in (results or {}).items() if r.get("uncertainty_meta")
        }
        qids = [q for q in sorted(query_ids) if q in metas and self.gold.get(q)]
        if not qids:
            return {}

        records: Dict[str, Dict] = {}
        todo = []
        for q in qids:
            cached = metas[q].get("criteria_eval")
            if self.reusable(cached, metas[q].get("criteria") or []):
                records[q] = cached
            else:
                todo.append(q)
        if todo:
            print(f"  Criteria eval: judging {len(todo)} quer{'y' if len(todo) == 1 else 'ies'}"
                  f" ({len(records)} reused) with {self.judge_model}")
            if run_dir:
                self.use_judgment_cache(Path(run_dir) / CRITERIA_JUDGMENTS_FILE)
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                fresh = list(pool.map(
                    lambda q: self.evaluate_query(q, metas[q].get("question", ""), metas[q].get("criteria") or []),
                    todo))
            records.update(zip(todo, fresh))
            if run_dir:
                for q in todo:
                    update_uncertainty_meta(run_dir, q, {"criteria_eval": records[q]})

        per_query = [records[q] for q in qids]
        infos = {q: metas[q].get("criteria_info") or {} for q in qids}
        return {"summary": self.summarize(per_query, infos), "per_query": per_query, "num_judged": len(todo)}

    def summarize(self, per_query: List[Dict], infos: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """The ``criteria`` group of ``summary.json``, laid out as the
        report's: ``method``, ``judge_model``, the counts, ``metrics`` and
        ``stats``.  *infos* (``{query_id: criteria_info}``) adds the
        extraction errors."""
        # A query whose judge failed has partial scores: left out of the means.
        scored = [r for r in per_query if not r.get("errors")]
        metrics: Dict[str, Any] = {m: _mean([r[m] for r in scored if r.get(m) is not None])
                                   for m in MODE_METRICS[self.mode]}
        stats: Dict[str, Any] = {"avg_num_criteria": _mean([r["num_criteria"] for r in per_query])}
        if self.mode == "entities":
            stats["avg_num_open"] = _mean([r["num_open"] for r in per_query])
        stats["avg_num_gold"] = _mean([r["num_gold"] for r in per_query])

        checked = [r["request_coverage"] for r in scored if r.get("request_coverage")]
        if checked:
            metrics["constraint_recall"] = _mean([c["constraint_recall"] for c in checked
                                                  if c.get("constraint_recall") is not None])
            metrics["exclusion_recall"] = _mean([c["exclusion_recall"] for c in checked
                                                 if c.get("exclusion_recall") is not None])
            metrics["violations_per_query"] = _mean([c["num_violations"] for c in checked])
            stats["avg_num_constraints"] = _mean([c["num_constraints"] for c in checked])
        if self.mode == "nuggets":
            stats["avg_unmatched_criteria"] = _mean([r["num_unmatched_criteria"] for r in scored])
            stats["num_judge_malformed"] = sum(r.get("judge_malformed", 0) for r in per_query)
        if infos:
            stats["num_extraction_errors"] = sum(
                bool((infos.get(r["query_id"]) or {}).get("errors")) for r in per_query
            )
        return {
            "method": self.method,
            "judge_model": self.judge_model,
            "num_evaluated": len(scored),
            "num_empty_criteria": sum(r["num_criteria"] == 0 for r in per_query),
            "num_judge_failures": len(per_query) - len(scored),
            "metrics": metrics,
            "stats": stats,
        }

    def summary_lines(self, summary: Dict[str, Any]) -> List[str]:
        """The terminal lines of the run summary for a :meth:`summarize`
        result (with the report comparison of :func:`compare_with_report`)."""
        c, m, s = summary, summary["metrics"], summary["stats"]
        lines = [f"Criteria ({c['method']}, {c['judge_model']}): {c['num_evaluated']} lists"
                 f" (empty {c['num_empty_criteria']}, judge failures {c['num_judge_failures']}),"
                 f" {_fmt(s.get('avg_num_criteria'), 1)} criteria vs {_fmt(s.get('avg_num_gold'), 1)} gold"]
        if self.mode == "entities":
            lines.append(f"  recall {_fmt(m.get('recall'))} | precision {_fmt(m.get('precision'))}")
        elif self.mode == "nuggets":
            lines.append(f"  coverage {_fmt(m.get('nugget_coverage'))}"
                         f" (weighted {_fmt(m.get('nugget_coverage_weighted'))})"
                         f" | unmatched criteria/query {_fmt(s.get('avg_unmatched_criteria'), 2)}")
            if "constraint_recall" in m:
                lines.append(f"  request constraints {_fmt(m.get('constraint_recall'))}"
                             f" | exclusions {_fmt(m.get('exclusion_recall'))}"
                             f" | violations/query {_fmt(m.get('violations_per_query'), 2)}")
            if "answered_if_asked" in m:
                lines.append(f"  report answers {_fmt(m.get('answered_if_asked'))} of the asked nuggets,"
                             f" {_fmt(m.get('answered_if_not_asked'))} of the others")
        else:
            lines.append(f"  recall strict {_fmt(m.get('recall_strict'))} / lenient {_fmt(m.get('recall_lenient'))}"
                         f" | precision strict {_fmt(m.get('precision_strict'))}"
                         f" / lenient {_fmt(m.get('precision_lenient'))}")
        return lines


# ----------------------------------------------------------------------
# Criteria against the report, nugget by nugget
# ----------------------------------------------------------------------

def compare_with_report(criteria_metrics: Dict[str, Any], report_records: List[Dict]) -> None:
    """Cross each nugget's criteria label (``nuggets`` mode: covered by the
    criteria) with its report label (Auto-ARGUE: answered by the report).

    Uses the queries scored by both, without judge errors.  Per query:
    ``answered_if_asked`` / ``answered_if_not_asked``, the report's coverage
    of the nuggets the criteria asked for / did not ask for (None without
    such nuggets).  With the two coverages, they fix the whole asked x
    answered table.  Their macro-averages go into the ``metrics`` of
    *criteria_metrics*' summary (:meth:`CriteriaEvaluator.evaluate_run`'s
    result); nothing is added when no query is scored by both.
    """
    reports = {str(r["query_id"]): r for r in report_records}
    rows: List[Dict[str, Optional[float]]] = []
    for rec in criteria_metrics["per_query"]:
        qid = str(rec["query_id"])
        if rec.get("errors") or "matches" not in rec or qid not in reports:
            continue
        answered = set(reports[qid].get("answered_nuggets") or [])
        labels = [(bool(m["covered"]), m["gold"] in answered) for m in rec["matches"]]
        row = {}
        for name, asked in (("answered_if_asked", True), ("answered_if_not_asked", False)):
            group = [lab[1] for lab in labels if lab[0] == asked]
            row[name] = round(sum(group) / len(group), 4) if group else None
        rows.append(row)
    if rows:
        criteria_metrics["summary"]["metrics"].update(
            {k: _mean([r[k] for r in rows if r[k] is not None]) for k in rows[0]})
