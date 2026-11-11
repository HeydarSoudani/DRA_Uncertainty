"""Belief update evaluation: what the belief model does with a gold document.

Run on the queries whose criteria are gold criteria with a gold answer and
gold documents per criterion (``evaluation.gold.criteria``: TRQA's members,
InfoSeek's hops; a criterion without both, the rest and the aggregate
criteria are left out).  Read from the saved belief files
(``belief/{qid}.jsonl``), so it scores any run with the belief model on
(monitor or inform), without calling the belief model.

**Gold event** (criterion k, document d, step t): d is one of k's gold
documents, new to the belief at step t (not in an earlier step whose update
call parsed: the update shows only such documents), and its
text as the update read it (the run's ``max_passage_chars`` cut) states k's
gold answer (string match, ``gold.states``).  A gold document whose answer
is past the cut is not an event (counted in the reachability).  When the
update call of step t failed, the document is shown again when retrieved
again: the event is its first occurrence with a parsed call, else a
``call_failed`` event at its first occurrence.

Per gold event:

- ``assigned``: the update wrote an observation of d on k (verified spans);
- ``delta``: the level of k's gold candidate after the step minus before
  (``core.types.STATUS_LEVEL``, -1 unlikely .. 4 certain; a gold answer that is
  not a candidate is at 0, unknown; several gold candidates: the highest);
  ``rises``: delta > 0, or the gold candidate was already "certain" and
  stays so (``at_ceiling``);
- ``correct_after`` / ``correct_end``: k's selected candidate (one-answer
  criterion), or one of them (several answers), is a gold answer, after the
  step / in the final belief;
- ``stage`` (the funnel, one per event): ``call_failed``,
  ``not_assigned``, ``no_gold_candidate`` (the gold answer is not among
  k's candidates after the step), ``not_selected``, ``correct_lost``
  (selected after the step, not at the end), ``correct_kept``.

**Control event** (criterion k, step t): no gold event of k at t, and the
update observed on k a new document that is not one of k's gold documents
and whose text does not state the gold answer.  Its ``delta`` is the
largest change of a non-gold candidate of k (absent = 0).  The AUROC of
``delta`` between the first gold event of each criterion and the control
events tells whether the belief rises because of the evidence; a gold
event already at "certain" (``at_ceiling``) cannot rise and is left out
of it.

A candidate is a gold answer when it matches one (``gold.states``), else
when the judge says so (``gold.AnswerJudge``, the run's ``judge_model``,
cached in ``belief_eval/judge_cache.jsonl``; no judge model: the match
only).

Only belief files of the current ``SCHEMA_VERSION`` are scored
(``load_belief_files``; an older file counts as no belief file).

Measures (``evaluate``): per query the rates over its first gold event of
each criterion; the headline is their mean over the queries with an event
(macro, bootstrap CI over queries), the pooled rate next to it (micro).  The
status calibration pools every step of every scored criterion: the
criterion's status probability (``core.types.status_probability``) against y,
its selected candidate(s) correct (Brier, ECE, AUROC).  The reachability:
per query, the share of scored criteria whose gold document was seen, and
whose gold event happened.  Every event is written to
``belief_eval/events.csv``.
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from belief_model_component import (AGGREGATE, MULTI_ASPECT, REST, STATUS_LEVEL, UNKNOWN, exclusive,
                                    status_probability)
from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import DATASET_SPECS
from utils.text_utils import cut_passage

from ..common import read_jsonl
from ..gold.criteria import load_gold_rows, match_row
from ..metrics import auroc, bootstrap_ci, brier, ece, mean
from .gold import CANDIDATE, AnswerJudge, states
from .writer import SCHEMA_VERSION

logger = logging.getLogger(__name__)

#: Folder of the belief evaluation's files in the run directory.
BELIEF_EVAL_DIR = "belief_eval"
#: Funnel stages of a gold event, in order.
STAGES = ("call_failed", "not_assigned", "no_gold_candidate", "not_selected", "correct_lost", "correct_kept")
#: Per-event rates (over the first gold event of each criterion).
EVENT_RATES = ("assigned", "rises", "correct_after", "correct_end")
#: Criterion kinds with no documents or no gold answers of their own.
_UNSCORED_KINDS = (REST, AGGREGATE)
_CERTAIN = STATUS_LEVEL["certain"]
EVENT_FIELDS = ("query_id", "criterion", "kind", "type", "first", "step", "doc_id", "assigned", "level_before",
                "level_after", "delta", "at_ceiling", "rises", "correct_after", "correct_end", "stage")


#: Text fields of a step or final line no evaluation reads (dropped by
#: ``load_belief_files``, which keeps every query of a run in memory).
_TEXT_KEYS = ("agent_turn", "reply", "references", "intermediate_answer_reasoning")

BeliefFile = Tuple[Optional[Dict], List[Dict], Optional[Dict]]


def read_belief_file(path: Path, light: bool = False) -> BeliefFile:
    """``(meta, steps, final)`` of a belief file (``(None, [], None)`` when
    missing); *light* drops the text fields no evaluation reads."""
    lines = read_jsonl(path)
    if light:
        lines = [{k: v for k, v in r.items() if k not in _TEXT_KEYS} for r in lines]
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    final = next((r for r in lines if r.get("record") == "final"), None)
    return meta, [r for r in lines if r.get("record") == "step"], final


# Before the shared cut (schema 38) the belief update cut a short-answer
# document at 3000 characters and read a report document whole.
LEGACY_PASSAGE_CHARS = 3000


def read_run_config(run_dir: Path) -> Dict[str, Any]:
    """The run's ``run_config.json`` (``{}`` when missing)."""
    path = Path(run_dir) / "run_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def passage_chars(config: Dict[str, Any], query_shape: str) -> Optional[int]:
    """Characters of one document the belief update of a run read (or, a
    run without the belief model, will read in training), from its
    ``run_config.json`` *config*: the belief section's
    ``max_passage_chars``, else the agent's (the shared cut, the same
    value), else the cut of the runs before it was recorded; None: whole
    documents."""
    for section in ("belief_model", "agent"):
        values = config.get(section) or {}
        if "max_passage_chars" in values:
            return values["max_passage_chars"]
    return None if query_shape == MULTI_ASPECT else LEGACY_PASSAGE_CHARS


def load_belief_files(run_dir: Path, query_ids: Iterable[str]) -> Dict[str, BeliefFile]:
    """``{qid: (meta, steps, final)}`` of the run's belief files of the
    current ``SCHEMA_VERSION`` (light, see ``read_belief_file``), read once
    for every evaluation; a query without one, or with an older one, is
    left out (``skipped`` counts these by reason)."""
    out: Dict[str, BeliefFile] = {}
    old: Counter = Counter()
    for qid in query_ids:
        meta, steps, final = read_belief_file(Path(run_dir) / "belief" / f"{qid}.jsonl", light=True)
        if meta is None:
            continue
        if meta.get("schema_version") != SCHEMA_VERSION:
            old[meta.get("schema_version")] += 1
            continue
        out[qid] = (meta, steps, final)
    if old:
        logger.warning("%d belief file(s) of another schema (%s), not %d: not evaluated",
                       sum(old.values()), dict(old), SCHEMA_VERSION)
    return out


def _by_id(beliefs: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {b["id"]: b for b in beliefs or []}


class _Query:
    """One query's belief file with its scored gold criteria."""

    def __init__(self, qid: str, meta: Dict, steps: List[Dict], final: Optional[Dict], gold: Dict[str, Dict],
                 names: List[str]) -> None:
        self.qid, self.meta, self.steps, self.final = qid, meta, steps, final
        self.gold = gold            # scored criterion id -> {id, text, kind, answers, doc_ids}
        self.names = names          # entity names of the gold structure (masked in the match)

    def needed_doc_ids(self) -> Set[str]:
        """The documents whose text the events read: the gold documents of
        a scored criterion and the documents observed on one."""
        gold = set().union(*(c["doc_ids"] for c in self.gold.values()))
        ids = {d["doc_id"] for s in self.steps for d in s.get("docs") or [] if d.get("doc_id") in gold}
        ids |= {o["doc_id"] for s in self.steps for o in s.get("observations") or []
                if o.get("doc_id") and o.get("id") in self.gold}
        return ids

    def candidate_names(self) -> Iterable[Tuple[str, str]]:
        """``(criterion id, name)`` of every candidate of a scored criterion
        in any saved belief."""
        for line in self.steps + ([self.final] if self.final else []):
            for b in line.get("beliefs") or []:
                if b["id"] in self.gold:
                    for c in b.get("candidates") or []:
                        yield b["id"], c.get("name") or ""


class BeliefUpdateEvaluator:
    """The belief update measures of a run (module docstring).

    Args:
        dataset: Dataset name; only datasets with per-criterion gold are
            scored (``evaluation.gold.criteria``).
        split: File-level split id (TRQA's gold bank).
        corpus_path: The corpus the run retrieved from (document texts).
        judge_model: Model of the candidate judge; None or "": match only.
    """

    def __init__(self, dataset: str, split: Optional[str], corpus_path: Optional[str],
                 judge_model: Optional[str] = None) -> None:
        self.dataset = dataset
        self.split = split
        self.corpus_path = corpus_path
        self.judge_model = judge_model or None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, files: Dict[str, BeliefFile], query_ids: Iterable[str], counts: Counter) -> List[_Query]:
        rows = load_gold_rows(self.dataset, self.split)
        queries = []
        for qid in sorted(query_ids):
            if qid not in files:
                counts["no_belief_file"] += 1
                continue
            meta, steps, final = files[qid]
            row = match_row(rows.get(qid, []), meta.get("criteria") or [])
            if row is None:
                counts["no_gold_criteria"] += 1
                continue
            gold = {}
            for run_c, gold_c in zip(meta["criteria"], row["criteria"]):
                if run_c["kind"] in _UNSCORED_KINDS or not gold_c.get("answers") or not gold_c.get("doc_ids"):
                    continue
                gold[run_c["id"]] = {"id": run_c["id"], "text": run_c["text"], "kind": run_c["kind"],
                                     "answers": list(gold_c["answers"]), "doc_ids": set(gold_c["doc_ids"])}
            if not gold:
                counts["no_scored_criterion"] += 1
                continue
            queries.append(_Query(qid, meta, steps, final, gold, list(row.get("entity_names") or [])))
        return queries

    def _shown_texts(self, queries: List[_Query], run_dir: Path) -> Dict[str, str]:
        """Each needed document's text as the update read it."""
        ids = set().union(*(q.needed_doc_ids() for q in queries)) if queries else set()
        if not ids or not self.corpus_path:
            return {}
        max_chars = passage_chars(read_run_config(run_dir), DATASET_SPECS[self.dataset].query_shape)
        contents = CorpusLookup(self.corpus_path).contents(ids)
        return {i: cut_passage(str(t).strip(), max_chars) for i, t in contents.items()}

    def _gold_names(self, queries: List[_Query], run_dir: Path) -> Dict[Tuple[str, str, str], bool]:
        """``(qid, criterion id, name) -> is a gold answer`` for every
        candidate name: the match, else the judge."""
        verdict: Dict[Tuple[str, str, str], bool] = {}
        todo: List[Tuple[Tuple[str, str, str], Dict[str, Any]]] = []
        for q in queries:
            for cid, name in set(q.candidate_names()):
                c = q.gold[cid]
                if states(name, c["answers"], q.names):
                    verdict[(q.qid, cid, name)] = True
                elif self.judge_model and name.strip():
                    todo.append(((q.qid, cid, name), c))
                else:
                    verdict[(q.qid, cid, name)] = False
        if todo:
            judge = AnswerJudge(self.judge_model, run_dir / BELIEF_EVAL_DIR / "judge_cache.jsonl")
            items = [(CANDIDATE, {"text": c["text"], "answers": c["answers"]}, key[2]) for key, c in todo]
            answers = judge.resolve(items)
            for (key, _), item in zip(todo, items):
                verdict[key] = answers.get(judge.key(*item), False)
        return verdict

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _query_events(self, q: _Query, shown: Dict[str, str],
                      is_gold: Dict[Tuple[str, str, str], bool]) -> Tuple[List[Dict], List[Tuple[float, int]], Dict]:
        """The events of one query, its calibration items ``(p, y)`` and its
        reachability counts."""

        def gold_level(belief: Optional[Dict], cid: str) -> int:
            levels = [STATUS_LEVEL.get(c.get("status"), 0) for c in (belief or {}).get("candidates") or []
                      if is_gold.get((q.qid, cid, c.get("name") or ""), False)]
            return max(levels) if levels else 0

        def has_gold_candidate(belief: Optional[Dict], cid: str) -> bool:
            return any(is_gold.get((q.qid, cid, c.get("name") or ""), False)
                       for c in (belief or {}).get("candidates") or [])

        def selected_correct(belief: Optional[Dict], cid: str) -> bool:
            selected = list((belief or {}).get("selected") or [])
            if exclusive(q.gold[cid]["kind"]):
                selected = selected[:1]
            return any(is_gold.get((q.qid, cid, name), False) for name in selected)

        def other_levels(belief: Optional[Dict], cid: str) -> Dict[str, int]:
            return {c.get("name") or "": STATUS_LEVEL.get(c.get("status"), 0)
                    for c in (belief or {}).get("candidates") or []
                    if not is_gold.get((q.qid, cid, c.get("name") or ""), False)}

        final_beliefs = _by_id((q.final or {}).get("beliefs")) or (
            _by_id(q.steps[-1].get("beliefs")) if q.steps else {})
        prev: Dict[str, Dict] = {}
        events: List[Dict] = []
        items: List[Tuple[float, int]] = []
        done: Set[Tuple[str, str]] = set()
        failed_first: Dict[Tuple[str, str], int] = {}
        seen_gold: Set[str] = set()
        judged: Set[str] = set()   # documents of a step whose update call parsed

        for step in q.steps:
            t = step.get("iteration")
            docs = step.get("docs") or []
            parsed = step.get("observations") is not None
            failed = (bool(step.get("called")) and not parsed) or any(
                str(e).startswith("belief_update:") for e in step.get("errors") or [])
            fresh = {d["doc_id"] for d in docs if d.get("doc_id") and d["doc_id"] not in judged}
            after = _by_id(step.get("beliefs")) or prev
            observed = {(o["id"], o.get("doc_id")) for o in step.get("observations") or [] if o.get("doc_id")}
            for cid, c in q.gold.items():
                seen_gold.update(cid for d in docs if d["doc_id"] in c["doc_ids"])
                gold_new = [d["doc_id"] for d in docs if d["doc_id"] in fresh and d["doc_id"] in c["doc_ids"]
                            and (cid, d["doc_id"]) not in done
                            and states(shown.get(d["doc_id"], ""), c["answers"], q.names)]
                for doc_id in gold_new:
                    if failed:
                        failed_first.setdefault((cid, doc_id), t)
                        continue
                    done.add((cid, doc_id))
                    failed_first.pop((cid, doc_id), None)
                    before_l, after_l = gold_level(prev.get(cid), cid), gold_level(after.get(cid), cid)
                    assigned = (cid, doc_id) in observed
                    correct_after = selected_correct(after.get(cid), cid)
                    correct_end = selected_correct(final_beliefs.get(cid), cid)
                    if not assigned:
                        stage = "not_assigned"
                    elif not has_gold_candidate(after.get(cid), cid):
                        stage = "no_gold_candidate"
                    elif not correct_after:
                        stage = "not_selected"
                    else:
                        stage = "correct_kept" if correct_end else "correct_lost"
                    at_ceiling = before_l == _CERTAIN
                    events.append({
                        "query_id": q.qid, "criterion": cid, "kind": c["kind"], "type": "gold", "step": t,
                        "doc_id": doc_id, "assigned": assigned, "level_before": before_l, "level_after": after_l,
                        "delta": after_l - before_l, "at_ceiling": at_ceiling,
                        "rises": after_l > before_l or (at_ceiling and after_l == _CERTAIN),
                        "correct_after": correct_after, "correct_end": correct_end, "stage": stage,
                    })
                if failed or gold_new:
                    continue
                controls = [d["doc_id"] for d in docs if d["doc_id"] in fresh and (cid, d["doc_id"]) in observed
                            and d["doc_id"] not in c["doc_ids"]
                            and not states(shown.get(d["doc_id"], ""), c["answers"], q.names)]
                if controls:
                    before_o, after_o = other_levels(prev.get(cid), cid), other_levels(after.get(cid), cid)
                    deltas = [after_o.get(n, 0) - before_o.get(n, 0) for n in set(before_o) | set(after_o)]
                    delta = max(deltas) if deltas else 0
                    events.append({
                        "query_id": q.qid, "criterion": cid, "kind": c["kind"], "type": "control", "step": t,
                        "doc_id": " ".join(controls), "assigned": True, "delta": delta, "rises": delta > 0,
                    })
            for cid in q.gold:
                b = after.get(cid)
                items.append((status_probability((b or {}).get("status") or UNKNOWN),
                              int(selected_correct(b, cid))))
            if parsed:
                judged |= fresh
            prev = after

        for (cid, doc_id), t in failed_first.items():
            events.append({"query_id": q.qid, "criterion": cid, "kind": q.gold[cid]["kind"], "type": "gold",
                           "step": t, "doc_id": doc_id, "assigned": False, "rises": False, "correct_after": False,
                           "correct_end": selected_correct(final_beliefs.get(cid), cid), "stage": "call_failed"})

        # The first gold event of each criterion (step order).
        firsts: Dict[str, Dict] = {}
        for e in sorted((e for e in events if e["type"] == "gold"), key=lambda e: e["step"]):
            firsts.setdefault(e["criterion"], e)
        for e in events:
            e["first"] = e["type"] == "gold" and firsts.get(e["criterion"]) is e
        reach = {"criteria": len(q.gold), "gold_seen": len(seen_gold), "gold_event": len(firsts)}
        return events, items, reach

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, run_dir, files: Dict[str, BeliefFile], query_ids: Iterable[str]) -> Dict[str, Any]:
        """The measures over the run's queries with scored gold criteria,
        from their belief *files* (``load_belief_files``; a query of
        *query_ids* without one counts as ``no belief file``):
        ``{per_query, rows, counts, events}`` (``rows``: ``(metric, value,
        ci_low, ci_high, n)``); ``{}`` when no query has one."""
        run_dir = Path(run_dir)
        counts: Counter = Counter()
        queries = self._load(files, query_ids, counts)
        if not queries:
            if counts:
                logger.info("belief update evaluation: no query with scored gold criteria (%s)", dict(counts))
            return {}
        shown = self._shown_texts(queries, run_dir)
        is_gold = self._gold_names(queries, run_dir)

        events: List[Dict] = []
        per_query: Dict[str, Dict[str, Any]] = {}
        groups = []   # per query: (first gold events, control events, calibration items, reach)
        for q in queries:
            q_events, items, reach = self._query_events(q, shown, is_gold)
            events += q_events
            firsts = [e for e in q_events if e.get("first")]
            controls = [e for e in q_events if e["type"] == "control"]
            groups.append((firsts, controls, items, reach))
            rec: Dict[str, Any] = {
                "belief_criteria": reach["criteria"],
                "belief_gold_seen": reach["gold_seen"] / reach["criteria"],
                "belief_gold_event": reach["gold_event"] / reach["criteria"],
            }
            for name in EVENT_RATES:
                rec[f"belief_{name}"] = mean([float(e[name]) for e in firsts]) if firsts else None
            per_query[q.qid] = rec

        rows = self._rows(groups, counts)
        self._write_events(run_dir, events)
        return {"per_query": per_query, "rows": rows, "counts": dict(counts), "events": len(events)}

    @staticmethod
    def _rows(groups: List[Tuple], counts: Counter) -> List[Tuple[str, Any, Any, Any, Any]]:
        rows: List[Tuple[str, Any, Any, Any, Any]] = []
        with_events = [g for g in groups if g[0]]
        firsts = [e for g in groups for e in g[0]]
        rows.append(("queries scored", len(groups), None, None, None))
        rows.append(("criteria scored", sum(g[3]["criteria"] for g in groups), None, None, None))
        for name, key in (("gold doc seen", "gold_seen"), ("gold event", "gold_event")):
            def stat(gs, key=key):
                return mean([g[3][key] / g[3]["criteria"] for g in gs])
            rows.append((f"reach: {name}", stat(groups), *bootstrap_ci(groups, stat), len(groups)))
        for name in EVENT_RATES:
            def macro(gs, name=name):
                return mean([mean([float(e[name]) for e in g[0]]) for g in gs if g[0]])
            rows.append((name, macro(with_events), *bootstrap_ci(with_events, macro), len(with_events)))
            rows.append((f"{name} (micro)", mean([float(e[name]) for e in firsts]), None, None, len(firsts)))

        def rising(g):
            # First gold events that can rise: a parsed call, below "certain".
            return [e for e in g[0] if "delta" in e and not e["at_ceiling"]]

        def rise_auroc(gs):
            gold = [e["delta"] for g in gs for e in rising(g)]
            controls = [e["delta"] for g in gs for e in g[1]]
            return auroc(np.array([1] * len(gold) + [0] * len(controls)), np.array(gold + controls, dtype=float))

        n_controls = sum(len(g[1]) for g in groups)
        rows.append(("rise AUROC (gold vs control)", rise_auroc(groups), *bootstrap_ci(groups, rise_auroc),
                     sum(len(rising(g)) for g in groups) + n_controls))
        rows.append(("control rises", mean([float(e["rises"]) for g in groups for e in g[1]]), None, None,
                     n_controls))
        stages = Counter(e["stage"] for e in firsts)
        for stage in STAGES:
            rows.append((f"funnel: {stage}", stages[stage] / len(firsts) if firsts else None, None, None,
                         len(firsts)))

        def cal(fn):
            def stat(gs):
                items = [x for g in gs for x in g[2]]
                return fn(np.array([y for _, y in items]), np.array([p for p, _ in items]))
            return stat

        n_items = sum(len(g[2]) for g in groups)
        rows.append(("status AUROC", cal(auroc)(groups), *bootstrap_ci(groups, cal(auroc)), n_items))
        rows.append(("status Brier", cal(brier)(groups), None, None, n_items))
        rows.append(("status ECE", cal(ece)(groups), None, None, n_items))
        for reason, n in sorted(counts.items()):
            rows.append((f"skipped: {reason.replace('_', ' ')}", n, None, None, None))
        return rows

    @staticmethod
    def _write_events(run_dir: Path, events: List[Dict]) -> None:
        path = run_dir / BELIEF_EVAL_DIR / "events.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for e in events:
                writer.writerow({k: int(v) if isinstance(v := e.get(k), bool) else v for k in EVENT_FIELDS})
