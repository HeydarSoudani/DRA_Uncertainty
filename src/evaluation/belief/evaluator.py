"""Belief evaluation: what the belief model does with the gold evidence, and
whether its final belief is correct and calibrated.

Run on the queries whose criteria are gold criteria
(``evaluation.gold.criteria``: TRQA's members, InfoSeek's hops).  Read from
the saved belief files (``belief/{qid}.jsonl``), so it scores any run with
the belief model on (monitor or inform), without calling the belief model.

**Scored criteria.** Every criterion with a gold answer: a closed one (a
member's value or a hop, with gold documents) and an aggregate (gold answer
only: no document states a computed value; scored at the end only).

**Gold event** (criterion k, document d, step t): d is one of k's gold
documents, new to the belief at step t (not in an earlier step whose update
call parsed: the update shows only such documents), and its text as the
update read it (the run's ``max_passage_tokens`` cut) states k's gold
answer (string match, ``gold.states``, else the judge when the run has a
judge model).  When the update call of step t failed, the document is shown
again when retrieved again: the event is its first occurrence with a parsed
call, else a ``call_failed`` event at its first occurrence.

**Control event** (criterion k, step t): no gold event of k at t, and the
update observed on k a new document that is not one of k's gold documents
and whose text does not state the gold answer.

The **gold candidate's probability** at a step: the status probability
(``core.types.status_probability``) of k's gold candidate (several: the
highest), "unknown" when the gold answer is not a candidate.  An event's
change is that probability after the step minus before; an event whose gold
candidate is already "certain" cannot rise and has no change.

A candidate is a gold answer when it matches one (``gold.states``), else
when the judge says so (``gold.AnswerJudge``, the run's ``judge_model``,
cached in ``belief_eval/judge_cache.jsonl``; no judge model: the match
only).  Only belief files of the current ``SCHEMA_VERSION`` are scored
(``load_belief_files``; an older file counts as no belief file).

Per criterion (a value is None where it does not apply):

    gold_change     the change at its first gold event (closed criteria)
    control_change  the mean change over its control events (closed
                    criteria)
    recall          the share of its gold documents the agent saw (closed
                    criteria)
    reached         at the end, its selected candidate is a gold answer
    reached_seen    ``reached``, for a closed criterion with a gold document
                    seen (the belief's own share: no seen gold document is
                    a retrieval failure)
    brier           (p - y)^2 of its final status probability p against
                    y = ``reached``

Each row of ``metrics.md`` is one of them twice: the mean over the queries
of each query's mean over its criteria (bootstrap CI over queries, n
queries), and ``(c)``: the mean over every criterion (n criteria).  Below the
table, the final status of every criterion with a Brier item, per status:
how many, how many reached (``calibration``).

In the files, for diagnosis: every gold and control event, with a gold
event's funnel stage (``call_failed``, ``not_assigned``,
``no_gold_candidate``, ``not_selected``, ``correct_lost``, ``correct_kept``),
in ``belief_eval/events.csv``; per query in ``per_query.csv`` the means of
its criteria and ``belief_correct`` (every scored closed criterion
reached).
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from belief_model_component import (AGGREGATE, MULTI_ASPECT, STATUS_LEVEL, UNKNOWN, exclusive,
                                    status_probability)
from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import DATASET_SPECS
from utils.text_utils import cut_chars, cut_passage

from ..common import read_jsonl
from ..gold.criteria import load_gold_rows, match_row
from ..metrics import bootstrap_ci, mean
from .gold import CANDIDATE, SPANS, AnswerJudge, states
from .writer import SCHEMA_VERSION

logger = logging.getLogger(__name__)

#: Folder of the belief evaluation's files in the run directory.
BELIEF_EVAL_DIR = "belief_eval"
#: The per-criterion values, in row order: (row name, key).
MEASURES = (("gold event: gold candidate change", "gold_change"),
            ("control event: gold candidate change", "control_change"),
            ("criterion recall", "recall"),
            ("reached answer", "reached"),
            ("reached answer, gold seen", "reached_seen"),
            ("status Brier", "brier"))
#: Columns of the calibration table below the main table.
CALIBRATION_COLUMNS = ("status", "p", "criteria", "reached", "share reached")
_CERTAIN = "certain"
EVENT_FIELDS = ("query_id", "criterion", "kind", "type", "first", "step", "doc_id", "assigned", "p_before",
                "p_after", "change", "correct_after", "correct_end", "stage")


#: Text fields no evaluation reads (dropped by ``load_belief_files``, which
#: keeps every query of a run in memory).
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
# document at 3000 characters and read a report document whole; from
# schema 38 to the token cut the shared cut was ``max_passage_chars``
# characters, ending in "...".
LEGACY_PASSAGE_CHARS = 3000


def read_run_config(run_dir: Path) -> Dict[str, Any]:
    """The run's ``run_config.json`` (``{}`` when missing)."""
    path = Path(run_dir) / "run_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _recorded(config: Dict[str, Any], key: str) -> Tuple[bool, Optional[int]]:
    """``(recorded, value)`` of *key* in the belief section of a
    ``run_config.json`` *config*, else in its agent section (the shared cut,
    the same value)."""
    for section in ("belief_model", "agent"):
        values = config.get(section) or {}
        if key in values:
            return True, values[key]
    return False, None


def passage_tokens(config: Dict[str, Any]) -> Optional[int]:
    """Tokens of one document the belief update of a run read (or, a run
    without the belief model, will read in training): its
    ``max_passage_tokens``; None: whole documents.  A run made before the
    token cut (it recorded characters, or nothing) cannot be replayed with
    the same text: ValueError."""
    recorded, value = _recorded(config, "max_passage_tokens")
    if not recorded:
        raise ValueError("the run was made before the shared token cut (no max_passage_tokens in its "
                         "run_config.json); rerun it")
    return value


def passage_cut(config: Dict[str, Any], query_shape: str) -> Callable[[str], str]:
    """The cut of one document as the belief update of a run read it, from
    its ``run_config.json`` *config*: ``max_passage_tokens`` (tokens), else
    an older run's ``max_passage_chars`` (characters), else the cut before
    either was recorded."""
    recorded, tokens = _recorded(config, "max_passage_tokens")
    if recorded:
        return lambda text: cut_passage(text, tokens)
    recorded, chars = _recorded(config, "max_passage_chars")
    if not recorded:
        chars = None if query_shape == MULTI_ASPECT else LEGACY_PASSAGE_CHARS
    return lambda text: cut_chars(text, chars)


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

    def final_beliefs(self) -> Dict[str, Dict[str, Any]]:
        """The belief per criterion at the end (the final line, else the
        last step)."""
        return _by_id((self.final or {}).get("beliefs")) or (
            _by_id(self.steps[-1].get("beliefs")) if self.steps else {})

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
    """The belief measures of a run (module docstring).

    Args:
        dataset: Dataset name; only datasets with per-criterion gold are
            scored (``evaluation.gold.criteria``).
        split: File-level split id (TRQA's gold bank).
        corpus_path: The corpus the run retrieved from (document texts).
        judge_model: Model of the candidate and document judge; None or "":
            match only.
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
                if not gold_c.get("answers"):
                    continue
                gold[run_c["id"]] = {"id": run_c["id"], "text": run_c["text"], "kind": run_c["kind"],
                                     "answers": list(gold_c["answers"]), "doc_ids": set(gold_c.get("doc_ids") or [])}
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
        cut = passage_cut(read_run_config(run_dir), DATASET_SPECS[self.dataset].query_shape)
        contents = CorpusLookup(self.corpus_path).contents(ids)
        return {i: cut(str(t).strip()) for i, t in contents.items()}

    @staticmethod
    def _resolve(judge: Optional[AnswerJudge], verdict: Dict, todo: List[Tuple[Any, Tuple[str, Dict, str]]]) -> None:
        """Fill *verdict* with the judge's answer on each ``(key, item)`` of
        *todo* (no judge: False)."""
        if not todo:
            return
        if judge is None:
            verdict.update({key: False for key, _ in todo})
            return
        answers = judge.resolve([item for _, item in todo])
        for key, item in todo:
            verdict[key] = answers.get(judge.key(*item), False)

    def _gold_names(self, queries: List[_Query], judge: Optional[AnswerJudge]) -> Dict[Tuple[str, str, str], bool]:
        """``(qid, criterion id, name) -> True`` when the name is a gold
        answer of a scored criterion: the match, else the judge."""
        verdict: Dict[Tuple[str, str, str], bool] = {}
        todo: List[Tuple[Tuple[str, str, str], Tuple[str, Dict, str]]] = []
        for q in queries:
            pairs = [(cid, name, q.gold[cid]["text"], q.gold[cid]["answers"], q.names)
                     for cid, name in set(q.candidate_names())]
            for cid, name, text, answers, others in pairs:
                key = (q.qid, cid, name)
                if states(name, answers, others):
                    verdict[key] = True
                elif name.strip():
                    todo.append((key, (CANDIDATE, {"text": text, "answers": answers}, name)))
                else:
                    verdict[key] = False
        self._resolve(judge, verdict, todo)
        return verdict

    def _gold_docs(self, queries: List[_Query], shown: Dict[str, str],
                   judge: Optional[AnswerJudge]) -> Dict[Tuple[str, str, str], bool]:
        """``(qid, criterion id, doc id) -> True`` when a seen gold document
        of the criterion states its gold answer in the text the update read:
        the match, else the judge."""
        verdict: Dict[Tuple[str, str, str], bool] = {}
        todo: List[Tuple[Tuple[str, str, str], Tuple[str, Dict, str]]] = []
        for q in queries:
            seen = {d["doc_id"] for s in q.steps for d in s.get("docs") or [] if d.get("doc_id")}
            for cid, c in q.gold.items():
                for doc_id in c["doc_ids"] & seen:
                    key, text = (q.qid, cid, doc_id), shown.get(doc_id, "")
                    if states(text, c["answers"], q.names):
                        verdict[key] = True
                    elif text:
                        todo.append((key, (SPANS, {"text": c["text"], "answers": c["answers"]}, text)))
                    else:
                        verdict[key] = False
        self._resolve(judge, verdict, todo)
        return verdict

    # ------------------------------------------------------------------
    # One query
    # ------------------------------------------------------------------

    def _query(self, q: _Query, shown: Dict[str, str], is_gold: Dict[Tuple[str, str, str], bool],
               doc_states: Dict[Tuple[str, str, str], bool]) -> Tuple[Dict[str, Dict], List[Dict]]:
        """The per-criterion values of one query (``{criterion id: values}``,
        module docstring) and its events."""

        def gold_name(cid: str, name: str) -> bool:
            return is_gold.get((q.qid, cid, name or ""), False)

        def gold_status(belief: Optional[Dict], cid: str) -> str:
            statuses = [c.get("status") for c in (belief or {}).get("candidates") or []
                        if gold_name(cid, c.get("name")) and c.get("status") in STATUS_LEVEL]
            return max(statuses, key=STATUS_LEVEL.get) if statuses else UNKNOWN

        def has_gold_candidate(belief: Optional[Dict], cid: str) -> bool:
            return any(gold_name(cid, c.get("name")) for c in (belief or {}).get("candidates") or [])

        def selected_correct(belief: Optional[Dict], cid: str) -> bool:
            selected = list((belief or {}).get("selected") or [])
            if exclusive(q.gold[cid]["kind"]):
                selected = selected[:1]
            return any(gold_name(cid, name) for name in selected)

        def change(before: Optional[Dict], after: Optional[Dict], cid: str) -> Dict[str, Any]:
            s_before, s_after = gold_status(before, cid), gold_status(after, cid)
            p_before, p_after = status_probability(s_before), status_probability(s_after)
            return {"p_before": p_before, "p_after": p_after,
                    "change": None if s_before == _CERTAIN else p_after - p_before}

        final = q.final_beliefs()
        prev: Dict[str, Dict] = {}
        events: List[Dict] = []
        done: Set[Tuple[str, str]] = set()
        failed_first: Dict[Tuple[str, str], int] = {}
        seen: Dict[str, Set[str]] = {cid: set() for cid in q.gold}
        judged: Set[str] = set()   # documents of a step whose update call parsed

        for step in q.steps:
            t = step.get("iteration")
            docs = [d for d in step.get("docs") or [] if d.get("doc_id")]
            # A call failed when it was made and no reply parsed; a parse
            # error of an attempt that a retry recovered leaves an error
            # line on a step whose call succeeded.
            parsed = step.get("observations") is not None
            failed = bool(step.get("called")) and not parsed
            fresh = {d["doc_id"] for d in docs if d["doc_id"] not in judged}
            after = _by_id(step.get("beliefs")) or prev
            observed = {(o["id"], o.get("doc_id")) for o in step.get("observations") or [] if o.get("doc_id")}
            for cid, c in q.gold.items():
                if not c["doc_ids"]:
                    continue
                seen[cid] |= {d["doc_id"] for d in docs if d["doc_id"] in c["doc_ids"]}
                gold_new = [d["doc_id"] for d in docs if d["doc_id"] in fresh and d["doc_id"] in c["doc_ids"]
                            and (cid, d["doc_id"]) not in done and doc_states.get((q.qid, cid, d["doc_id"]), False)]
                for doc_id in gold_new:
                    if failed:
                        failed_first.setdefault((cid, doc_id), t)
                        continue
                    done.add((cid, doc_id))
                    failed_first.pop((cid, doc_id), None)
                    assigned = (cid, doc_id) in observed
                    correct_after = selected_correct(after.get(cid), cid)
                    correct_end = selected_correct(final.get(cid), cid)
                    if not assigned:
                        stage = "not_assigned"
                    elif not has_gold_candidate(after.get(cid), cid):
                        stage = "no_gold_candidate"
                    elif not correct_after:
                        stage = "not_selected"
                    else:
                        stage = "correct_kept" if correct_end else "correct_lost"
                    events.append({"query_id": q.qid, "criterion": cid, "kind": c["kind"], "type": "gold",
                                   "step": t, "doc_id": doc_id, "assigned": assigned,
                                   **change(prev.get(cid), after.get(cid), cid),
                                   "correct_after": correct_after, "correct_end": correct_end, "stage": stage})
                if failed or gold_new:
                    continue
                controls = [d["doc_id"] for d in docs if d["doc_id"] in fresh and (cid, d["doc_id"]) in observed
                            and d["doc_id"] not in c["doc_ids"]
                            and not states(shown.get(d["doc_id"], ""), c["answers"], q.names)]
                if controls:
                    events.append({"query_id": q.qid, "criterion": cid, "kind": c["kind"], "type": "control",
                                   "step": t, "doc_id": " ".join(controls), "assigned": True,
                                   **change(prev.get(cid), after.get(cid), cid)})
            if parsed:
                judged |= fresh
            prev = after

        for (cid, doc_id), t in failed_first.items():
            events.append({"query_id": q.qid, "criterion": cid, "kind": q.gold[cid]["kind"], "type": "gold",
                           "step": t, "doc_id": doc_id, "assigned": False, "change": None, "correct_after": False,
                           "correct_end": selected_correct(final.get(cid), cid), "stage": "call_failed"})

        # The first gold event of each criterion (step order).
        firsts: Dict[str, Dict] = {}
        for e in sorted((e for e in events if e["type"] == "gold"), key=lambda e: e["step"]):
            firsts.setdefault(e["criterion"], e)
        for e in events:
            e["first"] = e["type"] == "gold" and firsts.get(e["criterion"]) is e

        values: Dict[str, Dict[str, Any]] = {}
        for cid, c in q.gold.items():
            has_docs = bool(c["doc_ids"])
            reached = selected_correct(final.get(cid), cid)
            controls = [e["change"] for e in events
                        if e["type"] == "control" and e["criterion"] == cid and e["change"] is not None]
            p = status_probability((final.get(cid) or {}).get("status") or UNKNOWN)
            values[cid] = {
                "kind": c["kind"],
                "gold_change": (firsts.get(cid) or {}).get("change"),
                "control_change": mean(controls) if controls else None,
                "recall": len(seen[cid]) / len(c["doc_ids"]) if has_docs else None,
                "reached": float(reached),
                "reached_seen": float(reached) if has_docs and seen[cid] else None,
                "brier": (p - float(reached)) ** 2,
                "status": (final.get(cid) or {}).get("status") or UNKNOWN,
            }
        return values, events

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, run_dir, files: Dict[str, BeliefFile], query_ids: Iterable[str]) -> Dict[str, Any]:
        """The measures over the run's queries with scored gold criteria,
        from their belief *files* (``load_belief_files``; a query of
        *query_ids* without one counts as ``no belief file``):
        ``{per_query, rows, calibration, events}`` (``rows``: ``(metric,
        value, ci_low, ci_high, n)``; ``calibration``: the rows of the
        calibration table, ``CALIBRATION_COLUMNS``); ``{}`` when no query
        has one."""
        run_dir = Path(run_dir)
        counts: Counter = Counter()
        queries = self._load(files, query_ids, counts)
        if counts:
            logger.info("belief evaluation: queries left out %s", dict(counts))
        if not queries:
            return {}
        shown = self._shown_texts(queries, run_dir)
        judge = AnswerJudge(self.judge_model, run_dir / BELIEF_EVAL_DIR / "judge_cache.jsonl") \
            if self.judge_model else None
        is_gold = self._gold_names(queries, judge)
        doc_states = self._gold_docs(queries, shown, judge)

        events: List[Dict] = []
        per_query: Dict[str, Dict[str, Any]] = {}
        groups: List[List[Dict[str, Any]]] = []   # per query: the values of its criteria
        for q in queries:
            values, q_events = self._query(q, shown, is_gold, doc_states)
            events += q_events
            groups.append(list(values.values()))
            rec: Dict[str, Any] = {"belief_criteria": len(values)}
            for _, key in MEASURES:
                rec[f"belief_{key}"] = _mean_of(values.values(), key)
            members = [v for v in values.values() if v["kind"] != AGGREGATE]
            rec["belief_correct"] = int(all(v["reached"] for v in members)) if members else None
            per_query[q.qid] = rec

        self._write_events(run_dir, events)
        return {"per_query": per_query, "rows": self._rows(groups), "calibration": self._calibration(groups),
                "events": len(events)}

    @staticmethod
    def _rows(groups: List[List[Dict[str, Any]]]) -> List[Tuple[str, Any, Any, Any, Any]]:
        rows: List[Tuple[str, Any, Any, Any, Any]] = []
        for name, key in MEASURES:
            scored = [g for g in groups if _mean_of(g, key) is not None]

            def per_query(gs, key=key):
                return mean([_mean_of(g, key) for g in gs])
            pooled = [v[key] for g in groups for v in g if v[key] is not None]
            rows.append((name, per_query(scored), *bootstrap_ci(scored, per_query), len(scored)))
            rows.append((f"{name} (c)", mean(pooled), None, None, len(pooled)))
        return rows

    @staticmethod
    def _calibration(groups: List[List[Dict[str, Any]]]) -> List[Tuple[str, float, int, int, Optional[float]]]:
        """Per status, highest first: ``(status, p, criteria, reached, share
        reached)`` over the criteria with a Brier item."""
        items = [v for g in groups for v in g if v["brier"] is not None]
        out = []
        for status in sorted(STATUS_LEVEL, key=STATUS_LEVEL.get, reverse=True):
            with_status = [v for v in items if v["status"] == status]
            hits = int(sum(v["reached"] for v in with_status))
            out.append((status, status_probability(status), len(with_status), hits,
                        hits / len(with_status) if with_status else None))
        return out

    @staticmethod
    def _write_events(run_dir: Path, events: List[Dict]) -> None:
        path = run_dir / BELIEF_EVAL_DIR / "events.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for e in events:
                writer.writerow({k: int(v) if isinstance(v := e.get(k), bool) else v for k in EVENT_FIELDS})


def _mean_of(values: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    """The mean of the defined *key* of *values*, None when there is none."""
    vals = [v[key] for v in values if v[key] is not None]
    return mean(vals) if vals else None
