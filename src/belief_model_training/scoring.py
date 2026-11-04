"""Scores of a belief model against the gold of its episodes.

One set of rows and measures, used to assess the teacher on the phase 1
episodes (``assess``, on its replayed calls), to reward the student's
chains in phase 3 (``rewards``) and to validate them.  Each belief update
call gives one row (``call_row``) with three parts, and each chain one
final row (``final_row``):

    format   the gate f_k (``format_ok``): the reply was not cut off,
             parses, names known criteria and new passages only, gives
             each nugget a valid status and evidence labels it knows, and
             updates each criterion once
    obs      valid observation (r_obs): the observations the gold decides,
             as an F1 = 2 TP / (2 TP + FP + FN) over documents and the
             agent's turn together; per gold criterion with gold answers
             (``nugget_gold.gold_criterion``):
               document  a passage among the criterion's gold documents
                         whose text as the call showed it (cut for a
                         short-answer query) has a gold answer should give
                         a "supports" observation with a gold value: one
                         does (TP, per passage and criterion) or none does
                         (FN); on a gold document, one with another value
                         that supports, or that contradicts a gold value,
                         is FP
               turn      a turn that states the gold answer
                         (``Episode.stated_at``) should give a "supports"
                         think observation with it: TP or FN
             FP also: an observation that did not enter the belief (no
             verified span, value not in the turn, a document observation
             on a computed criterion) and a value that is another
             criterion's gold answer (misassigned).  Every other
             observation (another passage, another value) is neutral
    cons     valid conversion against the call's own observations (r_cons):
             the share of checks passed, per rewritten criterion (each
             criterion with a new observation; status and exclusive on
             the nuggets the reply wrote, not those carried over):
               complete     it is in the reply, every value of its current
                            belief and of its new observations keeps a
                            nugget, no unknown or repeated value
                            (``BeliefState.apply`` issues)
               status       per nugget, its status fits its document
                            evidence (``status_consistent``)
               exclusive    a one-answer criterion has at most one nugget
                            at "almost certain" or above
               observed     a belief written for a criterion without a new
                            observation is a failed check
    cal      calibration of the rewritten nuggets (r_cal): per rewritten
             criterion with gold answers (not the aggregate, whose nuggets
             are always "unknown"), the mean 1 - (p - g)^2 over the
             nuggets the reply wrote (p the status probability, g the value is a gold
             answer), then the mean over the criteria
    final    the certainty score s_T of the chain's final belief
             (``belief_model_component.certainty``) against the
             correctness y of the agent's final answer: r_final =
             1 - (s_T - y)^2; the measures also report the other
             aggregators of the criterion scores (information only)

A part with nothing to score is None (neutral).  ``measures`` pools the
rows: observation precision / recall / F1 (documents and turns apart),
the misassigned rate, the violation rate of each consistency check, the
Brier score, ECE and reliability (gold rate per status) of the nuggets,
the mean of each reward part, and s_T against y: AUROC, Kendall's tau-b,
Brier score and ECE (10 equal-width bins).
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata

from belief_model_component.certainty import AGGREGATORS, CERTAINTY_AGGREGATOR, criterion_scores
from belief_model_component.criteria import (
    ISSUE_DROPPED_VALUE, ISSUE_DUPLICATE_UPDATE, ISSUE_NO_NEW_OBSERVATION, ISSUE_NO_STATUS, ISSUE_NOT_IN_REPLY,
    ISSUE_OVER_CAP, ISSUE_REPEATED_VALUE, ISSUE_UNKNOWN_CRITERION, ISSUE_UNKNOWN_EVIDENCE, ISSUE_UNKNOWN_VALUE,
    ISSUE_UNOBSERVED_VALUE, NOTE_NOT_NEW_PASSAGE, NOTE_REPEATED, NOTE_UNKNOWN_CRITERION, same_value,
)
from belief_model_component.types import (
    AGGREGATE, BELIEF_STATUSES, CONTRADICTS, NO_DOC_KINDS, STATUS_LEVEL, SUPPORTS, exclusive, status_probability,
)

from .episodes import Episode, UpdaterCall
from .nugget_gold import NuggetGold, gold_criterion, matches

# A reply with one of these fails the gate.
GATE_NOTES = (NOTE_UNKNOWN_CRITERION, NOTE_NOT_NEW_PASSAGE)
GATE_ISSUES = (ISSUE_UNKNOWN_CRITERION, ISSUE_DUPLICATE_UPDATE, ISSUE_NO_STATUS, ISSUE_UNKNOWN_EVIDENCE)
# Issues that fail a rewritten criterion's "complete" check.
INCOMPLETE_ISSUES = (ISSUE_NOT_IN_REPLY, ISSUE_UNKNOWN_VALUE, ISSUE_REPEATED_VALUE, ISSUE_OVER_CAP,
                     ISSUE_UNOBSERVED_VALUE, ISSUE_DROPPED_VALUE)
CONS_CHECKS = ("complete", "status", "exclusive", "observed")
HIGH_LEVEL = STATUS_LEVEL["almost certain"]


def _observations(outcome: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (outcome.get("think_observations") or []) + (outcome.get("doc_observations") or [])


def format_ok(outcome: Dict[str, Any], finish_reason: Optional[str]) -> bool:
    """The gate f_k of one call's *outcome* (``BeliefTracker.submit``)."""
    if finish_reason != "stop" or not outcome.get("parsed"):
        return False
    if any(o.get("note") in GATE_NOTES for o in _observations(outcome)):
        return False
    return not any(set(r.get("issues") or []) & set(GATE_ISSUES) for r in outcome.get("nugget_updates") or [])


def status_consistent(nugget: Dict[str, Any], others: List[str]) -> bool:
    """The nugget's status fits its document evidence: "doubtful" needs a
    document that contradicts its value and none that supports it,
    "unlikely" a document and none that supports, "possible" a document,
    "probable" a supporting document, "almost certain" and "certain" a
    supporting document and none that contradicts; "unknown" always fits
    (the cautious choice, rule 13).  An entry is on the nugget's value
    unless its value is another nugget's (*others*): values that name one
    answer in other words share a nugget."""
    value, status = nugget["value"], nugget["status"]
    docs = [e for e in nugget.get("evidence") or [] if str(e.get("label", "")).startswith("d")]
    own = [e for e in docs if same_value(e["value"], value) or not any(same_value(e["value"], o) for o in others)]
    support = any(e["stance"] == SUPPORTS for e in own)
    contra = any(e["stance"] == CONTRADICTS for e in own)
    if status == "doubtful":
        return contra and not support
    if status == "unlikely":
        return bool(docs) and not support
    if status == "unknown":
        return True
    if status == "possible":
        return bool(docs)
    if status == "probable":
        return support
    return support and not contra


def _kinds(episode: Episode) -> Dict[str, str]:
    return {c["id"]: c["kind"] for c in episode.gold_criteria or []}


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def _doc_counts(episode: Episode, call: UpdaterCall, outcome: Dict[str, Any], gold: NuggetGold) -> Dict[str, int]:
    # A gold document is expected to give an observation only when its text
    # as the call showed it (cut for a short-answer query) has the gold answer.
    shown = call.shown_text()
    expected = set()
    for c in episode.gold_criteria or []:
        if gold_criterion(episode, c["id"]) is not None and c["kind"] not in NO_DOC_KINDS:
            expected.update((c["id"], d) for d in set(shown) & set(c["doc_ids"]) if matches(episode, c, shown[d]))
    covered, fp, mis = set(), 0, 0
    for o in outcome.get("doc_observations") or []:
        if o.get("note") in GATE_NOTES:
            continue
        cid, doc = o["id"], o.get("doc_id")
        if not o.get("grounded") or o["note"] not in ("", NOTE_REPEATED):
            fp += 1
            continue
        c = gold_criterion(episode, cid)
        if c is None:
            continue
        on_gold_doc = doc in set(c["doc_ids"])
        if gold.correct(episode, cid, o["value"]):
            if o["stance"] == SUPPORTS and on_gold_doc:
                covered.add((cid, doc))
            elif o["stance"] == CONTRADICTS and on_gold_doc:
                fp += 1
        elif gold.misassigned(episode, cid, o["value"]):
            fp += 1
            mis += 1
        elif on_gold_doc and o["stance"] == SUPPORTS:
            fp += 1
    return {"tp": len(covered), "fp": fp, "fn": len(expected - covered), "misassigned": mis}


def _think_counts(episode: Episode, call: UpdaterCall, outcome: Dict[str, Any], gold: NuggetGold) -> Dict[str, int]:
    expected = {cid for cid in episode.stated_at(call.t) if gold_criterion(episode, cid) is not None}
    covered, fp, mis = set(), 0, 0
    for o in outcome.get("think_observations") or []:
        if o.get("note") in GATE_NOTES:
            continue
        cid = o["id"]
        if not o.get("grounded"):
            fp += 1
            continue
        if gold_criterion(episode, cid) is None:
            continue
        if gold.correct(episode, cid, o["value"]):
            if o["stance"] == SUPPORTS:
                covered.add(cid)
        elif gold.misassigned(episode, cid, o["value"]):
            fp += 1
            mis += 1
    return {"tp": len(covered), "fp": fp, "fn": len(expected - covered), "misassigned": mis}


def _cons_checks(episode: Episode, outcome: Dict[str, Any]) -> Dict[str, List[int]]:
    """Per consistency check, ``[passed, total]`` over the call's records."""
    kinds = _kinds(episode)
    checks: Dict[str, List[int]] = {name: [0, 0] for name in CONS_CHECKS}

    def check(name: str, ok: bool) -> None:
        checks[name][0] += int(ok)
        checks[name][1] += 1

    for r in outcome.get("nugget_updates") or []:
        issues = set(r.get("issues") or [])
        if ISSUE_NO_NEW_OBSERVATION in issues:
            check("observed", False)
            continue
        if r.get("after") is None:
            continue
        check("complete", not issues & set(INCOMPLETE_ISSUES))
        if r.get("written") is None:
            continue
        after = r["after"]
        written = _written(r)
        for n in written:
            check("status", status_consistent(n, [m["value"] for m in after if m is not n]))
        if exclusive(kinds.get(r["id"], "")):
            check("exclusive", sum(STATUS_LEVEL[n["status"]] >= HIGH_LEVEL for n in written) <= 1)
    return checks


def _written(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The nuggets of a nugget update record that the reply wrote: its
    ``after`` without the nuggets carried over unchanged (``carried``)."""
    carried = set(record.get("carried") or [])
    return [n for n in record.get("after") or [] if n["value"] not in carried]


def _cal_items(episode: Episode, outcome: Dict[str, Any], gold: NuggetGold) -> List[List[Tuple[str, bool]]]:
    """Per rewritten criterion with gold answers (not the aggregate), the
    nuggets the reply wrote as ``(status, is a gold answer)``."""
    out = []
    for r in outcome.get("nugget_updates") or []:
        if r.get("written") is None or not _written(r):
            continue
        c = gold_criterion(episode, r["id"])
        if c is None or c["kind"] == AGGREGATE:
            continue
        out.append([(n["status"], bool(gold.correct(episode, r["id"], n["value"]))) for n in _written(r)])
    return out


def call_row(episode: Episode, call: UpdaterCall, outcome: Dict[str, Any], finish_reason: Optional[str],
             gold: NuggetGold) -> Dict[str, Any]:
    """The row of one call: ``format``, the observation counts (``doc``,
    ``think``), the consistency checks (``cons``), the calibration items
    (``cal``) and the call's reward parts ``r_obs``, ``r_cons``, ``r_cal``
    (None: nothing to score).  A call that fails the gate scores 0 on
    each part."""
    row: Dict[str, Any] = {"final_call": call.final, "format": format_ok(outcome, finish_reason)}
    if not row["format"]:
        return {**row, "r_obs": 0.0, "r_cons": 0.0, "r_cal": 0.0}
    doc = _doc_counts(episode, call, outcome, gold)
    think = _think_counts(episode, call, outcome, gold)
    cons = _cons_checks(episode, outcome)
    cal = _cal_items(episode, outcome, gold)
    passed, total = sum(v[0] for v in cons.values()), sum(v[1] for v in cons.values())
    briers = [sum(1.0 - (status_probability(s) - g) ** 2 for s, g in items) / len(items) for items in cal]
    return {
        **row, "doc": doc, "think": think, "cons": cons, "cal": cal,
        "r_obs": _f1(doc["tp"] + think["tp"], doc["fp"] + think["fp"], doc["fn"] + think["fn"]),
        "r_cons": passed / total if total else None,
        "r_cal": sum(briers) / len(briers) if briers else None,
    }


def final_row(episode: Episode, nuggets: List[List[Dict[str, Any]]], answer_label: Optional[str],
              final_ok: bool) -> Dict[str, Any]:
    """The final row of a chain with final belief *nuggets* (per gold
    criterion), the answer turn's label (None: no final call) and whether
    the final call passed the gate: s_T per aggregator, y, and r_final
    (None when y is unknown; 0 when the final call failed the gate)."""
    kinds = [c["kind"] for c in episode.gold_criteria or []]
    scores = criterion_scores(nuggets, kinds, episode.query_shape, answer_label)
    s = {name: (agg(scores) if scores else None) for name, agg in AGGREGATORS.items()}
    y = episode.correct
    r = None
    if y is not None and s[CERTAINTY_AGGREGATOR] is not None:
        r = (1.0 - (s[CERTAINTY_AGGREGATOR] - float(y)) ** 2) if final_ok else 0.0
    return {"s": s, "y": y, "r_final": r}


def chain_values(episode: Episode, outcomes: Iterable[Dict[str, Any]],
                 nuggets: Optional[List[List[Dict[str, Any]]]] = None) -> List[Tuple[Episode, str, str]]:
    """``(episode, criterion id, value)`` of every observation and nugget of
    *outcomes* and of the final *nuggets*, for ``NuggetGold.resolve``."""
    out = []
    for o in outcomes:
        out += [(episode, x["id"], x["value"]) for x in _observations(o)]
        out += [(episode, r["id"], n["value"]) for r in o.get("nugget_updates") or [] for n in r.get("after") or []]
    ids = [c["id"] for c in episode.gold_criteria or []]
    for cid, ns in zip(ids, nuggets or []):
        out += [(episode, cid, n["value"]) for n in ns]
    return out


# ---------------------------------------------------------------------------
# Pooled measures
# ---------------------------------------------------------------------------

def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC of *s* for y=1 over y=0, a tie counting 1/2; NaN
    without both classes."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float(rankdata(s)[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _ece(y: np.ndarray, s: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins of *s*."""
    if len(y) == 0:
        return math.nan
    idx = np.minimum((s * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - s[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(y))


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else math.nan
    r = tp / (tp + fn) if tp + fn else math.nan
    return {"precision": p, "recall": r, "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan}


def _mean(values: Iterable[Optional[float]]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else math.nan


def _final_measures(prefix: str, s: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_auroc": _auroc(y, s),
        f"{prefix}_tau_b": float(kendalltau(s, y)[0]) if len(y) > 1 and len(set(y)) > 1 else math.nan,
        f"{prefix}_brier": float(((s - y) ** 2).mean()) if len(y) else math.nan,
        f"{prefix}_ece": _ece(y, s),
    }


def measures(rows: List[Dict[str, Any]], finals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The pooled measures over call *rows* (``call_row``) and final rows
    (``final_row``)."""
    scored = [r for r in rows if r["format"]]
    out: Dict[str, Any] = {"calls": len(rows), "format_rate": len(scored) / len(rows) if rows else math.nan}
    for side in ("doc", "think"):
        c = Counter()
        for r in scored:
            c.update(r[side])
        out.update({f"obs_{side}_{k}": v for k, v in _prf(c["tp"], c["fp"], c["fn"]).items()})
        out[f"obs_{side}_misassigned"] = c["misassigned"]
    cons: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in scored:
        for name, (passed, total) in r["cons"].items():
            cons[name][0] += passed
            cons[name][1] += total
    out.update({f"cons_{name}_violation_rate": (1 - cons[name][0] / cons[name][1]) if cons[name][1] else math.nan
                for name in CONS_CHECKS})
    pairs = [(status_probability(s), float(g), s) for r in scored for items in r["cal"] for s, g in items]
    p = np.array([x[0] for x in pairs])
    g = np.array([x[1] for x in pairs])
    out.update({
        "nuggets_scored": len(pairs),
        "nugget_brier": float(((p - g) ** 2).mean()) if pairs else math.nan,
        "nugget_ece": _ece(g, p),
        "nugget_auroc": _auroc(g, p),
    })
    for status in BELIEF_STATUSES:
        own = [x[1] for x in pairs if x[2] == status]
        out[f"gold_rate_{status.replace(' ', '_')}"] = sum(own) / len(own) if own else math.nan
        out[f"count_{status.replace(' ', '_')}"] = len(own)
    out.update({name: _mean(r[name] for r in rows) for name in ("r_obs", "r_cons", "r_cal")})
    judged = [f for f in finals if f["y"] is not None and f["s"][CERTAINTY_AGGREGATOR] is not None]
    y = np.array([float(f["y"]) for f in judged])
    out["final_judged"] = len(judged)
    out["r_final"] = _mean(f["r_final"] for f in finals)
    for name in AGGREGATORS:
        prefix = "certainty" if name == CERTAINTY_AGGREGATOR else f"certainty_{name}"
        out.update(_final_measures(prefix, np.array([f["s"][name] for f in judged]), y))
    return out


def assess(episodes: List[Episode], replays: List[Tuple[Episode, list, Any]], gold: NuggetGold) -> Dict[str, Any]:
    """The teacher's measures over its replayed calls: *replays* holds per
    episode ``(episode, calls, tracker)`` (``episodes.replay_teacher``,
    the calls before the first mismatch); the final row only for an
    episode replayed in full.  Resolves the gold of every value first."""
    items = []
    for ep, calls, tracker in replays:
        items += chain_values(ep, [c.outcome for c in calls], tracker.signal.nuggets())
    gold.resolve(items)
    rows, finals = [], []
    for ep, calls, tracker in replays:
        rows += [call_row(ep, c.call, c.outcome, "stop", gold) for c in calls]
        if ep.replay_error is None:
            final_ok = not calls or not calls[-1].call.final or format_ok(calls[-1].outcome, "stop")
            finals.append(final_row(ep, tracker.signal.nuggets(), tracker.signal.answer_label, final_ok))
    return {"episodes": len(episodes), **measures(rows, finals)}
