"""Scores of a belief model against the gold of its episodes, per
criterion.

One set of scores and measures, used to assess the teacher on the phase 1
episodes (``assess``, on its replayed calls), to reward the student's
chains in phase 3 (``rewards``) and to validate them.  A belief chain (the
calls of one episode) is scored by ``chain_score``.  Each criterion with
gold answers (``nugget_gold.gold_criterion``: not the rest criterion) is
its own sequence of beliefs, one per step of the episode (a step
without a call, or whose call failed, carries the belief of the step
before; the answer turn is the last step), and is scored on its own.  A
hop of a multi-hop query is a several-answers criterion, so y is whether
its selected candidates include the gold one (``value_gold``; another
valid answer is not counted against it):

    format   the gate (``format_ok``), per belief update call: the reply
             was not cut off, parses, names known criteria and passages of
             the call only, and gives each candidate update one valid status
             (``core.issues.GATE_ISSUES``); the chain's share of update calls
             that pass it (the final scorer's call has its own gate, in
             ``final``)
    assign   assignment of the sources to the criterion, from the
             observations (the trace), as an F1 = 2 TP / (2 TP + FP + FN)
             over the chain's calls, documents and turns together:
               document  a passage of the call (new to the belief)
                         among the criterion's gold documents
                         whose text as the call showed it (cut for a
                         short-answer query) has a gold answer should give
                         an observation whose spans state it
                         (``NuggetGold.stated``): one does (TP, per
                         passage) or none does (FN)
               turn      a turn of the call that states the gold
                         answer (``Episode.stated_at``; a call reads
                         the turns of the steps before it that made
                         none) should give a claim whose span states
                         it: TP or FN
             FP: an observation the reply wrote that did not enter the
             belief for a reason of its own
             (``core.issues.REJECTED_OBSERVATION``: no verified span, no
             span of the turn, a claim without a candidate or a valid
             claim, a document on a computed criterion).
             Every other observation is neutral (spans that give
             information about a criterion may name other criteria's
             answers).  A call whose reply did not parse is not counted
             here (it is charged in ``format`` only); its passages are
             shown again by the next call and counted there
    cal      calibration of the status at every step: 1 - (p - y)^2, p the
             status probability and y whether the criterion is correctly
             covered by its selected candidate(s) (``value_gold``: a one-answer
             criterion, its selected candidate is a gold answer; a
             several-answers one, the share of its gold answers its selected
             candidates cover, each answer counted once); the mean over
             the steps
    final    s_T against the correctness y of the agent's final answer:
             r_final = 1 - (s_T - y)^2 (``scorer_reward``; 0 when the final
             call fails the gate).  On a short-answer query s_T is the
             final scorer's (``belief_model_component.update.final_score``, the
             probability of the final answer's status; its gate: not cut off,
             parses);
             on a report, the certainty of the final beliefs
             (``belief_model_component.core.certainty``, a fixed function of the
             statuses).  The measures report the scorer's s_T next to the
             fixed function on the final beliefs (with its other
             aggregators) and its form anchored on the scorer's answer's
             candidates
             (``core.certainty.answer_certainty``), baselines only

The chain's ``assign`` and ``cal`` are the means over its criteria.  Two
more per-criterion measures are reported, not rewarded: the recall of its
gold documents among its document observations (``doc_recall_obs``), over
the gold documents that should give an observation; and the share of
steps whose selected candidate is correct (``value_accuracy``).

A part with nothing to score is None (neutral).  ``measures`` averages the
chains' per-criterion measures (per criterion over its steps, then over
the criteria, then over the chains), pools the observation counts
(precision / recall / F1, documents and turns apart), the (status, y) items of every step (Brier score, ECE, AUROC on
the items with a binary y, gold rate per status), the issue codes (rate
per call) and the removals (how many, and the share that removed a gold
answer), and compares s_T with y: AUROC, Kendall's tau-b, Brier score
and ECE (10 equal-width bins): ``certainty_scorer_*`` the final
scorer's, ``certainty_*`` the fixed function's, ``certainty_answer_*``
its anchored form.
"""

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from belief_model_component import (
    AGGREGATORS, BELIEF_STATUSES, CERTAINTY_AGGREGATOR, GATE_ISSUES, NO_DOC_KINDS, REJECTED_OBSERVATION, REMOVED,
    UNKNOWN, answer_certainty, criterion_scores, exclusive, status_probability,
)
from evaluation.belief.gold import CANDIDATE, SPANS
from evaluation.metrics import auroc as _auroc, brier as _brier, ece as _ece, mean_or_none, tau_b

from .episodes import Episode, UpdaterCall
from .nugget_gold import NuggetGold, gold_criterion, matches, states

SIDES = ("doc", "think")


def format_ok(outcome: Dict[str, Any], finish_reason: Optional[str]) -> bool:
    """The gate of one call's *outcome* (``BeliefTracker.submit``)."""
    if finish_reason != "stop" or not outcome.get("parsed"):
        return False
    return not any(i.get("code") in GATE_ISSUES for i in outcome.get("issues") or [])


def scorer_reward(outcome: Dict[str, Any], finish_reason: Optional[str], y: Optional[bool]) -> Optional[float]:
    """r_final of one final scorer reply: 1 - (s_T - y)^2, 0 when the reply
    fails its gate (cut off, or no status parsed); None when y is
    unknown."""
    if y is None:
        return None
    if finish_reason != "stop" or not outcome.get("parsed"):
        return 0.0
    return 1.0 - (outcome["certainty"] - float(y)) ** 2


def _kinds(episode: Episode) -> Dict[str, str]:
    return {c["id"]: c["kind"] for c in episode.gold_criteria or []}


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _is_doc(label: Any) -> bool:
    return str(label or "").startswith("d")


def _open_answers(episode: Episode, cid: str, selected: List[str]) -> Tuple[set, List[str], List[str]]:
    """For a several-answers criterion: the gold answers its *selected*
    candidates state by string match, the answers none states (open), and
    the candidates that match no answer (left for the judge, which is
    asked about the open answers only, so an alias of a matched answer
    does not count twice)."""
    c = gold_criterion(episode, cid)
    found, unmatched = set(), []
    for v in selected:
        hit = [a for a in c["answers"] if states(v, [a], episode.gold_names)]
        if hit:
            found.update(hit)
        else:
            unmatched.append(v)
    return found, [a for a in c["answers"] if a not in found], unmatched


def value_gold(episode: Episode, cid: str, belief: Dict[str, Any], kind: str, gold: NuggetGold) -> float:
    """y of a criterion's belief: whether criterion *cid* is correctly
    covered by its selected candidate(s).  A one-answer criterion: its (first) selected candidate is a gold
    answer (0 without one); a several-answers criterion: the share of its
    gold answers its selected candidates cover: the answers they state by
    string match, plus one per other candidate the judge finds to be one
    of the answers still open (``_open_answers``), at most as many as are
    open."""
    by = list(belief.get("selected") or [])
    if exclusive(kind):
        return float(bool(by) and bool(gold.correct(episode, cid, by[0])))
    found, open_answers, unmatched = _open_answers(episode, cid, by)
    extra = sum(1 for v in unmatched if open_answers and gold.correct(episode, cid, v, open_answers))
    total = len(found) + len(open_answers)
    return (len(found) + min(extra, len(open_answers))) / total


def _expected_docs(episode: Episode, call: UpdaterCall, cid: str) -> set:
    """The gold documents of *cid* new in the call whose shown text has a
    gold answer: each should give an observation with it."""
    c = gold_criterion(episode, cid)
    if c is None or c["kind"] in NO_DOC_KINDS:
        return set()
    shown = call.shown_text()
    return {d for d in set(shown) & set(c["doc_ids"]) if matches(episode, c, shown[d])}


def _call_counts(episode: Episode, call: UpdaterCall, outcome: Dict[str, Any], cid: str,
                 gold: NuggetGold) -> Dict[str, Counter]:
    """The observation counts of criterion *cid* in one call, per side
    (``tp``, ``fp``, ``fn``), the gold documents it observed (TP,
    ``doc["hits"]``) and those it should have (``doc["expected"]``)."""
    out = {side: Counter() for side in SIDES}
    c = gold_criterion(episode, cid)
    gold_docs = set(c["doc_ids"])
    expected_docs = _expected_docs(episode, call, cid)
    # The turns of the call that state the gold answer (a call reads the
    # turns of the steps before it that made none).
    expected_turns = {f"t{step}" for t, step in call.turn_positions(len(episode.steps))
                      if cid in episode.stated_at(t)}
    docs_hit, turns_hit = set(), set()
    for o in outcome.get("observations") or []:
        if o["id"] != cid:
            continue
        side = "doc" if _is_doc(o["label"]) else "think"
        # An observation whose spans state the gold answer is a hit; any
        # other is not scored (its spans may give the criterion's other
        # parts, or another candidate).
        if gold.stated(episode, cid, spans_text(o)):
            if side == "think":
                turns_hit.add(o["label"])
            elif o.get("doc_id") in gold_docs:
                docs_hit.add(o.get("doc_id"))
    for i in outcome.get("issues") or []:
        if i.get("id") == cid and i.get("code") in REJECTED_OBSERVATION:
            out["doc" if _is_doc(i.get("label")) else "think"]["fp"] += 1
    out["doc"]["tp"] += len(docs_hit & expected_docs)
    out["doc"]["fn"] += len(expected_docs - docs_hit)
    out["think"]["tp"] += len(turns_hit & expected_turns)
    out["think"]["fn"] += len(expected_turns - turns_hit)
    out["doc"]["hits"] = docs_hit & expected_docs  # type: ignore[assignment]
    out["doc"]["expected"] = expected_docs  # type: ignore[assignment]
    return out


def _unknown(cid: str) -> Dict[str, Any]:
    """The belief of a criterion without candidates (b_0)."""
    return {"id": cid, "status": UNKNOWN, "selected": [], "candidates": []}


def spans_text(observation: Dict[str, Any]) -> str:
    """The spans of a saved observation as one text, for the gold."""
    return " ... ".join(observation.get("spans") or [])


def chain_score(episode: Episode, calls: List[UpdaterCall], outcomes: List[Dict[str, Any]],
                finish_reasons: List[Optional[str]], gold: NuggetGold, beliefs: List[Dict[str, Any]],
                complete: bool = True) -> Dict[str, Any]:
    """The score of one belief chain: its *calls* in order with their
    *outcomes* (``BeliefTracker.submit``) and finish reasons, and its final
    *beliefs* (per criterion).  *complete*: the calls cover the whole
    episode (a replay that stopped at a mismatch does not; its steps are
    scored up to the last call, and it has no final score).

    Returns ``{format, assign, cal, final, criteria, issues, calls,
    removals, removed_gold}``: ``criteria`` per scored criterion ``{kind,
    doc, think, assign_f1, doc_recall_obs, value_accuracy, cal, items}``
    (``items`` the (status, y) of every step), ``issues`` the count of
    each issue code, ``removals`` the candidates removed and
    ``removed_gold`` those of them that state a gold answer of their
    criterion (string match only: ``matches``)."""
    kinds = _kinds(episode)
    scored = [c["id"] for c in episode.gold_criteria or [] if gold_criterion(episode, c["id"]) is not None]
    gates = [format_ok(o, f) for o, f in zip(outcomes, finish_reasons)]
    update_gates = [g for g, c in zip(gates, calls) if not c.is_scorer]
    by_t = {c.t: i for i, c in enumerate(calls)}
    step_calls = [c.t for c in calls if c.t is not None]
    n_steps = len(episode.steps) if complete else (max(step_calls) + 1 if step_calls else 0)
    positions: List[Optional[int]] = list(range(n_steps))
    # The final update's belief (a report); the final scorer leaves b_{T-1}
    # unchanged, already scored at the last step.
    if complete and None in by_t and not calls[by_t[None]].is_scorer:
        positions.append(None)

    per = {cid: {"doc": Counter(), "think": Counter(), "items": [], "hits": set(), "expected": set()}
           for cid in scored}
    current = {cid: _unknown(cid) for cid in kinds}
    # y per (criterion, selected candidates): a belief is often carried
    # over unchanged for many steps.
    ys: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    for pos in positions:
        i = by_t.get(pos)
        if i is not None:
            call, outcome = calls[i], outcomes[i]
            # A reply that did not parse is charged in ``format`` only: its
            # passages stay unjudged and are counted in the call that shows
            # them again.
            if outcome.get("parsed") and not call.is_scorer:
                for cid in scored:
                    counts = _call_counts(episode, call, outcome, cid, gold)
                    per[cid]["hits"] |= counts["doc"].pop("hits")
                    per[cid]["expected"] |= counts["doc"].pop("expected")
                    for side in SIDES:
                        per[cid][side].update(counts[side])
                current.update({b["id"]: b for b in outcome.get("beliefs") or []})
        for cid in scored:
            s = current.get(cid) or _unknown(cid)
            key = (cid, tuple(s.get("selected") or []))
            if key not in ys:
                ys[key] = value_gold(episode, cid, s, kinds[cid], gold)
            per[cid]["items"].append((s.get("status") or UNKNOWN, ys[key]))

    criteria: Dict[str, Dict[str, Any]] = {}
    for cid in scored:
        p = per[cid]
        doc, think = p["doc"], p["think"]
        items = p["items"]
        criteria[cid] = {
            "kind": kinds[cid],
            "doc": {k: doc[k] for k in ("tp", "fp", "fn")},
            "think": {k: think[k] for k in ("tp", "fp", "fn")},
            "assign_f1": _f1(doc["tp"] + think["tp"], doc["fp"] + think["fp"], doc["fn"] + think["fn"]),
            "doc_recall_obs": _ratio(len(p["hits"]), len(p["expected"])),
            "value_accuracy": mean_or_none(y for _, y in items),
            "cal": mean_or_none(1.0 - (status_probability(st) - y) ** 2 for st, y in items),
            "items": items,
        }

    removed = [(u.get("id"), u.get("candidate") or "") for o in outcomes for u in o.get("updates") or []
               if u.get("status") == REMOVED]

    final: Optional[Dict[str, Any]] = None
    if complete:
        final_ok = None not in by_t or gates[by_t[None]]
        ordered = [next((b for b in beliefs if b.get("id") == c["id"]), _unknown(c["id"]))
                   for c in episode.gold_criteria or []]
        kind_list = [c["kind"] for c in episode.gold_criteria or []]
        scores = criterion_scores(ordered, kind_list)
        s = {name: (agg(scores) if scores else None) for name, agg in AGGREGATORS.items()}
        y = episode.correct
        r = s_scorer = s_answer = None
        i = by_t.get(None)
        if i is not None and calls[i].is_scorer:
            outcome = outcomes[i]
            r = scorer_reward(outcome, finish_reasons[i], y)
            if final_ok:
                s_scorer = outcome["certainty"]
                s_answer = answer_certainty(ordered, kind_list, outcome["final_score"]["candidates"],
                                            outcome.get("ruled_out"))
        elif y is not None and s[CERTAINTY_AGGREGATOR] is not None:
            r = (1.0 - (s[CERTAINTY_AGGREGATOR] - float(y)) ** 2) if final_ok else 0.0
        final = {"s": s, "s_scorer": s_scorer, "s_answer": s_answer, "y": y, "r_final": r}

    return {
        "format": _ratio(sum(update_gates), len(update_gates)),
        "assign": mean_or_none(c["assign_f1"] for c in criteria.values()),
        "cal": mean_or_none(c["cal"] for c in criteria.values()),
        "final": final,
        "criteria": criteria,
        "issues": dict(Counter(i.get("code") for o in outcomes for i in o.get("issues") or [])),
        "calls": len(calls),
        "removals": len(removed),
        "removed_gold": sum(1 for cid, name in removed
                            if (g := gold_criterion(episode, cid)) is not None and matches(episode, g, name)),
    }


def chain_values(episode: Episode, outcomes: Iterable[Dict[str, Any]],
                 beliefs: Optional[List[Dict[str, Any]]] = None) -> List[Tuple]:
    """The texts ``chain_score`` asks the gold about, each once, for
    ``NuggetGold.resolve``: ``(episode, criterion id, SPANS, text)`` of
    every observation's spans, and per selected candidate of a criterion
    the call changed (``changed``) or of the final *beliefs*
    ``(episode, criterion id, CANDIDATE, text[, open answers])`` (a
    several-answers criterion: judged against its answers still open,
    ``value_gold``)."""
    kinds = _kinds(episode)
    # Each text once (the episode is the same for all: left out of the key).
    out: Dict[Tuple, None] = {}

    def candidates(s: Dict[str, Any]) -> None:
        cid, selected = s["id"], list(s.get("selected") or [])
        if not selected or gold_criterion(episode, cid) is None:
            return
        if exclusive(kinds.get(cid, "")):
            out[(cid, CANDIDATE, selected[0])] = None
            return
        _, open_answers, unmatched = _open_answers(episode, cid, selected)
        for v in unmatched if open_answers else []:
            out[(cid, CANDIDATE, v, tuple(open_answers))] = None

    for o in outcomes:
        for x in o.get("observations") or []:
            if x.get("spans"):
                out[(x["id"], SPANS, spans_text(x))] = None
        changed = set(o.get("changed") or [])
        for b in o.get("beliefs") or []:
            if b["id"] in changed:
                candidates(b)
    for b in beliefs or []:
        candidates(b)
    return [(episode, *item) for item in out]


# ---------------------------------------------------------------------------
# Pooled measures
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else math.nan
    r = tp / (tp + fn) if tp + fn else math.nan
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan
    return {"precision": p, "recall": r, "f1": f}


def _nan(value: Optional[float]) -> float:
    return math.nan if value is None else value


def _final_measures(prefix: str, s: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_auroc": _auroc(y, s),
        f"{prefix}_tau_b": tau_b(s, y),
        f"{prefix}_brier": _brier(y, s),
        f"{prefix}_ece": _ece(y, s),
    }


# Per-criterion measures, averaged over a chain's criteria, then over the chains.
CRITERION_MEASURES = ("assign_f1", "doc_recall_obs", "value_accuracy", "cal")


def measures(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The measures over the chain *scores* (``chain_score``)."""
    calls = sum(s["calls"] for s in scores)
    out: Dict[str, Any] = {"chains": len(scores), "calls": calls}
    crits = [c for s in scores for c in s["criteria"].values()]
    out["criteria_scored"] = len(crits)
    for side in SIDES:
        c = Counter()
        for x in crits:
            c.update(x[side])
        out.update({f"obs_{side}_{k}": v for k, v in _prf(c["tp"], c["fp"], c["fn"]).items()})
    for name in CRITERION_MEASURES:
        out[f"crit_{name}"] = _nan(mean_or_none(mean_or_none(c[name] for c in s["criteria"].values()) for s in scores))
    pairs = [(status_probability(st), float(y), st) for c in crits for st, y in c["items"]]
    p = np.array([x[0] for x in pairs])
    g = np.array([x[1] for x in pairs])
    binary = [x for x in pairs if x[1] in (0.0, 1.0)]
    out.update({
        "beliefs_scored": len(pairs),
        "status_brier": float(((p - g) ** 2).mean()) if pairs else math.nan,
        "status_ece": _ece(g, p),
        "status_auroc": _auroc(np.array([x[1] for x in binary]), np.array([x[0] for x in binary])),
    })
    for status in BELIEF_STATUSES:
        own = [x[1] for x in binary if x[2] == status]
        out[f"gold_rate_{status.replace(' ', '_')}"] = sum(own) / len(own) if own else math.nan
        out[f"count_{status.replace(' ', '_')}"] = len(own)
    issues = Counter()
    for s in scores:
        issues.update(s["issues"])
    out.update({f"issue_rate_{code}": n / calls for code, n in sorted(issues.items())} if calls else {})
    removals = sum(s["removals"] for s in scores)
    out["removals"] = removals
    out["removed_gold_rate"] = sum(s["removed_gold"] for s in scores) / removals if removals else math.nan
    finals = [s["final"] for s in scores if s["final"] is not None]
    out.update({"format": _nan(mean_or_none(s["format"] for s in scores)), "assign": _nan(mean_or_none(s["assign"] for s in scores)),
                "cal": _nan(mean_or_none(s["cal"] for s in scores)), "r_final": _nan(mean_or_none(f["r_final"] for f in finals))})
    judged = [f for f in finals if f["y"] is not None and f["s"][CERTAINTY_AGGREGATOR] is not None]
    y = np.array([float(f["y"]) for f in judged])
    out["final_judged"] = len(judged)
    for name in AGGREGATORS:
        prefix = "certainty" if name == CERTAINTY_AGGREGATOR else f"certainty_{name}"
        out.update(_final_measures(prefix, np.array([f["s"][name] for f in judged]), y))
    # The final scorer, and the fixed function anchored on its answer's candidates, on
    # the episodes it scored (the fixed function again on the same ones).
    scored_finals = [f for f in finals if f["y"] is not None and f.get("s_scorer") is not None]
    ys = np.array([float(f["y"]) for f in scored_finals])
    out["final_scored"] = len(scored_finals)
    for prefix, values in (("certainty_scorer", [f["s_scorer"] for f in scored_finals]),
                           ("certainty_answer", [f["s_answer"] for f in scored_finals]),
                           ("certainty_formula_scored", [f["s"][CERTAINTY_AGGREGATOR] for f in scored_finals])):
        out.update(_final_measures(prefix, np.array(values, dtype=float), ys))
    return out


def assess(episodes: List[Episode], replays: List[Tuple[Episode, list, Any]], gold: NuggetGold) -> Dict[str, Any]:
    """The teacher's measures over its replayed calls: *replays* holds per
    episode ``(episode, calls, tracker)`` (``episodes.replay_teacher``,
    the calls before the first mismatch); the final score only for an
    episode replayed in full.  Resolves the gold of every value first."""
    items = []
    for ep, calls, tracker in replays:
        items += chain_values(ep, [c.outcome for c in calls], tracker.signal.beliefs())
    gold.resolve(items)
    scores = [chain_score(ep, [c.call for c in calls], [c.outcome for c in calls], ["stop"] * len(calls), gold,
                          tracker.signal.beliefs() or [], complete=ep.replay_error is None)
              for ep, calls, tracker in replays]
    return {"episodes": len(episodes), **measures(scores)}
