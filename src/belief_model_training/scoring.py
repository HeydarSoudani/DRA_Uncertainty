"""Scores of a belief model against the gold of its episodes.

One set of standard measures (``measures``), used to assess the teacher on
the phase 1 episodes (``assess``) and the student on the validation chains
of phase 3; the phase 3 rewards (``rewards``) are their per-sample forms.
An updater call is scored on the two sides of the belief and the chain on
its end:

    retriever    coverage C_t against ``recall_so_far``: mean signed error
                 (> 0 = ahead of recall), RMSE, Kendall's tau-b pooled over
                 the calls; logged too: precision / recall / F1 of the
                 documents cited as support (verified span) against the
                 gold documents among the call's passages, pooled (micro)
    reasoner     the ``utilized`` judgments of the rises the call judges
                 against the gold (the agent turn states the criterion's
                 gold answer, ``utilized_gold``): accuracy, and precision /
                 recall / F1 of "utilized", pooled
    final        the final confidence C_T = C^ret_T x U_T against the
                 correctness y of the agent's final answer: AUROC, Brier
                 score, ECE (10 equal-width bins); the same for C^ret_T
                 alone (does the reasoner side add to it?)

Coverage, with negative levels clipped to 0:
``C_t = mean_k max(level_t(k), 0) / 4`` (``types.STATUS_LEVEL``);
utilization U_t = the used share of the judged rises
(``BeliefState.utilization``; 1 before any rise was judged).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata

from belief_model_component import STATUS_LEVEL, INITIAL_STATUS

from .episodes import Episode

TOP_LEVEL = max(STATUS_LEVEL.values())


def coverage(statuses: Sequence[str]) -> float:
    """C_t of a belief state: mean clipped status level over the criteria, in [0, 1]."""
    if not statuses:
        return 0.0
    return sum(max(STATUS_LEVEL[s], 0) for s in statuses) / (len(statuses) * TOP_LEVEL)


def confidence(cov: float, utilization: Optional[float]) -> float:
    """C = C^ret x U: the covered share the agent used (U = 1 before any
    rise was judged)."""
    return cov * (1.0 if utilization is None else utilization)


def cited_support(records: List[Dict[str, Any]]) -> Set[str]:
    """Documents cited as support, with at least one verified span, in a
    step's ``criteria_updates``; the records of invalid updates (``from``
    null: unknown criterion id, duplicate update) are not applied and are
    left out."""
    return {e["doc_id"] for r in records if r.get("from") is not None
            for e in r.get("support") or [] if any(e.get("span_verified") or [])}


def used_pairs(episode: Episode, t: Optional[int], used: List[Dict[str, Any]]) -> List[Tuple[bool, bool]]:
    """``(judged, gold)`` of each rise judged with a valid value on the
    agent turn of step *t* (None: the answer turn)."""
    stated = episode.stated_at(t) or set()
    return [(bool(u["utilized"]), u["id"] in stated) for u in used if u.get("utilized") is not None]


def call_row(episode: Episode, t: Optional[int], statuses: Sequence[str], utilization: Optional[float],
             cited: Set[str], gold: Set[str], used: List[Dict[str, Any]], final: bool) -> Dict[str, Any]:
    """The ``measures`` row of one scored updater call: at step *t* (None:
    the final call, on the answer turn), with the belief after it."""
    cov = coverage(statuses)
    recall = episode.steps[t].recall_so_far if t is not None else (
        episode.steps[-1].recall_so_far if episode.steps else None)
    return {"cited": cited, "gold": gold, "coverage": cov, "recall": recall,
            "used": used_pairs(episode, t, used), "confidence": confidence(cov, utilization),
            "final": final, "correct": episode.correct}


def teacher_trace(episode: Episode) -> List[Dict[str, Any]]:
    """Per call of the teacher's matched replay, a ``measures`` row.  The
    statuses after a step are the ``to`` of its saved records (unchanged
    when the updater was not called or failed), U from the saved
    ``utilized_updates``; the last row is the final one when the whole
    episode replayed."""
    statuses = [INITIAL_STATUS] * len(episode.criteria)
    used_gain = judged_gain = 0
    trace = []

    def add_used(used: List[Dict[str, Any]]) -> None:
        nonlocal used_gain, judged_gain
        for u in used:
            if u.get("utilized") is not None:
                judged_gain += u["rise"]
                used_gain += u["rise"] if u["utilized"] else 0

    for t, step in enumerate(episode.steps[:episode.replay_steps]):
        records = [r for r in step.teacher_updates if r.get("from") is not None]
        if len(records) == len(statuses):
            statuses = [r["to"] for r in records]
        add_used(step.teacher_used)
        trace.append(call_row(episode, t, statuses, used_gain / judged_gain if judged_gain else None,
                              cited_support(step.teacher_updates), episode.shown_gold(t), step.teacher_used, False))
    if trace and episode.replay_steps == len(episode.steps):
        if episode.teacher_final_used:
            add_used(episode.teacher_final_used)
            trace.append(call_row(episode, None, statuses, used_gain / judged_gain if judged_gain else None,
                                  set(), set(), episode.teacher_final_used, False))
        trace[-1]["final"] = True
    return trace


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


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else math.nan
    r = tp / (tp + fn) if tp + fn else math.nan
    f = 2 * p * r / (p + r) if tp else 0.0
    return p, r, f


def assess(episodes: List[Episode]) -> Dict[str, Any]:
    """The teacher's scores over *episodes* (the matched replay calls)."""
    return measures([row for ep in episodes for row in teacher_trace(ep)])


def _final_measures(prefix: str, s: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_auroc": _auroc(y, s),
        f"{prefix}_brier": float(((s - y) ** 2).mean()) if len(y) else math.nan,
        f"{prefix}_ece": _ece(y, s),
    }


def measures(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The measures over call rows (``call_row``); the final rows with a
    known ``correct`` give the final measures."""
    tp = fp = fn = 0
    cov, rec, conf, final_cov, y = [], [], [], [], []
    used = [p for r in rows for p in r["used"]]
    for t in rows:
        tp += len(t["cited"] & t["gold"])
        fp += len(t["cited"] - t["gold"])
        fn += len(t["gold"] - t["cited"])
        if t["recall"] is not None:
            cov.append(t["coverage"])
            rec.append(t["recall"])
        if t["final"] and t["correct"] is not None:
            conf.append(t["confidence"])
            final_cov.append(t["coverage"])
            y.append(float(t["correct"]))
    p, r, f = _prf(tp, fp, fn)
    up, ur, uf = _prf(sum(a and g for a, g in used), sum(a and not g for a, g in used),
                      sum(g and not a for a, g in used))
    c, g = np.array(cov), np.array(rec)
    yy = np.array(y)
    return {
        "calls": len(rows),
        "coverage_minus_recall": float((c - g).mean()) if len(c) else math.nan,
        "coverage_recall_rmse": float(np.sqrt(((c - g) ** 2).mean())) if len(c) else math.nan,
        "coverage_recall_tau_b": float(kendalltau(c, g)[0]) if len(c) > 1 else math.nan,
        "support_precision": p, "support_recall": r, "support_f1": f,
        "utilized_judged": len(used),
        "utilized_accuracy": sum(a == g for a, g in used) / len(used) if used else math.nan,
        "utilized_gold_rate": sum(g for _, g in used) / len(used) if used else math.nan,
        "utilized_precision": up, "utilized_recall": ur, "utilized_f1": uf,
        "final_judged": len(yy),
        **_final_measures("final", np.array(conf), yy),
        **_final_measures("final_coverage", np.array(final_cov), yy),
    }
