"""Scores of a belief model against the gold of its episodes.

One set of standard measures (``measures``), used to assess the teacher on
the phase 1 episodes (``assess``) and the student on the validation steps
of phase 3; the phase 3 rewards (``rewards``) are their per-sample forms.
A belief update at step t is scored on three parts:

    what moves     precision / recall / F1 of the documents cited as
                   support (with a verified span) against the gold
                   documents among the call's passages
                   (``Episode.shown_gold``), pooled over the steps (micro)
    how far        coverage C_t against ``recall_so_far``: mean signed
                   error (> 0 = ahead of recall), RMSE, and Kendall's
                   tau-b pooled over all steps
    where it ends  final coverage C_T against the correctness y of the
                   agent's final answer: AUROC, Brier score, ECE (10
                   equal-width bins)

Coverage, with negative levels clipped to 0:
``C_t = mean_k max(level_t(k), 0) / 4`` (``types.STATUS_LEVEL``).
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


def cited_support(records: List[Dict[str, Any]]) -> Set[str]:
    """Documents cited as support, with at least one verified span, in a
    step's ``criteria_updates``."""
    return {e["doc_id"] for r in records for e in r.get("support") or [] if any(e.get("span_verified") or [])}


def support_f1(cited: Set[str], gold: Set[str]) -> float:
    """F1 of the cited support documents against the new gold documents of
    a step; 1 when both are empty (nothing cited on a step without gold)."""
    if not cited and not gold:
        return 1.0
    tp = len(cited & gold)
    return 2 * tp / (len(cited) + len(gold))


def step_row(episode: Episode, t: int, cited: Set[str], coverage: float, final: bool,
             gold: Optional[Set[str]] = None) -> Dict[str, Any]:
    """The ``measures`` row of one scored belief update at step *t*; *gold*
    defaults to the gold of the teacher's call (``Episode.shown_gold``)."""
    return {"cited": cited, "coverage": coverage, "gold": episode.shown_gold(t) if gold is None else gold,
            "recall": episode.steps[t].recall_so_far, "final": final, "correct": episode.correct}


def teacher_trace(episode: Episode) -> List[Dict[str, Any]]:
    """Per step of the teacher's matched replay, a ``measures`` row:
    ``{cited, coverage, gold, recall, final, correct}``.  The statuses
    after a step are the ``to`` of its saved records (unchanged when the
    updater was not called or failed); the last row is the final one when
    the whole episode replayed."""
    statuses = [INITIAL_STATUS] * len(episode.criteria)
    trace = []
    for t, step in enumerate(episode.steps[:episode.replay_steps]):
        records = [r for r in step.teacher_updates if r.get("from") is not None]
        if len(records) == len(statuses):
            statuses = [r["to"] for r in records]
        trace.append(step_row(episode, t, cited_support(step.teacher_updates), coverage(statuses), False))
    if trace and episode.replay_steps == len(episode.steps):
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
    """The teacher's scores over *episodes* (the matched replay steps)."""
    return measures([row for ep in episodes for row in teacher_trace(ep)])


def measures(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The measures over step rows ``{cited, coverage, gold, recall,
    final, correct}``; the final rows with a known ``correct`` give the
    final-coverage measures."""
    tp = fp = fn = 0
    cov, rec, final, y = [], [], [], []
    for t in rows:
        tp += len(t["cited"] & t["gold"])
        fp += len(t["cited"] - t["gold"])
        fn += len(t["gold"] - t["cited"])
        if t["recall"] is not None:
            cov.append(t["coverage"])
            rec.append(t["recall"])
        if t["final"] and t["correct"] is not None:
            final.append(t["coverage"])
            y.append(float(t["correct"]))
    p, r, f = _prf(tp, fp, fn)
    c, g = np.array(cov), np.array(rec)
    s, yy = np.array(final), np.array(y)
    return {
        "support_precision": p, "support_recall": r, "support_f1": f,
        "steps": len(c),
        "coverage_minus_recall": float((c - g).mean()) if len(c) else math.nan,
        "coverage_recall_rmse": float(np.sqrt(((c - g) ** 2).mean())) if len(c) else math.nan,
        "coverage_recall_tau_b": float(kendalltau(c, g)[0]) if len(c) > 1 else math.nan,
        "final_judged": len(yy),
        "final_auroc": _auroc(yy, s),
        "final_brier": float(((s - yy) ** 2).mean()) if len(yy) else math.nan,
        "final_ece": _ece(yy, s),
    }
