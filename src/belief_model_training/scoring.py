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
    reasoner     the answer candidates after the call against the gold
                 intermediate answers (``candidate_gold``), on the gold
                 criteria with gold answers that have candidates or whose
                 gold answer the call's agent turn states
                 (``reasoner_items``): Brier score and AUROC of the
                 candidate probabilities p against their correctness g,
                 the gold rate, the accuracy of each criterion's top
                 candidate, and the missed rate (the turn states the gold
                 answer and no candidate matches it), pooled
    final        the final confidence C_T = C^ret_T x R_T against the
                 correctness y of the agent's final answer: AUROC, Brier
                 score, ECE (10 equal-width bins); the same for C^ret_T
                 alone (does the reasoner side add to it?)

Coverage, with negative levels clipped to 0:
``C_t = mean_k max(level_t(k), 0) / 4`` (``types.STATUS_LEVEL``) over the
criteria with a status (an aggregate has none);
reasoning confidence R_t = the mean over the criteria of their top
candidate's probability p = (level + 2) / 6 (``types.candidate_probability``;
0 without a candidate), rest criteria left out
(``criteria.reasoning_confidence``).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata

from belief_model_component import STATUS_LEVEL, candidate_probability, initial_status
from belief_model_component.criteria import reasoning_confidence

from .candidate_gold import CandidateGold
from .episodes import Episode

TOP_LEVEL = max(STATUS_LEVEL.values())


def coverage(statuses: Sequence[Optional[str]]) -> float:
    """C_t of a belief state: mean clipped status level over the criteria
    with a status (None: an aggregate), in [0, 1]."""
    levels = [max(STATUS_LEVEL[s], 0) for s in statuses if s is not None]
    return sum(levels) / (len(levels) * TOP_LEVEL) if levels else 0.0


def confidence(cov: float, candidates: Sequence[Sequence[Dict[str, Any]]], kinds: Sequence[str]) -> float:
    """C = C^ret x R: the coverage times the reasoning confidence of the
    candidate lists of criteria of *kinds*."""
    return cov * reasoning_confidence(list(candidates), list(kinds))


def criteria_kinds(episode: Episode) -> List[str]:
    """The kind of each of the episode's criteria, in order."""
    return [c["kind"] for c in episode.criteria]


def cited_support(records: List[Dict[str, Any]]) -> Set[str]:
    """Documents cited as support, with at least one verified span, in a
    step's ``criteria_updates``; records with ``from`` null are left out:
    invalid updates (unknown criterion id, duplicate update), which are not
    applied, and criteria without a status, which cite nothing."""
    return {e["doc_id"] for r in records if r.get("from") is not None
            for e in r.get("support") or [] if any(e.get("span_verified") or [])}


def reasoner_items(episode: Episode, t: Optional[int], candidates: Sequence[Sequence[Dict[str, Any]]],
                   gold: CandidateGold) -> List[Dict[str, Any]]:
    """One ``{pairs, stated, missed}`` per gold criterion with gold answers
    that has candidates after the call at step *t* (None: the final call),
    or whose gold answer the call's agent turn states: ``pairs`` holds
    ``(p, g)`` per candidate (probability, is a gold answer), ``missed`` is
    true when the turn states the gold answer and no candidate is one.
    *candidates* are in the order of the episode's gold criteria."""
    stated = episode.stated_at(t) or set()
    items = []
    for c, cands in zip(episode.gold_criteria or [], candidates):
        if not c["answers"] or not (cands or c["id"] in stated):
            continue
        pairs = [(candidate_probability(x["status"]), bool(gold.correct(episode, c["id"], x["value"])))
                 for x in cands]
        items.append({"pairs": pairs, "stated": c["id"] in stated,
                      "missed": c["id"] in stated and not any(g for _, g in pairs)})
    return items


def reasoner_reward(items: List[Dict[str, Any]]) -> Optional[float]:
    """r_reas of one call: the mean over *items* (``reasoner_items``) of
    the mean 1 - (p - g)^2 over the criterion's candidates, 0 for a missed
    answer; None without items."""
    if not items:
        return None
    return sum(0.0 if it["missed"] else sum(1.0 - (p - g) ** 2 for p, g in it["pairs"]) / len(it["pairs"])
               for it in items) / len(items)


def call_row(episode: Episode, t: Optional[int], statuses: Sequence[str],
             candidates: Sequence[Sequence[Dict[str, Any]]], cited: Set[str], gold_docs: Set[str],
             gold: CandidateGold, final: bool) -> Dict[str, Any]:
    """The ``measures`` row of one scored updater call: at step *t* (None:
    the final call, on the answer turn), with the belief after it."""
    cov = coverage(statuses)
    recall = episode.steps[t].recall_so_far if t is not None else (
        episode.steps[-1].recall_so_far if episode.steps else None)
    return {"cited": cited, "gold": gold_docs, "coverage": cov, "recall": recall,
            "reasoner": reasoner_items(episode, t, candidates, gold),
            "confidence": confidence(cov, candidates, criteria_kinds(episode)),
            "final": final, "correct": episode.correct}


def _replay_candidates(lists: List[List[Dict[str, Any]]], index: Dict[str, int],
                       records: List[Dict[str, Any]]) -> None:
    """Apply saved ``candidate_updates`` records to per-criterion *lists*."""
    for r in records:
        if r.get("after") is not None and r["id"] in index:
            lists[index[r["id"]]] = r["after"]


def teacher_candidate_values(episode: Episode) -> List[Tuple[str, str]]:
    """``(criterion id, value)`` of every candidate the teacher kept."""
    records = [r for s in episode.steps for r in s.teacher_candidates] + episode.teacher_final_candidates
    return [(r["id"], c["value"]) for r in records for c in r.get("after") or []]


def teacher_trace(episode: Episode, gold: CandidateGold) -> List[Dict[str, Any]]:
    """Per call of the teacher's matched replay, a ``measures`` row.  The
    statuses after a step are the ``to`` of its saved records (unchanged
    when the updater was not called or failed), the candidates the saved
    ``candidate_updates`` applied; the last row is the final one when the
    whole episode replayed."""
    statuses = [initial_status(c["kind"]) for c in episode.criteria]
    candidates: List[List[Dict[str, Any]]] = [[] for _ in episode.criteria]
    index = {c["id"]: k for k, c in enumerate(episode.criteria)}
    ids = list(index)
    trace = []
    for t, step in enumerate(episode.steps[:episode.replay_steps]):
        # One record per criterion, in criteria order, comes first
        # (``BeliefState.update``); records of invalid updates follow.
        records = step.teacher_updates[:len(ids)]
        if [r.get("id") for r in records] == ids:
            statuses = [r["to"] for r in records]
        _replay_candidates(candidates, index, step.teacher_candidates)
        trace.append(call_row(episode, t, statuses, candidates, cited_support(step.teacher_updates),
                              episode.shown_gold(t), gold, False))
    if trace and episode.replay_steps == len(episode.steps):
        if episode.teacher_final_output is not None:
            _replay_candidates(candidates, index, episode.teacher_final_candidates)
            trace.append(call_row(episode, None, statuses, candidates, set(), set(), gold, False))
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


def assess(episodes: List[Episode], gold: CandidateGold) -> Dict[str, Any]:
    """The teacher's scores over *episodes* (the matched replay calls); the
    teacher's candidates must be resolved in *gold* first
    (``teacher_candidate_values``)."""
    return measures([row for ep in episodes for row in teacher_trace(ep, gold)])


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
    items = [it for r in rows for it in r["reasoner"]]
    pairs = [pg for it in items for pg in it["pairs"]]
    tops = [max(it["pairs"], key=lambda pg: pg[0])[1] for it in items if it["pairs"]]
    stated = [it["missed"] for it in items if it["stated"]]
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
    pp, pg = np.array([p for p, _ in pairs]), np.array([float(g) for _, g in pairs])
    c, g = np.array(cov), np.array(rec)
    yy = np.array(y)
    return {
        "calls": len(rows),
        "coverage_minus_recall": float((c - g).mean()) if len(c) else math.nan,
        "coverage_recall_rmse": float(np.sqrt(((c - g) ** 2).mean())) if len(c) else math.nan,
        "coverage_recall_tau_b": float(kendalltau(c, g)[0]) if len(c) > 1 else math.nan,
        "support_precision": p, "support_recall": r, "support_f1": f,
        "candidates_scored": len(pairs),
        "candidate_brier": float(((pp - pg) ** 2).mean()) if len(pairs) else math.nan,
        "candidate_auroc": _auroc(pg, pp),
        "candidate_gold_rate": float(pg.mean()) if len(pairs) else math.nan,
        "top_candidate_accuracy": sum(tops) / len(tops) if tops else math.nan,
        "missed_rate": sum(stated) / len(stated) if stated else math.nan,
        "final_judged": len(yy),
        **_final_measures("final", np.array(conf), yy),
        **_final_measures("final_coverage", np.array(final_cov), yy),
    }
