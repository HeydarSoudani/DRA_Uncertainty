"""Shared measures of the evaluation and of belief model training.

One copy of each measure, used by the run evaluation (``evaluation.runner``:
uncertainty and belief update), by belief model training
(``belief_model_training.scoring``) and by the offline analyses.

    auroc         AUROC of a score for a binary label (Mann-Whitney U, ties
                  as halves); NaN when one class is missing
    brier         mean squared error of a probability against the label
    ece           expected calibration error, equal-width bins
    tau_b         Kendall's tau-b; NaN with fewer than two items or one class
    mean          the mean, NaN when empty
    mean_or_none  the mean of the defined values (None and NaN left out),
                  None when there is none
    bootstrap_ci  percentile CI of a statistic over resampled queries
    paired_diff_ci
                  CI of the difference of one statistic on two scores of the
                  same queries, from the same resamples

The CIs resample queries (the unit that is independent), never the items
within a query: the caller passes one group of items per query.
"""

import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata

#: Resamples of the bootstrap and its seed (fixed, so a rerun gives the same CI).
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
#: Bins of the expected calibration error.
ECE_BINS = 10


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC of score *s* for the binary label *y* (Mann-Whitney U, ties as
    halves); NaN when one class is missing."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return math.nan
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(y: np.ndarray, s: np.ndarray) -> float:
    """Mean of (s - y)^2; NaN when empty."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    return float(((s - y) ** 2).mean()) if len(y) else math.nan


def ece(y: np.ndarray, s: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error over *bins* equal-width bins of *s* (in
    [0, 1]); NaN when empty."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    if not len(y):
        return math.nan
    idx = np.minimum((s * bins).astype(int), bins - 1)
    return float(sum(abs(s[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(y))


def tau_b(s: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b of *s* and *y*; NaN with fewer than two items or a
    constant *y*."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return math.nan
    return float(kendalltau(s, y)[0])


def mean(values: Sequence[float]) -> float:
    """The mean of *values*, NaN when empty."""
    return float(np.mean(values)) if len(values) else math.nan


def mean_or_none(values: Iterable[Optional[float]], digits: Optional[int] = None) -> Optional[float]:
    """The mean of the defined *values* (None and NaN left out), rounded to
    *digits* when given; None when there is none."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return round(m, digits) if digits is not None else m


def _resamples(n: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_ci(groups: List, stat: Callable[[List], float], level: float = 0.95,
                 resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Percentile CI of ``stat(groups)`` over *resamples* resamples of
    *groups* (one group per query, drawn with replacement); resamples where
    the statistic is NaN are left out.  ``(nan, nan)`` with fewer than two
    groups or no defined resample."""
    if len(groups) < 2:
        return math.nan, math.nan
    values = [stat([groups[i] for i in idx]) for idx in _resamples(len(groups), resamples, seed)]
    values = np.array([v for v in values if not math.isnan(v)])
    if not len(values):
        return math.nan, math.nan
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def paired_diff_ci(groups: List, stat_a: Callable[[List], float], stat_b: Callable[[List], float],
                   level: float = 0.95, resamples: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    """``(diff, low, high)``: ``stat_a - stat_b`` on *groups* and its
    percentile CI, both statistics computed on the same resamples (a paired
    bootstrap)."""
    diff = stat_a(groups) - stat_b(groups)
    if len(groups) < 2:
        return diff, math.nan, math.nan
    values = []
    for idx in _resamples(len(groups), resamples, seed):
        sample = [groups[i] for i in idx]
        v = stat_a(sample) - stat_b(sample)
        if not math.isnan(v):
            values.append(v)
    if not values:
        return diff, math.nan, math.nan
    alpha = (1 - level) / 2
    return diff, float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))
