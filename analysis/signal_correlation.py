"""Correlation of the uncertainty-estimator signals with productive steps.

A step is productive when it found at least one relevant document not seen
in an earlier step of the same sample (``num_new_relevant > 0``, the same as
``new_item_precision > 0``).  Productive steps are frequent early and rarer
later, and most signals also decay with the step index, so a signal can
separate the two classes only because both follow the step index.  The
analysis has three parts:

- A, pooled AUROC (step index ignored): AUROC of the signal for the
  productive label over all steps of all queries, as in "When Deep Research
  Agents Stagnate".  It is the probability that a random productive step has
  a higher signal value than a random unproductive one (a tie counts 1/2);
  0.5 means no separation and below 0.5 means lower values go with
  productive steps.
- B, class distributions (step index ignored): the distribution of the
  signal on productive and unproductive steps, each class normalized to its
  own number of steps so the class imbalance does not hide the shape.
- C, turn-matched AUROC: the same probability with both steps of a pair
  taken at the same step index, so a signal that only follows the step
  index scores 0.5.  AUC(t) is computed per turn bin and plotted against the
  step index; the turn-matched AUROC pools the pairs of all turns, i.e. it
  is the mean of the per-turn AUROCs weighted by their number of pairs.
  Turn bins merge consecutive turns until a bin holds at least
  ``MIN_BIN_CLASS`` productive and ``MIN_BIN_CLASS`` unproductive steps (a
  shorter tail joins the last bin); inside a bin pairs are still matched on
  the exact turn, so the bins only group the display.  ``iteration`` is
  constant within a turn and is left out of part C.

Every part is computed on two step sets: all steps, and only the steps that
retrieved at least one new document (``num_new_docs > 0``).  Steps without
new documents can never be productive and have doc novelty and criteria
delta 0 by construction, so the second set shows what a signal knows beyond
"new documents arrived".  95% CIs come from a bootstrap over queries (steps
of one trajectory are not independent); the resamples are shared by all
signals and parts of a step set, so differences between them are paired.

Signals (step record fields of ``uncertainty/{query_id}.jsonl``):

- ``doc_novelty``           nu^D
- ``criteria_delta``        Delta^D
- ``query_novelty``         nu^q
- ``targeted_uncovered``    a_t reduced to a scalar: number of criteria a query
                            of the step targeted that were uncovered in
                            sigma_{t-1}
- ``iteration``             baseline: step index (parts A and B only)
- ``num_new_docs``          baseline: number of documents not seen before

Steps with a null gold (no qrels for the query, no documents) are dropped;
per signal, steps where the signal is null are dropped and counted.  Only
schema_version >= 5 files are read.  The default mode is ``monitor``: in
``inform`` mode the agent reads the signals, which changes the trajectory.

The run is selected with the same arguments as ``experiments/dra_inference.py``
(``--config``, ``--agentic-model``, ``--dataset``, ``--subset``, ``--retriever``,
``--uncertainty-estimator-mode`` and any YAML key as ``--flag`` override), and
its directory is built the same way:
``{output}/{dataset}_{split}[_{query_key}]_{retriever}/{run_name}/{ue_config}/``.

Output in ``<run_dir>/analysis/signal_correlation/``:

- ``{signal}.png`` and ``.pdf``: one column per step set; row 1 the class
  distributions with the pooled AUROC (A, B), row 2 AUC(t) with its CI band
  and the turn-matched AUROC (C), row 3 the productive and unproductive
  steps per turn bin.
- ``auc_by_turn.png`` and ``.pdf``: AUC(t) of every signal, one panel per
  step set.
- ``summary.csv``: per signal and step set, the counts and the pooled and
  turn-matched AUROC with their CIs.
- ``auc_by_turn.csv``: per signal, step set and turn bin, the counts and
  AUC(t) with its CI.

Usage::

    python analysis/signal_correlation.py --dataset browsecomp_plus --subset test \\
        --agentic-model uncertainty_aware --uncertainty-estimator-mode monitor
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# Repo root (utils) and src/ (component packages), as for experiments/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from indexing_corpus_dataset.dataset_loaders import resolve_split_id
from indexing_corpus_dataset.layout import DATASETS
from utils.cli_setup import (
    apply_config_to_args,
    load_run_config,
    parse_cli_overrides,
    resolve_dataset_defaults,
)
from utils.config import AGENTIC_MODEL_ALIAS, AGENTIC_MODEL_TO_LLM
from utils.io_utils import build_dataset_dir_name, build_run_name_for_pipeline, build_uncertainty_config_name

logger = logging.getLogger(__name__)

_OUTPUT_PREFIX = os.environ.get(
    "DRA_OUTPUT_ROOT", "/projects/0/prjs0834/heydars/DRA_training/run_outputs"
)
_CONFIG_DEFAULT = str(_REPO_ROOT / "experiments" / "configs" / "dra_inference.yaml")

MIN_SCHEMA_VERSION = 5
OUT_SUBDIR = Path("analysis") / "signal_correlation"
NUM_CONTINUOUS_BINS = 20
MIN_BIN_CLASS = 10
TURN_KEY = "iteration"

# ``by_turn``: part C applies (the signal is not constant within a turn).
SIGNALS = [
    {"key": "doc_novelty", "name": "Doc novelty (nu^D)", "short": "Doc novelty",
     "discrete": True, "by_turn": True},
    {"key": "criteria_delta", "name": "Criteria coverage change (Delta^D)", "short": "Criteria delta",
     "discrete": True, "by_turn": True},
    {"key": "query_novelty", "name": "Query novelty (nu^q)", "short": "Query novelty",
     "discrete": False, "by_turn": True},
    {"key": "targeted_uncovered", "name": "Targeted uncovered criteria (a_t)", "short": "Targeted uncovered",
     "discrete": True, "by_turn": True},
    {"key": "iteration", "name": "Step index (baseline)", "short": "Step index",
     "discrete": True, "by_turn": False},
    {"key": "num_new_docs", "name": "New docs (baseline)", "short": "New docs",
     "discrete": True, "by_turn": True},
]
TURN_SIGNALS = [s for s in SIGNALS if s["by_turn"]]
SUBSETS = [
    ("all", "All steps"),
    ("new_docs", "Steps with new docs"),
]

# Categorical slots 1 and 2 of the validated default palette for the two
# classes; slots 1-5, in order, for the signals of the overview figure, each
# with its own marker (three slots are below 3:1 on the surface, and the
# table ``auc_by_turn.csv`` carries the same values).  Text and grid in
# neutral ink.
COLOR_POS = "#2a78d6"
COLOR_NEG = "#eb6834"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES_MARKERS = ["o", "s", "^", "D", "v"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _targeted_uncovered(step: Dict[str, Any], criteria_ids: List[str]) -> Optional[int]:
    """Number of targeted criteria that were uncovered before the step."""
    targeted, before = step.get("criteria_targeted"), step.get("criteria_state_before")
    if targeted is None or before is None:
        return None
    index = {cid: k for k, cid in enumerate(criteria_ids)}
    return sum(1 for cid in targeted if cid in index and before[index[cid]] == "uncovered")


def load_steps(run_dir: Path, mode: str) -> pd.DataFrame:
    """One row per step of the run's uncertainty files, with the gold label."""
    rows: List[Dict[str, Any]] = []
    skipped_schema, skipped_mode = 0, 0
    for path in sorted((run_dir / "uncertainty").glob("*.jsonl")):
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if not lines or lines[0].get("record") != "meta":
            logger.warning("%s: no meta line, skipped", path)
            continue
        meta = lines[0]
        if (meta.get("schema_version") or 0) < MIN_SCHEMA_VERSION:
            skipped_schema += 1
            continue
        if meta.get("mode") != mode:
            skipped_mode += 1
            continue
        criteria_ids = [c["id"] for c in meta.get("criteria") or []]
        for step in lines[1:]:
            if step.get("record") != "step":
                continue
            num_new_relevant = step.get("num_new_relevant")
            rows.append({
                "query_id": str(meta["query_id"]),
                "productive": None if num_new_relevant is None else int(num_new_relevant > 0),
                "doc_novelty": step.get("doc_novelty"),
                "criteria_delta": step.get("criteria_delta"),
                "query_novelty": step.get("query_novelty"),
                "targeted_uncovered": _targeted_uncovered(step, criteria_ids),
                "iteration": step.get("iteration"),
                "num_new_docs": step.get("num_new_docs"),
            })
    if skipped_schema:
        logger.warning("%s: %d files with schema_version < %d skipped", run_dir, skipped_schema, MIN_SCHEMA_VERSION)
    if skipped_mode:
        logger.warning("%s: %d files with mode other than %s skipped", run_dir, skipped_mode, mode)
    return pd.DataFrame(rows, columns=["query_id", "productive"] + [s["key"] for s in SIGNALS])


# ---------------------------------------------------------------------------
# Pooled and turn-matched AUROC with a query-level bootstrap CI
# ---------------------------------------------------------------------------

def turn_bins(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Consecutive turn ranges ``(first, last)`` holding at least
    ``MIN_BIN_CLASS`` steps of each class; a shorter tail joins the last bin."""
    counts = df.groupby(TURN_KEY)["productive"].agg(["sum", "count"]).sort_index()
    bins: List[Tuple[int, int]] = []
    first, n_pos, n_neg = None, 0, 0
    for turn, row in counts.iterrows():
        first = turn if first is None else first
        n_pos += int(row["sum"])
        n_neg += int(row["count"] - row["sum"])
        if n_pos >= MIN_BIN_CLASS and n_neg >= MIN_BIN_CLASS:
            bins.append((int(first), int(turn)))
            first, n_pos, n_neg = None, 0, 0
    if first is not None:
        last = int(counts.index[-1])
        bins[-1:] = [(bins[-1][0], last)] if bins else [(int(first), last)]
    return bins


def _pair_counts(y: np.ndarray, s: np.ndarray) -> Tuple[float, int]:
    """``(wins, pairs)`` of productive over unproductive steps, a tie counting
    1/2; ``wins / pairs`` is the AUROC (Mann-Whitney U over the pairs)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0
    rank_sum = rankdata(s)[y == 1].sum()
    return float(rank_sum - n_pos * (n_pos + 1) / 2), n_pos * n_neg


def _statistics(
    y: np.ndarray, s: np.ndarray, turns: np.ndarray, turn_bin: Dict[int, int], n_bins: int,
) -> Tuple[float, float, np.ndarray]:
    """``(pooled, turn_matched, bin_auc)`` of one sample of steps; NaN where
    there is no pair.  An empty *turn_bin* leaves the turn-matched values NaN."""
    wins, pairs = _pair_counts(y, s)
    pooled = wins / pairs if pairs else np.nan
    bin_wins, bin_pairs = np.zeros(n_bins), np.zeros(n_bins)
    for turn, b in turn_bin.items():
        mask = turns == turn
        w, p = _pair_counts(y[mask], s[mask])
        bin_wins[b] += w
        bin_pairs[b] += p
    turn_matched = bin_wins.sum() / bin_pairs.sum() if bin_pairs.sum() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_auc = np.where(bin_pairs > 0, bin_wins / bin_pairs, np.nan)
    return pooled, turn_matched, bin_auc


def _value(x: float) -> Optional[float]:
    return None if np.isnan(x) else float(x)


def _ci(values: np.ndarray, point: float) -> Tuple[Optional[float], Optional[float]]:
    """95% percentile interval of the defined bootstrap values; none when the
    point estimate or every resample is undefined."""
    values = values[~np.isnan(values)]
    if np.isnan(point) or not len(values):
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def evaluate(
    df: pd.DataFrame, key: str, queries: np.ndarray, picks: np.ndarray,
    turn_bin: Dict[int, int], n_bins: int,
) -> Dict[str, Any]:
    """Pooled AUROC, turn-matched AUROC and per-bin AUC(t) of ``df[key]`` for
    ``df.productive``, with 95% CIs.

    Each row of *picks* is one resample: indices into *queries*, drawn with
    replacement, whose steps are all kept.
    """
    y = df["productive"].to_numpy(int)
    s = df[key].to_numpy(float)
    turns = df[TURN_KEY].to_numpy(int)
    pooled, turn_matched, bin_auc = _statistics(y, s, turns, turn_bin, n_bins)

    rows = df.groupby("query_id").indices
    empty = np.empty(0, dtype=int)
    groups = [rows.get(q, empty) for q in queries]
    boot_pooled, boot_turn, boot_bins = [], [], []
    for pick in picks:
        idx = np.concatenate([groups[g] for g in pick])
        b_pooled, b_turn, b_bins = _statistics(y[idx], s[idx], turns[idx], turn_bin, n_bins)
        boot_pooled.append(b_pooled)
        boot_turn.append(b_turn)
        boot_bins.append(b_bins)
    boot_bins = np.array(boot_bins).reshape(len(picks), n_bins)

    in_bin = pd.Series(turns).map(turn_bin)
    return {
        "auroc": _value(pooled),
        "auroc_ci": _ci(np.array(boot_pooled), pooled),
        "turn_auroc": _value(turn_matched),
        "turn_auroc_ci": _ci(np.array(boot_turn), turn_matched),
        "bin_auc": bin_auc,
        "bin_ci": [_ci(boot_bins[:, b], bin_auc[b]) for b in range(n_bins)],
        "bin_pos": np.array([int(y[(in_bin == b).to_numpy()].sum()) for b in range(n_bins)]),
        "bin_neg": np.array([int((y[(in_bin == b).to_numpy()] == 0).sum()) for b in range(n_bins)]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def _bin_label(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def _headline(label: str, value: Optional[float], ci: Tuple[Optional[float], Optional[float]]) -> str:
    if value is None:
        return f"{label} n/a (no productive/unproductive pair)"
    if ci[0] is None:
        return f"{label} {value:.2f}"
    return f"{label} {value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)


def _class_fractions(df: pd.DataFrame, key: str, discrete: bool):
    """Return ``(positions, tick_labels, width, frac_pos, frac_neg)``."""
    values = df[key].to_numpy(float)
    pos = values[df["productive"].to_numpy(int) == 1]
    neg = values[df["productive"].to_numpy(int) == 0]
    if discrete:
        cats = np.unique(values)
        frac_pos = np.array([(pos == c).sum() for c in cats]) / max(len(pos), 1)
        frac_neg = np.array([(neg == c).sum() for c in cats]) / max(len(neg), 1)
        positions = np.arange(len(cats), dtype=float)
        return positions, [_fmt(c) for c in cats], 1.0, frac_pos, frac_neg
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, NUM_CONTINUOUS_BINS + 1)
    frac_pos = np.histogram(pos, edges)[0] / max(len(pos), 1)
    frac_neg = np.histogram(neg, edges)[0] / max(len(neg), 1)
    return (edges[:-1] + edges[1:]) / 2, None, edges[1] - edges[0], frac_pos, frac_neg


def _distribution_panel(ax, df: pd.DataFrame, signal: Dict[str, Any], title: str, res: Dict[str, Any]) -> None:
    """Parts A and B: class distributions, pooled AUROC in the title."""
    _style(ax)
    ax.set_title(f"{title}\n{_headline('pooled AUROC', res['auroc'], res['auroc_ci'])}",
                 fontsize=9, color=TEXT_PRIMARY, loc="left")
    if df.empty:
        ax.text(0.5, 0.5, "no steps", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
        return

    positions, tick_labels, width, frac_pos, frac_neg = _class_fractions(df, signal["key"], signal["discrete"])
    bar = width * 0.4
    n_pos, n_neg = int(df["productive"].sum()), int((df["productive"] == 0).sum())
    # Two bars per position, separated by a small surface-colored gap.
    ax.bar(positions - bar / 2, frac_pos, width=bar, color=COLOR_POS, edgecolor=SURFACE, linewidth=1,
           label=f"productive (n={n_pos})")
    ax.bar(positions + bar / 2, frac_neg, width=bar, color=COLOR_NEG, edgecolor=SURFACE, linewidth=1,
           label=f"unproductive (n={n_neg})")
    if tick_labels is not None:
        step = max(1, len(tick_labels) // 15)
        ax.set_xticks(positions[::step])
        ax.set_xticklabels(tick_labels[::step])
    ax.set_xlabel(signal["name"], fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("fraction of steps in class", fontsize=9, color=TEXT_SECONDARY)
    _legend(ax, fontsize=8)


def _bin_axis(ax, bins: List[Tuple[int, int]], show_labels: bool) -> None:
    x = np.arange(len(bins))
    step = max(1, len(bins) // 15)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([_bin_label(*b) for b in bins[::step]] if show_labels else [])
    ax.set_xlim(-0.6, len(bins) - 0.4)


def _turn_panels(ax, ax_count, bins: List[Tuple[int, int]], res: Dict[str, Any]) -> None:
    """Part C: AUC(t) with its CI band, and the steps of each class per bin."""
    _style(ax)
    _style(ax_count)
    ax.set_title(_headline("turn-matched AUROC", res["turn_auroc"], res["turn_auroc_ci"]),
                 fontsize=9, color=TEXT_PRIMARY, loc="left")
    if not bins:
        ax.text(0.5, 0.5, "no steps", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
        return

    x = np.arange(len(bins))
    low = np.array([np.nan if c[0] is None else c[0] for c in res["bin_ci"]])
    high = np.array([np.nan if c[1] is None else c[1] for c in res["bin_ci"]])
    ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.fill_between(x, low, high, color=TEXT_PRIMARY, alpha=0.12, linewidth=0)
    ax.plot(x, res["bin_auc"], color=TEXT_PRIMARY, linewidth=1.5, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC(t)", fontsize=9, color=TEXT_SECONDARY)
    _bin_axis(ax, bins, show_labels=False)

    bar = 0.4
    ax_count.bar(x - bar / 2, res["bin_pos"], width=bar, color=COLOR_POS, edgecolor=SURFACE, linewidth=1,
                 label="productive")
    ax_count.bar(x + bar / 2, res["bin_neg"], width=bar, color=COLOR_NEG, edgecolor=SURFACE, linewidth=1,
                 label="unproductive")
    ax_count.set_ylabel("steps", fontsize=9, color=TEXT_SECONDARY)
    ax_count.set_xlabel("step index (turn bin)", fontsize=9, color=TEXT_SECONDARY)
    _bin_axis(ax_count, bins, show_labels=True)
    _legend(ax_count, fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0, 1))


def plot_signal(data: Dict[str, Dict[str, Any]], signal: Dict[str, Any], run_label: str, out_dir: Path) -> None:
    if signal["by_turn"]:
        fig, axes = plt.subplots(3, len(SUBSETS), figsize=(11, 9), facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 3, 1.2]})
    else:
        fig, axes = plt.subplots(1, len(SUBSETS), figsize=(11, 4), facecolor=SURFACE, squeeze=False)
    for col, (subset, title) in enumerate(SUBSETS):
        d = data[subset]
        _distribution_panel(axes[0, col], d["df"], signal, title, d["res"])
        if signal["by_turn"]:
            _turn_panels(axes[1, col], axes[2, col], d["bins"], d["res"])
    fig.suptitle(f"{signal['name']}  |  {run_label}", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{signal['key']}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_overview(per_subset: Dict[str, Dict[str, Any]], run_label: str, out_dir: Path) -> None:
    """AUC(t) of every part-C signal, one panel per step set."""
    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(11, 5.5), facecolor=SURFACE)
    for ax, (subset, title) in zip(axes, SUBSETS):
        _style(ax)
        ax.set_title(title, fontsize=9, color=TEXT_PRIMARY, loc="left")
        bins = per_subset[subset]["bins"]
        if not bins:
            ax.text(0.5, 0.5, "no steps", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
            continue
        x = np.arange(len(bins))
        ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)))
        for signal, color, marker in zip(TURN_SIGNALS, SERIES_COLORS, SERIES_MARKERS):
            res = per_subset[subset]["signals"][signal["key"]]["res"]
            pooled = "n/a" if res["auroc"] is None else f"{res['auroc']:.2f}"
            matched = "n/a" if res["turn_auroc"] is None else f"{res['turn_auroc']:.2f}"
            ax.plot(x, res["bin_auc"], color=color, linewidth=1.5, marker=marker, markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=f"{signal['short']}: pooled {pooled}, turn-matched {matched}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC(t)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_xlabel("step index (turn bin)", fontsize=9, color=TEXT_SECONDARY)
        _bin_axis(ax, bins, show_labels=True)
        _legend(ax, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(f"AUC(t) by signal  |  {run_label}", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"auc_by_turn.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


def analyze_run(run_dir: Path, mode: str, n_bootstrap: int, seed: int) -> Optional[pd.DataFrame]:
    steps = load_steps(run_dir, mode)
    n_null_gold = int(steps["productive"].isna().sum())
    steps = steps[steps["productive"].notna()].astype({"productive": int})
    if steps.empty:
        logger.warning("%s: no step with a gold label, skipped", run_dir)
        return None

    run_label = "/".join(run_dir.resolve().parts[-3:])
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    summary_rows: List[Dict[str, Any]] = []
    turn_rows: List[Dict[str, Any]] = []
    per_subset: Dict[str, Dict[str, Any]] = {}
    for subset, _ in SUBSETS:
        base = steps if subset == "all" else steps[steps["num_new_docs"] > 0]
        bins = turn_bins(base) if not base.empty else []
        turn_bin = {t: b for b, (first, last) in enumerate(bins) for t in range(first, last + 1)}
        # One set of query resamples per step set, shared by all signals.
        queries = base["query_id"].unique()
        picks = (rng.integers(len(queries), size=(n_bootstrap, len(queries)))
                 if len(queries) else np.empty((0, 0), dtype=int))
        per_subset[subset] = {"bins": bins, "signals": {}}

        for signal in SIGNALS:
            key = signal["key"]
            n_null = int(base[key].isna().sum())
            df = base[base[key].notna()].reset_index(drop=True)
            res = evaluate(df, key, queries, picks, turn_bin if signal["by_turn"] else {}, len(bins))
            per_subset[subset]["signals"][key] = {"df": df, "res": res}
            summary_rows.append({
                "signal": key, "subset": subset,
                "n_queries": df["query_id"].nunique(), "n_steps": len(df),
                "n_productive": int(df["productive"].sum()),
                "base_rate": _round(float(df["productive"].mean())) if len(df) else None,
                "n_null_signal": n_null,
                "auroc": _round(res["auroc"]),
                "auroc_ci_low": _round(res["auroc_ci"][0]), "auroc_ci_high": _round(res["auroc_ci"][1]),
                "turn_auroc": _round(res["turn_auroc"]),
                "turn_auroc_ci_low": _round(res["turn_auroc_ci"][0]),
                "turn_auroc_ci_high": _round(res["turn_auroc_ci"][1]),
            })
            if signal["by_turn"]:
                for b, (first, last) in enumerate(bins):
                    turn_rows.append({
                        "signal": key, "subset": subset, "turn_bin": _bin_label(first, last),
                        "turn_first": first, "turn_last": last,
                        "n_productive": int(res["bin_pos"][b]), "n_unproductive": int(res["bin_neg"][b]),
                        "auc": _round(_value(res["bin_auc"][b])),
                        "ci_low": _round(res["bin_ci"][b][0]), "ci_high": _round(res["bin_ci"][b][1]),
                    })

    for signal in SIGNALS:
        data = {
            subset: {"bins": per_subset[subset]["bins"], **per_subset[subset]["signals"][signal["key"]]}
            for subset, _ in SUBSETS
        }
        plot_signal(data, signal, run_label, out_dir)
    plot_overview(per_subset, run_label, out_dir)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(turn_rows).to_csv(out_dir / "auc_by_turn.csv", index=False)
    print(f"\n== {run_label}  ({steps['query_id'].nunique()} queries, {len(steps)} steps, "
          f"{n_null_gold} steps without gold dropped)")
    print(summary.to_string(index=False))
    print(f"-> {out_dir}")
    return summary


def _parse_args() -> argparse.Namespace:
    """Same run-selecting arguments as experiments/dra_inference.py.

    The frequently-varied knobs are CLI arguments; the rest come from the
    YAML ``--config`` and can be overridden with the matching ``--flag``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # ── File-backed config ─────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="Path to the YAML file holding the mostly-fixed pipeline variables (the one the run used). Any value in it can be overridden by passing the matching --flag on the CLI.")

    # ── Run selection (same as experiments/dra_inference.py) ───────────────
    parser.add_argument("--agentic-model", type=str, default="uncertainty_aware", choices=list(AGENTIC_MODEL_TO_LLM), help="Agent of the run; the LLM is selected automatically from the agent.")
    parser.add_argument("--dataset", type=str, default="browsecomp_plus", choices=list(DATASETS), help="Dataset of the run.")
    parser.add_argument("--subset", type=lambda v: None if v == "null" else v, default=None, help="Dataset subset/collection (unset or null = the dataset's default in layout.DATASET_SPECS). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical; browsecomp_plus: test; ragtime: unused.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever of the run.")
    parser.add_argument("--uncertainty-estimator-mode", type=str, default="monitor", choices=["monitor", "inform"], help="Uncertainty estimator mode of the run. 'monitor' (default): the signals never changed the trajectory. 'inform': the agent read them.")

    # ── Analysis ───────────────────────────────────────────────────────────
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Query-level bootstrap resamples for the AUROC CIs.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")

    args, extras = parser.parse_known_args()

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    cli_subset = args.subset
    config = load_run_config(args.config)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset

    # ── Derive --llm-model from --agentic-model ────────────────────────────-
    args.llm_model = AGENTIC_MODEL_TO_LLM[args.agentic_model]
    args.agentic_model = AGENTIC_MODEL_ALIAS.get(args.agentic_model, args.agentic_model)

    if args.output is None:
        args.output = _OUTPUT_PREFIX
    return args


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """The run directory experiments/dra_inference.py writes for *args*."""
    resolve_dataset_defaults(args)
    file_data_set = resolve_split_id(args.dataset, args.dataset_year, args.subset)
    run_name = build_run_name_for_pipeline(
        agentic_model=args.agentic_model, llm_model=args.llm_model, use_plan=args.use_plan,
    )
    dataset_dir = build_dataset_dir_name(args.dataset, file_data_set, args.query_key, args.retriever)
    ue_config = build_uncertainty_config_name(
        uncertainty_estimator_mode=args.uncertainty_estimator_mode,
        ensure_novel_seen_docs=args.ensure_novel_seen_docs,
    )
    return Path(args.output) / dataset_dir / run_name / ue_config


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = resolve_run_dir(args)
    if not (run_dir / "uncertainty").is_dir():
        raise SystemExit(f"error: no uncertainty/ folder in {run_dir}")
    analyze_run(run_dir, args.uncertainty_estimator_mode, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
