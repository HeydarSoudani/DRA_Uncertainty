"""Correlation of the belief-model signals with the step gold.

The gold of a step is its new-item precision, the fraction of the step's
documents that are relevant and not seen in an earlier step of the same
sample (``new_item_precision``).  It is used twice:

- binary: the step is productive when it found at least one new relevant
  document (``num_new_relevant > 0``, the same as ``new_item_precision > 0``);
  the measure is AUROC, the probability that a random productive step has a
  higher signal value than a random unproductive one (a tie counts 1/2);
  0.5 means no separation and below 0.5 means lower values go with
  productive steps.
- graded: ``new_item_precision`` itself; the measure is Kendall's tau-b, the
  rank correlation that accounts for ties (most steps have gold 0); 0 means
  no association and a negative value means lower values go with a higher
  gold.

Both are pooled over all steps (every turn) of all queries.  95% CIs come
from a bootstrap over queries (steps of one trajectory are not independent);
the resamples are shared by all signals, so differences between them are
paired.

Signals (step record fields of ``belief/{query_id}.jsonl``):

- ``doc_novelty``           nu^D
- ``criteria_delta``        Delta^D
- ``query_novelty``         nu^q

Steps with a null gold (no qrels for the query, no documents) are dropped;
per signal, steps where the signal is null are dropped and counted.

Final belief state.  The belief at the end of the run is also read as an
uncertainty estimate of the whole query: the final belief is the mean, over
the criteria, of each criterion's status level
(``belief_model_component.types.STATUS_LEVEL``, -2 to 4) in the last step's
``belief_after``, divided by 4 (-0.5 every criterion "doubtful", 0 every
criterion "unknown" as at b_0, 1 every criterion "certain").
Per query it is compared with:

- correctness (binary; ``correct`` of the LLM judge or ``exact_match`` of
  the numeric match in ``accuracy.jsonl``): AUROC.  Report datasets have no
  correctness, so this part is skipped for them.
- Recall@N over the seen documents (the last step's ``recall_so_far``, the
  evaluator's per-query Recall@N): Kendall's tau-b.

CIs from a bootstrap over queries.  Queries without a belief state, a
correctness label (judge errors) or a recall are dropped and counted.  Only
schema_version >= 15 files are read (the ``belief_*`` field names start
at 12, ``criteria_delta`` at 13, the Agent-BRACE statuses at 14, the signed
scale at 15).  The
default mode is ``monitor``: in ``inform`` mode the agent reads the signals,
which changes the trajectory.

The run is selected with the same arguments as ``experiments/inference/dra_inference.py``
(``--config``, ``--policy-model``, ``--dataset``, ``--subset``, ``--retriever``,
``--belief-mode`` and any YAML key as ``--flag`` override), and
its directory is built the same way:
``{output}/{dataset}_{split}[_{query_key}]_{retriever}/{run_name}/{belief_config}/``.

Output in ``<run_dir>/analysis/signal_correlation/``:

- ``distributions.png``: one panel per signal, the distribution of the
  signal on productive (green) and unproductive (red) steps, each class
  normalized to its own number of steps; the AUROC and tau-b are in the
  title.  The axes are fixed (see ``SIGNALS``) so runs compare directly.
- ``summary.csv``: per signal, the counts and the AUROC and tau-b with their
  CIs.
- ``final_state.png``: the final belief of correct (green) and
  incorrect (red) queries with the AUROC (left, only with correctness), and
  the final belief against Recall@N with the tau-b (right).
- ``final_state_summary.csv``: per target (correctness, Recall@N), the counts
  and the measure with its CI.

Usage::

    python analysis/signal_correlation.py --dataset browsecomp_plus --subset test \\
        --policy-model belief_aware --belief-mode monitor
"""

import argparse
import json
import logging
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from indexing_corpus_dataset.dataset_loaders import resolve_split_id
from indexing_corpus_dataset.layout import DATASETS, OUTPUT_ROOT
from utils.cli_setup import (
    apply_config_to_args,
    load_run_config,
    parse_cli_overrides,
    resolve_dataset_defaults,
)
from belief_model_component.types import STATUS_LEVEL
from utils.config import POLICY_MODEL_TO_LLM
from utils.io_utils import build_dataset_dir_name, build_run_name_for_pipeline, build_belief_config_name

logger = logging.getLogger(__name__)

_OUTPUT_PREFIX = str(OUTPUT_ROOT)
_CONFIG_DEFAULT = str(_REPO_ROOT / "experiments" / "configs" / "dra_inference.yaml")

MIN_SCHEMA_VERSION = 15
OUT_SUBDIR = Path("analysis") / "signal_correlation"
NUM_CONTINUOUS_BINS = 20
GOLD_KEY = "new_item_precision"

# ``range``: the fixed x range of the panel.  ``discrete``: one bar per
# integer in the range, values outside it joining the end bars; otherwise
# ``NUM_CONTINUOUS_BINS`` equal-width bins over the range.
SIGNALS = [
    {"key": "doc_novelty", "name": "Doc novelty (nu^D)", "discrete": False, "range": (0.0, 1.0)},
    {"key": "criteria_delta", "name": "Criteria change (Delta^D)", "discrete": True, "range": (-8, 12)},
    {"key": "query_novelty", "name": "Query novelty (nu^q)", "discrete": False, "range": (0.0, 1.0)},
]

# The final-state score, one value per query: -0.5 (every criterion
# "doubtful") to 1 (every criterion "certain"), see ``_final_belief``.
BELIEF = {"key": "belief", "name": "Final belief", "discrete": False, "range": (-0.5, 1.0)}
# Per-query correctness field of ``accuracy.jsonl``, by answer evaluator.
CORRECTNESS_KEYS = ("correct", "exact_match")

# Green for productive, red for unproductive steps; the two overlap, so each
# class also keeps a solid outline.  Text and grid in neutral ink.
COLOR_POS = "#1f9d55"
COLOR_NEG = "#d64545"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_belief(run_dir: Path, mode: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """``(query_id, step records)`` per belief file of the run with
    schema_version >= ``MIN_SCHEMA_VERSION``; empty when the run's belief
    model (``run_config.json``) is not in the given *mode*."""
    run_mode = (json.loads((run_dir / "run_config.json").read_text()).get("belief_model") or {}).get("mode")
    if run_mode != mode:
        logger.warning("%s: belief model mode %s in run_config.json, not %s; no belief file read", run_dir, run_mode, mode)
        return []
    samples: List[Tuple[str, List[Dict[str, Any]]]] = []
    skipped_schema = 0
    for path in sorted((run_dir / "belief").glob("*.jsonl")):
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if not lines or lines[0].get("record") != "meta":
            logger.warning("%s: no meta line, skipped", path)
            continue
        meta = lines[0]
        if (meta.get("schema_version") or 0) < MIN_SCHEMA_VERSION:
            skipped_schema += 1
            continue
        samples.append((str(meta["query_id"]), [s for s in lines[1:] if s.get("record") == "step"]))
    if skipped_schema:
        logger.warning("%s: %d files with schema_version < %d skipped", run_dir, skipped_schema, MIN_SCHEMA_VERSION)
    return samples


def load_steps(samples: List[Tuple[str, List[Dict[str, Any]]]]) -> pd.DataFrame:
    """One row per step, with both golds."""
    rows: List[Dict[str, Any]] = []
    for query_id, steps in samples:
        for step in steps:
            num_new_relevant = step.get("num_new_relevant")
            row = {
                "query_id": query_id,
                "productive": None if num_new_relevant is None else int(num_new_relevant > 0),
                GOLD_KEY: step.get(GOLD_KEY),
            }
            row.update({s["key"]: step.get(s["key"]) for s in SIGNALS})
            rows.append(row)
    return pd.DataFrame(rows, columns=["query_id", "productive", GOLD_KEY] + [s["key"] for s in SIGNALS])


def _final_belief(statuses: Optional[List[str]]) -> Optional[float]:
    """Mean status level (``STATUS_LEVEL``) of a belief state, divided by
    the top level: 1 every criterion "certain", 0 every criterion
    "unknown", negative when the passages are against them; None without
    one."""
    if not statuses:
        return None
    top = max(STATUS_LEVEL.values())
    return sum(STATUS_LEVEL[status] for status in statuses) / (len(statuses) * top)


def load_correctness(run_dir: Path) -> Dict[str, Optional[int]]:
    """Per query, 1 correct / 0 incorrect from ``accuracy.jsonl`` (None on a
    judge error); empty when the run has no answer evaluation."""
    path = run_dir / "accuracy.jsonl"
    if not path.is_file():
        return {}
    correctness: Dict[str, Optional[int]] = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line) if line.strip() else {}
            if "query_id" not in record:
                continue
            value = next((record[k] for k in CORRECTNESS_KEYS if record.get(k) is not None), None)
            ok = value is not None and not record.get("judge_error")
            correctness[str(record["query_id"])] = int(bool(value)) if ok else None
    return correctness


def load_queries(samples: List[Tuple[str, List[Dict[str, Any]]]], correctness: Dict[str, Optional[int]]) -> pd.DataFrame:
    """One row per query: the final belief and Recall@N of the last
    step, and the correctness (None when unknown)."""
    rows: List[Dict[str, Any]] = []
    for query_id, steps in samples:
        last = steps[-1] if steps else {}
        rows.append({
            "query_id": query_id,
            "belief": _final_belief(last.get("belief_after")),
            "recall": last.get("recall_so_far"),
            "correct": correctness.get(query_id),
        })
    return pd.DataFrame(rows, columns=["query_id", "belief", "recall", "correct"])


# ---------------------------------------------------------------------------
# AUROC and tau-b with a query-level bootstrap CI
# ---------------------------------------------------------------------------

def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC of *s* for productive over unproductive steps, a tie
    counting 1/2; NaN without both classes."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank_sum = rankdata(s)[y == 1].sum()
    return float(rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _tau(s: np.ndarray, g: np.ndarray) -> float:
    """Kendall's tau-b of *s* with *g*; NaN when either is constant."""
    if len(s) < 2 or len(np.unique(s)) < 2 or len(np.unique(g)) < 2:
        return np.nan
    return float(kendalltau(s, g)[0])


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
    binary: Optional[str] = None, graded: Optional[str] = None,
) -> Dict[str, Any]:
    """AUROC (for the 0/1 column *binary*) and tau-b (for the column
    *graded*) of ``df[key]`` with 95% CIs; a measure whose column is None is
    left out.

    Each row of *picks* is one resample: indices into *queries*, drawn with
    replacement, whose rows are all kept.
    """
    s = df[key].to_numpy(float)
    measures = {}
    if binary is not None:
        y = df[binary].to_numpy(int)
        measures["auroc"] = lambda idx: _auroc(y[idx], s[idx])
    if graded is not None:
        g = df[graded].to_numpy(float)
        measures["tau"] = lambda idx: _tau(s[idx], g[idx])

    rows = df.groupby("query_id").indices
    empty = np.empty(0, dtype=int)
    groups = [rows.get(q, empty) for q in queries]
    resamples = [np.concatenate([groups[k] for k in pick]) for pick in picks]

    res: Dict[str, Any] = {}
    everything = np.arange(len(df))
    for name, measure in measures.items():
        point = measure(everything)
        res[name] = _value(point)
        res[f"{name}_ci"] = _ci(np.array([measure(idx) for idx in resamples], dtype=float), point)
    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _headline(label: str, value: Optional[float], ci: Tuple[Optional[float], Optional[float]]) -> str:
    if value is None:
        return f"{label} n/a"
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


def _signal_axis(values: np.ndarray, signal: Dict[str, Any]):
    """Return ``(positions, tick_labels, width, index)``: the x positions of
    the fixed values (or bins) of *signal* and, per step, the position it
    falls in."""
    lo, hi = signal["range"]
    if signal["discrete"]:
        positions = np.arange(lo, hi + 1, dtype=float)
        labels = [str(int(v)) for v in positions]
        labels[0], labels[-1] = f"<={labels[0]}", f">={labels[-1]}"
        index = np.clip(np.rint(values), lo, hi).astype(int) - lo
        return positions, labels, 1.0, index
    edges = np.linspace(lo, hi, NUM_CONTINUOUS_BINS + 1)
    index = np.clip(np.digitize(values, edges[1:-1]), 0, NUM_CONTINUOUS_BINS - 1)
    return (edges[:-1] + edges[1:]) / 2, None, edges[1] - edges[0], index


def _title(name: str, res: Dict[str, Any]) -> str:
    """*name* over the measures *res* holds."""
    labels = {"auroc": "AUROC", "tau": "tau-b"}
    return f"{name}\n" + "  |  ".join(_headline(labels[m], res[m], res[f"{m}_ci"]) for m in labels if m in res)


def _distribution_panel(
    ax, df: pd.DataFrame, signal: Dict[str, Any], res: Dict[str, Any],
    label: str = "productive", unit: str = "steps", title: Optional[str] = None,
) -> None:
    """Positive and negative rows of the 0/1 column *label* per signal value,
    each class normalized to its own number of rows; the measures under
    *title* (default the signal name)."""
    _style(ax)
    ax.set_title(_title(title or signal["name"], res), fontsize=9, color=TEXT_PRIMARY, loc="left")
    lo, hi = signal["range"]
    pad = 0.5 if signal["discrete"] else 0.0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0.0, 1.0)
    if df.empty:
        ax.text(0.5, 0.5, f"no {unit}", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
        return

    positions, tick_labels, width, index = _signal_axis(df[signal["key"]].to_numpy(float), signal)
    y = df[label].to_numpy(int)
    frac_pos = np.bincount(index[y == 1], minlength=len(positions)) / max(int(y.sum()), 1)
    frac_neg = np.bincount(index[y == 0], minlength=len(positions)) / max(int((y == 0).sum()), 1)
    bar = width * 0.9
    for frac, color in ((frac_neg, COLOR_NEG), (frac_pos, COLOR_POS)):
        ax.bar(positions, frac, width=bar, color=color, alpha=0.45)
        ax.bar(positions, frac, width=bar, fill=False, edgecolor=color, linewidth=1.2)
    if tick_labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
    ax.set_xlabel(f"{signal['name']} value", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(f"fraction of {unit} in class", fontsize=9, color=TEXT_SECONDARY)


def plot_distributions(
    steps: pd.DataFrame, data: Dict[str, Dict[str, Any]], run_label: str, out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(5.5 * len(SIGNALS), 4.8), facecolor=SURFACE)
    for ax, signal in zip(np.atleast_1d(axes), SIGNALS):
        _distribution_panel(ax, data[signal["key"]]["df"], signal, data[signal["key"]]["res"])
    # The class counts are the same in every panel (the steps with a gold);
    # a panel's own count can be lower only by its null signal values.
    n_pos, n_neg = int(steps["productive"].sum()), int((steps["productive"] == 0).sum())
    _finish(fig, run_label, out_dir / "distributions.png",
            f"productive (n={n_pos})", f"unproductive (n={n_neg})")


def _finish(fig, run_label: str, path: Path, pos_label: Optional[str], neg_label: Optional[str]) -> None:
    """Class legend (left out without labels), run title, save."""
    if pos_label is not None:
        handles = [Patch(facecolor=color, alpha=0.45, edgecolor=color, label=label)
                   for color, label in ((COLOR_POS, pos_label), (COLOR_NEG, neg_label))]
        legend = fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)
    fig.suptitle(run_label, fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _recall_panel(ax, df: pd.DataFrame, res: Dict[str, Any]) -> None:
    """Final belief against Recall@N, one point per query, colored by
    correctness when known; tau-b in the title."""
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_title(_title("Final belief vs Recall@N", res), fontsize=9, color=TEXT_PRIMARY, loc="left")
    lo, hi = BELIEF["range"]
    pad = 0.02 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.03, 1.03)
    if df.empty:
        ax.text(0.5, 0.5, "no queries", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
    colors = [TEXT_SECONDARY if pd.isna(c) else COLOR_POS if c else COLOR_NEG for c in df["correct"]]
    ax.scatter(df["belief"], df["recall"], s=22, c=colors, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"{BELIEF['name']} value", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Recall@N", fontsize=9, color=TEXT_SECONDARY)


def plot_final_state(
    data: Dict[str, Dict[str, Any]], run_label: str, out_dir: Path,
) -> None:
    """Final belief by correctness (when the run has it) and against Recall@N."""
    keys = [k for k in ("correct", "recall") if k in data]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.8), facecolor=SURFACE)
    for ax, key in zip(np.atleast_1d(axes), keys):
        if key == "correct":
            _distribution_panel(ax, data[key]["df"], BELIEF, data[key]["res"], label="correct",
                                unit="queries", title=f"{BELIEF['name']} vs correctness")
        else:
            _recall_panel(ax, data[key]["df"], data[key]["res"])
    labels = (None, None)
    if "correct" in data:
        correct = data["correct"]["df"]["correct"]
        labels = (f"correct (n={int(correct.sum())})", f"incorrect (n={int((correct == 0).sum())})")
    _finish(fig, run_label, out_dir / "final_state.png", *labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


def _measure_columns(res: Dict[str, Any], names: Tuple[str, ...]) -> Dict[str, Optional[float]]:
    row: Dict[str, Optional[float]] = {}
    for name in names:
        row[name] = _round(res[name])
        row[f"{name}_ci_low"] = _round(res[f"{name}_ci"][0])
        row[f"{name}_ci_high"] = _round(res[f"{name}_ci"][1])
    return row


def analyze_steps(
    samples: List[Tuple[str, List[Dict[str, Any]]]], run_label: str, out_dir: Path, n_bootstrap: int, seed: int,
) -> None:
    """Per-step signals against the step gold: ``distributions.png``, ``summary.csv``."""
    steps = load_steps(samples)
    has_gold = steps["productive"].notna() & steps[GOLD_KEY].notna()
    n_null_gold = int((~has_gold).sum())
    steps = steps[has_gold].astype({"productive": int})
    if steps.empty:
        logger.warning("%s: no step with a gold label, step analysis skipped", run_label)
        return

    # One set of query resamples, shared by all signals.
    queries = steps["query_id"].unique()
    picks = np.random.default_rng(seed).integers(len(queries), size=(n_bootstrap, len(queries)))

    summary_rows: List[Dict[str, Any]] = []
    data: Dict[str, Dict[str, Any]] = {}
    for signal in SIGNALS:
        key = signal["key"]
        n_null = int(steps[key].isna().sum())
        df = steps[steps[key].notna()].reset_index(drop=True)
        res = evaluate(df, key, queries, picks, binary="productive", graded=GOLD_KEY)
        data[key] = {"df": df, "res": res}
        row = {
            "signal": key,
            "n_queries": df["query_id"].nunique(), "n_steps": len(df),
            "n_productive": int(df["productive"].sum()),
            "base_rate": _round(float(df["productive"].mean())) if len(df) else None,
            "mean_gold": _round(float(df[GOLD_KEY].mean())) if len(df) else None,
            "n_null_signal": n_null,
        }
        row.update(_measure_columns(res, ("auroc", "tau")))
        summary_rows.append(row)

    plot_distributions(steps, data, run_label, out_dir)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n== {run_label}  ({steps['query_id'].nunique()} queries, {len(steps)} steps, "
          f"{n_null_gold} steps without gold dropped)")
    print(summary[["signal", "n_steps", "n_productive", "auroc", "tau"]].to_string(index=False))


def analyze_final_state(
    samples: List[Tuple[str, List[Dict[str, Any]]]], correctness: Dict[str, Optional[int]],
    run_label: str, out_dir: Path, n_bootstrap: int, seed: int,
) -> None:
    """Final belief against correctness (AUROC) and Recall@N
    (tau-b): ``final_state.png``, ``final_state_summary.csv``."""
    queries_df = load_queries(samples, correctness)
    n_null_belief = int(queries_df["belief"].isna().sum())
    queries_df = queries_df[queries_df["belief"].notna()]
    # (target column, measure); correctness only when the run has it.
    targets = ([("correct", "auroc")] if correctness else []) + [("recall", "tau")]

    summary_rows: List[Dict[str, Any]] = []
    data: Dict[str, Dict[str, Any]] = {}
    for target, measure in targets:
        n_null = int(queries_df[target].isna().sum())
        df = queries_df[queries_df[target].notna()].reset_index(drop=True)
        if target == "correct":
            df = df.astype({"correct": int})
        queries = df["query_id"].unique()
        picks = np.random.default_rng(seed).integers(len(queries), size=(n_bootstrap, len(queries)))
        res = evaluate(df, "belief", queries, picks,
                       binary=target if measure == "auroc" else None, graded=target if measure == "tau" else None)
        data[target] = {"df": df, "res": res}
        row = {
            "target": "correctness" if target == "correct" else "Recall@N",
            "n_queries": len(df),
            "mean_target": _round(float(df[target].mean())) if len(df) else None,
            "mean_belief": _round(float(df["belief"].mean())) if len(df) else None,
            "n_null_target": n_null,
            "n_null_belief": n_null_belief,
            "measure": measure,
        }
        cols = _measure_columns(res, (measure,))
        row.update({"value": cols[measure], "ci_low": cols[f"{measure}_ci_low"], "ci_high": cols[f"{measure}_ci_high"]})
        summary_rows.append(row)

    plot_final_state(data, run_label, out_dir)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "final_state_summary.csv", index=False)
    print(f"\n-- final belief ({n_null_belief} queries without a belief state dropped)")
    print(summary[["target", "n_queries", "mean_target", "measure", "value", "ci_low", "ci_high"]].to_string(index=False))


def analyze_run(run_dir: Path, mode: str, n_bootstrap: int, seed: int) -> None:
    samples = read_belief(run_dir, mode)
    if not samples:
        logger.warning("%s: no belief file to analyze, skipped", run_dir)
        return
    run_label = "/".join(run_dir.resolve().parts[-3:])
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_steps(samples, run_label, out_dir, n_bootstrap, seed)
    analyze_final_state(samples, load_correctness(run_dir), run_label, out_dir, n_bootstrap, seed)
    print(f"-> {out_dir}")


def _parse_args() -> argparse.Namespace:
    """Same run-selecting arguments as experiments/inference/dra_inference.py.

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

    # ── Run selection (same as experiments/inference/dra_inference.py) ───────────────
    parser.add_argument("--policy-model", type=str, default="glm", choices=list(POLICY_MODEL_TO_LLM), help="Agent of the run; the LLM is selected automatically from the agent.")
    parser.add_argument("--dataset", type=str, default="browsecomp_plus", choices=list(DATASETS), help="Dataset of the run.")
    parser.add_argument("--subset", type=lambda v: None if v == "null" else v, default=None, help="Dataset subset/collection (unset or null = the dataset's default in layout.DATASET_SPECS). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical; browsecomp_plus: test; ragtime: unused.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever of the run.")
    parser.add_argument("--belief-mode", type=str, default="monitor", choices=["monitor", "inform"], help="Belief model mode of the run. 'monitor' (default): the signals never changed the trajectory. 'inform': the agent read them.")

    # ── Analysis ───────────────────────────────────────────────────────────
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Query-level bootstrap resamples for the AUROC and tau-b CIs.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")

    args, extras = parser.parse_known_args()

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    cli_subset = args.subset
    config = load_run_config(args.config)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset

    # ── Derive --llm-model from --policy-model ────────────────────────────-
    args.llm_model = POLICY_MODEL_TO_LLM[args.policy_model]

    if args.output is None:
        args.output = _OUTPUT_PREFIX
    return args


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """The run directory experiments/inference/dra_inference.py writes for *args*."""
    resolve_dataset_defaults(args)
    file_data_set = resolve_split_id(args.dataset, args.dataset_year, args.subset)
    run_name = build_run_name_for_pipeline(
        policy_model=args.policy_model, llm_model=args.llm_model, use_plan=args.use_plan,
    )
    dataset_dir = build_dataset_dir_name(args.dataset, file_data_set, args.query_key, args.retriever)
    belief_config = build_belief_config_name(belief_mode=args.belief_mode)
    return Path(args.output) / dataset_dir / run_name / belief_config


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = resolve_run_dir(args)
    if not (run_dir / "belief").is_dir():
        raise SystemExit(f"error: no belief/ folder in {run_dir}")
    analyze_run(run_dir, args.belief_mode, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()

# python analysis/signal_correlation.py
