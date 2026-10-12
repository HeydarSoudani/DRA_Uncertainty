"""Correlation of the uncertainty-estimator signals with the step gold.

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

Signals (step record fields of ``uncertainty/{query_id}.jsonl``):

- ``doc_novelty``           nu^D
- ``criteria_delta``        Delta^D
- ``query_novelty``         nu^q

Steps with a null gold (no qrels for the query, no documents) are dropped;
per signal, steps where the signal is null are dropped and counted.  Only
schema_version >= 10 files are read (the criteria semantics changed up to
10).  The default mode is ``monitor``: in ``inform`` mode the agent reads
the signals, which changes the trajectory.

The run is selected with the same arguments as ``experiments/dra_inference.py``
(``--config``, ``--agentic-model``, ``--dataset``, ``--subset``, ``--retriever``,
``--uncertainty-estimator-mode`` and any YAML key as ``--flag`` override), and
its directory is built the same way:
``{output}/{dataset}_{split}[_{query_key}]_{retriever}/{run_name}/{ue_config}/``.

Output in ``<run_dir>/analysis/signal_correlation/``:

- ``distributions.png``: one panel per signal, the distribution of the
  signal on productive (green) and unproductive (red) steps, each class
  normalized to its own number of steps; the AUROC and tau-b are in the
  title.  The axes are fixed (see ``SIGNALS``) so runs compare directly.
- ``summary.csv``: per signal, the counts and the AUROC and tau-b with their
  CIs.

Usage::

    python analysis/signal_correlation.py --dataset browsecomp_plus --subset test \\
        --agentic-model uncertainty_aware --uncertainty-estimator-mode monitor
"""

import argparse
import json
import logging
import math
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
from utils.config import AGENTIC_MODEL_ALIAS, AGENTIC_MODEL_TO_LLM
from utils.io_utils import build_dataset_dir_name, build_run_name_for_pipeline, build_uncertainty_config_name

logger = logging.getLogger(__name__)

_OUTPUT_PREFIX = str(OUTPUT_ROOT)
_CONFIG_DEFAULT = str(_REPO_ROOT / "experiments" / "configs" / "dra_inference.yaml")

MIN_SCHEMA_VERSION = 10
OUT_SUBDIR = Path("analysis") / "signal_correlation"
NUM_CONTINUOUS_BINS = 20
GOLD_KEY = "new_item_precision"

# ``range``: the fixed x range of the panel.  ``discrete``: one bar per
# integer in the range, values outside it joining the end bars; otherwise
# ``NUM_CONTINUOUS_BINS`` equal-width bins over the range.
SIGNALS = [
    {"key": "doc_novelty", "name": "Doc novelty (nu^D)", "discrete": False, "range": (0.0, 1.0)},
    {"key": "criteria_delta", "name": "Criteria coverage change (Delta^D)", "discrete": True, "range": (-3, 5)},
    {"key": "query_novelty", "name": "Query novelty (nu^q)", "discrete": False, "range": (0.0, 1.0)},
]

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

def load_steps(run_dir: Path, mode: str) -> pd.DataFrame:
    """One row per step of the run's uncertainty files, with both golds."""
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
        for step in lines[1:]:
            if step.get("record") != "step":
                continue
            num_new_relevant = step.get("num_new_relevant")
            row = {
                "query_id": str(meta["query_id"]),
                "productive": None if num_new_relevant is None else int(num_new_relevant > 0),
                GOLD_KEY: step.get(GOLD_KEY),
            }
            row.update({s["key"]: step.get(s["key"]) for s in SIGNALS})
            rows.append(row)
    if skipped_schema:
        logger.warning("%s: %d files with schema_version < %d skipped", run_dir, skipped_schema, MIN_SCHEMA_VERSION)
    if skipped_mode:
        logger.warning("%s: %d files with mode other than %s skipped", run_dir, skipped_mode, mode)
    return pd.DataFrame(rows, columns=["query_id", "productive", GOLD_KEY] + [s["key"] for s in SIGNALS])


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


def evaluate(df: pd.DataFrame, key: str, queries: np.ndarray, picks: np.ndarray) -> Dict[str, Any]:
    """AUROC (for ``df.productive``) and tau-b (for ``df.new_item_precision``)
    of ``df[key]`` with 95% CIs.

    Each row of *picks* is one resample: indices into *queries*, drawn with
    replacement, whose steps are all kept.
    """
    y = df["productive"].to_numpy(int)
    g = df[GOLD_KEY].to_numpy(float)
    s = df[key].to_numpy(float)
    measures = {"auroc": lambda idx: _auroc(y[idx], s[idx]), "tau": lambda idx: _tau(s[idx], g[idx])}

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


def _distribution_panel(ax, df: pd.DataFrame, signal: Dict[str, Any], res: Dict[str, Any]) -> None:
    """Productive and unproductive steps per signal value, each class
    normalized to its own number of steps; AUROC and tau-b in the title."""
    _style(ax)
    ax.set_title(f"{signal['name']}\n{_headline('AUROC', res['auroc'], res['auroc_ci'])}  |  "
                 f"{_headline('tau-b', res['tau'], res['tau_ci'])}",
                 fontsize=9, color=TEXT_PRIMARY, loc="left")
    lo, hi = signal["range"]
    pad = 0.5 if signal["discrete"] else 0.0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0.0, 1.0)
    if df.empty:
        ax.text(0.5, 0.5, "no steps", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
        return

    positions, tick_labels, width, index = _signal_axis(df[signal["key"]].to_numpy(float), signal)
    y = df["productive"].to_numpy(int)
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
    ax.set_ylabel("fraction of steps in class", fontsize=9, color=TEXT_SECONDARY)


def plot_distributions(
    steps: pd.DataFrame, data: Dict[str, Dict[str, Any]], run_label: str, out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(5.5 * len(SIGNALS), 4.8), facecolor=SURFACE)
    for ax, signal in zip(np.atleast_1d(axes), SIGNALS):
        _distribution_panel(ax, data[signal["key"]]["df"], signal, data[signal["key"]]["res"])
    # The class counts are the same in every panel (the steps with a gold);
    # a panel's own count can be lower only by its null signal values.
    n_pos, n_neg = int(steps["productive"].sum()), int((steps["productive"] == 0).sum())
    handles = [Patch(facecolor=color, alpha=0.45, edgecolor=color, label=label)
               for color, label in ((COLOR_POS, f"productive (n={n_pos})"),
                                    (COLOR_NEG, f"unproductive (n={n_neg})"))]
    legend = fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.suptitle(run_label, fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "distributions.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


def analyze_run(run_dir: Path, mode: str, n_bootstrap: int, seed: int) -> Optional[pd.DataFrame]:
    steps = load_steps(run_dir, mode)
    has_gold = steps["productive"].notna() & steps[GOLD_KEY].notna()
    n_null_gold = int((~has_gold).sum())
    steps = steps[has_gold].astype({"productive": int})
    if steps.empty:
        logger.warning("%s: no step with a gold label, skipped", run_dir)
        return None

    run_label = "/".join(run_dir.resolve().parts[-3:])
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # One set of query resamples, shared by all signals.
    queries = steps["query_id"].unique()
    picks = np.random.default_rng(seed).integers(len(queries), size=(n_bootstrap, len(queries)))

    summary_rows: List[Dict[str, Any]] = []
    data: Dict[str, Dict[str, Any]] = {}
    for signal in SIGNALS:
        key = signal["key"]
        n_null = int(steps[key].isna().sum())
        df = steps[steps[key].notna()].reset_index(drop=True)
        res = evaluate(df, key, queries, picks)
        data[key] = {"df": df, "res": res}
        row = {
            "signal": key,
            "n_queries": df["query_id"].nunique(), "n_steps": len(df),
            "n_productive": int(df["productive"].sum()),
            "base_rate": _round(float(df["productive"].mean())) if len(df) else None,
            "mean_gold": _round(float(df[GOLD_KEY].mean())) if len(df) else None,
            "n_null_signal": n_null,
        }
        for name in ("auroc", "tau"):
            row[name] = _round(res[name])
            row[f"{name}_ci_low"] = _round(res[f"{name}_ci"][0])
            row[f"{name}_ci_high"] = _round(res[f"{name}_ci"][1])
        summary_rows.append(row)

    plot_distributions(steps, data, run_label, out_dir)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n== {run_label}  ({steps['query_id'].nunique()} queries, {len(steps)} steps, "
          f"{n_null_gold} steps without gold dropped)")
    print(summary[["signal", "n_steps", "n_productive", "auroc", "tau"]].to_string(index=False))
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
    parser.add_argument("--agentic-model", type=str, default="glm", choices=list(AGENTIC_MODEL_TO_LLM), help="Agent of the run; the LLM is selected automatically from the agent.")
    parser.add_argument("--dataset", type=str, default="browsecomp_plus", choices=list(DATASETS), help="Dataset of the run.")
    parser.add_argument("--subset", type=lambda v: None if v == "null" else v, default=None, help="Dataset subset/collection (unset or null = the dataset's default in layout.DATASET_SPECS). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical; browsecomp_plus: test; ragtime: unused.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever of the run.")
    parser.add_argument("--uncertainty-estimator-mode", type=str, default="monitor", choices=["monitor", "inform"], help="Uncertainty estimator mode of the run. 'monitor' (default): the signals never changed the trajectory. 'inform': the agent read them.")

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

# python analysis/signal_correlation.py
