"""Collect the evaluation of many runs into one table.

Reads every run under ``--root`` that has a ``metrics.md``
(``evaluation.report``, written by ``dra_inference.py`` and its
``--eval-only``) with its ``run_config.json``, and writes one table: a row
per run (dataset, split, agent, belief mode, criteria source, finished and
unfinished queries), a column
per measure.

Usage::

    python analysis/collect_results.py --root run_outputs \\
        [--out results] [--group uncertainty belief] [--all] [--ci]

``--group`` keeps the measures of these groups only; by default the
headline measures (``HEADLINE``) are kept, ``--all`` keeps every measure of
the kept groups; ``--ci`` adds each measure's CI as ``[low, high]`` in its
cell (markdown only).  Outputs: ``{out}.md`` (aligned markdown table) and
``{out}.csv``.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.belief import read_run_config  # noqa: E402
from evaluation.report import GROUPS, METRICS_FILE, fmt, read_header, read_metrics, table  # noqa: E402

#: The headline measures, (group, metric), in column order.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("retrieval", "Recall@N"),
    ("generation", "accuracy"),
    ("generation", "nugget coverage"),
    ("generation", "nugget F1"),
    ("uncertainty", "s_T AUROC"),
    ("uncertainty", "s_T ECE"),
    ("uncertainty", "verbal AUROC"),
    ("uncertainty", "verbal ECE"),
    ("belief", "assigned"),
    ("belief", "rises"),
    ("belief", "correct_after"),
    ("belief", "correct_end"),
    ("belief", "rise AUROC (gold vs control)"),
    ("trajectory", "searches"),
    ("trajectory", "agent tokens"),
    ("trajectory", "belief tokens"),
)
RUN_COLUMNS = ("dataset", "split", "agent", "belief", "criteria", "queries", "unfinished")


def run_identity(run_dir: Path) -> Dict[str, str]:
    config = read_run_config(run_dir)
    data, agent, belief = config.get("dataset") or {}, config.get("agent") or {}, config.get("belief_model") or {}
    return {
        "dataset": data.get("dataset") or "?",
        "split": data.get("split") or "-",
        "agent": agent.get("policy_model") or "?",
        "belief": belief.get("mode") or "?",
        "criteria": belief.get("criteria_source_effective") or "-",
    }


def collect(root: Path, groups: List[str], all_metrics: bool,
            with_ci: bool) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, str]]]:
    """The columns and one row per run under *root*: the markdown cells (3
    digits, with the CI when *with_ci*) and the CSV cells (4 digits, no CI)."""
    md_rows, csv_rows, seen_metrics = [], [], []
    for path in sorted(root.rglob(METRICS_FILE)):
        run_dir = path.parent
        header = read_header(path)
        ident = {**run_identity(run_dir), "run": str(run_dir.relative_to(root)),
                 "queries": header.get("queries", ""), "unfinished": header.get("unfinished", "")}
        md_row, csv_row = dict(ident), dict(ident)
        for r in read_metrics(path):
            if r.group not in groups or (not all_metrics and (r.group, r.metric) not in HEADLINE):
                continue
            key = f"{r.group}: {r.metric}"
            if key not in seen_metrics:
                seen_metrics.append(key)
            cell = fmt(r.value, 3)
            if with_ci and r.ci_low is not None:
                cell += f" [{fmt(r.ci_low, 2)}, {fmt(r.ci_high, 2)}]"
            md_row[key] = cell
            csv_row[key] = fmt(r.value)
        md_rows.append(md_row)
        csv_rows.append(csv_row)
    if all_metrics:
        metric_cols = seen_metrics
    else:
        metric_cols = [f"{g}: {m}" for g, m in HEADLINE if f"{g}: {m}" in seen_metrics]
    return list(RUN_COLUMNS) + metric_cols + ["run"], md_rows, csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Folder searched for runs (metrics.md).")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output path without extension.")
    parser.add_argument("--group", nargs="+", default=list(GROUPS), choices=list(GROUPS), help="Groups kept.")
    parser.add_argument("--all", action="store_true", help="Every measure of the kept groups, not the headline.")
    parser.add_argument("--ci", action="store_true", help="Add the CIs to the markdown cells.")
    args = parser.parse_args()

    columns, md_rows, csv_rows = collect(args.root, args.group, args.all, args.ci)
    if not md_rows:
        raise SystemExit(f"no {METRICS_FILE} under {args.root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = table(columns, [[row.get(c, "") for c in columns] for row in md_rows])
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"{len(md_rows)} runs -> {args.out.with_suffix('.md')} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
