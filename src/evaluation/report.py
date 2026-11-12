"""The evaluation files of a run and its terminal log.

- ``metrics.md``: a header (run, dataset, agent, belief mode, queries),
  one aligned markdown table, one row per measure: ``group | metric |
  value | ci_low | ci_high | n`` (groups in order: retrieval, generation,
  uncertainty, belief, trajectory), then any extra section (a titled
  table, e.g. the belief status calibration).  Read in the editor as is, or
  in the markdown preview; the main table is parsed back by
  ``read_metrics`` (the cross-run collector, ``analysis/collect_results.py``).
- ``per_query.csv``: one row per query, every per-query value of every
  group (the aggregates of ``metrics.md`` are computed from them).
- the log: one line per group with the headline measures (``log_lines``).
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Group order of the files and the log.
GROUPS = ("retrieval", "generation", "uncertainty", "belief", "trajectory")
METRICS_FILE = "metrics.md"
PER_QUERY_FILE = "per_query.csv"
COLUMNS = ("group", "metric", "value", "ci_low", "ci_high", "n")


@dataclass
class Row:
    group: str
    metric: str
    value: Any
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None


def fmt(value: Any, digits: int = 4) -> str:
    """A cell: "" for None, "n/a" for NaN, ints as ints, floats rounded."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int,)) or (isinstance(value, float) and value.is_integer() and abs(value) >= 10):
        return str(int(value))
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def table(header: List[str], body: List[List[str]]) -> List[str]:
    """The lines of an aligned markdown table."""
    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
              for i in range(len(header))]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    return [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in body]


#: The note under the main table when a row is reported per criterion.
CRITERION_NOTE = ("A row with (c) is the mean over every criterion (n criteria); the same row without it is the "
                  "mean over the queries of each query's mean over its criteria (CI over queries, n queries).")


def write_metrics(path: Path, header: Dict[str, Any], rows: List[Row],
                  sections: Sequence[Tuple[str, List[str], List[List[str]]]] = ()) -> None:
    """Write ``metrics.md``: the *header* items as a list, the table, then
    each of *sections* (``(title, columns, cell rows)``) as a titled table."""
    order = {g: i for i, g in enumerate(GROUPS)}
    rows = sorted(rows, key=lambda r: order.get(r.group, len(GROUPS)))
    body = [[r.group, r.metric, fmt(r.value), fmt(r.ci_low), fmt(r.ci_high), fmt(r.n)] for r in rows]
    lines = ["# Evaluation", ""] + [f"- {k}: {v}" for k, v in header.items()] + [""]
    lines += table(list(COLUMNS), body)
    if any(r.metric.endswith(" (c)") for r in rows):
        lines += ["", CRITERION_NOTE]
    for title, columns, cells in sections:
        lines += ["", f"## {title}", ""] + table(columns, cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_header(path: Path) -> Dict[str, str]:
    """The header items of a ``metrics.md`` (``- key: value`` lines)."""
    out: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            break
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            out[key] = value
    return out


def read_metrics(path: Path) -> List[Row]:
    """The rows of a ``metrics.md`` (counts as ints, other values as floats,
    "n/a" as NaN, empty cells None)."""
    rows: List[Row] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            break   # the extra sections
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(COLUMNS) or cells[0] in ("group", "") or set(cells[0]) <= {"-"}:
            continue

        def num(cell: str):
            if cell == "":
                return None
            try:
                return int(cell) if cell.lstrip("-").isdigit() else float(cell)
            except ValueError:
                return math.nan if cell == "n/a" else cell

        rows.append(Row(cells[0], cells[1], num(cells[2]), num(cells[3]), num(cells[4]), num(cells[5])))
    return rows


def write_per_query(path: Path, per_query: Dict[str, Dict[str, Any]]) -> None:
    """Write ``per_query.csv``: one row per query, columns in first-seen order."""
    columns: List[str] = []
    for rec in per_query.values():
        columns += [k for k in rec if k not in columns]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id"] + columns)
        for qid in sorted(per_query):
            rec = per_query[qid]
            writer.writerow([qid] + [fmt(rec.get(c)) for c in columns])


def _get(rows: List[Row], group: str, metric: str) -> Optional[Row]:
    return next((r for r in rows if r.group == group and r.metric == metric), None)


def _val(rows: List[Row], group: str, metric: str, digits: int = 3) -> str:
    r = _get(rows, group, metric)
    return "n/a" if r is None else fmt(r.value, digits)


def _n(rows: List[Row], group: str, metric: str) -> str:
    r = _get(rows, group, metric)
    return "n/a" if r is None or r.n is None else str(r.n)


def _ci(rows: List[Row], group: str, metric: str) -> str:
    r = _get(rows, group, metric)
    if r is None:
        return "n/a"
    ci = f" [{fmt(r.ci_low, 2)},{fmt(r.ci_high, 2)}]" if r.ci_low is not None else ""
    return fmt(r.value, 3) + ci


def log_lines(title: str, rows: List[Row]) -> List[str]:
    """The terminal block: one line per group present in *rows*."""
    groups = {r.group for r in rows}
    out = [title]
    out.append(f"retrieval    Recall@N {_val(rows, 'retrieval', 'Recall@N')}"
               f" | Precision@N {_val(rows, 'retrieval', 'Precision@N')}"
               f" (N {_val(rows, 'retrieval', 'N (docs seen)', 1)})")
    if _get(rows, "generation", "accuracy"):
        out.append(f"generation   accuracy {_val(rows, 'generation', 'accuracy')}"
                   f" ({_val(rows, 'generation', 'correct')}/{_val(rows, 'generation', 'evaluated')})")
    elif _get(rows, "generation", "nugget coverage"):
        out.append(f"generation   nugget coverage {_val(rows, 'generation', 'nugget coverage')}"
                   f" | F1 {_val(rows, 'generation', 'nugget F1')}"
                   f" ({_val(rows, 'generation', 'evaluated')} reports)")
    if _get(rows, "uncertainty", "s_T AUROC"):
        line = (f"uncertainty  s_T AUROC {_ci(rows, 'uncertainty', 's_T AUROC')}"
                f" Brier {_val(rows, 'uncertainty', 's_T Brier')}")
        if _get(rows, "uncertainty", "verbal AUROC"):
            line += (f" | verbal AUROC {_ci(rows, 'uncertainty', 'verbal AUROC')}"
                     f" Brier {_val(rows, 'uncertainty', 'verbal Brier')}")
        out.append(line + f" (n {_val(rows, 'uncertainty', 'queries scored')})")
    elif _get(rows, "uncertainty", "s_T tau-b vs nugget coverage"):
        out.append(f"uncertainty  s_T tau-b vs coverage {_ci(rows, 'uncertainty', 's_T tau-b vs nugget coverage')}"
                   f" Brier {_val(rows, 'uncertainty', 's_T Brier vs nugget coverage')}"
                   f" (n {_val(rows, 'uncertainty', 'queries scored')})")
    if "belief" in groups:
        out.append(f"belief       gold change {_val(rows, 'belief', 'gold event: gold candidate change')}"
                   f" | control change {_val(rows, 'belief', 'control event: gold candidate change')}"
                   f" | recall {_val(rows, 'belief', 'criterion recall')}"
                   f" | reached {_val(rows, 'belief', 'reached answer')}"
                   f" (gold seen {_val(rows, 'belief', 'reached answer, gold seen')})"
                   f" | status Brier {_val(rows, 'belief', 'status Brier')}"
                   f" (n {_n(rows, 'belief', 'reached answer')})")
    line = (f"trajectory   searches {_val(rows, 'trajectory', 'searches', 1)}"
            f" | agent tok {_val(rows, 'trajectory', 'agent tokens', 0)}")
    if _get(rows, "trajectory", "belief tokens"):
        line += f" | belief tok {_val(rows, 'trajectory', 'belief tokens', 0)}"
    out.append(line + f" | forced {_val(rows, 'trajectory', 'forced answers')}")
    return out
