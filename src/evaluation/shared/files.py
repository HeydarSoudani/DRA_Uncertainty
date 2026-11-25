"""The evaluation files of a run, all under ``{run_dir}/eval/``.

    metrics.md      header, one table (:data:`COLUMNS`, groups in :data:`GROUPS`
                    order), then the extra sections; parsed back by
                    :func:`read_metrics`, :func:`read_header`, :func:`read_sections`
    per_query.csv   one row per query, every per-query value of every group
                    (a belief model's columns prefixed ``{model}/``); parsed
                    back by :func:`read_per_query`
    generation/     the generation evaluators' own files (``accuracy.jsonl``,
                    or the Auto-ARGUE cache and ``scores.tsv``)

The layout is described in ``evaluation/README.md``.  Paths here are
relative to the run directory.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .result import Row, Section

#: Group order of the files and the log.
GROUPS = ("retrieval", "generation", "uncertainty", "belief", "trajectory")
#: Folder of every evaluation file in the run directory.
EVAL_DIR = "eval"
METRICS_FILE = f"{EVAL_DIR}/metrics.md"
PER_QUERY_FILE = f"{EVAL_DIR}/per_query.csv"
#: The generation evaluators' files (``accuracy.jsonl`` or the Auto-ARGUE cache).
GENERATION_EVAL_DIR = f"{EVAL_DIR}/generation"
ACCURACY_FILE = f"{GENERATION_EVAL_DIR}/accuracy.jsonl"
#: Columns of the ``metrics.md`` table.
COLUMNS = ("group", "belief_model", "metric", "value", "ci_low", "ci_high", "n")


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


# ---------------------------------------------------------------------------
# metrics.md
# ---------------------------------------------------------------------------

def write_metrics(path: Path, header: Dict[str, Any], rows: List[Row], sections: Sequence[Section] = ()) -> None:
    """Write ``metrics.md``: the *header* items as a list, the table (rows in
    :data:`GROUPS` order), then each of *sections* as a titled table."""
    order = {g: i for i, g in enumerate(GROUPS)}
    rows = sorted(rows, key=lambda r: order.get(r.group, len(GROUPS)))
    body = [[r.group, r.belief_model, r.metric, fmt(r.value), fmt(r.ci_low), fmt(r.ci_high), fmt(r.n)]
            for r in rows]
    lines = ["# Evaluation", ""] + [f"- {k}: {v}" for k, v in header.items()] + [""]
    lines += table(list(COLUMNS), body)
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

        rows.append(Row(cells[0], cells[2], num(cells[3]), num(cells[4]), num(cells[5]), num(cells[6]),
                        belief_model=cells[1]))
    return rows


def read_sections(path: Path) -> List[Section]:
    """The extra sections of a ``metrics.md`` (``## title`` and its table),
    as :func:`write_metrics` takes them."""
    sections: List[Section] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), [], []))
        elif sections and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= {"-"}:
                continue   # the separator line
            title, columns, body = sections[-1]
            if not columns:
                columns.extend(cells)
            else:
                body.append(cells)
    return sections


# ---------------------------------------------------------------------------
# per_query.csv
# ---------------------------------------------------------------------------

def read_per_query(path: Path) -> Dict[str, Dict[str, Any]]:
    """The rows of a ``per_query.csv``, ``{query_id: {column: cell}}`` (cells
    as written, empty cells kept so an empty column stays)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row.pop("query_id"): row for row in csv.DictReader(fh)}


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
