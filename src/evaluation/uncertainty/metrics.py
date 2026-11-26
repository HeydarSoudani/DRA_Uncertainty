"""The ``uncertainty`` group: its metric names and its log lines."""

from typing import List, Optional

from ..shared.log import ci, get, n, val
from ..shared.result import Row

GROUP = "uncertainty"

#: The score of a judged query without a stated verbal confidence
#: (``verbal.py``): unknown, neither confident nor unconfident.
MISSING_CONFIDENCE = 0.5

# Scores (each has an AUROC and a Brier row).
VERBAL = "verbal"
RULE_BASED = "belief rule-based"
LLM_SCORE = "belief LLM score"

VERBAL_MISSING = f"verbal: missing (scored {MISSING_CONFIDENCE:g})"

# Report targets s_T is compared with (``report.py``).
QUERIES_SCORED = "queries scored"
COVERAGE = "nugget coverage"
COVERAGE_WEIGHTED = "nugget coverage (weighted)"


def auroc_row(score: str) -> str:
    """``"{score} AUROC"``."""
    return f"{score} AUROC"


def brier_row(score: str) -> str:
    """``"{score} Brier"``."""
    return f"{score} Brier"


def tau_vs(target: str) -> str:
    """``"s_T tau-b vs {target}"``."""
    return f"s_T tau-b vs {target}"


def brier_vs(target: str) -> str:
    """``"s_T Brier vs {target}"``."""
    return f"s_T Brier vs {target}"


def log_line(rows: List[Row]) -> Optional[str]:
    """The verbal confidence line of the generation; None without it."""
    if not get(rows, GROUP, auroc_row(VERBAL)):
        return None
    return (f"uncertainty  verbal AUROC {ci(rows, GROUP, auroc_row(VERBAL))}"
            f" Brier {val(rows, GROUP, brier_row(VERBAL))}"
            f" (n {n(rows, GROUP, auroc_row(VERBAL))},"
            f" missing {val(rows, GROUP, VERBAL_MISSING)})")


def model_log_line(rows: List[Row]) -> Optional[str]:
    """A belief model's line: the rule-based and LLM scores (QA), or s_T
    against nugget coverage (reports); None without either."""
    if get(rows, GROUP, auroc_row(LLM_SCORE)):
        return (f"uncertainty  rule-based AUROC {ci(rows, GROUP, auroc_row(RULE_BASED))}"
                f" Brier {val(rows, GROUP, brier_row(RULE_BASED))}"
                f" | LLM score AUROC {ci(rows, GROUP, auroc_row(LLM_SCORE))}"
                f" Brier {val(rows, GROUP, brier_row(LLM_SCORE))}"
                f" (n {n(rows, GROUP, auroc_row(LLM_SCORE))})")
    if get(rows, GROUP, tau_vs(COVERAGE)):
        return (f"uncertainty  s_T tau-b vs coverage {ci(rows, GROUP, tau_vs(COVERAGE))}"
                f" Brier {val(rows, GROUP, brier_vs(COVERAGE))}"
                f" (n {val(rows, GROUP, QUERIES_SCORED)})")
    return None
