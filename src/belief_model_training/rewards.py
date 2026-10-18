"""Phase 3 reward of one student belief update (teacher-forced step).

    R_t = f_t · (Σ_i w_i r_i / Σ_i w_i)   over the terms available at the step

    f_t   format gate: 1 when the reply was not cut off, parses, gives every
          listed criterion a valid status, names known criteria once, and
          every cited passage has spans that all occur in it; else 0
    r_st  what moves: F1 of the documents cited as support against the gold
          documents among the call's passages (``Episode.shown_gold``);
          only when either is non-empty
    r_sc  how far: 1 - (C_t - recall_so_far)^2
    r_ts  where it ends: 1 - (C_T - y)^2, on the episode's last updater
          call, when the correctness y of the final answer is known

C_t is the coverage after the reply (``scoring.coverage``).  The squared
errors are minimized by the expected target given the input, so they
reward calibrated coverage.  The terms are the per-sample forms of
``scoring.measures``.
"""

from typing import Any, Dict, Optional

from .config import RLConfig
from .env import StepPrompt
from .scoring import cited_support, coverage, support_f1

# Notes of BeliefState.update that mean an invalid reply.
_INVALID_NOTES = ("no valid status", "unknown criterion id", "duplicate update")


def format_ok(outcome: Dict[str, Any], finish_reason: Optional[str]) -> bool:
    if finish_reason != "stop" or not outcome.get("parsed"):
        return False
    for r in outcome["records"]:
        if any(n in (r.get("note") or "") for n in _INVALID_NOTES):
            return False
        for e in (r.get("support") or []) + (r.get("contradict") or []):
            if not e.get("span_verified") or not all(e["span_verified"]):
                return False
    return True


class Reward:
    def __init__(self, cfg: RLConfig) -> None:
        self.weights = {"st": cfg.reward_weight_st, "sc": cfg.reward_weight_sc, "ts": cfg.reward_weight_ts}

    def __call__(self, prompt: StepPrompt, outcome: Dict[str, Any], finish_reason: Optional[str]) -> Dict[str, Any]:
        """``{reward, format, st, sc, ts, coverage, cited}``; the terms are
        None when not available or when the format gate fails."""
        out: Dict[str, Any] = {"reward": 0.0, "format": 0.0, "st": None, "sc": None, "ts": None,
                               "coverage": None, "cited": set()}
        if not format_ok(outcome, finish_reason):
            return out
        item = prompt.item
        recall = item.episode.steps[item.t].recall_so_far
        c = coverage(outcome["statuses"])
        cited = cited_support(outcome["records"])
        terms = {
            "st": support_f1(cited, item.episode.shown_gold(item.t)),
            "sc": None if recall is None else 1.0 - (c - recall) ** 2,
            "ts": (1.0 - (c - float(item.episode.correct)) ** 2
                   if item.last and item.episode.correct is not None else None),
        }
        used = {k: v for k, v in terms.items() if v is not None and self.weights[k] > 0}
        total = sum(self.weights[k] for k in used)
        out.update(terms, format=1.0, coverage=c, cited=cited,
                   reward=sum(self.weights[k] * v for k, v in used.items()) / total if total else 1.0)
        return out
