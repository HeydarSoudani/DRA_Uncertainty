"""Phase 3 reward of one student belief update.

    R_t = f_t · (Σ_i w_i r_i / Σ_i w_i)   over the terms available at the step

    f_t   format gate: 1 when the reply was not cut off, parses, gives every
          listed criterion a valid status, names known criteria once, and
          every cited passage has spans that all occur in it; else 0
    r_st  what moves: F1 of the documents cited as support against the gold
          documents among the call's passages (``StepPrompt.gold``); 1 when
          both are empty (nothing cited on a step without gold), 0 when only
          one is
    r_sc  how far: 1 - (C_t - recall_so_far)^2, when recall_so_far is known
    r_ts  where it ends: 1 - (C_T - y)^2, when the prompt is ``final`` and
          the correctness y of the final answer is known

C_t is the coverage after the reply (``scoring.coverage``).  The squared
errors are minimized by the expected target given the input, so they
reward calibrated coverage.  The terms are the per-sample forms of
``scoring.measures``.

R is the logged reward and the validation measure.  The GRPO advantage is
computed from the terms (``trainer.group_advantages``): ``terms`` holds
every available term, 0 when the format gate fails, so a reply with a bad
format ranks last on each term.
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

    def available(self, prompt: StepPrompt) -> list:
        """The terms of *prompt*'s replies (weight > 0)."""
        episode = prompt.item.episode
        have = {"st": True, "sc": episode.steps[prompt.item.t].recall_so_far is not None,
                "ts": prompt.final and episode.correct is not None}
        return [k for k, w in self.weights.items() if w > 0 and have[k]]

    def __call__(self, prompt: StepPrompt, outcome: Dict[str, Any], finish_reason: Optional[str]) -> Dict[str, Any]:
        """``{reward, format, st, sc, ts, terms, coverage, cited}``; ``st``,
        ``sc``, ``ts`` are None when not available or when the format gate
        fails, ``terms`` the available ones (0 on a failed gate)."""
        keys = self.available(prompt)
        out: Dict[str, Any] = {"reward": 0.0, "format": 0.0, "st": None, "sc": None, "ts": None,
                               "terms": {k: 0.0 for k in keys}, "coverage": None, "cited": set()}
        if not format_ok(outcome, finish_reason):
            return out
        episode = prompt.item.episode
        c = coverage(outcome["statuses"])
        cited = cited_support(outcome["records"])
        values = {
            "st": lambda: support_f1(cited, prompt.gold),
            "sc": lambda: 1.0 - (c - episode.steps[prompt.item.t].recall_so_far) ** 2,
            "ts": lambda: 1.0 - (c - float(episode.correct)) ** 2,
        }
        terms = {k: values[k]() for k in keys}
        total = sum(self.weights[k] for k in terms)
        out.update(terms, terms=terms, format=1.0, coverage=c, cited=cited,
                   reward=sum(self.weights[k] * v for k, v in terms.items()) / total if total else 1.0)
        return out
