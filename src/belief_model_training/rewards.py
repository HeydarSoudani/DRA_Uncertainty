"""Phase 3 reward of one belief chain: the student's updater calls over a
whole episode, scored on every step and on the final belief.

    R = w * sum_k r_step_k + r_final          r_step_k = f_k * (r_ret_k + r_reas_k)

    f_k        format gate of call k: 1 when the reply was not cut off,
               parses, gives every criterion it lists a valid status, names
               known criteria once, every cited passage has spans that all
               occur in it (``episodes.span_verified``), has a
               ``candidates`` list when the call shows an agent turn, and
               every candidate it writes is kept (its value occurs in the
               agent's turns, a valid status, not repeated, within the cap:
               ``BeliefState.revise`` notes nothing); else 0
    r_ret_k    retriever: 1 - (C_t - recall_so_far)^2, the coverage after
               the call against the share of the query's relevant documents
               seen so far (not for the final call, which has no passage)
    r_reas_k   reasoner: the answer candidates after the call against the
               gold intermediate answers (``scoring.reasoner_reward``): per
               gold criterion with candidates or whose gold answer the
               call's turn states, the mean 1 - (p - g)^2 over its
               candidates (p = (level + 2) / 6, g = the candidate is a gold
               answer, ``candidate_gold``), 0 when the turn states the gold
               answer and no candidate is one; the mean over those
               criteria, 1 when there is none (nothing to get wrong)
    r_final    f_T * (1 - (C_T - y)^2): the final confidence C_T = C^ret_T
               x R_T (``scoring.confidence``) after the chain's last call
               against the correctness y of the agent's final answer (a
               Brier score); 0 when y is unknown

w is ``rl.reward_weight_step``.  The squared errors are minimized by the
expected target given the input, so they reward calibrated coverage and
confidence.

R is the logged reward and the validation measure.  The GRPO advantage is
computed per term (``trainer.group_advantages``, GDPO) from ``terms``:
``ret`` (sum of f_k r_ret_k), ``reas`` (sum of f_k r_reas_k) and ``final``,
weighted w, w and 1, so each is normalized within the episode's group of
chains before they are added.  The candidates of the chains must be
resolved in the gold first (``candidate_values``, ``CandidateGold.resolve``).
"""

from typing import Any, Dict, List, Optional, Tuple

from .candidate_gold import CandidateGold
from .config import RLConfig
from .episodes import Episode, UpdaterCall, span_verified
from .scoring import call_row, cited_support, confidence, coverage, criteria_kinds, reasoner_items, reasoner_reward

# Notes of BeliefState.update that mean an invalid reply.
_INVALID_NOTES = ("no valid status", "unknown criterion id", "duplicate update")

TERMS = ("ret", "reas", "final")


def format_ok(call: UpdaterCall, outcome: Dict[str, Any], finish_reason: Optional[str]) -> bool:
    if finish_reason != "stop" or not outcome.get("parsed"):
        return False
    for r in outcome["criteria_updates"]:
        if any(n in (r.get("note") or "") for n in _INVALID_NOTES):
            return False
        if not all(span_verified(e) for e in (r.get("support") or []) + (r.get("contradict") or [])):
            return False
    if call.agent_turn is not None and not outcome.get("candidates_parsed"):
        return False
    return not any(r["note"] for r in outcome["candidate_updates"])


def candidate_values(episode: Episode, outcomes: List[Dict[str, Any]]) -> List[Tuple[Episode, str, str]]:
    """``(episode, criterion id, value)`` of every candidate after the calls
    with *outcomes*, for ``CandidateGold.resolve``."""
    ids = [c["id"] for c in episode.gold_criteria or []]
    return [(episode, cid, c["value"]) for o in outcomes if o.get("candidates")
            for cid, cands in zip(ids, o["candidates"]) for c in cands]


class Reward:
    def __init__(self, cfg: RLConfig, gold: CandidateGold) -> None:
        self.w = cfg.reward_weight_step
        self.weights = {"ret": self.w, "reas": self.w, "final": 1.0}
        self.gold = gold

    def __call__(self, episode: Episode, calls: List[UpdaterCall], outcomes: List[Dict[str, Any]],
                 finish_reasons: List[Optional[str]], statuses: List[str],
                 candidates: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """The reward of one chain: its *calls* in order with their
        *outcomes* (``BeliefTracker.submit``) and finish reasons, and the
        belief after the last call (*statuses*, *candidates*).  Returns
        ``{reward, terms, format, ret, reas, final, rows}``: ``ret`` and
        ``reas`` are the mean per-call values (over the calls where they
        apply, logged), ``rows`` the ``scoring.measures`` rows of the calls
        whose format passed."""
        ret_sum = reas_sum = 0.0
        ret_vals: List[float] = []
        reas_vals: List[float] = []
        fmt: List[float] = []
        rows: List[Dict[str, Any]] = []
        for call, outcome, finish in zip(calls, outcomes, finish_reasons):
            f = 1.0 if format_ok(call, outcome, finish) else 0.0
            fmt.append(f)
            if not f:
                continue
            reas = reasoner_reward(reasoner_items(episode, call.t, outcome["candidates"], self.gold))
            reas = 1.0 if reas is None else reas
            reas_sum += reas
            reas_vals.append(reas)
            if call.t is not None and episode.steps[call.t].recall_so_far is not None:
                ret = 1.0 - (coverage(outcome["belief_after"]) - episode.steps[call.t].recall_so_far) ** 2
                ret_sum += ret
                ret_vals.append(ret)
            rows.append(call_row(episode, call.t, outcome["belief_after"], outcome["candidates"],
                                 cited_support(outcome["criteria_updates"]),
                                 episode.gold_among([d["doc_id"] for d in call.kept]), self.gold, False))
        last_ok = bool(fmt) and fmt[-1] == 1.0
        final = 0.0
        if episode.correct is not None and last_ok:
            final = 1.0 - (confidence(coverage(statuses), candidates, criteria_kinds(episode))
                           - float(episode.correct)) ** 2
        if rows and last_ok:
            rows[-1]["final"] = True
        terms = {"ret": ret_sum, "reas": reas_sum, "final": final}
        return {
            "reward": self.w * (ret_sum + reas_sum) + final, "terms": terms,
            "format": sum(fmt) / len(fmt) if fmt else None,
            "ret": sum(ret_vals) / len(ret_vals) if ret_vals else None,
            "reas": sum(reas_vals) / len(reas_vals) if reas_vals else None,
            "final": final if episode.correct is not None else None,
            "rows": rows,
        }
