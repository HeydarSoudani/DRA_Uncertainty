"""Phase 3 reward of one belief chain: the student's belief update calls
over a whole episode, scored on every step and on the final belief.

    R = w_obs * obs + w_cons * cons + w_cal * cal + w_final * r_final

    obs, cons, cal   the mean over the chain's calls of f_k * r_k of each
                     part (``scoring.call_row``): r_obs valid observation
                     (F1 on the observations the gold decides), r_cons
                     valid conversion against the call's own observations
                     (share of consistency checks passed), r_cal
                     calibration of the rewritten nuggets (mean
                     1 - (p - g)^2); f_k the gate (``scoring.format_ok``):
                     a call that fails it scores 0 on every part; a call
                     with nothing to score on a part does not count in its
                     mean
    r_final          1 - (s_T - y)^2: the certainty score of the final
                     belief against the correctness y of the agent's final
                     answer (``scoring.final_row``; 0 when the final call
                     fails the gate)

The means over steps keep a long episode from weighing more than a short
one (the chains of a group share their episode, so their calls are the
same steps).  A part a chain has nothing to score on (no call with it, y
unknown) is None: neutral in its group (``trainer.group_advantages``
takes it as the group mean) and left out of R.

R is the logged reward and the validation measure.  The GRPO advantage is
computed per part (GDPO) from ``terms``, each normalized within the
episode's group of chains before the weighted sum.  The values of the
chains must be resolved in the gold first (``scoring.chain_values``,
``NuggetGold.resolve``).
"""

from typing import Any, Dict, List, Optional

from .config import RLConfig
from .episodes import Episode, UpdaterCall
from .nugget_gold import NuggetGold
from .scoring import call_row, final_row

TERMS = ("obs", "cons", "cal", "final")


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class Reward:
    def __init__(self, cfg: RLConfig, gold: NuggetGold) -> None:
        self.weights = {"obs": cfg.reward_weight_obs, "cons": cfg.reward_weight_cons,
                        "cal": cfg.reward_weight_cal, "final": cfg.reward_weight_final}
        self.gold = gold

    def __call__(self, episode: Episode, calls: List[UpdaterCall], outcomes: List[Dict[str, Any]],
                 finish_reasons: List[Optional[str]], nuggets: List[List[Dict[str, Any]]],
                 answer_label: Optional[str]) -> Dict[str, Any]:
        """The reward of one chain: its *calls* in order with their
        *outcomes* (``BeliefTracker.submit``) and finish reasons, and its
        final belief (*nuggets*, the answer turn's label).  Returns
        ``{reward, terms, format, rows, final}``."""
        rows = [call_row(episode, c, o, f, self.gold) for c, o, f in zip(calls, outcomes, finish_reasons)]
        final_ok = not rows or not rows[-1]["final_call"] or rows[-1]["format"]
        final = final_row(episode, nuggets, answer_label, final_ok)
        terms = {"obs": _mean([r["r_obs"] for r in rows]), "cons": _mean([r["r_cons"] for r in rows]),
                 "cal": _mean([r["r_cal"] for r in rows]), "final": final["r_final"]}
        return {
            "reward": sum(self.weights[k] * v for k, v in terms.items() if v is not None),
            "terms": terms,
            "format": sum(r["format"] for r in rows) / len(rows) if rows else None,
            "rows": rows, "final": final,
        }
