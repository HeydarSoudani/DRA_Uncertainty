"""Phase 3 reward of one belief chain: the student's belief update calls
over a whole episode, scored per criterion at every step and on the final
belief (``scoring.chain_score``).

    R = w_format * format + w_assign * assign + w_cal * cal + w_final * r_final

    format   the share of the chain's calls that pass the gate
             (``scoring.format_ok``)
    assign   per criterion with gold, the F1 of its source assignment over
             the chain's calls (its gold documents and the turns that
             state its gold answer, from the observations); the mean over
             the criteria
    cal      per criterion, the mean over the episode's steps of
             1 - (p - y)^2 (p its status probability, y whether its
             selected candidate covers it correctly); the mean over the criteria
    r_final  1 - (s_T - y)^2: the certainty score of the final beliefs
             against the correctness y of the agent's final answer (0 when
             the final call fails the gate)

Per criterion over the steps, then over the criteria, so neither a long
episode nor a criterion with many updates weighs more.  A part a chain
has nothing to score on (no gold criterion, y unknown) is None: neutral in
its group (``trainer.group_advantages`` takes it as the group mean) and
left out of R.

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
from .scoring import chain_score

TERMS = ("format", "assign", "cal", "final")


class Reward:
    def __init__(self, cfg: RLConfig, gold: NuggetGold) -> None:
        self.weights = {"format": cfg.reward_weight_format, "assign": cfg.reward_weight_assign,
                        "cal": cfg.reward_weight_cal, "final": cfg.reward_weight_final}
        self.gold = gold

    def __call__(self, episode: Episode, calls: List[UpdaterCall], outcomes: List[Dict[str, Any]],
                 finish_reasons: List[Optional[str]], beliefs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The reward of one chain: its *calls* in order with their
        *outcomes* (``BeliefTracker.submit``) and finish reasons, and its
        final *beliefs*.  Returns ``{reward, terms, score}``
        (``score``: ``scoring.chain_score``)."""
        score = chain_score(episode, calls, outcomes, finish_reasons, self.gold, beliefs)
        terms = {"format": score["format"], "assign": score["assign"], "cal": score["cal"],
                 "final": score["final"]["r_final"] if score["final"] else None}
        return {
            "reward": sum(self.weights[k] * v for k, v in terms.items() if v is not None),
            "terms": terms,
            "score": score,
        }
