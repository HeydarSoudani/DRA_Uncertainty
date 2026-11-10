"""Phase 3 reward of one belief chain: the student's belief update calls
over a whole episode, scored per criterion at every step and on the final
belief (``scoring.chain_score``).

    R = w_format * format + w_assign * assign + w_cal * cal + w_final * r_final

    format   the share of the chain's belief update calls that pass the
             gate (``scoring.format_ok``)
    assign   per criterion with gold, the F1 of its source assignment over
             the chain's calls (its gold documents and the turns that
             state its gold answer, from the observations); the mean over
             the criteria
    cal      per criterion, the mean over the episode's steps of
             1 - (p - y)^2 (p its status probability, y whether its
             selected candidate covers it correctly); the mean over the criteria
    r_final  1 - (s_T - y)^2: s_T against the correctness y of the
             agent's final answer (0 when the final call fails the gate):
             the final scorer's on a short-answer query, the certainty of
             the final beliefs on a report (``scoring``)

Per criterion over the steps, then over the criteria, so neither a long
episode nor a criterion with many updates weighs more.  A part a chain
has nothing to score on (no gold criterion, y unknown) is None: neutral in
its group (``trainer.group_advantages`` takes it as the group mean) and
left out of R.

R is the logged reward and the validation measure.  The GRPO advantage is
computed per part (GDPO), each normalized within its group before the
weighted sum, in two streams.  The belief update calls of a chain take
the chain's advantage from ``chain_terms`` (format, assign, cal, and on a
report r_final), within the episode's group of chains.  On a short-answer
query r_final is the final scorer's alone: each of the ``group_size``
replies to the chain's scorer call (``env.Chain.scorer``, one b_{T-1})
takes its own advantage from its r_final (``scorer_terms``), within those
replies, so the update calls are not credited with the scorer's reply and
the scorer is not credited with the chain's belief.  The values of the
chains must be resolved in the gold first (``scoring.chain_values``,
``NuggetGold.resolve``).
"""

from typing import Any, Dict, List, Optional

from .config import RLConfig
from .episodes import Episode, UpdaterCall
from .nugget_gold import NuggetGold
from .scoring import chain_score, scorer_reward

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
        scorer = any(c.is_scorer for c in calls)
        return {
            "reward": sum(self.weights[k] * v for k, v in terms.items() if v is not None),
            "terms": terms,
            # The update calls' advantage: r_final of a scorer left out (its own stream).
            "chain_terms": {**terms, "final": None} if scorer else terms,
            "score": score,
        }

    def scorer_terms(self, episode: Episode, outcomes: List[Dict[str, Any]],
                     finish_reasons: List[Optional[str]]) -> List[Dict[str, Optional[float]]]:
        """The terms of each reply to one scorer call (``{"final": r_final}``)."""
        return [{"final": scorer_reward(o, f, episode.correct)} for o, f in zip(outcomes, finish_reasons)]
