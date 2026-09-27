"""Core data contracts shared across the whole training pipeline.

These dataclasses are the *spine* every service agrees on:

    PromptRecord   — one training question (+ gold signals for reward)
    Turn           — one policy generation: the context it was generated from
                     and the generated tokens, as one training sample
    Trajectory     — a full multi-turn rollout for one prompt
    RewardBreakdown— the reward's verdict for a trajectory

Per-turn samples.  The agents keep the whole run in the prompt they send each
turn (for the uncertainty-aware agent: one growing user message), so turn t+1's context
is not turn t's context followed by its output.  Each turn is therefore its own
training sample, ``prompt_token_ids ++ gen_token_ids``, and every turn of a
trajectory gets the trajectory's advantage.  Retrieved passages and any other
injected observation live inside later turns' prompts, so they are masked by
construction.

Two invariants (enforced in ``rollout.masking``):
    * ``len(token_ids) == len(loss_mask)`` for every Turn.
    * ``loss_mask`` is 0 on every prompt token and 1 on every generated token;
      only the policy's own generated tokens are trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt (input to a rollout)
# ---------------------------------------------------------------------------

@dataclass
class PromptRecord:
    """A single training prompt plus whatever gold signals reward may need.

    ``answers`` / ``qrels`` / ``criteria`` are optional so the same record type
    works for RL (needs reward signals) and for smoke tests (needs none).
    """

    id: str
    question: str
    answers: Optional[List[str]] = None            # gold answers -> outcome reward
    qrels: Optional[Dict[str, int]] = None         # {doc_id: relevance} -> marginal-recall process reward
    criteria: Optional[List[str]] = None           # info-need criteria -> coverage process reward
    meta: Dict[str, Any] = field(default_factory=dict)   # e.g. {"dataset": "trqa"}


# ---------------------------------------------------------------------------
# Turn + Trajectory (output of a rollout)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One policy generation and what the environment made of it.

    ``prompt_token_ids`` are the tokens of the context the policy saw (masked);
    ``gen_*`` are the policy's own tokens (trained on).  ``messages`` is the
    same context as chat messages, kept for SFT export and inspection.
    """

    index: int
    text: str = ""                                 # generated text, as returned by the policy
    action_kind: str = "search"                    # env's reading: "search" | "answer" | "retry" | "stop"
    query: Optional[str] = None                    # the search query, if action_kind == "search"
    valid: bool = True                             # False when the env rejected the turn's format

    messages: List[Dict[str, str]] = field(default_factory=list)
    prompt_token_ids: List[int] = field(default_factory=list)
    gen_token_ids: List[int] = field(default_factory=list)
    gen_logprobs: Optional[List[float]] = None     # per-token logprob under the behavior policy
    finish_reason: str = "stop"

    obs_text: str = ""                             # what the env appended after this turn (logs only)
    obs_doc_ids: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)   # env-specific per-turn data

    @property
    def token_ids(self) -> List[int]:
        return list(self.prompt_token_ids) + list(self.gen_token_ids)

    @property
    def loss_mask(self) -> List[int]:
        return [0] * len(self.prompt_token_ids) + [1] * len(self.gen_token_ids)


@dataclass
class RewardBreakdown:
    """Reward verdict for one trajectory.

    ``per_turn`` has one entry per *search* turn (process credit); ``outcome`` is
    the terminal answer reward; ``total`` is the combined scalar the trainer uses
    as the trajectory return before group-normalization.
    """

    outcome: float = 0.0
    per_turn: List[float] = field(default_factory=list)
    total: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A full multi-turn rollout for one prompt."""

    prompt_id: str
    group_id: str                                  # rollouts sharing a prompt+version form a GRPO group
    policy_version: int                            # weight version that GENERATED this trajectory
    turns: List[Turn] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = "answered"                  # set by the env, e.g. "answered" | "max_turns" | "format_failure"
    meta: Dict[str, Any] = field(default_factory=dict)   # env summary of the run

    reward: Optional[RewardBreakdown] = None       # filled by the reward model
    advantage: Optional[float] = None              # trajectory-level GRPO advantage (filled by trainer)

    # ---- convenience views for reward / trainer ----
    @property
    def num_search_turns(self) -> int:
        return sum(1 for t in self.turns if t.action_kind == "search")

    @property
    def answered(self) -> bool:
        return self.stop_reason == "answered"

    def samples(self) -> List[Turn]:
        """The training samples of this trajectory: one per generation."""
        return list(self.turns)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
