"""Phase 3 environments: where the student belief updater's samples come from.

``replay``: the student re-tracks the belief over the fixed observations of
phase 1 episodes (``episodes.BeliefTracker``; the agent ran in monitor mode,
so its trajectory does not depend on the belief).  Each updater call is one
sample, as in Agent-BRACE (one per belief generation).  The episodes of a
batch advance together, one updater call each per round, so every round is
one batched vLLM generation.  A reply that does not parse leaves the belief
unchanged and its documents unjudged, as at inference (no parse retry: the
reply is the sample).

``online``: the frozen agent reading the student's belief (inform mode).
TODO, with the rewards.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .episodes import BeliefTracker, DocumentStore, Episode
from .rollout import VLLMRolloutEngine


@dataclass
class BeliefSample:
    """One student updater call."""
    qid: str
    iteration: int
    messages: List[Dict[str, str]]
    prompt_ids: List[int]
    response_ids: List[int]
    old_log_probs: List[float]
    text: Optional[str]
    finish_reason: Optional[str]
    # From the tracker: parsed, criteria_delta, belief_before, belief_after, criteria_updates.
    outcome: Dict[str, Any] = field(default_factory=dict)


def replay_rollout(episodes: List[Episode], store: DocumentStore, engine: VLLMRolloutEngine, *,
                   temperature: float, max_new_tokens: int,
                   max_steps: Optional[int] = None) -> List[BeliefSample]:
    """Run the student over every episode; returns all its updater calls."""
    trackers = [BeliefTracker(ep, store.docs(ep)[:ep.replay_steps], max_steps) for ep in episodes]
    samples: List[BeliefSample] = []
    while True:
        calls = [(tr, tr.next_call()) for tr in trackers]
        calls = [(tr, c) for tr, c in calls if c is not None]
        if not calls:
            return samples
        gens = engine.generate([c.messages for _, c in calls], temperature=temperature,
                               max_new_tokens=max_new_tokens)
        for (tr, call), gen in zip(calls, gens):
            outcome = tr.submit(gen.text)
            samples.append(BeliefSample(
                qid=tr.episode.qid, iteration=call.iteration, messages=call.messages,
                prompt_ids=gen.prompt_ids, response_ids=gen.response_ids, old_log_probs=gen.logprobs,
                text=gen.text, finish_reason=gen.finish_reason, outcome=outcome,
            ))
