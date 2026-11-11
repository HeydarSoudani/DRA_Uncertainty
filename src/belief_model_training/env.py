"""Phase 3 environment: belief chains of the student over whole episodes
(GRPO groups of chains).

An episode of the ``rl`` split (off-mode run) has fixed observations: the
documents of each search step and the agent's turn of each step and its
answer turn (the belief never changes what the agent does).  A chain is the
student's belief updater run over the whole episode from the episode's gold
criteria (the init call is not trained here): one call per step that has a
document not judged yet or an agent turn to read, then the final call on the
answer turn, each call's prompt built from the chain's own b_{t-1}
(``episodes.BeliefTracker``), as at inference.  ``run_chains`` draws
``group_size`` chains per episode (temperature 1 in training, greedy for
validation); the calls of all chains are generated together, one round per
call.  On a short-answer query the final call is the final scorer's (the
second task): it gets ``group_size`` replies on the chain's own b_{T-1}
(``Chain.scorer``, its own GRPO group), the first of them the chain's
s_T.  A reply that does not parse (or a prompt too long) leaves the call's
documents unjudged and the belief unchanged, as at inference.

``EpisodeSampler`` draws the episodes of a batch, balanced on the
correctness of the agent's final answer.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .episodes import BeliefTracker, DocumentStore, Episode, UpdaterCall
from .rollout import VLLMRolloutEngine


@dataclass
class ChainCall:
    """One updater call of a chain and the student's reply."""
    call: UpdaterCall
    prompt_ids: List[int]
    response_ids: List[int]
    # vLLM's log-probs of the reply tokens (compared with the trainer's).
    rollout_log_probs: List[float]
    text: Optional[str]
    finish_reason: Optional[str]
    # BeliefTracker.submit
    outcome: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chain:
    """The student's belief chain over one episode; ``group`` indexes the
    episode in the batch.  ``calls`` are its calls in order (the scorer's
    first reply last, on a short-answer query); ``scorer`` every reply to
    its scorer call (empty without one); ``beliefs`` is the belief after
    the last call (per gold criterion)."""
    episode: Episode
    group: int
    calls: List[ChainCall] = field(default_factory=list)
    scorer: List[ChainCall] = field(default_factory=list)
    beliefs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def update_calls(self) -> List[ChainCall]:
        """The chain's belief update calls (not the scorer's)."""
        return [cc for cc in self.calls if not cc.call.is_scorer]


@dataclass
class _Run:
    chain: Chain
    tracker: BeliefTracker
    next_t: int = 0
    finished: bool = False


def _next_call(run: _Run) -> Optional[UpdaterCall]:
    """The chain's next updater call (steps in order, then the final call),
    or None when the chain is over."""
    steps = run.chain.episode.steps
    while run.next_t < len(steps):
        t = run.next_t
        run.next_t += 1
        call = run.tracker.open_step(t)
        if call is not None:
            return call
    if not run.finished:
        run.finished = True
        return run.tracker.open_final()
    return None


def run_chains(episodes: List[Episode], store: DocumentStore, engine: VLLMRolloutEngine, *, group_size: int,
               temperature: float) -> List[Chain]:
    """*group_size* chains per episode from its gold criteria; the chains of
    an episode are contiguous and share its index as ``group``.  A call's
    reply budget is the inference updater's or scorer's
    (``BeliefTracker.max_tokens``, larger for a multi-member query).  A scorer call
    gets *group_size* replies."""
    runs = [_Run(Chain(ep, g), BeliefTracker(ep, store.docs(ep), ep.gold_criteria_list()))
            for g, ep in enumerate(episodes) for _ in range(group_size)]
    active = list(runs)
    while active:
        pending = []
        for run in active:
            call = _next_call(run)
            if call is not None:
                pending.append((run, call))
        if not pending:
            break
        for scorer, n in ((False, 1), (True, group_size)):
            part = [(run, call) for run, call in pending if call.is_scorer == scorer]
            if not part:
                continue
            gens = engine.generate([c.messages for _, c in part], temperature=temperature, n=n,
                                   max_new_tokens=[run.tracker.max_tokens(c) for run, c in part])
            for (run, call), g in zip(part, gens):
                replies = [ChainCall(call, gen.prompt_ids, gen.response_ids, gen.logprobs, gen.text,
                                     gen.finish_reason) for gen in g]
                replies[0].outcome = run.tracker.submit(replies[0].text)
                for cc in replies[1:]:
                    cc.outcome = run.tracker.score_outcome(call, cc.text)
                run.chain.calls.append(replies[0])
                if scorer:
                    run.chain.scorer = replies
        active = [run for run, _ in pending]
    for run in runs:
        run.chain.beliefs = run.tracker.signal.beliefs() or []
    return [run.chain for run in runs]


class EpisodeSampler:
    """Draws the episodes of a batch without replacement, the episodes with
    a correct and with a wrong final answer weighing the same in total
    (episodes not judged weigh as one of the larger class)."""

    def __init__(self, episodes: List[Episode], rng: random.Random) -> None:
        self.episodes = episodes
        n = {y: sum(ep.correct is y for ep in episodes) for y in (True, False)}
        big = max(n.values()) or 1
        w = np.array([big / n[ep.correct] if ep.correct is not None and n[ep.correct] else 1.0 for ep in episodes])
        self.p = w / w.sum()
        self.rng = np.random.default_rng(rng.randrange(2 ** 32))

    def sample(self, n: int, exclude: Optional[Set[int]] = None) -> List[Episode]:
        """*n* episodes, none of the indices *exclude* (those already drawn
        in the iteration); fewer when not enough are left."""
        p = self.p.copy()
        if exclude:
            p[list(exclude)] = 0.0
        left = int((p > 0).sum())
        n = min(n, left)
        if n <= 0:
            return []
        idx = self.rng.choice(len(self.episodes), size=n, replace=False, p=p / p.sum())
        if exclude is not None:
            exclude.update(int(i) for i in idx)
        return [self.episodes[i] for i in idx]
