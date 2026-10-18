"""Phase 3 environment: teacher-forced belief update steps (GRPO groups).

A step item is one updater call of a phase 1 episode (``rl`` or ``val``
use): a step of the matched replay where the teacher was called and
replied.  Its prompt is built from the teacher's belief b_{t-1}: the
teacher's replies are replayed up to the step (``episodes.BeliefTracker``;
the agent ran in monitor mode, so the documents do not depend on the
belief).  The student writes ``group_size`` replies to the same prompt; each
is applied to a copy of b_{t-1}, which gives its statuses and cited
support for the reward.  The init call is not trained here (the criteria
stay as learned in SFT).

``StepSampler`` draws the step items of a batch: a share of steps with new
gold documents, at most a share from episodes without gold, the rest from
the other steps; within each pool the last calls of episodes are balanced
on the correctness of the final answer.
"""

import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .episodes import BeliefTracker, DocumentStore, Episode, UpdaterCall
from .rollout import VLLMRolloutEngine


@dataclass
class StepItem:
    """One updater call of an episode: ``t`` indexes ``episode.steps``."""
    episode: Episode
    t: int
    # The call shows gold documents (Episode.shown_gold) / is the episode's last updater call
    # (only when the whole episode replayed).
    has_gold: bool
    last: bool


@dataclass
class StepPrompt:
    """The student's prompt of a step item, and b_{t-1} (``signal``) to
    apply its replies to."""
    item: StepItem
    call: UpdaterCall
    signal: Any


@dataclass
class StudentSample:
    """One student reply; ``group`` indexes the prompt of the batch."""
    prompt: StepPrompt
    group: int
    prompt_ids: List[int]
    response_ids: List[int]
    # vLLM's log-probs of the reply tokens (compared with the trainer's).
    rollout_log_probs: List[float]
    text: Optional[str]
    finish_reason: Optional[str]
    # apply_reply: parsed, then records and statuses.
    outcome: Dict[str, Any] = field(default_factory=dict)


def step_items(episodes: List[Episode]) -> List[StepItem]:
    """The updater calls of *episodes* within their matched replay, the
    teacher's failed calls left out."""
    items: List[StepItem] = []
    for ep in episodes:
        calls = [t for t, s in enumerate(ep.steps[:ep.replay_steps or 0]) if s.teacher_reply is not None]
        complete = ep.replay_steps == len(ep.steps)
        items += [StepItem(ep, t, bool(ep.shown_gold(t)), complete and t == calls[-1]) for t in calls]
    return items


def build_prompts(items: List[StepItem], store: DocumentStore) -> List[StepPrompt]:
    """The prompts of *items*, in order; each episode is replayed once, up
    to its last item.  An item whose step needs no call in the replay is
    left out (not expected within the matched replay)."""
    by_episode: Dict[int, List[int]] = defaultdict(list)
    for i, item in enumerate(items):
        by_episode[id(item.episode)].append(i)
    out: List[Optional[StepPrompt]] = [None] * len(items)
    for idxs in by_episode.values():
        ep = items[idxs[0]].episode
        wanted: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            wanted[items[i].t].append(i)
        tracker = BeliefTracker(ep, store.docs(ep)[:ep.replay_steps])
        for t in range(max(wanted) + 1):
            call = tracker.open_step(t)
            if call is None:
                continue
            for i in wanted.get(t, []):
                out[i] = StepPrompt(items[i], call, _snapshot(tracker.signal))
            tracker.submit(ep.steps[t].teacher_reply)
    return [p for p in out if p is not None]


def _snapshot(signal: Any) -> Any:
    """A copy of the belief signal that shares its updater (prompt and
    parse only)."""
    return copy.deepcopy(signal, memo={id(signal.updater): signal.updater})


def apply_reply(prompt: StepPrompt, raw: Optional[str]) -> Dict[str, Any]:
    """The reply *raw* applied to a copy of b_{t-1}: ``{parsed, records,
    statuses}`` (``records`` as ``BeliefState.update``)."""
    signal = _snapshot(prompt.signal)
    call = prompt.call
    updates = signal.updater.parse(raw, call.kept, call.iteration) if raw is not None else None
    if updates is None:
        return {"parsed": False, "records": [], "statuses": signal.statuses()}
    _, records = signal.apply(call.docs, updates, raw)
    return {"parsed": True, "records": records, "statuses": signal.statuses()}


def rollout_groups(prompts: List[StepPrompt], engine: VLLMRolloutEngine, *, group_size: int,
                   temperature: float, max_new_tokens: int) -> List[StudentSample]:
    """*group_size* student replies per prompt, in one batched generation;
    the samples of a group are contiguous."""
    groups = engine.generate([p.call.messages for p in prompts], temperature=temperature,
                             max_new_tokens=max_new_tokens, n=group_size)
    return [
        StudentSample(
            prompt=prompt, group=g, prompt_ids=gen.prompt_ids, response_ids=gen.response_ids,
            rollout_log_probs=gen.logprobs, text=gen.text, finish_reason=gen.finish_reason,
            outcome=apply_reply(prompt, gen.text),
        )
        for g, (prompt, gens) in enumerate(zip(prompts, groups)) for gen in gens
    ]


class StepSampler:
    """Draws the step items of a batch (with replacement): ``gold_share``
    from steps with new gold, ``no_gold_share`` at most (and at most their
    natural share) from episodes without gold, the rest from the other
    steps of episodes with gold."""

    def __init__(self, items: List[StepItem], gold_share: float, no_gold_share: float, rng: random.Random) -> None:
        gold = [i for i in items if i.has_gold]
        no_gold = [i for i in items if not i.episode.has_gold]
        other = [i for i in items if not i.has_gold and i.episode.has_gold]
        no_gold_share = min(no_gold_share, len(no_gold) / max(len(items), 1))
        pools = [(gold, gold_share), (no_gold, no_gold_share), (other, max(1.0 - gold_share - no_gold_share, 0.0))]
        pools = [(p, s) for p, s in pools if p and s > 0]
        total = sum(s for _, s in pools)
        self.pools = [(p, s / total, _balance_weights(p)) for p, s in pools]
        self.rng = rng

    def sample(self, n: int) -> List[StepItem]:
        counts = [int(round(n * s)) for _, s, _ in self.pools]
        counts[-1] += n - sum(counts)
        out: List[StepItem] = []
        for (pool, _, weights), c in zip(self.pools, counts):
            out += self.rng.choices(pool, weights=weights, k=max(c, 0))
        return out


def _balance_weights(pool: List[StepItem]) -> List[float]:
    """Sampling weights that give the last calls with a correct and with a
    wrong final answer equal total weight; other items weigh 1."""
    n_y = {y: sum(i.last and i.episode.correct is y for i in pool) for y in (True, False)}
    if not all(n_y.values()):
        return [1.0] * len(pool)
    n_last = n_y[True] + n_y[False]
    return [n_last / (2 * n_y[i.episode.correct]) if i.last and i.episode.correct is not None else 1.0
            for i in pool]
