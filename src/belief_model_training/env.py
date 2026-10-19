"""Phase 3 environment: belief update steps (GRPO groups) from the teacher's
or the student's own belief chain.

A step item is one updater call of a phase 1 episode (``rl`` or ``val``
use): a step of the matched replay where the teacher was called and
replied.  The agent ran in monitor mode, so the documents of every step do
not depend on the belief, and a step's prompt can be built from any belief
chain over the episode:

    teacher-forced  from the teacher's belief b_{t-1}: the teacher's replies
                    replayed up to the step (``build_prompts``)
    student state   from the student's own b_{t-1}: the student's criteria
                    (its init call) and its replies to the earlier steps,
                    greedy as at inference (``student_prompts``, DAgger-style:
                    the states the student meets when it runs alone)

The student writes ``group_size`` replies to a prompt; each is applied to a
copy of b_{t-1}, which gives its statuses and cited support for the reward.
The gold of a prompt is the relevant documents among its call's passages,
which differ between chains when a failed or unparsed call leaves documents
unjudged.  r_ts applies to a prompt when it is ``final``: the last updater
call of the student's chain (the step of the teacher's last call, a chain's
calls following the same documents).  The init call is not trained here
(the criteria stay as learned in SFT).

``run_student_chains`` runs the student's chain over whole episodes for the
free-running validation.

``StepSampler`` draws the step items of a batch: a share of steps with new
gold documents, at most a share from episodes without gold, the rest from
the other steps; within each pool the last calls of episodes are balanced
on the correctness of the final answer.
"""

import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from belief_model_component import Criterion

from .episodes import BeliefTracker, DocumentStore, Episode, UpdaterCall
from .rollout import VLLMRolloutEngine

# LLMCriteriaSource's max_tokens: the init call at inference.
CRITERIA_MAX_TOKENS = 4096


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
    # The relevant documents among the call's passages.
    gold: Set[str]
    # r_ts applies: the last updater call of the chain.
    final: bool
    # "teacher" (teacher-forced) or "student" (the student's own state).
    state: str = "teacher"


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


def _prompt(item: StepItem, call: UpdaterCall, signal: Any, final: bool, state: str) -> StepPrompt:
    gold = item.episode.gold_among([d["doc_id"] for d in call.kept])
    return StepPrompt(item, call, _snapshot(signal), gold, final, state)


def build_prompts(items: List[StepItem], store: DocumentStore, *, final: bool = True) -> List[StepPrompt]:
    """The teacher-forced prompts of *items*, in order; each episode is
    replayed once, up to its last item.  *final* False: r_ts applies to
    none (a teacher-forced final belief is mostly the teacher's).  An item
    whose step needs no call in the replay is left out (not expected within
    the matched replay)."""
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
                out[i] = _prompt(items[i], call, tracker.signal, final and items[i].last, "teacher")
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


# ---------------------------------------------------------------------------
# The student's own belief chain
# ---------------------------------------------------------------------------

def student_criteria(episodes: List[Episode], engine: VLLMRolloutEngine) -> List[List[Criterion]]:
    """The student's criteria of each episode: its init call, greedy as at
    inference (empty when the reply does not parse)."""
    extractors = [ep.criteria_extractor() for ep in episodes]
    gens = engine.generate([x.messages(ep.query) for x, ep in zip(extractors, episodes)], temperature=0.0,
                           max_new_tokens=CRITERIA_MAX_TOKENS)
    return [x.parse(g[0].text)[0] if g[0].text is not None else [] for x, g in zip(extractors, gens)]


@dataclass
class ChainCall:
    """One updater call of a student chain: its prompt (b_{t-1}) and the
    student's greedy reply (None when not generated)."""
    t: int
    prompt: StepPrompt
    text: Optional[str] = None
    finish_reason: Optional[str] = None


def run_student_chains(
    episodes: List[Episode], criteria: List[List[Criterion]], targets: List[int], store: DocumentStore,
    engine: VLLMRolloutEngine, *, max_new_tokens: int, reply_at_target: bool,
) -> List[List[ChainCall]]:
    """The student's belief chain over each episode from its *criteria*, up
    to step ``targets[e]``: the calls of all episodes are generated
    together, one round per step, greedy as at inference.  The call at the
    target step gets a reply only with *reply_at_target*.  A reply that does
    not parse (or a prompt too long) leaves the call's documents unjudged,
    as at inference.  The prompts' ``final`` is left False."""
    trackers = [BeliefTracker(ep, store.docs(ep), crit) for ep, crit in zip(episodes, criteria)]
    nxt = [0] * len(episodes)
    calls: List[List[ChainCall]] = [[] for _ in episodes]
    while True:
        pending: List[Tuple[int, ChainCall]] = []
        for e, (ep, tracker) in enumerate(zip(episodes, trackers)):
            while nxt[e] <= targets[e]:
                t = nxt[e]
                nxt[e] += 1
                call = tracker.open_step(t)
                if call is None:
                    continue
                cc = ChainCall(t, _prompt(StepItem(ep, t, False, False), call, tracker.signal, False, "student"))
                cc.prompt.item.has_gold = bool(cc.prompt.gold)
                calls[e].append(cc)
                if t == targets[e] and not reply_at_target:
                    tracker.submit(None)
                    break
                pending.append((e, cc))
                break
        if not pending:
            return calls
        gens = engine.generate([cc.prompt.call.messages for _, cc in pending], temperature=0.0,
                               max_new_tokens=max_new_tokens)
        for (e, cc), g in zip(pending, gens):
            cc.text, cc.finish_reason = g[0].text, g[0].finish_reason
            trackers[e].submit(cc.text)


def student_prompts(items: List[StepItem], store: DocumentStore, engine: VLLMRolloutEngine, *,
                    max_new_tokens: int) -> Tuple[List[StepPrompt], List[StepItem]]:
    """The prompts of *items* from the student's own belief: per episode,
    the student's criteria and its greedy chain up to the item's step.
    Returns ``(prompts, rest)``: *rest* are the items without a student
    prompt (no criteria parsed, or no call at the step in the student's
    chain).  r_ts applies to an item that is the teacher's last call."""
    by_episode: Dict[int, List[StepItem]] = defaultdict(list)
    for item in items:
        by_episode[id(item.episode)].append(item)
    groups = list(by_episode.values())
    episodes = [g[0].episode for g in groups]
    criteria = student_criteria(episodes, engine)
    keep = [e for e, c in enumerate(criteria) if c]
    rest = [item for e, g in enumerate(groups) if not criteria[e] for item in g]
    chains = run_student_chains(
        [episodes[e] for e in keep], [criteria[e] for e in keep], [max(i.t for i in groups[e]) for e in keep],
        store, engine, max_new_tokens=max_new_tokens, reply_at_target=False,
    )
    prompts: List[StepPrompt] = []
    for e, chain in zip(keep, chains):
        by_t = {cc.t: cc.prompt for cc in chain}
        for item in groups[e]:
            base = by_t.get(item.t)
            if base is None:
                rest.append(item)
                continue
            prompts.append(StepPrompt(item, base.call, base.signal, base.gold, item.last, "student"))
    return prompts, rest


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
        """*n* items in random order (the pools mixed)."""
        counts = [int(round(n * s)) for _, s, _ in self.pools]
        counts[-1] += n - sum(counts)
        out: List[StepItem] = []
        for (pool, _, weights), c in zip(self.pools, counts):
            out += self.rng.choices(pool, weights=weights, k=max(c, 0))
        self.rng.shuffle(out)
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
