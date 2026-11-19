"""One belief chain over an episode: the inference belief model itself.

A chain is ``belief_model_component.BeliefModel`` run over a saved episode
exactly as the belief stage runs it (``orchestration.belief_stage``):
reset (the beliefs), one ``observe`` per search iteration, ``finish`` on the
answer turn (the final scorer's call).  Only the LLM client differs: every
call of the belief model goes through a ``ChainClient``, which records it
(its kind, step and attempt, the messages and the reply):

* ``ReplayClient`` returns the teacher's saved reply of the call (SFT
  samples: the prompts are rebuilt by the belief model, byte for byte);
* ``rl.rollout.SampledClient`` samples the student's reply (RL and
  validation).

The beliefs are the dataset's gold beliefs (``GoldBeliefs``, as the
belief stage gives a training dataset's query), or the student's own init
call (``LLMBeliefSource`` on the chain's client).  The client's sampling
settings are its own: the temperature the belief model passes is ignored.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from indexing_corpus_dataset.layout import DATASET_SPECS, extraction_kind

from .config import GOLD_INIT
from .runs import DocumentStore, Episode

#: Call kinds.
INIT, UPDATE, SCORE = "init", "update", "score"
#: The model name the student's calls are recorded under.
STUDENT = "student"


@dataclass
class CallRecord:
    """One LLM call of a chain."""
    kind: str
    step: Optional[int]
    attempt: int
    messages: List[Dict[str, str]]
    reply: str
    max_tokens: int = 4096
    # Sampled calls only (rl.rollout): token ids, vLLM log-probs, finish reason.
    prompt_ids: List[int] = field(default_factory=list)
    response_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    finish_reason: Optional[str] = None


class ChainClient:
    """The LLM client of one chain (``complete(messages, **kwargs)``):
    records every call; subclasses produce the reply (``_reply``)."""

    def __init__(self) -> None:
        self.calls: List[CallRecord] = []
        self._kind: Optional[str] = None
        self._step: Optional[int] = None
        self._attempt = 0

    def begin(self, kind: str, step: Optional[int]) -> None:
        """The next calls are those of *kind* at *step* (their attempts counted from 1)."""
        self._kind, self._step, self._attempt = kind, step, 0

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 4096, **_: Any) -> str:
        self._attempt += 1
        record = CallRecord(self._kind, self._step, self._attempt, [dict(m) for m in messages], "", max_tokens)
        self._reply(record, max_tokens)
        self.calls.append(record)
        return record.reply

    def _reply(self, record: CallRecord, max_tokens: int) -> None:
        raise NotImplementedError


class ReplayClient(ChainClient):
    """Returns the saved reply of each call (*replies*, by ``(kind,
    step)``); a call without one gets an empty reply and is listed in
    ``unexpected``."""

    def __init__(self, replies: Dict[Tuple[str, int], str]) -> None:
        super().__init__()
        self.replies = replies
        self.unexpected: List[Tuple[str, Optional[int]]] = []

    def _reply(self, record: CallRecord, max_tokens: int) -> None:
        reply = self.replies.get((record.kind, record.step))
        if reply is None:
            self.unexpected.append((record.kind, record.step))
        record.reply = reply or ""


def gold_belief_source(episode: Episode):
    """The episode's gold beliefs as the belief stage gives them
    (``init.sources.GoldBeliefSource``: id, text, the kind of the shape)."""
    from belief_model_component import Belief, BeliefSource, belief_kinds

    rows = episode.gold_beliefs
    beliefs = [Belief(id=c["id"], text=c["text"], kind=kind)
               for c, kind in zip(rows, belief_kinds(episode.query_shape, [c["text"] for c in rows]))]

    class GoldBeliefs(BeliefSource):
        def get(self, query_id, query):
            return list(beliefs), {"belief_source": "gold", "errors": []}

    return GoldBeliefs()


def init_source(dataset: str, client: Any):
    """The belief init call of *dataset*'s queries on *client*, as a query
    without gold beliefs gets it (the dataset's shape and cap)."""
    from belief_model_component import LLMBeliefSource

    spec = DATASET_SPECS[dataset]
    return LLMBeliefSource(llm_client=client, max_beliefs=spec.max_beliefs, model_name=STUDENT,
                           query_shape=spec.query_shape, kind=extraction_kind(dataset))


def build_belief_model(episode: Episode, client: ChainClient, beliefs: str = GOLD_INIT):
    """The belief model of a chain: *client* for every call, the run's
    passage cut, the gold beliefs or the student's own init."""
    from belief_model_component import BeliefModel, build_belief_updater, build_final_scorer

    source = gold_belief_source(episode) if beliefs == GOLD_INIT else init_source(episode.dataset, client)
    updater = build_belief_updater(client, model=STUDENT, query_shape=episode.query_shape,
                                   max_passage_tokens=episode.max_passage_tokens)
    scorer = build_final_scorer(client, model=STUDENT, query_shape=episode.query_shape)
    if scorer is None:
        raise ValueError(f"{episode.key}: query shape {episode.query_shape} has no final scorer (a report)")
    return BeliefModel(belief_source=source, belief_updater=updater, qrels={}, final_scorer=scorer)


@dataclass
class ChainResult:
    episode: Episode
    client: ChainClient
    beliefs: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def calls(self) -> List[CallRecord]:
        return self.client.calls

    @property
    def s_T(self) -> Optional[float]:
        """The final scorer's probability, None when its reply did not parse."""
        if self.final is None or not self.final.get("final_score"):
            return None
        return self.final.get("certainty")


def run_chain(episode: Episode, client: ChainClient, store: DocumentStore, beliefs: str = GOLD_INIT,
              on_step: Optional[Callable[[int], None]] = None) -> ChainResult:
    """The belief model over *episode* with *client* (see the module
    docstring).  An exception stops the chain (``error``)."""
    result = ChainResult(episode, client)
    model = None
    try:
        model = build_belief_model(episode, client, beliefs)
        client.begin(INIT, None)
        model.reset(query_id=episode.qid, query=episode.question)
        result.beliefs = model.init_summary()["beliefs"]
        if not result.beliefs:
            result.error = "no beliefs"
            return result
        for k, it in enumerate(episode.saved.iterations, 1):
            if on_step is not None:
                on_step(k)
            client.begin(UPDATE, k)
            model.observe(subqueries=it.subqueries, docs=store.docs(episode, it.doc_ids),
                          iter_num=it.agent_iteration, agent_turn=it.agent_turn)
        client.begin(SCORE, len(episode.saved.iterations) + 1)
        model.finish(episode.saved.answer_turn)
    except Exception as e:  # noqa: BLE001 - one chain, reported by the caller
        result.error = f"{type(e).__name__}: {e}"
    if model is not None:
        result.steps = list(model.steps)
        result.final = model.final
    return result
