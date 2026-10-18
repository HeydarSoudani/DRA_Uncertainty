"""Episodes: the belief model's view of one agent run (phase 1 output).

An episode is one query of a monitor-mode inference run: the query, its
criteria (the init call's output) and the model that wrote them, and per
search iteration the documents the belief model observed (in the order it
saw them) and the teacher updater's reply.  In monitor mode the belief never
changes what the agent does, so these observations are fixed and any
updater step can be rebuilt from the teacher's belief before it (phase 3).

Document text is not saved in ``belief/{qid}.jsonl``; it is read back from
the corpus (``CorpusLookup.contents``, the field the retrievers return).

``teacher_init_call`` rebuilds the init call (the criteria extraction
prompt) with the episode's criteria as the reply.  ``replay_teacher`` re-runs the inference belief updater on the episode with
the teacher's saved replies in place of the LLM: it gives the exact prompt
of every teacher call (the SFT inputs) and checks the replay against the
saved ``criteria_updates`` (statuses, cited passages and span checks, which
depend on the passage text), so an episode whose documents or prompts
cannot be rebuilt is caught.

The episode also carries the gold the phase 3 rewards need, so that they read
no qrels or run files: per step the gold documents first seen in it
(``annotate_gold``), the passages its updater call showed (``replay_teacher``)
and ``recall_so_far``, and the final correctness of the agent's answer.
``Episode.shown_gold`` is the gold among a call's passages: a document of a
failed or unparsed call stays unjudged and is shown again in a later call,
so it counts in that call.  ``use`` (``assign_use``) puts each query in ``sft``, ``rl``
or ``val`` by a hash of its id, so a query keeps its use when the pool grows.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from belief_model_component import BeliefUpdateSignal, Criterion, LLMCriteriaSource, build_belief_judges
from belief_model_component.belief_model import unique_docs
from evaluation.belief.writer import _clean
from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import DATASET_SPECS, default_corpus_path

from .config import Split

logger = logging.getLogger(__name__)

# Belief files written with the current statuses and updater prompts.
MIN_SCHEMA_VERSION = 15


@dataclass
class EpisodeStep:
    iteration: int
    # The documents observed in the iteration, in the order the belief model saw them.
    doc_ids: List[str]
    # The teacher updater's reply (null when it was not called).
    teacher_output: Optional[str]
    # The updater call raised (no reply).
    teacher_failed: bool
    # The step's saved criteria_updates, for the replay check.
    teacher_updates: List[Dict[str, Any]] = field(default_factory=list)
    # Relevant documents seen up to and including the step / all relevant (saved).
    recall_so_far: Optional[float] = None
    # Relevant documents of the step not seen in an earlier step (annotate_gold).
    new_gold_ids: List[str] = field(default_factory=list)
    # The passages of the step's replayed updater call (null when it made none).
    shown_doc_ids: Optional[List[str]] = None

    @property
    def teacher_reply(self) -> Optional[str]:
        """The reply the belief was updated from: None when the updater was
        not called or the call failed."""
        return None if self.teacher_failed else self.teacher_output


@dataclass
class Episode:
    qid: str
    dataset: str
    subset: Optional[str]
    dataset_year: Optional[str]
    query: str
    query_shape: str
    criteria: List[Dict[str, str]]
    steps: List[EpisodeStep]
    run_dir: str
    # The criteria cap of the init prompt, and the model that wrote the criteria.
    max_criteria: Optional[int] = None
    criteria_model: Optional[str] = None
    # Steps whose replay matched the saved run (all of them when it matched).
    replay_steps: Optional[int] = None
    replay_error: Optional[str] = None
    # The agent of the run, and the pool source of the query (its id prefix).
    policy_model: Optional[str] = None
    source: Optional[str] = None
    # Relevant documents of the query; the agent's final answer is correct
    # (null when it was not judged or the judge failed).
    num_relevant: Optional[int] = None
    correct: Optional[bool] = None
    # sft, rl or val (assign_use).
    use: Optional[str] = None

    @property
    def split(self) -> Split:
        return Split(self.dataset, self.subset, self.dataset_year)

    @property
    def has_gold(self) -> bool:
        """Some step found a relevant document."""
        return any(s.new_gold_ids for s in self.steps)

    def shown_gold(self, t: int) -> Set[str]:
        """The relevant documents among the passages of step *t*'s updater
        call, which the updater judges there for the first time."""
        gold = {i for s in self.steps for i in s.new_gold_ids}
        return gold.intersection(self.steps[t].shown_doc_ids or [])

    def criteria_list(self) -> List[Criterion]:
        return [Criterion(**c) for c in self.criteria]

    def criteria_extractor(self) -> LLMCriteriaSource:
        """The init call of the episode's query (prompt and parse; no LLM)."""
        cap = self.max_criteria or DATASET_SPECS[self.dataset].max_criteria
        return LLMCriteriaSource(_NoLLM(), max_criteria=cap, query_shape=self.query_shape)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        data = dict(data)
        data["steps"] = [EpisodeStep(**s) for s in data["steps"]]
        return cls(**data)


def _call_failed(errors: List[str]) -> bool:
    """The step's updater call raised (not a parse failure)."""
    return any(e.startswith("belief_update: belief_updater: ") and "parse (attempt" not in e
               and "empty text for doc" not in e for e in errors)


def read_episode(path: Path, split: Split, run_dir: Path) -> Optional[Episode]:
    """The episode of one ``belief/{qid}.jsonl`` file, or None when the file
    is older than ``MIN_SCHEMA_VERSION`` or has no criteria."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    if meta is None or (meta.get("schema_version") or 0) < MIN_SCHEMA_VERSION:
        return None
    if not meta.get("criteria"):
        return None
    steps = []
    for r in lines:
        if r.get("record") != "step":
            continue
        steps.append(EpisodeStep(
            iteration=r["iteration"],
            doc_ids=[d["doc_id"] for d in r.get("docs") or []],
            teacher_output=r.get("belief_updater_output"),
            teacher_failed=_call_failed(r.get("errors") or []),
            teacher_updates=r.get("criteria_updates") or [],
            recall_so_far=r.get("recall_so_far"),
        ))
    qid = str(meta["query_id"])
    return Episode(
        qid=qid, dataset=split.dataset, subset=split.subset,
        dataset_year=split.dataset_year, query=meta["question"],
        query_shape=meta.get("query_shape") or DATASET_SPECS[split.dataset].query_shape,
        criteria=meta["criteria"], steps=steps, run_dir=str(run_dir),
        max_criteria=meta.get("max_criteria"),
        criteria_model=(meta.get("criteria_info") or {}).get("model") or meta.get("llm_criteria"),
        policy_model=meta.get("agent"), source=qid.split("_", 1)[0] if "_" in qid else None,
        num_relevant=meta.get("num_relevant"),
    )


def annotate_gold(episode: Episode, relevant_ids: Set[str]) -> Optional[str]:
    """Set each step's ``new_gold_ids`` from the query's relevant documents
    and check them against the saved recall: returns the first difference,
    or None.  On a difference the gold is kept (it comes from the qrels) and
    the saved ``recall_so_far`` is left as it is."""
    seen: Set[str] = set()
    for step in episode.steps:
        step.new_gold_ids = [i for i in dict.fromkeys(step.doc_ids) if i in relevant_ids and i not in seen]
        seen.update(step.new_gold_ids)
    if episode.num_relevant is not None and episode.num_relevant != len(relevant_ids):
        return f"num_relevant {episode.num_relevant} saved, {len(relevant_ids)} in the qrels"
    found = 0
    for step in episode.steps:
        found += len(step.new_gold_ids)
        if step.recall_so_far is None or not relevant_ids:
            continue
        recall = round(found / len(relevant_ids), 4)
        if abs(recall - step.recall_so_far) > 1e-4:
            return f"iteration {step.iteration}: recall_so_far {step.recall_so_far} saved, {recall} from the qrels"
    return None


def assign_use(qid: str, sft_share: float, val_share: float) -> str:
    """``sft``, ``rl`` or ``val`` by a hash of *qid*: the shares of the pool
    are ``sft_share``, ``val_share`` and the rest; a query's use depends on
    its id only."""
    u = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    if u < sft_share:
        return "sft"
    if u < sft_share + val_share:
        return "val"
    return "rl"


def read_episodes(path: Path) -> List[Episode]:
    return [Episode.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterator[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentStore:
    """Document text by id, one ``CorpusLookup`` per corpus."""

    def __init__(self) -> None:
        self._lookups: Dict[Path, CorpusLookup] = {}

    def docs(self, episode: Episode) -> List[List[Dict[str, str]]]:
        """Per step, its documents as the retrievers return them (``doc_id``
        and the corpus ``contents`` as ``relevant_text``); a document missing
        from the corpus has no text."""
        path = default_corpus_path(episode.dataset, episode.subset)
        lookup = self._lookups.setdefault(path, CorpusLookup(path))
        contents = lookup.contents({i for s in episode.steps for i in s.doc_ids})
        return [[{"doc_id": i, "relevant_text": contents.get(i, "")} for i in s.doc_ids] for s in episode.steps]


# ---------------------------------------------------------------------------
# Belief tracking over an episode
# ---------------------------------------------------------------------------

class _NoLLM:
    """Stands in for the updater's LLM where the caller makes the calls."""

    def complete(self, messages, **kwargs) -> str:
        raise RuntimeError("this updater is driven through prompt() and parse()")


@dataclass
class UpdaterCall:
    """One updater call of an episode: the prompt and what it is applied to."""
    iteration: int
    messages: List[Dict[str, str]]
    docs: List[Dict[str, Any]]
    kept: List[Dict[str, Any]]
    belief_before: List[str]


def teacher_init_call(episode: Episode) -> Dict[str, Any]:
    """The init call of *episode* with its criteria as the reply:
    ``{messages, output, verified}``; ``verified`` is whether the reply
    parses back to the same criteria (texts and kinds)."""
    extractor = episode.criteria_extractor()
    criteria = episode.criteria_list()
    output = extractor.target(criteria)
    parsed, _ = extractor.parse(output)
    return {
        "messages": extractor.messages(episode.query), "output": output,
        "verified": [c.to_dict() for c in parsed] == [c.to_dict() for c in criteria],
    }


class BeliefTracker:
    """The inference belief model (``BeliefUpdateSignal`` with the
    ``LLMBeliefUpdater`` of the query shape) over the fixed steps of an
    episode, from the episode's criteria, with the LLM calls made by the
    caller.

    ``open_step(t)`` starts step *t* and returns its updater call, or None
    when its documents need none (all judged before, or none with text; the
    step is then applied at once); ``submit`` applies the reply to the open
    call."""

    def __init__(self, episode: Episode, docs: List[List[Dict[str, Any]]]) -> None:
        self.episode = episode
        self._docs = docs
        self.updater = build_belief_judges(_NoLLM(), model="trained", query_shape=episode.query_shape)[0]
        self.signal = BeliefUpdateSignal(self.updater)
        self.signal.reset(episode.query, episode.criteria_list())
        self._pending: Optional[UpdaterCall] = None

    def open_step(self, t: int) -> Optional[UpdaterCall]:
        if self._pending is not None:
            raise RuntimeError("the previous updater call has no reply yet")
        docs = self.signal.unjudged(unique_docs(self._docs[t]))
        if not docs:
            return None
        messages, kept, _ = self.updater.prompt(self.episode.query, self.signal.state, docs)
        if messages is None:
            self.signal.apply(docs, [], None)
            return None
        self._pending = UpdaterCall(self.episode.steps[t].iteration, messages, docs, kept, self.signal.statuses())
        return self._pending

    def submit(self, raw: Optional[str]) -> Dict[str, Any]:
        """Apply the reply *raw* (None: the call failed) to the open call;
        returns the step's outcome."""
        call = self._pending
        if call is None:
            raise RuntimeError("no open updater call")
        self._pending = None
        updates = self.updater.parse(raw, call.kept, call.iteration) if raw is not None else None
        delta, records = self.signal.apply(call.docs, updates, raw)
        return {
            "iteration": call.iteration, "parsed": updates is not None, "criteria_delta": delta,
            "belief_before": call.belief_before, "belief_after": self.signal.statuses(),
            "criteria_updates": records,
        }


def _canonical(records: List[Dict[str, Any]]) -> str:
    return json.dumps(_clean(records), sort_keys=True, ensure_ascii=False)


def replay_teacher(
    episode: Episode, docs: List[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """Re-run the belief update of *episode* with the teacher's saved replies.

    Returns ``(calls, matched, error)``: one ``{iteration, messages, output,
    parsed, spans_verified}`` per teacher call before the first step whose
    replay differs from the saved run, the number of steps before it, and
    the reason of the difference (None when every step matches).  Sets the
    ``shown_doc_ids`` of every step with a call.
    """
    tracker = BeliefTracker(episode, docs)
    calls: List[Dict[str, Any]] = []
    for t, step in enumerate(episode.steps):
        call = tracker.open_step(t)
        if call is None:
            if step.teacher_output is not None or step.teacher_failed:
                return calls, t, f"iteration {step.iteration}: the saved run called the updater, the replay does not"
            continue
        step.shown_doc_ids = [d["doc_id"] for d in call.kept]
        if step.teacher_output is None and not step.teacher_failed:
            return calls, t, f"iteration {step.iteration}: the replay calls the updater, the saved run did not"
        outcome = tracker.submit(step.teacher_reply)
        if _canonical(outcome["criteria_updates"]) != _canonical(step.teacher_updates):
            return calls, t, f"iteration {step.iteration}: replayed criteria_updates differ from the saved ones"
        if step.teacher_reply is not None:
            cited = [e for r in outcome["criteria_updates"] for e in r["support"] + r["contradict"]]
            calls.append({
                "iteration": step.iteration, "messages": call.messages, "output": step.teacher_output,
                "parsed": outcome["parsed"],
                "spans_verified": all(all(e["span_verified"]) for e in cited),
            })
    return calls, len(episode.steps), None
