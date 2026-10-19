"""Episodes: the belief model's view of one agent run.

An episode is one query of an inference run (``experiments/inference/dra_inference.py``)
on a training pool split, in ``monitor`` or ``off`` mode: the query and per
search iteration the documents the agent was shown, in order
(``trajectory/{qid}.jsonl``, ``read_episode``).  Its steps are numbered as
the belief model numbers them (``iteration``, from 1, one per search
iteration); ``agent_iteration`` is the agent's own counter, the key that
joins a trajectory iteration to its belief step.  The belief never changes
what the agent does, so these observations are fixed and any updater step
can be rebuilt from any belief chain over the episode (phases 2 and 3).

A monitor run also has the teacher belief model's view
(``belief/{qid}.jsonl``, ``attach_teacher``): its criteria (the init call's
output) and the model that wrote them, and per iteration the teacher
updater's reply.  The belief file must have one step per search iteration
of the trajectory, with the same documents.

Document text is read back from the corpus (``CorpusLookup.contents``, the
field the retrievers return).

``teacher_init_call`` rebuilds the init call (the criteria extraction
prompt) with the episode's criteria as the reply.  ``replay_teacher`` re-runs the inference belief updater on the episode with
the teacher's saved replies in place of the LLM: it gives the exact prompt
of every teacher call (the SFT inputs) and checks the replay against the
saved ``criteria_updates`` (statuses, cited passages and span checks, which
depend on the passage text), so an episode whose documents or prompts
cannot be rebuilt is caught.

The episode also carries the gold of the rewards and scores
(``annotate_gold``, from the qrels): per step the gold documents first
seen in it and ``recall_so_far``, and the final correctness of the agent's
answer.  The gold of an updater call is the gold among its passages
(``Episode.gold_among``): a document of a failed or unparsed call stays
unjudged and is shown again in a later call, so it counts in that call.
"""

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

# Belief files written with the current statuses, updater prompts and layout.
MIN_SCHEMA_VERSION = 16


@dataclass
class EpisodeStep:
    # The belief model's step number (from 1) and the agent's own iteration.
    iteration: int
    agent_iteration: int
    # The documents shown in the iteration, in order (all its searches).
    doc_ids: List[str]
    # Relevant documents of the step not seen in an earlier step, and the
    # share of the query's relevant documents seen up to the step (annotate_gold).
    new_gold_ids: List[str] = field(default_factory=list)
    recall_so_far: Optional[float] = None
    # Monitor runs (attach_teacher): the teacher updater's reply (null when
    # it was not called), the call raised (no reply), the saved
    # criteria_updates (replay check) and recall_so_far (gold check).
    teacher_output: Optional[str] = None
    teacher_failed: bool = False
    teacher_updates: List[Dict[str, Any]] = field(default_factory=list)
    saved_recall: Optional[float] = None
    # The passages of the step's call in the teacher's replay (null when it made none).
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
    # The training pool split (layout.POOL_SPLITS).
    subset: Optional[str]
    dataset_year: Optional[str]
    query: str
    query_shape: str
    steps: List[EpisodeStep]
    run_dir: str
    # Monitor runs: the teacher's criteria, the criteria cap of the init
    # prompt (run_config.json) and the model that wrote the criteria.
    criteria: List[Dict[str, str]] = field(default_factory=list)
    max_criteria: Optional[int] = None
    criteria_model: Optional[str] = None
    # Steps whose teacher replay matched the saved run (all of them when it matched).
    replay_steps: Optional[int] = None
    replay_error: Optional[str] = None
    # The agent of the run, and the pool source of the query (its id prefix).
    policy_model: Optional[str] = None
    source: Optional[str] = None
    # Relevant documents of the query; the agent's final answer is correct
    # (null when it was not judged or the judge failed).
    num_relevant: Optional[int] = None
    correct: Optional[bool] = None

    @property
    def split(self) -> Split:
        return Split(self.dataset, self.subset, self.dataset_year)

    @property
    def has_gold(self) -> bool:
        """Some step found a relevant document."""
        return any(s.new_gold_ids for s in self.steps)

    def gold_among(self, doc_ids: List[str]) -> Set[str]:
        """The relevant documents among *doc_ids* (the passages of one
        updater call, of any belief chain over the episode)."""
        gold = {i for s in self.steps for i in s.new_gold_ids}
        return gold.intersection(doc_ids)

    def shown_gold(self, t: int) -> Set[str]:
        """The relevant documents among the passages of step *t*'s updater
        call in the teacher's replay, which the updater judges there for the
        first time."""
        return self.gold_among(self.steps[t].shown_doc_ids or [])

    def criteria_list(self) -> List[Criterion]:
        return [Criterion(**c) for c in self.criteria]

    def criteria_extractor(self) -> LLMCriteriaSource:
        """The init call of the episode's query (prompt and parse; no LLM)."""
        cap = self.max_criteria or DATASET_SPECS[self.dataset].max_criteria
        return LLMCriteriaSource(_NoLLM(), max_criteria=cap, query_shape=self.query_shape)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _call_failed(errors: List[str]) -> bool:
    """The step's updater call raised (not a parse failure)."""
    return any(e.startswith("belief_update: belief_updater: ") and "parse (attempt" not in e
               and "empty text for doc" not in e for e in errors)


def _iteration(value: Any) -> int:
    """The iteration of a trajectory step: ``"3"``, or ``"3.2"`` for the
    second search of iteration 3."""
    return int(str(value).split(".", 1)[0])


def read_episode(path: Path, split: Split, run_dir: Path) -> Optional[Episode]:
    """The episode of one ``trajectory/{qid}.jsonl`` file: one step per
    search iteration, in order, with its shown documents (those of all its
    searches, first occurrence kept; none when its searches returned
    nothing), or None when the run of the query did not finish (no meta
    line)."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    if meta is None:
        return None
    docs: Dict[int, List[str]] = {}
    for r in lines:
        if r.get("record") is None and r.get("search_query"):
            docs.setdefault(_iteration(r["iter"]), []).extend(r.get("seen_docs") or [])
    qid = str(meta["qid"])
    return Episode(
        qid=qid, dataset=split.dataset, subset=split.subset, dataset_year=split.dataset_year,
        query=meta["question"], query_shape=DATASET_SPECS[split.dataset].query_shape,
        steps=[EpisodeStep(iteration=n, agent_iteration=i, doc_ids=list(dict.fromkeys(d)))
               for n, (i, d) in enumerate(docs.items(), 1)],
        run_dir=str(run_dir), source=qid.split("_", 1)[0] if "_" in qid else None,
    )


def attach_teacher(episode: Episode, path: Path) -> Optional[str]:
    """Add the teacher's view of ``belief/{qid}.jsonl`` to *episode*;
    returns why it cannot be used (old schema, no criteria, other question,
    or belief steps that are not the trajectory's search iterations with
    their documents), or None.

    A belief step is joined to the trajectory iteration of its
    ``agent_iteration``, and must have the episode step's number and
    documents."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    if meta is None or (meta.get("schema_version") or 0) < MIN_SCHEMA_VERSION:
        return f"belief file older than schema {MIN_SCHEMA_VERSION}"
    if not meta.get("criteria"):
        return "no criteria"
    if meta.get("question") != episode.query:
        return "belief file question differs from the trajectory's"
    by_agent_iteration = {s.agent_iteration: s for s in episode.steps}
    joined = set()
    for r in lines:
        if r.get("record") != "step":
            continue
        step = by_agent_iteration.get(int(r["agent_iteration"]))
        if step is None:
            return f"belief step {r['iteration']}: agent iteration {r['agent_iteration']} has no search in the trajectory"
        if int(r["iteration"]) != step.iteration:
            return (f"belief step {r['iteration']}: agent iteration {r['agent_iteration']} is "
                    f"search iteration {step.iteration} of the trajectory")
        doc_ids = list(dict.fromkeys(d["doc_id"] for d in r.get("docs") or []))
        if doc_ids != step.doc_ids:
            return f"belief step {r['iteration']}: documents differ from the trajectory's"
        step.teacher_output = r.get("belief_updater_output")
        step.teacher_failed = _call_failed(r.get("errors") or [])
        step.teacher_updates = r.get("criteria_updates") or []
        step.saved_recall = r.get("recall_so_far")
        joined.add(step.iteration)
    missing = [s.agent_iteration for s in episode.steps if s.iteration not in joined]
    if missing:
        return f"agent iterations {missing} of the trajectory have no belief step"
    episode.criteria = meta["criteria"]
    episode.criteria_model = (meta.get("criteria_info") or {}).get("model")
    episode.num_relevant = meta.get("num_relevant")
    return None


def annotate_gold(episode: Episode, relevant_ids: Set[str]) -> Optional[str]:
    """Set each step's ``new_gold_ids`` and ``recall_so_far`` from the
    query's relevant documents; on a monitor episode, check them against
    the saved ones: returns the first difference, or None.  On a
    difference the gold from the qrels is kept."""
    seen: Set[str] = set()
    for step in episode.steps:
        step.new_gold_ids = [i for i in step.doc_ids if i in relevant_ids and i not in seen]
        seen.update(step.new_gold_ids)
        step.recall_so_far = round(len(seen) / len(relevant_ids), 4) if relevant_ids else None
    saved = episode.num_relevant
    episode.num_relevant = len(relevant_ids)
    if saved is not None and saved != len(relevant_ids):
        return f"num_relevant {saved} saved, {len(relevant_ids)} in the qrels"
    for step in episode.steps:
        if step.saved_recall is not None and step.recall_so_far is not None \
                and abs(step.recall_so_far - step.saved_recall) > 1e-4:
            return f"iteration {step.iteration}: recall_so_far {step.saved_recall} saved, {step.recall_so_far} from the qrels"
    return None


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
    episode, from *criteria* (default: the episode's, the teacher's), with
    the LLM calls made by the caller.

    ``open_step(t)`` starts step *t* and returns its updater call, or None
    when its documents need none (all judged before, or none with text; the
    step is then applied at once); ``submit`` applies the reply to the open
    call."""

    def __init__(self, episode: Episode, docs: List[List[Dict[str, Any]]],
                 criteria: Optional[List[Criterion]] = None) -> None:
        self.episode = episode
        self._docs = docs
        self.updater = build_belief_judges(_NoLLM(), model="trained", query_shape=episode.query_shape)[0]
        self.signal = BeliefUpdateSignal(self.updater)
        self.signal.reset(episode.query, episode.criteria_list() if criteria is None else criteria)
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
