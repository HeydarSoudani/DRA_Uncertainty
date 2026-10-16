"""Episodes: the belief model's view of one agent run (phase 1 output).

An episode is one query of a monitor-mode inference run: the query, its
criteria, and per search iteration the documents the belief model observed
(in the order it saw them) and the teacher updater's reply.  In monitor mode
the belief never changes what the agent does, so these observations are
fixed and the belief can be re-tracked over them by any updater (phase 3).

Document text is not saved in ``belief/{qid}.jsonl``; it is read back from
the corpus (``CorpusLookup.contents``, the field the retrievers return).

``replay_teacher`` re-runs the inference belief updater on the episode with
the teacher's saved replies in place of the LLM: it gives the exact prompt
of every teacher call (the SFT inputs) and checks the replay against the
saved ``criteria_updates`` (statuses, cited passages and span checks, which
depend on the passage text), so an episode whose documents or prompts
cannot be rebuilt is caught.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from belief_model_component import BeliefUpdateSignal, Criterion, build_belief_judges
from evaluation.belief.writer import SCHEMA_VERSION, _clean
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
    # Steps whose replay matched the saved run (all of them when it matched).
    replay_steps: Optional[int] = None
    replay_error: Optional[str] = None

    @property
    def split(self) -> Split:
        return Split(self.dataset, self.subset, self.dataset_year)

    def criteria_list(self) -> List[Criterion]:
        return [Criterion(**c) for c in self.criteria]

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
        ))
    return Episode(
        qid=str(meta["query_id"]), dataset=split.dataset, subset=split.subset,
        dataset_year=split.dataset_year, query=meta["question"],
        query_shape=meta.get("query_shape") or DATASET_SPECS[split.dataset].query_shape,
        criteria=meta["criteria"], steps=steps, run_dir=str(run_dir),
    )


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


class BeliefTracker:
    """The inference belief update (``BeliefUpdateSignal`` with the
    ``LLMBeliefUpdater`` of the query shape) over the fixed steps of an
    episode, with the updater's LLM calls made by the caller.

    ``open_step(t)`` starts step *t* and returns its updater call, or None
    when its documents need none (all judged before, or none with text; the
    step is then applied at once); ``submit`` applies the reply to the open
    call.  ``next_call`` walks the steps in order."""

    def __init__(self, episode: Episode, docs: List[List[Dict[str, Any]]],
                 max_steps: Optional[int] = None) -> None:
        self.episode = episode
        self._docs = docs[:max_steps] if max_steps else docs
        self.updater = build_belief_judges(_NoLLM(), model="trained", query_shape=episode.query_shape)[0]
        self.signal = BeliefUpdateSignal(self.updater)
        self.signal.reset(episode.query, episode.criteria_list())
        self._next = 0
        self._pending: Optional[UpdaterCall] = None

    @property
    def num_steps(self) -> int:
        return len(self._docs)

    def open_step(self, t: int) -> Optional[UpdaterCall]:
        if self._pending is not None:
            raise RuntimeError("the previous updater call has no reply yet")
        self._next = t + 1
        docs = self.signal.unjudged(_unique(self._docs[t]))
        if not docs:
            return None
        messages, kept, _ = self.updater.prompt(self.episode.query, self.signal.state, docs)
        if messages is None:
            self.signal.apply(docs, [], None)
            return None
        self._pending = UpdaterCall(self.episode.steps[t].iteration, messages, docs, kept, self.signal.statuses())
        return self._pending

    def next_call(self) -> Optional[UpdaterCall]:
        """The next updater call, or None when the episode is over."""
        if self._pending is not None:
            return self._pending
        while self._next < len(self._docs):
            call = self.open_step(self._next)
            if call is not None:
                return call
        return None

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


def _unique(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for d in docs:
        if d["doc_id"] in seen:
            continue
        seen.add(d["doc_id"])
        out.append(d)
    return out


def _canonical(records: List[Dict[str, Any]]) -> str:
    return json.dumps(_clean(records), sort_keys=True, ensure_ascii=False)


def replay_teacher(
    episode: Episode, docs: List[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """Re-run the belief update of *episode* with the teacher's saved replies.

    Returns ``(calls, matched, error)``: one ``{iteration, messages, output,
    parsed, spans_verified}`` per teacher call before the first step whose
    replay differs from the saved run, the number of steps before it, and
    the reason of the difference (None when every step matches).
    """
    tracker = BeliefTracker(episode, docs)
    calls: List[Dict[str, Any]] = []
    for t, step in enumerate(episode.steps):
        call = tracker.open_step(t)
        if call is None:
            if step.teacher_output is not None or step.teacher_failed:
                return calls, t, f"iteration {step.iteration}: the saved run called the updater, the replay does not"
            continue
        if step.teacher_output is None and not step.teacher_failed:
            return calls, t, f"iteration {step.iteration}: the replay calls the updater, the saved run did not"
        outcome = tracker.submit(None if step.teacher_failed else step.teacher_output)
        if _canonical(outcome["criteria_updates"]) != _canonical(step.teacher_updates):
            return calls, t, f"iteration {step.iteration}: replayed criteria_updates differ from the saved ones"
        if not step.teacher_failed:
            cited = [e for r in outcome["criteria_updates"] for e in r["support"] + r["contradict"]]
            calls.append({
                "iteration": step.iteration, "messages": call.messages, "output": step.teacher_output,
                "parsed": outcome["parsed"],
                "spans_verified": all(all(e["span_verified"]) for e in cited),
            })
    return calls, len(episode.steps), None
