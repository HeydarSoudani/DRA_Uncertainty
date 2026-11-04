"""Episodes: the belief model's view of one agent run.

An episode is one query of an inference run (``experiments/inference/dra_inference.py``)
on a training pool split, in ``monitor`` or ``off`` mode: the query and per
search iteration the documents the agent was shown, in order, and the
agent's turn of the iteration (its reasoning and queries,
``agent_turn.agent_turn_text``), then its answer turn
(``trajectory/{qid}.jsonl`` and ``generation/{qid}.md``, ``read_episode``).  Its steps are numbered as
the belief model numbers them (``iteration``, from 1, one per search
iteration); ``agent_iteration`` is the agent's own counter, the key that
joins a trajectory iteration to its belief step.  The belief never changes
what the agent does, so these observations are fixed and any updater step
can be rebuilt from any belief chain over the episode (phases 2 and 3).

A monitor run also has the teacher belief model's view
(``belief/{qid}.jsonl``, ``attach_teacher``): its criteria (the gold
criteria of the pool, or the init call's output) and the model that wrote
them, per iteration the teacher updater's reply and the agent turn it read,
and the final call on the answer turn.  The belief file must have one step
per search iteration of the trajectory, with the same documents.

Document text is read back from the corpus (``CorpusLookup.contents``, the
field the retrievers return).

``teacher_init_call`` rebuilds the init call (the criteria extraction
prompt; uncapped for gold criteria, within the cap for silver ones) with
the episode's criteria as the reply.  ``replay_teacher``
re-runs the inference belief updater on the episode with the teacher's
saved replies in place of the LLM: it gives the exact prompt of every
teacher call (the SFT inputs), the final call on the answer turn included,
and checks the replay against the saved ``criteria_updates`` (statuses,
cited passages and span checks, which depend on the passage text) and
``candidate_updates`` (anchors, which depend on the agent turns), so an
episode whose documents or prompts cannot be rebuilt is caught.

The episode also carries the gold of the rewards and scores
(``annotate_gold``, from the qrels): per step the gold documents first
seen in it and ``recall_so_far``, and the final correctness of the agent's
answer; with the gold criteria of the pool (``gold_criteria``), which
criteria each agent turn states the gold answer of (``candidate_gold``).
The gold of an updater call is the gold among its passages
(``Episode.gold_among``): a document of a failed or unparsed call stays
unjudged and is shown again in a later call, so it counts in that call.
"""

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from belief_model_component import BeliefUpdateSignal, Criterion, LLMCriteriaSource, build_belief_judges
from belief_model_component.agent_turn import agent_turn_text, answer_turn
from belief_model_component.belief_model import unique_docs
from evaluation.belief import SCHEMA_VERSION
from evaluation.belief.writer import _clean
from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import DATASET_SPECS, GOLD_CRITERIA, default_corpus_path
from utils.io_utils import read_jsonl

from .config import Split


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
    # The agent's turn of the iteration (reasoning and queries): from the
    # trajectory, or as the teacher read it (attach_teacher).
    agent_turn: Optional[str] = None
    # Monitor runs: the saved candidate_updates (replay check).
    teacher_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # Gold criteria (candidate_gold): the ids of the criteria whose gold
    # answer the step's agent turn states.
    stated: Optional[Set[str]] = None

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
    # Monitor runs: the run's criteria, the cap of a silver init prompt
    # (run_config.json), the model that wrote the criteria and their kind
    # (gold or silver, ``criteria_kind`` of the criteria info).
    criteria: List[Dict[str, str]] = field(default_factory=list)
    max_criteria: Optional[int] = None
    criteria_model: Optional[str] = None
    criteria_kind: Optional[str] = None
    # Steps whose teacher replay matched the saved run (all of them when it matched).
    replay_steps: Optional[int] = None
    replay_error: Optional[str] = None
    # The agent of the run, and the pool source of the query (its gold annotation row).
    policy_model: Optional[str] = None
    source: Optional[str] = None
    # Relevant documents of the query; the agent's final answer is correct
    # (null when it was not judged or the judge failed).
    num_relevant: Optional[int] = None
    correct: Optional[bool] = None
    # The agent's answer turn (null without an answer); monitor runs: the
    # teacher's final call on it (reply, failed, saved candidate_updates).
    answer_turn: Optional[str] = None
    teacher_final_output: Optional[str] = None
    teacher_final_failed: bool = False
    teacher_final_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # The gold criteria of the pool (queries with texts_ok only): per
    # criterion {id, kind, relations, answers, doc_ids, text}, answers and
    # doc_ids empty when the source states none; the criteria ids the
    # answer turn states the gold answer of (candidate_gold).
    gold_criteria: Optional[List[Dict[str, Any]]] = None
    answer_stated: Optional[Set[str]] = None
    # Every entity name of the gold structure (InfoSeek's tree; else empty),
    # masked when matching an answer inside a longer name.
    gold_names: List[str] = field(default_factory=list)
    # Set queries: the members with their values and documents (the gold of
    # the rest criterion, matched member by member); else empty.
    gold_members: List[Dict[str, Any]] = field(default_factory=list)

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

    def gold_criteria_list(self) -> List[Criterion]:
        """The gold criteria, as the belief model reads them."""
        return [Criterion(id=c["id"], text=c["text"], kind=c["kind"]) for c in self.gold_criteria or []]

    def stated_at(self, t: Optional[int]) -> Optional[Set[str]]:
        """The criteria whose gold answer the agent turn of step *t* states
        (None: the answer turn)."""
        return self.answer_stated if t is None else self.steps[t].stated

    def criteria_extractor(self) -> LLMCriteriaSource:
        """The init call of the episode's query (prompt and parse; no LLM):
        uncapped for gold criteria, as gold is never capped; silver ones
        within the run's cap (the dataset's by default)."""
        if self.criteria_kind == GOLD_CRITERIA:
            cap = None
        else:
            cap = self.max_criteria or DATASET_SPECS[self.dataset].max_criteria
        return LLMCriteriaSource(_NoLLM(), max_criteria=cap, query_shape=self.query_shape)


def span_verified(entry: Dict[str, Any]) -> bool:
    """A cited passage of a ``criteria_updates`` record has spans and all of
    them occur in it (``span_verified``); one with no spans is not verified."""
    spans = entry.get("span_verified") or []
    return bool(spans) and all(spans)


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
    nothing) and its agent turn (the reasoning and queries of the
    iteration's lines), then the answer turn (the reasoning after the last
    search and ``generation/{qid}.md``), or None when the run of the query
    did not finish (no meta line)."""
    lines = read_jsonl(path)
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    if meta is None:
        return None
    steps = [r for r in lines if r.get("record") is None]
    docs: Dict[int, List[str]] = {}
    for r in steps:
        if r.get("search_query"):
            docs.setdefault(_iteration(r["iter"]), []).extend(r.get("seen_docs") or [])
    turns = {}
    for i in docs:
        own = [r for r in steps if _iteration(r["iter"]) == i]
        turns[i] = agent_turn_text([r.get("think") for r in own], [r.get("search_query") for r in own])
    qid = str(meta["qid"])
    generation = Path(run_dir) / "generation" / f"{qid}.md"
    answer = generation.read_text(encoding="utf-8") if generation.exists() else ""
    return Episode(
        qid=qid, dataset=split.dataset, subset=split.subset, dataset_year=split.dataset_year,
        query=meta["question"], query_shape=DATASET_SPECS[split.dataset].query_shape,
        steps=[EpisodeStep(iteration=n, agent_iteration=i, doc_ids=list(dict.fromkeys(d)), agent_turn=turns[i])
               for n, (i, d) in enumerate(docs.items(), 1)],
        run_dir=str(run_dir),
        answer_turn=answer_turn(steps, answer),
    )


def attach_teacher(episode: Episode, path: Path) -> Optional[str]:
    """Add the teacher's view of ``belief/{qid}.jsonl`` to *episode*;
    returns why it cannot be used (other schema, no criteria, other question,
    or belief steps that are not the trajectory's search iterations with
    their documents), or None.

    A belief step is joined to the trajectory iteration of its
    ``agent_iteration``, and must have the episode step's number and
    documents."""
    lines = read_jsonl(path)
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    if meta is None or meta.get("schema_version") != SCHEMA_VERSION:
        return f"belief file of schema {meta and meta.get('schema_version')}, not {SCHEMA_VERSION}"
    if not meta.get("criteria"):
        return "no criteria"
    if meta.get("question") != episode.query:
        return "belief file question differs from the trajectory's"
    by_agent_iteration = {s.agent_iteration: s for s in episode.steps}
    joined = set()
    for r in lines:
        if r.get("record") != "step":
            continue
        if r.get("agent_iteration") is None:
            return f"belief step {r.get('iteration')}: no agent iteration"
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
        step.teacher_candidates = r.get("candidate_updates") or []
        step.saved_recall = r.get("recall_so_far")
        # The turn the teacher read (the trajectory's may differ in steps
        # that are not search lines).
        if r.get("agent_turn") is not None:
            step.agent_turn = r["agent_turn"]
        joined.add(step.iteration)
    missing = [s.agent_iteration for s in episode.steps if s.iteration not in joined]
    if missing:
        return f"agent iterations {missing} of the trajectory have no belief step"
    episode.criteria = meta["criteria"]
    episode.criteria_model = (meta.get("criteria_info") or {}).get("model")
    episode.criteria_kind = (meta.get("criteria_info") or {}).get("criteria_kind")
    episode.num_relevant = meta.get("num_relevant")
    final = meta.get("final_update")
    if final is not None:
        episode.answer_turn = final.get("agent_turn")
        episode.teacher_final_output = final.get("belief_updater_output")
        episode.teacher_final_failed = _call_failed(final.get("errors") or [])
        episode.teacher_final_candidates = final.get("candidate_updates") or []
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


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentStore:
    """Document text by id, one ``CorpusLookup`` per corpus; the documents
    of the last *max_episodes* episodes asked for are kept in memory."""

    def __init__(self, max_episodes: int = 1024) -> None:
        self._lookups: Dict[Path, CorpusLookup] = {}
        self._cache: "OrderedDict[Tuple[str, str], List[List[Dict[str, str]]]]" = OrderedDict()
        self.max_episodes = max_episodes

    def docs(self, episode: Episode) -> List[List[Dict[str, str]]]:
        """Per step, its documents as the retrievers return them (``doc_id``
        and the corpus ``contents`` as ``relevant_text``); a document missing
        from the corpus has no text.  The lists are shared between calls:
        callers do not change them."""
        key = (episode.run_dir, episode.qid)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = default_corpus_path(episode.dataset, episode.subset)
        lookup = self._lookups.setdefault(path, CorpusLookup(path))
        contents = lookup.contents({i for s in episode.steps for i in s.doc_ids})
        docs = [[{"doc_id": i, "relevant_text": contents.get(i, "")} for i in s.doc_ids] for s in episode.steps]
        self._cache[key] = docs
        if len(self._cache) > self.max_episodes:
            self._cache.popitem(last=False)
        return docs


# ---------------------------------------------------------------------------
# Belief tracking over an episode
# ---------------------------------------------------------------------------

class _NoLLM:
    """Stands in for the updater's LLM where the caller makes the calls."""

    def complete(self, messages, **kwargs) -> str:
        raise RuntimeError("this updater is driven through prompt() and parse()")


@dataclass
class UpdaterCall:
    """One updater call of an episode: the prompt and what it is applied to.
    ``t`` indexes ``episode.steps`` (None: the final call, on the answer
    turn, which reads its answer candidates and has no passage)."""
    t: Optional[int]
    iteration: int
    messages: List[Dict[str, str]]
    docs: List[Dict[str, Any]]
    kept: List[Dict[str, Any]]
    belief_before: List[str]
    agent_turn: Optional[str]

    @property
    def final(self) -> bool:
        return self.t is None


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
    when it needs none (no unjudged document with text and no agent turn to
    read; the step is then applied at once); ``open_final()`` returns the
    final call on the answer turn, or None (no answer turn, or a query
    without answer candidates); ``submit`` applies the reply to the open
    call."""

    def __init__(self, episode: Episode, docs: List[List[Dict[str, Any]]],
                 criteria: Optional[List[Criterion]] = None) -> None:
        self.episode = episode
        self._docs = docs
        self.updater = build_belief_judges(_NoLLM(), model="trained", query_shape=episode.query_shape)[0]
        self.signal = BeliefUpdateSignal(self.updater)
        self.signal.reset(episode.query, episode.criteria_list() if criteria is None else criteria)
        self._pending: Optional[UpdaterCall] = None

    def _open(self, t: Optional[int], docs: List[Dict[str, Any]], agent_turn: Optional[str]) -> Optional[UpdaterCall]:
        state = self.signal.state
        messages, kept, _ = self.updater.prompt(self.episode.query, state, docs, agent_turn)
        if messages is None:
            self.signal.apply(docs, [], None, None)
            return None
        iteration = self.episode.steps[t].iteration if t is not None else len(self.episode.steps) + 1
        self._pending = UpdaterCall(t, iteration, messages, docs, kept, self.signal.statuses(), agent_turn)
        return self._pending

    def open_step(self, t: int) -> Optional[UpdaterCall]:
        if self._pending is not None:
            raise RuntimeError("the previous updater call has no reply yet")
        docs = self.signal.unjudged(unique_docs(self._docs[t]))
        turn = self.episode.steps[t].agent_turn if self.updater.candidates else None
        if not docs and turn is None:
            return None
        return self._open(t, docs, turn)

    def open_final(self) -> Optional[UpdaterCall]:
        if self._pending is not None:
            raise RuntimeError("the previous updater call has no reply yet")
        if not self.updater.candidates or self.episode.answer_turn is None:
            return None
        return self._open(None, [], self.episode.answer_turn)

    def submit(self, raw: Optional[str]) -> Dict[str, Any]:
        """Apply the reply *raw* (None: the call failed) to the open call;
        returns the call's outcome.  The final call keeps the statuses
        (it has no passage)."""
        call = self._pending
        if call is None:
            raise RuntimeError("no open updater call")
        self._pending = None
        updates, cands = self.updater.parse(raw, call.kept, call.iteration) if raw is not None else (None, None)
        if call.final and updates is not None:
            updates = []
        delta, records, revised = self.signal.apply(call.docs, updates, cands, raw, call.agent_turn)
        return {
            "iteration": call.iteration, "parsed": updates is not None, "candidates_parsed": cands is not None,
            "criteria_delta": delta,
            "belief_before": call.belief_before, "belief_after": self.signal.statuses(),
            "criteria_updates": records, "candidate_updates": revised,
            "candidates": self.signal.candidates(),
        }


def _canonical(records: List[Dict[str, Any]]) -> str:
    return json.dumps(_clean(records), sort_keys=True, ensure_ascii=False)


def replay_teacher(
    episode: Episode, docs: List[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """Re-run the belief update of *episode* with the teacher's saved replies.

    Returns ``(calls, matched, error)``: one ``{iteration, messages, output,
    parsed, spans_verified}`` per teacher call before the first step whose
    replay differs from the saved run (the final call on the answer turn
    last), the number of steps before it, and the reason of the difference
    (None when every step and the final call match).  Sets the
    ``shown_doc_ids`` of every step with a call.
    """
    tracker = BeliefTracker(episode, docs)
    calls: List[Dict[str, Any]] = []
    steps: List[Tuple[Optional[int], Optional[str], bool, List[Dict[str, Any]], List[Dict[str, Any]]]] = [
        (t, s.teacher_output, s.teacher_failed, s.teacher_updates, s.teacher_candidates)
        for t, s in enumerate(episode.steps)
    ] + [(None, episode.teacher_final_output, episode.teacher_final_failed, [], episode.teacher_final_candidates)]
    for t, output, failed, saved_updates, saved_candidates in steps:
        name = f"iteration {episode.steps[t].iteration}" if t is not None else "the final call"
        matched = t if t is not None else len(episode.steps)
        call = tracker.open_step(t) if t is not None else tracker.open_final()
        if call is None:
            if output is not None or failed:
                return calls, matched, f"{name}: the saved run called the updater, the replay does not"
            continue
        if t is not None:
            episode.steps[t].shown_doc_ids = [d["doc_id"] for d in call.kept]
        if output is None and not failed:
            return calls, matched, f"{name}: the replay calls the updater, the saved run did not"
        reply = None if failed else output
        outcome = tracker.submit(reply)
        if t is not None and _canonical(outcome["criteria_updates"]) != _canonical(saved_updates):
            return calls, matched, f"{name}: replayed criteria_updates differ from the saved ones"
        if _canonical(outcome["candidate_updates"]) != _canonical(saved_candidates):
            return calls, matched, f"{name}: replayed candidate_updates differ from the saved ones"
        if reply is not None:
            cited = [e for r in outcome["criteria_updates"] for e in r["support"] + r["contradict"]]
            written = [c for r in outcome["candidate_updates"] for c in r["written"]]
            calls.append({
                "iteration": call.iteration, "messages": call.messages, "output": output,
                "parsed": outcome["parsed"],
                "spans_verified": all(span_verified(e) for e in cited) and all(c["anchored"] for c in written),
            })
    return calls, len(episode.steps), None
