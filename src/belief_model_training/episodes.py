"""Episodes: the belief model's view of one agent run.

An episode is one query of an inference run (``experiments/inference/dra_inference.py``)
on a training pool split, in ``monitor`` or ``off`` mode: the query and per
search iteration the documents the agent was shown, in order, and the
agent's turn of the iteration (its reasoning,
``agent_turn.agent_turn_text``), then its answer turn
(``trajectory/{qid}.jsonl`` and ``generation/{qid}.md``, ``read_episode``).  Its steps are numbered as
the belief model numbers them (``iteration``, from 1, one per search
iteration); ``agent_iteration`` is the agent's own counter, the key that
joins a trajectory iteration to its belief step.  The belief never changes
what the agent does, so these observations are fixed and any belief
update can be rebuilt from any belief chain over the episode (phases 2
and 3).

A monitor run also has the teacher belief model's view
(``belief/{qid}.jsonl``, ``attach_teacher``): its criteria and the model
that wrote them, per iteration the teacher's reply, the agent turn it read
and the records the reply gave (observations, candidate updates, changed
criteria, issues and the beliefs after it), and the final call on the answer turn (the
``final`` line: the final scorer's call on a short-answer query, one more
update on a report).  The belief file must have one step per search iteration
of the trajectory, with the same documents.

Document text is read back from the corpus (``CorpusLookup.contents``, the
field the retrievers return).

``BeliefTracker`` runs the inference belief update (``BeliefUpdateSignal``
with the ``LLMBeliefUpdater`` of the query shape, the same call triggers:
a step without a new passage makes no call and its turn is
read by the next one) over an episode with the LLM calls made by the
caller: the student's chains in phase 3, the
teacher's saved replies in ``replay_teacher``, which gives the exact prompt
of every teacher call (the SFT inputs), the final call on the answer turn
included (``ScorerCall``: the belief model's second task, on a
short-answer query), and checks the replay against the saved records (span checks
depend on the passage text and on the agent turns), so an episode
whose documents or prompts cannot be rebuilt is caught.

The episode also carries the gold of the rewards and scores: per step the
relevant documents first seen in it and ``recall_so_far`` (``annotate_gold``,
from the qrels), the final correctness of the agent's answer, and with the
gold criteria of the pool (``gold_criteria``: answers and documents per
criterion) which criteria each agent turn states the gold answer of
(``nugget_gold.annotate_stated``).
"""

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from belief_model_component import BeliefUpdateSignal, Criterion, build_belief_updater, build_final_scorer
from belief_model_component.final_score import check_issues, parse_final_score
from belief_model_component.agent_turn import agent_turn_text, answer_turn
from belief_model_component.belief_model import unique_docs
from belief_model_component.signals import CallPlan
from belief_model_component.updater import Passage
from evaluation.belief import SCHEMA_VERSION
from evaluation.belief.writer import _clean
from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import DATASET_SPECS, default_corpus_path
from utils.io_utils import read_jsonl

from .config import Split

# The records of one belief update (``BeliefUpdateSignal``), as saved.
RECORD_KEYS = ("observations", "updates", "changed", "issues", "beliefs")


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
    # Monitor runs (attach_teacher): the teacher's reply (null when it was
    # not called), the call raised (no reply), the saved records of the
    # reply (replay check) and recall_so_far (gold check).
    teacher_output: Optional[str] = None
    teacher_failed: bool = False
    teacher_records: Dict[str, Any] = field(default_factory=dict)
    saved_recall: Optional[float] = None
    # The agent's turn of the iteration (its reasoning): from the
    # trajectory, or as the teacher read it (attach_teacher).
    agent_turn: Optional[str] = None
    # Gold criteria (nugget_gold): the ids of the criteria whose gold
    # answer the step's agent turn states.
    stated: Optional[Set[str]] = None

    @property
    def teacher_reply(self) -> Optional[str]:
        """The reply the belief was updated from: None when the call was not
        made or failed."""
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
    # Characters of one document the run's belief update read
    # (``runs.passage_chars``); None: whole documents.
    max_passage_chars: Optional[int] = None
    # Monitor runs: the run's criteria, the model that wrote them.
    criteria: List[Dict[str, str]] = field(default_factory=list)
    criteria_model: Optional[str] = None
    replay_error: Optional[str] = None
    # The pool source of the query (its gold annotation row).
    source: Optional[str] = None
    # Relevant documents of the query; the agent's final answer is correct
    # (null when it was not judged or the judge failed).
    num_relevant: Optional[int] = None
    correct: Optional[bool] = None
    # The agent's answer turn (null without an answer); monitor runs: the
    # teacher's final call on it (reply, failed, saved records).
    answer_turn: Optional[str] = None
    teacher_final_output: Optional[str] = None
    teacher_final_failed: bool = False
    teacher_final_records: Dict[str, Any] = field(default_factory=dict)
    # The final status of the teacher's final scorer (null: none, or a report).
    teacher_final_label: Optional[str] = None
    # The gold criteria of the pool (queries with texts_ok only): per
    # criterion {id, kind, relations, answers, doc_ids, text}, answers and
    # doc_ids empty when the source states none; the criteria ids the
    # answer turn states the gold answer of (nugget_gold).
    gold_criteria: Optional[List[Dict[str, Any]]] = None
    answer_stated: Optional[Set[str]] = None
    # Every entity name of the gold structure (InfoSeek's tree; else empty),
    # masked when matching an answer inside a longer name.
    gold_names: List[str] = field(default_factory=list)

    @property
    def split(self) -> Split:
        return Split(self.dataset, self.subset, self.dataset_year)

    @property
    def has_gold(self) -> bool:
        """Some step found a relevant document."""
        return any(s.new_gold_ids for s in self.steps)

    @property
    def gold_by_id(self) -> Dict[str, Dict[str, Any]]:
        """Per id, its gold criterion; built once per ``gold_criteria``
        list (read in every inner loop of the scoring)."""
        cached = self.__dict__.get("_gold_by_id")
        if cached is None or cached[0] is not self.gold_criteria:
            cached = (self.gold_criteria, {c["id"]: c for c in self.gold_criteria or []})
            self.__dict__["_gold_by_id"] = cached
        return cached[1]

    def gold_criteria_list(self) -> List[Criterion]:
        """The gold criteria, as the belief model reads them."""
        return [Criterion(id=c["id"], text=c["text"], kind=c["kind"]) for c in self.gold_criteria or []]

    def stated_at(self, t: Optional[int]) -> Set[str]:
        """The criteria whose gold answer the agent turn of step *t* states
        (None: the answer turn)."""
        return (self.answer_stated if t is None else self.steps[t].stated) or set()


def _call_failed(errors: List[str], prefix: str = "belief_update: ") -> bool:
    """The step's belief update call (the final scorer's: *prefix*
    ``"final_score: "``) raised (not a parse failure, not a document
    without text)."""
    return any(e.startswith(prefix) and "parse (attempt" not in e for e in errors)


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
        turns[i] = agent_turn_text([r.get("think") for r in own])
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


def _records(line: Dict[str, Any]) -> Dict[str, Any]:
    return {k: line.get(k) for k in RECORD_KEYS}


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
    final = next((r for r in lines if r.get("record") == "final"), None)
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
        step.teacher_output = r.get("reply")
        step.teacher_failed = _call_failed(r.get("errors") or [])
        step.teacher_records = _records(r)
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
    episode.num_relevant = meta.get("num_relevant")
    # The run made no final call: the answer turn it saw was empty.
    episode.answer_turn = None
    if final is not None:
        episode.answer_turn = final.get("agent_turn")
        episode.teacher_final_output = final.get("reply")
        scorer = final.get("final_score") is not None or any(
            e.startswith("final_score: ") for e in final.get("errors") or [])
        episode.teacher_final_failed = _call_failed(final.get("errors") or [],
                                                    "final_score: " if scorer else "belief_update: ")
        episode.teacher_final_records = _records(final)
        episode.teacher_final_label = final.get("certainty_label")
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
    """One belief update call of an episode: its prompt and what it reads.
    ``t`` indexes ``episode.steps`` of the step that made the call (None:
    the final call, on the answer turn); ``iteration`` is that step's
    number (the answer turn's: the last step + 1); ``turns`` the agent's
    turns the call reads, ``(step number, text)``, those of earlier steps
    without a call first (``BeliefUpdateSignal.plan``)."""
    t: Optional[int]
    iteration: int
    messages: List[Dict[str, str]]
    plan: CallPlan
    # Per passage, its text as the call showed it (``LLMBeliefUpdater.shown_text``).
    shown: List[str] = field(default_factory=list)
    is_scorer = False

    @property
    def passages(self) -> List[Passage]:
        return self.plan.passages

    @property
    def turns(self) -> List[Tuple[int, str]]:
        return self.plan.turns

    def turn_positions(self, n_steps: int) -> List[Tuple[Optional[int], int]]:
        """Per turn of the call, ``(t, step number)``: *t* its index in the
        episode's steps (None: the answer turn, step ``n_steps + 1``)."""
        return [(step - 1 if step <= n_steps else None, step) for step, _ in self.turns]

    def shown_text(self) -> Dict[str, str]:
        """Per document of the call, its text as the call showed it."""
        return {p.doc_id: text for p, text in zip(self.passages, self.shown)}


@dataclass
class ScorerCall:
    """The final scorer's call of an episode (``final_score``, a
    short-answer query): its prompt, on b_{T-1} and the answer turn;
    ``iteration`` the answer turn's (the last step + 1)."""
    iteration: int
    messages: List[Dict[str, str]]
    t: Optional[int] = None
    is_scorer = True


class BeliefTracker:
    """The inference belief update over the fixed steps of an episode, from
    *criteria*, with the LLM calls made by the caller.

    ``open_step(t)`` starts step *t* and returns its call, or None when it
    makes none (no unjudged document with text: its turn waits for the
    next call, as at inference);
    ``open_final()`` returns the final call on the answer turn, or None (no
    answer turn): a ``ScorerCall`` on a short-answer query (the belief
    stays b_{T-1}), else one more update; ``submit`` applies the reply to
    the open call."""

    def __init__(self, episode: Episode, docs: List[List[Dict[str, Any]]], criteria: List[Criterion]) -> None:
        self.episode = episode
        self._docs = docs
        self.updater = build_belief_updater(_NoLLM(), model="trained", query_shape=episode.query_shape,
                                            max_passage_chars=episode.max_passage_chars)
        self.scorer = build_final_scorer(_NoLLM(), model="trained", query_shape=episode.query_shape)
        self.signal = BeliefUpdateSignal(self.updater)
        self.signal.reset(episode.query, criteria)
        self._pending: Optional[Any] = None

    def max_tokens(self, call: Any) -> int:
        """The reply budget of *call* (the inference updater's or scorer's)."""
        return self.scorer.max_tokens if call.is_scorer else self.updater.max_tokens

    def _open(self, t: Optional[int], iteration: int, plan: Optional[CallPlan]) -> Optional[UpdaterCall]:
        if plan is None:
            return None
        messages = self.updater.prompt(self.episode.query, self.signal.state, plan.turns, plan.passages)
        self._pending = UpdaterCall(t, iteration, messages, plan,
                                    [self.updater.shown_text(p.text) for p in plan.passages])
        return self._pending

    def open_step(self, t: int) -> Optional[UpdaterCall]:
        if self._pending is not None:
            raise RuntimeError("the previous call has no reply yet")
        if self.signal.state is None:
            return None
        step = self.episode.steps[t]
        plan, _ = self.signal.plan(unique_docs(self._docs[t]), step.iteration, step.agent_turn)
        return self._open(t, step.iteration, plan)

    def open_final(self) -> Optional[UpdaterCall]:
        if self._pending is not None:
            raise RuntimeError("the previous call has no reply yet")
        if self.signal.state is None or self.episode.answer_turn is None:
            return None
        iteration = len(self.episode.steps) + 1
        if self.scorer is not None:
            self._pending = ScorerCall(iteration, self.scorer.prompt(self.episode.query, self.signal.state,
                                                                     self.episode.answer_turn))
            return self._pending
        plan, _ = self.signal.plan([], iteration, self.episode.answer_turn, final=True)
        return self._open(None, iteration, plan)

    def score_outcome(self, call: ScorerCall, raw: Optional[str]) -> Dict[str, Any]:
        """The outcome of the scorer's reply *raw* (None: the call failed)
        to *call*, as ``submit`` gives it, without touching the belief (a
        GRPO group scores several replies to one scorer call):
        ``{iteration, parsed, observations, updates, changed, issues,
        beliefs, ruled_out, final_score, certainty, certainty_label}``
        (``ruled_out``: ``BeliefState.ruled_out_lists`` of b_{T-1}, for
        ``certainty.answer_certainty``)."""
        score = parse_final_score(raw) if raw is not None else None
        return {"iteration": call.iteration, "parsed": score is not None, "observations": None, "updates": [],
                "changed": [], "issues": [], "beliefs": self.signal.beliefs(),
                "ruled_out": self.signal.state.ruled_out_lists(),
                "final_score": ({**score.to_dict(), "issues": check_issues(score, self.signal.state)}
                                if score else None),
                "certainty": score.probability if score else None,
                "certainty_label": score.status if score else None}

    def submit(self, raw: Optional[str]) -> Dict[str, Any]:
        """Apply the reply *raw* (None: the call failed) to the open call;
        returns the call's outcome: ``{iteration, parsed, observations,
        updates, changed, issues, beliefs}`` (``beliefs`` every criterion's
        belief after the call; no observation and no change when the
        reply did not parse)."""
        call = self._pending
        if call is None:
            raise RuntimeError("no open call")
        self._pending = None
        if call.is_scorer:
            return self.score_outcome(call, raw)
        parsed = self.updater.parse(raw, call.turns, call.passages, call.iteration) if raw is not None else None
        # As the inference belief model records a call whose reply did not parse.
        out: Dict[str, Any] = {"iteration": call.iteration, "parsed": parsed is not None,
                               "observations": None, "updates": [], "changed": [], "issues": []}
        if parsed is not None:
            out.update(self.signal.apply(parsed, call.plan, call.iteration))
        out["beliefs"] = self.signal.beliefs()
        return out


def _canonical(records: Any) -> str:
    return json.dumps(_clean(records), sort_keys=True, ensure_ascii=False)


@dataclass
class TeacherCall:
    """One call of the teacher's replay: the call (an update, or the final
    scorer's), the teacher's reply (None: the call failed) and the
    replayed outcome."""
    call: Any
    output: Optional[str]
    outcome: Dict[str, Any]


def replay_teacher(episode: Episode, docs: List[List[Dict[str, Any]]],
                   ) -> Tuple[List[TeacherCall], Optional[str], Optional[BeliefTracker]]:
    """Re-run the belief update of *episode* with the teacher's saved replies.

    Returns ``(calls, error, tracker)``: the teacher's calls before the
    first step whose replay differs from the saved run (the final call on
    the answer turn last), the reason of the difference (None when every step and the final call match), and the
    tracker after the replay (its belief is the teacher's final one when
    everything matched).
    """
    tracker = BeliefTracker(episode, docs, [Criterion(**c) for c in episode.criteria])
    calls: List[TeacherCall] = []
    steps = [(t, s.teacher_output, s.teacher_failed, s.teacher_records) for t, s in enumerate(episode.steps)]
    steps.append((None, episode.teacher_final_output, episode.teacher_final_failed, episode.teacher_final_records))
    for t, output, failed, saved in steps:
        name = f"iteration {episode.steps[t].iteration}" if t is not None else "the final call"
        call = tracker.open_step(t) if t is not None else tracker.open_final()
        if call is None:
            if output is not None or failed:
                return calls, f"{name}: the saved run made the call, the replay does not", tracker
            continue
        if output is None and not failed:
            return calls, f"{name}: the replay makes the call, the saved run did not", tracker
        reply = None if failed else output
        outcome = tracker.submit(reply)
        for key in RECORD_KEYS:
            if _canonical(outcome.get(key)) != _canonical(saved.get(key)):
                return calls, f"{name}: replayed {key} differ from the saved ones", tracker
        if call.is_scorer and outcome["certainty_label"] != episode.teacher_final_label:
            return calls, (f"{name}: replayed final status {outcome['certainty_label']!r}, saved "
                           f"{episode.teacher_final_label!r}"), tracker
        calls.append(TeacherCall(call, reply, outcome))
    return calls, None, tracker
