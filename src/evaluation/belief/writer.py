"""Per-query belief file ``belief/{query_id}.jsonl``.

Saves what ``BeliefModel`` records into one JSONL file per query
under ``belief/``: a ``meta`` line, then one ``step`` line per search
iteration, then one ``final`` line (the final scorer on the agent's
answer turn, or on a report one more update, when it has one).  The files are the input of the offline analysis
and of belief model training.  Every line carries ``query_id``;
correctness labels stay in the accuracy output (``accuracy.jsonl``, same
``query_id`` key).  The run settings (agent, mode, criteria and belief
update models, criteria cap, retriever) are in the run's
``run_config.json``; the per-iteration seen doc ids, queries and
``<belief_state>`` tags in the trajectory file
(``trajectory/{query_id}.jsonl``).

Per-query JSONL schema (``belief/{query_id}.jsonl``), schema_version 45::

    line 1  {"record": "meta", "schema_version": 45, "query_id": "q1", "question": "...",
             "num_criteria": 3, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6,
             "criteria": [{"id": "c1", "text": "...", "kind": "closed"}, ...],
             "criteria_info": {"model": ..., "query_shape": "multi_hop", "errors": [], "bank": "hit"},
             "tokens": {"input_tokens": 51234, "output_tokens": 8120, "total_tokens": 59354, "num_calls": 9}}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "new_item_precision": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "recall_so_far": 0.3333, "num_relevant_seen": 2, "num_relevant": 6,
             "certainty": 0.4807, "called": true,
             "intermediate_answers": null, "intermediate_answer_status": "disabled",
             "docs": [{"doc_id": "doc17", "new": true, "label": "d1"}, ...],
             "beliefs": [{"id": "c1", "status": "certain", "selected": ["Osby"],
                          "candidates": [{"name": "Osby", "status": "certain", "basis": "passage",
                                          "added_step": 1,
                                          "support": [{"label": "d1", "doc_id": "doc17", "title": "Osby",
                                                       "step": 1, "spans": ["..."], "against": false}],
                                          "claim": "probable", "claim_step": 1}, ...],
                          "proposals": [{"name": "Varby", "claim": "possible", "step": 1,
                                         "added_step": 1}]}, ...],
             "updates": [{"id": "c1", "candidate": "Osby", "reason": "New: d1 states ...",
                          "support": ["d1"], "against": [],
                          "written": "certain", "previous": null, "status": "certain"}, ...],
             "changed": ["c1"],
             "observations": [{"id": "c1", "label": "t1", "step": 1, "candidate": "Osby", "claim": "probable",
                               "spans": ["..."]},
                              {"id": "c1", "label": "d1", "step": 1, "doc_id": "doc17", "spans": ["..."]}],
             "issues": [{"id": "c2", "code": "no_span", "detail": "no verified span", "label": "d3"}],
             "agent_turn": "...",
             "reply": "...",
             "intermediate_answer_reasoning": null, "errors": []}
    ...
    last    {"record": "final", "query_id": "q1", "iteration": 9, "certainty": 0.7,
             "certainty_label": "probable", "certainty_formula": 0.5, "certainty_answer_formula": 0.5,
             "called": true,
             "final_score": {"status": "probable",
                             "candidates": [{"id": "c1", "candidate": "Osby"}, ...],
                             "reason": "...", "issues": []},
             "beliefs": [...], "updates": [], "changed": [], "observations": null, "issues": [],
             "references": [{"id": "c1", "references": [{"label": "d1", "step": 1, "doc_id": "doc17",
                                                          "title": "...", "spans": ["..."]}, ...],
                             "removed": [{"name": "Nils Berg", "status": "possible", "basis": "reasoning",
                                          "added_step": 1, "removed_step": 2, "reason": "Was possible: ...",
                                          "against": true}]}, ...],
             "agent_turn": "... Final answer: ...", "reply": "...", "errors": []}

The fields are described in ``belief_model_component/README.md``
(step and final records, meta line) and ``belief_model_component.criteria.sources``
(``BeliefState.apply``: the issue codes).  A criterion's ``kind`` is
``closed`` (one fact with one answer: a clue, a hop, a member's value),
``aggregate`` (it combines the criteria before it: a multi-clue query's
target, "the <type> that meets c1 to c<k>", or how a multi-member query's
answer follows from the members' values) or ``aspect`` (one aspect of a
report topic).  A step line is the iteration's update plus the light
state: ``beliefs`` is every criterion's belief after the step (an
aspect's ``status`` is its own, its facts' ``status`` "kept"),
``updates`` the reply's candidate updates with their reasons (an
aspect's status item has ``candidate`` null), ``observations`` the
observations that entered in the step (with their verified spans; on a
computed aggregate, a passage states an input it needs).  The
final line's ``certainty`` is s_T and its ``references`` gathers every
criterion's observations over the sample, and the candidates the update
removed from it (``removed``, never shown).  The readable belief is in the
trajectory md (``belief_model_component.views.log_view``).

schema_version 45 (was 44): two status sets.  Kinds ``hop``, ``rest`` and
``target`` are gone: a hop is ``closed`` (one answer), a multi-member
query has no rest criterion, a target is an ``aggregate``.  An
aggregate's statuses are derived by code: the lowest status of its
inputs, "unlikely" when a passage speaks against it (issue
``aggregate_status`` when the reply wrote another).  A report's aspect
has one status of the scale (``unknown`` with no fact; ``certain`` needs
every fact supported by two documents), written by the reply's item
without a candidate; its facts are ``kept`` or removed.  The
``coverage`` field of step and final lines is gone, and the certainty
formula counts every criterion (the target was left out).  Issues
``bad_coverage``, ``coverage_no_passage``, ``coverage_no_fact`` become
``bad_aspect_status``, ``aspect_no_passage``, ``aspect_no_fact``, with
``aspect_one_source``, ``bad_fact_status`` (a fact neither "kept" nor
"removed") and ``not_computed`` (an update naming a computed answer no
claim gives) new.  The reply's ``reasoning_observations`` have
no ``turn`` (code finds it from the span).

schema_version 44 (was 43): same record fields, new numbers.  A status
becomes a probability by the table ``core.types.STATUS_PROBABILITY``
(unlikely 0.10, unknown 0.25, possible 0.50, probable 0.70, almost
certain 0.90, certain 0.97; was (level + 2) / 6) in every number read
from a status (``certainty`` of the final scorer, ``certainty_formula``,
``certainty_answer_formula``, a step's ``certainty``).  The certainty
formula of a short-answer query is the lowest criterion probability (was
the geometric mean); of a report, their mean.

schema_version 43 (was 42): step and final lines add ``coverage``, the
coverage updates of a report's aspects (criterion kind ``aspect``; an
aspect's status is its coverage, "unknown" while no passage supports a
fact of it); a multi-member criterion's subject is never its candidate
(issue ``subject_candidate``), and an aggregate takes passage
observations (a member's conditions); the document cut is
``max_passage_tokens`` (the first tokens, was ``max_passage_chars``).

Added within schema 42 (no new version, so earlier runs stay readable):
the meta line's ``tokens``, the belief update and final scorer tokens of
the sample (null when not counted; absent in earlier files).  A multi-clue
query (BrowseComp-Plus, shape renamed from single_target; set renamed
multi_member) lists its clues, then a target criterion ("the <type> that
meets c1 to c<k>", kind ``target``), then the asked answer when it is not
the target.  The target takes no observation or update (issues
``derived_criterion``, ``derived_update``); its candidates are derived
after each call (the names listed on every clue, at the lowest of their
statuses there, with their support), and it is left out of s_t and
``certainty_answer_formula``.  Only multi-clue files change; earlier
BrowseComp-Plus files have no target criterion.

schema_version 40 (was 39): a proposal records the step it was proposed
in (``added_step``; ``step`` is its latest claim's), so a dropped
proposal's ``added_step`` is that step; past ``MAX_PROPOSALS`` the least
recently claimed proposal is dropped (as before), and an update without
a passage is now held to the same cap.  ``certainty_answer_formula`` scores an answer's candidate a
passage ruled out as "unlikely" (was "unknown").  New issues: "New:" on a
proposed name (``wrong_previous``), ``unknown_candidate`` on a rest
criterion.  Criterion ids in a reply are read case-insensitively ("C1"),
and a label string with several labels ("d1, d2") is split.

schema_version 39 (was 38): the final scorer writes the status of the
final answer on the belief's scale (unlikely, possible, probable, almost
certain, certain: ``certainty_label``; ``certainty`` its
``core.types.status_probability``) in place of an ICD 203 likelihood, and per
criterion only the answer's candidate (``final_score``: ``{status,
candidates, reason, issues}``, no check).  Its prompt shares the belief
update's definitions and status definitions.

schema_version 38 (was 37): s_T of a short-answer query is the final
scorer's (``belief_model_component.update.final_score``): one call reads b_{T-1}
and the answer turn (no final update) and writes, per criterion, the
answer's candidate and its check, a reason and a likelihood (a word of
ICD 203, ``certainty_label``; ``certainty`` its probability, null when
the call failed).  The final line adds ``certainty_label``,
``certainty_formula`` (the fixed function of b_{T-1}),
``certainty_answer_formula`` (its form on the checks' candidates) and
``final_score``; a report keeps the final update (``certainty`` =
``certainty_formula``, the others null).  A removed candidate records
``against`` (ruled out: a passage or the agent's claim spoke against
it, not a proposal dropped for room); the final scorer reads those names
as removed.  One document length for the agent and the belief
update (the run's ``max_passage_tokens``, recorded in run_config's
``belief_model``): report documents are cut too.  A passage's title is
its metadata title, else its front matter's, else its first line.

schema_version 37 (was 36): a criterion is judged on its own parts: a part
that refers to another criterion is met when a passage links the candidate
to a candidate of that criterion, whatever that candidate's status, so a
change of the referenced criterion no longer lets a status move without
new support (``no_new_support`` again only for one answer, another
candidate's support changed).  A call is made only for a passage new to
the belief or on the final turn: no "cited again" passages
(``passages[]`` drops ``again``).  Criteria refer only to criteria before
them: gold multi-hop texts are written in a chain order (the row's
criteria reordered and renumbered), and an extraction with a forward
reference is asked once more (``criteria_info.warnings`` "order: ...").

schema_version 36 (was 35): a multi-hop criterion names only its own hop
and refers to the hop it depends on by its id ("the engineer who was born
in the town of c1"); its hops before the answer are of kind ``hop``
("several answers": candidates do not compete), the answer ``closed``.
The agent's reasoning only proposes names: a reasoning observation is a
claim (``{candidate, claim}``: a status, or "rejected") on one candidate;
a claimed name no passage supports is a proposal of its criterion
(``beliefs[].proposals``, no status, not a rival), a passage that
supports it makes it a candidate, a rejected claim drops it.  A
candidate's support is saved with it (``support``: the passage
observations that support it or speak against it, with their spans and
titles) and shown to the next update as quoted spans with the passage
title (no labels); its basis follows from it.  A candidate update writes
the passages of the call that support the candidate or speak against it
(``updates[].support`` / ``against``); "removed" needs one that speaks
against it.  A name that passages only speak against is not added (a
proposal stays one: ``against_only``).  Hard limits (code): a candidate
without support is at most "possible" (``no_support``, replaces
``turn_only``); a candidate whose update adds no support keeps its status
unless, for a one-answer criterion, another candidate's support changed
in the call (a proposal added or dropped does not count), or a criterion
its text refers to changed (``no_new_support``, replaces
``turn_only_change``).  New flags: ``certain_with_rival`` and
``unequal_tie`` (replace ``competing_candidates``), ``bad_claim``;
``repeated_observation`` dropped (a passage shown again adds its new
spans).  A step makes a call only when it has a new passage or its turn
cites a judged one again (shown "cited again": an id in citation brackets
or after "doc" / "docid" / "id"; a bare number elsewhere is no citation);
otherwise ``called`` is false, the belief is carried forward and the turn
is read by the next call (the reply's claims carry their turn label, its
passage observations the step of the call), the final call always made.

schema_version 35 (was 34): the status of a candidate is defined per
criterion type, in the prompt of the dataset's query shape
(``prompts/belief_update/criterion_statuses.txt``): "one answer" (its
candidates compete: the status counts the other candidates, "certain" only
when no other meets a part), "several answers" (the passages on the
candidate only) and "computed" (the agent's computation).  Prompt Rule 5
is dropped.  "doubtful" is no longer a status (scale "unlikely" ..
"certain", levels -1 .. 4): a candidate a passage rules out, or a
reasoning-only one the turn rejects, is removed (the update writes
"removed"; ``updates`` record ``status: "removed"``), kept in the final
line's ``references[].removed``; a removal without the observation that
can make it is kept and flagged (``bad_removal``, ``remove_unlisted``), a
removed candidate added again is flagged (``readded``).  A listed passage
candidate of a one-answer criterion may change without a new passage when
another of its "passage" candidates changes in the step.  ``competing_candidates``
flags more than one candidate above "possible".  The certainty score's q_k
is the criterion's status probability (no share between candidates).

schema_version 34 (was 33): the candidates of a one-answer criterion
compete (prompt Rule 5: a candidate is at most "possible" when another
meets as many parts of the criterion, at most "unlikely" when another
meets more).  A reasoning-only candidate, and a new one in a step with
no new passage, is moved into "unlikely" .. "possible" (``turn_only``;
was: set to "possible"), so a reasoning rejection is kept; a listed
reasoning-only candidate may move in that range, a listed passage one
still keeps its status without a new passage (``turn_only_change``).
``competing_candidates`` flags a candidate above "possible" with another
at "possible" or above (was: two above "possible").  The certainty score
divides a one-answer criterion's q_k by max(1, the summed probability of
its candidates at "possible" or above).  Same record fields.

schema_version 33 (was 32) replaces the statement of a criterion by its
candidates: per criterion a list of candidates, each with its status, its
basis ("passage" or "reasoning", set by code) and the step it was added
in; the update writes, per candidate that is new or whose status changes,
a reason ("Was <status>: ..." or "New: ...", saved in ``updates``, never
shown) and the new status, and code keeps every other candidate, so the
list only grows.  ``statements`` becomes ``beliefs`` (``{id, status,
selected, candidates}``: the criterion's status is the highest of its
candidates, "unknown" without one; candidates are listed highest status
first, then "passage" before "reasoning", then in the order added; the
selected candidates are those at "possible" or above, for a one-answer
criterion only the first listed); new ``updates``; ``changed`` the
criteria with a candidate added or whose status changed.  The agent's tag
shows the candidates, no ``<sources>``.  Reasoning only names candidates: a
reasoning-only candidate is "possible" (``turn_only``); in a step with no
new passage on the criterion, a new candidate is "possible" too
(``turn_only``) and a listed one keeps its status, up or down, whatever
its basis (``turn_only_change``, replaces ``turn_only_drop``).  A
candidate's name is matched in spans and titles as whole words.  New issue codes: ``empty_candidate``, ``bad_status``,
``duplicate_candidate``, ``unexpected_candidate``, ``no_reason_opening``,
``wrong_previous``, ``not_listed``, ``listed_as_new``, ``near_duplicate``,
``unchanged``, ``competing_candidates``; ``no_opening``,
``extra_selected``, ``no_selected``, ``duplicate_statement`` and
``unexpected_statement`` dropped.

schema_version 32 (was 31): passages set the status.  The update writes
a statement only for a criterion whose belief changes, so a criterion
with a new observation and no statement keeps its own (``missing_statement``
dropped), and ``changed`` lists the criteria whose statement the reply
rewrote (was: those with a new observation).  New issue code
``turn_only_drop``: a status that rested on a passage, lowered in a step
with no new passage observation on the criterion, is not taken (the old
statement is kept).  ``unknown_candidate`` also accepts the title of a
passage observed on the criterion.

schema_version 31 (was 30) keeps the belief at two levels.  Shown: the
statement of a criterion is free text with a fixed opening (status,
selected candidates), written by the update and read back verbatim by the
next update (no quotes added) and by the agent; a statement record is
``{id, status, selected, text, cited, support_cited}`` (no ``by``,
``rivals``, ``reason``).  Saved: an observation is the spans of its source
on the criterion, with no value (``{id, label, step, [doc_id], spans}``),
a think observation's spans verified against the turn; the final line adds
``references``.  The agent's turn no longer lists its search queries
(``query_value`` dropped).  New issue codes: ``no_opening`` (replaces
``no_status``), ``unknown_candidate``, ``extra_selected``, ``no_selected``;
``unknown_value``, ``no_source``, ``extra_leader``, ``no_by`` dropped.

schema_version 30 (was 29) drops the stance of an observation: the
update prompt is split into two subtasks, observations (which values each
source states for each criterion, no label) and statements (which sources
support a value, "given", and which contradict it, "against"); code checks
that a cited source has an observation on the criterion.  An observation
record has no ``stance`` field; the other saved fields are those of 29.

schema_version 29 (was 28) grounds the status in the passages: the
update prompt is rewritten (a passage supports a value only when it states
a condition of the criterion, another entity never contradicts one, a
search query is no value, settled criteria still take new passages), each
cited document in the current statement comes with a quote of its span,
code lowers a status above "possible" that no passage supports
(``turn_only``) and drops a think value that is a search query
(``query_value``), and "doubtful" scores 1/12 instead of 0 in the
certainty.  The saved fields are those of 28.

schema_version 28 (was 27) replaces the belief of a criterion (status,
covered_by, candidates with evidence) by one statement (status, the values
it is covered by and its rivals, each with its sources, reason), weighs
the agent's turns as well as the documents in the status, reads the answer
turn as an ordinary update in the ``final`` line (no answer ratings; meta
``final_*`` dropped), scores certainty from the statuses only (aggregate
criteria now count), and saves one ``issues`` list per step in place of
the per-record notes (``think_observations`` / ``passage_observations`` become
``observations``, ``beliefs_after`` ``statements``, ``belief_updates``
``changed``, ``belief_output`` ``reply``, ``passages`` the ``label`` of
``docs``, ``seen_before`` ``new``).  ``iteration`` counts from 1 for every
agent; ``agent_iteration`` is the agent's own counter.  A signal that could
not be computed is null, never 0, and every float is finite and rounded to
4 decimals.  A query without any search still gets its meta line.  The file
is written atomically.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


SCHEMA_VERSION = 45

# Kept in memory for the agent, not saved (the trajectory step has it).
_UNSAVED_STEP_KEYS = ("belief_state_tag",)


def clean_record(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None,
    floats rounded to 4 decimals, tuples and sets to lists."""
    if isinstance(obj, dict):
        return {str(k): clean_record(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_record(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_record(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return round(value, 4) if math.isfinite(value) else None
    return obj


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(clean_record(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def save_belief(query_id: str, question: str, result: Dict[str, Any], output_dir) -> None:
    """Write ``{output_dir}/{query_id}.jsonl`` from the result dict.

    Reads ``belief_meta``, ``belief_steps`` and ``belief_final`` (attached by
    ``_attach_belief_stats`` in the agent mixin); does nothing when the
    belief model was off.  Writes to a temporary file first, so an interrupted
    write never leaves a truncated file.
    """
    meta = result.get("belief_meta")
    if meta is None:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = {
        "record": "meta", "schema_version": SCHEMA_VERSION,
        "query_id": query_id, "question": question, **meta,
    }
    path = out_dir / f"{query_id}.jsonl"
    tmp_path = out_dir / f".{query_id}.jsonl.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(_dump(header) + "\n")
        for step in result.get("belief_steps") or []:
            step = {k: v for k, v in step.items() if k not in _UNSAVED_STEP_KEYS}
            f.write(_dump({"record": "step", "query_id": query_id, **step}) + "\n")
        final = result.get("belief_final")
        if final is not None:
            f.write(_dump({"record": "final", "query_id": query_id, **final}) + "\n")
    os.replace(tmp_path, path)
