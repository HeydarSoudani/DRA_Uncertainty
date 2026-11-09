"""Per-query belief file ``belief/{query_id}.jsonl``.

Saves what ``BeliefModel`` records into one JSONL file per query
under ``belief/``: a ``meta`` line, then one ``step`` line per search
iteration, then one ``final`` line (the update on the agent's answer
turn, when it has one).  The files are the input of the offline analysis
and of belief model training.  Every line carries ``query_id``;
correctness labels stay in the accuracy output (``accuracy.jsonl``, same
``query_id`` key).  The run settings (agent, mode, criteria and belief
update models, criteria cap, retriever) are in the run's
``run_config.json``; the per-iteration seen doc ids, queries and
``<belief_state>`` tags in the trajectory file
(``trajectory/{query_id}.jsonl``).

Per-query JSONL schema (``belief/{query_id}.jsonl``), schema_version 35::

    line 1  {"record": "meta", "schema_version": 35, "query_id": "q1", "question": "...",
             "num_criteria": 3, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6,
             "criteria": [{"id": "c1", "text": "...", "kind": "closed"}, ...],
             "criteria_info": {"model": ..., "query_shape": "multi_hop", "errors": [], "bank": "hit"}}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "new_item_precision": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "recall_so_far": 0.3333, "num_relevant_seen": 2, "num_relevant": 6,
             "certainty": 0.4807,
             "intermediate_answers": null, "intermediate_answer_status": "disabled",
             "docs": [{"doc_id": "doc17", "new": true, "label": "d1"}, ...],
             "beliefs": [{"id": "c1", "status": "certain", "selected": ["Osby"],
                          "candidates": [{"name": "Osby", "status": "certain", "basis": "passage",
                                          "added_step": 1}, ...]}, ...],
             "updates": [{"id": "c1", "candidate": "Osby", "reason": "New: d1 states ...",
                          "written": "certain", "previous": null, "status": "certain"}, ...],
             "changed": ["c1"],
             "observations": [{"id": "c1", "label": "t1", "step": 1, "spans": ["..."]},
                              {"id": "c1", "label": "d1", "step": 1, "doc_id": "doc17", "spans": ["..."]}],
             "issues": [{"id": "c2", "code": "no_span", "detail": "no verified span", "label": "d3"}],
             "agent_turn": "...",
             "reply": "...",
             "intermediate_answer_reasoning": null, "errors": []}
    ...
    last    {"record": "final", "query_id": "q1", "iteration": 9, "certainty": 0.6934,
             "beliefs": [...], "updates": [...], "changed": [...], "observations": [...], "issues": [...],
             "references": [{"id": "c1", "references": [{"label": "d1", "step": 1, "doc_id": "doc17",
                                                          "title": "...", "spans": ["..."]}, ...],
                             "removed": [{"name": "Nils Berg", "status": "possible", "basis": "reasoning",
                                          "added_step": 1, "removed_step": 2, "reason": "Was possible: ..."}]}, ...],
             "agent_turn": "... Final answer: ...", "reply": "...", "errors": []}

The fields are described in ``belief_model_component.belief_model``
(step and final records, meta line) and ``belief_model_component.criteria``
(``BeliefState.apply``: the issue codes).  A criterion's ``kind`` is
``closed`` (one fact with one answer), ``rest`` (the members of a set the
extractor did not list), ``aggregate`` (how a set query's answer follows
from the members' values) or ``aspect`` (one aspect of a report topic).  A
step line is the iteration's update plus the light state: ``beliefs``
is every criterion's belief after the step, ``updates`` the reply's
candidate updates with their reasons, ``observations`` the
observations that entered in the step (with their verified spans).  The
final line's ``certainty`` is s_T and its ``references`` gathers every
criterion's observations over the sample, and the candidates the update
removed from it (``removed``, never shown).  The readable belief is in the
trajectory md (``belief_model_component.belief_view``).

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


SCHEMA_VERSION = 35

# Kept in memory for the agent, not saved (the trajectory step has it).
_UNSAVED_STEP_KEYS = ("belief_state_tag",)


def _clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None,
    floats rounded to 4 decimals, tuples and sets to lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return round(value, 4) if math.isfinite(value) else None
    return obj


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(_clean(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


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
