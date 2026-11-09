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

Per-query JSONL schema (``belief/{query_id}.jsonl``), schema_version 32::

    line 1  {"record": "meta", "schema_version": 32, "query_id": "q1", "question": "...",
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
             "statements": [{"id": "c1", "status": "certain", "selected": ["Osby"],
                             "text": "It is certain that c1 is covered, by \"Osby\": d1 states ...",
                             "cited": ["d1"], "support_cited": ["d1"]}, ...],
             "changed": ["c1"],
             "observations": [{"id": "c1", "label": "t1", "step": 1, "spans": ["..."]},
                              {"id": "c1", "label": "d1", "step": 1, "doc_id": "doc17", "spans": ["..."]}],
             "issues": [{"id": "c2", "code": "no_span", "detail": "no verified span", "label": "d3"}],
             "agent_turn": "...",
             "reply": "...",
             "intermediate_answer_reasoning": null, "errors": []}
    ...
    last    {"record": "final", "query_id": "q1", "iteration": 9, "certainty": 0.6934,
             "statements": [...], "changed": [...], "observations": [...], "issues": [...],
             "references": [{"id": "c1", "references": [{"label": "d1", "step": 1, "doc_id": "doc17",
                                                          "title": "...", "spans": ["..."]}, ...]}, ...],
             "agent_turn": "... Final answer: ...", "reply": "...", "errors": []}

The fields are described in ``belief_model_component.belief_model``
(step and final records, meta line) and ``belief_model_component.criteria``
(``BeliefState.apply``: the issue codes).  A criterion's ``kind`` is
``closed`` (one fact with one answer), ``rest`` (the members of a set the
extractor did not list), ``aggregate`` (how a set query's answer follows
from the members' values) or ``aspect`` (one aspect of a report topic).  A
step line is the iteration's update plus the light state: ``statements``
is every criterion's statement after the step, ``observations`` the
observations that entered in the step (with their verified spans).  The
final line's ``certainty`` is s_T and its ``references`` gathers every
criterion's observations over the sample.  The readable belief is in the
trajectory md (``belief_model_component.belief_view``).

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


SCHEMA_VERSION = 32

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
