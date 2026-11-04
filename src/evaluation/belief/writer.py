"""Per-query belief file ``belief/{query_id}.jsonl``.

Saves what ``BeliefModel`` records into one JSONL file per query
under ``belief/``: a ``meta`` line, then one ``step`` line per search
iteration.  The files are the input of the offline analysis and of the
scoring model g_phi.  Every line carries ``query_id``; correctness labels
stay in the accuracy output (``accuracy.jsonl``, same ``query_id`` key).
The run settings (agent, mode, criteria and belief update models, criteria
cap, retriever) are in the run's ``run_config.json``; the per-iteration seen doc
ids, queries and ``<belief_state>`` tags in the trajectory file
(``trajectory/{query_id}.jsonl``).

Per-query JSONL schema (``belief/{query_id}.jsonl``), schema_version 24::

    line 1  {"record": "meta", "schema_version": 24, "query_id": "q1", "question": "...",
             "num_criteria": 3, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6,
             "criteria": [{"id": "c1", "text": "...", "kind": "closed"}, ...],
             "criteria_info": {"model": ..., "query_shape": "multi_hop", "errors": [], "bank": "hit"},
             "final_nuggets": [[{"value": "Osby", "status": "certain",
                                 "evidence": [{"label": "d1", "stance": "supports", "value": "Osby"},
                                              {"label": "t2", "stance": "supports", "value": "Osby"}]}], [], ...],
             "final_certainty": 0.5,
             "final_update": {"agent_turn": "...", "think_observations": [...], "nugget_updates": [...],
                              "belief_output": "...", "errors": []} | null}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "new_item_precision": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "recall_so_far": 0.3333, "num_relevant_seen": 2, "num_relevant": 6,
             "certainty": 0.3333,
             "intermediate_answers": ["..."], "intermediate_answer_status": "ok",
             "docs": [{"doc_id": "doc17", "seen_before": false}, ...],
             "nuggets_after": [[{"value": "Osby", "status": "certain",
                                 "evidence": [{"label": "d1", "stance": "supports", "value": "Osby"}]}], [], ...],
             "passages": [{"label": "d1", "doc_id": "doc17"}, ...],
             "think_observations": [{"id": "c1", "label": "t1", "step": 1, "stance": "supports",
                                     "value": "Osby", "grounded": true, "note": ""}],
             "doc_observations": [{"id": "c1", "label": "d1", "step": 1, "stance": "supports", "value": "Osby",
                                   "doc_id": "doc17", "spans": ["..."], "span_verified": [true],
                                   "grounded": true, "note": ""}],
             "nugget_updates": [{"id": "c1", "after": [...], "reason": "...", "note": "",
                                 "issues": [], "carried": [...]}],
             "agent_turn": "Reasoning: ... Search queries: ...",
             "belief_output": "...",
             "intermediate_answer_reasoning": "...", "errors": []}
    ...

A criterion's ``kind`` is ``closed`` (one fact), ``open`` (several parts
or answers), ``rest`` (the members of a set the extractor did not list) or
``aspect`` (one aspect of a report topic), new in schema_version 7; it is
saved only, the belief updater does not use it.  schema_version 8 drops ``criteria_eval`` (the criteria
are assessed offline, ``analysis/criteria_reachability.py``).
schema_version 9 drops the graded gold (``new_item_graded_recall``,
``new_gain``, ``total_gain``, ``graded_recall_so_far``, ``gain_seen``).
schema_version 10 adds ``criteria_sources_after`` (distinct supporting
documents per criterion); only cited passages with a verified span are
attached.  schema_version 11 is the belief model: the criteria state is one
ICD 203 word of estimative probability per criterion,
rewritten every step by the belief updater from every document of the step
(``criteria_delta`` sums the label moves over levels 0 to 6),
``criteria_updates`` has one record per criterion with the label as
``written``, ``criteria_judge`` and ``criteria_judge_output`` become
``belief_updater`` and ``belief_updater_output``, and ``criteria_evidence``
gives each criterion's ``label``.  schema_version 12 renames
``criteria_state_before`` / ``criteria_state_after`` / ``final_criteria_state``
to ``belief_before`` / ``belief_after`` / ``final_belief`` and
``criteria_delta`` to ``belief_delta``; schema_version 13 renames
``belief_delta`` back to ``criteria_delta``.  schema_version 14 replaces
the ICD 203 labels with the seven statuses of Agent-BRACE's belief scale
(``belief_model_component.types.BELIEF_STATUSES``; levels ``STATUS_LEVEL``: doubtful 0,
unknown 1 (b_0), unlikely 2, possible 3, probable 4, almost certain 5,
confirmed 6), so ``criteria_delta`` sums the status moves and a
contradiction lowers it; the belief updater sees only the documents it has
not judged in an earlier step and updates only the criteria they support,
contradict or relate to (``criteria_updates`` notes the others as not
updated); ``criteria_evidence`` gives each criterion's ``status`` (was
``label``).  schema_version 15 makes the scale signed, centred on
"unknown": doubtful -2, unlikely -1 (the passages point against the
criterion), unknown 0 (b_0), possible 1, probable 2, almost certain 3,
certain 4 (was "confirmed"); the positive levels rate how much of the
criterion the passages give, not how many passages give it, so a positive
``criteria_delta`` is more coverage and a negative one evidence against.
schema_version 16 keeps one home per fact: the run settings leave the meta
line (``agent``, ``llm_model``, ``dataset``, ``task``, ``query_shape``,
``llm_criteria``, ``max_criteria``, ``add_intermediate_answer``, ``mode``,
``criteria_source``, ``belief_updater``, ``query_scorer``: see
``run_config.json``; ``encoder``: the retriever's, ``searcher.retriever_name``
there), and the steps drop ``subqueries`` (the texts of
``queries``) and ``belief_state_tag`` (the trajectory step's
``belief_state``).  schema_version 17 adds the reasoner side of the
belief: the updater call of a step also judges, against the agent's turn
of the step (``agent_turn``), whether the agent used the passages that
raised criteria in the previous call (``utilized_updates``); per
criterion, ``utilized_before`` / ``utilized_after`` sum the used gains and
``utilization`` is the used share of the judged gains; the meta line adds
``final_utilized``, ``final_utilization`` and ``final_update`` (the rises of
the last step judged against the answer turn).  schema_version 18
replaces the use judgments with answer candidates: per criterion, the
answers the agent considers for it (from its memory or the passages), each
with a belief status read as how likely it is correct, rewritten by the
updater call of every step from the step's ``agent_turn``
(``candidate_updates``; ``candidates_before`` / ``candidates_after``), and
``reasoning_confidence`` (mean top-candidate probability) in place of
``utilization``; the meta line has ``final_candidates`` and
``final_reasoning_confidence`` in place of ``final_utilized`` and
``final_utilization``, and ``final_update`` reads the answer turn whenever
there is one.  schema_version 19 aligns the criteria of a set query
with the gold criteria of the training pools (members, the rest
criterion, then the aggregate: how the answer follows from the members'
values; no "complete list" criterion); the aggregate has answer
candidates only, so its status is null in ``belief_before`` /
``belief_after`` / ``final_belief`` and its ``criteria_updates`` record
has ``from`` / ``to`` null; ``reasoning_confidence`` leaves out the rest
criterion.  schema_version 20 replaces statuses and candidates with
nuggets (Agent-BRACE style): per criterion, the observations of the
documents (retriever judge: a document supports or contradicts a value,
with verified spans) and of the agent's turns (reasoner judge: the turn
gives or rejects a value, anchored in the turn), each labelled by its
source ("d<n>", numbered per query; "t<step>"), and its nuggets (one per
answer: value, status, evidence labels) rewritten by the belief judge
from the observation log (``doc_observations``, ``think_observations``,
``nugget_updates``, ``nuggets_after``; ``retriever_output``,
``reasoner_output``, ``belief_output``).  Dropped: ``criteria_delta``,
``belief_before`` / ``belief_after`` / ``final_belief``,
``candidates_before`` / ``candidates_after`` / ``final_candidates``,
``criteria_updates``, ``candidate_updates``, ``criteria_evidence``,
``belief_updater_output``; ``reasoning_confidence`` becomes
``belief_confidence`` (criteria with one answer only, null for a report).
The non-belief signals are dropped too: ``doc_novelty``,
``query_novelty``, ``criteria_targeted``, ``criteria_attempts_after``,
``final_criteria_attempts``, ``queries`` (the texts are in ``agent_turn``
and the trajectory) and the ``novelty`` of each ``docs`` entry.
schema_version 21 makes the belief update one call (the same prompt for
the prompted teacher and the trained student): it reads the agent's turn
and the step's new passages and writes the think observations, the
document observations and the new nuggets of the criteria with a new
observation; one raw reply ``belief_output`` replaces
``retriever_output``, ``reasoner_output`` and ``belief_output``.  The
belief is self-contained, with no observation log: each nugget's
``evidence`` holds its sources' observations (``{label, stance, value,
[spans]}``, was a list of labels).  A document is labelled when first
shown to the call (``passages``, new), not at its first observation, so
some labels have no observation.  ``criteria_sources_after`` counts the
documents that support one of the criterion's nuggets' values.
schema_version 22 replaces ``belief_confidence`` / ``final_belief_confidence``
with the certainty score ``certainty`` / ``final_certainty``
(``belief_model_component.certainty``: per criterion its most likely
nugget's probability, after the final call the nugget of the agent's
answer for the answer criterion of a multi-hop query, unknown without a
nugget; the geometric mean over the criteria, rest and aggregate left
out), adds ``issues`` (codes, ``criteria.ISSUE_*``) to every
``nugget_updates`` record (a value of the current belief left without a
nugget is new among them), and keeps a document observation on a label
that is no new passage of the call (``doc_id`` null, noted).
schema_version 23 checks a document observation's spans against the
passage as the call showed it (cut for a short-answer query), so a span
from past the cut is not verified (``span_verified`` false), and carries
a current nugget the reply leaves out (or writes invalidly) over
unchanged: its value is in the record's ``carried`` (only when there is
one) and its issue ``dropped_value`` stays.
schema_version 24 makes a step line this iteration's update plus a light
state: a nugget's evidence entries are ``{label, stance, value}`` (no
``spans``; a source's spans are saved once, in the ``doc_observations``
of the step that observed it), in ``nuggets_after``, ``nugget_updates``'
``after`` and ``final_nuggets``; ``nugget_updates`` drops ``written``
(the list as written is in ``belief_output``; a criterion the reply left
out has the issue ``not_in_reply``); ``criteria_sources_after`` is dropped
(it follows from ``nuggets_after``).  The readable belief is in the
trajectory md (``belief_model_component.belief_view``).
``iteration`` counts from 1 for
every agent; ``agent_iteration`` is the agent's own counter.  A signal that could not be computed is null, never 0,
and every float is finite and rounded to 4 decimals.  A query without any
search still gets its meta line.  The file is written atomically.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


SCHEMA_VERSION = 24

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

    Reads ``belief_meta`` and ``belief_steps`` (attached by
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
    os.replace(tmp_path, path)
