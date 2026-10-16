"""Per-query belief file ``belief/{query_id}.jsonl``.

Saves what ``BeliefModel`` records into one JSONL file per query
under ``belief/``: a ``meta`` line, then one ``step`` line per search
iteration.  The files are the input of the offline analysis and of the
scoring model g_phi.  Every line carries ``query_id``; correctness labels
stay in the accuracy output (``accuracy.jsonl``, same ``query_id`` key).
The per-iteration seen doc ids live in the trajectory file
(``trajectory/{query_id}.jsonl``).

Per-query JSONL schema (``belief/{query_id}.jsonl``), schema_version 14::

    line 1  {"record": "meta", "schema_version": 14, "query_id": "q1", "question": "...",
             "agent": "react", "llm_model": "...", "dataset": "browsecomp_plus",
             "llm_criteria": "...", "max_criteria": 8,
             "criteria_source": "llm", "belief_updater": "llm:...",
             "query_scorer": "llm:...", "encoder": "...",
             "num_criteria": 4, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6,
             "criteria": [{"id": "c1", "text": "...", "kind": "closed"}, ...],
             "criteria_info": {"model": ..., "query_shape": "set", "errors": []},
             "final_belief": ["almost certain", "unknown", ...],
             "final_criteria_attempts": [2, 0, ...],
             "criteria_evidence": [{"id": "c1", "kind": "closed", "status": "almost certain",
                                    "evidence": [{"doc_id": "d1", "step": 1, "role": "support",
                                                  "spans": ["..."], "span_verified": [true]}]}, ...]}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "doc_novelty": 1.0, "criteria_delta": 5, "query_novelty": 1.0,
             "new_item_precision": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "recall_so_far": 0.3333, "num_relevant_seen": 2, "num_relevant": 6,
             "intermediate_answers": ["..."], "intermediate_answer_status": "ok",
             "subqueries": ["..."],
             "queries": [{"text": "...", "max_sim_to_earlier": null, "novelty": 1.0,
                          "target_scores": [1.0, 0.5, ...]}],
             "docs": [{"doc_id": "d1", "seen_before": false, "novelty": 1.0}, ...],
             "belief_before": ["unknown", ...], "belief_after": ["almost certain", ...],
             "criteria_sources_after": [1, 0, ...],
             "criteria_targeted": ["c1"], "criteria_attempts_after": [1, 0, ...],
             "criteria_updates": [{"id": "c1", "from": "unknown", "to": "almost certain",
                                   "written": "almost certain", "support": [{"doc_id": "d1", ...}],
                                   "contradict": [], "reason": "...", "note": ""}, ...],
             "belief_updater_output": "...",
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


SCHEMA_VERSION = 15


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
            f.write(_dump({"record": "step", "query_id": query_id, **step}) + "\n")
    os.replace(tmp_path, path)
