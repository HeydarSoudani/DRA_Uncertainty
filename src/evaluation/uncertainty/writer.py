"""Per-query uncertainty file ``uncertainty/{query_id}.jsonl``.

Saves what ``UncertaintyEstimator`` records into one JSONL file per query
under ``uncertainty/``: a ``meta`` line, then one ``step`` line per search
iteration.  The files are the input of the offline analysis and of the
scoring model g_phi.  Every line carries ``query_id``; correctness labels
stay in the accuracy output (``accuracy.jsonl``, same ``query_id`` key).
The per-iteration seen doc ids live in the trajectory file
(``trajectory/{query_id}.jsonl``).

Per-query JSONL schema (``uncertainty/{query_id}.jsonl``), schema_version 8::

    line 1  {"record": "meta", "schema_version": 8, "query_id": "q1", "question": "...",
             "agent": "react", "llm_model": "...", "dataset": "browsecomp_plus",
             "llm_criteria": "...", "max_criteria": 8,
             "criteria_source": "llm", "criteria_judge": "llm:...",
             "query_scorer": "llm:...", "encoder": "...",
             "num_criteria": 4, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6, "total_gain": 9,
             "criteria": [{"id": "c1", "text": "...", "kind": "closed"}, ...],
             "criteria_info": {"model": ..., "query_shape": "set", "errors": []},
             "final_criteria_state": ["fully_covered", "uncovered", ...],
             "final_criteria_attempts": [2, 0, ...],
             "criteria_evidence": [{"id": "c1", "kind": "closed", "status": "fully_covered", "missing": "",
                                    "evidence": [{"doc_id": "d1", "step": 1, "role": "support",
                                                  "spans": ["..."], "span_verified": [true]}]}, ...]}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "doc_novelty": 1.0, "criteria_delta": 3, "query_novelty": 1.0,
             "new_item_precision": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "recall_so_far": 0.3333, "num_relevant_seen": 2, "num_relevant": 6,
             "new_item_graded_recall": 0.3333, "new_gain": 3, "total_gain": 9,
             "graded_recall_so_far": 0.3333, "gain_seen": 3,
             "intermediate_answers": ["..."], "intermediate_answer_status": "ok",
             "subqueries": ["..."],
             "queries": [{"text": "...", "max_sim_to_earlier": null, "novelty": 1.0,
                          "target_scores": [1.0, 0.5, ...]}],
             "docs": [{"doc_id": "d1", "seen_before": false, "novelty": 1.0}, ...],
             "criteria_state_before": ["uncovered", ...], "criteria_state_after": ["fully_covered", ...],
             "criteria_targeted": ["c1"], "criteria_attempts_after": [1, 0, ...],
             "criteria_updates": [{"id": "c1", "from": "uncovered", "to": "fully_covered",
                                   "proposed": "fully_covered", "support": [{"doc_id": "d1", ...}],
                                   "contradict": [], "reason": "...", "missing": "", "applied": true, "note": ""}, ...],
             "criteria_judge_output": "...",
             "intermediate_answer_reasoning": "...", "errors": []}
    ...

A criterion's ``kind`` is ``closed`` (one fact) or ``open`` (several parts
or answers; fully covered only with several supporting documents), new in
schema_version 7.  schema_version 8 drops ``criteria_eval`` (the criteria
are assessed offline, ``analysis/criteria_reachability.py``).  ``iteration`` counts from 1 for every agent; ``agent_iteration`` is the
agent's own counter.  A signal that could not be computed is null, never 0,
and every float is finite and rounded to 4 decimals.  A query without any
search still gets its meta line.  The file is written atomically.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


SCHEMA_VERSION = 8


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


def save_uncertainty(query_id: str, question: str, result: Dict[str, Any], output_dir) -> None:
    """Write ``{output_dir}/{query_id}.jsonl`` from the result dict.

    Reads ``uncertainty_meta`` and ``uncertainty_steps`` (attached by
    ``_attach_uncertainty_stats`` in the agent mixin); does nothing when the
    estimator was off.  Writes to a temporary file first, so an interrupted
    write never leaves a truncated file.
    """
    meta = result.get("uncertainty_meta")
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
        for step in result.get("uncertainty_steps") or []:
            f.write(_dump({"record": "step", "query_id": query_id, **step}) + "\n")
    os.replace(tmp_path, path)
