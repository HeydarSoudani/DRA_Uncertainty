"""Per-query belief file ``belief/{model}/{query_id}.jsonl``.

Saves what ``BeliefModel`` records into one JSONL file per query under
``belief/{model}/`` (one folder per belief model): a ``meta`` line, then one
``step`` line per search iteration, then, on a short-answer query, the final
update's ``step`` line (``final_update`` true: the update on the answer
turn's reasoning, without the final answer), then one ``final`` line (the
final scorer on b_T and the final answer; a report has no final update and
its s_T is the certainty of the last step's belief), when the query has an
answer turn. The files are the input of the offline analysis and of belief
model training. Every line carries ``query_id``; correctness labels stay in
the accuracy output (``eval/generation/accuracy.jsonl``, same ``query_id``
key). The run settings (agent, beliefs and belief update models, belief cap,
retriever) are in the run's ``run_config.json``; the per-iteration seen doc
ids and queries in the trajectory file (``trajectory/{query_id}.jsonl``),
which the belief stage reads (``pipeline.belief_stage``).

Per-query JSONL schema (``belief/{model}/{query_id}.jsonl``), schema_version 57::

    line 1  {"record": "meta", "schema_version": 57, "query_id": "q1", "question": "...",
             "num_beliefs": 3, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6,
             "beliefs": [{"id": "B1", "text": "...", "kind": "closed"}, ...],
             "belief_info": {"model": ..., "query_shape": "multi_hop", "errors": [], "bank": "hit"},
             "tokens": {"input_tokens": 51234, "output_tokens": 8120, "total_tokens": 59354, "num_calls": 9}}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0, "final_update": false,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "new_item_precision": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "recall_so_far": 0.3333, "num_relevant_seen": 2, "num_relevant": 6,
             "certainty": 0.4807, "called": true,
             "docs": [{"doc_id": "doc17", "new": true, "label": "d1"}, ...],
             "beliefs": [{"id": "B1", "status": "almost certain", "most_probable": "Osby", "selected": ["Osby"],
                          "candidates": [{"name": "Osby", "status": "almost certain", "basis": "passage",
                                          "added_step": 1,
                                          "support": [{"label": "d1", "doc_id": "doc17", "title": "Osby",
                                                       "step": 1, "spans": ["..."], "against": false}],
                                          "claim": "keep", "claim_step": 1}, ...],
                          "proposals": [{"name": "Varby", "claim": "keep", "step": 1,
                                         "added_step": 1}]}, ...],
             "updates": [{"id": "B1", "reason": "d1 states directly ...",
                          "passages": [{"candidate": "Osby", "stance": "+", "passage": "d1"}],
                          "written": {"most_probable": "Osby", "wep": "almost certain"},
                          "previous": "unknown", "most_probable": "Osby", "status": "almost certain"}, ...],
             "changed": ["B1"],
             "observations": [{"id": "B1", "label": "t1", "step": 1, "candidate": "Osby", "claim": "keep",
                               "spans": ["..."]},
                              {"id": "B1", "label": "d1", "step": 1, "doc_id": "doc17", "candidate": "Osby",
                               "stance": "+", "spans": ["..."]}],
             "issues": [{"id": "B2", "code": "span_unverified", "detail": "no verified span", "label": "d3"}],
             "notes": [{"id": "B1", "code": "candidate_readded", "detail": "'Varby': dropped before, written again"}],
             "agent_turn": "...",
             "reply": "...", "errors": []}
    ...
            {"record": "step", "query_id": "q1", "iteration": 9, "agent_iteration": null, "final_update": true,
             "num_subqueries": 0, "num_docs": 0, "num_new_docs": 0, "new_item_precision": null, ...,
             "certainty": 0.5, "called": true, "docs": [], "beliefs": [...], "updates": [...], ...,
             "agent_turn": "<the answer turn's reasoning>", "reply": "...", "errors": []}
    last    {"record": "final", "query_id": "q1", "iteration": 9, "certainty": 0.7,
             "certainty_label": "probable", "certainty_formula": 0.5,
             "called": true,
             "final_score": {"status": "probable", "reason": "..."},
             "beliefs": [...],
             "references": [{"id": "B1", "references": [{"label": "d1", "step": 1, "doc_id": "doc17",
                                                          "title": "...", "spans": ["..."]}, ...],
                             "removed": [{"name": "Nils Berg", "status": null, "basis": "reasoning",
                                          "added_step": 1, "removed_step": 2,
                                          "reason": "dropped in t2: ..."}]}, ...],
             "agent_turn": "<the final answer>", "reply": "...", "errors": []}

The fields are described in ``belief_model_component/README.md``
(step and final records, meta line) and ``belief_model_component.init.sources``
(``BeliefState.apply``: the issue codes).  A belief's ``kind`` is
``closed`` (one fact with one answer: a clue, a hop, a member's value),
``aggregate`` (it combines the beliefs before it: a multi-clue query's
target, "the <type> that meets B1 to B<k>", or how a multi-member query's
answer follows from the members' values) or ``aspect`` (one aspect of a
report topic).  A step line is the iteration's update plus the light
state: ``beliefs`` is every belief's state after the step (an
aspect's ``status`` is its own, its facts' ``status`` "kept"),
``updates`` the reply's candidate updates with their reasons (an
aspect's status item has ``candidate`` null), ``observations`` the
observations that entered in the step (with their verified spans).  The
final line's ``certainty`` is s_T and its ``references`` gathers every
belief's observations over the sample, and the candidates the update
removed from it (``removed``, never shown).  The readable belief is in
``belief/{model}/{query_id}.md`` (``belief_model_component.views.log_view``).

The changes of every earlier schema are in
``belief_model_component/SCHEMA_CHANGELOG.md``.
"""

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from utils.io import atomic_open, read_jsonl

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 57
#: The oldest schema the evaluation reads (``load_belief_files``): from 54 on
#: every field it reads exists under its current name (54 renamed
#: "criteria" to "beliefs"); an older file is left out.
EVAL_MIN_SCHEMA = 54
#: One folder per belief model: ``{run_dir}/belief/{model}/`` (:func:`belief_dir`).
BELIEF_DIR = "belief"


def belief_dir(run_dir: Union[str, Path], model: str) -> Path:
    """``{run_dir}/belief/{model}/``: the belief files of the belief model
    *model* (its short ``--belief-model`` name, ``utils.models.BELIEF_MODEL_TO_LLM``)."""
    return Path(run_dir) / BELIEF_DIR / model


def belief_models(run_dir: Union[str, Path]) -> List[str]:
    """The belief models with a folder in the run (sorted names)."""
    root = Path(run_dir) / BELIEF_DIR
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def clean_record(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None,
    floats rounded to 4 decimals, tuples and sets to lists."""
    if isinstance(obj, dict):
        return {str(k): clean_record(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_record(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        if isinstance(obj, float):  # np.float64 too (a float subclass)
            value = float(obj)
            return round(value, 4) if math.isfinite(value) else None
        return obj
    # numpy values (imported only when one is met: this package stays light).
    import numpy as np
    if isinstance(obj, np.ndarray):
        return clean_record(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return clean_record(float(obj))
    return obj


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(clean_record(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def save_belief(run_dir: Union[str, Path], model: str, query_id: str, question: str, meta: Dict[str, Any],
                steps: List[Dict[str, Any]], final: Optional[Dict[str, Any]]) -> None:
    """Write ``{run_dir}/belief/{model}/{query_id}.jsonl``: the *meta* line
    (``BeliefModel.meta``), one line per step record (``BeliefModel.steps``)
    and the *final* record (``BeliefModel.final``, none when None).  Writes
    to a temporary file first, so an interrupted write never leaves a
    truncated file.
    """
    out_dir = belief_dir(run_dir, model)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = {
        "record": "meta", "schema_version": SCHEMA_VERSION,
        "query_id": query_id, "question": question, **meta,
    }
    with atomic_open(out_dir / f"{query_id}.jsonl") as f:
        f.write(_dump(header) + "\n")
        for step in steps:
            f.write(_dump({"record": "step", "query_id": query_id, **step}) + "\n")
        if final is not None:
            f.write(_dump({"record": "final", "query_id": query_id, **final}) + "\n")


#: Text fields no evaluation reads (dropped by ``load_belief_files``, which
#: keeps every query of a run in memory).
_TEXT_KEYS = ("agent_turn", "reply", "references")

BeliefFile = Tuple[Optional[Dict], List[Dict], Optional[Dict]]


def read_belief_file(path: Union[str, Path], light: bool = False) -> BeliefFile:
    """``(meta, steps, final)`` of a belief file (``(None, [], None)`` when
    missing); *light* drops the text fields no evaluation reads."""
    path = Path(path)
    lines = read_jsonl(path) if path.exists() else []
    if light:
        lines = [{k: v for k, v in r.items() if k not in _TEXT_KEYS} for r in lines]
    meta = next((r for r in lines if r.get("record") == "meta"), None)
    final = next((r for r in lines if r.get("record") == "final"), None)
    return meta, [r for r in lines if r.get("record") == "step"], final


def load_belief_files(run_dir: Union[str, Path], model: str, query_ids: Iterable[str]) -> Dict[str, BeliefFile]:
    """``{qid: (meta, steps, final)}`` of the belief model *model*'s files
    (light, see ``read_belief_file``), read once for every evaluation; a
    query without one is left out.  A file of an older schema from
    ``EVAL_MIN_SCHEMA`` on is kept: the evaluation reads only fields these
    schemas share (a step's ``certainty``, ``docs``, ``called``,
    ``observations``, ``issues`` and ``notes``; the final line's
    ``certainty``, ``certainty_label``, ``certainty_formula`` and
    ``final_score``), and ``belief_schemas`` reports them; an info line
    counts them by schema.  A file older than ``EVAL_MIN_SCHEMA`` (or with
    no schema) is left out, with a warning."""
    out: Dict[str, BeliefFile] = {}
    old: Counter = Counter()
    too_old: Counter = Counter()
    folder = belief_dir(run_dir, model)
    for qid in query_ids:
        meta, steps, final = read_belief_file(folder / f"{qid}.jsonl", light=True)
        if meta is None:
            continue
        schema = meta.get("schema_version")
        if not isinstance(schema, int) or schema < EVAL_MIN_SCHEMA:
            too_old[schema] += 1
            continue
        if schema != SCHEMA_VERSION:
            old[schema] += 1
        out[qid] = (meta, steps, final)
    if too_old:
        logger.warning("%s: %d belief file(s) older than schema %d (%s) left out of the evaluation;"
                       " run the belief stage again", model, sum(too_old.values()), EVAL_MIN_SCHEMA, dict(too_old))
    if old:
        logger.info("%s: %d belief file(s) of an older schema (%s), not %d: evaluated as they are",
                    model, sum(old.values()), dict(old), SCHEMA_VERSION)
    return out


def belief_schemas(run_dir: Union[str, Path], model: str) -> Dict[Any, int]:
    """``{schema_version: files}`` of the belief model *model*'s files (their meta lines)."""
    counts: Counter = Counter()
    folder = belief_dir(run_dir, model)
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        try:
            with open(path) as f:
                counts[json.loads(f.readline()).get("schema_version")] += 1
        except (OSError, ValueError, AttributeError):
            counts[None] += 1
    return dict(counts)


def believed_queries(run_dir: Union[str, Path], model: str) -> Tuple[set, set]:
    """``(done, stale)`` query ids of the belief stage of the belief model *model*.

    A query is done when ``belief/{model}/{qid}.jsonl`` (written last and
    atomically, :func:`save_belief`) has a meta line of the current
    ``SCHEMA_VERSION``; a file of another schema is stale: the belief stage
    runs the query again (the evaluation scores it as it is until then).
    """
    folder = belief_dir(run_dir, model)
    if not folder.is_dir():
        return set(), set()
    done, stale = set(), set()
    for path in folder.glob("*.jsonl"):
        try:
            with open(path) as f:
                meta = json.loads(f.readline() or "{}")
        except (OSError, ValueError):
            meta = {}
        (done if meta.get("schema_version") == SCHEMA_VERSION else stale).add(path.stem)
    return done, stale
