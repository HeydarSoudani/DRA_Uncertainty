"""Shared experiment-result I/O helpers (local filesystem).

Functions for reading/writing experiment run outputs (JSON, TREC, generation
markdown) and for resuming interrupted pipeline runs.
"""

import json as _json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

from utils.config import AGENT_META_KEYS

try:
    from orjson import loads as _json_loads
except ImportError:
    _json_loads = _json.loads


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write *text* to *path* through a temp file and ``os.replace``, so a
    crash never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """The rows of a JSONL file; blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield _json_loads(line)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """All rows of a JSONL file (:func:`iter_jsonl`)."""
    return list(iter_jsonl(path))


def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    """Write *rows* as JSONL, atomically (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def setup_output_dirs(
    run_dir: Union[str, Path],
    subdirs: List[str],
) -> Dict[str, str]:
    """Create output sub-directories under *run_dir* and return their paths.

    Args:
        run_dir: Root run directory.
        subdirs: List of sub-directory names,
                 e.g. ``["retrieval", "generation", "trajectory"]``.

    Returns:
        Dict mapping each sub-directory name to its full path string.
    """
    rd = Path(str(run_dir).rstrip("/"))
    rd.mkdir(parents=True, exist_ok=True)
    result: Dict[str, str] = {}
    for name in subdirs:
        d = rd / name
        d.mkdir(parents=True, exist_ok=True)
        result[name] = str(d)
    return result


def _last_line(path: Path) -> bytes:
    """The last non-empty line of *path*, read from the end of the file."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        size = 4096
        while True:
            start = max(0, end - size)
            f.seek(start)
            chunk = f.read(end - start).rstrip(b"\n")
            cut = chunk.rfind(b"\n")
            if cut >= 0 or start == 0:
                return chunk[cut + 1:]
            size *= 2


def is_finished(trajectory_file: Union[str, Path]) -> bool:
    """The query's trajectory file ends with its meta line, which
    ``evaluation.trajectory.save_trajectory`` writes after every other file
    of the query."""
    try:
        return _last_line(Path(trajectory_file)).startswith(b'{"record":"meta"')
    except OSError:
        return False


def scan_queries(run_dir: Union[str, Path], require_belief: bool = False) -> Tuple[set, set]:
    """``(finished, unfinished)`` query ids of the run.

    A query is finished when ``trajectory/{qid}.jsonl`` ends with its meta
    line (and, with *require_belief*, ``belief/{qid}.jsonl`` exists: the
    belief model is on); a trajectory file without it is a query whose run
    stopped, and it is run again on resume.
    """
    traj_dir = Path(run_dir) / "trajectory"
    if not traj_dir.is_dir():
        return set(), set()
    belief_dir = Path(run_dir) / "belief"
    finished, unfinished = set(), set()
    for path in traj_dir.glob("*.jsonl"):
        done = is_finished(path) and (not require_belief or (belief_dir / path.name).exists())
        (finished if done else unfinished).add(path.stem)
    return finished, unfinished


def get_processed_queries(run_dir: Union[str, Path], require_belief: bool = False) -> set:
    """The finished query ids of the run (:func:`scan_queries`), skipped
    when an interrupted run is resumed."""
    return scan_queries(run_dir, require_belief)[0]


def load_result_from_trec(trec_file: Union[str, Path], query_id: str) -> dict:
    """Reconstruct a minimal result dict from a saved per-query TREC file.

    The TREC file format written by ``SurfacedDocEvaluator.save_item`` is::

        qid Q0 doc_id rank score iter_N

    Args:
        trec_file: Path to the ``.trec`` file.
        query_id:  Query identifier (used only for warning messages).

    Returns:
        Dict ``{"trajectory": [{"docs": [...]}, ...]}`` or ``{}`` on failure.
    """
    iterations: Dict[int, list] = {}
    try:
        with open(trec_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                doc_id, score, run_tag = parts[2], float(parts[4]), parts[5]
                try:
                    iter_idx = int(run_tag.split("_")[1])
                except (IndexError, ValueError):
                    iter_idx = 1
                iterations.setdefault(iter_idx, []).append(
                    {"doc_id": doc_id, "rank_score": score}
                )
    except Exception as e:
        print(f"Warning: could not load {trec_file}: {e}")
        return {}

    if not iterations:
        return {}

    trajectory = [{"docs": iterations[i]} for i in sorted(iterations)]
    return {"trajectory": trajectory}


def load_result_from_saved_files(run_dir: Union[str, Path], query_id: str) -> dict:
    """Reconstruct a result dict from the saved per-query files.

    The evaluation reads every run back this way, after a run as in
    ``--eval-only``, so both score the same result.  Sources:

    * ``trajectory/{qid}.jsonl``: the steps (seen doc ids, no doc text) and the
      meta line (counts, :data:`utils.config.AGENT_META_KEYS`);
    * ``generation/{qid}.md``: the final generation;
    * ``retrieval/cited``, ``seen`` and ``surfaced`` TRECs: the cited list and
      the per-step seen and surfaced lists.

    Args:
        run_dir:  Root directory of the run.
        query_id: Query identifier.

    Returns:
        Reconstructed result dict, or ``{}`` when the query is not finished
        (:func:`is_finished`).
    """
    run_dir_str = str(run_dir).rstrip("/")

    def _file_path(subdir: str, filename: str) -> str:
        return f"{run_dir_str}/{subdir}/{filename}"

    def _read_text(subdir: str, filename: str) -> str:
        return Path(_file_path(subdir, filename)).read_text(encoding="utf-8")

    # ── 1. Trajectory JSONL ──────────────────────────────────────────────────
    # One line per trajectory step, then the ``{"record": "meta", ...}`` line.
    # The full surfaced ranking is not stored here (it lives in
    # retrieval/surfaced/*.trec, reconstructed in step 4b); belief-model
    # signals are under belief/{qid}.jsonl and are not reloaded.
    try:
        content = _read_text("trajectory", f"{query_id}.jsonl")
    except OSError:
        return {}
    meta: dict = {}
    trajectory: list = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _json_loads(line)
        if obj.get("record") == "meta":
            meta = obj
        else:
            trajectory.append(obj)
    if not meta:
        return {}
    result: dict = {
        "trajectory":     trajectory,
        "num_steps":      meta.get("num_steps", 0),
        "num_searches":   meta.get("num_searches", 0),
        "num_iterations": meta.get("num_iterations"),
    }
    for key in AGENT_META_KEYS:
        if meta.get(key):
            result[key] = meta[key]

    # ── 2. Generation ─────────────────────────────────────────────────────────
    try:
        result["generation"] = _read_text("generation", f"{query_id}.md")
    except OSError:
        result["generation"] = ""

    # ── 3. Cited docs from the cited TREC ────────────────────────────────────
    try:
        content = _read_text("retrieval/cited", f"{query_id}.trec")
    except OSError:
        content = ""
    doc_ids = [{"doc_id": parts[2]} for parts in (line.split() for line in content.splitlines())
               if len(parts) >= 3]
    if doc_ids:
        result["cited_docs_ranked_list"] = doc_ids

    # ── 4. Seen-doc iterations from the seen TREC, when the trajectory has none
    if not any(step.get("seen_docs") for step in trajectory):
        seen_parsed = load_result_from_trec(_file_path("retrieval/seen", f"{query_id}.trec"), query_id)
        seen_iters = [[{"doc_id": d["doc_id"]} for d in step["docs"] if d.get("doc_id")]
                      for step in seen_parsed.get("trajectory", [])]
        if seen_iters:
            result["seen_docs_iterations"] = seen_iters

    # ── 4b. Surfaced iterations from the surfaced TREC ───────────────────────
    # The trajectory JSONL keeps only the seen (top-k) doc ids, not the full
    # surfaced ranking: the per-step surfaced lists come from the surfaced TREC.
    surf_parsed = load_result_from_trec(_file_path("retrieval/surfaced", f"{query_id}.trec"), query_id)
    surf_iters = [step["docs"] for step in surf_parsed.get("trajectory", []) if step.get("docs")]
    if surf_iters:
        result["surfaced_docs_iterations"] = surf_iters

    return result


# ===========================================================================
# Run / output directory naming
# ===========================================================================

def build_dataset_dir_name(dataset: str, split: str, query_key: str, retriever: str) -> str:
    """Name of the per-dataset output directory: ``{dataset}_{split}[_{query_key}]_{retriever}``.

    The query key is part of the name only when it is not the dataset's
    default (layout.DATASET_SPECS), e.g. ``neuclir_2024_news_qwen3_emb_4b`` for
    the default ``request`` and ``neuclir_2024_news_topic_title_qwen3_emb_4b``.
    """
    from indexing_corpus_dataset.layout import DATASET_SPECS

    qk_part = f"_{query_key}" if query_key != DATASET_SPECS[dataset].query_key else ""
    return f"{dataset}_{split}{qk_part}_{retriever}"


def build_run_name_for_pipeline(policy_model: str, llm_model: str, **kwargs) -> str:
    """Build a consistent output directory name for the current pipeline run.

    Template: {agent_name}_{backend}_{model}
    e.g. ``glm_api_glm-4.7-flash``, ``oss_20b_vllm_gpt-oss-20b`` or
    ``react_wo_plan_vllm_claude-sonnet-4-6``.  *policy_model* is the CLI
    ``--policy-model`` name (``oss_20b``, not the internal alias ``oss``).
    Dataset/retriever/query_key info lives in the parent dataset_dir; the
    fixed searcher config lives in the run_config.json sidecar.
    """
    from utils.config import resolve_agent_backend, model_display_name

    name = policy_model
    if policy_model == "react":
        name = "react_w_plan" if kwargs.get("use_plan", False) else "react_wo_plan"
    backend, _slug = resolve_agent_backend(policy_model)
    return f"{name}_{backend}_{model_display_name(llm_model)}"


def build_belief_config_name(**kwargs) -> str:
    """Build the belief-model directory name (the run's varying knob).

    Searcher/retrieval settings are fixed and recorded in run_config.json
    rather than the path.  Only the belief model mode is surfaced here.

    Examples: ``belief-off``, ``belief-monitor``, ``belief-inform``.
    """
    return f"belief-{kwargs.get('belief_mode', 'off')}"


#: Sections of ``run_config.json`` that the queries of one run directory
#: share: a resume with other values is refused (``check_run_config``).
#: ``evaluation`` may change between calls (it only re-judges).
LOCKED_RUN_CONFIG_SECTIONS = ("dataset", "agent", "searcher", "belief_model")


def build_run_config(policy_model: str, llm_model: str, *, dataset: str, split: str,
                     query_key: str, **kwargs) -> Dict[str, Any]:
    """The full resolved configuration of a run, saved as
    ``run_dir/run_config.json`` (:func:`check_run_config`).

    Every setting that changes what the agent does or what the belief model
    writes is here, with defaults resolved (the criteria cap, the updater
    and query scorer models), so the run directory's name stays short and
    the per-query files carry no run settings.  With the belief model off,
    ``belief_model`` holds the mode only.  *policy_model* is the CLI
    ``--policy-model`` name (``oss_20b``, not the internal alias ``oss``);
    ``max_output_tokens_total`` is the value the agent uses
    (:func:`utils.config.effective_max_output_tokens`).
    """
    from utils.config import effective_max_output_tokens, model_display_name, resolve_agent_backend
    from indexing_corpus_dataset.layout import DATASET_SPECS

    backend, slug = resolve_agent_backend(policy_model)
    mode = kwargs["belief_mode"]
    belief: Dict[str, Any] = {"mode": mode}
    if mode != "off":
        llm_criteria = kwargs.get("llm_criteria") or None
        judge = (kwargs.get("belief_judge_model") or llm_criteria) if llm_criteria else None
        max_criteria = kwargs.get("max_criteria")
        belief.update({
            "llm_criteria": llm_criteria,
            "max_criteria": DATASET_SPECS[dataset].max_criteria if max_criteria is None else max_criteria,
            "belief_judge_model": judge,
            "query_scorer_model": (kwargs.get("query_scorer_model") or judge) if judge else None,
            "criteria_bank": bool(kwargs["criteria_bank"]),
            "add_intermediate_answer": bool(kwargs["add_intermediate_answer"]),
        })
    return {
        "dataset": {
            "dataset": dataset,
            "split": split,
            "query_key": query_key,
            "min_relevance_score": kwargs.get("min_relevance_score"),
        },
        "agent": {
            "policy_model": policy_model,
            "backend": backend,
            "openrouter_slug": slug,
            "llm_model": llm_model,
            "model_display": model_display_name(llm_model),
            "temperature": kwargs.get("temperature"),
            "llm_top_p": kwargs.get("llm_top_p"),
            "use_plan": kwargs.get("use_plan", False),
            "max_iteration": kwargs.get("max_iteration"),
            "max_retries": kwargs.get("max_retries"),
            "max_passage_chars": kwargs.get("max_passage_chars"),
            "llm_max_tokens_per_call": kwargs.get("llm_max_tokens_per_call"),
            "max_output_tokens_total": effective_max_output_tokens(policy_model, kwargs["max_output_tokens_total"]),
            "max_extend_steps": kwargs.get("max_extend_steps"),
            "hard_mode": kwargs.get("hard_mode"),
        },
        "searcher": {
            "retriever_name": kwargs.get("retriever_name", "e5"),
            "top_k": kwargs.get("top_k"),
            "seen_top_k": kwargs.get("seen_top_k"),
            "rerank_top_k": kwargs.get("rerank_top_k"),
            "post_retrieval_reranker_name": kwargs.get("post_retrieval_reranker_name", "null"),
            "post_fusion_reranker_name": kwargs.get("post_fusion_reranker_name", "null"),
            "retrieval_input": kwargs.get("retrieval_input", "subquery"),
            "post_fusion_reranker_input": kwargs.get("post_fusion_reranker_input", "original_query"),
        },
        "belief_model": belief,
        "evaluation": {
            "judge_model": kwargs.get("judge_model"),
        },
    }


def run_config_diffs(old: Dict[str, Any], new: Dict[str, Any],
                     sections=LOCKED_RUN_CONFIG_SECTIONS) -> List[str]:
    """``section.key: old -> new`` for every key of *sections* that differs."""
    diffs = []
    for section in sections:
        a, b = old.get(section) or {}, new.get(section) or {}
        diffs += [f"{section}.{k}: {a.get(k)!r} -> {b.get(k)!r}"
                  for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return diffs


def check_run_config(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Write *config* to ``run_dir/run_config.json``; when the run directory
    already has one (a resumed run), first check that the locked sections
    (:data:`LOCKED_RUN_CONFIG_SECTIONS`) are unchanged, so every query of
    the directory ran with the same settings.  Raises SystemExit with the
    differences otherwise."""
    path = Path(run_dir) / "run_config.json"
    if path.exists():
        diffs = run_config_diffs(_json.loads(path.read_text()), config)
        if diffs:
            raise SystemExit(
                f"{path} was written with other settings than this run's: {'; '.join(diffs)}. "
                "Its queries would mix both; use the same settings, another --output, or remove the run directory."
            )
    atomic_write_text(path, _json.dumps(config, indent=2))


def update_run_config_evaluation(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Set the ``evaluation`` section of the run's ``run_config.json`` to
    *config*'s (``--eval-only``: the section that may change between calls);
    the locked sections are left as they are.  A run without the file is
    left alone."""
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        return
    saved = _json.loads(path.read_text())
    if saved.get("evaluation") != config["evaluation"]:
        saved["evaluation"] = config["evaluation"]
        atomic_write_text(path, _json.dumps(saved, indent=2))
