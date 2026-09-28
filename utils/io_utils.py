"""Shared experiment-result I/O helpers (local filesystem).

Functions for reading/writing experiment run outputs (JSON, TREC, generation
markdown) and for resuming interrupted pipeline runs.
"""

import json as _json
from pathlib import Path
from typing import Dict, List, Union

from utils.config import AGENT_META_KEYS

try:
    from orjson import loads as _json_loads
except ImportError:
    _json_loads = _json.loads


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


def get_processed_queries(run_dir: Union[str, Path], require_uncertainty: bool = False) -> set:
    """Return query IDs that already have a saved retrieval TREC file.

    Enables resuming an interrupted pipeline run by skipping previously
    completed queries.

    Args:
        run_dir: Root directory of the current run.
            Results are expected under ``{run_dir}/retrieval/surfaced/*.trec``.
        require_uncertainty: Also require ``{run_dir}/uncertainty/{qid}.jsonl``
            (set when the uncertainty estimator is on), so a query whose run
            stopped between the two writes is run again.

    Returns:
        Set of query-ID strings (filename stems of existing ``.trec`` files),
        or an empty set if the surfaced retrieval directory does not exist.
    """
    retrieval_dir = Path(run_dir) / "retrieval" / "surfaced"
    if not retrieval_dir.exists():
        return set()
    try:
        processed = {f.stem for f in retrieval_dir.glob("*.trec")}
        if require_uncertainty:
            uncertainty_dir = Path(run_dir) / "uncertainty"
            processed &= {f.stem for f in uncertainty_dir.glob("*.jsonl")}
        return processed
    except Exception as e:
        print(f"Warning: could not read existing results from {retrieval_dir}: {e}")
        return set()


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
                doc_id, rank, score, run_tag = parts[2], int(parts[3]), float(parts[4]), parts[5]
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


def load_result_from_saved_files(
    run_dir: Union[str, Path], query_id: str, *, lightweight: bool = False,
) -> dict:
    """Reconstruct a full result dict from all saved per-query files.

    Tries to load from the trajectory JSON first (which contains trajectory,
    generation, num_steps, num_searches, and agent-specific metadata).  Falls
    back to TREC-only reconstruction when the trajectory JSON is unavailable.

    Also loads the generation markdown and cited-docs TREC when available and
    the trajectory JSON didn't already provide them.

    Args:
        run_dir:  Root directory of the run.  Contains retrieval/
                  (with surfaced/, seen/, cited/ subdirs), trajectory/,
                  and generation/ subdirectories.
        query_id: Query identifier.
        lightweight: When True, skip the (potentially large) trajectory JSON
                     and reconstruct from TREC + generation + cited-docs files
                     only.  Evaluators need only doc_ids and scores, not the
                     full document text stored in trajectory JSONs.

    Returns:
        Reconstructed result dict, or ``{}`` on failure.
    """
    run_dir_str = str(run_dir).rstrip("/")

    def _file_path(subdir: str, filename: str) -> str:
        return f"{run_dir_str}/{subdir}/{filename}"

    def _read_text(subdir: str, filename: str) -> str:
        return Path(_file_path(subdir, filename)).read_text(encoding="utf-8")

    result: dict = {}

    # ── 1. Try trajectory JSONL (richest source) ────────────────────────────
    # One line per trajectory step plus a ``{"record": "meta", ...}`` line (last
    # in files written online, first in older runs — position is irrelevant here).  The full surfaced ranking is not stored here (it lives in
    # retrieval/surfaced/*.trec, reconstructed in step 5b); uncertainty signals
    # are under uncertainty/{qid}.jsonl (step 6).
    if not lightweight:
        try:
            content = _read_text("trajectory", f"{query_id}.jsonl")
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
            if meta or trajectory:
                result = {
                    "trajectory":     trajectory,
                    "generation":     meta.get("generation", ""),
                    "num_steps":      meta.get("num_steps", 0),
                    "num_searches":   meta.get("num_searches", 0),
                    "num_iterations": meta.get("num_iterations"),
                }
                for key in AGENT_META_KEYS:
                    if meta.get(key):
                        result[key] = meta[key]
        except (FileNotFoundError, OSError):
            pass
        except Exception as e:
            print(f"Warning: could not load trajectory JSONL for {query_id}: {e}")

    # ── 2. Fall back to TREC if no trajectory loaded ─────────────────────────
    if not result:
        try:
            trec_path = _file_path("retrieval/surfaced", f"{query_id}.trec")
            result = load_result_from_trec(trec_path, query_id)
        except (FileNotFoundError, OSError):
            pass

    if not result:
        return {}

    # ── 3. Load generation from MD if not in trajectory JSON ─────────────────
    if not result.get("generation"):
        try:
            result["generation"] = _read_text("generation", f"{query_id}.md")
        except Exception:
            pass

    # ── 4. Load cited docs from TREC if not already present ──────────────────
    if not result.get("cited_docs_ranked_list"):
        try:
            content = _read_text("retrieval/cited", f"{query_id}.trec")
            doc_ids = []
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2]:
                    doc_ids.append({"doc_id": parts[2]})
            if doc_ids:
                result["cited_docs_ranked_list"] = doc_ids
        except Exception:
            pass

    # ── 5. Load seen-doc iterations from TREC if not in trajectory ──────────
    if not result.get("seen_docs_iterations"):
        try:
            seen_trec_path = _file_path("retrieval/seen", f"{query_id}.trec")
            seen_parsed = load_result_from_trec(seen_trec_path, query_id)
            if seen_parsed:
                seen_iters = []
                for step in seen_parsed.get("trajectory", []):
                    docs = step.get("docs", [])
                    seen_iters.append([{"doc_id": d["doc_id"]} for d in docs if d.get("doc_id")])
                if seen_iters:
                    result["seen_docs_iterations"] = seen_iters
        except Exception:
            pass

    # ── 5b. Reconstruct surfaced iterations from surfaced TREC ───────────────
    # The trajectory JSONL keeps only the seen (top-k) doc ids, not the full
    # surfaced ranking.  Rebuild the per-step surfaced lists from the surfaced
    # TREC so SurfacedDocEvaluator stays correct on resume.  (Fresh runs skip
    # this: their in-memory trajectory still holds the full ``docs`` lists.)
    has_surfaced = any(step.get("docs") for step in result.get("trajectory", []))
    if not has_surfaced and not result.get("surfaced_docs_iterations"):
        try:
            surf_trec_path = _file_path("retrieval/surfaced", f"{query_id}.trec")
            surf_parsed = load_result_from_trec(surf_trec_path, query_id)
            surf_iters = [
                step["docs"]
                for step in surf_parsed.get("trajectory", [])
                if step.get("docs")
            ]
            if surf_iters:
                result["surfaced_docs_iterations"] = surf_iters
        except (FileNotFoundError, OSError):
            pass

    # ── 6. Load uncertainty signals from uncertainty/{qid}.jsonl ─────────────
    # A ``{"record": "meta", ...}`` line, then one ``{"record": "step", ...}``
    # line per search iteration (see ``evaluation.uncertainty_evaluator``).
    # Rebuilt into the ``uncertainty_meta`` / ``uncertainty_steps`` keys the
    # agent attaches, so UncertaintyEvaluator can aggregate them on resume.
    if "uncertainty_meta" not in result:
        try:
            content = _read_text("uncertainty", f"{query_id}.jsonl")
            meta = None
            steps = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = _json_loads(line)
                record = obj.pop("record", None)
                obj.pop("query_id", None)
                if record == "meta":
                    obj.pop("question", None)
                    obj.pop("schema_version", None)
                    meta = obj
                elif record == "step":
                    steps.append(obj)
            if meta is not None:
                result["uncertainty_meta"] = meta
                result["uncertainty_steps"] = steps
        except (FileNotFoundError, OSError):
            pass
        except Exception as e:
            print(f"Warning: could not load uncertainty JSONL for {query_id}: {e}")

    return result


# ===========================================================================
# Run / output directory naming
# ===========================================================================

def build_run_name_for_pipeline(agentic_model: str, llm_model: str, **kwargs) -> str:
    """Build a consistent output directory name for the current pipeline run.

    Template: {agent_name}_{backend}_{model}
    e.g. ``glm_api_glm-4.7-flash`` or ``react_wo_plan_vllm_claude-sonnet-4-6``.
    Dataset/retriever/query_key info lives in the parent dataset_dir; the
    fixed searcher config lives in the run_config.json sidecar.
    """
    from utils.config import resolve_agent_backend, model_display_name

    name = agentic_model
    if agentic_model == "react":
        name = "react_w_plan" if kwargs.get("use_plan", False) else "react_wo_plan"
    backend, _slug = resolve_agent_backend(agentic_model)
    return f"{name}_{backend}_{model_display_name(llm_model)}"


def build_uncertainty_config_name(**kwargs) -> str:
    """Build the uncertainty-estimator directory name (the run's varying knob).

    Searcher/retrieval settings are fixed and recorded in run_config.json
    rather than the path.  Only the estimator mode and, when it is on, the
    criteria judge are surfaced here, so a resumed run never mixes judges.

    Examples: ``ue-off``, ``ue-monitor_nli``, ``ue-monitor_llm_novel``.
    """
    mode = kwargs.get("uncertainty_estimator", "off")
    name = f"ue-{mode}"
    if mode != "off":
        name += f"_{kwargs.get('criteria_judge') or 'none'}"
    if kwargs.get("ensure_novel_seen_docs", False):
        name += "_novel"
    return name


def write_run_config(run_dir: Union[str, Path], agentic_model: str,
                     llm_model: str, **kwargs) -> None:
    """Persist the full run configuration to ``run_dir/run_config.json``.

    Captures the fixed searcher/retrieval settings that used to live in the
    folder name, plus agent/model/uncertainty-estimator metadata, so nothing is lost when
    the path is simplified.
    """
    from utils.config import resolve_agent_backend, model_display_name

    backend, slug = resolve_agent_backend(agentic_model)
    config = {
        "agent": {
            "agentic_model": agentic_model,
            "backend": backend,
            "openrouter_slug": slug,
            "llm_model": llm_model,
            "model_display": model_display_name(llm_model),
            "use_plan": kwargs.get("use_plan", False),
            "ua_max_turns": kwargs.get("ua_max_turns", 8),
            "ua_max_passage_chars": kwargs.get("ua_max_passage_chars", 1500),
            "ua_max_format_retries": kwargs.get("ua_max_format_retries", 2),
            "ua_max_tokens_per_call": kwargs.get("ua_max_tokens_per_call", 4096),
            "ua_disable_native_thinking": kwargs.get("ua_disable_native_thinking", True),
            "ua_show_novelty": kwargs.get("ua_show_novelty", True),
            "ua_show_criteria": kwargs.get("ua_show_criteria", True),
            "ua_criteria_mode": kwargs.get("ua_criteria_mode", "auto"),
            "ua_criteria_model": kwargs.get("ua_criteria_model", ""),
            "ua_max_criteria": kwargs.get("ua_max_criteria", 8),
            "ua_stabilization_window": kwargs.get("ua_stabilization_window", 15),
            "ua_criteria_max_tokens": kwargs.get("ua_criteria_max_tokens", 1024),
            "ua_evidence_top_k": kwargs.get("ua_evidence_top_k", 5),
            "ua_evidence_chars": kwargs.get("ua_evidence_chars", 1500),
        },
        "searcher": {
            "retriever_name": kwargs.get("retriever_name", "e5"),
            "seen_top_k": kwargs.get("seen_top_k", 10),
            "rerank_top_k": kwargs.get("rerank_top_k", 100),
            "post_retrieval_reranker_name": kwargs.get("post_retrieval_reranker_name", "null"),
            "post_fusion_reranker_name": kwargs.get("post_fusion_reranker_name", "null"),
            "retrieval_input": kwargs.get("retrieval_input", "subquery"),
            "post_fusion_reranker_input": kwargs.get("post_fusion_reranker_input", "original_query"),
            "ensure_novel_seen_docs": kwargs.get("ensure_novel_seen_docs", False),
        },
        "uncertainty_estimator": {
            "mode": kwargs.get("uncertainty_estimator", "off"),
            "llm_criteria": kwargs.get("llm_criteria"),
            "max_criteria": kwargs.get("max_criteria", 8),
            "criteria_judge": kwargs.get("criteria_judge", "none"),
            "criteria_judge_model": kwargs.get("criteria_judge_model", ""),
        },
    }
    path = Path(run_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        _json.dump(config, fh, indent=2)
