"""The run's settings as the stages write them: the run directory's name and
``run_config.json`` (built, checked against a resumed run, updated).

``run_config.json`` holds every setting that changes what the agent does
(:func:`build_run_config`, the generate stage), the belief stage's
settings per belief model (``belief_models``, :func:`build_belief_config`)
and the evaluation's judge model; the per-query files carry no run settings.
Reading it: ``utils.run_files.run_config``.
"""

import json as _json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.io import atomic_write_text, file_lock
from utils.run_files.run_config import BELIEF_CONFIG_SECTION, RUN_CONFIG_FILE, belief_config_of, read_run_config


def build_run_name_for_pipeline(policy_model: str, llm_model: str, **kwargs) -> str:
    """Build a consistent output directory name for the current pipeline run.

    Template: {agent_name}_{backend}_{model}
    e.g. ``glm_api_glm-4.7-flash``, ``oss_20b_vllm_gpt-oss-20b`` or
    ``react_wo_plan_vllm_claude-sonnet-4-6``.  *policy_model* is the CLI
    ``--policy-model`` name (``oss_20b``, not the internal alias ``oss``).
    Dataset/retriever/query_key info lives in the parent dataset_dir; the
    fixed searcher config lives in the run_config.json sidecar.
    """
    from utils.models import resolve_agent_backend, model_display_name

    name = policy_model
    if policy_model == "react":
        name = "react_w_plan" if kwargs.get("use_plan", False) else "react_wo_plan"
    backend, _slug = resolve_agent_backend(policy_model)
    return f"{name}_{backend}_{model_display_name(llm_model)}"


#: Sections of ``run_config.json`` that the trajectories of one run
#: directory share: a generate stage with other values is refused
#: (``check_run_config``), a belief or eval stage with another ``dataset``
#: section too (``check_run_dataset``).  ``belief_models`` is the belief
#: stage's: ``{model: settings}``, one entry per belief model (its short
#: ``--belief-model`` name, the folder ``belief/{model}/``); other settings
#: of a model replace that model's beliefs only (``belief_config_diffs``).
#: ``evaluation`` may change between calls (it only re-judges).
LOCKED_RUN_CONFIG_SECTIONS = ("dataset", "agent", "searcher")


#: Keys of the locked sections recorded for information only: a path that
#: moves with the data root (another machine, ``$DRA_DATA_ROOT``) while the
#: dataset and split, which name the documents, stay the same.  A change is
#: printed (:func:`print_moved_paths`), not refused; a run saved before the
#: key is accepted.
UNLOCKED_RUN_CONFIG_KEYS = {"dataset": ("corpus_path",)}


def build_run_config(policy_model: str, llm_model: str, *, dataset: str, split: str,
                     query_key: str, **kwargs) -> Dict[str, Any]:
    """The configuration of a run's generate stage, saved as
    ``run_dir/run_config.json`` (:func:`check_run_config`).

    Every setting that changes what the agent does is here, with defaults
    resolved, so the run directory's name stays short and the per-query
    files carry no run settings; the belief stage adds its own section
    (:func:`build_belief_config`).  *policy_model* is the CLI
    ``--policy-model`` name (``oss_20b``, not the internal alias ``oss``);
    ``max_output_tokens_total`` is the value the agent uses
    (:func:`utils.models.effective_max_output_tokens`).
    """
    from utils.models import effective_max_output_tokens, model_display_name, resolve_agent_backend

    backend, slug = resolve_agent_backend(policy_model)
    return {
        "dataset": {
            "dataset": dataset,
            "split": split,
            "query_key": query_key,
            "min_relevance_score": kwargs.get("min_relevance_score"),
            # The documents the agent searched (information, not locked:
            # UNLOCKED_RUN_CONFIG_KEYS).
            "corpus_path": kwargs.get("corpus_path"),
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
            "max_passage_tokens": kwargs.get("max_passage_tokens"),
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
        # The OpenRouter hosts the agent's calls are pinned to (OSS on the
        # API only; information, not a locked section: the pin is code).
        "openrouter_provider": _openrouter_provider(policy_model, backend),
        "evaluation": {
            "judge_model": kwargs.get("judge_model"),
        },
    }


def build_belief_config(dataset: str, *, max_passage_tokens: Optional[int], **kwargs) -> Dict[str, Any]:
    """One belief model's entry of the ``belief_models`` section of
    ``run_config.json``: the belief stage's settings (the belief cap as ``resolve_dataset_defaults`` set
    it from the dataset).  *max_passage_tokens* is the run's (its agent
    section): the belief update reads the text the agent was shown."""
    from belief_model_component import belief_provider_pin
    from indexing_corpus_dataset.shared.layout import effective_belief_source

    return {
        "belief_model": kwargs.get("belief_model") or None,
        "max_beliefs": kwargs.get("max_beliefs"),
        "belief_source": kwargs["belief_source"],
        "belief_source_effective": effective_belief_source(kwargs["belief_source"], dataset),
        "max_passage_tokens": max_passage_tokens,
        # The OpenRouter hosts its calls are routed to (None: local or unpinned).
        "openrouter_provider": belief_provider_pin(kwargs.get("belief_model") or ""),
    }


def _openrouter_provider(policy_model: str, backend: str) -> Optional[Dict[str, Any]]:
    """The provider pin of the agent's OpenRouter calls, None when it has
    none (an agent other than OSS, or served by vLLM)."""
    from utils.models import POLICY_MODEL_ALIAS
    if backend != "api" or POLICY_MODEL_ALIAS.get(policy_model, policy_model) != "oss":
        return None
    from deep_research_agents.agents.oss_agent import OPENROUTER_PROVIDER_PIN
    return dict(OPENROUTER_PROVIDER_PIN)


def run_config_diffs(old: Dict[str, Any], new: Dict[str, Any],
                     sections=LOCKED_RUN_CONFIG_SECTIONS) -> List[str]:
    """``section.key: old -> new`` for every key of *sections* that differs
    (the keys of ``UNLOCKED_RUN_CONFIG_KEYS`` left out)."""
    diffs = []
    for section in sections:
        a, b = old.get(section) or {}, new.get(section) or {}
        skip = UNLOCKED_RUN_CONFIG_KEYS.get(section, ())
        diffs += [f"{section}.{k}: {a.get(k)!r} -> {b.get(k)!r}"
                  for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)]
    return diffs


def print_moved_paths(old: Dict[str, Any], new: Dict[str, Any]) -> None:
    """Print each key of ``UNLOCKED_RUN_CONFIG_KEYS`` whose saved value
    (*old*, when it has one) differs from this call's (*new*)."""
    for section, keys in UNLOCKED_RUN_CONFIG_KEYS.items():
        a, b = old.get(section) or {}, new.get(section) or {}
        for k in keys:
            if a.get(k) is not None and a.get(k) != b.get(k):
                print(f"[WARN] {section}.{k} of the run: {a.get(k)!r}; this call reads {b.get(k)!r}. "
                      "It must hold the same documents.")


def _write(path: Path, config: Dict[str, Any]) -> None:
    atomic_write_text(path, _json.dumps(config, indent=2))


def check_run_config(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Write *config* (:func:`build_run_config`) to ``run_dir/run_config.json``,
    keeping its belief section; when the run directory already has one (a
    resumed run), first check that the locked sections
    (:data:`LOCKED_RUN_CONFIG_SECTIONS`) are unchanged, so every query of
    the directory ran with the same settings.  Raises SystemExit with the
    differences otherwise.  Every writer of the file holds its lock
    (``utils.io.file_lock``), so stages running at once lose no update."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    with file_lock(path):
        saved = read_run_config(run_dir)
        if saved:
            print_moved_paths(saved, config)
            diffs = run_config_diffs(saved, config)
            if diffs:
                raise SystemExit(
                    f"{path} was written with other settings than this run's: {'; '.join(diffs)}. "
                    "Its queries would mix both; use the same settings, another --output, or remove the run directory."
                )
        if BELIEF_CONFIG_SECTION in saved:
            config = {**config, BELIEF_CONFIG_SECTION: saved[BELIEF_CONFIG_SECTION]}
        _write(path, config)


def belief_config_diffs(run_dir: Union[str, Path], model: str, section: Dict[str, Any]) -> List[str]:
    """How *section* (:func:`build_belief_config`) differs from the saved
    entry of the belief model *model* (``key: old -> new``), empty when it
    is the same or the run has none.  Reads only."""
    old = belief_config_of(read_run_config(run_dir), model)
    if old is None:
        return []
    return run_config_diffs({model: old}, {model: section}, sections=(model,))


def _require_run_config(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """The run's ``run_config.json``; SystemExit when it has none."""
    saved = read_run_config(run_dir)
    if not saved:
        raise SystemExit(f"No run in {run_dir} (no run_config.json): run --stage generate first.")
    return saved


def set_belief_config(run_dir: Union[str, Path], model: str, section: Dict[str, Any]) -> None:
    """Set the entry of the belief model *model* in the run's
    ``run_config.json`` to *section* (:func:`build_belief_config`); the
    other models' entries are kept.  The run must have a ``run_config.json``
    (the generate stage's).  The belief stage calls it after removing the
    model's beliefs of other settings (:func:`belief_config_diffs`), so an
    entry never names settings its beliefs were not made with."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    with file_lock(path):
        saved = _require_run_config(run_dir)
        models = saved.setdefault(BELIEF_CONFIG_SECTION, {})
        if models.get(model) != section:
            models[model] = section
            _write(path, saved)


def check_run_dataset(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """The belief and eval stages: stop when the ``dataset`` section of
    *config* (:func:`build_run_config`, from this call's settings) differs
    from the run's.  These stages read the saved run with this call's
    qrels and corpus, which must be the generate stage's (a moved corpus
    path is printed, :func:`print_moved_paths`)."""
    saved = _require_run_config(run_dir)
    print_moved_paths(saved, config)
    diffs = run_config_diffs(saved, config, sections=("dataset",))
    if diffs:
        raise SystemExit(f"{Path(run_dir) / RUN_CONFIG_FILE} was generated with other dataset settings than "
                         f"this call's: {'; '.join(diffs)}. Use the generate stage's settings.")


def update_run_config_evaluation(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Set the ``evaluation`` section of the run's ``run_config.json`` to
    *config*'s (the belief and eval stages: the section that may change
    between calls); the other sections are left as they are.  A run without
    the file is left alone."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    if not path.exists():
        return
    with file_lock(path):
        saved = read_run_config(run_dir)
        if saved.get("evaluation") != config["evaluation"]:
            saved["evaluation"] = config["evaluation"]
            _write(path, saved)
