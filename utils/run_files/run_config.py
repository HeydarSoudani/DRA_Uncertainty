"""The run's settings: the run directory's name and ``run_config.json``.

``run_config.json`` holds every setting that changes what the agent does
(:func:`build_run_config`, the generate stage), the belief stage's section
(:func:`build_belief_config`) and the evaluation's judge model; the
per-query files carry no run settings.
"""

import json as _json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from utils.io_utils import atomic_write_text
from utils.text_utils import cut_chars, cut_passage

RUN_CONFIG_FILE = "run_config.json"


def read_run_config(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """The run's ``run_config.json`` (``{}`` when missing)."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    return _json.loads(path.read_text()) if path.exists() else {}


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


#: Sections of ``run_config.json`` that the trajectories of one run
#: directory share: a generate stage with other values is refused
#: (``check_run_config``), a belief or eval stage with another ``dataset``
#: section too (``check_run_dataset``).  ``belief_model`` is the belief
#: stage's: other values replace the run's beliefs (``belief_config_diffs``).
#: ``evaluation`` may change between calls (it only re-judges).
LOCKED_RUN_CONFIG_SECTIONS = ("dataset", "agent", "searcher")
BELIEF_CONFIG_SECTION = "belief_model"
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
    (:func:`utils.config.effective_max_output_tokens`).
    """
    from utils.config import effective_max_output_tokens, model_display_name, resolve_agent_backend

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
    """The ``belief_model`` section of ``run_config.json``: the belief
    stage's settings (the belief cap as ``resolve_dataset_defaults`` set
    it from the dataset).  *max_passage_tokens* is the run's (its agent
    section): the belief update reads the text the agent was shown."""
    from indexing_corpus_dataset.layout import effective_belief_source

    return {
        "belief_model": kwargs.get("belief_model") or None,
        "max_beliefs": kwargs.get("max_beliefs"),
        "belief_source": kwargs["belief_source"],
        "belief_source_effective": effective_belief_source(kwargs["belief_source"], dataset),
        "max_passage_tokens": max_passage_tokens,
    }


def _openrouter_provider(policy_model: str, backend: str) -> Optional[Dict[str, Any]]:
    """The provider pin of the agent's OpenRouter calls, None when it has
    none (an agent other than OSS, or served by vLLM)."""
    from utils.config import POLICY_MODEL_ALIAS
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


def check_run_config(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Write *config* (:func:`build_run_config`) to ``run_dir/run_config.json``,
    keeping its belief section; when the run directory already has one (a
    resumed run), first check that the locked sections
    (:data:`LOCKED_RUN_CONFIG_SECTIONS`) are unchanged, so every query of
    the directory ran with the same settings.  Raises SystemExit with the
    differences otherwise."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    saved: Dict[str, Any] = {}
    if path.exists():
        saved = _json.loads(path.read_text())
        print_moved_paths(saved, config)
        diffs = run_config_diffs(saved, config)
        if diffs:
            raise SystemExit(
                f"{path} was written with other settings than this run's: {'; '.join(diffs)}. "
                "Its queries would mix both; use the same settings, another --output, or remove the run directory."
            )
    if BELIEF_CONFIG_SECTION in saved:
        config = {**config, BELIEF_CONFIG_SECTION: saved[BELIEF_CONFIG_SECTION]}
    atomic_write_text(path, _json.dumps(config, indent=2))


def belief_model_of(section: Optional[Dict[str, Any]]) -> Optional[str]:
    """The belief model of a saved belief section."""
    return (section or {}).get("belief_model")


def belief_config_diffs(run_dir: Union[str, Path], section: Dict[str, Any]) -> List[str]:
    """How *section* (:func:`build_belief_config`) differs from the belief
    section of the run's ``run_config.json`` (``section.key: old -> new``),
    empty when it is the same or the run has none.  Reads only."""
    saved = _json.loads((Path(run_dir) / RUN_CONFIG_FILE).read_text())
    old = saved.get(BELIEF_CONFIG_SECTION)
    if old is None:
        return []
    return run_config_diffs({BELIEF_CONFIG_SECTION: old}, {BELIEF_CONFIG_SECTION: section},
                            sections=(BELIEF_CONFIG_SECTION,))


def set_belief_config(run_dir: Union[str, Path], section: Dict[str, Any]) -> None:
    """Set the belief section of the run's ``run_config.json`` to *section*
    (:func:`build_belief_config`).  The run must have a ``run_config.json``
    (the generate stage's).  The belief stage calls it after removing the
    beliefs of other settings (:func:`belief_config_diffs`), so the file
    never names settings its beliefs were not made with."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    saved = _json.loads(path.read_text())
    if saved.get(BELIEF_CONFIG_SECTION) != section:
        saved[BELIEF_CONFIG_SECTION] = section
        atomic_write_text(path, _json.dumps(saved, indent=2))


def check_run_dataset(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """The belief and eval stages: stop when the ``dataset`` section of
    *config* (:func:`build_run_config`, from this call's settings) differs
    from the run's.  These stages read the saved run with this call's
    qrels and corpus, which must be the generate stage's (a moved corpus
    path is printed, :func:`print_moved_paths`)."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    saved = _json.loads(path.read_text())
    print_moved_paths(saved, config)
    diffs = run_config_diffs(saved, config, sections=("dataset",))
    if diffs:
        raise SystemExit(f"{path} was generated with other dataset settings than this call's: {'; '.join(diffs)}. "
                         "Use the generate stage's settings.")


def update_run_config_evaluation(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Set the ``evaluation`` section of the run's ``run_config.json`` to
    *config*'s (the belief and eval stages: the section that may change
    between calls); the other sections are left as they are.  A run without
    the file is left alone."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    if not path.exists():
        return
    saved = _json.loads(path.read_text())
    if saved.get("evaluation") != config["evaluation"]:
        saved["evaluation"] = config["evaluation"]
        atomic_write_text(path, _json.dumps(saved, indent=2))


# Before the shared cut (schema 38) the belief update cut a short-answer
# document at 3000 characters and read a report document whole; from
# schema 38 to the token cut the shared cut was ``max_passage_chars``
# characters, ending in "...".
LEGACY_PASSAGE_CHARS = 3000


def _recorded(config: Dict[str, Any], key: str) -> Tuple[bool, Optional[int]]:
    """``(recorded, value)`` of *key* in the belief section of a
    ``run_config.json`` *config*, else in its agent section (the shared cut,
    the same value)."""
    for section in ("belief_model", "agent"):
        values = config.get(section) or {}
        if key in values:
            return True, values[key]
    return False, None


def passage_tokens(config: Dict[str, Any]) -> Optional[int]:
    """Tokens of one document the belief update of a run read (or, a run
    without the belief model, will read in training): its
    ``max_passage_tokens``; None: whole documents.  A run made before the
    token cut (it recorded characters, or nothing) cannot be replayed with
    the same text: ValueError."""
    recorded, value = _recorded(config, "max_passage_tokens")
    if not recorded:
        raise ValueError("the run was made before the shared token cut (no max_passage_tokens in its "
                         "run_config.json); rerun it")
    return value


def passage_cut(config: Dict[str, Any], query_shape: str) -> Callable[[str], str]:
    """The cut of one document as the belief update of a run read it, from
    its ``run_config.json`` *config*: ``max_passage_tokens`` (tokens), else
    an older run's ``max_passage_chars`` (characters), else the cut before
    either was recorded."""
    recorded, tokens = _recorded(config, "max_passage_tokens")
    if recorded:
        return lambda text: cut_passage(text, tokens)
    recorded, chars = _recorded(config, "max_passage_chars")
    if not recorded:
        from belief_model_component import MULTI_ASPECT
        chars = None if query_shape == MULTI_ASPECT else LEGACY_PASSAGE_CHARS
    return lambda text: cut_chars(text, chars)
