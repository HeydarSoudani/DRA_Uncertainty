"""The inference CLI's run configuration: the file-backed knobs
(``experiments/configs/dra_inference.yaml``), the dataset defaults, and the
kwargs of :func:`pipeline.run_pipeline`.
"""

import asyncio
import logging
from pathlib import Path

import yaml

from indexing_corpus_dataset.shared.layout import apply_dataset_defaults, dataset_root, default_corpus_path, default_index_dir
from utils.documents import MAX_PASSAGE_TOKENS
from utils.yaml_config import apply_config_to_args, validate_choices

logger = logging.getLogger(__name__)


# ============================================================================
# File-backed run configuration
# ============================================================================
# The mostly-fixed pipeline variables live in a YAML config file
# (experiments/configs/dra_inference.yaml) instead of as argparse `add_argument`
# calls.  The frequently-varied knobs stay on the CLI; everything below is
# sourced from the file but can still be overridden by an explicit CLI flag.
#
# Precedence (lowest -> highest):
#   FILE_BACKED_DEFAULTS  <  YAML config file  <  explicit CLI flag

# Built-in fallbacks, used only when a key is absent from the YAML file so that
# downstream ``args.<attr>`` access never raises.  Keep in sync with the
# shipped experiments/configs/dra_inference.yaml.
FILE_BACKED_DEFAULTS = {
    # LLM
    # None = auto: resolved per agent by utils.models.resolve_temperature.
    "llm_temperature": None,
    "llm_max_tokens_per_call": 2000,
    "llm_top_p": 1.0,
    "max_output_tokens_total": 120000,
    "request_timeout": 300,
    # Dataset (null -> auto-resolved from --dataset)
    "data_path": None,
    "subset": None,
    "dataset_year": None,
    "query_key": None,
    "qrels_data_path": None,
    "min_relevance_score": None,
    "query_ids_file": None,
    # Retriever
    "index_dir": None,
    "corpus_path": None,
    "top_k": 100,
    "seen_top_k": 5,
    # Searcher
    "post_retrieval_reranker": "null",
    "post_fusion_reranker": "null",
    "rerank_top_k": 100,
    "retrieval_input": "subquery",
    "post_fusion_reranker_input": "original_query",
    # Agent-specific
    "use_plan": False,
    "max_extend_steps": 5,
    "hard_mode": True,
    "max_passage_tokens": MAX_PASSAGE_TOKENS,
    # Belief model (the belief stage; the model itself is --belief-model)
    "max_beliefs": None,
    "belief_source": "gold",
    "belief_workers": 8,
    # Evaluation
    "judge_model": "openrouter/qwen/qwen3-32b",
    # Execution & output
    "output": None,
    "max_iteration": 100,
    "max_retries": 3,
    "verbose": True,
    "gpu_ids": None,
}

# Allowed values for the file-backed enum knobs (argparse ``choices=``
# equivalents) so a typo in the YAML or a CLI override fails fast.
_RERANKER_CHOICES = ["null", "batched_reranker", "rankllama", "rank1",
                     "qwen3_reranker", "listwise", "rank_r1", "monot5"]
FILE_BACKED_CHOICES = {
    "post_retrieval_reranker": _RERANKER_CHOICES,
    "post_fusion_reranker": _RERANKER_CHOICES,
    "retrieval_input": ["subquery", "original_query+subquery", "reasoning+subquery"],
    "post_fusion_reranker_input": ["original_query", "original_query+subqueries",
                                   "original_query+reasoning", "reasoning+subqueries"],
    "belief_source": ["gold", "silver"],
}


# A CLI override is coerced to the type of its config default -- which carries
# no type when that default is ``None`` (the auto-resolved knobs).  These keys
# name the type explicitly so ``--llm-temperature 0.3`` still arrives as a float
# rather than as the string "0.3".
FILE_BACKED_TYPES = {
    "llm_temperature": float,
    "min_relevance_score": int,
    "max_beliefs": int,
    "max_passage_tokens": int,
}


def load_run_config(config_path) -> dict:
    """Load the YAML run-config of file-backed (mostly-fixed) variables.

    Returns a dict over the FILE_BACKED_DEFAULTS keys: built-in fallbacks
    overlaid with whatever the YAML file provides.  Unknown keys in the file
    are ignored with a warning.
    """
    merged = dict(FILE_BACKED_DEFAULTS)
    if not config_path:
        return merged
    path = Path(config_path)
    if not path.exists():
        print(f"WARNING: config file not found: {config_path} — using built-in defaults")
        return merged
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - set(FILE_BACKED_DEFAULTS)
    if unknown:
        logger.warning("Ignoring unknown keys in %s: %s", config_path, sorted(unknown))
    merged.update({k: v for k, v in data.items() if k in FILE_BACKED_DEFAULTS})
    validate_choices(merged, FILE_BACKED_CHOICES)
    return merged


def apply_run_config(args, config: dict, overrides: dict) -> None:
    """Merge the run config (:func:`load_run_config`) and the CLI overrides
    onto ``args`` (:func:`utils.yaml_config.apply_config_to_args`, with the
    types and choices of the file-backed knobs)."""
    apply_config_to_args(args, config, overrides, types=FILE_BACKED_TYPES, choices=FILE_BACKED_CHOICES)


def resolve_dataset_defaults(args) -> None:
    """Auto-select dataset-related CLI arguments that were left as None.

    Split, query key, relevance threshold and belief cap come from the
    dataset's :data:`~indexing_corpus_dataset.shared.layout.DATASET_SPECS` entry.
    """
    if args.data_path is None:
        args.data_path = str(dataset_root(args.dataset))
        print(f"Auto-selected data path: {args.data_path}")

    apply_dataset_defaults(args, ("dataset_year", "subset", "query_key", "min_relevance_score", "max_beliefs"))

    if args.index_dir is None:
        args.index_dir = str(default_index_dir(args.dataset))
        print(f"Auto-selected index directory: {args.index_dir}")
    if args.corpus_path is None:
        args.corpus_path = str(default_corpus_path(args.dataset, args.subset))
        print(f"Auto-selected corpus path: {args.corpus_path}")

    if getattr(args, "qrels_data_path", None) is None:
        args.qrels_data_path = args.data_path


def assemble_pipeline_kwargs(args, llm_client, retriever, workers: int, verbose: bool, gpu_ids=None) -> dict:
    """Build the kwargs dict to pass to run_pipeline() (includes worker_config)."""
    pipeline_kwargs = {
        "retriever":                  retriever,
        "llm_client":                 llm_client,
        "llm_model":                  args.llm_model,
        "top_k":                      args.top_k,
        "seen_top_k":                 args.seen_top_k,
        "dataset":                    args.dataset,
        "min_relevance_score":        args.min_relevance_score,
        "max_iteration":              args.max_iteration,
        "max_retries":                args.max_retries,
        "llm_max_tokens_per_call":    args.llm_max_tokens_per_call,
        "retriever_name":             args.retriever,
    }

    pipeline_kwargs["qrels_data_path"] = args.qrels_data_path
    pipeline_kwargs["corpus_path"] = args.corpus_path
    pipeline_kwargs["temperature"] = args.llm_temperature
    pipeline_kwargs["llm_top_p"] = args.llm_top_p
    pipeline_kwargs["max_output_tokens_total"] = args.max_output_tokens_total
    pipeline_kwargs["use_plan"] = args.use_plan
    pipeline_kwargs["max_extend_steps"] = args.max_extend_steps
    pipeline_kwargs["hard_mode"] = args.hard_mode
    pipeline_kwargs["max_passage_tokens"] = args.max_passage_tokens

    pipeline_kwargs["belief_model"] = args.belief_model
    pipeline_kwargs["belief_model_name"] = args.belief_model_name
    pipeline_kwargs["max_beliefs"] = args.max_beliefs
    pipeline_kwargs["belief_source"] = args.belief_source
    pipeline_kwargs["belief_workers"] = args.belief_workers

    # Keys named as the CLI args: a worker rebuilds its LLM client and
    # retriever with setup_llm / setup_retriever_from_args on this dict.
    worker_config = {
        "policy_model":            args.policy_model,
        "policy_model_cli":        args.policy_model_cli,
        "llm_model":               args.llm_model,
        "llm_temperature":         args.llm_temperature,
        "llm_max_tokens_per_call": args.llm_max_tokens_per_call,
        "llm_top_p":               args.llm_top_p,
        "request_timeout":         args.request_timeout,
        "dataset":                 args.dataset,
        "retriever":               args.retriever,
        "index_dir":               args.index_dir,
        "corpus_path":             args.corpus_path,
        "top_k":                   args.top_k,
        "seen_top_k":              args.seen_top_k,
        "max_iteration":           args.max_iteration,
        "max_extend_steps":        args.max_extend_steps,
        "max_retries":             args.max_retries,
        "hard_mode":               args.hard_mode,
        "max_passage_tokens":       args.max_passage_tokens,
        "verbose":                 verbose,
        "gpu_ids":                 gpu_ids or [],  # real GPU ids; empty: CPU workers
        "post_retrieval_reranker_type": args.post_retrieval_reranker,
        "post_fusion_reranker_type":    args.post_fusion_reranker,
        "rerank_top_k":                 args.rerank_top_k,
        "retrieval_input":              args.retrieval_input,
        "post_fusion_reranker_input":   args.post_fusion_reranker_input,
        "max_output_tokens_total":       args.max_output_tokens_total,
        "use_plan":                      args.use_plan,
        "quiet":                         args.quiet,
    }

    pipeline_kwargs["workers"]       = workers
    pipeline_kwargs["worker_config"] = worker_config
    pipeline_kwargs["policy_model_cli"] = args.policy_model_cli
    return pipeline_kwargs


def cleanup_event_loop() -> None:
    """Cancel any pending async tasks and close the current event loop."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            if not loop.is_closed():
                loop.close()
    except Exception:
        pass
