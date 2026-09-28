"""Pipeline orchestration: retriever/uncertainty-estimator factories and multi-GPU workers.

This is the wiring layer that assembles heavy component packages
(``uncertainty_estimator``, ``searcher_component``, ``deep_research_agents``,
``evaluation``) into a runnable agent. It is consumed by the ``run_pipeline``
entry point in ``experiments/dra_inference.py``.

Leaf-level helpers (LLM-client factory, CLI arg resolution, output naming) live
in ``utils`` so this module is the only place that depends on the heavy
component packages — keeping ``utils`` a true leaf layer.

The worker functions (_build_components_from_config, _init_worker, gpu_worker)
must stay at module level to be picklable by multiprocessing.spawn.
"""

import contextlib
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from utils.config import (
    OPENROUTER_BASE_URL,
    SELF_MANAGED_LLM_AGENTS,
    _RetrieverConfig,
    get_reranker_configs,
    is_local_finetuned,
    resolve_agent_backend,
)
from reasoner_component import create_generator, disable_native_thinking
from utils.text_utils import _build_cited_docs_ranked_list, build_references_section
from utils.trajectory_logger import TrajectoryLogger

logger = logging.getLogger(__name__)


# ===========================================================================
# Retriever factory
# ===========================================================================

def setup_retriever_from_args(args):
    """Instantiate the local retriever from parsed CLI args."""
    from searcher_component.retriever import (
        BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever,
    )

    config = _RetrieverConfig(
        retriever_name=args.retriever,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        topk=args.top_k,
    )
    if args.dataset == "browsecomp_plus" and args.retriever.startswith("qwen3_emb"):
        config.retrieval_query_max_length = 8196
    if args.dataset == "browsecomp_plus" and args.retriever == "agentir_4b":
        config.retrieval_query_max_length = 8196
    if args.retriever == "bm25":
        return BM25Retriever(config)
    elif args.retriever in ("spladepp", "spladev3"):
        return SPLADERetriever(config)
    elif args.retriever in ["rerank_l6", "rerank_l12"]:
        return RerankRetriever(config)
    else:
        return DenseRetriever(config)


# ===========================================================================
# Uncertainty estimator factory
# ===========================================================================

# Datasets whose answers are short enough for per-step intermediate answers.
INTERMEDIATE_ANSWER_DATASETS = ("browsecomp_plus", "trqa")
# Agents that write a long report rather than a short answer.
NO_INTERMEDIATE_ANSWER_AGENTS = ("cpm_report",)


def build_uncertainty_estimator(
    mode: str,
    retriever=None,
    qrels=None,
    llm_criteria: Optional[str] = None,
    max_criteria: int = 8,
    criteria_judge: str = "none",
    criteria_judge_model: str = "",
    agent=None,
    agentic_model: Optional[str] = None,
    dataset: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    """Build the uncertainty estimator, or None when *mode* is ``"off"``.

    Shared by the main process (run_pipeline) and spawned GPU workers.

    Args:
        mode: ``"off"``, ``"monitor"`` (observe only) or ``"inform"``
            (observe and inject a ``<certainty>`` tag into the trajectory).
        retriever: Retriever whose encoder gives the novelty embeddings
            (dense retrievers only; others leave nu^q null and nu^D id-only).
        qrels: Ground-truth relevance judgements, for marginal recall.
        llm_criteria: Model that extracts each query's criteria.  Without
            it the criteria-based signals are null.
        max_criteria: Cap on the number of criteria per query.
        criteria_judge: ``none``, ``nli`` or ``llm``; the judge behind
            criteria_delta and criteria_targeting.
        criteria_judge_model: NLI model (nli) or judge LLM (llm); "" = the
            default NLI model or *llm_criteria*.
        agent: Agent instance; its ``answer_from_trajectory`` gives the
            per-step intermediate answers.
        agentic_model: Agent type name (answer format, intermediate answer
            gating).
        dataset: Dataset name (intermediate answer gating).
        llm_model: The agent's LLM; saved with the other run settings in
            every meta line.
    """
    if mode == "off":
        return None
    if mode not in ("monitor", "inform"):
        raise ValueError(f"unknown uncertainty estimator mode {mode!r}; expected 'off', 'monitor' or 'inform'")
    criteria_judge = (criteria_judge or "none").lower()

    from uncertainty_estimator import (
        LLMCriteriaSource, UncertaintyEstimator, build_criteria_judges, encode_fn_from_retriever,
    )

    criteria_source = None
    if llm_criteria:
        criteria_source = LLMCriteriaSource(
            llm_client=disable_native_thinking(create_generator(llm_criteria, backend="api")),
            model_name=llm_criteria,
            max_criteria=max_criteria,
        )
        print(f"Uncertainty estimator: criteria from {llm_criteria}")
    else:
        logger.warning("No --llm-criteria model; criteria-based signals disabled")

    encode_fn, encoder_name = encode_fn_from_retriever(retriever) if retriever is not None else (None, None)
    if encode_fn is None:
        logger.warning("Retriever has no local encoder; query_novelty is null and doc_novelty uses ids only")

    intermediate_answer_fn = None
    if (
        agent is not None
        and agentic_model not in NO_INTERMEDIATE_ANSWER_AGENTS
        and dataset in INTERMEDIATE_ANSWER_DATASETS
        and hasattr(agent, "answer_from_trajectory")
    ):
        intermediate_answer_fn = agent.answer_from_trajectory
        print(f"Uncertainty estimator: intermediate answers via {agentic_model}.answer_from_trajectory")

    doc_judge, query_scorer = None, None
    if criteria_source is None:
        if criteria_judge != "none":
            logger.warning("criteria_judge=%s ignored: no criteria source", criteria_judge)
    elif criteria_judge == "llm":
        judge_model = criteria_judge_model or llm_criteria
        doc_judge, query_scorer = build_criteria_judges(
            "llm", model=judge_model, llm_client=disable_native_thinking(create_generator(judge_model, backend="api")),
        )
    else:
        doc_judge, query_scorer = build_criteria_judges(
            criteria_judge, model=criteria_judge_model, encode_fn=encode_fn, encoder_name=encoder_name,
        )
    if doc_judge is not None:
        print(f"Uncertainty estimator: criteria judge {doc_judge.name}, "
              f"query scorer {query_scorer.name if query_scorer else None}")

    return UncertaintyEstimator(
        criteria_source=criteria_source,
        doc_judge=doc_judge,
        query_scorer=query_scorer,
        encode_fn=encode_fn,
        encoder_name=encoder_name,
        qrels=qrels or {},
        intermediate_answer_fn=intermediate_answer_fn,
        agentic_model=agentic_model or "",
        run_info={
            "agent": agentic_model,
            "llm_model": llm_model,
            "dataset": dataset,
            "llm_criteria": llm_criteria or None,
            "max_criteria": max_criteria,
            "mode": mode,
        },
        inform=mode == "inform",
    )


# ===========================================================================
# Agent factory
# ===========================================================================

def build_agent(
    agentic_model: str,
    llm_model: str,
    llm_client=None,
    retriever=None,
    *,
    agentic_model_cli: Optional[str] = None,
    dataset: Optional[str] = None,
    max_iteration: int = 100,
    seen_top_k: int = 5,
    verbose: bool = False,
    search_tool=None,
    use_plan: bool = False,
    max_output_tokens_total: int = 40000,
    temperature: float = 0.0,
    max_extend_steps: int = 5,
    max_retries: int = 3,
    hard_mode: bool = True,
    max_passage_chars: int = 4000,
    ua_max_turns: int = 8,
    ua_max_passage_chars: int = 1500,
    ua_max_format_retries: int = 2,
    ua_max_tokens_per_call: int = 4096,
    ua_disable_native_thinking: bool = True,
):
    """Instantiate an agent and attach its search tool.

    Shared by both the main process (run_pipeline, single-GPU) and spawned GPU
    workers (_init_worker) so the per-agent construction logic lives in one
    place.

    The agent-specific keyword arguments are assembled into ``_reasoning_extra``
    based on ``agentic_model`` and forwarded to the agent constructor.
    """
    from deep_research_agents.agents import AGENT_MAP

    model_class = AGENT_MAP[agentic_model]
    # Backend (API vs vLLM) is resolved on the user-facing CLI name because the
    # OpenRouter registry distinguishes oss_20b / oss_120b, which both collapse
    # to the internal agent name "oss".
    _cli_name = agentic_model_cli or agentic_model
    _reasoning_extra: dict = {}
    if use_plan and agentic_model == "react":
        _reasoning_extra["use_plan"] = True
    if agentic_model == "glm":
        _reasoning_extra["max_output_tokens"] = min(max_output_tokens_total, 20000)
        # Prefer OpenRouter when the agent is in the registry (0 local GPU);
        # otherwise fall through to the agent's vLLM defaults (localhost:6008).
        _backend, _slug = resolve_agent_backend(_cli_name)
        if _backend == "api":
            _reasoning_extra["model_url"] = OPENROUTER_BASE_URL
            _reasoning_extra["model_name"] = _slug
            _reasoning_extra["api_key"] = os.getenv("OPENROUTER_API_KEY")
    elif agentic_model == "oss":
        # Cap per-call output like GLM so the prompt/history still fits inside
        # the 131072-token vLLM window (gpt-oss native max = OpenRouter's max).
        # On the OpenRouter Responses path truncation:"auto" reclaims the full
        # window regardless of this ceiling.
        _reasoning_extra["max_output_tokens"] = min(max_output_tokens_total, 20000)
        # Prefer OpenRouter (Responses API) when oss_20b / oss_120b is in the
        # registry; otherwise use the agent's local vLLM defaults (localhost:6008).
        _backend, _slug = resolve_agent_backend(_cli_name)
        if _backend == "api":
            _reasoning_extra["model_url"] = OPENROUTER_BASE_URL
            _reasoning_extra["model_name"] = _slug
            _reasoning_extra["api_key"] = os.getenv("OPENROUTER_API_KEY")
        else:
            _reasoning_extra["model_name"] = f"openai/{llm_model}"
    elif agentic_model == "tongyi":
        _reasoning_extra["max_tokens_per_step"] = min(max_output_tokens_total, 20000)
    elif agentic_model == "cpm_explore":
        _reasoning_extra["max_output_tokens"] = min(max_output_tokens_total, 16384)
        # The run's resolved temperature (utils.config.resolve_temperature),
        # not a falsy-guarded default: 0.0 is a temperature a user can ask for.
        _reasoning_extra["temperature"] = temperature
    elif agentic_model == "cpm_report":
        _reasoning_extra["max_extend_steps"] = max_extend_steps
        _reasoning_extra["max_retries"] = max_retries
        _reasoning_extra["hard_mode"] = hard_mode
        _reasoning_extra["max_passage_chars"] = max_passage_chars
        _reasoning_extra["model_name"] = llm_model
    elif agentic_model == "uncertainty_aware":
        _reasoning_extra["max_turns"] = ua_max_turns
        _reasoning_extra["max_passage_chars"] = ua_max_passage_chars
        _reasoning_extra["max_format_retries"] = ua_max_format_retries
        _reasoning_extra["max_tokens_per_call"] = ua_max_tokens_per_call
        _reasoning_extra["disable_native_thinking"] = ua_disable_native_thinking

    agent = model_class(
        llm_client=llm_client,
        retriever=retriever,
        max_iteration=max_iteration,
        seen_top_k=seen_top_k,
        verbose=verbose,
        **_reasoning_extra,
    )

    # Inject search tool into agents that inherit from BasicAgent
    # (avoids modifying every subclass constructor).
    if search_tool is not None and hasattr(agent, "search_tool"):
        agent.search_tool = search_tool

    return agent


# ===========================================================================
# Multi-GPU workers (must remain top-level for multiprocessing.spawn pickling)
# ===========================================================================

def _build_components_from_config(worker_config: dict):
    """Rebuild LLM client and retriever from a serializable config dict.

    Called inside spawned worker processes to avoid pickling live objects.

    Returns:
        (llm_client, retriever)  — llm_client may be None for self-managed agents.
    """
    agentic_model = worker_config["agentic_model"]
    llm_model     = worker_config["llm_model"]
    dataset       = worker_config["dataset"]

    llm_client = None

    if agentic_model in SELF_MANAGED_LLM_AGENTS:
        pass
    elif is_local_finetuned(agentic_model, llm_model):
        hf_model = llm_model
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:6008/v1")
        llm_client = create_generator(
            hf_model,
            backend="vllm",
            api_base=api_base,
            api_key="EMPTY",
            litellm_prefix="openai",  # preserve model string "openai/<hf_model>"
            temperature=worker_config["llm_temperature"],
            max_tokens=worker_config["llm_max_tokens_per_call"],
            request_timeout=worker_config.get("request_timeout", 300),
        )
    else:
        # Let create_generator infer the backend from the model name: an
        # ``hf/``-prefixed (or registered) model routes to HFGenerator, ordinary
        # slugs route to APIGenerator.
        llm_client = create_generator(
            llm_model,
            temperature=worker_config["llm_temperature"],
            top_p=worker_config["llm_top_p"],
            max_completion_tokens=worker_config["llm_max_tokens_per_call"],
            metadata={"model": llm_model},
            request_timeout=worker_config.get("request_timeout"),
        )

    from searcher_component.retriever import (
        BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever,
    )

    cfg = _RetrieverConfig(
        retriever_name=worker_config["retriever_type"],
        index_dir=worker_config.get("index_dir"),
        corpus_path=worker_config.get("corpus_path"),
        topk=worker_config["top_k"],
    )
    if dataset == "browsecomp_plus" and worker_config["retriever_type"].startswith("qwen3_emb"):
        cfg.retrieval_query_max_length = 8196
    retriever_type = worker_config["retriever_type"]
    if retriever_type == "bm25":
        retriever = BM25Retriever(cfg)
    elif retriever_type in ("spladepp", "spladev3"):
        retriever = SPLADERetriever(cfg)
    elif retriever_type in ["rerank_l6", "rerank_l12"]:
        retriever = RerankRetriever(cfg)
    else:
        retriever = DenseRetriever(cfg)

    return llm_client, retriever


def _silence_hf_progress_bars() -> None:
    """Turn off HuggingFace/datasets tqdm bars and downgrade their loggers.

    Called from the quiet path of :func:`_init_worker` (and from the main
    process) so model/corpus loading stays silent in non-interactive runs.
    Each import is guarded: the helper must never be the reason a worker dies.
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for mod, attr in (
        ("transformers.utils.logging", "disable_progress_bar"),
        ("datasets.utils.logging", "disable_progress_bar"),
        ("huggingface_hub.utils", "disable_progress_bars"),
    ):
        try:
            __import__(mod)
            getattr(sys.modules[mod], attr)()
        except Exception:
            pass
    for name in ("transformers", "datasets", "sentence_transformers", "faiss"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _init_worker(worker_id: int, worker_config: dict):
    """Initialise a GPU worker: pin GPU, load models, build agent.

    Returns:
        (agent, search_tool, verbose)
    """
    gpu_ids = worker_config.get("gpu_ids", [])
    verbose  = worker_config.get("verbose", False)
    if worker_config.get("quiet", False):
        logging.getLogger("agents").setLevel(logging.ERROR)
        logging.getLogger("agent_tools").setLevel(logging.ERROR)
        logging.getLogger("utils").setLevel(logging.ERROR)
        logging.getLogger("prompts").setLevel(logging.ERROR)
        # The retriever's from_pretrained() draws a 398-step "Loading weights"
        # tqdm bar per worker.  Under sbatch stdout is not a TTY, so every
        # redraw is appended verbatim and four workers alone add ~1 MB of
        # carriage-return noise to the .out file.
        _silence_hf_progress_bars()
    if gpu_ids and worker_id < len(gpu_ids):
        physical_gpu = gpu_ids[worker_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        print(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES={physical_gpu}", flush=True)
        import torch
        if torch.cuda.is_available():
            n_visible = torch.cuda.device_count()
            if n_visible == 1:
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(physical_gpu)
            print(f"[Worker {worker_id}] torch.cuda.current_device()={torch.cuda.current_device()}, "
                  f"device_count={n_visible}", flush=True)
    else:
        print(f"[Worker {worker_id}] No specific GPU assigned (gpu_ids={gpu_ids})", flush=True)

    import shutil
    _java_bin = shutil.which("java")
    if _java_bin:
        _java_real = os.path.realpath(_java_bin)
        _java_home = os.path.dirname(os.path.dirname(_java_real))
        _jvm_so = os.path.join(_java_home, "lib", "server", "libjvm.so")
        if os.path.isfile(_jvm_so):
            os.environ["JAVA_HOME"] = _java_home
            os.environ["JVM_PATH"] = _jvm_so

    # By this point the JVM is normally already up: "spawn" re-imports __main__
    # in the worker, which reaches searcher_component.retriever and boots the VM
    # at import time -- applying these same options there.  Calling add_options()
    # on a running VM raises ValueError (not ImportError), which used to kill
    # every worker before it processed a single query, so guard on vm_running.
    try:
        import jnius_config
        if not jnius_config.vm_running:
            jnius_config.add_options(
                '-Xmx2g',
                '-Xms512m',
                '-XX:ParallelGCThreads=4',
                '-XX:ConcGCThreads=2',
            )
    except ImportError:
        pass

    print(f"[Worker {worker_id}] Building components (LLM, retriever)...", flush=True)
    llm_client, retriever = _build_components_from_config(worker_config)
    print(f"[Worker {worker_id}] Components built", flush=True)

    agentic_model = worker_config["agentic_model"]
    dataset = worker_config["dataset"]

    from searcher_component import RetrievalSearchTool
    from searcher_component.rerankers import build_reranker_from_config

    _reranker_configs = get_reranker_configs(worker_config.get("rerank_top_k", 100))

    _post_ret_type = worker_config.get("post_retrieval_reranker_type", "null")
    _post_fus_type = worker_config.get("post_fusion_reranker_type", "null")
    _post_ret_reranker = build_reranker_from_config(_post_ret_type, _reranker_configs) if _post_ret_type != "null" else None
    _post_fus_reranker = build_reranker_from_config(_post_fus_type, _reranker_configs) if _post_fus_type != "null" else None

    search_tool = None
    if retriever is not None:
        search_tool = RetrievalSearchTool(
            retriever=retriever,
            post_retrieval_reranker=_post_ret_reranker,
            post_fusion_reranker=_post_fus_reranker,
            top_k=worker_config.get("top_k", 100),
            rerank_top_k=worker_config.get("rerank_top_k", 100),
            retrieval_input=worker_config.get("retrieval_input", "subquery"),
            post_fusion_reranker_input=worker_config.get("post_fusion_reranker_input", "original_query"),
            ensure_novel_seen_docs=worker_config.get("ensure_novel_seen_docs", False),
            seen_top_k=worker_config.get("seen_top_k", 5),
        )

    print(f"[Worker {worker_id}] Creating agent ({agentic_model})...", flush=True)
    llm_model = worker_config["llm_model"]
    agent = build_agent(
        agentic_model=agentic_model,
        agentic_model_cli=worker_config.get("agentic_model_cli", agentic_model),
        dataset=worker_config["dataset"],
        llm_model=llm_model,
        llm_client=llm_client,
        retriever=retriever,
        max_iteration=worker_config.get("max_iteration", 100),
        seen_top_k=worker_config.get("seen_top_k", 5),
        verbose=verbose,
        search_tool=search_tool,
        use_plan=worker_config.get("use_plan", False),
        max_output_tokens_total=worker_config.get("max_output_tokens_total", 40000),
        temperature=worker_config.get("temperature", 0.7),
        max_extend_steps=worker_config.get("max_extend_steps", 5),
        max_retries=worker_config.get("max_retries", 3),
        hard_mode=worker_config.get("hard_mode", True),
        max_passage_chars=worker_config.get("max_passage_chars", 4000),
        ua_max_turns=worker_config.get("ua_max_turns", 8),
        ua_max_passage_chars=worker_config.get("ua_max_passage_chars", 1500),
        ua_max_format_retries=worker_config.get("ua_max_format_retries", 2),
        ua_max_tokens_per_call=worker_config.get("ua_max_tokens_per_call", 4096),
        ua_disable_native_thinking=worker_config.get("ua_disable_native_thinking", True),
    )

    estimator = build_uncertainty_estimator(
        mode=worker_config.get("uncertainty_estimator_mode", "off"),
        retriever=retriever,
        qrels=worker_config.get("qrels"),
        llm_criteria=worker_config.get("llm_criteria"),
        max_criteria=worker_config.get("max_criteria", 8),
        criteria_judge=worker_config.get("criteria_judge", "none"),
        criteria_judge_model=worker_config.get("criteria_judge_model", ""),
        agent=agent if hasattr(agent, "uncertainty_estimator") else None,
        agentic_model=agentic_model,
        dataset=dataset,
        llm_model=llm_model,
    )
    if estimator is not None and hasattr(agent, "uncertainty_estimator"):
        agent.uncertainty_estimator = estimator

    print(f"[Worker {worker_id}] Initialisation complete", flush=True)
    return agent, search_tool, verbose


def gpu_worker(worker_id: int, query_items: list, temp_dir_str: str, worker_config: dict, progress_queue=None, init_lock=None) -> dict:
    """Worker process: pin to one GPU, rebuild the agent, process queries, save files."""
    _lock_ctx = init_lock if init_lock is not None else contextlib.nullcontext()

    try:
        with _lock_ctx:
            agent, search_tool, verbose = _init_worker(
                worker_id, worker_config,
            )
    except Exception as exc:
        print(f"[Worker {worker_id}] INIT FAILED: {exc}", flush=True)
        traceback.print_exc()
        if progress_queue is not None:
            progress_queue.put(None)
        return {}

    temp_dir       = Path(temp_dir_str)
    retrieval_dir  = str(temp_dir / "retrieval" / "surfaced")
    generation_dir = str(temp_dir / "generation")
    trajectory_dir = str(temp_dir / "trajectory")
    cited_doc_dir  = str(temp_dir / "retrieval" / "cited")
    seen_doc_dir   = str(temp_dir / "retrieval" / "seen")
    uncertainty_dir = str(temp_dir / "uncertainty")
    for _d in [retrieval_dir, generation_dir, trajectory_dir, cited_doc_dir, seen_doc_dir, uncertainty_dir]:
        Path(_d).mkdir(parents=True, exist_ok=True)

    from evaluation import SurfacedDocEvaluator, GenerationEvaluator, TrajectoryEvaluator, UncertaintyEvaluator, CitedDocEvaluator, SeenDocEvaluator
    _ret_eval        = SurfacedDocEvaluator(qrels={}, k_values=[])
    _gen_eval        = GenerationEvaluator()
    _traj_eval       = TrajectoryEvaluator()
    _uncertainty_eval = UncertaintyEvaluator()
    _cited_eval      = CitedDocEvaluator(qrels={}, k_values=[])
    _seen_eval       = SeenDocEvaluator(qrels={}, k_values=[])

    results     = {}
    temperature = worker_config.get("temperature", 0.7)
    total       = len(query_items)
    max_iteration = worker_config.get("max_iteration", "?")

    _cb_state = [None, 0]

    def _status_cb(stage: str, iteration: int):
        if progress_queue is not None:
            progress_queue.put(
                (worker_id, _cb_state[0], _cb_state[1], total, "update",
                 (stage, iteration, max_iteration))
            )

    for idx, (query_id, query_text) in enumerate(query_items, 1):
        _cb_state[0] = query_id
        _cb_state[1] = idx
        if search_tool is not None:
            search_tool.reset()
        if progress_queue is not None:
            progress_queue.put((worker_id, query_id, idx, total, "processing", None))
        if verbose:
            print(
                f"  [Worker {worker_id}] [{idx}/{len(query_items)}] Processing query: {query_id}\n    Query text: {query_text}",
                flush=True,
            )
        # Streams each step to trajectory/{qid}.jsonl + .md as it happens, so a
        # worker that dies mid-query still leaves that query's work on disk.
        traj_logger = TrajectoryLogger(
            trajectory_dir, query_id, query_text,
            agent_name=worker_config.get("agentic_model", ""),
            model=worker_config.get("llm_model", ""),
        )
        result = agent.run_single(
            query_id=query_id,
            query_text=query_text,
            temperature=temperature,
            status_callback=_status_cb if progress_queue is not None else None,
            trajectory_logger=traj_logger,
        )
        if result is None:
            if verbose:
                print(f"  [Worker {worker_id}] ✗ Skipping {query_id}", flush=True)
            if progress_queue is not None:
                progress_queue.put((worker_id, query_id, idx, total, "skipped", None))
            continue
        result["cited_docs_ranked_list"] = _build_cited_docs_ranked_list(result)
        references = build_references_section(result)
        if references:
            result["generation"] = result["generation"].rstrip() + references
        results[query_id] = result
        _ret_eval.save_item(query_id, result, retrieval_dir)
        _gen_eval.save_item(query_id, result, generation_dir)
        _traj_eval.save_item(query_id, query_text, result, trajectory_dir)
        _cited_eval.save_item(query_id, result, cited_doc_dir)
        _seen_eval.save_item(query_id, result, seen_doc_dir)
        _uncertainty_eval.save_item(query_id, query_text, result, uncertainty_dir)
        if progress_queue is not None:
            num_iters = result.get("num_iterations", "?")
            progress_queue.put((worker_id, query_id, idx, total, "done", f"{num_iters}/{max_iteration}"))
        if verbose:
            print(f"  [Worker {worker_id}] ✓ Saved: {query_id}", flush=True)

    agent.cleanup()
    # As run_pipeline does: drop the estimator's model and callback references.
    estimator = getattr(agent, "uncertainty_estimator", None)
    if estimator is not None:
        estimator.close()
    if verbose:
        print(
            f"[Worker {worker_id}] Completed {len(results)}/{len(query_items)} queries",
            flush=True,
        )
    if progress_queue is not None:
        progress_queue.put(None)
    return results
