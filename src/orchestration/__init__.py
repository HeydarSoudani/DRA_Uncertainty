"""Pipeline orchestration: retriever/belief-model factories and multi-GPU workers.

This is the wiring layer that assembles heavy component packages
(``belief_model_component``, ``searcher_component``, ``deep_research_agents``,
``evaluation``) into a runnable agent. It is consumed by the ``run_pipeline``
entry point in ``experiments/inference/dra_inference.py``.

Leaf-level helpers (LLM-client factory, CLI arg resolution, output naming) live
in ``utils`` so this module is the only place that depends on the heavy
component packages — keeping ``utils`` a true leaf layer.

The worker functions (_init_worker, gpu_worker) must stay at module level to
be picklable by multiprocessing.spawn.
"""

import contextlib
import io
import logging
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from indexing_corpus_dataset.layout import DATASET_SPECS, query_max_length
from utils.config import (
    OPENROUTER_BASE_URL,
    _RetrieverConfig,
    effective_max_output_tokens,
    get_reranker_configs,
    resolve_agent_backend,
)
from reasoner_component import create_generator, disable_native_thinking
from reasoner_component.setup import setup_llm
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
    config.retrieval_query_max_length = query_max_length(args.dataset, args.retriever)
    if args.retriever == "bm25":
        return BM25Retriever(config)
    elif args.retriever in ("spladepp", "spladev3"):
        return SPLADERetriever(config)
    elif args.retriever in ["rerank_l6", "rerank_l12"]:
        return RerankRetriever(config)
    else:
        return DenseRetriever(config)


# ===========================================================================
# Belief model factory
# ===========================================================================

# Agents that write a long report rather than a short answer.
NO_INTERMEDIATE_ANSWER_AGENTS = ("cpm_report",)


def _belief_llm(model: str):
    """LLM client of a belief model call, native reasoning off.  A ``vllm/``
    model (e.g. a trained belief model) is served locally at
    ``$BELIEF_VLLM_API_BASE`` (default ``$VLLM_API_BASE``, so it can sit
    next to a locally served agent); any other model goes through the API."""
    if model.startswith("vllm/"):
        return disable_native_thinking(create_generator(model, api_base=os.getenv("BELIEF_VLLM_API_BASE")))
    return disable_native_thinking(create_generator(model, backend="api"))


def build_belief_model(
    mode: str,
    *,
    max_criteria: int,
    add_intermediate_answer: bool,
    retriever=None,
    qrels=None,
    llm_criteria: Optional[str] = None,
    belief_judge_model: str = "",
    query_scorer_model: str = "",
    agent=None,
    policy_model: Optional[str] = None,
    dataset: Optional[str] = None,
    criteria_bank: Optional[str] = None,
):
    """Build the belief model, or None when *mode* is ``"off"``.

    Shared by the main process (run_pipeline) and spawned GPU workers.

    Args:
        mode: ``"off"``, ``"monitor"`` (observe only) or ``"inform"``
            (observe and inject a ``<belief_state>`` tag into the trajectory).
        retriever: Retriever whose encoder gives the novelty embeddings
            (dense retrievers only; others leave nu^q null).
        qrels: Thresholded relevance judgements, for the new-item precision
            and the recall so far.
        llm_criteria: Model that extracts each query's criteria.  Without
            it the criteria-based signals are null.
        max_criteria: Cap on the number of criteria per query (for a report
            request, the number the extractor may add, asked in the prompt);
            resolved from the dataset by ``resolve_dataset_defaults``.
        belief_judge_model: LLM behind the belief state and criteria_delta
            (belief updater); "" = *llm_criteria*.  A ``vllm/`` model is
            served locally (``_belief_llm``), e.g. a trained belief model.
        query_scorer_model: LLM behind the criteria attempts (targeting
            scorer); "" = the belief updater's model.
        add_intermediate_answer: Ask the agent for an intermediate answer
            after every search iteration (one extra LLM call per step).
        agent: Agent instance; its ``answer_from_trajectory`` gives the
            per-step intermediate answers.
        policy_model: Agent type name (answer format, intermediate answer
            gating).
        dataset: Dataset name; intermediate answers only for ``task == "qa"``.
        criteria_bank: Criteria bank file (``layout.criteria_bank_path``):
            a query's criteria are read from it, and extracted and added
            only when missing.  None: extracted for every sample.
    """
    if mode == "off":
        return None
    if mode not in ("monitor", "inform"):
        raise ValueError(f"unknown belief model mode {mode!r}; expected 'off', 'monitor' or 'inform'")

    from belief_model_component import (
        BankedCriteriaSource, LLMCriteriaSource, LLMQueryScorer, BeliefModel, build_belief_judges,
        encode_fn_from_retriever,
    )

    query_shape = DATASET_SPECS[dataset].query_shape if dataset else "single_target"
    criteria_source = None
    if llm_criteria:
        criteria_source = LLMCriteriaSource(
            llm_client=_belief_llm(llm_criteria),
            max_criteria=max_criteria,
            model_name=llm_criteria,
            query_shape=query_shape,
        )
        if criteria_bank:
            criteria_source = BankedCriteriaSource(criteria_source, criteria_bank)
            print(f"Belief model: criteria from the bank {criteria_bank}, missing ones from {llm_criteria}")
        else:
            print(f"Belief model: criteria from {llm_criteria}")
    else:
        logger.warning("No --llm-criteria model; criteria-based signals disabled")

    encode_fn, _ = encode_fn_from_retriever(retriever) if retriever is not None else (None, None)
    if encode_fn is None:
        logger.warning("Retriever has no local encoder; query_novelty is null")

    intermediate_answer_fn = None
    if not add_intermediate_answer:
        print("Belief model: intermediate answers disabled (--add-intermediate-answer false)")
    elif (
        agent is not None
        and policy_model not in NO_INTERMEDIATE_ANSWER_AGENTS
        # Only short-answer tasks: a report has no answer candidates.
        and dataset is not None and DATASET_SPECS[dataset].task == "qa"
        and hasattr(agent, "answer_from_trajectory")
    ):
        intermediate_answer_fn = agent.answer_from_trajectory
        print(f"Belief model: intermediate answers via {policy_model}.answer_from_trajectory")

    belief_updater, query_scorer = None, None
    if criteria_source is not None:
        judge_model = belief_judge_model or llm_criteria
        belief_updater, query_scorer = build_belief_judges(
            _belief_llm(judge_model), model=judge_model,
            query_shape=query_shape,
        )
        if query_scorer_model and query_scorer_model != judge_model:
            query_scorer = LLMQueryScorer(_belief_llm(query_scorer_model), model_name=query_scorer_model)
        print(f"Belief model: belief updater {belief_updater.name}, query scorer {query_scorer.name}")

    return BeliefModel(
        criteria_source=criteria_source,
        belief_updater=belief_updater,
        query_scorer=query_scorer,
        encode_fn=encode_fn,
        qrels=qrels or {},
        intermediate_answer_fn=intermediate_answer_fn,
        policy_model=policy_model or "",
        inform=mode == "inform",
    )


# ===========================================================================
# Agent factory
# ===========================================================================

def build_agent(
    policy_model: str,
    llm_model: str,
    llm_client=None,
    retriever=None,
    *,
    policy_model_cli: Optional[str] = None,
    dataset: Optional[str] = None,
    max_iteration: int = 100,
    seen_top_k: int = 5,
    verbose: bool = False,
    search_tool=None,
    use_plan: bool = False,
    max_output_tokens_total: int,
    temperature: float,
    max_extend_steps: int = 5,
    max_retries: int = 3,
    hard_mode: bool = True,
    max_passage_chars: int = 4000,
):
    """Instantiate an agent and attach its search tool.

    Shared by both the main process (run_pipeline, single-GPU) and spawned GPU
    workers (_init_worker) so the per-agent construction logic lives in one
    place.

    The agent-specific keyword arguments are assembled into ``_reasoning_extra``
    based on ``policy_model`` and forwarded to the agent constructor.
    """
    from deep_research_agents.agents import AGENT_MAP

    model_class = AGENT_MAP[policy_model]
    # Backend (API vs vLLM) is resolved on the user-facing CLI name because the
    # OpenRouter registry distinguishes oss_20b / oss_120b, which both collapse
    # to the internal agent name "oss".
    _cli_name = policy_model_cli or policy_model
    # The capped value (utils.config.OUTPUT_TOKEN_CAPS); run_config.json records the same.
    max_output_tokens = effective_max_output_tokens(policy_model, max_output_tokens_total)
    _reasoning_extra: dict = {}
    if use_plan and policy_model == "react":
        _reasoning_extra["use_plan"] = True
    if policy_model == "glm":
        _reasoning_extra["max_output_tokens"] = max_output_tokens
        # Prefer OpenRouter when the agent is in the registry (0 local GPU);
        # otherwise fall through to the agent's vLLM defaults (localhost:6008).
        _backend, _slug = resolve_agent_backend(_cli_name)
        if _backend == "api":
            _reasoning_extra["model_url"] = OPENROUTER_BASE_URL
            _reasoning_extra["model_name"] = _slug
            _reasoning_extra["api_key"] = os.getenv("OPENROUTER_API_KEY")
    elif policy_model == "oss":
        # Cap per-call output like GLM so the prompt/history still fits inside
        # the 131072-token vLLM window (gpt-oss native max = OpenRouter's max).
        # On the OpenRouter Responses path truncation:"auto" reclaims the full
        # window regardless of this ceiling.
        _reasoning_extra["max_output_tokens"] = max_output_tokens
        # Prefer OpenRouter (Responses API) when oss_20b / oss_120b is in the
        # registry; otherwise use the agent's local vLLM defaults (localhost:6008).
        _backend, _slug = resolve_agent_backend(_cli_name)
        if _backend == "api":
            _reasoning_extra["model_url"] = OPENROUTER_BASE_URL
            _reasoning_extra["model_name"] = _slug
            _reasoning_extra["api_key"] = os.getenv("OPENROUTER_API_KEY")
        else:
            _reasoning_extra["model_name"] = f"openai/{llm_model}"
    elif policy_model == "tongyi":
        _reasoning_extra["max_tokens_per_step"] = max_output_tokens
    elif policy_model == "cpm_explore":
        _reasoning_extra["max_output_tokens"] = max_output_tokens
        # The run's resolved temperature (utils.config.resolve_temperature),
        # not a falsy-guarded default: 0.0 is a temperature a user can ask for.
        _reasoning_extra["temperature"] = temperature
    elif policy_model == "cpm_report":
        _reasoning_extra["max_extend_steps"] = max_extend_steps
        _reasoning_extra["max_retries"] = max_retries
        _reasoning_extra["hard_mode"] = hard_mode
        _reasoning_extra["max_passage_chars"] = max_passage_chars
        _reasoning_extra["model_name"] = llm_model
    elif policy_model == "belief_aware":
        _reasoning_extra["max_passage_chars"] = max_passage_chars
        _reasoning_extra["max_retries"] = max_retries
        # Report datasets get the report prompts (task from the dataset).
        if dataset is not None:
            spec = DATASET_SPECS[dataset]
            _reasoning_extra["task"] = spec.task
            _reasoning_extra["report_chars"] = spec.report_chars

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
    """Rebuild LLM client and retriever from a serializable config dict, with
    the main process's ``setup_llm`` and :func:`setup_retriever_from_args`
    (the worker config carries their argument names).

    Returns:
        (llm_client, retriever)  — llm_client may be None for self-managed agents.
    """
    args = SimpleNamespace(**worker_config)
    return setup_llm(args, num_gpus=1), setup_retriever_from_args(args)


def _silence_hf_progress_bars() -> None:
    """Turn off HuggingFace/datasets tqdm bars and downgrade their loggers.

    Called from :func:`quiet_logging` so model/corpus loading stays silent in
    non-interactive runs.
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


def quiet_logging() -> None:
    """The ``--quiet`` setup of a process (the main one and each GPU worker):
    agent and utils loggers at ERROR, no HF/datasets progress bars."""
    for name in ("agents", "agent_tools", "utils", "prompts"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # The retriever's from_pretrained() draws a 398-step "Loading weights"
    # tqdm bar per process.  Under sbatch stdout is not a TTY, so every
    # redraw is appended verbatim to the .out file.
    _silence_hf_progress_bars()


def _init_worker(worker_id: int, worker_config: dict):
    """Initialise a GPU worker: pin GPU, load models, build agent.

    Returns:
        (agent, search_tool, verbose)
    """
    gpu_ids = worker_config["gpu_ids"]
    verbose  = worker_config["verbose"]
    if worker_config["quiet"]:
        quiet_logging()
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

    policy_model = worker_config["policy_model"]
    dataset = worker_config["dataset"]

    from searcher_component import RetrievalSearchTool
    from searcher_component.rerankers import build_reranker_from_config

    _reranker_configs = get_reranker_configs(worker_config["rerank_top_k"])

    _post_ret_type = worker_config["post_retrieval_reranker_type"]
    _post_fus_type = worker_config["post_fusion_reranker_type"]
    _post_ret_reranker = build_reranker_from_config(_post_ret_type, _reranker_configs) if _post_ret_type != "null" else None
    _post_fus_reranker = build_reranker_from_config(_post_fus_type, _reranker_configs) if _post_fus_type != "null" else None

    search_tool = None
    if retriever is not None:
        search_tool = RetrievalSearchTool(
            retriever=retriever,
            post_retrieval_reranker=_post_ret_reranker,
            post_fusion_reranker=_post_fus_reranker,
            rerank_top_k=worker_config["rerank_top_k"],
            retrieval_input=worker_config["retrieval_input"],
            post_fusion_reranker_input=worker_config["post_fusion_reranker_input"],
        )

    print(f"[Worker {worker_id}] Creating agent ({policy_model})...", flush=True)
    llm_model = worker_config["llm_model"]
    agent = build_agent(
        policy_model=policy_model,
        policy_model_cli=worker_config["policy_model_cli"],
        dataset=dataset,
        llm_model=llm_model,
        llm_client=llm_client,
        retriever=retriever,
        max_iteration=worker_config["max_iteration"],
        seen_top_k=worker_config["seen_top_k"],
        verbose=verbose,
        search_tool=search_tool,
        use_plan=worker_config["use_plan"],
        max_output_tokens_total=worker_config["max_output_tokens_total"],
        temperature=worker_config["llm_temperature"],
        max_extend_steps=worker_config["max_extend_steps"],
        max_retries=worker_config["max_retries"],
        hard_mode=worker_config["hard_mode"],
        max_passage_chars=worker_config["max_passage_chars"],
    )

    belief_model = build_belief_model(
        mode=worker_config["belief_mode"],
        retriever=retriever,
        qrels=worker_config.get("qrels"),
        llm_criteria=worker_config["llm_criteria"],
        max_criteria=worker_config["max_criteria"],
        belief_judge_model=worker_config["belief_judge_model"],
        query_scorer_model=worker_config["query_scorer_model"],
        add_intermediate_answer=worker_config["add_intermediate_answer"],
        agent=agent if hasattr(agent, "belief_model") else None,
        policy_model=policy_model,
        dataset=dataset,
        criteria_bank=worker_config.get("criteria_bank"),
    )
    if belief_model is not None and hasattr(agent, "belief_model"):
        agent.belief_model = belief_model

    print(f"[Worker {worker_id}] Initialisation complete", flush=True)
    return agent, search_tool, verbose


def run_query(agent, run_dir, query_id: str, query_text: str, *, policy_model: str, llm_model: str,
              answer: Optional[str] = None, temperature: float = 0.7, status_callback=None) -> Optional[dict]:
    """Run one query and save its files under *run_dir* (None: nothing is
    saved); the one per-query path of the sequential run and of the GPU
    workers.  Returns the result, or None when the query failed: the error
    is logged and the caller goes on with its next query.

    The trajectory is streamed to ``trajectory/{qid}.jsonl`` and ``.md`` as
    it happens, so an interrupted query still leaves its work on disk; the
    result gets its cited-doc list and a References section, then
    ``evaluation.runner.save_query_outputs`` writes every file.  A query
    that fails before the trajectory meta line is written stays unfinished
    and is run again on resume.
    """
    from evaluation.runner import save_query_outputs

    try:
        traj_logger = TrajectoryLogger(
            Path(run_dir) / "trajectory", query_id, query_text,
            agent_name=policy_model, model=llm_model, answer=answer,
        ) if run_dir else None
        result = agent.run_single(
            query_id=query_id,
            query_text=query_text,
            temperature=temperature,
            status_callback=status_callback,
            trajectory_logger=traj_logger,
        )
        if result is None:
            return None
        result["cited_docs_ranked_list"] = _build_cited_docs_ranked_list(result)
        # Append a References section to the generation (CPMReport already has one)
        references = build_references_section(result)
        if references:
            result["generation"] = result["generation"].rstrip() + references
        if run_dir:
            save_query_outputs(run_dir, query_id, query_text, result)
        return result
    except Exception as exc:
        logger.error("Query %s failed: %s", query_id, exc)
        traceback.print_exc()
        return None


def gpu_worker(worker_id: int, query_items: list, run_dir: str, worker_config: dict, progress_queue=None) -> List[str]:
    """Worker process: pin to one GPU, rebuild the agent, process queries and
    save their files straight to *run_dir*.

    Returns the ids of the queries it finished; the parent reads the results
    back from their files.
    """
    # With live progress bars in the parent, this process's init prints would
    # land between the bars' in-place redraws and scramble them.  Hold them
    # back; on failure they go to stderr, otherwise worker 0 forwards the
    # belief model config lines (the same in every worker) to be printed once.
    live_bars = bool(worker_config.get("live_bars", False))
    init_out = io.StringIO()
    try:
        with (contextlib.redirect_stdout(init_out) if live_bars else contextlib.nullcontext()):
            agent, search_tool, verbose = _init_worker(
                worker_id, worker_config,
            )
    except Exception as exc:
        if live_bars:
            sys.stderr.write(init_out.getvalue())
        print(f"[Worker {worker_id}] INIT FAILED: {exc}", file=sys.stderr if live_bars else sys.stdout, flush=True)
        traceback.print_exc()
        if progress_queue is not None:
            progress_queue.put(None)
        return []
    if live_bars and worker_id == 0 and progress_queue is not None:
        for line in init_out.getvalue().splitlines():
            if line.startswith("Belief model:"):
                progress_queue.put(("log", line))

    from evaluation.runner import QUERY_OUTPUT_DIRS

    run_path = Path(run_dir)
    for _d in QUERY_OUTPUT_DIRS:
        (run_path / _d).mkdir(parents=True, exist_ok=True)

    finished: List[str] = []
    temperature = worker_config["llm_temperature"]
    total       = len(query_items)
    max_iteration = worker_config["max_iteration"]

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
        if progress_queue is not None:
            progress_queue.put((worker_id, query_id, idx, total, "processing", None))
        if verbose:
            print(
                f"  [Worker {worker_id}] [{idx}/{len(query_items)}] Processing query: {query_id}\n    Query text: {query_text}",
                flush=True,
            )
        result = run_query(
            agent, run_path, query_id, query_text,
            policy_model=worker_config["policy_model"], llm_model=worker_config["llm_model"],
            answer=(worker_config.get("answers") or {}).get(query_id), temperature=temperature,
            status_callback=_status_cb if progress_queue is not None else None,
        )
        if result is None:
            if verbose:
                print(f"  [Worker {worker_id}] ✗ Skipping {query_id}", flush=True)
            if progress_queue is not None:
                progress_queue.put((worker_id, query_id, idx, total, "skipped", None))
            continue
        finished.append(query_id)
        if progress_queue is not None:
            num_iters = result.get("num_iterations", "?")
            progress_queue.put((worker_id, query_id, idx, total, "done", f"{num_iters}/{max_iteration}"))
        if verbose:
            print(f"  [Worker {worker_id}] ✓ Saved: {query_id}", flush=True)

    agent.cleanup()
    # As run_pipeline does: drop the belief model's model and callback references.
    belief_model = getattr(agent, "belief_model", None)
    if belief_model is not None:
        belief_model.close()
    if verbose:
        print(
            f"[Worker {worker_id}] Completed {len(finished)}/{len(query_items)} queries",
            flush=True,
        )
    if progress_queue is not None:
        progress_queue.put(None)
    return finished
