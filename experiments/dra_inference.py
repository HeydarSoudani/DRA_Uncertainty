"""Run deep research agents pipeline.

Supported agents via --agentic-model (the LLM is selected automatically per agent):
    cpm_report      Writing-as-Reasoning agent (CPMReport)         → openbmb/AgentCPM-Report (vLLM)
    searchr1        SearchR1 reasoning agent                       → PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3 (vLLM)
    research        ReSearch reasoning agent                       → agentrl/ReSearch-Qwen-7B-Instruct (vLLM)
    stepsearch      StepSearch reasoning agent                     → Zill1/StepSearch-7B-Instruct (vLLM)
    react           ReAct reasoning agent (--use-plan optional)    → claude-sonnet-4-6 (API)
    selfask         SelfAsk reasoning agent                        → claude-sonnet-4-6 (API)
    searcho1        SearchO1 reasoning agent                       → claude-sonnet-4-6 (API)
    webweaver       WebWeaver outline agent                        → Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (vLLM)
    drtulu          DR-Tulu reasoning agent                        → rl-research/DR-Tulu-8B (vLLM)
    glm             GLM reasoning agent                            → zai-org/GLM-4.7-Flash (vLLM)
    oss_20b         GPT-OSS-20B reasoning agent                    → gpt-oss-20b (vLLM)
    oss_120b        GPT-OSS-120B reasoning agent                   → gpt-oss-120b (vLLM)
    tongyi          Tongyi-DeepResearch ReAct agent                → Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (vLLM)
    cpm_explore     AgentCPM-Explore deep search agent             → openbmb/AgentCPM-Explore (vLLM)
    uncertainty_aware  Uncertainty-aware search agent (reads <certainty> in inform mode) → qwen/qwen3.6-27b (OpenRouter API)

Agentic workflows:
    ReAct-style (react, selfask, searcho1, research, searchr1, stepsearch, drtulu, glm, oss_20b, oss_120b, tongyi, cpm_explore):
        Query → [Think → Search → Observe]* → Report → Evaluate
        Instruction-tuned : react, selfask, searcho1
        RL-trained        : research, searchr1, stepsearch, drtulu, tongyi, cpm_explore, glm, oss_20b, oss_120b

    Outline-style (webweaver):
        Query → [Think → Search → Write_outline]* → Outline → [Think → Retrieve → Write_section]* → Report → Evaluate

    Report-style (cpm_report):
        Query → Search → Init Plan → [Search → Write]* → [Extend Plan → [Search → Write]*]* → Report → Evaluate


Output structure:
    run_outputs/{dataset}_{split}_{query_key}_{retriever}/{agent}_{backend}_{model}/{uncertainty_config}/
    e.g. run_outputs/neuclir_2024_news_e5/oss_vllm_gpt-oss-20b/ue-monitor/
    ├── run_config.json              full agent/searcher/uncertainty-estimator settings
    ├── retrieval/
    │   ├── surfaced/
    │   │   └── {query_id}.trec      per-query surfaced-doc TREC (all iters, col 6 = iter_N; raw retriever output)
    │   ├── seen/
    │   │   └── {query_id}.trec      per-query seen-doc TREC file (docs shown to the LLM)
    │   ├── cited/
    │   │   └── {query_id}.trec      per-query cited-doc TREC file (docs cited by the LLM)
    │   └── fusion_{method}.trec     deduped fusion ranking, single aggregate over all queries
    ├── generation/
    │   └── {query_id}.md            per-query generation output
    ├── trajectory/
    │   ├── {query_id}.jsonl         per-query trajectory: one line per step + a trailing meta line
    │   └── {query_id}.md            same trajectory, human-readable, written live (one block per step)
    ├── uncertainty/
    │   └── {query_id}.jsonl         per-query uncertainty signals: meta line + one line per iteration
    ├── accuracy.jsonl               per-query answer correctness (datasets with answers)
    ├── report_eval/                 Auto-ARGUE inputs, cached judgments and per-query scores.tsv
    └── summary.json                 grouped run metrics:
                                       num_queries,
                                       retrieval  {seen, cited, fusion},
                                       generation {correctness | nuggets, stats},
                                       criteria, trajectory
"""

import argparse
import warnings
import logging
import time
import os
import concurrent.futures
import multiprocessing
import threading
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# Suppress async cleanup noise
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("asyncio.sslproto").setLevel(logging.CRITICAL)

from indexing_corpus_dataset.dataset_loaders import graded_qrels as to_graded_qrels, load_qrels, load_split, resolve_split_id
from indexing_corpus_dataset.layout import DATASETS, DATASET_SPECS, OUTPUT_ROOT, criteria_bank_path

from deep_research_agents.agents import ALL_AGENTS
from utils.config import AGENTIC_MODEL_TO_LLM, AGENTIC_MODEL_ALIAS, resolve_temperature
from orchestration import (
    gpu_worker,
    setup_retriever_from_args,
    _silence_hf_progress_bars,
)
from utils.llm_client import setup_llm
from utils.cli_setup import (
    resolve_dataset_defaults,
    detect_num_gpus,
    assemble_pipeline_kwargs,
    cleanup_event_loop,
    load_run_config,
    parse_cli_overrides,
    apply_config_to_args,
    _sm_bool,
)
from utils.text_utils import build_references_section, _build_cited_docs_ranked_list
from utils.trajectory_logger import TrajectoryLogger
from utils.io_utils import (
    get_processed_queries,
    setup_output_dirs,
    build_dataset_dir_name,
    build_run_name_for_pipeline,
    build_uncertainty_config_name,
    write_run_config,
)
from evaluation.gold import load_gold_units
from evaluation.runner import QUERY_OUTPUT_DIRS, build_evaluators, evaluate_and_save, load_run_results, save_query_outputs
from evaluation.retrieval.fusion import run_fusion_eval

_OUTPUT_PREFIX = str(OUTPUT_ROOT)
_CONFIG_DEFAULT = str(Path(__file__).resolve().parent / "configs" / "dra_inference.yaml")


# ============================================================================
# Pipeline
# ============================================================================
def run_pipeline(data_path: str, subset: Optional[str] = None, dataset_year: Optional[str] = None, query_key: Optional[str] = None, output_path: Optional[str] = None, agentic_model: str = "cpm_report", limit: Optional[int] = None, verbose: bool = True, num_gpus: int = 1, worker_config: Optional[dict] = None, **kwargs):
    """Execute the deep research agents pipeline on a dataset.

    Args:
        data_path:     Path to dataset directory.
        subset:        Dataset subset identifier, e.g. "news" or "technical"
                       for neuclir, "wiki2" for trqa.
        dataset_year:  Dataset year (neuclir "2024", ragtime "2025"); trqa
                       carries its eval split (test|validation) here.
        query_key:     Key in each JSONL query record to use as the query text.
                       None = the dataset's default (layout.DATASET_SPECS),
                       e.g. "request" for neuclir.  Records without the key
                       are not run.
        output_path:   Root path for saving results (optional).
        agentic_model: Agent to use; one of ALL_AGENTS.
        limit:         Cap the number of queries (for quick tests).
        verbose:       Whether to print detailed logs.
        num_gpus:      Number of GPU workers for query-level parallelism.
                       1 = sequential (default). >1 = spawn one process per GPU,
                       split queries across workers, merge temp files at the end.
        worker_config: Serialisable config dict passed to each GPU worker so it
                       can rebuild the agent stack.  Required when num_gpus > 1.
        **kwargs:      Agent-specific and evaluation parameters.
    """
    if agentic_model not in ALL_AGENTS:
        raise ValueError(
            f"Unknown agentic_model '{agentic_model}'. Choose from: {', '.join(ALL_AGENTS)}"
        )

    # ==================== Resolve file-level dataset identifier ====================
    # Build the split string used to locate queries/qrels files on disk
    # (e.g. "2024_news", "2025", "wiki2_test"); see resolve_split_id.
    dataset = kwargs.pop("dataset", "trqa")
    qrels_data_path = kwargs.pop("qrels_data_path", None) or data_path
    file_data_set = resolve_split_id(dataset, dataset_year, subset)

    if query_key is None:
        query_key = DATASET_SPECS[dataset].query_key

    # ==================== Load Dataset ====================
    # Single-pass load of queries + qrels + answers; filter to queries with qrels.
    min_rel_score = kwargs.get("min_relevance_score")
    same_qrels_path = qrels_data_path == data_path
    queries, qrels, answers = load_split(
        data_path, file_data_set, query_key=query_key,
        min_relevance_score=min_rel_score, only_with_qrels=same_qrels_path,
    )
    if not same_qrels_path:
        # qrels live in a separate directory; reload and re-filter against them.
        qrels = load_qrels(qrels_data_path, file_data_set, min_relevance_score=min_rel_score)
        queries = {qid: q for qid, q in queries.items() if qid in qrels}
    # Graded metrics use every grade with the official gains, not the threshold.
    graded_qrels = to_graded_qrels(
        load_qrels(qrels_data_path, file_data_set), DATASET_SPECS[dataset].relevance_gains,
    )
    if answers:
        print(f"Loaded {len(answers)} ground-truth answers (accuracy evaluation available)")

    # Keep the full set of questions for accuracy evaluation (before resume filtering)
    all_questions = dict(queries)

    # Gold of the criteria eval: only when the estimator extracts criteria
    # (monitor / inform) and the dataset has criteria gold.
    _estimator_mode = kwargs.get("uncertainty_estimator_mode", "off")
    criteria_gold = None
    if _estimator_mode != "off" and kwargs.get("llm_criteria") and DATASET_SPECS[dataset].criteria_gold:
        criteria_gold = load_gold_units(dataset, data_path, file_data_set)
    # Criteria bank of the split: criteria are read from it, extracted only when missing.
    criteria_bank = (str(criteria_bank_path(dataset, file_data_set))
                     if _estimator_mode != "off" and kwargs.get("criteria_bank", True) else None)

    # ==================== Resume: skip already-processed queries ====================
    llm_model = kwargs.pop("llm_model", "claude-sonnet-4-5")
    run_name = None
    run_dir  = None
    processed: set = set()

    if output_path:
        run_name = build_run_name_for_pipeline(agentic_model=agentic_model, llm_model=llm_model, **kwargs)
        retriever_label = kwargs.get("retriever_name", "e5")
        dataset_dir = build_dataset_dir_name(dataset, file_data_set, query_key, retriever_label)
        uncertainty_config_name = build_uncertainty_config_name(**kwargs)

        run_dir = str(Path(output_path) / dataset_dir / run_name / uncertainty_config_name)

        print(f"\n{'=' * 80}")
        print(f"[OUTPUT] Loading/saving results from: {run_dir}")
        print(f"{'=' * 80}")

        processed = get_processed_queries(
            run_dir, require_uncertainty=kwargs.get("uncertainty_estimator_mode", "off") != "off",
        )
        if processed:
            print(f"Found {len(processed)} already processed queries — skipping them")
            original_count = len(queries)
            queries = {qid: q for qid, q in queries.items() if qid not in processed}
            print(f"Remaining: {len(queries)} queries (skipped {original_count - len(queries)})")

        if not queries:
            print("All queries already processed — loading results from disk for evaluation")

    # ==================== Limit ====================
    if limit is not None and limit > 0:
        queries = dict(list(queries.items())[:limit])
        print(f"Limited to {len(queries)} queries for testing")

    # ==================== Eval-only mode ====================
    eval_only = kwargs.get("eval_only", False)
    if eval_only:
        if not run_dir:
            print("--eval-only requires a valid --output directory with existing results.")
            return
        _run_dir_str = str(run_dir)
        if not Path(_run_dir_str).exists():
            print(f"--eval-only: run_dir does not exist: {run_dir}")
            return

        processed_all = get_processed_queries(run_dir)
        if not processed_all:
            print(f"No retrieval data found in {_run_dir_str.rstrip('/')}/retrieval/surfaced. "
                  "Run the full pipeline first (without --eval-only).")
            return

        results = load_run_results(run_dir, processed_all)
        if not results:
            print(f"Could not load any results from {_run_dir_str}.")
            return
        print(f"Loaded {len(results)} queries for evaluation")

        evaluators = build_evaluators(qrels, kwargs, answers=answers, questions=all_questions,
                                      dataset=dataset, graded_qrels=graded_qrels,
                                      data_path=data_path, split=file_data_set,
                                      criteria_gold=criteria_gold)

        # Fusion runs first so its per-method surfaced-doc metrics can be folded
        # into the single summary.json written by evaluate_and_save.
        fusion_metrics = run_fusion_eval(results, qrels, kwargs, run_dir, gain_qrels=graded_qrels)

        # The LLM-as-judge evaluators (BrowseComp-Plus accuracy, Auto-ARGUE
        # reports) call the OpenRouter-hosted judge directly and reuse the
        # verdicts saved in the run directory; no local server to start.
        evaluate_and_save(results, evaluators, run_dir, fusion_metrics=fusion_metrics)
        return

    # ==================== Inject qrels into worker_config for the multi-GPU estimator ==
    if worker_config is not None and _estimator_mode != "off":
        worker_config["qrels"] = qrels
        worker_config["graded_qrels"] = graded_qrels
        worker_config["criteria_gold"] = criteria_gold
        worker_config["criteria_bank"] = criteria_bank

    # ==================== Build search tool ====================
    from searcher_component.searcher import RetrievalSearchTool

    _retriever = kwargs.get("retriever")
    search_tool = None
    if _retriever is not None:
        search_tool = RetrievalSearchTool(
            retriever=_retriever,
            post_retrieval_reranker=kwargs.get("post_retrieval_reranker"),
            post_fusion_reranker=kwargs.get("post_fusion_reranker"),
            top_k=kwargs.get("top_k", 100),
            rerank_top_k=kwargs.get("rerank_top_k", 100),
            retrieval_input=kwargs.get("retrieval_input", "subquery"),
            post_fusion_reranker_input=kwargs.get("post_fusion_reranker_input", "original_query"),
            ensure_novel_seen_docs=kwargs.get("ensure_novel_seen_docs", False),
            seen_top_k=kwargs.get("seen_top_k", 5),
        )

    # ==================== Instantiate Agent + Uncertainty estimator ====================
    # In multi-GPU mode neither the agent nor the estimator is built in the main
    # process; each worker spawns its own instances on its assigned GPU (see
    # orchestration._init_worker).  Building them here would create unused LLM
    # clients that are immediately torn down.
    from orchestration import build_agent, build_uncertainty_estimator

    agent = None
    estimator = None
    if num_gpus <= 1 and queries:
        agent = build_agent(
            agentic_model=agentic_model,
            agentic_model_cli=kwargs.get("agentic_model_cli", agentic_model),
            dataset=dataset,
            llm_model=llm_model,
            llm_client=kwargs.get("llm_client"),
            retriever=_retriever,
            max_iteration=kwargs.get("max_iteration", 100),
            seen_top_k=kwargs.get("seen_top_k", 5),
            verbose=verbose,
            search_tool=search_tool,
            use_plan=kwargs.get("use_plan", False),
            max_output_tokens_total=kwargs.get("max_output_tokens_total", 40000),
            temperature=kwargs.get("temperature", 0.0),
            max_extend_steps=kwargs.get("max_extend_steps", 5),
            max_retries=kwargs.get("max_retries", 3),
            hard_mode=kwargs.get("hard_mode", True),
            max_passage_chars=kwargs.get("max_passage_chars", 4000),
        )

        # Build the estimator AFTER the agent so it can take the agent's
        # intermediate-answer hook.
        estimator = build_uncertainty_estimator(
            mode=_estimator_mode,
            retriever=_retriever,
            qrels=qrels,
            graded_qrels=graded_qrels,
            llm_criteria=kwargs.get("llm_criteria"),
            max_criteria=kwargs.get("max_criteria"),
            criteria_judge_model=kwargs.get("criteria_judge_model", ""),
            add_intermediate_answer=kwargs.get("add_intermediate_answer", True),
            agent=agent if hasattr(agent, "uncertainty_estimator") else None,
            agentic_model=agentic_model,
            dataset=dataset,
            llm_model=llm_model,
            criteria_gold=criteria_gold,
            judge_model=kwargs.get("judge_model"),
            criteria_bank=criteria_bank,
        )
        if estimator is not None and hasattr(agent, "uncertainty_estimator"):
            agent.uncertainty_estimator = estimator

    # ==================== Setup output dirs + evaluators ====================
    evaluators = build_evaluators(qrels, kwargs, answers=answers, questions=all_questions,
                                  dataset=dataset, graded_qrels=graded_qrels,
                                  data_path=data_path, split=file_data_set,
                                  criteria_gold=criteria_gold)

    trajectory_dir = None
    if output_path:
        trajectory_dir = setup_output_dirs(run_dir, QUERY_OUTPUT_DIRS)["trajectory"]
        write_run_config(run_dir, agentic_model=agentic_model, llm_model=llm_model, **kwargs)
        print(f"\nProcessing {len(queries)} queries, saving results to {run_dir}/...")

    # ==================== Loop: run + save per query ====================
    results     = {}
    query_items = list(queries.items())

    if num_gpus > 1 and output_path and worker_config is not None:
        # ── Multi-GPU path ──────────────────────────────────────────────────
        # Split queries round-robin across GPU workers so load is balanced even
        # when queries vary in difficulty.
        chunks = [query_items[i::num_gpus] for i in range(num_gpus)]

        # The live bars redraw on every per-iteration progress event.  That is
        # what you want on a TTY and useless in an sbatch .out file, where each
        # redraw is appended verbatim (~1 MB per run).  Under --quiet the bars
        # are disabled and replaced by one line per finished query.
        quiet = bool(worker_config.get("quiet", False))

        print(f"\nParallel mode: {num_gpus} workers")
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i}: {len(chunk)} queries")

        bars = [
            tqdm(
                total=len(chunks[i]),
                position=i,
                leave=True,
                desc=f"[Worker {i}]",
                bar_format=(
                    "{desc} {percentage:3.0f}%|{bar}|"
                    " {n}/{total} [{elapsed}<{remaining}] {postfix}"
                ),
                dynamic_ncols=True,
                disable=quiet,
            )
            for i in range(num_gpus)
        ]
        if not quiet:
            for i, bar in enumerate(bars):
                bar.set_postfix_str("(—, —/—, starting)", refresh=True)

        mp_ctx = multiprocessing.get_context("spawn")
        _manager = mp_ctx.Manager()
        progress_queue = _manager.Queue()

        def _drain_progress(stop_event):
            done_count = 0
            while not stop_event.is_set():
                try:
                    item = progress_queue.get(timeout=0.3)
                except Exception:
                    continue
                if item is None:
                    done_count += 1
                    if done_count >= num_gpus:
                        break
                    continue
                w_id, qid, idx, total, stage, iter_info = item
                bar = bars[w_id]
                if quiet:
                    # Only per-query completions are worth a line; the
                    # "processing"/"update" ticks are the high-frequency ones.
                    if stage not in ("processing", "update"):
                        bar.n = idx
                        print(
                            f"[Worker {w_id}] {idx}/{total} done  ({qid}, {stage})",
                            flush=True,
                        )
                    continue
                if stage == "processing":
                    bar.set_postfix_str(f"({qid}, —/—, starting)", refresh=True)
                elif stage == "update":
                    stage_name, iteration, max_iter_val = iter_info
                    bar.set_postfix_str(
                        f"({qid}, {iteration}/{max_iter_val}, {stage_name})", refresh=True
                    )
                else:
                    bar.n = idx
                    iter_str = iter_info if iter_info else "—/—"
                    bar.set_postfix_str(f"({qid}, {iter_str}, {stage})", refresh=True)
                    bar.refresh()

        _drain_stop = threading.Event()
        drain_thread = threading.Thread(target=_drain_progress, args=(_drain_stop,), daemon=True)
        drain_thread.start()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_gpus, mp_context=mp_ctx
        ) as executor:
            futures = {
                executor.submit(gpu_worker, i, chunks[i], str(run_dir), worker_config, progress_queue, None): i
                for i in range(num_gpus)
            }
            for future in concurrent.futures.as_completed(futures):
                worker_id = futures[future]
                try:
                    worker_results = future.result()
                    results.update(worker_results)
                    tqdm.write(
                        f"[Worker {worker_id}] Completed {len(worker_results)}/{len(chunks[worker_id])} queries"
                    )
                except Exception as e:
                    tqdm.write(f"  Worker {worker_id} failed: {e}")

        _drain_stop.set()
        drain_thread.join(timeout=3)
        _manager.shutdown()

        for bar in bars:
            bar.close()

        print(f"\nCompleted {len(results)} queries (parallel)")

    else:
        # ── Sequential path (default / single GPU) ──────────────────────────
        for idx, (query_id, query_text) in enumerate(query_items, 1):
            if search_tool is not None:
                search_tool.reset()
            _answer_line = f"\n  Answer: {answers[query_id]}" if answers.get(query_id) else ""
            print(f"\n[{idx}/{len(query_items)}] Processing query: {query_id}\n  Query text: {query_text}{_answer_line}")
            # Streams each step to trajectory/{qid}.jsonl + .md as it happens,
            # so an interrupted run still leaves this query's work on disk.
            traj_logger = TrajectoryLogger(
                trajectory_dir, query_id, query_text,
                agent_name=agentic_model, model=llm_model,
                answer=answers.get(query_id),
            ) if trajectory_dir else None
            result = agent.run_single(
                query_id=query_id,
                query_text=query_text,
                temperature=kwargs.get("temperature", 0.7),
                trajectory_logger=traj_logger,
            )
            if result is None:
                print(f"  ✗ Skipping {query_id} (error during processing)")
                continue
            result["cited_docs_ranked_list"] = _build_cited_docs_ranked_list(result)
            # Append a References section to the generation (skips CPMReport which already has one)
            references = build_references_section(result)
            if references:
                result["generation"] = result["generation"].rstrip() + references
            results[query_id] = result

            if output_path:
                save_query_outputs(run_dir, query_id, query_text, result)
                print(f"  ✓ Saved: {query_id}")

        if agent:
            agent.cleanup()
        print(f"\nCompleted {len(results)} queries")

    # ==================== Read the run back from disk ====================
    # Every query of the run (this batch and earlier ones) is evaluated from
    # its saved files, exactly as --eval-only does, so both write the same
    # summary.json.  Without an output directory the in-memory results are
    # evaluated.
    if run_dir:
        results = load_run_results(run_dir, get_processed_queries(run_dir))

    # ==================== Evaluate + Save ====================
    _vllm_mgr = kwargs.get("vllm_manager")
    _total_gpus = kwargs.get("total_gpus_on_machine", 8)

    # ── Aggressively release ALL GPU memory ──────────────────────────
    # Break every reference chain to GPU-resident objects so gc can
    # collect them before we reclaim the GPUs.
    #
    # Reference chains that keep the encoder model alive:
    #   1. the estimator: its answer hook (bound method → agent → retriever.encoder),
    #      the encode closures of the novelty signals and the query scorer,
    #      and the NLI judge model
    #   2. agent.search_tool.retriever → retriever.encoder
    #   3. agent.retriever → retriever.encoder
    #   4. kwargs["retriever"] → retriever.encoder

    # 1) Sever closure/bound-method/model refs inside the estimator
    if estimator is not None:
        estimator.close()

    # 2) Move the encoder model off GPU *before* dropping references.
    #    Accelerate's device_map hooks can prevent gc from freeing GPU
    #    tensors even after all Python refs are gone; .cpu() forces the
    #    move and remove_hook_from_submodules detaches dispatch hooks.
    _enc = getattr(_retriever, "encoder", None) if _retriever is not None else None
    if _enc is not None and hasattr(_enc, "model"):
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(_enc.model)
        except Exception:
            pass
        _enc.model.cpu()
        del _enc.model
    del _enc

    # 3) Drop all local + kwargs references
    del agent, search_tool, _retriever, estimator
    kwargs.pop("retriever", None)
    kwargs.pop("post_retrieval_reranker", None)
    kwargs.pop("post_fusion_reranker", None)

    # 4) Force garbage collection and return GPU memory to CUDA
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) Kill vLLM servers, orphan processes, and verify GPUs are free
    if _vllm_mgr is not None:
        _vllm_mgr.shutdown_and_release_gpus(_total_gpus)

    # ==================== Multi-fusion evaluation ====================
    # Fusion runs first so its per-method surfaced-doc metrics can be folded
    # into the single summary.json written by evaluate_and_save.
    fusion_metrics = {}
    if results:
        fusion_metrics = run_fusion_eval(results, qrels, kwargs, run_dir, gain_qrels=graded_qrels)

    # The LLM-as-judge evaluators (BrowseComp-Plus accuracy, Auto-ARGUE reports)
    # call the OpenRouter-hosted judge directly; no local server to start.
    evaluate_and_save(results, evaluators, run_dir, fusion_metrics=fusion_metrics)

    # ==================== Final status ==========================================
    if output_path and run_dir:
        print(f"\n{'=' * 80}")
        print(f"[OUTPUT] All results saved to: {run_dir}")
        print(f"{'=' * 80}")


# ============================================================================
# CLI
# ============================================================================
def _none_if_null(value: str) -> Optional[str]:
    """Read the literal ``null`` on the CLI as None (auto-select), like the YAML."""
    return None if value == "null" else value


def _parse_args():
    """Build and parse the CLI argument parser.

    Only the frequently-varied knobs are declared as CLI arguments.  The
    mostly-fixed variables live in a YAML config file (``--config``, default
    experiments/configs/dra_inference.yaml) and are merged onto ``args`` afterwards;
    any of them can still be overridden by passing the matching ``--flag`` on
    the command line (handled via ``parse_known_args`` -> ``apply_config_to_args``).
    """
    parser = argparse.ArgumentParser(
        description="Run deep research agents pipeline (unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,  # only full flag names, e.g. --uncertainty-estimator-mode
    )

    # ── File-backed config ─────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="Path to the YAML file holding the mostly-fixed pipeline variables. Any value in it can be overridden by passing the matching --flag on the CLI.")

    # ── Frequently-varied knobs (everything else lives in --config) ─────────
    parser.add_argument("--agentic-model", type=str, default="glm", choices=list(AGENTIC_MODEL_TO_LLM), help="Agent to run; the LLM is selected automatically from the agent. uncertainty_aware = SearchR1-style agent that reads the <certainty> tag in inform mode, where its system prompt explains it (monitor/off: no tag and no explanation); cpm_report = Writing-as-Reasoning (report generation); searchr1/research/stepsearch/react/selfask/searcho1 = Reasoning-augmented retrieval; glm/oss_20b/oss_120b/tongyi = vendor-specific ReAct agents.")
    parser.add_argument("--dataset", type=str, default="ragtime", choices=list(DATASETS), help="Dataset; all use local indices.")
    parser.add_argument("--subset", type=_none_if_null, default="test", help="Dataset subset/collection (unset or null = the dataset's default in layout.DATASET_SPECS). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical; browsecomp_plus: test; ragtime: unused.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever; its index must be built for --dataset.")
    parser.add_argument("--uncertainty-estimator-mode", type=str, default="monitor", choices=["off", "monitor", "inform"], help="Uncertainty estimator mode. 'off': disabled. 'monitor': at the end of each search iteration compute and save the per-step uncertainty signals (doc/query novelty, criteria change, criteria attempts, new-item recall, intermediate answers) to uncertainty/{qid}.jsonl; the trajectory is never changed. 'inform': as monitor, and also append a <certainty> tag (criteria states, retrieval signals doc_novelty/criteria_delta, attempts per criterion, reasoning signal query_novelty; never gold-based signals) to the trajectory after each iteration's search results.")

    # ── Run-control flags ───────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="Cap number of queries (for quick tests)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPU workers for query-level parallelism. 0 = auto-detect from torch.cuda.device_count(). Each worker loads its own model instance on its assigned GPU.")
    parser.add_argument("--eval-only", type=_sm_bool, nargs="?", const=True, default=False, help="Skip agent execution and evaluate the run from its saved files (the run must have been completed at least once). Runs every evaluator of the dataset (generation, trajectory, seen/cited docs, fusion; answer accuracy where the dataset has answers: LLM judge via --judge-model for BrowseComp-Plus, numeric match for TRQA; Auto-ARGUE report scores for NeuCLIR and RAGTIME; criteria vs the dataset's criteria gold when the estimator is on) and writes the same summary.json as the run itself. Judge verdicts are reused from accuracy.jsonl, report_eval/ and the criteria scores (uncertainty meta lines), so an unchanged run makes no LLM call.")
    parser.add_argument("--quiet", type=_sm_bool, nargs="?", const=True, default=False, help="Print minimal logs (overrides verbose)")

    args, extras = parser.parse_known_args()

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    cli_subset = args.subset
    config = load_run_config(args.config)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset

    # ── Derive --llm-model from --agentic-model ────────────────────────────-
    args.llm_model = AGENTIC_MODEL_TO_LLM[args.agentic_model]
    args.agentic_model_cli = args.agentic_model
    args.agentic_model = AGENTIC_MODEL_ALIAS.get(args.agentic_model, args.agentic_model)

    # ── Resolve the sampling temperature from the agent ────────────────────-
    # Left on auto (null in the YAML), each agent gets the value it was tuned
    # at; an explicit --llm-temperature / YAML value wins for every agent.
    # Resolved here, before assemble_pipeline_kwargs, so the generator, the
    # worker config and the recorded run_config all see the one number.
    _explicit_temp = args.llm_temperature
    args.llm_temperature = resolve_temperature(args.agentic_model, _explicit_temp)
    if _explicit_temp is None:
        print(f"Auto-selected temperature: {args.llm_temperature} (from --agentic-model {args.agentic_model})")

    if args.output is None:
        args.output = _OUTPUT_PREFIX

    return args

def main():
    """Main entry point."""
    args    = _parse_args()

    verbose = args.verbose and not args.quiet

    if args.quiet:
        logging.getLogger("agents").setLevel(logging.ERROR)
        logging.getLogger("agent_tools").setLevel(logging.ERROR)
        logging.getLogger("utils").setLevel(logging.ERROR)
        logging.getLogger("prompts").setLevel(logging.ERROR)
        # Same treatment the workers get in _init_worker: no HF/datasets tqdm
        # bars, which single-GPU runs would otherwise draw in this process.
        _silence_hf_progress_bars()

    resolve_dataset_defaults(args)

    # ── Parse --gpu-ids and reconcile with --num-gpus ────────────────────
    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        if args.num_gpus <= 1:
            args.num_gpus = len(gpu_ids)
    num_gpus            = detect_num_gpus(args.num_gpus)

    # ── Auto-start vLLM servers if needed ─────────────────────────────────
    from utils.vllm_manager import VLLMServerManager
    vllm_manager = VLLMServerManager()

    # Detect the *real* total GPU count on the machine (not the user's
    # --num-gpus which is the desired worker count).
    try:
        import torch
        total_gpus_on_machine = max(1, torch.cuda.device_count())
    except ImportError:
        total_gpus_on_machine = num_gpus

    # ── GPU plan: retriever first (fixed footprint), then agent ──────────
    from utils.config import resolve_agent_backend, RETRIEVER_VRAM_GB
    _agent_backend, _agent_slug = resolve_agent_backend(args.agentic_model_cli)
    print(f"[GPU Plan] total GPUs : {total_gpus_on_machine}")
    print(f"[GPU Plan] retriever  : {args.retriever} "
          f"(~{RETRIEVER_VRAM_GB} GB fp16, FAISS on CPU — shares a worker GPU)")
    if _agent_backend == "api":
        print(f"[GPU Plan] agent      : {args.agentic_model} → "
              f"OpenRouter '{_agent_slug}' (0 local GPU)")
    else:
        print(f"[GPU Plan] agent      : {args.agentic_model} → "
              f"local vLLM (reserves leftmost GPUs)")

    if args.eval_only:
        # Eval-only: skip LLM/reranker vLLM servers.  The LLM-as-judge
        # evaluators use the OpenRouter-hosted judge, so no GPU is needed for it.
        if num_gpus > 1:
            gpu_ids = gpu_ids or list(range(min(num_gpus, total_gpus_on_machine)))
    elif gpu_ids is None:
        # Let the manager allocate GPUs: vLLM servers get leftmost GPUs,
        # remaining GPUs go to pipeline workers.
        worker_gpu_ids = vllm_manager.auto_start(args, total_gpus=total_gpus_on_machine)
        if worker_gpu_ids is not None:
            gpu_ids = worker_gpu_ids
            num_gpus = len(worker_gpu_ids)
            args.num_gpus = num_gpus
        elif num_gpus > 1:
            gpu_ids = list(range(min(num_gpus, total_gpus_on_machine)))
    else:
        # User specified --gpu-ids explicitly; still check whether vLLM
        # servers need to be started and adjust the worker set accordingly.
        worker_gpu_ids = vllm_manager.auto_start(args, total_gpus=total_gpus_on_machine)
        if worker_gpu_ids is not None:
            gpu_ids = worker_gpu_ids
            num_gpus = len(worker_gpu_ids)
            args.num_gpus = num_gpus

    # ── Restrict main process to pipeline GPUs before any CUDA init ───────
    # When --gpu-ids is specified the main process must not touch vLLM's
    # GPUs (typically 0..3).  Setting CUDA_VISIBLE_DEVICES early prevents
    # accidental CUDA context creation on those devices.
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    if args.eval_only:
        pipeline_kwargs = assemble_pipeline_kwargs(args, llm_client=None, retriever=None, num_gpus=num_gpus, verbose=verbose, gpu_ids=gpu_ids)
    elif num_gpus > 1:
        # Multi-GPU: each worker loads its own retriever on its assigned GPU.
        # Skip loading in the main process to avoid OOM on shared GPUs.
        llm_client = setup_llm(args, num_gpus)
        pipeline_kwargs = assemble_pipeline_kwargs(args, llm_client, retriever=None, num_gpus=num_gpus, verbose=verbose, gpu_ids=gpu_ids)
    else:
        llm_client  = setup_llm(args, num_gpus)
        retriever   = setup_retriever_from_args(args)
        pipeline_kwargs = assemble_pipeline_kwargs(args, llm_client, retriever, num_gpus, verbose, gpu_ids=gpu_ids)

    # ── Deep-research-pipeline-specific kwargs ────────────────────────
    pipeline_kwargs.update({
        "fusion_k": args.fusion_k,
        "fusion_methods": args.fusion_methods,
        "eval_only": args.eval_only,
        "vllm_manager": vllm_manager,
        "total_gpus_on_machine": total_gpus_on_machine,
        "judge_model": args.judge_model,
    })

    # ── Optional post-retrieval & post-fusion rerankers ─────────────────
    # In multi-GPU mode each worker builds its own reranker on its assigned
    # GPU (see _init_worker).  Loading one in the main process would waste
    # GPU memory on a device that a worker needs (e.g. rankllama = ~14 GB).
    if num_gpus <= 1 and not args.eval_only:
        from utils.config import get_reranker_configs
        _reranker_configs = get_reranker_configs(args.rerank_top_k)
        from searcher_component.rerankers import build_reranker_from_config
        if args.post_retrieval_reranker != "null":
            pipeline_kwargs["post_retrieval_reranker"] = build_reranker_from_config(
                args.post_retrieval_reranker, _reranker_configs,
            )
        if args.post_fusion_reranker != "null":
            pipeline_kwargs["post_fusion_reranker"] = build_reranker_from_config(
                args.post_fusion_reranker, _reranker_configs,
            )
    pipeline_kwargs["rerank_top_k"] = args.rerank_top_k
    pipeline_kwargs["retrieval_input"] = args.retrieval_input
    pipeline_kwargs["post_fusion_reranker_input"] = args.post_fusion_reranker_input
    pipeline_kwargs["post_retrieval_reranker_name"] = args.post_retrieval_reranker
    pipeline_kwargs["post_fusion_reranker_name"] = args.post_fusion_reranker
    pipeline_kwargs["ensure_novel_seen_docs"] = args.ensure_novel_seen_docs

    try:
        run_pipeline(
            data_path=args.data_path,
            subset=args.subset,
            dataset_year=args.dataset_year,
            query_key=args.query_key,
            output_path=args.output,
            agentic_model=args.agentic_model,
            limit=args.limit,
            verbose=verbose,
            **pipeline_kwargs,
        )
        print("Pipeline execution completed successfully")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise
    finally:
        time.sleep(0.5)
        cleanup_event_loop()
        vllm_manager.shutdown()

if __name__ == "__main__":
    main()


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
#   run_outputs/{dataset}_{split}_{query_key}_{retriever}/{agent}_{backend}_{model}/{uncertainty_config}/
#   e.g. run_outputs/neuclir_2024_news_e5/oss_vllm_gpt-oss-20b/ue-monitor/
#     ├── run_config.json          full agent/searcher/uncertainty-estimator settings
#     ├── retrieval/
#     │   ├── surfaced/
#     │   │   └── {query_id}.trec   per-query surfaced-doc TREC (raw retriever output, col 6 = iter_N)
#     │   ├── seen/
#     │   │   └── {query_id}.trec   per-query seen-doc TREC file (docs shown to the LLM)
#     │   ├── cited/
#     │   │   └── {query_id}.trec   per-query cited-doc TREC file (docs cited by the LLM)
#     │   └── fusion_{method}.trec  deduped fusion ranking, single aggregate over all queries
#     ├── generation/
#     │   └── {query_id}.md     per-query generation output
#     ├── trajectory/
#     │   ├── {query_id}.jsonl  per-query trajectory: one line per step + a trailing meta line
#     │   └── {query_id}.md     same trajectory, human-readable, written live
#     ├── uncertainty/
#     │   └── {query_id}.jsonl  per-query uncertainty signals: meta line + one line per iteration
#     ├── accuracy.jsonl        per-query answer correctness (datasets with answers)
#     ├── report_eval/          Auto-ARGUE inputs, cached judgments and per-query scores.tsv
#     └── summary.json          grouped: num_queries, retrieval{seen,cited,fusion},
#                                        generation{correctness|nuggets,stats},
#                                        criteria, trajectory
#
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#   CUDA_VISIBLE_DEVICES=5,6 python experiments/dra_inference.py --dataset browsecomp_plus --limit 1
#   CUDA_VISIBLE_DEVICES=0,1,2 python experiments/dra_inference.py --dataset neuclir --limit 1
#   python experiments/dra_inference.py --dataset neuclir --num-gpus 6 --quiet --limit 6
#   python experiments/dra_inference.py --limit 2
