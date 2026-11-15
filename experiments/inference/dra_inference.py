"""Run deep research agents pipeline.

Supported agents via --policy-model (the LLM is selected automatically per agent):
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

Agentic workflows:
    ReAct-style (react, selfask, searcho1, research, searchr1, stepsearch, drtulu, glm, oss_20b, oss_120b, tongyi, cpm_explore):
        Query → [Think → Search → Observe]* → Report → Evaluate
        Instruction-tuned : react, selfask, searcho1
        RL-trained        : research, searchr1, stepsearch, drtulu, tongyi, cpm_explore, glm, oss_20b, oss_120b

    Outline-style (webweaver):
        Query → [Think → Search → Write_outline]* → Outline → [Think → Retrieve → Write_section]* → Report → Evaluate

    Report-style (cpm_report):
        Query → Search → Init Plan → [Search → Write]* → [Extend Plan → [Search → Write]*]* → Report → Evaluate


Stages (--stage):
    generate   Run the agent on the queries without a finished trajectory, save
               its files, then evaluate the run.
    belief     Run the belief model over the finished trajectories without a
               belief file (orchestration.belief_stage: no agent, no retrieval;
               document text from the corpus by id), then evaluate the run.
               Other belief settings than the run's replace its beliefs.
    eval       Evaluate the run from its saved files (judge verdicts reused).
    all        generate, then belief, then evaluate.

Output structure: see OUTPUT STRUCTURE at the end of this file.
"""

import argparse
import shutil
import warnings
import logging
import time
import os
import concurrent.futures
import multiprocessing
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
load_dotenv()

# Suppress async cleanup noise
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("asyncio.sslproto").setLevel(logging.CRITICAL)

from indexing_corpus_dataset.dataset_loaders import load_qrels, load_split, resolve_split_id
from indexing_corpus_dataset.layout import CRITERIA_SOURCES, DATASETS, DATASET_SPECS, OUTPUT_ROOT

from deep_research_agents.agents import ALL_AGENTS
from utils.config import POLICY_MODEL_TO_LLM, POLICY_MODEL_ALIAS, resolve_temperature
from orchestration import (
    gpu_worker,
    quiet_logging,
    setup_retriever_from_args,
)
from utils.llm_client import setup_llm
from utils.progress import WorkerBars
from utils.cli_setup import (
    resolve_dataset_defaults,
    count_gpus,
    resolve_workers,
    assemble_pipeline_kwargs,
    cleanup_event_loop,
    load_run_config,
    parse_cli_overrides,
    apply_config_to_args,
    _sm_bool,
)
from utils.io_utils import setup_output_dirs
from run_files import QUERY_OUTPUT_DIRS, believed_queries, scan_queries
from run_files.run_config import (
    belief_config_diffs,
    build_belief_config,
    build_dataset_dir_name,
    build_run_config,
    build_run_name_for_pipeline,
    check_run_config,
    check_run_dataset,
    read_run_config,
    set_belief_config,
    update_run_config_evaluation,
)
from evaluation.metrics_file import BELIEF_EVAL_DIR
from evaluation.runner import build_evaluators, evaluate_and_save, load_run_results

_OUTPUT_PREFIX = str(OUTPUT_ROOT)
_CONFIG_DEFAULT = str(Path(__file__).resolve().parents[1] / "configs" / "dra_inference.yaml")

#: The stages of a run (see the module docstring).
STAGES = ("generate", "belief", "eval", "all")
GENERATES = ("generate", "all")
BELIEVES = ("belief", "all")


# ============================================================================
# Pipeline
# ============================================================================
@dataclass
class RunPlan:
    """What a run works on, settled before any model is loaded (see
    :func:`prepare_run`)."""
    dataset: str
    split: str                      # file-level split id, e.g. "wiki2_test"
    query_key: str
    queries: Dict[str, str]         # the queries still to generate (resume and --limit applied)
    all_questions: Dict[str, str]   # every query of the split, for accuracy evaluation
    qrels: Dict[str, Dict[str, int]]         # at min_relevance_score
    answers: Dict[str, str]
    run_dir: str
    limit: Optional[int]            # --limit: the cap of each stage's queries


def prepare_run(data_path: str, output_path: str, policy_model: str, subset: Optional[str] = None,
                dataset_year: Optional[str] = None, query_key: Optional[str] = None,
                limit: Optional[int] = None, **kwargs) -> RunPlan:
    """Load the split, name the run directory and find the queries left to
    generate (the generate stage).

    Done before any model, retriever or vLLM server is set up, so a run with
    nothing left to generate sets up none.

    Args:
        data_path:     Path to dataset directory.
        output_path:   Root path for saving results.
        policy_model:  Agent to use (internal name); one of ALL_AGENTS.
        subset:        Dataset subset identifier, e.g. "news" or "technical"
                       for neuclir, "wiki2" for trqa.
        dataset_year:  Dataset year (neuclir "2024", ragtime "2025"); trqa
                       carries its eval split (test|validation) here.
        query_key:     Key in each JSONL query record to use as the query text.
                       None = the dataset's default (layout.DATASET_SPECS),
                       e.g. "request" for neuclir.  Records without the key
                       are not run.
        limit:         Cap the number of queries (for quick tests), applied
                       to the queries left to generate.
        **kwargs:      The pipeline kwargs (``assemble_pipeline_kwargs``).
    """
    if policy_model not in ALL_AGENTS:
        raise ValueError(
            f"Unknown policy_model '{policy_model}'. Choose from: {', '.join(ALL_AGENTS)}"
        )
    kwargs = dict(kwargs)

    # ==================== Resolve file-level dataset identifier ====================
    # Build the split string used to locate queries/qrels files on disk
    # (e.g. "2024_news", "2025", "wiki2_test"); see resolve_split_id.
    dataset = kwargs.pop("dataset")
    qrels_data_path = kwargs.pop("qrels_data_path", None) or data_path
    file_data_set = resolve_split_id(dataset, dataset_year, subset)

    if query_key is None:
        query_key = DATASET_SPECS[dataset].query_key

    # ==================== Load Dataset ====================
    # The qrels are read once; the evaluation and the belief model use the
    # grades at min_relevance_score.  Only queries with a relevant doc are run.
    queries, all_qrels, answers = load_split(data_path, file_data_set, query_key=query_key)
    if qrels_data_path != data_path:
        all_qrels = load_qrels(qrels_data_path, file_data_set)
    min_rel_score = kwargs.get("min_relevance_score")
    qrels = {}
    for qid, docs in all_qrels.items():
        kept = {did: rel for did, rel in docs.items() if min_rel_score is None or rel >= min_rel_score}
        if kept:
            qrels[qid] = kept
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    if answers:
        print(f"Loaded {len(answers)} ground-truth answers (accuracy evaluation available)")

    # Only the queries listed in the file (one id per line).
    query_ids_file = kwargs.pop("query_ids_file", None)
    if query_ids_file:
        wanted = {line.strip() for line in Path(query_ids_file).read_text().splitlines() if line.strip()}
        queries = {qid: q for qid, q in queries.items() if qid in wanted}
        print(f"Kept {len(queries)} queries listed in {query_ids_file} ({len(wanted)} ids)")

    # Keep the full set of questions for accuracy evaluation (before resume filtering)
    all_questions = dict(queries)

    stage = kwargs["stage"]
    if stage in BELIEVES:
        from belief_model_component import check_criteria_inputs
        check_criteria_inputs(dataset, kwargs["criteria_source"], kwargs["belief_model"], kwargs["max_criteria"])

    # ==================== Resume: skip already-generated queries ====================
    # Named and recorded by the CLI --policy-model name (oss_20b, not oss).
    llm_model = kwargs.pop("llm_model")
    policy_model_cli = kwargs.pop("policy_model_cli")
    run_name = build_run_name_for_pipeline(policy_model=policy_model_cli, llm_model=llm_model, **kwargs)
    dataset_dir = build_dataset_dir_name(dataset, file_data_set, query_key, kwargs["retriever_name"])
    run_dir = str(Path(output_path) / dataset_dir / run_name)

    print(f"\n{'=' * 80}")
    print(f"[OUTPUT] Loading/saving results from: {run_dir}")
    print(f"{'=' * 80}")

    run_config = build_run_config(
        policy_model_cli, llm_model, dataset=dataset, split=file_data_set, query_key=query_key, **kwargs,
    )
    if stage in GENERATES:
        # Every trajectory of the run directory is generated with the same
        # settings: a resume with other ones stops here, before any query runs.
        check_run_config(run_dir, run_config)
    else:
        # The belief and eval stages read the saved run with this call's
        # qrels and corpus (the dataset section, which must be the run's);
        # only the evaluation settings (e.g. the judge) may change.
        if not (Path(run_dir) / "run_config.json").exists():
            raise SystemExit(f"No run in {run_dir} (no run_config.json): run --stage generate first.")
        check_run_dataset(run_dir, run_config)
        update_run_config_evaluation(run_dir, run_config)

    if stage not in GENERATES:
        queries = {}
    else:
        generated = scan_queries(run_dir)[0]
        if generated:
            print(f"Found {len(generated)} already generated queries — skipping them")
            original_count = len(queries)
            queries = {qid: q for qid, q in queries.items() if qid not in generated}
            print(f"Remaining: {len(queries)} queries (skipped {original_count - len(queries)})")
        if limit is not None and limit > 0:
            queries = dict(list(queries.items())[:limit])
            print(f"Limited to {len(queries)} queries for testing")
        if not queries:
            print("All queries already generated")

    if stage == "all" and queries and (kwargs["belief_model"] or "").startswith("vllm/"):
        _check_belief_server_survives(kwargs["belief_model"])

    return RunPlan(
        dataset=dataset, split=file_data_set, query_key=query_key, queries=queries,
        all_questions=all_questions, qrels=qrels, answers=answers,
        run_dir=run_dir, limit=limit,
    )


def _check_belief_server_survives(belief_model: str) -> None:
    """``--stage all`` with a ``vllm/`` belief model: after generating, the
    GPU release kills every process on the known server ports
    (``VLLMServerManager.cleanup_ports``), so a belief model served on one
    of them would be gone before the belief stage.  Stops before any query
    runs."""
    from urllib.parse import urlparse

    from belief_model_component import belief_vllm_api_base
    from utils.vllm_manager import VLLMServerManager

    base = belief_vllm_api_base()
    port = urlparse(base).port
    if port in VLLMServerManager.cleanup_ports():
        raise SystemExit(
            f"--stage all with belief model {belief_model} at {base}: port {port} is cleared after the generate "
            "stage, so the belief model's server would be killed before the belief stage. Run --stage generate, "
            "then --stage belief, or serve the belief model on another port ($BELIEF_VLLM_API_BASE)."
        )


def run_pipeline(plan: RunPlan, data_path: str, policy_model: str = "cpm_report", verbose: bool = True,
                 workers: int = 1, worker_config: Optional[dict] = None, **kwargs):
    """Run the stages of ``kwargs["stage"]`` (see the module docstring), then
    evaluate the whole run.

    Args:
        plan:          From :func:`prepare_run`.
        data_path:     Path to dataset directory (nuggets of the report datasets).
        policy_model:  Agent to use (internal name); one of ALL_AGENTS.
        verbose:       Whether to print detailed logs.
        workers:       Query-level worker processes of the generate stage.
                       1 = sequential (default). >1 = one process per worker
                       (one GPU each on a GPU machine, CPU otherwise), queries
                       split across them; each saves its queries' files
                       straight to the run directory.
        worker_config: Serialisable config dict passed to each worker so it
                       can rebuild the agent stack.  Required when workers > 1.
        **kwargs:      Agent-specific, belief and evaluation parameters.
    """
    dataset, run_dir, answers = plan.dataset, plan.run_dir, plan.answers
    stage = kwargs["stage"]

    evaluators = build_evaluators(plan.qrels, kwargs, answers=answers, questions=plan.all_questions,
                                  dataset=dataset,
                                  data_path=data_path, split=plan.split)

    if stage in GENERATES and plan.queries:
        _run_queries(plan, policy_model, kwargs["llm_model"], verbose, workers, worker_config, kwargs)
    if stage in BELIEVES:
        _run_belief(plan, verbose, kwargs)

    # ==================== Read the run back from disk ====================
    # Every query of the run is evaluated from its saved files, whatever the
    # stage, so every stage writes the same eval/metrics.md and eval/per_query.csv.
    finished, unfinished = scan_queries(run_dir)
    if unfinished:
        print(f"\n{len(unfinished)} unfinished queries (no meta line in trajectory/{{qid}}.jsonl); "
              f"they are generated again on resume: "
              f"{', '.join(sorted(unfinished)[:20])}{' ...' if len(unfinished) > 20 else ''}")
    if not finished:
        print(f"No finished query in {run_dir.rstrip('/')}/trajectory. Run --stage generate first.")
        return
    results = load_run_results(run_dir, finished)
    if not results:
        print(f"Could not load any results from {run_dir}.")
        return

    # The LLM-as-judge evaluators (BrowseComp-Plus accuracy, Auto-ARGUE reports)
    # call the OpenRouter-hosted judge directly and reuse the verdicts saved in
    # the run directory; no local server to start.
    evaluate_and_save(results, evaluators, run_dir, num_unfinished=len(unfinished))

    # ==================== Final status ==========================================
    print(f"\n{'=' * 80}")
    print(f"[OUTPUT] All results saved to: {run_dir}")
    print(f"{'=' * 80}")


def _run_belief(plan: RunPlan, verbose: bool, kwargs: dict) -> None:
    """The belief stage: the belief model over the run's finished
    trajectories without a belief file (``orchestration.belief_stage``).

    Its settings are the ``belief_model`` section of ``run_config.json``
    (``build_belief_config``; the passage cut, retriever and agent are the
    generate stage's, read from the file; the corpus is this call's, of
    the run's dataset and split, ``check_run_dataset``); when they differ
    from the saved ones, the run's beliefs (``belief/``, ``eval/belief/``)
    are removed, then the new settings saved, and every query is run
    again.  A belief file of an older schema is made again too.  With
    ``--limit`` (``--stage all``), the queries this call generated come
    first."""
    from belief_model_component import build_criteria_source, prepare_criteria
    from orchestration import build_belief_model
    from orchestration.belief_stage import run_belief_stage

    run_dir = Path(plan.run_dir)
    saved = read_run_config(run_dir)
    max_passage_tokens = saved["agent"]["max_passage_tokens"]
    section = build_belief_config(
        plan.dataset, max_passage_tokens=max_passage_tokens,
        **{k: kwargs[k] for k in ("belief_model", "max_criteria", "criteria_source")},
    )
    diffs = belief_config_diffs(run_dir, section)
    if diffs:
        print(f"\n[Belief] Other belief settings than the run's ({'; '.join(diffs)}): "
              f"removing its beliefs (belief/, {BELIEF_EVAL_DIR}/) and running every query again")
        # Removed before the new settings are saved: an interrupted call
        # never leaves old beliefs under the new settings.
        for name in ("belief", BELIEF_EVAL_DIR):
            shutil.rmtree(run_dir / name, ignore_errors=True)
    set_belief_config(run_dir, section)

    finished = scan_queries(run_dir)[0]
    done, stale = believed_queries(run_dir)
    todo = [qid for qid in plan.all_questions if qid in finished and qid not in done]
    todo.sort(key=lambda qid: qid not in plan.queries)   # this call's generated queries first
    if done:
        print(f"[Belief] Found {len(done & finished)} queries with a belief file — skipping them")
    if stale & finished:
        print(f"[Belief] {len(stale & finished)} belief files of an older schema: running those queries again")
    if plan.limit is not None and plan.limit > 0:
        todo = todo[:plan.limit]
    if not todo:
        print("[Belief] No finished trajectory without a belief file")
        return

    # Criteria, made and banked before the workers start; the workers'
    # belief models share this source and read every query's from the bank.
    source = build_criteria_source(plan.dataset, plan.split, belief_model=kwargs["belief_model"],
                                   max_criteria=kwargs["max_criteria"], criteria_source=kwargs["criteria_source"])
    prepare_criteria(source, {qid: plan.all_questions[qid] for qid in todo})

    def build_model():
        return build_belief_model(
            dataset=plan.dataset, belief_model=kwargs["belief_model"], criteria=source,
            qrels=plan.qrels, max_passage_tokens=max_passage_tokens,
        )

    print(f"\n[Belief] Running the belief model on {len(todo)} queries, saving to {run_dir}/belief/...")
    n_done = run_belief_stage(
        run_dir, todo, build_model=build_model, corpus_path=kwargs["corpus_path"],
        retriever_name=saved["searcher"]["retriever_name"], policy_model=saved["agent"]["policy_model"],
        workers=kwargs["belief_workers"], verbose=verbose,
    )
    print(f"[Belief] Completed {n_done}/{len(todo)} queries")


def _run_queries(plan: RunPlan, policy_model: str, llm_model: str, verbose: bool, workers: int,
                 worker_config: Optional[dict], kwargs: dict) -> None:
    """Run the agent on ``plan.queries`` and save each query's files under
    ``plan.run_dir``, then release every GPU the run held."""
    run_dir, answers = plan.run_dir, plan.answers

    # The gold answers, for the header of each query's trajectory .md.
    if worker_config is not None:
        worker_config["answers"] = answers

    # ==================== Build search tool ====================
    from searcher_component.searcher import RetrievalSearchTool

    _retriever = kwargs.get("retriever")
    search_tool = None
    if _retriever is not None:
        search_tool = RetrievalSearchTool(
            retriever=_retriever,
            post_retrieval_reranker=kwargs.get("post_retrieval_reranker"),
            post_fusion_reranker=kwargs.get("post_fusion_reranker"),
            rerank_top_k=kwargs["rerank_top_k"],
            retrieval_input=kwargs["retrieval_input"],
            post_fusion_reranker_input=kwargs["post_fusion_reranker_input"],
        )

    # ==================== Instantiate Agent ====================
    # With several workers the agent is not built in the main process; each
    # worker spawns its own instance on its assigned GPU or the CPU (see
    # orchestration._init_worker).  Building it here would create unused LLM
    # clients that are immediately torn down.
    from orchestration import build_agent, run_query

    agent = None
    if workers <= 1:
        agent = build_agent(
            policy_model=policy_model,
            policy_model_cli=kwargs["policy_model_cli"],
            dataset=plan.dataset,
            llm_model=llm_model,
            llm_client=kwargs.get("llm_client"),
            retriever=_retriever,
            max_iteration=kwargs["max_iteration"],
            seen_top_k=kwargs["seen_top_k"],
            verbose=verbose,
            search_tool=search_tool,
            use_plan=kwargs["use_plan"],
            max_output_tokens_total=kwargs["max_output_tokens_total"],
            temperature=kwargs["temperature"],
            max_extend_steps=kwargs["max_extend_steps"],
            max_retries=kwargs["max_retries"],
            hard_mode=kwargs["hard_mode"],
            max_passage_tokens=kwargs["max_passage_tokens"],
        )

    setup_output_dirs(run_dir, QUERY_OUTPUT_DIRS)
    print(f"\nProcessing {len(plan.queries)} queries, saving results to {run_dir}/...")

    # ==================== Loop: run + save per query ====================
    # Only the number of finished queries is kept: the evaluation reads the
    # results back from their files.
    n_done = 0
    query_items = list(plan.queries.items())

    if workers > 1 and worker_config is not None:
        # ── Parallel path ───────────────────────────────────────────────────
        # Split queries round-robin across the workers so load is balanced even
        # when queries vary in difficulty; never more workers than queries.
        num_workers = min(workers, len(query_items))
        chunks = [query_items[i::num_workers] for i in range(num_workers)]

        # One bar per worker (utils.progress.WorkerBars): live on a terminal,
        # one line per finished query off it (an sbatch .out file).
        bars = WorkerBars([len(c) for c in chunks], "Worker")
        worker_config = {**worker_config, "live_bars": bars.live}

        print(f"\nParallel mode: {num_workers} workers")
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i}: {len(chunk)} queries")

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
                    if done_count >= num_workers:
                        break
                    continue
                w_id, qid, _idx, _total, stage, iter_info = item
                if stage == "processing":
                    bars.status(w_id, f"{qid}, starting")
                elif stage == "update":
                    stage_name, iteration, max_iter_val = iter_info
                    bars.status(w_id, f"{qid}, turn {iteration}/{max_iter_val}, {stage_name}")
                else:
                    bars.done(w_id, qid, f"turns {iter_info}" if iter_info else stage)

        _drain_stop = threading.Event()
        drain_thread = threading.Thread(target=_drain_progress, args=(_drain_stop,), daemon=True)
        drain_thread.start()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp_ctx
        ) as executor:
            futures = {
                executor.submit(gpu_worker, i, chunks[i], run_dir, worker_config, progress_queue): i
                for i in range(num_workers)
            }
            for future in concurrent.futures.as_completed(futures):
                worker_id = futures[future]
                try:
                    worker_done = future.result()
                    n_done += len(worker_done)
                    bars.write(
                        f"[Worker {worker_id}] Completed {len(worker_done)}/{len(chunks[worker_id])} queries"
                    )
                except Exception as e:
                    bars.write(f"  Worker {worker_id} failed: {e}")

        _drain_stop.set()
        drain_thread.join(timeout=3)
        _manager.shutdown()
        bars.close()

        print(f"\nCompleted {n_done} queries (parallel)")

    else:
        # ── Sequential path (default / one worker) ──────────────────────────
        # Not verbose: the same bar as one parallel worker.  Verbose: the
        # agent's trace, each query under a header line.
        max_iteration = kwargs["max_iteration"]
        bars = None if verbose else WorkerBars([len(query_items)], "Worker")
        for idx, (query_id, query_text) in enumerate(query_items, 1):
            status_cb = None
            if bars is None:
                _answer_line = f"\n  Answer: {answers[query_id]}" if answers.get(query_id) else ""
                print(f"\n[{idx}/{len(query_items)}] Processing query: {query_id}\n  Query text: {query_text}{_answer_line}")
            else:
                bars.status(0, f"{query_id}, starting")

                def status_cb(stage: str, iteration: int, _qid=query_id):
                    bars.status(0, f"{_qid}, turn {iteration}/{max_iteration}, {stage}")
            result = run_query(
                agent, run_dir, query_id, query_text,
                policy_model=policy_model, llm_model=llm_model,
                answer=answers.get(query_id), temperature=kwargs["temperature"],
                status_callback=status_cb,
            )
            if result is None:
                if bars is None:
                    print(f"  ✗ Skipping {query_id} (error during processing)")
                else:
                    bars.done(0, query_id, "skipped")
                continue
            n_done += 1
            if bars is None:
                print(f"  ✓ Saved: {query_id}")
            else:
                bars.done(0, query_id, f"turns {result.get('num_iterations', '?')}/{max_iteration}")
        if bars is not None:
            bars.close()

        if agent:
            agent.cleanup()
        print(f"\nCompleted {n_done} queries")

    # ==================== Release the GPUs ====================
    _vllm_mgr = kwargs.get("vllm_manager")
    _total_gpus = kwargs["total_gpus_on_machine"]

    # ── Aggressively release ALL GPU memory ──────────────────────────
    # Break every reference chain to GPU-resident objects so gc can
    # collect them before we reclaim the GPUs.
    #
    # Reference chains that keep the encoder model alive:
    #   1. agent.search_tool.retriever → retriever.encoder
    #   2. agent.retriever → retriever.encoder
    #   3. kwargs["retriever"] → retriever.encoder

    # 1) Move the encoder model off GPU *before* dropping references.
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

    # 2) Drop all local + kwargs references
    del agent, search_tool, _retriever
    kwargs.pop("retriever", None)
    kwargs.pop("post_retrieval_reranker", None)
    kwargs.pop("post_fusion_reranker", None)

    # 3) Force garbage collection and return GPU memory to CUDA
    import gc
    gc.collect()
    if _total_gpus == 0:
        # CPU machine: no GPU memory to return, no vLLM server was started.
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4) Kill vLLM servers, orphan processes, and verify GPUs are free
    if _vllm_mgr is not None:
        _vllm_mgr.shutdown_and_release_gpus(_total_gpus)


# ============================================================================
# Run settings
# ============================================================================
def _resolve_args(args, extras):
    """Merge the YAML config and CLI overrides onto ``args`` and derive the agent settings."""

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    cli_subset = args.subset
    cli_criteria_source = args.criteria_source
    cli_belief_model = args.belief_model
    config = load_run_config(args.config)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset
    if cli_criteria_source is not None:
        args.criteria_source = cli_criteria_source
    if cli_belief_model is not None:
        args.belief_model = cli_belief_model

    # ── Derive --llm-model from --policy-model ────────────────────────────-
    args.llm_model = POLICY_MODEL_TO_LLM[args.policy_model]
    args.policy_model_cli = args.policy_model
    args.policy_model = POLICY_MODEL_ALIAS.get(args.policy_model, args.policy_model)

    # ── Resolve the sampling temperature from the agent ────────────────────-
    # Left on auto (null in the YAML), each agent gets the value it was tuned
    # at; an explicit --llm-temperature / YAML value wins for every agent.
    # Resolved here, before assemble_pipeline_kwargs, so the generator, the
    # worker config and the recorded run_config all see the one number.
    _explicit_temp = args.llm_temperature
    args.llm_temperature = resolve_temperature(args.policy_model, _explicit_temp)
    if _explicit_temp is None:
        print(f"Auto-selected temperature: {args.llm_temperature} (from --policy-model {args.policy_model_cli})")

    if args.output is None:
        args.output = _OUTPUT_PREFIX

    return args


def main():
    """Main entry point."""
    args    = _resolve_args(*parse_args())

    verbose = args.verbose and not args.quiet

    if args.quiet:
        # Same treatment the workers get in _init_worker: no HF/datasets tqdm
        # bars, which single-GPU runs would otherwise draw in this process.
        quiet_logging()

    resolve_dataset_defaults(args)

    from utils.vllm_manager import VLLMServerManager
    vllm_manager = VLLMServerManager()

    # Workers and GPUs of the generate stage, set below when there are
    # queries to generate: the belief and eval stages use no GPU (they run
    # on a CPU machine), so nothing here touches torch or CUDA for them.
    workers = 1
    gpu_ids = None
    total_gpus_on_machine = 0

    def _pipeline_kwargs(llm_client=None, retriever=None) -> dict:
        kwargs = assemble_pipeline_kwargs(args, llm_client, retriever, workers, verbose, gpu_ids=gpu_ids)
        # ── Deep-research-pipeline-specific kwargs ────────────────────────
        kwargs.update({
            "stage": args.stage,
            "vllm_manager": vllm_manager,
            "total_gpus_on_machine": total_gpus_on_machine,
            "judge_model": args.judge_model,
            "query_ids_file": args.query_ids_file,
            "rerank_top_k": args.rerank_top_k,
            "retrieval_input": args.retrieval_input,
            "post_fusion_reranker_input": args.post_fusion_reranker_input,
            "post_retrieval_reranker_name": args.post_retrieval_reranker,
            "post_fusion_reranker_name": args.post_fusion_reranker,
        })
        return kwargs

    try:
        # ── Queries left to run, before any model is set up ──────────────
        pipeline_kwargs = _pipeline_kwargs()
        plan = prepare_run(
            data_path=args.data_path,
            output_path=args.output,
            policy_model=args.policy_model,
            subset=args.subset,
            dataset_year=args.dataset_year,
            query_key=args.query_key,
            limit=args.limit,
            **pipeline_kwargs,
        )

        # With nothing to generate (the belief or eval stage, or every query
        # generated) no vLLM server, LLM client, retriever or reranker is set
        # up.  The belief stage reads the documents from the corpus and calls
        # its own model (belief_workers threads); the LLM-as-judge evaluators
        # use the OpenRouter-hosted judge, so no GPU is needed for them.
        if not plan.queries and (args.workers != 1 or args.gpu_ids is not None):
            print("[Workers] Nothing to generate: --workers and --gpu-ids are ignored "
                  "(the belief stage runs --belief-workers threads)")
        if plan.queries:
            # ── Workers and GPUs ─────────────────────────────────────────────
            total_gpus_on_machine = count_gpus()
            if args.gpu_ids is not None:
                gpu_ids = [int(x) for x in str(args.gpu_ids).split(",")]
            workers = resolve_workers(args.workers, total_gpus_on_machine, gpu_ids)

            # ── GPU plan: retriever first (fixed footprint), then agent ──────────
            from utils.config import resolve_agent_backend, RETRIEVER_VRAM_GB
            _agent_backend, _agent_slug = resolve_agent_backend(args.policy_model_cli)
            print(f"[GPU Plan] total GPUs : {total_gpus_on_machine}")
            if total_gpus_on_machine:
                print(f"[GPU Plan] retriever  : {args.retriever} "
                      f"(~{RETRIEVER_VRAM_GB} GB fp16, FAISS on CPU — shares a worker GPU)")
            else:
                print(f"[GPU Plan] retriever  : {args.retriever} on CPU (no GPU on this machine; "
                      f"slow for a dense encoder, one copy in RAM per worker)")
            if _agent_backend == "api":
                print(f"[GPU Plan] agent      : {args.policy_model_cli} → "
                      f"OpenRouter '{_agent_slug}' (0 local GPU)")
            else:
                print(f"[GPU Plan] agent      : {args.policy_model_cli} → "
                      f"local vLLM (reserves leftmost GPUs)")

            # ── Auto-start vLLM servers if needed ─────────────────────────────────
            # The manager allocates GPUs: vLLM servers get the leftmost GPUs, the
            # remaining ones (or those of --gpu-ids) go to pipeline workers.
            worker_gpu_ids = vllm_manager.auto_start(args, total_gpus=total_gpus_on_machine)
            if worker_gpu_ids is not None:
                gpu_ids = worker_gpu_ids
                workers = max(1, len(worker_gpu_ids))
            elif gpu_ids is None and workers > 1 and total_gpus_on_machine:
                gpu_ids = list(range(total_gpus_on_machine))
            # One worker per GPU on a GPU machine; CPU workers otherwise.
            if gpu_ids and workers > len(gpu_ids):
                print(f"[Workers] {workers} workers asked, {len(gpu_ids)} GPUs: running {len(gpu_ids)} (one per GPU)")
                workers = len(gpu_ids)
            # Never more workers than queries.
            workers = min(workers, len(plan.queries))
            if gpu_ids is not None:
                gpu_ids = gpu_ids[:max(workers, 1)]
            args.workers = workers
            print(f"[Workers] {workers} query worker(s) on "
                  f"{f'GPUs {gpu_ids}' if gpu_ids else ('the CPU' if not total_gpus_on_machine else 'the visible GPUs')}")

            # ── Restrict main process to pipeline GPUs before any CUDA init ───────
            # When --gpu-ids is specified the main process must not touch vLLM's
            # GPUs (typically 0..3).  Setting CUDA_VISIBLE_DEVICES early prevents
            # accidental CUDA context creation on those devices.
            if gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

            if workers > 1:
                # Several workers: each builds its own LLM client, retriever and
                # rerankers on its assigned GPU (or the CPU).  Skip them in the
                # main process to avoid OOM on shared GPUs.
                pipeline_kwargs = _pipeline_kwargs()
            else:
                pipeline_kwargs = _pipeline_kwargs(setup_llm(args, workers), setup_retriever_from_args(args))
                # ── Optional post-retrieval & post-fusion rerankers ─────────────────
                from utils.config import get_reranker_configs
                from searcher_component.rerankers import build_reranker_from_config
                _reranker_configs = get_reranker_configs(args.rerank_top_k)
                if args.post_retrieval_reranker != "null":
                    pipeline_kwargs["post_retrieval_reranker"] = build_reranker_from_config(
                        args.post_retrieval_reranker, _reranker_configs,
                    )
                if args.post_fusion_reranker != "null":
                    pipeline_kwargs["post_fusion_reranker"] = build_reranker_from_config(
                        args.post_fusion_reranker, _reranker_configs,
                    )

        run_pipeline(
            plan,
            data_path=args.data_path,
            policy_model=args.policy_model,
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

# ============================================================================
# CLI arguments
# ============================================================================
def _none_if_null(value: str) -> Optional[str]:
    """Read the literal ``null`` on the CLI as None (auto-select), like the YAML."""
    return None if value == "null" else value


def parse_args():
    """Build the CLI argument parser and parse the command line.

    Only the frequently-varied knobs are declared as CLI arguments.  The
    mostly-fixed variables live in a YAML config file (``--config``, default
    experiments/configs/dra_inference.yaml) and are merged onto ``args`` in
    ``_resolve_args``; any of them can still be overridden by passing the
    matching ``--flag`` on the command line (returned in ``extras``).
    """
    parser = argparse.ArgumentParser(
        description="Run deep research agents pipeline (unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,  # only full flag names, e.g. --criteria-source
    )

    # ── Stage ───────────────────────────────────────────────────────────────
    parser.add_argument("--stage", type=str, default="all", choices=list(STAGES), help="What to run. 'generate': the agent on the queries without a finished trajectory (trajectory/, retrieval/, generation/), then the evaluation. 'belief': the belief model over the finished trajectories without a belief file (belief/{qid}.jsonl and .md; no agent and no retrieval: the documents' text is read from the corpus by id), then the evaluation; belief settings (--belief-model, --criteria-source) other than the run's replace its beliefs. 'eval': the evaluation only, from the saved files (every evaluator of the dataset; judge verdicts reused from eval/generation/ and eval/belief/, so an unchanged run makes no LLM call). 'all': generate, then belief, then the evaluation. Every stage writes the same eval/metrics.md and eval/per_query.csv.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="Path to the YAML file holding the mostly-fixed pipeline variables. Any value in it can be overridden by passing the matching --flag on the CLI.")
    # ── Generation (the belief and eval stages need the same values to find the run directory) ──
    parser.add_argument("--dataset", type=str, default="browsecomp_plus", choices=list(DATASETS), help="Dataset; all use local indices.")
    parser.add_argument("--subset", type=_none_if_null, default=None, help="Dataset subset/collection (unset or null = the dataset's default in layout.DATASET_SPECS). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical; browsecomp_plus: test; ragtime: unused.")
    parser.add_argument("--policy-model", type=str, default="oss_20b", choices=list(POLICY_MODEL_TO_LLM), help="Agent to run; the LLM is selected automatically from the agent. cpm_report = Writing-as-Reasoning (report generation); searchr1/research/stepsearch/react/selfask/searcho1 = Reasoning-augmented retrieval; glm/oss_20b/oss_120b/tongyi = vendor-specific ReAct agents.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever; its index must be built for --dataset.")
    parser.add_argument("--workers", type=int, default=1, help="Query-level worker processes of the generate stage; each loads its own retriever and agent. On a GPU machine one worker per GPU (capped at the GPUs, or at --gpu-ids); on a CPU machine CPU processes (the retriever on CPU, one copy in RAM each). 0 = auto: the GPU count, 1 without GPU. Ignored by the belief and eval stages (no GPU; the belief stage runs --belief-workers threads).")
    # ── Belief ──────────────────────────────────────────────────────────────
    parser.add_argument("--belief-model", type=str, default=None, help="Backbone LLM of the belief model, for its three calls: criteria init (banked under this model), belief update and final score. A vllm/<name> model is served at $BELIEF_VLLM_API_BASE (default $VLLM_API_BASE). Unset = belief_model in --config.")
    parser.add_argument("--criteria-source", type=str, default=None, choices=list(CRITERIA_SOURCES), help="Criteria of the belief model: gold (TRQA: made from the intermediate info; trqa_validation: the dataset rows) or silver (extracted by --belief-model). A choice only on TRQA and trqa_validation; elsewhere the dataset's own source is used. Unset = criteria_source in --config.")
    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="Cap the queries each stage runs (for quick tests): the generate stage's queries left to generate, the belief stage's trajectories left without a belief file")
    parser.add_argument("--quiet", type=_sm_bool, nargs="?", const=True, default=False, help="Print minimal logs (overrides verbose)")

    args, extras = parser.parse_known_args()
    return args, extras


if __name__ == "__main__":
    main()


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
#   {output}/{dataset}_{split}[_{query_key}]_{retriever}/{agent}_{backend}_{model}/
#   ({output} = DRA_OUTPUT_ROOT; _{query_key} only when not the dataset's default)
#   e.g. run_outputs/neuclir_2024_news_qwen3_emb_4b/oss_20b_vllm_gpt-oss-20b/
#   ({agent} = the --policy-model name)
#
#   One home per fact: each file below holds its facts only.
#     ├── run_config.json       every resolved setting: dataset, agent, searcher (generate
#     │                         stage; a resume with other settings is refused), belief_model
#     │                         (belief stage; other settings replace the beliefs), evaluation
#     ├── retrieval/
#     │   ├── surfaced/{qid}.trec   surfaced-doc TREC (raw retriever output, col 6 = iter_N)
#     │   ├── seen/{qid}.trec       seen-doc TREC (docs shown to the LLM)
#     │   └── cited/{qid}.trec      cited-doc TREC (docs cited by the LLM)
#     ├── generation/{qid}.md   the final generation (with its References section)
#     ├── trajectory/
#     │   ├── {qid}.jsonl       one line per step (queries, seen doc ids, search_step of a
#     │   │                     search iteration), then the meta line (counts, agent data,
#     │   │                     agent_generation); written last: the meta line marks the
#     │   │                     query as generated
#     │   └── {qid}.md          the same trajectory, human-readable, written live
#     ├── belief/               belief stage (orchestration.belief_stage)
#     │   ├── {qid}.jsonl       meta line + one line per search iteration + final line
#     │   │                     (run_files.belief); written last: marks the query as done
#     │   └── {qid}.md          the same belief, human-readable
#     └── eval/                 the evaluation (every stage, evaluation.runner)
#         ├── metrics.md        one table: group | metric | value | ci_low | ci_high | n
#         │                     (retrieval, generation, uncertainty, belief, trajectory)
#         ├── per_query.csv     every per-query value of every group
#         ├── generation/       accuracy.jsonl: per-query answer correctness (datasets with
#         │                     answers); or Auto-ARGUE inputs, cached judgments and
#         │                     per-query scores.tsv (report datasets)
#         └── belief/           belief update evaluation: events.csv, judge_cache.jsonl
#
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#   python experiments/inference/dra_inference.py --stage generate --dataset browsecomp_plus --limit 1 --workers 1
#   python experiments/inference/dra_inference.py --stage generate --dataset trqa --subset wiki2 --limit 1 --workers 1
#   python experiments/inference/dra_inference.py --stage belief --dataset browsecomp_plus --limit 1
#   python experiments/inference/dra_inference.py --stage belief --dataset trqa --subset wiki2 --limit 1

#   python experiments/inference/dra_inference.py --stage generate --dataset browsecomp_plus --limit 2 --workers 2
#   python experiments/inference/dra_inference.py --stage generate --dataset trqa --subset wiki2 --limit 1 --workers 1
#   python experiments/inference/dra_inference.py --stage belief --dataset browsecomp_plus --limit 2 --workers 2
#   python experiments/inference/dra_inference.py --stage belief --dataset trqa --subset wiki2 --limit 1
