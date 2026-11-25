"""The generate stage: the agent on the plan's queries, each query's files
saved under the run directory, then every GPU of the run released.

One worker runs the queries in this process (:func:`_run_sequential`);
several run them in spawned worker processes, one per GPU (or CPU
processes), queries split round-robin (:func:`_run_parallel`).
"""

import concurrent.futures
import multiprocessing
import threading
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, List, Optional, Tuple

from utils.io_utils import setup_output_dirs
from utils.progress import WorkerBars
from utils.run_files import QUERY_OUTPUT_DIRS, scan_queries

from .factories import build_agent, build_search_tool
from .resources import offload_encoder, release_gpus
from .run_plan import RunPlan
from .workers import gpu_worker, run_query


def run_generate_stage(plan: RunPlan, policy_model: str, llm_model: str, verbose: bool, workers: int,
                       worker_config: Optional[dict], kwargs: dict) -> None:
    """Run the agent on ``plan.queries`` and save each query's files under
    ``plan.run_dir``, then release every GPU the run held.

    ``worker_config`` (``utils.cli_setup.assemble_pipeline_kwargs``) is
    required with several workers: each rebuilds its agent from it."""
    if workers > 1 and worker_config is None:
        raise ValueError(f"{workers} workers need a worker_config to rebuild the agent in each worker")
    run_dir = plan.run_dir

    # ==================== Build search tool ====================
    retriever = kwargs.get("retriever")
    search_tool = build_search_tool(
        retriever,
        post_retrieval_reranker=kwargs.get("post_retrieval_reranker"),
        post_fusion_reranker=kwargs.get("post_fusion_reranker"),
        rerank_top_k=kwargs["rerank_top_k"],
        retrieval_input=kwargs["retrieval_input"],
        post_fusion_reranker_input=kwargs["post_fusion_reranker_input"],
    )

    # ==================== Instantiate Agent ====================
    # With several workers the agent is not built in the main process; each
    # worker spawns its own instance on its assigned GPU or the CPU (see
    # pipeline.workers._init_worker).  Building it here would create unused
    # LLM clients that are immediately torn down.
    agent = None
    if workers <= 1:
        agent = build_agent(
            policy_model=policy_model,
            policy_model_cli=kwargs["policy_model_cli"],
            dataset=plan.dataset,
            llm_model=llm_model,
            llm_client=kwargs.get("llm_client"),
            retriever=retriever,
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
    query_items = list(plan.queries.items())
    if workers > 1:
        n_done = _run_parallel(query_items, run_dir, workers, worker_config, plan.answers)
        print(f"\nCompleted {n_done} queries (parallel)")
    else:
        n_done = _run_sequential(agent, query_items, plan, policy_model, llm_model, verbose, kwargs)
        agent.cleanup()
        print(f"\nCompleted {n_done} queries")

    # ==================== Release the GPUs ====================
    # Break every reference chain to GPU-resident objects so gc can
    # collect them before the GPUs are reclaimed.  Reference chains that
    # keep the encoder model alive:
    #   1. agent.search_tool.retriever → retriever.encoder
    #   2. agent.retriever → retriever.encoder
    #   3. kwargs["retriever"] → retriever.encoder
    # 1) Move the encoder model off GPU *before* dropping references.
    offload_encoder(retriever)
    # 2) Drop all local + kwargs references
    del agent, search_tool, retriever
    kwargs.pop("retriever", None)
    kwargs.pop("post_retrieval_reranker", None)
    kwargs.pop("post_fusion_reranker", None)
    # 3) Collect, return GPU memory to CUDA, stop the vLLM servers.
    release_gpus(kwargs.get("vllm_manager"), kwargs["total_gpus_on_machine"],
                 kwargs.get("cuda_visible_devices_before"))


def _run_sequential(agent, query_items: List[Tuple[str, str]], plan: RunPlan, policy_model: str, llm_model: str,
                    verbose: bool, kwargs: dict) -> int:
    """The queries in this process, one after the other; returns the number
    finished.  Not verbose: the same bar as one parallel worker.  Verbose:
    the agent's trace, each query under a header line."""
    run_dir, answers = plan.run_dir, plan.answers
    max_iteration = kwargs["max_iteration"]
    n_done = 0
    bars = None if verbose else WorkerBars([len(query_items)], "Worker")
    for idx, (query_id, query_text) in enumerate(query_items, 1):
        status_cb = None
        if bars is None:
            answer_line = f"\n  Answer: {answers[query_id]}" if answers.get(query_id) else ""
            print(f"\n[{idx}/{len(query_items)}] Processing query: {query_id}\n  Query text: {query_text}{answer_line}")
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
    return n_done


def _run_parallel(query_items: List[Tuple[str, str]], run_dir: str, workers: int, worker_config: dict,
                  answers: Dict[str, str]) -> int:
    """The queries in *workers* spawned processes (``gpu_worker``), split
    round-robin so load is balanced even when queries vary in difficulty;
    never more workers than queries.  Each worker gets the gold answers of
    its own queries (the header of each trajectory .md).  Returns the
    number finished."""
    num_workers = min(workers, len(query_items))
    chunks = [query_items[i::num_workers] for i in range(num_workers)]

    # One bar per worker (utils.progress.WorkerBars): live on a terminal,
    # one line per finished query off it (an sbatch .out file).
    bars = WorkerBars([len(c) for c in chunks], "Worker")
    configs = [
        {**worker_config, "live_bars": bars.live,
         "answers": {qid: answers[qid] for qid, _ in chunk if qid in answers}}
        for chunk in chunks
    ]

    print(f"\nParallel mode: {num_workers} workers")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: {len(chunk)} queries")

    mp_ctx = multiprocessing.get_context("spawn")
    manager = mp_ctx.Manager()
    progress_queue = manager.Queue()

    # Progress messages of gpu_worker: (worker_id, qid, kind, info), then
    # None when the worker ends.
    def drain_progress(stop_event):
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
            w_id, qid, kind, info = item
            if kind == "processing":
                bars.status(w_id, f"{qid}, starting")
            elif kind == "update":
                stage_name, iteration, max_iter_val = info
                bars.status(w_id, f"{qid}, turn {iteration}/{max_iter_val}, {stage_name}")
            else:
                bars.done(w_id, qid, f"turns {info}" if info else kind)

    drain_stop = threading.Event()
    drain_thread = threading.Thread(target=drain_progress, args=(drain_stop,), daemon=True)
    drain_thread.start()

    n_done = 0
    broken: List[int] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=mp_ctx
    ) as executor:
        futures = {
            executor.submit(gpu_worker, i, chunks[i], run_dir, configs[i], progress_queue): i
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
            except BrokenProcessPool:
                # A worker process died (e.g. killed out of memory): the
                # pool stops every worker.  Their finished queries are on
                # disk, counted below.
                broken.append(worker_id)
            except Exception as e:
                bars.write(f"  Worker {worker_id} failed: {e}")

    if broken:
        finished = scan_queries(run_dir)[0]
        for worker_id in sorted(broken):
            k = sum(qid in finished for qid, _ in chunks[worker_id])
            n_done += k
            bars.write(f"[Worker {worker_id}] process died (the worker pool broke, e.g. out of memory): "
                       f"{k}/{len(chunks[worker_id])} queries finished; the other "
                       f"{len(chunks[worker_id]) - k} run again on resume")

    drain_stop.set()
    drain_thread.join(timeout=3)
    manager.shutdown()
    bars.close()
    return n_done
