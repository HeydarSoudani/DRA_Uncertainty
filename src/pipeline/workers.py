"""The generate stage's per-query run and its GPU workers.

:func:`run_query` is the one per-query path of the sequential run and of
the GPU workers.  The worker functions (:func:`_init_worker`,
:func:`gpu_worker`) must stay at module level to be picklable by
multiprocessing.spawn.
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

from reasoner_component import setup_llm
from utils.run_files.trajectory_logger import TrajectoryLogger
from utils.citations import ranked_cited_docs, with_references_section

from .factories import build_agent, build_rerankers, build_search_tool, setup_retriever_from_args
from .logging_setup import quiet_logging

logger = logging.getLogger(__name__)


def run_query(agent, run_dir, query_id: str, query_text: str, *, policy_model: str, llm_model: str,
              temperature: float, answer: Optional[str] = None, status_callback=None) -> Optional[dict]:
    """Run one query and save its files under *run_dir*; the one per-query
    path of the sequential run and of the GPU workers.  Returns the result,
    or None when the query failed: the error is logged and the caller goes
    on with its next query.

    The trajectory is streamed to ``trajectory/{qid}.jsonl`` and ``.md`` as
    it happens, so an interrupted query still leaves its work on disk; the
    result gets its cited-doc list and a References section, then
    ``utils.run_files.save_query_outputs`` writes every file.  A query
    that fails before the trajectory meta line is written stays unfinished
    and is run again on resume.
    """
    from utils.run_files import save_query_outputs

    try:
        traj_logger = TrajectoryLogger(
            Path(run_dir) / "trajectory", query_id, query_text,
            agent_name=policy_model, model=llm_model, answer=answer,
        )
        result = agent.run_single(
            query_id=query_id,
            query_text=query_text,
            temperature=temperature,
            status_callback=status_callback,
            trajectory_logger=traj_logger,
        )
        if result is None:
            return None
        result["cited_docs_ranked_list"] = ranked_cited_docs(result)
        # The generation as the agent returned it, before the References
        # section (which also renumbers docid citations): the answer the
        # belief stage reads (trajectory meta line).
        result["agent_generation"] = result["generation"]
        # Append a References section to the generation (CPMReport already has one)
        result["generation"] = with_references_section(result)
        save_query_outputs(run_dir, query_id, query_text, result)
        return result
    except Exception as exc:
        logger.error("Query %s failed: %s", query_id, exc)
        traceback.print_exc()
        return None


# ===========================================================================
# Multi-GPU workers (must remain top-level for multiprocessing.spawn pickling)
# ===========================================================================

def _init_worker(worker_id: int, worker_config: dict):
    """Initialise a worker: pin its GPU (none: a CPU worker), load models,
    build the agent; returns the agent.

    The worker config carries the argument names of the main process's
    ``setup_llm`` and ``setup_retriever_from_args``, so the LLM client and
    the retriever are rebuilt with them.  Its ``gpu_ids`` are real GPU ids
    (``pipeline.resources.plan_generate_workers``).
    """
    gpu_ids = worker_config["gpu_ids"]
    if worker_config["quiet"]:
        quiet_logging()
    if gpu_ids and worker_id < len(gpu_ids):
        gpu = str(gpu_ids[worker_id])
        # Set before CUDA starts in this process: its one visible GPU is device 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES={gpu}", flush=True)
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"[Worker {worker_id}] torch.cuda.current_device()={torch.cuda.current_device()}, "
                  f"device_count={torch.cuda.device_count()}", flush=True)
    else:
        print(f"[Worker {worker_id}] No GPU assigned: retriever on the CPU", flush=True)

    print(f"[Worker {worker_id}] Building components (LLM, retriever)...", flush=True)
    args = SimpleNamespace(**worker_config)
    llm_client, retriever = setup_llm(args), setup_retriever_from_args(args)
    print(f"[Worker {worker_id}] Components built", flush=True)

    policy_model = worker_config["policy_model"]

    post_retrieval_reranker, post_fusion_reranker = build_rerankers(
        worker_config["post_retrieval_reranker_type"], worker_config["post_fusion_reranker_type"],
        worker_config["rerank_top_k"],
    )
    search_tool = build_search_tool(
        retriever,
        post_retrieval_reranker=post_retrieval_reranker,
        post_fusion_reranker=post_fusion_reranker,
        rerank_top_k=worker_config["rerank_top_k"],
        retrieval_input=worker_config["retrieval_input"],
        post_fusion_reranker_input=worker_config["post_fusion_reranker_input"],
    )

    print(f"[Worker {worker_id}] Creating agent ({policy_model})...", flush=True)
    agent = build_agent(
        policy_model=policy_model,
        policy_model_cli=worker_config["policy_model_cli"],
        dataset=worker_config["dataset"],
        llm_model=worker_config["llm_model"],
        llm_client=llm_client,
        retriever=retriever,
        max_iteration=worker_config["max_iteration"],
        seen_top_k=worker_config["seen_top_k"],
        verbose=worker_config["verbose"],
        search_tool=search_tool,
        use_plan=worker_config["use_plan"],
        max_output_tokens_total=worker_config["max_output_tokens_total"],
        temperature=worker_config["llm_temperature"],
        max_extend_steps=worker_config["max_extend_steps"],
        max_retries=worker_config["max_retries"],
        hard_mode=worker_config["hard_mode"],
        max_passage_tokens=worker_config["max_passage_tokens"],
    )

    print(f"[Worker {worker_id}] Initialisation complete", flush=True)
    return agent


def gpu_worker(worker_id: int, query_items: list, run_dir: str, worker_config: dict, progress_queue=None) -> List[str]:
    """Worker process: pin to one GPU, rebuild the agent, process queries and
    save their files straight to *run_dir* (its folders made by the parent,
    ``pipeline.generate_stage``).

    Progress goes to *progress_queue* as ``(worker_id, qid, kind, info)``
    (kind: processing, update, done, skipped), then None when it ends.
    Returns the ids of the queries it finished; the parent reads the
    results back from their files.
    """
    # With live progress bars in the parent, this process's init prints would
    # land between the bars' in-place redraws and scramble them.  Hold them
    # back; on failure they go to stderr.
    live_bars = bool(worker_config.get("live_bars", False))
    init_out = io.StringIO()
    try:
        with (contextlib.redirect_stdout(init_out) if live_bars else contextlib.nullcontext()):
            agent = _init_worker(worker_id, worker_config)
    except Exception as exc:
        if live_bars:
            sys.stderr.write(init_out.getvalue())
        print(f"[Worker {worker_id}] INIT FAILED: {exc}", file=sys.stderr if live_bars else sys.stdout, flush=True)
        traceback.print_exc()
        if progress_queue is not None:
            progress_queue.put(None)
        return []

    def report(qid: str, kind: str, info=None) -> None:
        if progress_queue is not None:
            progress_queue.put((worker_id, qid, kind, info))

    verbose = worker_config["verbose"]
    temperature = worker_config["llm_temperature"]
    max_iteration = worker_config["max_iteration"]
    answers = worker_config.get("answers") or {}
    finished: List[str] = []
    current = [None]   # the query of the status callback

    def status_cb(stage: str, iteration: int):
        report(current[0], "update", (stage, iteration, max_iteration))

    for idx, (query_id, query_text) in enumerate(query_items, 1):
        current[0] = query_id
        report(query_id, "processing")
        if verbose:
            print(
                f"  [Worker {worker_id}] [{idx}/{len(query_items)}] Processing query: {query_id}\n    Query text: {query_text}",
                flush=True,
            )
        result = run_query(
            agent, run_dir, query_id, query_text,
            policy_model=worker_config["policy_model"], llm_model=worker_config["llm_model"],
            answer=answers.get(query_id), temperature=temperature,
            status_callback=status_cb if progress_queue is not None else None,
        )
        if result is None:
            if verbose:
                print(f"  [Worker {worker_id}] ✗ Skipping {query_id}", flush=True)
            report(query_id, "skipped")
            continue
        finished.append(query_id)
        report(query_id, "done", f"{result.get('num_iterations', '?')}/{max_iteration}")
        if verbose:
            print(f"  [Worker {worker_id}] ✓ Saved: {query_id}", flush=True)

    agent.cleanup()
    if verbose:
        print(
            f"[Worker {worker_id}] Completed {len(finished)}/{len(query_items)} queries",
            flush=True,
        )
    if progress_queue is not None:
        progress_queue.put(None)
    return finished
