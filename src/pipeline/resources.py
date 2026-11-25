"""The generate stage's GPUs: the plan of its workers and GPUs (vLLM
servers first), and their release at the end of the stage.

GPUs are named by their real ids (``CUDA_VISIBLE_DEVICES`` / ``nvidia-smi
-i`` values; ``--gpu-ids`` too), never by their position among the visible
ones: with ``CUDA_VISIBLE_DEVICES=2,3`` the first GPU is ``"2"``.

The belief and eval stages use no GPU (they run on a CPU machine), so
nothing here is called for them.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from utils.cli_setup import count_gpus, resolve_workers


@dataclass
class WorkerPlan:
    """The generate stage's workers and GPUs (:func:`plan_generate_workers`)."""
    workers: int
    gpu_ids: Optional[List[str]]    # the workers' GPU ids; None: the visible GPUs; []: the CPU
    total_gpus: int                 # the GPUs visible to this process before the plan
    visible_before: Optional[str]   # CUDA_VISIBLE_DEVICES before the plan (None: unset)


def _parse_gpu_ids(value) -> Optional[List[str]]:
    """``--gpu-ids`` (``"4,5"``, ``4`` or a YAML list) as GPU ids; None when unset."""
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return [str(g).strip() for g in items if str(g).strip()]


def plan_generate_workers(args, vllm_manager, n_queries: int) -> WorkerPlan:
    """Plan the generate stage's workers and GPUs and start its vLLM servers.

    vLLM servers get the leftmost visible GPUs (``vllm_manager``).  The
    workers get the GPUs of ``--gpu-ids`` (which must not be a server's),
    else the GPUs the servers left: one worker per GPU on a GPU machine,
    CPU workers otherwise (also when the servers took every GPU), never
    more workers than queries.  Restricts this process to the workers' GPUs
    (``CUDA_VISIBLE_DEVICES``; :func:`release_gpus` restores it).
    """
    from reasoner_component import run_servers
    from reasoner_component.vllm_server.process import gpu_ids as ids_of_positions
    from utils.config import RETRIEVER_VRAM_GB, resolve_agent_backend

    visible_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    total_gpus = count_gpus()
    requested = _parse_gpu_ids(args.gpu_ids)
    workers = resolve_workers(args.workers, total_gpus, requested)

    # ── GPU plan: retriever first (fixed footprint), then agent ──────────
    agent_backend, agent_slug = resolve_agent_backend(args.policy_model_cli)
    print(f"[GPU Plan] total GPUs : {total_gpus}")
    if total_gpus:
        print(f"[GPU Plan] retriever  : {args.retriever} "
              f"(~{RETRIEVER_VRAM_GB} GB fp16, FAISS on CPU — shares a worker GPU)")
    else:
        print(f"[GPU Plan] retriever  : {args.retriever} on CPU (no GPU on this machine; "
              f"slow for a dense encoder, one copy in RAM per worker)")
    if agent_backend == "api":
        print(f"[GPU Plan] agent      : {args.policy_model_cli} → "
              f"OpenRouter '{agent_slug}' (0 local GPU)")
    else:
        print(f"[GPU Plan] agent      : {args.policy_model_cli} → "
              f"local vLLM (reserves leftmost GPUs)")

    # ── Auto-start vLLM servers if needed ─────────────────────────────────
    # The manager returns the GPU positions it left free; as ids below.
    free_positions = vllm_manager.start(run_servers(args), total_gpus=total_gpus)
    gpu_ids: Optional[List[str]] = requested
    if free_positions is not None:
        free = ids_of_positions(free_positions)
        if requested is not None:
            taken = [g for g in requested if g not in free]
            if taken:
                raise SystemExit(f"--gpu-ids {','.join(requested)}: GPU(s) {','.join(taken)} are used by "
                                 f"the vLLM servers; the free GPUs are {free or 'none'}")
        else:
            gpu_ids = free
            workers = max(1, len(free))
    elif gpu_ids is None and workers > 1 and total_gpus:
        gpu_ids = ids_of_positions(range(total_gpus))
    # One worker per GPU on a GPU machine; CPU workers otherwise.
    if gpu_ids and workers > len(gpu_ids):
        print(f"[Workers] {workers} workers asked, {len(gpu_ids)} GPUs: running {len(gpu_ids)} (one per GPU)")
        workers = len(gpu_ids)
    # Never more workers than queries.
    workers = min(workers, n_queries)
    if gpu_ids:
        gpu_ids = gpu_ids[:workers]
    print(f"[Workers] {workers} query worker(s) on "
          f"{f'GPUs {gpu_ids}' if gpu_ids else ('the CPU' if not total_gpus or gpu_ids == [] else 'the visible GPUs')}")

    # ── Restrict this process (and the workers it spawns) to their GPUs ───
    # The vLLM servers already run with their own CUDA_VISIBLE_DEVICES.
    # With every GPU taken by the servers, none: the retriever on the CPU.
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    elif gpu_ids == [] and total_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return WorkerPlan(workers=workers, gpu_ids=gpu_ids, total_gpus=total_gpus, visible_before=visible_before)


def offload_encoder(retriever) -> None:
    """Move the retriever's encoder model off the GPU and drop it.

    Accelerate's device_map hooks can prevent gc from freeing GPU tensors
    even after all Python refs are gone; ``.cpu()`` forces the move and
    ``remove_hook_from_submodules`` detaches dispatch hooks.
    """
    _enc = getattr(retriever, "encoder", None) if retriever is not None else None
    if _enc is not None and hasattr(_enc, "model"):
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(_enc.model)
        except Exception:
            pass
        _enc.model.cpu()
        del _enc.model


def release_gpus(vllm_manager, total_gpus_on_machine: int, visible_before: Optional[str]) -> None:
    """Return the GPU memory to CUDA, restore ``CUDA_VISIBLE_DEVICES`` as it
    was before :func:`plan_generate_workers`, then stop the vLLM servers and
    check every GPU of the run is free (the next stage, e.g. a vLLM belief
    model, sees the GPUs as the run started with them).

    The caller drops every reference to a GPU-resident object (agent,
    search tool, retriever, rerankers) first, so gc can collect them.
    """
    import gc
    gc.collect()
    if visible_before is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_before
    if total_gpus_on_machine == 0:
        # CPU machine: no GPU memory to return, no vLLM server was started.
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Kill vLLM servers, orphan processes, and verify GPUs are free
    if vllm_manager is not None:
        vllm_manager.release_gpus(total_gpus_on_machine)
