"""Running vLLM servers: the processes, the GPUs they get, their cleanup.

``VLLMServer``         one ``vllm`` process for a ``ServerSpec``
``VLLMServerManager``  the servers of a pipeline run: GPU split, start, stop,
                       GPU release for the next stage
``serve``              one server for the length of a ``with`` block

TP is fitted to the GPU memory (``fit_tp``); the command line and environment
of a server come from ``server_command`` / ``server_env``.

GPU numbering: the pipeline counts GPUs 0..n-1 as this process sees them (as
torch does); ``CUDA_VISIBLE_DEVICES`` of a server and ``nvidia-smi -i`` take
the ids those positions stand for (``gpu_ids``).
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .registry import DOWNLOAD_DIR, HF_HOME, LOG_DIR, MODELS_DIR, ServerSpec, VLLMServerError, cleanup_ports

# Headroom on top of the weights for the KV cache, activations and CUDA graphs.
_TP_MEM_SAFETY = 1.2


def visible_gpus() -> List[str]:
    """The GPU ids this process may use: ``$CUDA_VISIBLE_DEVICES`` when it is
    set, else every GPU ``nvidia-smi -L`` lists (none when it fails)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [str(i) for i in range(sum(line.startswith("GPU ") for line in out.splitlines()))]


def gpu_ids(positions: Iterable[int]) -> List[str]:
    """The GPU ids (``CUDA_VISIBLE_DEVICES`` / ``nvidia-smi -i`` values) of
    the GPU *positions* this process counts from 0: with
    ``CUDA_VISIBLE_DEVICES=2,3`` position 0 is GPU ``"2"``.  A position past
    the visible GPUs (or none visible) stands for itself."""
    visible = visible_gpus()
    return [visible[p] if p < len(visible) else str(p) for p in positions]


def per_gpu_gb() -> Optional[float]:
    """The memory of the first visible GPU in GB, None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                              "-i", gpu_ids([0])[0]],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().split("\n")[0]) / 1024
    except Exception:  # noqa: BLE001 - no GPU, no driver, unexpected output
        return None


def fit_tp(spec: ServerSpec, gpu_gb: Optional[float], *, max_tp: int) -> int:
    """The smallest TP in ``spec.allowed_tp`` (at most *max_tp*) whose GPUs
    hold the weights with headroom.  Without ``spec.model_gb`` or *gpu_gb*,
    the allowed TP nearest ``spec.tp_size``."""
    allowed = sorted(t for t in spec.allowed_tp if t >= 1)
    capped = [t for t in allowed if t <= max_tp]

    if spec.model_gb is None or gpu_gb is None:
        tp = min(capped, key=lambda t: abs(t - spec.tp_size)) if capped else spec.tp_size
        reason = "static default" if spec.model_gb is None else "GPU memory unknown"
        print(f"[vLLM] {spec.label}: TP={tp} ({reason})")
        return tp

    usable = gpu_gb * spec.gpu_memory_utilization
    need = spec.model_gb * _TP_MEM_SAFETY
    tp = next((t for t in capped if t * usable >= need), None)
    if tp is None:
        tp = capped[-1] if capped else allowed[0]
        print(f"[vLLM] {spec.label}: needs ~{need:.0f} GB, a GPU has {usable:.0f} GB usable: TP={tp} (may not fit)")
    else:
        print(f"[vLLM] {spec.label}: ~{spec.model_gb:.0f} GB weights, {usable:.0f} GB usable per GPU: TP={tp}")
    return tp


def _local_weights(model: str) -> str:
    """The pre-staged weights directory of *model* when it has one, else *model*."""
    local = MODELS_DIR / model.replace("/", "--")
    if local.is_dir() and any(local.glob("*.safetensors")):
        return str(local)
    return model


def server_command(spec: ServerSpec, tp_size: int) -> List[str]:
    """The command line of *spec*'s server at TP *tp_size*."""
    weights = _local_weights(spec.model)
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", weights, "--port", str(spec.port)]
    if weights != spec.model:
        # Clients ask for the HF name, not the staged path.
        cmd += ["--served-model-name", spec.model]
    cmd += [
        "--tensor-parallel-size", str(tp_size),
        "--download-dir", DOWNLOAD_DIR,
        "--trust-remote-code",
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
    ]
    if spec.max_model_len is not None:
        cmd += ["--max-model-len", str(spec.max_model_len)]
    if spec.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(spec.max_num_seqs)]
    if spec.enforce_eager:
        cmd.append("--enforce-eager")
    if spec.dtype:
        cmd += ["--dtype", spec.dtype]
    if spec.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if spec.lora:
        cmd += ["--enable-lora", "--lora-modules", "=".join(spec.lora)]
        if spec.max_lora_rank is not None:
            cmd += ["--max-lora-rank", str(spec.max_lora_rank)]
    if spec.tool_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", spec.tool_parser]
    cmd += spec.extra_args
    return cmd


def server_env(gpus: Sequence[str]) -> Dict[str, str]:
    """The environment of a server on *gpus*."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["HF_HOME"] = HF_HOME
    # The V1 engine needs /dev/shm for multiprocessing; containers with a few
    # GB of it crash silently.  vLLM 0.11 dropped the V0 engine, so there is
    # nothing to fall back to: only warn.
    try:
        shm = os.statvfs("/dev/shm")
        small_shm = shm.f_frsize * shm.f_blocks / 1024 ** 3 < 8
    except OSError:
        small_shm = False
    if small_shm:
        if _has_v0_engine():
            env.setdefault("VLLM_USE_V1", "0")
            print("[vLLM] /dev/shm is under 8 GB: V1 engine off")
        else:
            print("[vLLM] WARNING: /dev/shm is under 8 GB; the vLLM engine may crash (enlarge it)")
    return env


def _has_v0_engine() -> bool:
    """True when the installed vLLM still has the V0 engine (before 0.11)."""
    try:
        from packaging.version import InvalidVersion, Version
        return Version(vllm_version()) < Version("0.11")
    except (ImportError, InvalidVersion):
        return False


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def served_models(api_base: str) -> List[Dict[str, Any]]:
    """The ``/models`` entries of the server at *api_base*; empty when nothing
    answers there as one."""
    try:
        with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=5) as reply:
            return json.load(reply).get("data") or []
    except Exception:  # noqa: BLE001 - not a vLLM server
        return []


class VLLMServer:
    """One vLLM server process: *spec* on *gpus* (TP = their number).

    ``start`` reuses a server already on the spec's port when it serves the
    model, and raises when the port is taken by anything else.  ``stop``
    leaves a reused server running.
    """

    def __init__(self, spec: ServerSpec, gpus: Sequence[int | str]):
        self.spec = spec
        self.gpus = [str(g) for g in gpus]
        self._proc: Optional[subprocess.Popen] = None
        self._log = None

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"vllm_{self.spec.port}.log"

    def start(self) -> None:
        spec = self.spec
        if port_in_use(spec.port):
            names = [m.get("id") for m in served_models(spec.api_base)]
            if spec.name not in names:
                raise VLLMServerError(f"port {spec.port} for {spec.label} is taken by another process "
                                      f"(serving {names or 'no vLLM model'}); free it first")
            print(f"[vLLM] {spec.label} already served on port {spec.port}: using that server")
            return
        if not self.gpus:
            raise VLLMServerError(f"{spec.label} needs at least one GPU")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[vLLM] Launching {spec.label} on GPUs {self.gpus}, port {spec.port} (log: {self.log_path})")
        self._log = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            server_command(spec, len(self.gpus)), stdout=self._log, stderr=subprocess.STDOUT,
            env=server_env(self.gpus),
            start_new_session=True,  # own process group: stop() kills vLLM's workers too
        )

    def wait_healthy(self, timeout: float = 900) -> None:
        """Wait for ``/health``; raises when the process dies or *timeout* passes."""
        url = f"http://127.0.0.1:{self.spec.port}/health"
        start, interval = time.time(), 2.0
        while time.time() - start < timeout:
            if self._proc is not None and self._proc.poll() is not None:
                raise VLLMServerError(f"{self.spec.label} died (exit code {self._proc.returncode}); "
                                      f"end of {self.log_path}:\n{self._log_tail()}")
            try:
                with urllib.request.urlopen(url, timeout=5) as reply:
                    if reply.status == 200:
                        print(f"[vLLM] {self.spec.label} is healthy ({time.time() - start:.0f}s)")
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(interval)
            interval = min(interval * 1.2, 10.0)
        self.stop()
        raise VLLMServerError(f"{self.spec.label} not healthy after {timeout:.0f}s; see {self.log_path}")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            print(f"[vLLM] Stopping {self.spec.label} (PID {proc.pid})")
            _kill_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_group(proc, signal.SIGKILL)
                proc.wait(timeout=5)
        if self._log is not None:
            self._log.close()
            self._log = None

    def _log_tail(self, lines: int = 80) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"


def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, OSError):
        proc.send_signal(sig)


class VLLMServerManager:
    """The vLLM servers of one pipeline run: they get the leftmost GPUs, the
    pipeline workers the rest.

    Usage::

        manager = VLLMServerManager()
        worker_gpus = manager.start(specs, total_gpus=4)   # None: no server needed
        try:
            ...                                            # run the pipeline
        finally:
            manager.shutdown()
    """

    def __init__(self) -> None:
        self._servers: List[VLLMServer] = []
        self._hooked = False

    def start(self, specs: Iterable[ServerSpec], total_gpus: int, *,
              health_timeout: float = 900) -> Optional[List[int]]:
        """Fit each server's TP, start the servers, wait for them; return the
        GPU ids left for the pipeline workers (None when *specs* is empty)."""
        specs = list(specs)
        if not specs:
            print("[vLLM] No vLLM servers needed for this configuration.")
            return None

        # Keep a GPU for the workers when there are several; when the models
        # still take them all, the retriever falls back to CPU (warned below).
        gpu_gb = per_gpu_gb()
        sizes = [fit_tp(spec, gpu_gb, max_tp=max(1, total_gpus - 1)) for spec in specs]
        if sum(sizes) > total_gpus:
            wanted = "\n".join(f"  - {s.label}: TP={tp} on port {s.port}" for s, tp in zip(specs, sizes))
            raise VLLMServerError(
                f"the vLLM servers need {sum(sizes)} GPUs, {total_gpus} are available:\n{wanted}\n"
                "Use a node with larger GPUs, route the agent through OpenRouter (OPENROUTER_API_KEY), "
                "or drop the vLLM reranker (--post-retrieval-reranker null).")

        servers, offset = [], 0
        for spec, tp in zip(specs, sizes):
            servers.append(VLLMServer(spec, gpu_ids(range(offset, offset + tp))))
            offset += tp
        workers = list(range(offset, total_gpus))
        self._print_plan(servers, workers, total_gpus)

        self._hook_cleanup()
        for server in servers:
            server.start()
            self._servers.append(server)
        for server in servers:
            server.wait_healthy(health_timeout)
        return workers

    def shutdown(self) -> None:
        """Stop every server this manager started."""
        servers, self._servers = self._servers, []
        for server in reversed(servers):
            server.stop()

    def release_gpus(self, total_gpus: int, *, timeout: float = 120) -> None:
        """Stop the servers, kill whatever still holds an agent or reranker
        port, and wait until the GPUs are free for the next stage."""
        ports = {s.spec.port for s in self._servers} | cleanup_ports()
        self.shutdown()
        for port in sorted(ports):
            _kill_port_holders(port)
        _free_torch_memory()
        _wait_for_free_gpus(gpu_ids(range(total_gpus)), timeout=timeout)

    def _hook_cleanup(self) -> None:
        if self._hooked:
            return
        self._hooked = True
        atexit.register(self.shutdown)
        if threading.current_thread() is threading.main_thread():
            def on_sigterm(signum, _frame):
                self.shutdown()
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, on_sigterm)

    @staticmethod
    def _print_plan(servers: List[VLLMServer], workers: List[int], total_gpus: int) -> None:
        print(f"\n{'=' * 70}\n[vLLM] GPU allocation ({total_gpus} GPUs) — vLLM {vllm_version()}\n{'=' * 70}")
        for server in servers:
            spec = server.spec
            print(f"  {spec.label}\n    model {spec.model}, port {spec.port}, GPUs {server.gpus} (TP={len(server.gpus)})")
        print(f"  Pipeline workers: GPUs {workers}\n{'=' * 70}\n")
        if not workers:
            print("[vLLM] WARNING: no GPU left for the pipeline workers: retrieval runs on the CPU (slow).")


@contextmanager
def serve(spec: ServerSpec, *, health_timeout: float = 900) -> Iterator[str]:
    """Serve *spec* for the ``with`` block and yield its API base.

    A server already on the spec's port that serves the model is used and left
    running.  Otherwise one starts on the first visible GPUs (TP fitted to
    them) and stops when the block ends, also on an error.
    """
    gpus: List[str] = []
    if not port_in_use(spec.port):
        visible = visible_gpus()
        if not visible:
            raise VLLMServerError(f"{spec.label} is served by vLLM and needs a GPU; none is visible")
        gpus = visible[:fit_tp(spec, per_gpu_gb(), max_tp=len(visible))]
    server = VLLMServer(spec, gpus)
    try:
        server.start()
        server.wait_healthy(health_timeout)
        yield spec.api_base
    finally:
        server.stop()


def _kill_port_holders(port: int) -> None:
    try:
        out = subprocess.run(["fuser", "-n", "tcp", str(port)], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    for pid in out.split():
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"[vLLM] Killed PID {pid} still on port {port}")
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def _free_torch_memory() -> None:
    import gc
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def _wait_for_free_gpus(ids: List[str], *, min_free: float = 0.85, timeout: float = 120) -> None:
    """Poll nvidia-smi until every GPU of *ids* (``gpu_ids``) has *min_free*
    of its memory free."""
    if not ids:
        return
    print(f"[vLLM] Waiting for GPUs {ids} to free their memory (need {min_free:.0%} free)...")
    start, last_print = time.time(), 0.0
    while time.time() - start < timeout:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits",
                                  "-i", ",".join(map(str, ids))], capture_output=True, text=True, timeout=10).stdout
            usage = [tuple(int(x) for x in line.split(",")) for line in out.strip().split("\n")]
            if time.time() - last_print >= 10:
                last_print = time.time()
                for gpu, (free, total) in zip(ids, usage):
                    print(f"  GPU {gpu}: {free} MB free of {total} MB")
            if all(free / total >= min_free for free, total in usage):
                print(f"[vLLM] GPUs {ids} free ({time.time() - start:.0f}s)")
                return
        except Exception:  # noqa: BLE001 - nvidia-smi hiccup: try again
            pass
        time.sleep(3)
    print(f"[vLLM] WARNING: GPUs {ids} still in use after {timeout:.0f}s; going on")


def vllm_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("vllm")
    except PackageNotFoundError:
        return "not installed"
