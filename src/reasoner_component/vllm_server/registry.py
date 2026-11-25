"""The vLLM server registry: how every self-hosted model is served.

``ServerSpec`` is one model's server settings; ``SERVERS`` indexes every spec
by the model name a client sends (``server_spec`` / ``require_spec``).  To
serve a new model, add its spec to ``_SPECS``.

Importing this module starts nothing and imports neither ``vllm`` nor ``torch``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Paths honour the environment first, so SLURM jobs (which don't source
# ~/.bashrc) set them once for this module and the serve scripts alike.
HF_HOME = os.environ.get("HF_HOME", "/projects/0/prjs0834/heydars/.cache/huggingface")
DOWNLOAD_DIR = os.environ.get("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
LOG_DIR = Path(os.environ.get("DRA_VLLM_LOG_DIR", "/tmp/vllm_server_logs"))
# Pre-staged weights (for offline nodes): ``<DRA_MODELS_DIR>/<org>--<model>/``.
MODELS_DIR = Path(os.environ.get("DRA_MODELS_DIR", "/projects/0/prjs0834/heydars/DRA_training/models"))

# One port per role.  The belief port stays outside sshd's X11 forwarding range
# (6000 + display number, taken on login nodes) and outside ``cleanup_ports``:
# the belief stage starts its own server after the generate stage.
AGENT_PORT = 6008
JUDGE_PORT = 6009
BELIEF_PORT = 7010
RERANKER_PORT = 8000
RANK_R1_PORT = 8001


def local_api_base(port: int) -> str:
    """The OpenAI-compatible API of a server on this node at *port*."""
    return f"http://127.0.0.1:{port}/v1"


# Where the self-managed agents reach their server by default.
AGENT_API_BASE = local_api_base(AGENT_PORT)


@dataclass(frozen=True)
class ServerSpec:
    """How to serve one model.

    ``model`` is the weights (an HF id or a local path) and the name the
    server registers them under; a LoRA server is asked for its adapter
    alias instead (``name``).  TP is fitted to the GPUs (``fit_tp``) from
    ``model_gb``, the approximate weight footprint, within ``allowed_tp``
    (TP must divide the model's (kv-)head count); without ``model_gb`` it is
    ``tp_size``.
    """

    model: str
    port: int
    role: str                                   # "agent" | "reranker" | "belief" | "judge"
    label: str
    tp_size: int = 1
    model_gb: Optional[float] = None
    allowed_tp: Tuple[int, ...] = (1, 2, 4)
    max_model_len: Optional[int] = 65536
    max_num_seqs: Optional[int] = 64
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    dtype: Optional[str] = None
    tool_parser: Optional[str] = None           # --enable-auto-tool-choice --tool-call-parser
    lora: Optional[Tuple[str, str]] = None      # (alias, adapter): --lora-modules alias=adapter
    max_lora_rank: Optional[int] = None
    extra_args: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """The model name a client sends."""
        return self.lora[0] if self.lora else self.model

    @property
    def api_base(self) -> str:
        return local_api_base(self.port)


def _agent(model: str, label: str, **kw) -> ServerSpec:
    return ServerSpec(model=model, port=AGENT_PORT, role="agent", label=label, **kw)


def _reranker(model: str, label: str, port: int = RERANKER_PORT, **kw) -> ServerSpec:
    return ServerSpec(model=model, port=port, role="reranker", label=label, **kw)


def _qwen7b_agent(model: str, label: str) -> ServerSpec:
    # Qwen2.5-7B finetunes: 32k native context, 28 heads / 4 kv heads.
    return _agent(model, label, max_model_len=32768, model_gb=16.0, allowed_tp=(1, 2))


def _qwen35_4b_belief(model: str, label: str) -> ServerSpec:
    # Tuned for throughput: the belief stage sends --belief-workers requests at
    # once, and every call shares the system prompt.  Paged attention reserves
    # nothing for max_model_len up front, so it covers the longest belief call.
    return ServerSpec(
        model=model, port=BELIEF_PORT, role="belief", label=label,
        max_model_len=65536, max_num_seqs=128, dtype="bfloat16", enable_prefix_caching=True,
        model_gb=9.0, allowed_tp=(1,),
        extra_args=("--language-model-only",),  # text only: the vision tower is not loaded
    )


_BASE_BELIEF = _qwen35_4b_belief("Qwen/Qwen3.5-4B", "Belief model Qwen3.5-4B")

_SPECS: Tuple[ServerSpec, ...] = (
    # ── Agents ────────────────────────────────────────────────────────────────
    _agent("openai/gpt-oss-20b", "GPT-OSS-20B", max_model_len=131072, max_num_seqs=16,
           enforce_eager=True, model_gb=13.0, allowed_tp=(1,)),            # mxfp4
    _agent("openai/gpt-oss-120b", "GPT-OSS-120B", tp_size=2, max_model_len=131072, max_num_seqs=16,
           enforce_eager=True, model_gb=63.0),                             # mxfp4 MoE
    _agent("zai-org/GLM-4.7-Flash", "GLM-4.7-Flash", max_model_len=202752, max_num_seqs=16,
           enable_prefix_caching=True, model_gb=60.0, tool_parser="glm47"),  # 20 heads: TP | 20
    _agent("Alibaba-NLP/Tongyi-DeepResearch-30B-A3B", "Tongyi-DeepResearch-30B", tp_size=2,
           max_model_len=131072, max_num_seqs=16, enable_prefix_caching=True,
           model_gb=61.0, tool_parser="hermes"),                           # MoE, kv_heads=4
    _agent("openbmb/AgentCPM-Report", "AgentCPM-Report", model_gb=16.0, allowed_tp=(1, 2)),
    _agent("openbmb/AgentCPM-Explore", "AgentCPM-Explore", max_model_len=32768,
           model_gb=16.0, allowed_tp=(1, 2)),
    _agent("rl-research/DR-Tulu-8B", "DR-Tulu-8B", model_gb=16.0, allowed_tp=(1, 2)),
    _qwen7b_agent("PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3", "Search-R1-7B"),
    _qwen7b_agent("agentrl/ReSearch-Qwen-7B-Instruct", "ReSearch-Qwen-7B"),
    _qwen7b_agent("Zill1/StepSearch-7B-Instruct", "StepSearch-7B"),
    # ── Rerankers ─────────────────────────────────────────────────────────────
    _reranker("jhu-clsp/rank1-7b", "Rank1-7B reranker", max_model_len=4096, dtype="float16",
              model_gb=15.0, allowed_tp=(1, 2)),
    _reranker("Qwen/Qwen3-Reranker-0.6B", "Qwen3-Reranker-0.6B", max_model_len=8192,
              enable_prefix_caching=True, model_gb=2.0, allowed_tp=(1,)),
    _reranker("Qwen/Qwen3-Reranker-4B", "Qwen3-Reranker-4B", max_model_len=8192,
              enable_prefix_caching=True, model_gb=9.0, allowed_tp=(1,)),
    _reranker("Qwen/Qwen3-Reranker-8B", "Qwen3-Reranker-8B", max_model_len=8192,
              enable_prefix_caching=True, model_gb=16.0, allowed_tp=(1, 2)),
    _reranker("Qwen/Qwen3-8B", "Qwen3-8B listwise reranker", max_model_len=8192,
              enable_prefix_caching=True, model_gb=16.0, allowed_tp=(1, 2)),
    _reranker("castorini/rank_zephyr_7b_v1_full", "RankZephyr-7B listwise reranker", max_model_len=8192,
              enable_prefix_caching=True, model_gb=15.0, allowed_tp=(1, 2)),
    _reranker("Qwen/Qwen2.5-7B-Instruct", "Rank-R1-7B setwise reranker (LoRA)", port=RANK_R1_PORT,
              max_model_len=8192, model_gb=16.0, allowed_tp=(1, 2),
              lora=("rank-r1", "ielabgroup/Rank-R1-7B-v0.1"), max_lora_rank=32),
    # ── Judge (the pipeline's judge runs on OpenRouter; served by hand) ──────
    ServerSpec(model="Qwen/Qwen3-32B", port=JUDGE_PORT, role="judge", label="Qwen3-32B judge",
               max_model_len=16384, max_num_seqs=4, model_gb=64.0),
    # ── Belief models ─────────────────────────────────────────────────────────
    _BASE_BELIEF,
)


def _trained_belief_specs() -> List[ServerSpec]:
    """The trained belief models (``utils.config.TRAINED_BELIEF_RUNS``): their
    checkpoint path, served with their student's settings."""
    from utils.config import TRAINED_BELIEF_RUNS, student_weights, trained_belief_checkpoint

    students = {_BASE_BELIEF.model: _BASE_BELIEF}
    return [
        dataclasses.replace(students[student_weights(student)],
                            model=str(trained_belief_checkpoint(student, run)), label=f"Belief model {name}")
        for name, (student, run) in TRAINED_BELIEF_RUNS.items()
    ]


def _index(specs: Iterable[ServerSpec]) -> Dict[str, ServerSpec]:
    index: Dict[str, ServerSpec] = {}
    for spec in specs:
        if spec.name in index:
            raise ValueError(f"two vLLM server specs for {spec.name}")
        index[spec.name] = spec
    return index


SERVERS: Dict[str, ServerSpec] = _index([*_SPECS, *_trained_belief_specs()])


class VLLMServerError(RuntimeError):
    """A vLLM server could not be planned, started or reused."""


def bare_model_name(model: str) -> str:
    """*model* without its ``vllm/`` routing prefix."""
    return model.removeprefix("vllm/")


def server_spec(model: str) -> Optional[ServerSpec]:
    """The spec of the server for *model* (``vllm/`` prefix ignored);
    None when no local server serves it."""
    return SERVERS.get(bare_model_name(model))


def require_spec(model: str) -> ServerSpec:
    """``server_spec`` that raises when *model* has no spec."""
    spec = server_spec(model)
    if spec is None:
        raise VLLMServerError(f"no vLLM server spec for {bare_model_name(model)!r} in "
                              f"reasoner_component/vllm_server/registry.py (known: {', '.join(sorted(SERVERS))})")
    return spec


def cleanup_ports() -> set:
    """The ports of every agent and reranker server: after the generate stage
    ``VLLMServerManager.release_gpus`` kills any process bound to one, also a
    server this process did not start."""
    return {spec.port for spec in SERVERS.values() if spec.role in ("agent", "reranker")}
