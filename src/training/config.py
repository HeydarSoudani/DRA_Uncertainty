"""Training configuration — dataclasses + YAML loader.

Mirrors the inference convention (a YAML file holding mostly-fixed knobs, with
CLI overrides merged on top) but is self-contained: it does NOT import
``utils.cli_setup`` so the training path stays decoupled from inference-specific
argument logic.

Everything has a default so a bare ``TrainingConfig()`` is a valid CPU smoke-test
config (all components default to ``mock``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional


# The policy model the agent is trained INTO -- the student of the distillation
# plan, served by vLLM for rollouts (``rollout.model``) and used as the
# SFT cold-start backbone (``sft.base_model``).  Kept in one place so the two
# cannot drift: training a different backbone than the one being rolled out is
# silent and produces garbage advantages.
#
# Qwen3.5-9B is the student of the trajectory-distillation plan: it is the
# backbone DeepSearch-World validated in an offline verifiable search
# environment, and it shares its tokenizer with the Qwen3.6-27B teacher that
# generates the trajectories (utils.config.AGENTIC_MODEL_TO_LLM["belief"]),
# which is what keeps token-level on-policy distillation available after SFT.
DEFAULT_POLICY_MODEL: str = "Qwen/Qwen3.5-9B"


# ---------------------------------------------------------------------------
# Section configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    source: str = "synthetic"                 # "synthetic" (CPU test) | "dataset" (indexing_corpus_dataset)
    dataset: str = "trqa"                     # used when source == "dataset"
    dataset_year: Optional[str] = None        # TRQA: the split (test | validation)
    subset: Optional[str] = None              # TRQA: wiki1 | wiki2
    query_key: str = "text"
    data_path: Optional[str] = None           # None = the inference default for the dataset
    qrels_data_path: Optional[str] = None
    min_relevance_score: Optional[int] = None
    limit: Optional[int] = None
    shuffle: bool = True                      # shuffle prompts (seed: rl.seed)
    num_synthetic: int = 8                    # size of the synthetic dataset


@dataclass
class RetrievalConfig:
    """The retriever the agent searches against (rollout.tool == "retrieval").

    Built as ``dra_inference.py`` builds it; empty paths take the inference
    defaults for ``data.dataset``.
    """
    retriever: str = "qwen3_emb_4b"
    index_dir: Optional[str] = None
    corpus_path: Optional[str] = None
    top_k: int = 100                          # docs retrieved per search (envs slice what they show)
    retrieval_input: str = "subquery"


@dataclass
class RolloutConfig:
    policy: str = "mock"                      # "mock" | "server" (vLLM, token-level)
    tool: str = "mock"                        # "mock" | "retrieval"
    server_url: Optional[str] = None          # vLLM base url, when policy == "server"
    model: Optional[str] = DEFAULT_POLICY_MODEL   # served model name (ignored by policy == "mock")
    tokenizer: Optional[str] = None           # HF tokenizer of the policy; None = model
    chat_template_kwargs: Dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    request_timeout: float = 600.0
    max_turns: int = 5                        # reference env only; agent envs carry their own cap
    max_gen_tokens: int = 512                 # reference env only
    max_generations: int = 64                 # hard cap on policy calls per rollout (any env)
    temperature: float = 1.0
    top_k_docs: int = 5                       # passages shown per search (inference: seen_top_k)
    group_size: int = 4                       # rollouts per prompt (GRPO group)
    concurrency: int = 8                      # max concurrent rollouts (async)
    mock_malformed_rate: float = 0.0          # policy == "mock": share of turns with no action


@dataclass
class RewardConfig:
    # "auto" (trqa_exact on TRQA, llm_judge otherwise) | "trqa_exact" | "llm_judge" | "mock"
    outcome_metric: str = "auto"
    judge_model: str = "openrouter/qwen/qwen3-32b"   # llm_judge; the inference default
    lambda_process: float = 0.0               # weight on process reward (0 = outcome only)
    clamp_process: bool = True                # bound process shaping so it can't dominate outcome
    signals: tuple = ("doc_novelty", "marginal_recall")  # TODO(reward): which controller signals to use


@dataclass
class BufferConfig:
    staleness_k: int = 1                      # max weight-version lag tolerated (async)
    capacity: int = 4096


@dataclass
class TrainerConfig:
    backend: str = "mock"                     # "mock" | "grpo" (veRL adapter, TODO)
    lr: float = 1e-6
    kl_coef: float = 0.001
    save_dir: Optional[str] = None
    save_every: int = 50
    # backend-specific extras (veRL config path, parallelism, ...) go here
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTConfig:
    backend: str = "none"                     # "none" (build data only) | "verl" (TODO)
    data_out: Optional[str] = None            # JSONL of per-turn chat samples; None = {output_dir}/sft/sft_data.jsonl
    base_model: Optional[str] = DEFAULT_POLICY_MODEL   # cold-start backbone; keep == rollout.model
    teacher_model: str = "openrouter/qwen/qwen3.6-27b"  # generates the SFT trajectories
    teacher_temperature: float = 0.6          # the belief agent's inference temperature
    samples_per_prompt: int = 1               # teacher rollouts per prompt
    min_outcome: float = 1.0                  # keep trajectories whose outcome reward reaches this
    drop_format_retries: bool = True          # keep only trajectories with no re-asked turn
    concurrency: int = 8
    epochs: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    mode: str = "sync"                        # "sync" (validate first) | "async" (production)
    total_steps: int = 4                      # trainer updates
    prompts_per_step: int = 2                 # prompts sampled per sync step
    seed: int = 0


@dataclass
class TrainingConfig:
    stage: str = "rl"                         # "rl" | "sft" (build SFT data, then the SFT backend)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    # Agent-specific settings, parsed by the entry script that plugs the agent in.
    agent: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None          # rollouts, SFT data, run config
    save_rollouts: bool = True                # write every scored group under output_dir/rollouts
    verbose: bool = True


# ---------------------------------------------------------------------------
# (de)serialization + merge
# ---------------------------------------------------------------------------

def to_dict(cfg) -> Dict[str, Any]:
    from dataclasses import asdict
    return asdict(cfg)


def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Uses ``get_type_hints`` so ``from __future__ import annotations`` (which turns
    field types into strings) still resolves nested dataclass fields.
    """
    if not is_dataclass(cls):
        return data
    from typing import get_type_hints
    type_hints = get_type_hints(cls)
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = type_hints.get(f.name)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """Load a TrainingConfig from YAML (optional) with dict overrides merged on top."""
    data: Dict[str, Any] = {}
    if path:
        import yaml  # local import: not needed for the pure-Python smoke path
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        _deep_update(data, overrides)
    return _from_dict(TrainingConfig, data)
