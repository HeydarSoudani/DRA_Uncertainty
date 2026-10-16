"""Settings of belief model training: one section per phase plus the shared one.

The YAML file (``experiments/configs/belief_model_training.yaml``) has the
same sections and keys as the dataclasses below; CLI flags and
``--set section.key=value`` override it.  An unknown section or key is an
error.  Training hyperparameters default to Agent-BRACE's recipes
(arXiv 2605.11436; ``recipes/finetune_belief_qwen25_3b.sh`` for SFT,
``recipes/textworld_basic_ppo_decouple_joint_belief_vllm.sh`` for RL).

A split is written ``dataset[:subset[:dataset_year]]`` (the arguments of
``experiments/dra_inference.py``), e.g. ``trqa:wiki1:validation``,
``browsecomp_plus``, ``neuclir:news:2024``.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from indexing_corpus_dataset.layout import DATASET_SPECS, OUTPUT_ROOT, resolve_split_id


@dataclass(frozen=True)
class Split:
    """One dataset split, as ``experiments/dra_inference.py`` addresses it."""
    dataset: str
    subset: Optional[str] = None
    dataset_year: Optional[str] = None

    @classmethod
    def parse(cls, text: str) -> "Split":
        parts = [p.strip() or None for p in text.split(":")]
        if not parts[0] or parts[0] not in DATASET_SPECS or len(parts) > 3:
            raise ValueError(f"bad split {text!r}; expected dataset[:subset[:dataset_year]] "
                             f"with dataset in {list(DATASET_SPECS)}")
        spec = DATASET_SPECS[parts[0]]
        subset = parts[1] if len(parts) > 1 and parts[1] else spec.subset
        year = parts[2] if len(parts) > 2 and parts[2] else spec.dataset_year
        return cls(parts[0], subset, year)

    @property
    def split_id(self) -> str:
        """The on-disk split id (``layout.resolve_split_id``), e.g. ``wiki1_validation``."""
        return resolve_split_id(self.dataset, self.dataset_year, self.subset)

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.split_id}"

    def __str__(self) -> str:
        return ":".join([self.dataset, self.subset or "", self.dataset_year or ""])


@dataclass
class SharedConfig:
    run_name: str = "default"
    # Root of the training runs; null = OUTPUT_ROOT/belief_model_training.
    output_root: Optional[str] = None
    seed: int = 0
    # Agent, retriever and belief model settings of the inference runs in
    # phases 1 and 4 (max_iteration, seen_top_k, ...): never repeated here.
    inference_config: str = "experiments/configs/dra_inference.yaml"
    retriever: str = "qwen3_emb_4b"
    # dra_inference workers in phases 1 and 4.
    num_gpus: int = 4


@dataclass
class QueriesConfig:
    """Phase 0: the training queries.  TODO: to be designed."""


@dataclass
class TrajectoriesConfig:
    """Phase 1: frozen agent + teacher belief model in monitor mode."""
    agent: str = "glm"
    # Belief updater whose replies are the SFT targets.
    teacher_model: str = "openrouter/qwen/qwen3.6-27b"
    # Criteria extraction (frozen; criteria come from the criteria bank).
    criteria_model: str = "openrouter/qwen/qwen3.6-27b"
    # SFT samples only from replies whose cited spans all occur in their passage.
    require_verified_spans: bool = True


@dataclass
class SFTConfig:
    """Phase 2: SFT warm-up of the student on the teacher's replies (LoRA, then merged)."""
    student_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 2e-5
    num_epochs: int = 5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    # Samples longer than this (prompt + reply, tokens) are dropped.
    max_length: int = 16384
    logging_steps: int = 5
    save_steps: int = 100


@dataclass
class RLConfig:
    """Phase 3: PPO on the belief updater call, ported from Agent-BRACE's
    BeliefStateLMTrainer (no critic, no KL by default)."""
    # Start checkpoint; null = the merged phase 2 model.
    student_init: Optional[str] = None
    # LLM judge of the rewards.  TODO: the rewards are to be designed.
    judge_model: Optional[str] = None
    # replay: the student re-tracks the belief over the phase 1 trajectories
    # (fixed observations).  online: TODO.
    env: str = "replay"
    num_iterations: int = 200
    episodes_per_batch: int = 16
    # Steps of an episode replayed per batch; null = all.
    max_episode_steps: Optional[int] = None
    learning_rate: float = 1e-5
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 4
    clip_ratio: float = 0.2
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    grad_clip: float = 1.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    temperature: float = 1.0
    max_new_tokens: int = 4096
    vllm_gpu_memory_utilization: float = 0.55
    vllm_max_model_len: int = 32768
    vllm_max_num_seqs: int = 16
    save_every: int = 20
    # The updated weights pass from the trainer to vLLM through this directory.
    sync_dir: str = "/dev/shm/belief_model_training_sync"


@dataclass
class EvalConfig:
    """Phase 4: the frozen agent with each belief model, scored by the
    existing evaluators (correctness, Recall@N, GradedRecall@N)."""
    # Checkpoint to evaluate; null = the final phase 3 model.
    belief_model: Optional[str] = None
    # Agent; null = the phase 1 agent.
    agent: Optional[str] = None
    splits: List[str] = field(default_factory=list)
    # off: no belief model; teacher / student: that belief model in inform mode.
    variants: List[str] = field(default_factory=lambda: ["off", "teacher", "student"])
    # GPUs of the inference workers ("" = dra_inference decides); with the
    # student variant, leave serve_gpu out of them.
    inference_gpu_ids: str = "1,2,3"
    # GPU and port of the vLLM server of the student.
    serve_gpu: int = 0
    serve_port: int = 6020
    serve_max_model_len: int = 32768
    serve_gpu_memory_utilization: float = 0.85


SECTIONS = {
    "shared": SharedConfig,
    "queries": QueriesConfig,
    "trajectories": TrajectoriesConfig,
    "sft": SFTConfig,
    "rl": RLConfig,
    "eval": EvalConfig,
}


@dataclass
class TrainingConfig:
    shared: SharedConfig = field(default_factory=SharedConfig)
    queries: QueriesConfig = field(default_factory=QueriesConfig)
    trajectories: TrajectoriesConfig = field(default_factory=TrajectoriesConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: Any, current: Any, key: str) -> Any:
    """A ``--set`` string as the type of the key's current value."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("null", "none"):
        return None
    if isinstance(current, bool):
        if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"{key}: expected a boolean, got {value!r}")
        return value.lower() in ("true", "1", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """The config of *path* (YAML; None = the defaults) with *overrides*
    (``{"section.key": value}``) applied."""
    cfg = TrainingConfig()
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    items = [(f"{section}.{key}", value)
             for section, values in (data or {}).items() for key, value in (values or {}).items()]
    items += list((overrides or {}).items())
    for dotted, value in items:
        section, _, key = dotted.partition(".")
        if section not in SECTIONS:
            raise ValueError(f"unknown config section {section!r}; expected one of {list(SECTIONS)}")
        obj = getattr(cfg, section)
        if key not in {f.name for f in dataclasses.fields(obj)}:
            raise ValueError(f"unknown config key {dotted!r}")
        setattr(obj, key, _coerce(value, getattr(obj, key), dotted))
    return cfg


class RunPaths:
    """Files of one training run under ``{output_root}/{run_name}``."""

    def __init__(self, cfg: TrainingConfig) -> None:
        root = Path(cfg.shared.output_root) if cfg.shared.output_root else OUTPUT_ROOT / "belief_model_training"
        self.root = root / cfg.shared.run_name
        self.config_file = self.root / "config.yaml"
        # Phase 0
        self.queries_file = self.root / "phase0_queries" / "queries.jsonl"
        # Phase 1
        self.trajectories_dir = self.root / "phase1_trajectories"
        self.inference_runs = self.trajectories_dir / "runs"
        self.query_ids_dir = self.trajectories_dir / "query_ids"
        self.episodes_file = self.trajectories_dir / "episodes.jsonl"
        self.sft_file = self.trajectories_dir / "sft.jsonl"
        # Phase 2
        self.sft_dir = self.root / "phase2_sft"
        self.sft_merged = self.sft_dir / "merged"
        # Phase 3
        self.rl_dir = self.root / "phase3_rl"
        self.rl_final = self.rl_dir / "final"
        self.rl_metrics = self.rl_dir / "metrics.jsonl"
        # Phase 4
        self.eval_dir = self.root / "phase4_eval"
        self.eval_summary = self.eval_dir / "summary.json"
