"""Settings of belief model training: one section per phase plus the shared one.

The YAML file (``experiments/configs/belief_model_training.yaml``) has the
same sections and keys as the dataclasses below; CLI flags and
``--set section.key=value`` override it.  An unknown section or key is an
error.  Training hyperparameters default to Agent-BRACE's recipes
(arXiv 2605.11436; ``recipes/finetune_belief_qwen25_3b.sh`` for SFT,
``recipes/textworld_basic_ppo_decouple_joint_belief_vllm.sh`` for RL).
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from indexing_corpus_dataset.layout import BUILT_DATASETS, DATASET_SPECS, OUTPUT_ROOT, resolve_split_id


@dataclass(frozen=True)
class Split:
    """One dataset split, as ``experiments/inference/dra_inference.py`` addresses it."""
    dataset: str
    subset: Optional[str] = None
    dataset_year: Optional[str] = None

    @property
    def split_id(self) -> str:
        """The on-disk split id (``layout.resolve_split_id``), e.g. ``wiki1_validation``."""
        return resolve_split_id(self.dataset, self.dataset_year, self.subset)

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.split_id}"


@dataclass
class SharedConfig:
    # Training pool of experiments/training/build_training_pool.py: deep_qa
    # (InfoSeek-Hard, WebShaper, FRAMES) or trqa_validation (TRQA wiki1 +
    # wiki2 validation).
    training_data: str = "deep_qa"
    run_name: str = "default"
    # Root of the training runs; null = OUTPUT_ROOT/belief_model_training.
    output_root: Optional[str] = None
    seed: int = 0
    # Agent, retriever and belief model settings of the inference runs in
    # phase 1 (max_iteration, seen_top_k, ...): never repeated here.
    inference_config: str = "experiments/configs/dra_inference.yaml"
    retriever: str = "qwen3_emb_4b"
    # dra_inference workers in phase 1.
    num_gpus: int = 4
    # Query targeting scorer (criteria attempts) of phase 1: frozen,
    # not part of the trained belief model; null = trajectories.teacher_belief_model.
    query_scorer_model: Optional[str] = None
    # Longest sequence (prompt + reply, tokens) of the student, in SFT and RL:
    # longer SFT samples are dropped, longer RL prompts get no reply.
    max_seq_len: int = 32768


@dataclass
class TrajectoriesConfig:
    """Phase 1: frozen agent + teacher belief model in monitor mode, on the
    training pool ``shared.training_data``."""
    policy_model: str = "glm"
    # Teacher belief model: its criteria (init) and belief updates are the SFT targets.
    teacher_belief_model: str = "openrouter/qwen/qwen3.6-27b"
    # SFT samples only from replies whose cited spans all occur in their passage.
    require_verified_spans: bool = True
    # The next N pool queries not run yet (null = all), as dra_inference --limit.
    limit: Optional[int] = None
    # Shares of the queries for SFT and validation (episodes.assign_use); the rest is for RL.
    sft_share: float = 0.30
    val_share: float = 0.05


@dataclass
class SFTConfig:
    """Phase 2: SFT warm-up of the student on the teacher's replies (LoRA, then merged)."""
    student_belief_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    # Agent-BRACE uses 2e-5; 1e-4 is the usual LoRA rate (LoRA trains few weights).
    learning_rate: float = 1e-4
    # Upper bound: the checkpoint with the lowest validation loss is kept.
    num_epochs: int = 3
    # Effective batch per GPU 1 x 32 = Agent-BRACE's 4 x 8; one sample per
    # forward pass bounds the memory of samples up to shared.max_seq_len.
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    warmup_ratio: float = 0.03
    logging_steps: int = 5


@dataclass
class RLConfig:
    """Phase 3: GRPO on belief updates from the teacher's and the student's
    own belief chains (env, rewards), with the PPO update of Agent-BRACE's
    BeliefStateLMTrainer (no critic, no KL by default)."""
    # Start checkpoint of the student belief model; null = the merged phase 2 model.
    init_belief_model: Optional[str] = None
    num_iterations: int = 200
    # Prompts (updater steps of rl episodes) per iteration, and replies per prompt (GRPO group).
    prompts_per_batch: int = 32
    group_size: int = 8
    # Batch mix: share of steps with new gold; at most this share from episodes without gold.
    gold_step_share: float = 0.5
    no_gold_share: float = 0.15
    # Share of the prompts from the student's own belief chain (the rest
    # teacher-forced), reached linearly over the first student_state_ramp of
    # the iterations (scheduled sampling; 0 = from the first iteration).
    student_state_share: float = 1.0
    student_state_ramp: float = 0.5
    # Weights of the reward terms (rewards.py): state tracking, state correctness, task success.
    reward_weight_st: float = 1.0
    reward_weight_sc: float = 1.0
    reward_weight_ts: float = 1.0
    # GDPO advantage: each term normalized within its group by max(std, adv_std_floor).
    adv_std_floor: float = 0.05
    # Validation every val_every iterations on at most val_max_episodes val episodes
    # (fully replayed): the student's own chain, and teacher-forced.
    val_every: int = 20
    val_max_episodes: int = 50
    learning_rate: float = 1e-5
    ppo_epochs: int = 1
    # 256 replies per iteration / 32 = 8 optimizer steps per iteration.
    ppo_mini_batch_size: int = 32
    # Ratio clipped to [1 - clip_ratio_low, 1 + clip_ratio_high] (DAPO's clip-higher).
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    grad_clip: float = 1.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    temperature: float = 1.0
    # The inference updater's max_tokens (LLMBeliefUpdater).
    max_new_tokens: int = 8192
    vllm_gpu_memory_utilization: float = 0.55
    vllm_max_num_seqs: int = 64
    # A step_{n}/ checkpoint every save_every iterations; the last save_total_limit are kept.
    save_every: int = 20
    save_total_limit: int = 3
    # The updated weights pass from the trainer to vLLM through this directory.
    sync_dir: str = "/dev/shm/belief_model_training_sync"


SECTIONS = {
    "shared": SharedConfig,
    "trajectories": TrajectoriesConfig,
    "sft": SFTConfig,
    "rl": RLConfig,
}


@dataclass
class TrainingConfig:
    shared: SharedConfig = field(default_factory=SharedConfig)
    trajectories: TrajectoriesConfig = field(default_factory=TrajectoriesConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def training_split(self) -> Split:
        """The split of ``shared.training_data`` that dra_inference runs."""
        return Split(self.shared.training_data, DATASET_SPECS[self.shared.training_data].subset)

    @property
    def query_scorer_model(self) -> str:
        return self.shared.query_scorer_model or self.trajectories.teacher_belief_model


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
    if cfg.shared.training_data not in BUILT_DATASETS:
        raise ValueError(f"shared.training_data: expected one of {sorted(BUILT_DATASETS)}, got {cfg.shared.training_data!r}")
    return cfg


class RunPaths:
    """Files of one training run under ``{output_root}/{training_data}/{run_name}``."""

    def __init__(self, cfg: TrainingConfig) -> None:
        root = Path(cfg.shared.output_root) if cfg.shared.output_root else OUTPUT_ROOT / "belief_model_training"
        self.root = root / cfg.shared.training_data / cfg.shared.run_name
        # Phase 1
        self.trajectories_dir = self.root / "phase1_trajectories"
        self.inference_runs = self.trajectories_dir / "runs"
        self.episodes_file = self.trajectories_dir / "episodes.jsonl"
        self.sft_file = self.trajectories_dir / "sft.jsonl"
        self.sft_val_file = self.trajectories_dir / "sft_val.jsonl"
        self.teacher_scores_file = self.trajectories_dir / "teacher_scores.json"
        # Phase 2
        self.sft_dir = self.root / "phase2_sft"
        self.sft_merged = self.sft_dir / "merged"
        # Phase 3
        self.rl_dir = self.root / "phase3_rl"
        self.rl_final = self.rl_dir / "final"
        self.rl_metrics = self.rl_dir / "metrics.jsonl"
        self.rl_validation = self.rl_dir / "validation.jsonl"
