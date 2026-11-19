"""Settings of belief model training, one section per phase plus ``shared``.

The YAML file (``experiments/configs/belief_training.yaml``) has the same
sections and keys as the dataclasses below, plus the ``data`` section: the
settings of the training dataset build (phase 1), whose sub-sections are
those of ``indexing_corpus_dataset.training_datasets.config``.  CLI
``--set section.key=value`` overrides a key.  An unknown section or key is
an error.

Hyperparameters follow Agent-BRACE (arXiv 2605.11436): SFT LoRA r 64 /
alpha 128, lr 2e-5, 5 epochs; RL lr 1e-5, PPO clip 0.2, dual-clip 3.0,
no critic, group-normalized advantage, temperature 0.7.
"""

import dataclasses
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from indexing_corpus_dataset.layout import BUILT_DATASETS, OUTPUT_ROOT
from utils.config import BELIEF_MODEL_TO_LLM, BELIEF_TRAINING_ROOT, POLICY_MODEL_TO_LLM, trained_belief_checkpoint
from utils.sectioned_config import load_sectioned

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Starting beliefs of an RL run: the dataset's gold beliefs, or the
#: student's own init call (trained with the chain's advantage).
GOLD_INIT, OWN_INIT = "gold", "own"
RL_BELIEFS = (GOLD_INIT, OWN_INIT)
#: The training run of each RL starting-belief choice (its folder under
#: BELIEF_TRAINING_ROOT, and its served name in utils.config.TRAINED_BELIEF_RUNS).
RL_RUNS = {GOLD_INIT: "rl", OWN_INIT: "rl_own_init"}
SFT_RUN = "sft"


@dataclass
class SharedConfig:
    # Training datasets (phase 1 builds them; their sft / rl / val splits feed
    # phases 2 to 4).  Their episodes are pooled: one model for every shape.
    training_datasets: List[str] = field(default_factory=lambda: ["infoseek", "trqa_validation"])
    # Frozen agents whose trajectories are the episodes (phase 2 runs them).
    policy_models: List[str] = field(default_factory=lambda: ["oss_20b", "glm"])
    retriever: str = "qwen3_emb_4b"
    # Teacher belief model (a --belief-model short name of dra_inference):
    # its replies on the sft and val splits are the SFT samples.
    teacher: str = "qwen3.6_27b"
    # The student: Hugging Face weights of the SFT start.
    student: str = "Qwen/Qwen3.5-4B"
    # The inference settings of phase 2 (agent, retriever, belief stage);
    # its ``output`` is where the runs are (null = OUTPUT_ROOT).
    inference_config: str = "experiments/configs/dra_inference.yaml"
    seed: int = 0
    # Longest student sequence (prompt + reply, tokens): longer SFT samples
    # are dropped, an RL prompt longer than this gets an empty reply.
    max_seq_len: int = 32768


@dataclass
class TrajectoriesConfig:
    """Phase 2: dra_inference --stage generate on a split, then --stage
    belief with the teacher on the sft and val splits."""
    # Query worker processes of the generate stage (dra_inference --workers).
    workers: int = 1
    # Belief stage threads (dra_inference --belief-workers); null = the inference config's.
    belief_workers: Optional[int] = None
    quiet: bool = True


@dataclass
class SFTConfig:
    """Phase 3: SFT of the student on the teacher's replies (LoRA, then merged)."""
    # Datasets whose gold beliefs are init samples (question -> gold beliefs).
    # Multi-hop only by default: the beliefs of a multi-hop question are in
    # its words; TRQA's members are not in the question.
    init_datasets: List[str] = field(default_factory=lambda: ["infoseek"])
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # Attention, Gated DeltaNet (Qwen3.5: in_proj_qkv, in_proj_z, out_proj) and
    # MLP projections; names a model lacks are skipped.
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 2e-5
    # Upper bound: the epoch with the lowest validation loss is merged.
    num_epochs: int = 5
    # Effective batch 1 x 32 per GPU = Agent-BRACE's 4 x 8 (one sample per
    # forward pass bounds the memory of long samples).
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    warmup_ratio: float = 0.03
    logging_steps: int = 5


@dataclass
class RLConfig:
    """Phase 4: GRPO over belief chains, reward = Brier of the final score."""
    # Starting beliefs: gold (run ``rl``) or own (the student's init call, run ``rl_own_init``).
    beliefs: str = GOLD_INIT
    # Start checkpoint; null = the SFT model (``sft/final``).
    init_model: Optional[str] = None
    num_iterations: int = 200
    # Episodes per iteration, chains per episode (the GRPO group), and
    # final-score replies per chain (each scored; their mean is the chain's reward).
    episodes_per_batch: int = 8
    group_size: int = 8
    scorer_samples: int = 4
    temperature: float = 0.7
    # Episodes with more search iterations (with a document) than this are
    # left out (null = none): set from the phase 2 statistics.
    max_chain_steps: Optional[int] = None
    # Rollouts per iteration to replace groups whose chains all have the same reward.
    refill_rounds: int = 2
    learning_rate: float = 1e-5
    ppo_epochs: int = 1
    # Chains per optimizer step (all the calls of a chain in one mini-batch).
    ppo_mini_batch_chains: int = 4
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    grad_clip: float = 1.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    # Validation (greedy chains on the val episodes; selects final/ by s_T
    # AUROC) before training and every val_every iterations.
    val_every: int = 10
    val_max_episodes: Optional[int] = None
    vllm_gpu_memory_utilization: float = 0.55
    vllm_max_num_seqs: int = 128
    save_every: int = 20
    save_total_limit: int = 3
    # The updated weights pass from the trainer to vLLM through this directory.
    sync_dir: str = "/dev/shm/belief_training_sync"


SECTIONS = {
    "shared": SharedConfig,
    "trajectories": TrajectoriesConfig,
    "sft": SFTConfig,
    "rl": RLConfig,
}
#: The section with the dataset build settings (phase 1).
DATA_SECTION = "data"


@dataclass
class TrainingConfig:
    shared: SharedConfig = field(default_factory=SharedConfig)
    trajectories: TrajectoriesConfig = field(default_factory=TrajectoriesConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    # The ``data`` section as written (phase 1 hands it to the build).
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def inference_config_path(self) -> Path:
        path = Path(self.shared.inference_config)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def inference_output(self) -> Path:
        """The root of the inference runs: ``output`` of the inference
        config (null = OUTPUT_ROOT; relative to the repo root)."""
        data = yaml.safe_load(self.inference_config_path.read_text()) or {}
        return REPO_ROOT / data["output"] if data.get("output") else OUTPUT_ROOT

    def build_config(self):
        """The dataset build settings of the ``data`` section
        (``training_datasets.config.load_config``, with its checks)."""
        from indexing_corpus_dataset.training_datasets.config import load_config as load_build_config

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            yaml.safe_dump(self.data, f)
            f.flush()
            return load_build_config(f.name)


def load_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """The config of *path* (YAML; None = the defaults) with *overrides*
    (``{"section.key": value}``) applied."""
    data = (yaml.safe_load(Path(path).read_text()) or {}) if path else {}
    build = data.pop(DATA_SECTION, None) or {}
    items = {f"{s}.{k}": v for s, values in data.items() for k, v in (values or {}).items()}
    items.update(overrides or {})
    cfg = load_sectioned(TrainingConfig(), SECTIONS, None, items)
    cfg.data = build
    sh = cfg.shared
    unknown = sorted(set(sh.training_datasets) - BUILT_DATASETS)
    if unknown:
        raise ValueError(f"shared.training_datasets: {unknown} are not training datasets ({sorted(BUILT_DATASETS)})")
    unknown = sorted(set(sh.policy_models) - set(POLICY_MODEL_TO_LLM))
    if unknown:
        raise ValueError(f"shared.policy_models: unknown agents {unknown}")
    if sh.teacher not in BELIEF_MODEL_TO_LLM:
        raise ValueError(f"shared.teacher: {sh.teacher!r} is no --belief-model name ({sorted(BELIEF_MODEL_TO_LLM)})")
    if cfg.rl.beliefs not in RL_BELIEFS:
        raise ValueError(f"rl.beliefs: expected one of {list(RL_BELIEFS)}, got {cfg.rl.beliefs!r}")
    if cfg.rl.group_size < 2 or cfg.rl.scorer_samples < 1:
        raise ValueError("rl.group_size must be >= 2 and rl.scorer_samples >= 1")
    return cfg


class RunPaths:
    """The files of the training runs under ``BELIEF_TRAINING_ROOT``
    (``utils.config``): ``stats/`` (phase 2 statistics), ``sft/`` (phase
    3) and ``rl/`` or ``rl_own_init/`` (phase 4); the ``final/`` of each
    run is what dra_inference serves (``utils.config.TRAINED_BELIEF_RUNS``)."""

    def __init__(self) -> None:
        self.root = BELIEF_TRAINING_ROOT
        self.stats_dir = self.root / "stats"
        self.sft_dir = self.root / SFT_RUN
        self.sft_samples = self.sft_dir / "samples.jsonl"
        self.sft_final = trained_belief_checkpoint(SFT_RUN)

    def rl_dir(self, beliefs: str) -> Path:
        return self.root / RL_RUNS[beliefs]

    def rl_final(self, beliefs: str) -> Path:
        return trained_belief_checkpoint(RL_RUNS[beliefs])
