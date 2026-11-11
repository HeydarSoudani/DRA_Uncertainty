"""Phase 2: SFT warm-up of the student belief model (Agent-BRACE's
``finetune_belief_state_sft_alfworld.py``: completion-only loss, LoRA, then
merged into full weights for RL).

The samples come from the monitor runs of phase 1 on the ``sft`` (training)
and ``val`` (validation) splits (``build_samples``, on the main process,
written to ``samples.jsonl``).  Only queries with gold criteria are used
(``runs.load_episodes``).  A sample is one belief update call: the
teacher's replies on the gold criteria replayed (``episodes.replay_teacher``,
the final call on the answer turn included), each call within the matched
replay that passes the gate (``scoring.format_ok``) and, with
``require_verified_spans``, that is clean (``core.issues.UNCLEAN_REPLY``:
every observation entered the belief, no status corrected, no cited source
without an observation, no reason that misreads the belief, no removal
code did not take, no competing one-answer candidates).  On a short-answer
query the final call is the final scorer's (the second task,
``task: score``): its sample is kept when the reply parses, its answer's
candidates have no issue (``update.final_score.check_issues``) and its status agrees
with the correctness y of the agent's answer (|p - y| <= ``SCORE_AGREEMENT``,
rejection sampling, so the student does not copy the teacher's
miscalibration; "possible", p = 1/2, is kept for both y, so the student
also learns to hedge); the counts of kept score samples per y are printed (the
two should not be far apart).  The init call (criteria
extraction) is not trained: the student runs on the gold criteria.
The input is the chat-template prompt the inference belief model sends
(``add_generation_prompt``, native reasoning off, as served by vLLM),
followed by the teacher's reply and the end-of-turn token: the sequence the
model generates at inference.  The loss is on the reply and end-of-turn
token only, averaged over the reply tokens of the batch (token-mean); the
logits are computed at the reply positions only (a full-vocabulary logit
row per prompt token would not fit in memory).
Samples longer than ``shared.max_seq_len`` are dropped.

Validation (the ``val`` samples): the loss after every epoch; a checkpoint
is saved every epoch and the one with the lowest validation loss is merged
(``num_epochs`` is an upper bound).  Without validation samples the last
epoch is merged.

Multi-GPU: launch with ``torchrun --nproc_per_node N`` (data parallel; the
effective batch is per_device_batch_size x gradient_accumulation_steps x N).

The teacher is scored on the sft and val episodes before training
(``scoring.assess``: per criterion the source assignment and the status
calibration, the share of well-formed calls and the certainty score
against correctness, the measures of the phase 3 rewards), so its quality
is known: SFT copies its errors.

Outputs (``phase2_sft/``): ``samples.jsonl`` ({set: train / val, task:
update / score, qid, dataset, split, iteration, messages, completion}),
``teacher_scores.json``, ``checkpoints/`` (LoRA; removed at the start of
a run), ``adapter/`` and ``merged/`` (full weights, the RL start).
"""

import contextlib
import io
import json
import logging
import math
import shutil
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils.io_utils import read_jsonl, write_jsonl

from belief_model_component import UNCLEAN_REPLY

from .config import RunPaths, TrainingConfig
from .episodes import DocumentStore, Episode, replay_teacher
from .runs import check_same_settings, load_episodes, nugget_gold
from .scoring import assess, format_ok

logger = logging.getLogger(__name__)

# A final scorer sample is kept when its probability is within this of y
# (inclusive: "possible", 1/2, is kept for both y).
SCORE_AGREEMENT = 0.5


def chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """The prompt text of *messages* as the inference server builds it."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def load_student(path: str):
    """The student belief model at *path* in bf16, in its checkpoint's own
    class: a vision-language checkpoint (Qwen3.5) keeps its vision encoder,
    frozen and unused, so that the saved weights keep the layout vLLM loads."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    # Importing the model's module prints transformers' own docstring check
    # ("[ERROR] `loss` is part of ...'s signature, but not documented").
    with contextlib.redirect_stdout(io.StringIO()):
        if getattr(AutoConfig.from_pretrained(path), "vision_config", None) is None:
            return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    model.model.visual.requires_grad_(False)
    return model


def load_processor(path: str):
    """The tokenizer, or for a vision-language checkpoint the processor, saved
    next to the student's weights (``vllm serve`` needs the processor files)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path)


def tokenize_sample(tokenizer, messages: List[Dict[str, str]], completion: str) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(chat_prompt(tokenizer, messages), add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


class _Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def _clean(outcome: Dict) -> bool:
    """The reply has no issue of ``core.issues.UNCLEAN_REPLY``: every
    observation entered the belief or repeats one in it, no status was
    corrected by code, every reason opens with the candidate's current
    status (or "New:") and cites sources observed on its criterion, every
    removal names a listed candidate and was taken, and no one-answer
    criterion is left with competing candidates; so the reply is the
    belief it trains."""
    return not any(i.get("code") in UNCLEAN_REPLY for i in outcome.get("issues") or [])


def teacher_samples(episode: Episode, store: DocumentStore, require_verified_spans: bool,
                    counts: Counter) -> Tuple[List[Dict], Tuple]:
    """The SFT samples of one monitor episode and its replay
    ``(episode, calls, tracker)`` (teacher scores); sets its
    ``replay_error``."""
    calls, episode.replay_error, tracker = replay_teacher(episode, store.docs(episode))
    if episode.replay_error is not None:
        counts["replay_mismatch"] += 1
        logger.warning("[phase 2] %s %s: %s", episode.split.name, episode.qid, episode.replay_error)
    base = {"qid": episode.qid, "dataset": episode.dataset, "split": episode.split.split_id}
    out = []
    for c in calls:
        task = "score" if c.call.is_scorer else "update"
        prefix = "score_" if c.call.is_scorer else ""
        counts[f"{prefix}teacher_calls"] += 1
        if c.output is None:
            counts[f"{prefix}dropped_failed"] += 1
            continue
        if not format_ok(c.outcome, "stop"):
            counts[f"{prefix}dropped_gate"] += 1
            continue
        if c.call.is_scorer:
            if c.outcome["final_score"]["issues"]:
                counts["score_dropped_issues"] += 1
                continue
            if episode.correct is None or abs(c.outcome["certainty"] - float(episode.correct)) > SCORE_AGREEMENT:
                counts["score_dropped_disagree"] += 1
                continue
            counts[f"score_samples_y{int(episode.correct)}"] += 1
        elif require_verified_spans and not _clean(c.outcome):
            counts["dropped_unclean"] += 1
            continue
        out.append({**base, "task": task, "iteration": c.call.iteration, "messages": c.call.messages,
                    "completion": c.output})
    return out, (episode, calls, tracker)


def build_samples(cfg: TrainingConfig, paths: RunPaths) -> None:
    """``samples.jsonl`` (train: the sft split, val: the val split) and
    ``teacher_scores.json`` from the monitor runs of phase 1."""
    sc = cfg.sft
    check_same_settings(cfg, ["sft", "val"])
    store = DocumentStore()
    rows: List[Dict] = []
    episodes: List[Episode] = []
    replays: List[Tuple] = []
    for name, set_name in (("sft", "train"), ("val", "val")):
        counts: Counter = Counter()
        for episode in load_episodes(cfg, name, teacher=True, tag="[phase 2]"):
            samples, replay = teacher_samples(episode, store, sc.require_verified_spans, counts)
            counts["samples"] += len(samples)
            rows += [{"set": set_name, **r} for r in samples]
            episodes.append(episode)
            replays.append(replay)
        print(f"[phase 2] {name}: {dict(counts)}")
    write_jsonl(paths.sft_samples, rows)
    scores = assess(episodes, replays, nugget_gold(cfg))
    paths.teacher_scores_file.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"[phase 2] teacher on {len(episodes)} sft + val episodes: "
          f"{json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in scores.items()})}")
    print(f"[phase 2] wrote {len(rows)} samples to {paths.sft_samples}")


def load_samples(tokenizer, rows: List[Dict], max_length: int):
    """The samples *rows*, tokenized, and the number of those longer than
    *max_length*, dropped."""
    out: List[Dict[str, List[int]]] = []
    dropped = 0
    for r in rows:
        sample = tokenize_sample(tokenizer, r["messages"], r["completion"])
        if len(sample["input_ids"]) > max_length:
            dropped += 1
            continue
        out.append(sample)
    return out, dropped


def _describe(samples: List[Dict[str, List[int]]], dropped: int, max_length: int) -> str:
    if not samples:
        return f"0 ({dropped} longer than {max_length} tokens dropped)"
    lengths = [len(s["input_ids"]) for s in samples]
    replies = [sum(t != -100 for t in s["labels"]) for s in samples]
    return (f"{len(samples)} ({dropped} longer than {max_length} tokens dropped); tokens mean "
            f"{sum(lengths) / len(samples):.0f}, max {max(lengths)}; reply tokens mean "
            f"{sum(replies) / len(samples):.0f}, max {max(replies)}")


def _quiet_libraries() -> None:
    """Keep the log to the run summary, the progress bar and the losses:
    no weight-loading bar, no transformers notices (e.g. the torch fallback of
    the causal-conv1d kernel, the special-token alignment of the Trainer), no
    kernel compilation messages of fla / TileLang."""
    import transformers

    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    with contextlib.suppress(ImportError):
        import tilelang  # noqa: F401  (sets its own logger to INFO on import)
    for name in ("tilelang", "fla"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _completion_trainer(base):
    """*base* (``transformers.Trainer``) with the loss computed from the
    logits at the reply positions only."""

    class CompletionTrainer(base):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # The reply tokens of the whole accumulated batch (all devices) are
            # counted for a token-mean loss (num_items_in_batch).
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Position t predicts token t + 1: keep the positions whose next token is a reply token.
            targets = inputs["labels"][:, 1:]
            positions = targets.ne(-100).any(0).nonzero().squeeze(1)
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            logits_to_keep=positions, use_cache=False)
            targets = targets[:, positions]
            loss = F.cross_entropy(outputs.logits.flatten(0, 1).float(), targets.flatten(),
                                   ignore_index=-100, reduction="sum")
            if model.training and num_items_in_batch is not None:
                loss = loss / num_items_in_batch
                if self.args.average_tokens_across_devices:
                    loss = loss * self.accelerator.num_processes  # num_items_in_batch counts all devices
            else:
                loss = loss / targets.ne(-100).sum().clamp(min=1)
            return (loss, outputs) if return_outputs else loss

    return CompletionTrainer


def run_sft(cfg: TrainingConfig, paths: RunPaths) -> None:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise SystemExit(f"Phase 2 needs peft (pip install peft): {e}") from e
    from accelerate import PartialState
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    sc = cfg.sft
    state = PartialState()
    # The main process builds the samples, the others then read them.
    checkpoints = paths.sft_dir / "checkpoints"
    with state.main_process_first():
        if state.is_main_process:
            paths.sft_dir.mkdir(parents=True, exist_ok=True)
            # A run starts over: the Trainer's rotation would count the checkpoints of an earlier run.
            shutil.rmtree(checkpoints, ignore_errors=True)
            build_samples(cfg, paths)
        rows = read_jsonl(paths.sft_samples)
    tokenizer = AutoTokenizer.from_pretrained(sc.student_belief_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = cfg.shared.max_seq_len
    train, train_dropped = load_samples(tokenizer, [r for r in rows if r["set"] == "train"], max_len)
    val, val_dropped = load_samples(tokenizer, [r for r in rows if r["set"] == "val"], max_len)
    # Loss on the validation samples (selects the checkpoint).
    eval_sets = {"all": val} if val else None

    _quiet_libraries()
    model = load_student(sc.student_belief_model)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=sc.lora_r, lora_alpha=sc.lora_alpha, lora_dropout=sc.lora_dropout, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=list(sc.lora_target_modules),
    ))

    batch = sc.per_device_batch_size * sc.gradient_accumulation_steps * state.num_processes
    steps = math.ceil(len(train) / batch) * sc.num_epochs
    if state.is_main_process:
        trainable, total = model.get_nb_trainable_parameters()
        print("\n".join([
            "[phase 2] SFT",
            f"  student        {sc.student_belief_model}",
            f"  LoRA           r {sc.lora_r}, alpha {sc.lora_alpha}, dropout {sc.lora_dropout}; "
            f"trainable {trainable / 1e6:.1f}M of {total / 1e9:.2f}B parameters",
            f"  train samples  {_describe(train, train_dropped, max_len)}",
            f"  val samples    {_describe(val, val_dropped, max_len)}",
            f"  batch          {sc.per_device_batch_size} per device x {sc.gradient_accumulation_steps} accumulation "
            f"x {state.num_processes} GPU(s) = {batch}",
            f"  steps          {steps} ({sc.num_epochs} epochs; validation and checkpoint every epoch)",
            f"  optimizer      lr {sc.learning_rate:g}, warmup {sc.warmup_ratio:g} of the steps, seed {cfg.shared.seed}",
        ]), flush=True)

    args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=sc.num_epochs,
        per_device_train_batch_size=sc.per_device_batch_size,
        per_device_eval_batch_size=sc.per_device_batch_size,
        gradient_accumulation_steps=sc.gradient_accumulation_steps,
        learning_rate=sc.learning_rate,
        warmup_steps=sc.warmup_ratio,  # a value below 1 is a share of the steps
        logging_steps=sc.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_sets else "no",
        load_best_model_at_end=bool(eval_sets),
        metric_for_best_model="eval_all_loss" if eval_sets else None,
        greater_is_better=False,
        prediction_loss_only=True,
        save_total_limit=sc.num_epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=cfg.shared.seed,
    )
    trainer = _completion_trainer(Trainer)(
        model=model, args=args, train_dataset=train, eval_dataset=eval_sets,
        data_collator=_Collator(tokenizer.pad_token_id), processing_class=tokenizer,
    )
    trainer.train()
    if eval_sets:
        print(f"[phase 2] best checkpoint {trainer.state.best_model_checkpoint} "
              f"(validation loss {trainer.state.best_metric:.4f})")

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(str(paths.sft_dir / "adapter"))
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(paths.sft_merged))
        load_processor(sc.student_belief_model).save_pretrained(str(paths.sft_merged))
        print(f"[phase 2] merged model saved to {paths.sft_merged}")
