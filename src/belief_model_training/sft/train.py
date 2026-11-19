"""Phase 3: SFT of the student on the teacher's replies (Agent-BRACE's belief
SFT: completion-only loss, LoRA, then merged into full weights).

The main process builds the samples (``sft.samples``), the others read
them.  The input is the chat-template prompt the inference belief model
sends, then the reply and the end-of-turn token; the loss is on the reply
and end-of-turn token only, averaged over the reply tokens of the batch
(token-mean); the logits are computed at the reply positions only.
Samples longer than ``shared.max_seq_len`` are dropped.

Validation loss (the val set) after every epoch; a checkpoint every
epoch, and the one with the lowest validation loss is merged into
``sft/final`` (the served ``qwen3.5_4b_sft`` and the RL start).  Without
validation samples the last epoch is merged.

Multi-GPU: ``torchrun --nproc_per_node N`` (data parallel; the effective
batch is per_device_batch_size x gradient_accumulation_steps x N).

Outputs (``sft/``): ``samples.jsonl``, ``checkpoints/`` (LoRA; removed at
the start of a run), ``adapter/``, ``final/``.
"""

import math
import shutil
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils.io_utils import read_jsonl

from ..config import RunPaths, TrainingConfig
from ..student import load_processor, load_student, quiet_libraries, tokenize_sample
from .samples import build_samples


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


def load_samples(tokenizer, rows: List[Dict], max_length: int) -> Tuple[List[Dict[str, List[int]]], int]:
    """The samples *rows*, tokenized, and the number longer than *max_length* (dropped)."""
    out, dropped = [], 0
    for r in rows:
        sample = tokenize_sample(tokenizer, r["messages"], r["completion"])
        if len(sample["input_ids"]) > max_length:
            dropped += 1
            continue
        out.append(sample)
    return out, dropped


def describe(samples: List[Dict[str, List[int]]], dropped: int, max_length: int) -> str:
    if not samples:
        return f"0 ({dropped} longer than {max_length} tokens dropped)"
    lengths = [len(s["input_ids"]) for s in samples]
    replies = [sum(t != -100 for t in s["labels"]) for s in samples]
    return (f"{len(samples)} ({dropped} longer than {max_length} tokens dropped); tokens mean "
            f"{sum(lengths) / len(samples):.0f}, max {max(lengths)}; reply tokens mean "
            f"{sum(replies) / len(samples):.0f}, max {max(replies)}")


def _completion_trainer(base):
    """*base* (``transformers.Trainer``) with the loss computed from the
    logits at the reply positions only."""

    class CompletionTrainer(base):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # The reply tokens of the whole accumulated batch (all devices)
            # are counted for a token-mean loss (num_items_in_batch).
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


def run_sft(cfg: TrainingConfig) -> None:
    from accelerate import PartialState
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments

    sc, sh = cfg.sft, cfg.shared
    paths = RunPaths()
    state = PartialState()
    checkpoints = paths.sft_dir / "checkpoints"
    with state.main_process_first():
        if state.is_main_process:
            paths.sft_dir.mkdir(parents=True, exist_ok=True)
            # A run starts over: the Trainer's rotation would count an earlier run's checkpoints.
            shutil.rmtree(checkpoints, ignore_errors=True)
            build_samples(cfg, paths)
        rows = read_jsonl(paths.sft_samples)
    processor = load_processor(sh.student)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = sh.max_seq_len
    train, train_dropped = load_samples(processor, [r for r in rows if r["set"] == "train"], max_len)
    val, val_dropped = load_samples(processor, [r for r in rows if r["set"] == "val"], max_len)
    if not train:
        raise SystemExit(f"[phase 3] no train sample in {paths.sft_samples}: run phase 2 on the sft split first")
    eval_sets = {"all": val} if val else None

    quiet_libraries()
    model = load_student(sh.student)
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
            "[phase 3] SFT",
            f"  student        {sh.student}",
            f"  LoRA           r {sc.lora_r}, alpha {sc.lora_alpha}, dropout {sc.lora_dropout}; "
            f"trainable {trainable / 1e6:.1f}M of {total / 1e9:.2f}B parameters",
            f"  train samples  {describe(train, train_dropped, max_len)}",
            f"  val samples    {describe(val, val_dropped, max_len)}",
            f"  batch          {sc.per_device_batch_size} per device x {sc.gradient_accumulation_steps} "
            f"accumulation x {state.num_processes} GPU(s) = {batch}",
            f"  steps          {steps} ({sc.num_epochs} epochs; validation and checkpoint every epoch)",
            f"  optimizer      lr {sc.learning_rate:g}, warmup {sc.warmup_ratio:g} of the steps, seed {sh.seed}",
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
        seed=sh.seed,
    )
    trainer = _completion_trainer(Trainer)(
        model=model, args=args, train_dataset=train, eval_dataset=eval_sets,
        data_collator=_Collator(tokenizer.pad_token_id), processing_class=tokenizer,
    )
    trainer.train()
    if eval_sets:
        print(f"[phase 3] best checkpoint {trainer.state.best_model_checkpoint} "
              f"(validation loss {trainer.state.best_metric:.4f})")

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(str(paths.sft_dir / "adapter"))
        merged = trainer.model.merge_and_unload()
        shutil.rmtree(paths.sft_final, ignore_errors=True)
        merged.save_pretrained(str(paths.sft_final))
        processor.save_pretrained(str(paths.sft_final))
        print(f"[phase 3] merged model saved to {paths.sft_final} (served as qwen3.5_4b_sft)")
