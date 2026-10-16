"""Phase 2: SFT warm-up of the student belief updater (Agent-BRACE's
``finetune_belief_state_sft_alfworld.py``: completion-only loss, LoRA, then
merged into full weights for RL).

A sample is one teacher updater call (``phase1_trajectories/sft.jsonl``).
The input is the chat-template prompt the inference updater sends
(``add_generation_prompt``, native reasoning off, as served by vLLM),
followed by the teacher's reply and the end-of-turn token: the sequence the
model generates at inference.  The loss is on the reply and end-of-turn
token only.  Samples longer than ``max_length`` are dropped.

Multi-GPU: launch with ``torchrun --nproc_per_node N`` (data parallel; the
effective batch is per_device_batch_size x gradient_accumulation_steps x N).

Outputs (``phase2_sft/``): ``checkpoints/`` (LoRA), ``adapter/`` and
``merged/`` (full weights, the RL start).
"""

import json
import logging
import random
from typing import Any, Dict, List

import torch

from .config import RunPaths, TrainingConfig

logger = logging.getLogger(__name__)


def chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """The prompt text of *messages* as the inference server builds it."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


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


def run_sft(cfg: TrainingConfig, paths: RunPaths) -> None:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise SystemExit(f"Phase 2 needs peft (pip install peft): {e}") from e
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    sc = cfg.sft
    if not paths.sft_file.exists():
        raise FileNotFoundError(f"{paths.sft_file} not found; run phase 1 first")
    tokenizer = AutoTokenizer.from_pretrained(sc.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(line) for line in paths.sft_file.read_text().splitlines() if line.strip()]
    data = [tokenize_sample(tokenizer, r["messages"], r["completion"]) for r in rows]
    kept = [d for d in data if len(d["input_ids"]) <= sc.max_length]
    random.Random(cfg.shared.seed).shuffle(kept)
    print(f"[phase 2] {len(kept)} SFT samples ({len(data) - len(kept)} longer than {sc.max_length} tokens dropped); "
          f"student {sc.student_model}")

    model = AutoModelForCausalLM.from_pretrained(sc.student_model, dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=sc.lora_r, lora_alpha=sc.lora_alpha, lora_dropout=sc.lora_dropout, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=list(sc.lora_target_modules),
    ))

    args = TrainingArguments(
        output_dir=str(paths.sft_dir / "checkpoints"),
        num_train_epochs=sc.num_epochs,
        per_device_train_batch_size=sc.per_device_batch_size,
        gradient_accumulation_steps=sc.gradient_accumulation_steps,
        learning_rate=sc.learning_rate,
        warmup_ratio=sc.warmup_ratio,
        logging_steps=sc.logging_steps,
        save_steps=sc.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=cfg.shared.seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=kept,
        data_collator=_Collator(tokenizer.pad_token_id), processing_class=tokenizer,
    )
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(str(paths.sft_dir / "adapter"))
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(paths.sft_merged))
        tokenizer.save_pretrained(str(paths.sft_merged))
        print(f"[phase 2] merged model saved to {paths.sft_merged}")
