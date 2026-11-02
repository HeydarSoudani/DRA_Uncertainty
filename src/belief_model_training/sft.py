"""Phase 2: SFT warm-up of the student belief model (Agent-BRACE's
``finetune_belief_state_sft_alfworld.py``: completion-only loss, LoRA, then
merged into full weights for RL).

The samples come from the monitor runs of phase 1 on the ``sft`` (training)
and ``val`` (validation) splits (``build_samples``, on the main process,
written to ``samples.jsonl``).  Only queries with gold criteria are used
(``runs.load_episodes``).  A sample is one call: an init call (the
criteria extraction prompt with the query's gold criteria as the target,
``episodes.teacher_init_call``; only when they parse back) or an updater
call (the teacher's replies on the gold criteria replayed,
``episodes.replay_teacher``, the final call on the answer turn included;
each call within the matched replay that parsed and, with
``require_verified_spans``, whose cited spans all occur in their passage
or the agent's turn), in one mixed set, unweighted.
The input is the chat-template prompt the inference belief model sends
(``add_generation_prompt``, native reasoning off, as served by vLLM),
followed by the teacher's reply and the end-of-turn token: the sequence the
model generates at inference.  The loss is on the reply and end-of-turn
token only, averaged over the reply tokens of the batch (token-mean); the
logits are computed at the reply positions only (a full-vocabulary logit
row per prompt token would not fit in memory).
Samples longer than ``shared.max_seq_len`` are dropped.

Validation (the ``val`` samples): the loss on all samples and on the init and update samples separately, after
every epoch; a checkpoint is saved every epoch and the one with the lowest
validation loss is merged (``num_epochs`` is an upper bound).  Without
validation samples the last epoch is merged.

Multi-GPU: launch with ``torchrun --nproc_per_node N`` (data parallel; the
effective batch is per_device_batch_size x gradient_accumulation_steps x N).

The teacher is scored on the sft and val episodes before training
(``scoring.assess``: coverage against ``recall_so_far``, its answer
candidates against the gold intermediate answers, the final confidence
against correctness),
so its quality is known: SFT copies its errors.

Outputs (``phase2_sft/``): ``samples.jsonl`` ({set: train / val, qid,
dataset, split, call, iteration, messages, completion}),
``teacher_scores.json``, ``checkpoints/`` (LoRA; removed at the start of
a run), ``adapter/`` and ``merged/`` (full weights, the RL start).
"""

import contextlib
import io
import json
import logging
import math
import random
import shutil
from collections import Counter
from typing import Dict, List

import torch
import torch.nn.functional as F

from utils.io_utils import read_jsonl, write_jsonl

from .config import RunPaths, TrainingConfig
from .episodes import DocumentStore, Episode, replay_teacher, teacher_init_call
from belief_model_component.criteria import GOLD_CRITERIA_MODEL

from .runs import candidate_gold, check_same_settings, load_episodes
from .scoring import assess

logger = logging.getLogger(__name__)


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


def teacher_samples(episode: Episode, store: DocumentStore, teacher: str, require_verified_spans: bool,
                    counts: Counter) -> List[Dict]:
    """The SFT samples of one monitor episode; sets its ``replay_steps``,
    ``replay_error`` and ``shown_doc_ids`` (teacher scores)."""
    calls, episode.replay_steps, episode.replay_error = replay_teacher(episode, store.docs(episode))
    if episode.replay_error is not None:
        counts["replay_mismatch"] += 1
        logger.warning("[phase 2] %s %s: %s", episode.split.name, episode.qid, episode.replay_error)
    base = {"qid": episode.qid, "dataset": episode.dataset, "split": episode.split.split_id}
    out = []
    if episode.criteria_model not in (teacher, GOLD_CRITERIA_MODEL):
        counts["init_not_teacher"] += 1
    else:
        init = teacher_init_call(episode)
        if init["verified"]:
            out.append({**base, "call": "init", "iteration": 0, "messages": init["messages"],
                        "completion": init["output"]})
        else:
            counts["init_not_reproduced"] += 1
    for call in calls:
        counts["teacher_calls"] += 1
        if not call["parsed"]:
            counts["dropped_unparsed"] += 1
            continue
        if require_verified_spans and not call["spans_verified"]:
            counts["dropped_unverified_spans"] += 1
            continue
        out.append({**base, "call": "update", "iteration": call["iteration"], "messages": call["messages"],
                    "completion": call["output"]})
    return out


def build_samples(cfg: TrainingConfig, paths: RunPaths) -> None:
    """``samples.jsonl`` (train: the sft split, val: the val split) and
    ``teacher_scores.json`` from the monitor runs of phase 1."""
    sc = cfg.sft
    teacher = cfg.trajectories.teacher_belief_model
    check_same_settings(cfg, ["sft", "val"])
    store = DocumentStore()
    rows: List[Dict] = []
    episodes: List[Episode] = []
    for name, set_name in (("sft", "train"), ("val", "val")):
        counts: Counter = Counter()
        for episode in load_episodes(cfg, name, teacher=True, tag="[phase 2]"):
            samples = teacher_samples(episode, store, teacher, sc.require_verified_spans, counts)
            counts.update(f"{r['call']}_samples" for r in samples)
            rows += [{"set": set_name, **r} for r in samples]
            episodes.append(episode)
        print(f"[phase 2] {name}: {dict(counts)}")
    write_jsonl(paths.sft_samples, rows)
    scores = assess(episodes, candidate_gold(cfg))
    paths.teacher_scores_file.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"[phase 2] teacher on {len(episodes)} sft + val episodes: "
          f"{json.dumps({k: round(v, 3) for k, v in scores.items()})}")
    print(f"[phase 2] wrote {len(rows)} samples to {paths.sft_samples}")


def load_samples(tokenizer, rows: List[Dict], max_length: int):
    """The samples *rows* by call (``init`` / ``update``), tokenized, and the
    number of those longer than *max_length*, dropped."""
    out: Dict[str, List[Dict[str, List[int]]]] = {"init": [], "update": []}
    dropped = 0
    for r in rows:
        sample = tokenize_sample(tokenizer, r["messages"], r["completion"])
        if len(sample["input_ids"]) > max_length:
            dropped += 1
            continue
        out[r.get("call", "update")].append(sample)
    return out, dropped


def _describe(samples: Dict[str, List[Dict[str, List[int]]]], dropped: int, max_length: int) -> str:
    kept = samples["init"] + samples["update"]
    if not kept:
        return f"0 ({dropped} longer than {max_length} tokens dropped)"
    lengths = [len(s["input_ids"]) for s in kept]
    replies = [sum(t != -100 for t in s["labels"]) for s in kept]
    return (f"{len(kept)} ({len(samples['init'])} init, {len(samples['update'])} update; "
            f"{dropped} longer than {max_length} tokens dropped); tokens mean {sum(lengths) / len(kept):.0f}, "
            f"max {max(lengths)}; reply tokens mean {sum(replies) / len(kept):.0f}, max {max(replies)}")


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
    kept = train["init"] + train["update"]
    random.Random(cfg.shared.seed).shuffle(kept)
    val, val_dropped = load_samples(tokenizer, [r for r in rows if r["set"] == "val"], max_len)
    # Loss on all validation samples (selects the checkpoint) and per call.
    eval_sets = {"all": val["init"] + val["update"], **{c: v for c, v in val.items() if v}} if any(val.values()) else None

    _quiet_libraries()
    model = load_student(sc.student_belief_model)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=sc.lora_r, lora_alpha=sc.lora_alpha, lora_dropout=sc.lora_dropout, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=list(sc.lora_target_modules),
    ))

    batch = sc.per_device_batch_size * sc.gradient_accumulation_steps * state.num_processes
    steps = math.ceil(len(kept) / batch) * sc.num_epochs
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
        model=model, args=args, train_dataset=kept, eval_dataset=eval_sets,
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
