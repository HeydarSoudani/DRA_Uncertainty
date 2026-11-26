"""The student model: its chat prompt, weights and tokenization.

The prompt of a call is the model's chat template over the belief model's
messages with ``add_generation_prompt`` and native reasoning off, as the
inference server builds it (``reasoner_component.disable_native_thinking``:
``chat_template_kwargs.enable_thinking`` false).  A reply is followed by the
end-of-turn token: the sequence the model generates at inference.

A vision-language checkpoint (Qwen3.5) is loaded in its own class with its
vision encoder frozen and unused, so the saved weights keep the layout vLLM
loads (``--language-model-only``)."""

import contextlib
import io
import logging
from typing import Dict, List

import torch


def chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """The prompt text of *messages* as the inference server builds it."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def text_tokenizer(tokenizer):
    """The text tokenizer of a processor (a vision-language checkpoint's)."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def tokenize_sample(tokenizer, messages: List[Dict[str, str]], completion: str) -> Dict[str, List[int]]:
    """Input ids of prompt + reply + end of turn, labels on the reply and
    end of turn only."""
    tok = text_tokenizer(tokenizer)
    prompt_ids = tok(chat_prompt(tokenizer, messages), add_special_tokens=False)["input_ids"]
    completion_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": prompt_ids + completion_ids, "labels": [-100] * len(prompt_ids) + completion_ids}


def load_student(path: str):
    """The student at *path* in bf16 (see the module docstring)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    # Importing the model's module prints transformers' own docstring checks.
    with contextlib.redirect_stdout(io.StringIO()):
        if getattr(AutoConfig.from_pretrained(path), "vision_config", None) is None:
            return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    model.model.visual.requires_grad_(False)
    return model


def load_processor(path: str):
    """The tokenizer, or the processor of a vision-language checkpoint,
    saved next to the weights (``vllm serve`` needs its files)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path)


def quiet_libraries() -> None:
    """No weight-loading bars, transformers notices or kernel messages."""
    import transformers

    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    with contextlib.suppress(ImportError):
        import tilelang  # noqa: F401  (sets its own logger to INFO on import)
    for name in ("tilelang", "fla"):
        logging.getLogger(name).setLevel(logging.WARNING)
