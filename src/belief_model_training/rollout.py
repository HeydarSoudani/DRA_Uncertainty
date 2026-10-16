"""vLLM generation of the student belief updater during RL.

As Agent-BRACE's belief vLLM: batched sampling (temperature 1.0) that keeps
``log pi_old`` of every sampled token for PPO.  The engine shares its GPU
with the trainer: ``sleep`` frees the GPU memory for the update and
``load_weights`` brings the engine back with the updated weights
(vLLM sleep mode and ``reload_weights``, in place of Agent-BRACE's rebuild
of the engine).

Prompts are built with the model's chat template exactly as the inference
server builds them (``sft.chat_prompt``).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .sft import chat_prompt

logger = logging.getLogger(__name__)


@dataclass
class Generation:
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float]
    text: Optional[str]
    finish_reason: Optional[str]


class VLLMRolloutEngine:
    def __init__(self, model_path: str, *, gpu_memory_utilization: float, max_model_len: int,
                 max_num_seqs: int, seed: int = 0) -> None:
        from vllm import LLM

        self.max_model_len = max_model_len
        self.llm = LLM(
            model=model_path, dtype="bfloat16", enable_sleep_mode=True, seed=seed,
            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: List[List[Dict[str, str]]], *, temperature: float,
                 max_new_tokens: int) -> List[Generation]:
        """One sampled reply per chat prompt.  A prompt that leaves no room
        to generate in ``max_model_len`` gets no reply (``text`` None)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompt_ids = [self.tokenizer(chat_prompt(self.tokenizer, m), add_special_tokens=False)["input_ids"]
                      for m in prompts]
        fits = [i for i, p in enumerate(prompt_ids) if len(p) < self.max_model_len]
        out: List[Generation] = [Generation(p, [], [], None, "prompt_too_long") for p in prompt_ids]
        if not fits:
            return out
        params = [SamplingParams(temperature=temperature, top_p=1.0, logprobs=0,
                                 max_tokens=min(max_new_tokens, self.max_model_len - len(prompt_ids[i])))
                  for i in fits]
        results = self.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[i]) for i in fits],
                                    sampling_params=params, use_tqdm=False)
        for i, res in zip(fits, results):
            seq = res.outputs[0]
            ids = [int(t) for t in seq.token_ids]
            lps = [float(step[tok].logprob) for step, tok in zip(seq.logprobs or [], ids)]
            if len(lps) != len(ids):
                raise RuntimeError("vLLM returned no log-prob for some sampled tokens")
            out[i] = Generation(prompt_ids[i], ids, lps, seq.text, seq.finish_reason)
        return out

    def sleep(self) -> None:
        """Free the engine's GPU memory (weights and KV cache are discarded)."""
        self.llm.sleep(level=2)

    def load_weights(self, weights_dir: str) -> None:
        """Wake the engine with the weights saved in *weights_dir*."""
        self.llm.wake_up(tags=["weights"])
        self.llm.collective_rpc("reload_weights", kwargs={"weights_path": weights_dir})
        self.llm.wake_up(tags=["kv_cache"])
        self.llm.reset_prefix_cache()
