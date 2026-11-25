"""Rollouts of RL and validation: the student's belief chains on vLLM.

``VLLMRolloutEngine`` is the student on an in-process vLLM engine (as
Agent-BRACE's belief vLLM): it shares its GPU with the trainer (``sleep``
frees the GPU memory for the update, ``load_weights`` brings the engine
back with the updated weights).  The prompts are text only: the vision
encoder of a vision-language student (Qwen3.5) is not loaded
(``language_model_only``).

``run_chains`` runs many chains at once, each the inference belief model
(``chain.run_chain``) in its own thread with a ``SampledClient``: a
thread's call waits in a ``LockstepBatcher`` until every running chain is
waiting on a call, and then all those calls are generated in one vLLM
batch.  The log-prob of every sampled token is kept to check the trainer's
recomputed ``log pi_old`` against it.
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..shared.chain import CallRecord, ChainClient, ChainResult, run_chain
from ..shared.runs import DocumentStore, Episode
from ..shared.student import chat_prompt


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
            max_num_seqs=max_num_seqs, language_model_only=True, enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def prompt_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        return self.tokenizer(chat_prompt(self.tokenizer, messages), add_special_tokens=False)["input_ids"]

    def generate(self, prompts: List[List[Dict[str, str]]], *, temperature: float,
                 max_new_tokens: Union[int, Sequence[int]], n: int = 1) -> List[List[Generation]]:
        """*n* sampled replies per chat prompt within *max_new_tokens* (one
        budget for all, or one per prompt).  A prompt that leaves no room to
        generate in ``max_model_len`` gets *n* empty replies (finish reason
        ``prompt_too_long``)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        ids = [self.prompt_ids(m) for m in prompts]
        out = [[Generation(p, [], [], "", "prompt_too_long") for _ in range(n)] for p in ids]
        fits = [i for i, p in enumerate(ids) if len(p) < self.max_model_len]
        if not fits:
            return out
        budgets = [max_new_tokens] * len(prompts) if isinstance(max_new_tokens, int) else list(max_new_tokens)
        params = [SamplingParams(n=n, temperature=temperature, top_p=1.0, logprobs=0,
                                 max_tokens=min(budgets[i], self.max_model_len - len(ids[i])))
                  for i in fits]
        results = self.llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in fits],
                                    sampling_params=params, use_tqdm=False)
        for i, res in zip(fits, results):
            for j, seq in enumerate(sorted(res.outputs, key=lambda s: s.index)):
                tokens = [int(t) for t in seq.token_ids]
                lps = [float(step[tok].logprob) for step, tok in zip(seq.logprobs or [], tokens)]
                if len(lps) != len(tokens):
                    raise RuntimeError("vLLM returned no log-prob for some sampled tokens")
                out[i][j] = Generation(ids[i], tokens, lps, seq.text, seq.finish_reason)
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


class _Request:
    def __init__(self, messages: List[Dict[str, str]], max_tokens: int) -> None:
        self.messages = messages
        self.max_tokens = max_tokens
        self.result: Optional[Generation] = None
        self.error: Optional[BaseException] = None


class LockstepBatcher:
    """Gathers the calls of the running chains into one generation round
    (see the module docstring)."""

    def __init__(self, engine, temperature: float, workers: int) -> None:
        self.engine = engine
        self.temperature = temperature
        self._cv = threading.Condition()
        self._pending: List[_Request] = []
        self._active = workers

    def submit(self, messages: List[Dict[str, str]], max_tokens: int) -> Generation:
        req = _Request(messages, max_tokens)
        with self._cv:
            self._pending.append(req)
            self._cv.notify_all()
            while req.result is None and req.error is None:
                self._cv.wait()
        if req.error is not None:
            raise req.error
        return req.result

    def done(self) -> None:
        """A chain's thread ended."""
        with self._cv:
            self._active -= 1
            self._cv.notify_all()

    def serve(self) -> None:
        """Generate rounds until every chain has ended (the calling thread)."""
        while True:
            with self._cv:
                while self._active > 0 and len(self._pending) < self._active:
                    self._cv.wait()
                if not self._pending:
                    return
                batch, self._pending = self._pending, []
            try:
                gens = self.engine.generate([r.messages for r in batch], temperature=self.temperature,
                                            max_new_tokens=[r.max_tokens for r in batch])
                error = None
            except BaseException as e:  # noqa: BLE001 - handed to every waiting chain
                gens, error = None, e
            with self._cv:
                for k, r in enumerate(batch):
                    if error is not None:
                        r.error = error
                    else:
                        r.result = gens[k][0]
                self._cv.notify_all()


class SampledClient(ChainClient):
    """A chain's client whose replies are sampled through the batcher."""

    def __init__(self, batcher: LockstepBatcher) -> None:
        super().__init__()
        self._batcher = batcher

    def _reply(self, record: CallRecord, max_tokens: int) -> None:
        gen = self._batcher.submit(record.messages, max_tokens)
        record.reply = gen.text or ""
        record.prompt_ids = gen.prompt_ids
        record.response_ids = gen.response_ids
        record.logprobs = gen.logprobs
        record.finish_reason = gen.finish_reason


def run_chains(jobs: List[Tuple[Episode, str]], engine, store: DocumentStore, temperature: float) -> List[ChainResult]:
    """One chain per ``(episode, beliefs)`` of *jobs*, all at once (see the
    module docstring); in the order of *jobs*."""
    if not jobs:
        return []
    batcher = LockstepBatcher(engine, temperature, len(jobs))
    results: List[Optional[ChainResult]] = [None] * len(jobs)

    def work(k: int) -> None:
        episode, beliefs = jobs[k]
        try:
            results[k] = run_chain(episode, SampledClient(batcher), store, beliefs)
        finally:
            batcher.done()

    threads = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(len(jobs))]
    for t in threads:
        t.start()
    batcher.serve()
    for t in threads:
        t.join()
    return results


def extra_score_replies(chains: List[ChainResult], engine, n: int, temperature: float) -> List[List[CallRecord]]:
    """*n* more replies to each chain's final-score call (its first attempt's
    messages and budget), sampled together; none for a chain without that call."""
    from ..shared.chain import SCORE

    firsts = [next((c for c in r.calls if c.kind == SCORE and c.attempt == 1), None) for r in chains]
    asked = [k for k, c in enumerate(firsts) if c is not None]
    out: List[List[CallRecord]] = [[] for _ in chains]
    if n <= 0 or not asked:
        return out
    gens = engine.generate([firsts[k].messages for k in asked], temperature=temperature, n=n,
                           max_new_tokens=[firsts[k].max_tokens for k in asked])
    for k, group in zip(asked, gens):
        f = firsts[k]
        for g in group:
            out[k].append(CallRecord(SCORE, f.step, 1, f.messages, g.text or "", f.max_tokens,
                                     g.prompt_ids, g.response_ids, g.logprobs, g.finish_reason))
    return out
