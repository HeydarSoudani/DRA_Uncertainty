"""Walkthrough: running a reasoner on a self-hosted vLLM server.

  registry  reasoner_component/vllm_server/registry.py   how each model is served
  server    reasoner_component/vllm_server/process.py    starting / stopping it
  client    reasoner_component/vllm.py                   VLLMGenerator

Run on a GPU node:  python experiments/inference/vllm_example.py
"""

import asyncio
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel

from reasoner_component import VLLMGenerator, create_generator
from reasoner_component.vllm_server import VLLMServerManager, require_spec, serve
from reasoner_component.vllm_server.process import visible_gpus

# ── 0. Pick the model ───────────────────────────────────────────────────────
# Must be in registry.py (python -m reasoner_component.vllm_server list).

MODEL = "vllm/Qwen/Qwen3.5-4B"

MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": "What is the capital of the Netherlands?"},
]


# ── 1. Start the server ─────────────────────────────────────────────────────

@contextmanager
def run_server(model: str) -> Iterator[str]:
    """Serve *model* for the ``with`` block (waits until it is up); yields its URL."""
    spec = require_spec(model)
    with serve(spec) as api_base:
        print(f"[example] {spec.label} is up at {api_base}")
        yield api_base


# ── 2. Build the client ─────────────────────────────────────────────────────

def build_client(api_base: str) -> VLLMGenerator:
    """Defaults for every call; each call can override them."""
    return create_generator(MODEL, api_base=api_base, temperature=0.0, max_tokens=2000)


# ── 3. Use it ───────────────────────────────────────────────────────────────

def complete(generator: VLLMGenerator) -> str:
    # strip_think=False would keep the reasoning as <think>...</think>
    answer = generator.complete(MESSAGES)
    print(f"[complete] {answer}")
    return answer


async def acomplete_many(generator: VLLMGenerator) -> List[str]:
    # All sent at once: the server batches them (up to max_num_seqs).
    questions = ["What is the capital of France?", "Who wrote Hamlet?", "What is 17 * 23?"]
    answers = await asyncio.gather(
        *(generator.acomplete([{"role": "user", "content": q}]) for q in questions)
    )
    for q, a in zip(questions, answers):
        print(f"[acomplete] {q} -> {a}")
    return answers


class Answer(BaseModel):
    answer: str
    confidence: float


def structured_output(generator: VLLMGenerator) -> Answer:
    # The server constrains the reply to Answer's JSON schema.
    messages = [{"role": "user", "content": "What is the capital of the Netherlands? "
                 "Answer in JSON with your answer and your confidence (0 to 1)."}]
    result = generator.complete_with_structured_output(messages, Answer)
    print(f"[structured] {result.answer} (confidence {result.confidence})")
    return result


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def tool_calls(generator: VLLMGenerator) -> List[Dict[str, Any]]:
    """Needs a server with a tool parser (spec.tool_parser): an agent model,
    e.g. MODEL = "vllm/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B".  Qwen3.5-4B has none."""
    messages = [{"role": "user", "content": "Who won the 2022 Nobel Prize in Physics? Search for it."}]
    text, calls = generator.complete_with_tools(messages, [SEARCH_TOOL])
    # calls: [{"name": "search", "arguments": {"query": ...}}]
    print(f"[tools] text: {text!r}, calls: {calls}")
    return calls


def stop_on_pattern(generator: VLLMGenerator) -> Tuple[str, Optional[str]]:
    """Streams the reply and stops it at the first regex match: how a text-based
    agent hands control back after writing a search."""
    messages = [{"role": "user", "content": "Who won the 2022 Nobel Prize in Physics? To search, "
                 "write <search>your query</search> and wait for the results."}]
    text, match = generator.complete_with_stop_on_pattern(messages, [r"<search>.*?</search>"])
    # match: the matched text (None: the reply ended without one); text: what came before it
    print(f"[stop] match: {match!r}, text before: {text[-200:]!r}")
    return text, match


def disable_thinking(generator: VLLMGenerator) -> str:
    """Switches the model's reasoning off for every later call of *generator*.
    For one call only: complete(..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})."""
    generator.disable_thinking()
    answer = generator.complete(MESSAGES, strip_think=False)  # no <think> block to keep
    print(f"[no thinking] {answer}")
    return answer


# ── 4. Inspect and clean up ─────────────────────────────────────────────────

def cleanup(generator: VLLMGenerator) -> None:
    """Prints the token usage of every call so far and closes the client's
    connections; the server itself stops when run_server's block ends."""
    print(f"[tokens] {generator.token_meter.snapshot()}")
    generator.cleanup()


def main():
    ### ----------------------------------------------------------
    # ── Option 1: `with` block, the server stops on leaving it ──
    with run_server(MODEL) as api_base:        # 1
        generator = build_client(api_base)     # 2
        complete(generator)                    # 3
        asyncio.run(acomplete_many(generator))
        structured_output(generator)
        # tool_calls(generator)               # needs an agent model, see its docstring
        stop_on_pattern(generator)
        disable_thinking(generator)
        cleanup(generator)                     # 4
    # No separate stop call is needed: the server stops when the with run_server(...) block ends.
    print("[example] server stopped")          # 5


    ### ------------------------------------
    # ── Option 2: start and stop by hand ──
    spec = require_spec(MODEL)
    manager = VLLMServerManager()
    manager.start([spec], total_gpus=len(visible_gpus()))   # 1: starts, waits until up
    try:
        generator = build_client(spec.api_base)             # 2
        complete(generator)                                 # 3
        cleanup(generator)                                  # 4
    finally:
        manager.shutdown()                                  # 5: also on an error


if __name__ == "__main__":
    main()
