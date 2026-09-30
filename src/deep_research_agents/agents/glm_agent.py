"""GLM agent: faithful port of the AgentIR GLM client.

Adapted from AgentIR/evaluation/search_agent/glm_client.py
(run_conversation_with_tools) to the pipeline's BasicAgent interface.

LLM calls  : OpenAI Chat Completions API → vLLM server
             (Responses API not used because vLLM's --tool-call-parser glm47
              only works with the Chat Completions endpoint.)
Retrieval  : pipeline local retriever (self.retrieve_documents)
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai

from reasoner_component import REASONING_FALLBACK_PREFIX

from .base_agent import BasicAgent
from deep_research_agents.prompts.glm.user import QUERY_TEMPLATE
from deep_research_agents.prompts.answer_prompts import (
    FINAL_ANSWER_INSTRUCTION,
    OSS_FORMAT,
)
from utils.config import InferenceConfig

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "glm"
GLM_SYSTEM_PROMPT = (_PROMPT_DIR / "system.txt").read_text()


def _parse_tool_calls_from_text(text: str, defined_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse <tool_call> XML tags from reasoning content (GLM-specific quirk).

    GLM sometimes emits tool calls inside reasoning content rather than in the
    standard tool_calls field. This function extracts them.

    Returns tool calls in Chat Completions format (for use in assistant message).
    """

    def _get_argument_type(func_name: str, arg_key: str) -> str:
        for t in defined_tools:
            func = t.get("function", {})
            if func.get("name") == func_name:
                props = func.get("parameters", {}).get("properties", {})
                if arg_key in props:
                    return props[arg_key].get("type", "string")
        return "string"

    tool_calls = []
    tool_call_strs = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    for call in tool_call_strs:
        func_name_match = re.match(r"([^\n<]+)", call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        if func_name:
            pairs = re.findall(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                call,
                re.DOTALL,
            )
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = _get_argument_type(func_name, arg_key)
                if arg_type != "string":
                    try:
                        arg_value = json.loads(arg_value)
                    except Exception:
                        pass
                arguments[arg_key] = arg_value

            call_id = "tool-call-" + str(uuid.uuid4())
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            })
    return tool_calls


class GLM_Agent(BasicAgent):
    """GLM deep research agent using the OpenAI Chat Completions API."""

    AGENT_NAME = "GLM"

    def __init__(self, llm_client=None, retriever=None, max_iteration: int = 100, seen_top_k: int = 5, model_url: Optional[str] = None, model_name: str = "zai-org/GLM-4.7-Flash", max_output_tokens: int = 20000, system_prompt: Optional[str] = None, verbose: bool = True, api_key: Optional[str] = None, **kwargs) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)

        self.model_url = model_url or os.getenv(
            "GLM_API_BASE", "http://localhost:6008/v1"
        )
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        self._api_key = api_key or os.getenv("GLM_API_KEY", "EMPTY")

        # OpenRouter omits reasoning tokens unless they are explicitly requested,
        # and returns them under ``reasoning`` rather than vLLM's
        # ``reasoning_content`` (handled in ``_read_reasoning``).  Opt in so the
        # think block is emitted; vLLM ignores this extra field. Spread into
        # every chat.completions.create() call via ``_chat_sampling_kwargs``.
        if "openrouter.ai" in (self.model_url or ""):
            self._sampling = {"extra_body": {"reasoning": {"enabled": True}}}

        # Per-call output caps.
        self._max_tokens_per_call = 4096
        self._intermediate_answer_max_tokens = 1024

        from utils.token_meter import TokenMeter
        self.token_meter = TokenMeter()

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            model_name=model_name,
            api_base=self.model_url,
            api_key=self._api_key,
            max_output_tokens=max_output_tokens,
            reasoning_effort=None,
            system_prompt=GLM_SYSTEM_PROMPT,
            format_instructions=OSS_FORMAT,
        )

    # _get_tool_definitions() and _format_search_results() inherited from BasicAgent

    def _read_reasoning(self, message: Any) -> Optional[str]:
        """Extract the reasoning/think text from an assistant message.

        Provider-agnostic: vLLM exposes the think block as
        ``reasoning_content``; OpenRouter uses ``reasoning`` (a plain string)
        and ``reasoning_details`` (structured/encrypted blocks).  We read
        whichever is present so the trajectory captures reasoning on both
        backends.
        """
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        if reasoning:
            return reasoning

        details = getattr(message, "reasoning_details", None)
        if details:
            parts: List[str] = []
            for d in details:
                if isinstance(d, dict):
                    text = d.get("text") or d.get("summary")
                else:
                    text = getattr(d, "text", None) or getattr(d, "summary", None)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
        return None

    # ── Intermediate answer (Chat Completions, same client as main loop) ────

    def answer_from_trajectory(self, original_query: str, trajectory: Any, instruction: str) -> str:
        """Same direct openai client as the main loop, within the output budget."""
        cfg = self.inference_config
        messages = self._intermediate_answer_messages(trajectory, instruction)
        remaining_tokens = max(cfg.max_output_tokens - getattr(self, "_cumulative_output_tokens", 0), 1024)
        client = openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=messages,
            max_tokens=min(remaining_tokens, self._intermediate_answer_max_tokens),
            **self._chat_sampling_kwargs(),
        )
        message = response.choices[0].message
        raw = message.content or ""
        reasoning = self._read_reasoning(message)
        if not raw.strip() and reasoning:
            return REASONING_FALLBACK_PREFIX + reasoning
        return raw

    # _force_answer_chat_in_conversation() and _force_answer_chat_compressed()
    # inherited from BasicAgent

    # run_single() inherited from BasicAgent (handles 3-value inference return)

    # ── Main inference loop ───────────────────────────────────────────────────

    def inference(self, query: str, generation_temp: float = 0.7) -> Tuple[List[Dict[str, Any]], str, int]:
        """GLM Chat Completions inference loop.

        Faithful to AgentIR/evaluation/search_agent/glm_client.py
        (run_conversation_with_tools), using vLLM's Chat Completions API
        with tool calling (--tool-call-parser glm47).

        Returns:
            reasoning_path  : list of per-step dicts (pipeline-compatible)
            prediction      : extracted answer string
            num_iterations  : number of LLM iterations used
        """
        cfg = self.inference_config
        client = openai.OpenAI(
            base_url=cfg.api_base,
            api_key=cfg.api_key,
        )

        formatted_query = QUERY_TEMPLATE.format(Question=query)
        messages: List[Any] = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": formatted_query},
        ]
        tools = self._get_tool_definitions()

        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""
        self._cumulative_output_tokens = 0

        self._print(f"Query: {query}")

        self._reasoning_only_retries = 0
        iteration = 1
        while iteration <= self.max_iteration:
            remaining_tokens = min(self.max_output_tokens - self._cumulative_output_tokens, self._max_tokens_per_call)
            if remaining_tokens <= 0:
                logger.info("Global output token budget exhausted, forcing final answer")
                self._print("Token budget exhausted, forcing final answer in conversation")
                forced_text = self._force_answer_chat_in_conversation(messages)
                if not forced_text:
                    self._print("Retrying with compressed evidence fallback")
                    forced_text = self._force_answer_chat_compressed(
                        query, reasoning_path,
                    )
                if forced_text:
                    prediction = forced_text
                    reasoning_path.append({
                        "action_type": "context_limit",
                        "think": FINAL_ANSWER_INSTRUCTION,
                        "generation": forced_text,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                break
            try:
                # `generation_temp` (the pipeline-wide default, often 0.0) is
                # intentionally NOT forwarded — greedy decoding degrades some
                # reasoning ("thinking") models. Instead, per-agent sampling
                # overrides (see `_sampling`) carry the model card's official
                # params. For GLM this is empty, so the server's
                # generation_config decides.
                response = client.chat.completions.create(
                    model=cfg.model_name,
                    messages=messages,
                    tools=tools,
                    max_tokens=remaining_tokens,
                    **self._chat_sampling_kwargs(),
                )
            except Exception as e:
                logger.warning(f"Iteration {iteration} API error: {e}")
                self._print("Context limit hit, forcing final answer in conversation")
                forced_text = self._force_answer_chat_in_conversation(messages)
                if not forced_text:
                    self._print("Retrying with compressed evidence fallback")
                    forced_text = self._force_answer_chat_compressed(
                        query, reasoning_path,
                    )
                if forced_text:
                    prediction = forced_text
                    reasoning_path.append({
                        "action_type": "context_limit",
                        "think": FINAL_ANSWER_INSTRUCTION,
                        "generation": forced_text,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                break

            if response.usage:
                self._cumulative_output_tokens += response.usage.completion_tokens or 0
            self.token_meter.record_usage(getattr(response, "usage", None))

            message = response.choices[0].message
            cur_reasoning = self._read_reasoning(message)
            cur_text = message.content
            official_tool_calls = message.tool_calls or []

            # Diagnostic: set DRA_DEBUG_RAW=1 to dump the raw server response per
            # iteration (reasoning vs content vs tool_calls vs finish_reason).
            # Off by default → no effect on normal runs or other agents.
            if os.getenv("DRA_DEBUG_RAW"):
                _raw_rc = self._read_reasoning(message)
                _fr = response.choices[0].finish_reason
                _ct = response.usage.completion_tokens if response.usage else "?"
                print(
                    f"[RAW iter {iteration}] finish={_fr} compl_tokens={_ct} "
                    f"| reasoning_content(len={len(_raw_rc or '')})={(_raw_rc or '')[:160]!r} "
                    f"| content(len={len(cur_text or '')})={(cur_text or '')[:160]!r} "
                    f"| tool_calls={[ (t.function.name, t.function.arguments[:80]) for t in official_tool_calls ]}",
                    flush=True,
                )

            # vLLM bug: reasoning-only response with no content and no tool calls
            if cur_reasoning and not cur_text and not official_tool_calls:
                self._reasoning_only_retries += 1
                if self._reasoning_only_retries > 5:
                    logger.warning("Too many reasoning-only responses, breaking loop")
                    break
                messages.append(message)
                continue

            # Normalise tool calls to a uniform list of dicts
            function_calls: List[Dict[str, Any]] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in official_tool_calls
            ]

            # GLM quirk: parse tool calls embedded in reasoning text
            if cur_reasoning and not function_calls:
                parsed_from_reasoning = _parse_tool_calls_from_text(
                    cur_reasoning, tools,
                )
                if parsed_from_reasoning:
                    function_calls = [
                        {
                            "id": ptc["id"],
                            "name": ptc["function"]["name"],
                            "arguments": ptc["function"]["arguments"],
                        }
                        for ptc in parsed_from_reasoning
                    ]
                    # Create synthetic assistant message with these tool calls
                    messages.append({
                        "role": "assistant",
                        "content": cur_text or "",
                        "tool_calls": parsed_from_reasoning,
                    })
                else:
                    messages.append(message)
            else:
                messages.append(message)

            # Record answer turn if text is present
            if cur_text:
                if not function_calls:
                    self._notify_progress("answer", iteration)
                    self._vprint(iteration, "think", (cur_reasoning or "(no reasoning)")[:100])
                    self._vprint(iteration, "answer", cur_text[:100])
                    reasoning_path.append({
                        "action_type": "answer",
                        "think": cur_reasoning or "",
                        "generation": cur_text,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                prediction = cur_text

            # Terminate if no function calls
            if not function_calls:
                break

            # Process function calls; the iteration's searches are observed together
            n_searches = sum(1 for fc in function_calls if fc["name"] == "search")
            _iter_subqueries: List[str] = []
            _iter_seen_docs: List[Dict[str, Any]] = []

            for idx, tc in enumerate(function_calls):
                sub_iter = idx if n_searches > 1 else None
                try:
                    arguments = json.loads(tc["arguments"])
                    search_query = arguments.get("query", "").strip()

                    if tc["name"] == "search":
                        if idx == 0 and (cur_reasoning or cur_text):
                            self._notify_progress("think", iteration)
                            think_parts = [p for p in [cur_reasoning, cur_text] if p]
                            self._vprint(iteration, "think", " | ".join(think_parts)[:100])

                        if not search_query:
                            logger.warning(
                                f"Iteration {iteration}: empty search query, returning error to model. "
                                f"Raw arguments: {tc['arguments']!r}"
                            )
                            self._vprint(iteration, "search", "(empty query – asking model to retry)", sub_iter=sub_iter)
                            messages.append(self._build_tool_response_message(
                                tc["id"],
                                "Error: search query is empty. Please provide a non-empty, specific search query.",
                            ))
                            continue

                        self._notify_progress("search", iteration)
                        self._vprint(iteration, "search", search_query, sub_iter=sub_iter)
                        if self.search_tool is not None:
                            docs = self.search_tool.execute(
                                search_query,
                                original_query=query,
                                reasoning=cur_reasoning if cur_reasoning else None,
                            )
                        else:
                            docs = self.retrieve_documents(search_query, original_query=query)
                        result_text = self._format_search_results(docs)
                        self._vprint_docs(iteration, docs[:self.seen_top_k], sub_iter=sub_iter)

                        reasoning_path.append({
                            "action_type": "search",
                            "think": cur_reasoning if idx == 0 else "",
                            "search_query": search_query,
                            "docs": docs,
                            "all_docs": docs,
                            "component_doc_ids": [
                                d.get("doc_id", "")
                                for d in docs[: self.seen_top_k]
                            ],
                            "iteration": iteration,
                            "sub_iter": sub_iter,
                            "tokens": self._step_tokens() if idx == 0 else None,
                        })
                    else:
                        result_text = f"Error: Tool {tc['name']} not found"

                    messages.append(self._build_tool_response_message(tc["id"], result_text))
                    if tc["name"] == "search" and search_query:
                        _iter_subqueries.append(search_query)
                        _iter_seen_docs.extend(docs[:self.seen_top_k])

                except Exception as e:
                    error_msg = f"Error executing {tc.get('name', 'unknown')}: {e}"
                    logger.warning(error_msg)
                    messages.append(self._build_tool_response_message(
                        tc.get("id", ""), error_msg,
                    ))

            if _iter_subqueries:
                tag = self._observe_step(
                    _iter_subqueries, _iter_seen_docs, iteration, query,
                    trajectory=messages,
                )
                self._append_certainty(messages, tag, reasoning_path[-1])

            iteration += 1

        if not prediction:
            self._print("Max iterations reached without answer, forcing final answer in conversation")
            forced_text = self._force_answer_chat_in_conversation(messages)
            if not forced_text:
                self._print("Retrying with compressed evidence fallback")
                forced_text = self._force_answer_chat_compressed(
                    query, reasoning_path,
                )
            if forced_text:
                prediction = forced_text
                reasoning_path.append({
                    "action_type": "max_iter_force",
                    "think": FINAL_ANSWER_INSTRUCTION,
                    "generation": forced_text,
                    "docs": [],
                    "component_doc_ids": [],
                })
            if not prediction:
                prediction = "No answer found."

        num_iterations = iteration
        num_searches = sum(
            1 for s in reasoning_path if s.get("action_type") == "search"
        )
        self._print(
            f"Done: {num_iterations} iters, {len(reasoning_path)} steps, "
            f"{num_searches} searches, answer {len(prediction)} chars"
        )

        return reasoning_path, prediction, num_iterations
