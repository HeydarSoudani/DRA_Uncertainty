"""Base class for retrieval agents."""

import copy
import logging
import time
import traceback
from typing import Callable, Dict, List, Any, Optional, Union

from utils.llm_client import LiteLLMClient
from searcher_component import normalize_retrieval_response
from deep_research_agents.prompts.answer_prompts import (
    FINAL_ANSWER_INSTRUCTION,
    TAG_FORMAT,
)
from utils.text_utils import passages2string, format_as_json  # noqa: F401 – re-exported for back-compat
from utils.text_utils import reduce_reasoning_path, build_evidence_summary
from utils.config import InferenceConfig
from utils.text_utils import verbose_print, verbose_print_search_results, verbose_print_uncertainty  # noqa: F401
from utils.text_utils import get_think as _get_think, get_query as _get_query, get_answer as _get_answer  # noqa: E501

logger = logging.getLogger(__name__)


class AgentVerboseMixin:
    """Mixin providing verbose logging and uncertainty-estimator helpers."""

    @property
    def _display_name(self) -> str:
        return getattr(self, "name", None) or getattr(self, "AGENT_NAME", "Agent")

    @property
    def _is_verbose(self) -> bool:
        return getattr(self, "verbose", True)

    def _print(self, message: str) -> None:
        if self._is_verbose:
            print(f"[{self._display_name}] {message}")

    def _vprint(self, iter_num: int, component: str, message: str, *, sub_iter: int = None) -> None:
        if self._is_verbose:
            verbose_print(iter_num, component, message, agent_name=self._display_name, sub_iter=sub_iter)

    def _vprint_docs(self, iter_num: int, docs: list, *, sub_iter: int = None) -> None:
        if self._is_verbose:
            verbose_print_search_results(iter_num, docs, agent_name=self._display_name, sub_iter=sub_iter)

    def _vprint_uncertainty(self, iter_num: int, record: dict) -> None:
        if self._is_verbose:
            verbose_print_uncertainty(iter_num, record, agent_name=self._display_name)

    # -- Online trajectory logging --

    def _record_step(self, reasoning_path: list, step: dict) -> dict:
        """Append *step* to *reasoning_path* and stream it to the trajectory log.

        Every ``reasoning_path.append(...)`` goes through here, so a step
        reaches disk the moment it is complete rather than when the query
        finishes.  An interrupted run then still leaves everything that ran
        before the interruption in ``trajectory/{qid}.jsonl`` and
        ``trajectory/{qid}.md``.

        Callers that mutate a step after appending it must log it *after* the
        mutation instead, or the logged copy will be missing those fields.
        """
        reasoning_path.append(step)
        logger_ = getattr(self, "_traj_logger", None)
        if logger_ is not None:
            logger_.log_step(step)
        return step

    def _log_block(self, text: str, *, title: str = None) -> None:
        """Write a markdown-only note into the trajectory log.

        For what explains the trajectory without belonging to it: phase
        banners, parse errors and retries.  Never touches the
        JSONL.
        """
        logger_ = getattr(self, "_traj_logger", None)
        if logger_ is not None:
            logger_.log_block(text, title=title)

    def _begin_trajectory_log(self, trajectory_logger) -> None:
        """Attach and open the per-query trajectory log (no-op when absent)."""
        self._traj_logger = trajectory_logger
        if trajectory_logger is not None:
            trajectory_logger.start()

    def _end_trajectory_log(self, result: dict = None, error: BaseException = None) -> None:
        """Write the trajectory log's footer and close it.

        Called from ``run_single``'s ``finally`` so an exception or a kill still
        leaves a closed, flushed file marked as incomplete.
        """
        logger_ = getattr(self, "_traj_logger", None)
        if logger_ is None:
            return
        try:
            if result is not None:
                logger_.finalize(result)
        finally:
            logger_.close(error=error)
            self._traj_logger = None

    def _observe_step(
        self,
        subqueries: Union[str, List[str]],
        docs: List[Dict[str, Any]],
        iter_num: int,
        original_query: Optional[str] = None,
        *,
        seen_docs: Optional[List[Dict[str, Any]]] = None,
        trajectory: Any = None,
    ) -> None:
        """Pass one finished search iteration to the uncertainty estimator.

        Called once per iteration, after all its searches, with the
        iteration's queries, the documents the agent was shown and its
        conversation so far (*trajectory*, for the intermediate answer).  The
        estimator only records signals; the trajectory is never changed.
        *seen_docs*, when given, are printed in verbose mode.

        The estimator is isolated from the agent run: its errors are logged,
        never raised; the tokens its intermediate answer spends on the
        agent's client are removed from the agent's token meter; its wall
        time is added to ``self._uncertainty_seconds`` so time limits can
        exclude it.
        """
        if seen_docs is not None:
            self._vprint_docs(iter_num, seen_docs)
        estimator = getattr(self, "uncertainty_estimator", None)
        if estimator is None:
            return
        started = time.monotonic()
        meter = self._token_meter() if hasattr(self, "_token_meter") else None
        saved = (meter.input_tokens, meter.output_tokens, meter.num_calls) if meter is not None else None
        try:
            record = estimator.observe(
                subqueries=[subqueries] if isinstance(subqueries, str) else list(subqueries),
                docs=docs,
                iter_num=iter_num,
                original_query=original_query or "",
                trajectory=trajectory,
            )
            self._vprint_uncertainty(iter_num, record)
        except Exception:
            logger.warning("Uncertainty estimator failed at iteration %s; the agent run continues",
                           iter_num, exc_info=True)
        finally:
            if saved is not None:
                meter.input_tokens, meter.output_tokens, meter.num_calls = saved
            self._uncertainty_seconds = getattr(self, "_uncertainty_seconds", 0.0) + time.monotonic() - started

    def _reset_uncertainty_estimator(self, query_id: Optional[str], query_text: str) -> None:
        self._uncertainty_seconds = 0.0
        estimator = getattr(self, "uncertainty_estimator", None)
        if estimator is None:
            return
        try:
            estimator.reset(query_id=query_id, query=query_text)
        except Exception:
            logger.warning("Uncertainty estimator reset failed for %s", query_id, exc_info=True)

    def _attach_uncertainty_stats(self, result: dict) -> None:
        estimator = getattr(self, "uncertainty_estimator", None)
        if estimator is None:
            return
        try:
            result["uncertainty_meta"] = estimator.meta()
            result["uncertainty_steps"] = list(estimator.steps)
        except Exception:
            logger.warning("Uncertainty estimator meta failed", exc_info=True)


def _strip_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove tool-related content from a message list.

    Providers like Bedrock reject messages that contain tool_calls or
    role="tool" entries when no ``tools=`` parameter is supplied. This
    helper converts such messages into plain text so the conversation
    can be sent as a regular completion call (e.g. the intermediate answer
    generation).
    """
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            sanitized.append({
                "role": "user",
                "content": f"[Search Result]\n{msg.get('content', '')}",
            })
        elif msg.get("tool_calls"):
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            if not new_msg.get("content"):
                new_msg["content"] = "(searching...)"
            sanitized.append(new_msg)
        else:
            sanitized.append(msg)
    return sanitized


class BasicAgent(AgentVerboseMixin):
    """Base class for retrieval agents."""

    def __init__(self, llm_client: LiteLLMClient, retriever: Optional[Any] = None, max_iteration: int = 100, seen_top_k: int = 5, search_tool=None, uncertainty_estimator=None):
        """Initialize BasicAgent.

        Args:
            llm_client: LiteLLM client for generation
            retriever: Retriever endpoint client (optional for no-retrieval models)
            max_iteration: Maximum iterations for multi-step methods
            seen_top_k: Number of top docs passed to the next component (visible
                to the model). These docs are also considered as cited docs.
                Doc IDs for these are recorded in the trajectory.
            search_tool: Optional RetrievalSearchTool instance. When provided,
                retrieve_documents() delegates to it (supporting fusion and
                reranking). Falls back to raw retriever if not set.
            uncertainty_estimator: Optional UncertaintyEstimator. When
                provided, _observe_step() records the uncertainty signals of
                each search iteration; the trajectory is never changed.
        """
        self.generator = llm_client
        self.retriever = retriever
        self.search_tool = search_tool
        self.uncertainty_estimator = uncertainty_estimator
        self.max_iteration = max_iteration
        self.seen_top_k = seen_top_k
        # Set per query by run_single(); see _record_step().
        self._traj_logger = None

        self.inference_config: InferenceConfig = InferenceConfig()

        # Per-agent Chat Completions sampling overrides spread into every
        # client.chat.completions.create() call. Empty by default (let the
        # served model's generation_config decide). Agents that are sensitive to
        # decoding populate this with their official sampling params (e.g. GLM
        # sets extra_body to request reasoning tokens over OpenRouter).
        self._sampling: Dict[str, Any] = {}

    def _chat_sampling_kwargs(self) -> Dict[str, Any]:
        """Sampling kwargs to spread into chat.completions.create() calls.

        Returns a shallow copy so callers can mutate (e.g. add max_tokens)
        without clobbering the agent-level defaults.
        """
        return dict(getattr(self, "_sampling", {}) or {})

    # Agent display name used by _display_name property; subclasses can override.
    AGENT_NAME: str = "Agent"

    def _token_meter(self):
        """Return the active :class:`TokenMeter`, or None if unavailable.

        Prefers an agent-owned ``self.token_meter`` (vendor agents that hold
        their own provider client) and falls back to the generator's meter
        (LiteLLM/HF-backed agents using ``self.generator``).
        """
        meter = getattr(self, "token_meter", None)
        if meter is not None:
            return meter
        for client_attr in ("generator", "_vllm_client"):
            meter = getattr(getattr(self, client_attr, None), "token_meter", None)
            if meter is not None:
                return meter
        return None

    def _step_tokens(self):
        """Tokens consumed since the previous trajectory step (or None)."""
        meter = self._token_meter()
        return meter.since_last_step() if meter is not None else None

    def get_intermediate_answer_llm(self) -> Optional[LiteLLMClient]:
        """Return a LiteLLM client for the intermediate answer.

        Self-managed agents (those with model_name / model_url attributes)
        automatically get a hosted_vllm/ LiteLLM wrapper pointing at the
        same vLLM server so the intermediate answer uses the same
        backbone model.  API-backed agents fall back to self.generator.
        Subclasses with non-standard routing (e.g. Bedrock) can still
        override this method.
        """
        model_name = getattr(self, "model_name", None)
        model_url = getattr(self, "model_url", None)
        if model_name and model_url:
            api_key = getattr(self, "_api_key", "EMPTY")
            return LiteLLMClient(
                model=f"hosted_vllm/{model_name}",
                api_base=model_url,
                api_key=api_key,
            )
        return self.generator

    def get_system_prompt(self) -> Optional[str]:
        """Return the agent's system prompt, or None if not set."""
        return getattr(self, "system_prompt", None) or getattr(self, "_system_prompt", None)

    # ------------------------------------------------------------------
    # Default document formatting and tool definitions
    # ------------------------------------------------------------------

    def _format_search_results(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as JSON for tool response messages."""
        return format_as_json(docs, self.seen_top_k)

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return search tool definition in the appropriate API format.

        Dispatches on ``inference_config.api_type``.  Subclasses with
        additional tools should override this method.
        """
        cfg = self.inference_config
        desc = (
            "Search for information using the search engine. "
            f"Returns top {self.seen_top_k} results."
        )
        if cfg.api_type == "responses_api":
            return [{
                "type": "function",
                "name": "search",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }]
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                },
            },
        }]

    # ------------------------------------------------------------------
    # Unified Responses API call (used by force answer + intermediate answer)
    # ------------------------------------------------------------------

    def _make_responses_api_call(
        self,
        messages: List[Any],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens_override: Optional[int] = None,
        return_reasoning_fallback: bool = False,
    ) -> Optional[str]:
        """Make a Responses API call using ``self.inference_config``.

        Returns the text content from the first message item.  When
        *return_reasoning_fallback* is True and no message item exists,
        the reasoning summary text is returned (prefixed with
        ``[reasoning_fallback]`` so callers can distinguish it).
        Returns None only on API failure or truly empty output.
        """
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

        request: Dict[str, Any] = {
            "model": cfg.model_name,
            "max_output_tokens": max_output_tokens_override or cfg.max_output_tokens,
            "input": messages,
            "truncation": "auto",
        }
        if cfg.reasoning_effort is not None:
            request["reasoning"] = {
                "effort": cfg.reasoning_effort,
                "summary": "detailed",
            }
        if tools:
            request["tools"] = tools

        try:
            response = client.responses.create(**request)
        except Exception as e:
            logger.warning("Responses API call failed: %s", e)
            return None

        for item in response.output:
            if getattr(item, "type", None) == "message":
                return "\n".join(p.text for p in item.content)

        if return_reasoning_fallback:
            reasoning_parts = []
            for item in response.output:
                if getattr(item, "type", None) == "reasoning":
                    reasoning_parts.extend(
                        p.text for p in item.content if hasattr(p, "text")
                    )
            if reasoning_parts:
                return "[reasoning_fallback]" + "\n".join(reasoning_parts)

        return None

    # ------------------------------------------------------------------
    # Trim first iteration helper
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_attr(msg: Any, key: str, default: Any = None) -> Any:
        """Get an attribute from a message, whether it's a dict or pydantic object."""
        if isinstance(msg, dict):
            return msg.get(key, default)
        return getattr(msg, key, default)

    @classmethod
    def _trim_first_iteration(cls, messages: List[Any]) -> List[Any]:
        """Remove the first search iteration from the message history.

        Returns a new list: ``[prefix] + [iter_2 … iter_L]``.
        Works for both Responses API (type-keyed items) and Chat Completions
        (role-keyed messages).  If fewer than 2 iterations exist, returns a
        copy of *messages* unchanged.
        """
        prefix_end = 0
        for i, msg in enumerate(messages):
            if cls._msg_attr(msg, "role") in ("system", "user"):
                prefix_end = i + 1
            else:
                break

        if prefix_end >= len(messages):
            return list(messages)

        first_item = messages[prefix_end]

        if cls._msg_attr(first_item, "type") is not None:
            seen_tool_output = False
            for i in range(prefix_end, len(messages)):
                if cls._msg_attr(messages[i], "type") == "function_call_output":
                    seen_tool_output = True
                elif seen_tool_output:
                    return messages[:prefix_end] + messages[i:]
        else:
            assistant_count = 0
            for i in range(prefix_end, len(messages)):
                if cls._msg_attr(messages[i], "role") == "assistant":
                    assistant_count += 1
                    if assistant_count == 2:
                        return messages[:prefix_end] + messages[i:]

        return list(messages)

    # ------------------------------------------------------------------
    # Shared force-answer (Responses API agents)
    # ------------------------------------------------------------------

    def _force_answer_responses_api_in_conversation(
        self,
        messages: List[Any],
    ) -> Optional[str]:
        """Append force-answer instruction to conversation and call the API.

        Uses ``inference_config`` for model, max_output_tokens, reasoning_effort.
        Trims the first iteration to free context space for the instruction.
        """
        cfg = self.inference_config
        trimmed = self._trim_first_iteration(messages)
        trimmed.append({
            "role": "user",
            "content": f"{FINAL_ANSWER_INSTRUCTION}\n\n{cfg.format_instructions}",
        })
        return self._make_responses_api_call(trimmed)

    def _force_answer_responses_api_compressed(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Standalone prompt with compressed evidence — last-resort fallback.

        Uses ``inference_config`` for model, max_output_tokens, reasoning_effort.
        """
        cfg = self.inference_config
        prompt = self._build_force_answer_prompt(
            query, reasoning_path, cfg.format_instructions,
        )
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._make_responses_api_call(messages)

    # ------------------------------------------------------------------
    # Shared force-answer (Chat Completions agents)
    # ------------------------------------------------------------------

    def _force_answer_chat_in_conversation(
        self,
        messages: List[Any],
    ) -> Optional[str]:
        """Append force-answer instruction to conversation and call Chat Completions API.

        Uses ``inference_config`` for model, api_base, api_key, max_output_tokens.
        Trims the first iteration to free context space for the instruction.
        """
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        trimmed = self._trim_first_iteration(messages)
        trimmed.append({
            "role": "user",
            "content": f"{FINAL_ANSWER_INSTRUCTION}\n\n{cfg.format_instructions}",
        })
        force_max_tokens = min(cfg.max_output_tokens, 4096)
        try:
            response = client.chat.completions.create(
                model=cfg.model_name,
                messages=trimmed,
                max_tokens=force_max_tokens,
                **self._chat_sampling_kwargs(),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Force answer (chat, in conversation) failed: %s", e)
            return None

    def _force_answer_chat_compressed(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Standalone prompt with compressed evidence — Chat Completions fallback.

        Uses ``inference_config`` for model, api_base, api_key, max_output_tokens.
        """
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        prompt = self._build_force_answer_prompt(
            query, reasoning_path, cfg.format_instructions,
        )
        messages: List[Dict[str, str]] = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        force_max_tokens = min(cfg.max_output_tokens, 4096)
        try:
            response = client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                max_tokens=force_max_tokens,
                **self._chat_sampling_kwargs(),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Force answer (chat, compressed) failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Intermediate answer (asked by the uncertainty estimator every turn)
    # ------------------------------------------------------------------

    def answer_from_trajectory(
        self,
        original_query: str,
        trajectory: Any,
        instruction: str,
    ) -> str:
        """Raw model output for *instruction* asked after *trajectory*.

        Used by ``IntermediateAnswerSignal``, which owns the instruction and
        the parsing.  The default appends the instruction to a copy of the
        conversation and calls the agent's own model; Responses API agents
        continue the live conversation.  Agents with another context or
        client override this.  Raises on failure; a reasoning-only reply is
        returned with the ``[reasoning_fallback]`` prefix.
        """
        messages = self._intermediate_answer_messages(trajectory, instruction)
        if self.inference_config.api_type == "responses_api":
            return self._make_responses_api_call(messages, return_reasoning_fallback=True) or ""
        llm = self.get_intermediate_answer_llm()
        if llm is None:
            raise RuntimeError("no LLM client available")
        return llm.complete(
            _strip_tool_messages(messages),
            strip_think=False,
            return_reasoning_fallback=True,
            max_tokens=self.inference_config.max_output_tokens,
        ) or ""

    @staticmethod
    def _intermediate_answer_messages(trajectory: Any, instruction: str) -> List[Dict[str, Any]]:
        """A copy of the conversation with *instruction* as the last user turn."""
        if not isinstance(trajectory, list):
            raise ValueError(f"trajectory is not a message list ({type(trajectory).__name__})")
        messages = copy.deepcopy(trajectory)
        messages.append({"role": "user", "content": instruction})
        return messages

    def _notify_progress(self, stage: str, iteration: int) -> None:
        """Report current iteration and stage to the progress bar callback."""
        cb = getattr(self, "_status_callback", None)
        if cb is not None:
            cb(stage, iteration)

    # ------------------------------------------------------------------
    # run_single building blocks (shared by the base loop and overrides)
    # ------------------------------------------------------------------

    @staticmethod
    def _count_searches(reasoning_path: List[Dict[str, Any]]) -> int:
        """Number of trajectory steps that actually retrieved documents."""
        return sum(
            1 for step in reasoning_path
            if step.get("docs") or step.get("all_docs")
        )

    def _token_usage_delta(self, start: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        """Token usage accrued since *start* (from :meth:`_token_meter().snapshot`)."""
        meter = self._token_meter()
        if meter is None or start is None:
            return None
        end = meter.snapshot()
        return {k: end[k] - start[k] for k in
                ("input_tokens", "output_tokens", "total_tokens", "num_calls")}

    # ------------------------------------------------------------------
    # Common run interface (shared by all reasoning agents)
    # ------------------------------------------------------------------

    def run_single(self, query_id: str, query_text: str, temperature: float = 0.7, status_callback=None, trajectory_logger=None) -> Optional[Dict[str, Any]]:
        """Process a single query and return its normalised result dict.

        Args:
            query_id:   Identifier for the query.
            query_text: The query string.
            temperature: Sampling temperature passed to inference().
            status_callback: Optional callable(stage: str, iteration: int) invoked at
                             each retrieval step.  Used to stream live progress to the
                             parent process's tqdm bars in multi-GPU runs.
            trajectory_logger: Optional :class:`utils.trajectory_logger.TrajectoryLogger`
                             that streams each step to disk as it happens, so an
                             interrupted run still leaves a readable trajectory.

        Returns:
            result_dict with keys: query, generation, num_steps, num_searches, trajectory.
            Returns None on error.

        Note:
            Supports inference() returning either 2 values (reasoning_path, prediction)
            or 3 values (reasoning_path, prediction, num_iterations).
        """
        self._status_callback = status_callback
        self._search_iter     = 0
        self._begin_trajectory_log(trajectory_logger)
        _error: Optional[BaseException] = None
        _result: Optional[Dict[str, Any]] = None
        # Reset the uncertainty estimator (creates the query's criteria, loads its qrels)
        self._reset_uncertainty_estimator(query_id, query_text)
        # Snapshot token usage so we can report per-query totals.
        _meter = self._token_meter()
        _tok_start = _meter.snapshot() if _meter is not None else None
        if _meter is not None:
            _meter.since_last_step()  # clear per-step cursor for this query
        try:
            inference_result = self.inference(query_text, generation_temp=temperature)

            # Support both 2-value and 3-value returns from inference()
            if len(inference_result) == 3:
                reasoning_path, prediction, num_iterations = inference_result
            else:
                reasoning_path, prediction = inference_result
                num_iterations = None

            num_searches = self._count_searches(reasoning_path)

            result = {
                "query": query_text,
                "generation": str(prediction) if prediction else "",
                "num_steps": len(reasoning_path),
                "num_searches": num_searches,
                "trajectory": reasoning_path,
            }
            if num_iterations is not None:
                result["num_iterations"] = num_iterations

            # Attach uncertainty signals when available
            self._attach_uncertainty_stats(result)

            # Attach per-query token usage when a meter is available.
            token_usage = self._token_usage_delta(_tok_start)
            if token_usage is not None:
                result["token_usage"] = token_usage

            if num_iterations is not None:
                logger.info(
                    f"  ✓ {num_iterations} iters, {num_searches} searches, "
                    f"{len(reasoning_path)} steps"
                )
            else:
                logger.info(f"  ✓ {num_searches} searches, {len(reasoning_path)} steps")
            _result = result
            return result

        except Exception as e:
            _error = e
            logger.error(f"  ✗ Error processing {query_id}: {e}")
            traceback.print_exc()
            return None
        finally:
            self._end_trajectory_log(result=_result, error=_error)
            self._status_callback = None
            self._search_iter     = 0

    def cleanup(self):
        """Release any resources held by the agent."""
        pass

    # ------------------------------------------------------------------
    # Helpers used by subclasses
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shared tool-call loop helpers (GLM, OSS)
    # ------------------------------------------------------------------

    def _build_tool_response_message(
        self,
        call_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Build a tool response message in the appropriate API format.

        Chat Completions API → ``{"role": "tool", "tool_call_id": ..., "content": ...}``
        Responses API        → ``{"type": "function_call_output", "call_id": ..., "output": ...}``
        """
        cfg = self.inference_config
        if cfg.api_type == "responses_api":
            return {"type": "function_call_output", "call_id": call_id, "output": content}
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    # ------------------------------------------------------------------
    # Other helpers
    # ------------------------------------------------------------------

    def get_unique_docs(self, docs_lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate documents based on doc_id."""
        return list({doc['doc_id']: doc for doc in docs_lst}.values())

    def get_think(self, text: str) -> Optional[str]:
        """Extract thinking process from <think> tags."""
        return _get_think(text)

    def get_query(self, text: str) -> Optional[str]:
        """Extract search query from <search> tags."""
        return _get_query(text)

    def get_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer> tags."""
        return _get_answer(text)

    def _rebuild_step_text(self, step: Dict[str, Any]) -> str:
        """Reconstruct prompt text for one reasoning step from structured data.

        Default uses ``<think>/<search>/<information>`` tags (SearchR1/StepSearch).
        Override in subclasses for agents with different prompt formats.
        """
        think = step.get("think", "")
        sq = step.get("search_query", step.get("query", ""))
        docs = step.get("docs", [])
        is_stripped = step.get("_docs_stripped", False)
        seen_top_k = getattr(self, "seen_top_k", 5)

        output_text = ""
        if think:
            output_text += f"<think>{think}</think>\n"
        if sq:
            output_text += f"<search>{sq}</search>"

        if is_stripped:
            search_results = "(earlier search results omitted for brevity)"
        else:
            search_results = passages2string(docs[:seen_top_k])

        return f"\n\n{output_text}<information>{search_results}</information>\n\n"

    def _rebuild_windowed_prompt(
        self,
        initial_prompt: str,
        reasoning_path: List[Dict[str, Any]],
        answer_nudge: str,
        keep_last: int = 3,
    ) -> str:
        """Rebuild a windowed conversation prompt from reasoning_path.

        Reconstructs the agent's conversation with full document context for
        recent search turns and stripped context for older turns, then appends
        the answer nudge.  Same pattern as early stopping (real conversation
        + nudge) but with reduced context to fit within the limit.
        """
        pruned = reduce_reasoning_path(reasoning_path, keep_last=keep_last)

        prompt = initial_prompt
        for step in pruned:
            if step.get("search_query") or step.get("query"):
                prompt += self._rebuild_step_text(step)

        prompt += answer_nudge
        return prompt

    def _build_force_answer_prompt(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
        format_instructions: str,
    ) -> str:
        """Build a compact prompt for forcing a final answer.

        Reusable across agents that handle their own LLM calls (e.g. GLM,
        OSS).  Agents using ``self.generator`` should prefer
        ``_force_answer_on_context_limit`` which wraps this.
        """
        evidence = build_evidence_summary(reasoning_path, seen_top_k=self.seen_top_k)
        return (
            f"{FINAL_ANSWER_INSTRUCTION}\n\n"
            f"Question: {query}\n\n"
            f"Evidence collected:\n{evidence}\n\n"
            f"{format_instructions}"
        )

    def _force_answer_on_context_limit(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
        format_instructions: str,
        extract_fn: Callable[[Optional[str]], Optional[str]],
        *,
        generation_temp: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
        generator_kwargs: Optional[Dict[str, Any]] = None,
        initial_prompt: Optional[str] = None,
        answer_nudge: Optional[str] = None,
    ) -> Optional[str]:
        """Force a final answer when context limit is reached.

        Two modes depending on whether ``initial_prompt`` is provided:

        **Windowed mode** (``initial_prompt`` set): Rebuilds the conversation
        from ``initial_prompt`` + pruned ``reasoning_path`` using
        ``_rebuild_windowed_prompt``, then appends the answer nudge.

        **Summary mode** (``initial_prompt`` is ``None``): Falls back to
        ``_build_force_answer_prompt`` which builds a standalone prompt with
        a windowed evidence summary.

        Returns the extracted answer string, or ``None`` on failure.
        """
        self._print("Context limit hit, forcing final answer from collected evidence")

        if initial_prompt is not None:
            nudge = answer_nudge if answer_nudge is not None else f"\n{format_instructions}"
            prompt = self._rebuild_windowed_prompt(
                initial_prompt, reasoning_path, answer_nudge=nudge,
            )
        else:
            prompt = self._build_force_answer_prompt(query, reasoning_path, format_instructions)

        messages: List[Dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if assistant_prefill is not None:
            messages.append({"role": "assistant", "content": assistant_prefill})

        try:
            kwargs: Dict[str, Any] = {"temperature": generation_temp, "max_tokens": max_tokens}
            if generator_kwargs:
                kwargs.update(generator_kwargs)
            forced_output = self.generator.complete(messages, **kwargs)
            answer = extract_fn(forced_output)
            self._vprint(-1, "forced_answer", (answer or "(no answer)")[:300])
            reasoning_path.append({
                'action_type': 'context_limit',
                'think': FINAL_ANSWER_INSTRUCTION,
                'prediction': answer,
            })
            return answer
        except Exception as e2:
            logger.warning(f"Forced final answer also failed: {e2}")
            return None

    def retrieve_documents(self, query: str, *, original_query: Optional[str] = None, reasoning: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for a query.

        Delegates to ``self.search_tool`` when available (supports fusion
        and reranking).  Falls back to the raw retriever otherwise.

        Args:
            query: The search query string.
            original_query: The original user query, forwarded to the search
                tool so that ``retrieval_input`` / ``post_fusion_reranker_input``
                composition modes can combine it with sub-queries.
            reasoning: Optional trajectory context forwarded to the search
                tool for use with ``retrieval_input`` / ``post_fusion_reranker_input``
                composition modes.
        """
        self._search_iter = getattr(self, "_search_iter", 0) + 1

        if self.search_tool is not None:
            return self.search_tool.execute(query, original_query=original_query, reasoning=reasoning)

        if not self.retriever:
            return []

        results = self.retriever.retrieve(query)
        return normalize_retrieval_response(results)


class TagReasoningAgent(BasicAgent):
    """Base for reasoning agents using a think/search/answer tag loop.

    Covers the shared inference pattern used by SearchR1, StepSearch, and
    ReSearch.  Subclasses configure prompt formatting and answer detection
    via constructor args and optional method overrides.

    Configuration points:
        _format_initial_prompt(question) — build the initial user prompt.
        _build_messages(input_prompt)    — wrap prompt into message list.
        _has_answer(output_text)         — detect whether output contains a final answer.
        _extract_prediction(output_text) — pull the prediction from the answer output.
        curr_step_template               — format string for appending search steps.
        _system_prompt                   — optional system message (None = omit).
    """

    AGENT_NAME = "TagReasoning"

    def __init__(self, llm_client: LiteLLMClient, retriever: Optional[Any] = None, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self._system_prompt: Optional[str] = None
        self.curr_step_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
        )

    # -- Configuration points (override in subclasses) --

    def _format_initial_prompt(self, question: str) -> str:
        raise NotImplementedError

    def _build_messages(self, input_prompt: str) -> List[Dict[str, str]]:
        if self._system_prompt:
            return [
                {'role': 'system', 'content': self._system_prompt},
                {'role': 'user', 'content': input_prompt},
            ]
        return [{'role': 'user', 'content': input_prompt}]

    def _has_answer(self, output_text: str) -> bool:
        return '</answer>' in output_text

    def _extract_prediction(self, output_text: str) -> Optional[str]:
        return self.get_answer(output_text)

    # -- Shared inference loop --

    def inference(self, question: str, generation_temp: float = 0.7) -> tuple:
        input_prompt = self._format_initial_prompt(question)
        messages = self._build_messages(input_prompt)

        reasoning_path: List[Dict[str, Any]] = []
        for iter_idx in range(self.max_iteration):
            iter_num = iter_idx
            self._notify_progress("think", iter_num)
            try:
                output_text = self.generator.complete(messages, temperature=generation_temp)
            except Exception as e:
                logger.warning(f"Iteration {iter_num} API error: {e}")
                self._force_answer_on_context_limit(
                    question, reasoning_path,
                    f"{TAG_FORMAT}\n<answer>",
                    lambda out: (out or "").split("</answer>")[0].strip(),
                    generation_temp=generation_temp,
                    system_prompt=self._system_prompt,
                    initial_prompt=self._format_initial_prompt(question),
                    answer_nudge=f"\n{FINAL_ANSWER_INSTRUCTION}\n<answer>",
                )
                break

            if self._has_answer(output_text):
                one_step_think = self.get_think(output_text)
                prediction = self._extract_prediction(output_text)
                self._vprint(iter_num, "think", one_step_think or "(no think)")
                self._notify_progress("answer", iter_num)
                self._vprint(iter_num, "answer", prediction or "(no answer)")
                reasoning_path.append({'think': one_step_think, 'prediction': prediction, 'tokens': self._step_tokens()})
                break

            tmp_query = self.get_query(output_text)
            think_text = self.get_think(output_text)
            self._vprint(iter_num, "think", think_text or "(no think)")

            if tmp_query:
                self._notify_progress("search", iter_num)
                self._vprint(iter_num, "search", tmp_query)
                search_docs = self.retrieve_documents(
                    tmp_query,
                    original_query=question,
                    reasoning=think_text if think_text else None,
                )
                search_results = passages2string(search_docs[:self.seen_top_k])

            else:
                search_docs, search_results = [], ''

            reasoning_path.append({
                'think': think_text,
                'search_query': tmp_query,
                'docs': search_docs,
                'component_doc_ids': [d.get('doc_id', '') for d in search_docs[:self.seen_top_k]],
                'tokens': self._step_tokens(),
            })

            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = self._build_messages(input_prompt)

            if tmp_query:
                seen_docs = search_docs[:self.seen_top_k]
                self._observe_step(
                    tmp_query, seen_docs, iter_num, question,
                    seen_docs=seen_docs,
                    trajectory=messages,
                )

        prediction = reasoning_path[-1].get('prediction') if reasoning_path else None

        if not prediction:
            action_type = 'max_iter_force'
            force_input = input_prompt + (
                f"\n{FINAL_ANSWER_INSTRUCTION} {TAG_FORMAT}\n<answer>"
            )
            force_messages = self._build_messages(force_input)
            force_output = self.generator.complete(force_messages, temperature=generation_temp)
            prediction = (force_output or "").split("</answer>")[0].strip()
            self._vprint(iter_num, "think", FINAL_ANSWER_INSTRUCTION)
            self._vprint(iter_num, "finish", prediction or "(no answer)")
            reasoning_path.append({
                'action_type': action_type,
                'think': FINAL_ANSWER_INSTRUCTION,
                'prediction': prediction,
                'tokens': self._step_tokens(),
            })

        num_iterations = iter_idx + 1 if reasoning_path else 0
        return reasoning_path, prediction, num_iterations
