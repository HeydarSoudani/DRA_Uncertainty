"""UncertaintyAwareAgent: a SearchR1-style search agent that reads a <certainty> tag.

Loop, per query::

    iteration t  policy writes <reasoning> <search>; code retrieves and shows
                 <information>; the shared uncertainty estimator observes the
                 iteration and code appends its <certainty step="t"> tag
    end          policy writes <reasoning> <answer>; when ``max_iteration``
                 runs out first, or the transcript no longer fits the context
                 window, one more call asks for the answer (the forced answer)

The policy never writes the tag.  Everything behind it (criteria, signals,
intermediate answers, ``uncertainty/{qid}.jsonl``) is the shared
:class:`uncertainty_estimator.UncertaintyEstimator`, exactly as for any agent;
see ``uncertainty_estimator.certainty`` for the tag layout.  The
``--uncertainty-estimator-mode`` flag applies as for the other agents:

    inform   the tag is appended after each search, and the system prompt
             explains it
    monitor  the signals are computed and saved, but the trajectory carries no
             tag and the system prompt never mentions it
    off      no estimator; the agent is a plain SearchR1-style agent

The run is one growing transcript in the user message, as in the SearchR1
family.  ``run_single`` and retrieval come from :class:`BasicAgent`, the
trajectory streams through the standard logger, and the per-turn records ride
on the result under ``ua_*`` keys, which the trajectory meta line persists
(``utils.config.AGENT_META_KEYS``).

Settings are the shared ones of ``dra_inference.yaml``: ``max_iteration`` (turn
cap, enforced in code and never stated to the policy), ``max_retries`` (format
re-asks), ``max_passage_chars``, ``llm_max_tokens_per_call``, ``seen_top_k``
(passages per search) and the run temperature.  The backbone's own reasoning
mode is always off.

The protocol's reasoning tag is ``<reasoning>``, not ``<think>``: with native
thinking off, Qwen3.x's chat template pre-fills an empty ``<think></think>``,
so the model treats thinking as done and never opens ``<think>`` again.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from reasoner_component import disable_native_thinking, is_context_window_error
from utils.text_utils import doc_id, doc_text, doc_title
from deep_research_agents.prompts.answer_prompts import (
    FINAL_ANSWER_INSTRUCTION,
    MAX_TURNS_ANSWER_INSTRUCTION,
    REASONING_TAG_FORMAT,
    force_answer_prompt,
)
from deep_research_agents.prompts.uncertainty_aware import (
    render_format_error,
    render_system,
    render_user,
)

from .base_agent import BasicAgent

logger = logging.getLogger(__name__)


# ── Turn protocol ─────────────────────────────────────────────────────────────
# A turn is ``<reasoning> … </reasoning>`` followed by exactly one ``<search>``
# or ``<answer>``.  :func:`parse_turn` returns the parsed pieces plus
#
#     errors    format failures, answered by a re-ask: no action, empty action.
#     warnings  repaired and logged: no <reasoning>, reasoning written without
#               tags (wrapped), a <certainty> or <information> written
#               by the policy (dropped).
#
# Anything after the first closing action tag is cut before parsing (see
# :func:`truncate_after_action`), so a second action or trailing reasoning
# never reaches the parser.  Parsing is regex-based rather than XML: model
# output is close to XML but not reliably well-formed.

ACTION_SEARCH = "search"
ACTION_ANSWER = "answer"

REASONING_TAG = "reasoning"

# The protocol's <reasoning> block, or a <think> block (provider reasoning the
# client re-inlined, or a model that falls back to its native tag).
_THINK_RE = re.compile(r"<(reasoning|think)>(.*?)</\1>", re.DOTALL)
_STRAY_THINK_RE = re.compile(r"</?(?:reasoning|think)>")
_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_INJECTED_RE = re.compile(r"<(certainty|information)\b[^>]*>.*?</\1>", re.DOTALL)

# Characters of a think shown in the verbose [think] line.
_THINK_PREVIEW_CHARS = 150


def _preview(text: str, max_chars: int = _THINK_PREVIEW_CHARS) -> str:
    """*text* on one line, cut to *max_chars* with "..." appended."""
    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[:max_chars].rstrip() + "..."


@dataclass
class Turn:
    think: Optional[str]
    action: Optional[str]          # "search" | "answer" | None
    action_text: Optional[str]
    text: str                      # the turn as it goes into the history
    native_reasoning: str = ""     # earlier reasoning blocks (provider reasoning re-inlined)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def truncate_after_action(text: str) -> str:
    """Cut the turn after its first closing action tag and close an open one.

    Stop sequences end generation at ``</search>`` / ``</answer>`` but are not
    returned by every provider, and not honoured by every provider either.
    """
    text = text or ""
    ends = [text.find(tag) + len(tag) for tag in ("</search>", "</answer>") if tag in text]
    if ends:
        return text[:min(ends)]
    for tag in ("search", "answer"):
        opened = text.rfind(f"<{tag}>")
        if opened != -1 and f"</{tag}>" not in text[opened:]:
            return text.rstrip() + f"</{tag}>"
    return text


def parse_turn(text: str) -> Turn:
    """Parse one policy turn."""
    errors: List[str] = []
    warnings: List[str] = []
    text = truncate_after_action(text)

    if _INJECTED_RE.search(text):
        warnings.append("the turn contains <certainty> or <information>; dropped")
        text = _INJECTED_RE.sub("", text)

    am = _ACTION_RE.search(text)
    action = action_text = None
    if am is None:
        errors.append("no <search> or <answer> in the turn")
    else:
        action, action_text = am.group(1), am.group(2).strip()
        if not action_text:
            errors.append(f"empty <{action}>")

    # The protocol's think is the last reasoning block before the action.
    # Earlier blocks are provider reasoning the client re-inlined.
    thinks = list(_THINK_RE.finditer(text))
    before = [m for m in thinks if am is None or m.start() < am.start()]
    think = None
    native = ""
    if before:
        think = before[-1].group(2).strip() or None
        native = "\n\n".join(m.group(2).strip() for m in before[:-1])
    elif am is not None:
        # With native thinking off, Qwen3-family models reason in untagged
        # prose before the action.  That prose is the turn's think: keep it,
        # so the history shows the protocol and the model its own reasoning.
        think = _STRAY_THINK_RE.sub("", text[:am.start()]).strip() or None
        if think:
            warnings.append(f"untagged reasoning; wrapped in <{REASONING_TAG}>")
    if think is None:
        warnings.append(f"no <{REASONING_TAG}> block")

    if action and action_text:
        parts = [f"<{REASONING_TAG}>{think}</{REASONING_TAG}>"] if think else []
        parts.append(f"<{action}>{action_text}</{action}>")
        history_text = "\n".join(parts)
    else:
        history_text = text.strip()

    return Turn(think=think, action=action, action_text=action_text, text=history_text,
                native_reasoning=native, errors=errors, warnings=warnings)


# ── Passage labels ────────────────────────────────────────────────────────────
# Every passage shown to the policy gets a label ``d1, d2, …`` that is unique in
# the run and stable: the same ``doc_id`` always gets the same label.

@dataclass
class DocRegistry:
    label_of: Dict[str, str] = field(default_factory=dict)   # doc_id -> dN
    doc_of: Dict[str, str] = field(default_factory=dict)     # dN -> doc_id

    def register(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Label *docs* (new ids get the next label); returns their labels."""
        labels: List[str] = []
        for doc in docs:
            did = doc_id(doc)
            label = self.label_of.get(did)
            if label is None:
                label = f"d{len(self.label_of) + 1}"
                self.label_of[did] = label
                self.doc_of[label] = did
            labels.append(label)
        return labels


def unique_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """*docs* in order, without repeated ids and without docs that have none
    (they cannot be labelled)."""
    seen = set()
    unique = []
    for doc in docs:
        did = doc_id(doc)
        if did and did not in seen:
            seen.add(did)
            unique.append(doc)
    return unique


def render_information(docs: List[Dict[str, Any]], labels: List[str], max_chars: int) -> str:
    parts = [f"[{label}] {doc_title(doc)}\n{doc_text(doc, max_length=max_chars)}"
             for doc, label in zip(docs, labels)]
    body = "\n\n".join(parts) if parts else "No documents retrieved."
    return f"<information>\n{body}\n</information>"


# ── Transcript ────────────────────────────────────────────────────────────────
# The transcript is the question followed by one block per search iteration:
# the turn, its <information> and (inform mode) its <certainty> tag.  When it
# no longer fits the context window, the forced answer reads a windowed copy
# in which only the last ``_KEEP_LAST_RESULTS`` iterations keep their passages.

_KEEP_LAST_RESULTS = 3
_OMITTED_INFORMATION = "<information>\n(earlier search results omitted)\n</information>"


@dataclass
class SearchBlock:
    turn: str
    information: str
    certainty: Optional[str] = None


def render_transcript(question: str, blocks: List[SearchBlock],
                      keep_last: Optional[int] = None) -> str:
    """The policy's user message; with *keep_last*, older passages are omitted."""
    parts = [render_user(question)]
    first_kept = 0 if keep_last is None else max(len(blocks) - keep_last, 0)
    for i, block in enumerate(blocks):
        information = block.information if i >= first_kept else _OMITTED_INFORMATION
        parts.append(f"\n\n{block.turn}\n{information}\n")
        if block.certainty:
            parts.append(f"{block.certainty}\n")
    return "".join(parts)


# ── Agent ─────────────────────────────────────────────────────────────────────

STOP_SEQUENCES = ["</search>", "</answer>"]
# Intermediate answer: stop at the answer, and before a search the model
# starts instead (it would go on to invent the <information> block).
ANSWER_STOP_SEQUENCES = ["</answer>", "<search>"]

# Forced-answer step types, shared with the other agents and the trajectory
# evaluator's force-answer set.
ACTION_MAX_ITER_FORCE = "max_iter_force"
ACTION_CONTEXT_LIMIT = "context_limit"

# How a run ends (``ua_outcome.end``).  "answered" is the normal end and
# "forced_answer" an answer asked for once the turn cap or the context window
# is reached (``ua_outcome.forced_by``).  The others are failures and are
# deliberately not in the trajectory evaluator's terminal set, so they count as
# runs that never finished; "max_turns" and "context_limit" are the two forced
# answers that gave no <answer>.
END_ANSWERED = "answered"
END_FORCED = "forced_answer"
END_MAX_TURNS = "max_turns"
END_CONTEXT_LIMIT = "context_limit"
END_FORMAT = "format_failure"
END_ERROR = "llm_error"


class UncertaintyAwareAgent(BasicAgent):
    AGENT_NAME = "UncertaintyAware"

    def __init__(self, llm_client, retriever: Optional[Any] = None, max_iteration: int = 100,
                 seen_top_k: int = 5, verbose: bool = True, max_passage_chars: int = 4000,
                 max_retries: int = 3):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self.max_passage_chars = max_passage_chars
        self.max_retries = max_retries
        # The backbone's own reasoning mode is always off, as in the Search-R1
        # family: the protocol's <reasoning> is the only reasoning.  With it on,
        # the model can spend the whole output cap on hidden reasoning and
        # return no action.  Applies to every call on this client, the
        # intermediate answers included.
        disable_native_thinking(self.generator)
        self._extras: Dict[str, Any] = {}

    def _result_extras(self) -> Dict[str, Any]:
        return self._extras

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call(self, messages: List[Dict[str, str]], temperature: float,
              stop: Optional[List[str]] = None) -> str:
        # The output cap is the client's (llm_max_tokens_per_call).
        # strip_think=False: the client's default cuts everything up to the
        # last </think>, which would drop the turn's reasoning before parsing.
        kwargs: Dict[str, Any] = {"temperature": temperature, "strip_think": False}
        if stop:
            kwargs["stop"] = stop
        return self.generator.complete(messages, **kwargs) or ""

    @staticmethod
    def _messages(system: str, transcript: str) -> List[Dict[str, str]]:
        # The whole run so far is one user message, as in the SearchR1 family.
        return [{"role": "system", "content": system},
                {"role": "user", "content": transcript}]

    def answer_from_trajectory(self, original_query: str, trajectory: Any, instruction: str) -> str:
        """Intermediate answer with the policy's call settings (native
        thinking off), greedy.

        The instruction joins the transcript's user message rather than
        following it as a second user turn: the run is one transcript, and a
        separate turn after it reads as the next turn of the search loop.
        """
        messages = self._intermediate_answer_messages(trajectory, instruction)
        instruction_msg = messages.pop()
        messages[-1]["content"] += f"\n\n{instruction_msg['content']}"
        return truncate_after_action(self._call(messages, 0.0, stop=ANSWER_STOP_SEQUENCES))

    def _generate_turn(self, messages: List[Dict[str, str]], t: int,
                       temperature: float) -> Tuple[Optional[Turn], int, Optional[Exception]]:
        """One turn, re-asked on format errors.

        Returns (turn, re-asks, error): turn is None and error the exception
        when the LLM call fails.
        """
        extra: List[Dict[str, str]] = []
        turn: Optional[Turn] = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call(messages + extra, temperature, stop=STOP_SEQUENCES)
            except Exception as exc:
                logger.warning("UncertaintyAware turn %d: LLM call failed: %s", t, exc)
                self._log_block(f"LLM call failed: {exc}", title=f"Turn {t}: error")
                return None, attempt, exc
            turn = parse_turn(raw)
            if turn.ok:
                return turn, attempt, None
            self._log_block(
                "\n".join(f"- {e}" for e in turn.errors) + f"\n\n```\n{raw.strip()[:3000]}\n```",
                title=f"Turn {t}: format errors (attempt {attempt + 1})",
            )
            extra = [
                {"role": "assistant", "content": turn.text or raw},
                {"role": "user", "content": render_format_error(turn.errors)},
            ]
        return turn, self.max_retries, None

    def _force_answer(self, system: str, question: str, blocks: List[SearchBlock],
                      temperature: float, forced_by: str, iteration: int) -> Dict[str, Any]:
        """Ask for the answer once the run cannot go on; one call, no search.

        *forced_by* is ``END_MAX_TURNS`` (turn cap) or ``END_CONTEXT_LIMIT``
        (the transcript no longer fits).  As for the intermediate answer, the
        instruction joins the transcript.  At the turn cap the full transcript
        is tried first and the windowed one only when it does not fit; at the
        context limit the windowed one is used directly.

        Returns the trajectory step; it has ``prediction`` when the call gave
        an <answer>, ``errors`` otherwise.
        """
        at_cap = forced_by == END_MAX_TURNS
        instruction = MAX_TURNS_ANSWER_INSTRUCTION if at_cap else FINAL_ANSWER_INSTRUCTION
        prompt = force_answer_prompt(REASONING_TAG_FORMAT, instruction)
        step: Dict[str, Any] = {
            "iteration": iteration,
            "action_type": ACTION_MAX_ITER_FORCE if at_cap else ACTION_CONTEXT_LIMIT,
        }
        forced: Optional[Turn] = None
        error: Optional[str] = None
        for keep_last in ((None, _KEEP_LAST_RESULTS) if at_cap else (_KEEP_LAST_RESULTS,)):
            transcript = render_transcript(question, blocks, keep_last=keep_last)
            try:
                raw = self._call(self._messages(system, f"{transcript}\n\n{prompt}"),
                                 temperature, stop=ANSWER_STOP_SEQUENCES)
            except Exception as exc:
                error = str(exc)
                logger.warning("UncertaintyAware forced answer: LLM call failed: %s", exc)
                self._log_block(f"LLM call failed: {exc}", title="Forced answer: error")
                if is_context_window_error(exc):
                    continue
                break
            forced, error = parse_turn(raw), None
            step["windowed"] = keep_last is not None
            break

        if forced is not None and forced.ok and forced.action == ACTION_ANSWER:
            self._vprint(iteration, "forced answer", forced.action_text)
            step.update({"think": forced.think, "prediction": forced.action_text,
                         "generation": forced.action_text})
        else:
            step["errors"] = error or "; ".join(
                (forced.errors if forced else []) or ["the forced turn is not an <answer>"])
        step["tokens"] = self._step_tokens()
        return step

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def inference(self, question: str, generation_temp: float = 0.6) -> tuple:
        self._extras = {}

        # Read per query: the estimator is attached after the agent is built.
        informs = self._informs
        system = render_system(inform=informs)
        registry = DocRegistry()
        blocks: List[SearchBlock] = []
        records: List[Dict[str, Any]] = []
        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""
        end = END_MAX_TURNS
        turns = 0
        searches = 0

        for t in range(self.max_iteration):
            self._notify_progress("turn", t)
            transcript = render_transcript(question, blocks)
            turn, attempts, error = self._generate_turn(self._messages(system, transcript), t, generation_temp)
            if turn is None:
                if is_context_window_error(error):
                    end = END_CONTEXT_LIMIT
                    break
                end = END_ERROR
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": END_ERROR, "errors": str(error),
                    "tokens": self._step_tokens(),
                })
                break
            turns = t + 1
            rec: Dict[str, Any] = {
                "turn": t,
                "valid": turn.ok,
                "errors": turn.errors,
                "warnings": turn.warnings,
                "format_retries": attempts,
                "action": turn.action,
                "action_text": turn.action_text,
                "native_reasoning_chars": len(turn.native_reasoning),
            }
            records.append(rec)
            if turn.warnings:
                self._log_block("; ".join(turn.warnings), title=f"Turn {t}: warnings")
            if turn.think:
                self._vprint(t, "think", _preview(turn.think))

            if not turn.ok:
                end = END_FORMAT
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": END_FORMAT, "think": turn.think,
                    "errors": "; ".join(turn.errors), "tokens": self._step_tokens(),
                })
                break

            if turn.action == ACTION_ANSWER:
                prediction = turn.action_text
                end = END_ANSWERED
                self._vprint(t, "answer", prediction)
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": ACTION_ANSWER, "think": turn.think,
                    "prediction": prediction, "generation": prediction,
                    "tokens": self._step_tokens(),
                })
                break

            # -- search, then (inform mode) the <certainty> tag that closes this iteration --
            query = turn.action_text
            self._notify_progress("search", t)
            self._vprint(t, "search", query)
            docs = self.retrieve_documents(query, original_query=question, reasoning=turn.think)
            shown = unique_docs(docs)[:self.seen_top_k]
            labels = registry.register(shown)
            block = SearchBlock(turn=turn.text,
                                information=render_information(shown, labels, self.max_passage_chars))
            blocks.append(block)
            step = {
                "iteration": t, "action_type": ACTION_SEARCH, "think": turn.think,
                "search_query": query, "docs": docs,
                "component_doc_ids": [doc_id(d) for d in shown],
                "labels": " ".join(labels),
                "tokens": self._step_tokens(),
            }

            tag = self._observe_step(
                query, shown, t, question,
                seen_docs=shown,
                trajectory=self._messages(system, render_transcript(question, blocks)),
            )
            if tag:
                block.certainty = tag
                step["certainty"] = tag
            self._record_step(reasoning_path, step)
            rec.update({"search": searches, "labels": labels, "certainty": tag})
            searches += 1

        forced_by = end if end in (END_MAX_TURNS, END_CONTEXT_LIMIT) else None
        if forced_by:
            self._notify_progress("force_answer", turns)
            forced_step = self._force_answer(system, question, blocks, generation_temp, forced_by, turns)
            self._record_step(reasoning_path, forced_step)
            if forced_step.get("prediction"):
                prediction = forced_step["prediction"]
                end = END_FORCED

        self._extras = {
            "ua_records": records,
            "ua_doc_labels": dict(registry.doc_of),
            "ua_outcome": {
                "end": end,
                "forced_by": forced_by,
                "answer": prediction or None,
                "num_turns": turns,
                "num_searches": searches,
                "format_retries": sum(r.get("format_retries", 0) for r in records),
                # Inform mode only: searches without a tag (the estimator
                # failed or had nothing to show).  None in monitor/off mode.
                "missing_certainty": sum(1 for r in records
                                         if r.get("action") == ACTION_SEARCH and not r.get("certainty"))
                                     if informs else None,
                "config": {
                    "max_iteration": self.max_iteration,
                    "max_retries": self.max_retries,
                    "max_passage_chars": self.max_passage_chars,
                },
            },
        }
        return reasoning_path, prediction, turns
