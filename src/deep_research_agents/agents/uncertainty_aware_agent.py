"""UncertaintyAwareAgent: a SearchR1-style search agent that reads a <certainty> tag.

Loop, per query::

    iteration t  policy writes <think> <search>; code retrieves and shows
                 <information>; the shared uncertainty estimator observes the
                 iteration and code appends its <certainty step="t"> tag
    end          policy writes <think> <answer>, or ``max_turns`` runs out
                 (a run that never answers is a failure)

The policy never writes the tag.  Everything behind it (criteria, signals,
intermediate answers, ``uncertainty/{qid}.jsonl``) is the shared
:class:`uncertainty_estimator.UncertaintyEstimator`, exactly as for any agent
run with ``--uncertainty-estimator-mode inform``; see
``uncertainty_estimator.certainty`` for the tag layout.  This agent always runs
in ``inform`` mode (the pipeline forces it, whatever the flag says).  What sets
it apart from the other agents is only its system prompt, which explains the
tag.

The run is one growing transcript in the user message, as in the SearchR1
family.  ``run_single`` and retrieval come from :class:`BasicAgent`, the
trajectory streams through the standard logger, and the per-turn records ride
on the result under ``ua_*`` keys, which the trajectory meta line persists
(``utils.config.AGENT_META_KEYS``).

Settings come from the ``ua_*`` keys of ``dra_inference.yaml``.  The
pipeline's ``seen_top_k`` (passages per search) and run temperature apply; its
``max_iteration`` does not: this agent's turn cap is ``ua_max_turns``.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from reasoner_component import no_thinking_extra_body
from utils.config import InferenceConfig
from utils.text_utils import doc_text, doc_title
from deep_research_agents.prompts.uncertainty_aware import (
    render_format_error,
    render_system,
    render_user,
)

from .base_agent import BasicAgent

logger = logging.getLogger(__name__)


# ── Turn protocol ─────────────────────────────────────────────────────────────
# A turn is ``<think> … </think>`` followed by exactly one ``<search>`` or
# ``<answer>``.  :func:`parse_turn` returns the parsed pieces plus
#
#     errors    format failures, answered by a re-ask: no action, empty action.
#     warnings  repaired and logged: no <think>, a <certainty> or <information>
#               written by the policy (dropped).
#
# Anything after the first closing action tag is cut before parsing (see
# :func:`truncate_after_action`), so a second action or a trailing <think>
# never reaches the parser.  Parsing is regex-based rather than XML: model
# output is close to XML but not reliably well-formed.

ACTION_ANSWER = "answer"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_INJECTED_RE = re.compile(r"<(certainty|information)\b[^>]*>.*?</\1>", re.DOTALL)


@dataclass
class Turn:
    think: Optional[str]
    action: Optional[str]          # "search" | "answer" | None
    action_text: Optional[str]
    text: str                      # the turn as it goes into the history
    native_reasoning: str = ""     # extra <think> blocks (provider reasoning re-inlined)
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

    # The protocol's think is the last <think> before the action.  Earlier
    # blocks are provider reasoning the client re-inlined into the content.
    thinks = list(_THINK_RE.finditer(text))
    before = [m for m in thinks if am is None or m.start() < am.start()]
    think = None
    native = ""
    if before:
        think = before[-1].group(1).strip() or None
        native = "\n\n".join(m.group(1).strip() for m in before[:-1])
    if think is None:
        warnings.append("no <think> block")

    if action and action_text:
        parts = [f"<think>{think}</think>"] if think else []
        parts.append(f"<{action}>{action_text}</{action}>")
        history_text = "\n".join(parts)
    else:
        history_text = text.strip()

    return Turn(think=think, action=action, action_text=action_text, text=history_text,
                native_reasoning=native, errors=errors, warnings=warnings)


# ── Passage labels ────────────────────────────────────────────────────────────
# Every passage shown to the policy gets a label ``d1, d2, …`` that is unique in
# the run and stable: the same ``doc_id`` always gets the same label.

def _doc_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("doc_id") or doc.get("id") or "")


@dataclass
class DocRegistry:
    label_of: Dict[str, str] = field(default_factory=dict)   # doc_id -> dN
    doc_of: Dict[str, str] = field(default_factory=dict)     # dN -> doc_id

    def register(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Label *docs* (new ids get the next label); returns their labels."""
        labels: List[str] = []
        for doc in docs:
            did = _doc_id(doc)
            if not did:
                continue
            label = self.label_of.get(did)
            if label is None:
                label = f"d{len(self.label_of) + 1}"
                self.label_of[did] = label
                self.doc_of[label] = did
            if label not in labels:
                labels.append(label)
        return labels


def render_information(docs: List[Dict[str, Any]], registry: DocRegistry, max_chars: int) -> str:
    parts = []
    for doc in docs:
        label = registry.label_of.get(_doc_id(doc))
        if label is None:
            continue
        parts.append(f"[{label}] {doc_title(doc)}\n{doc_text(doc, max_length=max_chars)}")
    body = "\n\n".join(parts) if parts else "No documents retrieved."
    return f"<information>\n{body}\n</information>"


# ── Agent ─────────────────────────────────────────────────────────────────────

STOP_SEQUENCES = ["</search>", "</answer>"]

# Terminal step types.  "answer" is the normal end; the others are failures
# and are deliberately not in the trajectory evaluator's terminal set, so they
# count as runs that never finished.
END_ANSWERED = "answered"
END_MAX_TURNS = "max_turns"
END_FORMAT = "format_failure"
END_ERROR = "llm_error"


@dataclass
class UncertaintyAwareAgentConfig:
    # Generation steps before the run counts as a failure.
    max_turns: int = 8
    # Characters of one passage shown inside <information>.
    max_passage_chars: int = 1500
    # Re-asks after a malformed turn before the run stops.
    max_format_retries: int = 2
    # Output-token cap for one policy turn.
    max_tokens_per_call: int = 4096
    # Turn the backbone's own reasoning mode off, so the only <think> in a
    # turn is the protocol's.
    disable_native_thinking: bool = True


class UncertaintyAwareAgent(BasicAgent):
    AGENT_NAME = "UncertaintyAware"

    def __init__(self, llm_client, retriever: Optional[Any] = None, max_iteration: int = 100,
                 seen_top_k: int = 5, verbose: bool = True, max_turns: int = 8,
                 max_passage_chars: int = 1500, max_format_retries: int = 2,
                 max_tokens_per_call: int = 4096, disable_native_thinking: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self.cfg = UncertaintyAwareAgentConfig(
            max_turns=max_turns,
            max_passage_chars=max_passage_chars,
            max_format_retries=max_format_retries,
            max_tokens_per_call=max_tokens_per_call,
            disable_native_thinking=disable_native_thinking,
        )
        self.inference_config = InferenceConfig(api_type="chat_completion")
        self._extras: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    def _attach_uncertainty_stats(self, result: dict) -> None:
        # run_single's per-result hook, called before the trajectory log is
        # finalised; also used to attach the per-turn records.
        super()._attach_uncertainty_stats(result)
        result.update(self._extras)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int,
              stop: Optional[List[str]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        # Passing extra_body per call replaces the configured one (OpenRouter
        # provider pin), so the helper copies and extends it.
        body = no_thinking_extra_body(self.generator) if self.cfg.disable_native_thinking else None
        if body is not None:
            kwargs["extra_body"] = body
        return self.generator.complete(messages, **kwargs) or ""

    def answer_from_trajectory(self, original_query: str, trajectory: Any, instruction: str) -> str:
        """Intermediate answer with the policy's call settings (native
        thinking off, per-turn token cap), greedy."""
        messages = self._intermediate_answer_messages(trajectory, instruction)
        return self._call(messages, 0.0, self.cfg.max_tokens_per_call)

    @staticmethod
    def _messages(system: str, transcript: str) -> List[Dict[str, str]]:
        # The whole run so far is one user message, as in the SearchR1 family.
        return [{"role": "system", "content": system},
                {"role": "user", "content": transcript}]

    def _generate_turn(self, messages, t, temperature) -> Tuple[Optional[Turn], int, Optional[str]]:
        """One turn, re-asked on format errors.

        Returns (turn, re-asks, error): turn is None and error the message when
        the LLM call fails.
        """
        extra: List[Dict[str, str]] = []
        turn: Optional[Turn] = None
        for attempt in range(self.cfg.max_format_retries + 1):
            try:
                raw = self._call(messages + extra, temperature, self.cfg.max_tokens_per_call,
                                 stop=STOP_SEQUENCES)
            except Exception as exc:
                logger.warning("UncertaintyAware turn %d: LLM call failed: %s", t, exc)
                self._log_block(f"LLM call failed: {exc}", title=f"Turn {t}: error")
                return None, attempt, str(exc)
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
        return turn, self.cfg.max_format_retries, None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def inference(self, question: str, generation_temp: float = 0.6) -> tuple:
        cfg = self.cfg
        self._extras = {}

        system = render_system(cfg.max_turns)
        transcript = render_user(question)
        registry = DocRegistry()
        records: List[Dict[str, Any]] = []
        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""
        end = END_MAX_TURNS
        turns = 0
        searches = 0

        for t in range(cfg.max_turns):
            self._notify_progress("turn", t)
            turn, attempts, error = self._generate_turn(self._messages(system, transcript), t, generation_temp)
            if turn is None:
                end = END_ERROR
                self._record_step(reasoning_path, {
                    "iteration": t, "action_type": END_ERROR, "errors": error,
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
                self._vprint(t, "think", turn.think)

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
                    "iteration": t, "action_type": "answer", "think": turn.think,
                    "prediction": prediction, "generation": prediction,
                    "tokens": self._step_tokens(),
                })
                break

            # -- search, then the <certainty> tag that closes this iteration --
            query = turn.action_text
            self._notify_progress("search", t)
            self._vprint(t, "search", query)
            docs = self.retrieve_documents(query, original_query=question, reasoning=turn.think)
            shown = docs[:self.seen_top_k]
            labels = registry.register(shown)
            transcript += f"\n\n{turn.text}\n{render_information(shown, registry, cfg.max_passage_chars)}\n"
            step = {
                "iteration": t, "action_type": "search", "think": turn.think,
                "search_query": query, "docs": docs,
                "component_doc_ids": [d.get("doc_id", "") for d in shown],
                "labels": " ".join(labels),
                "tokens": self._step_tokens(),
            }

            tag = self._observe_step(
                query, shown, t, question,
                seen_docs=shown,
                trajectory=self._messages(system, transcript),
            )
            if tag:
                transcript += f"{tag}\n"
                step["certainty"] = tag
            self._record_step(reasoning_path, step)
            rec.update({"search": searches, "labels": labels, "certainty": tag})
            searches += 1

        if end == END_MAX_TURNS:
            self._record_step(reasoning_path, {"iteration": turns, "action_type": END_MAX_TURNS})

        self._extras = {
            "ua_records": records,
            "ua_doc_labels": dict(registry.doc_of),
            "ua_outcome": {
                "end": end,
                "answer": prediction or None,
                "num_turns": turns,
                "num_searches": searches,
                "format_retries": sum(r.get("format_retries", 0) for r in records),
                # Searches without a tag (the estimator failed or had nothing to show).
                "missing_certainty": sum(1 for r in records
                                         if r.get("action") == "search" and not r.get("certainty")),
                "config": asdict(cfg),
            },
        }
        return reasoning_path, prediction, turns
