"""Online (streaming) trajectory logging.

The trajectory of a query used to reach disk only *after* the agent returned,
which meant an interrupted run — Ctrl-C, an OOM kill, a hung API call — left no
trace at all for the query that was in flight.  :class:`TrajectoryLogger` writes
each step the moment it is recorded, to two files that live side by side:

    trajectory/{qid}.jsonl   machine-readable, one line per step (unchanged
                             schema; the ``{"record": "meta", ...}`` header line
                             is appended *last*, once the result is known)
    trajectory/{qid}.md      human-readable, one block per step, flushed after
                             every write

Both are opened in truncate mode, so re-running a query after a crash replaces
its partial log rather than appending to it.

The ``meta`` line moving to the end of the JSONL needs no reader change:
:func:`utils.io_utils.load_result_from_saved_files` dispatches on ``record ==
"meta"`` per line, so its position is irrelevant.

The step-shaping helpers below are shared with
:func:`evaluation.trajectory.save_trajectory`, which writes the same JSONL at the end
of a query as a self-healing safety net.  Keeping one implementation is what
stops the online and end-of-run files from drifting apart.
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.config import AGENT_META_KEYS
from utils.text_utils import doc_text, doc_title

logger = logging.getLogger(__name__)

_BACKTICK_RUN = re.compile(r"`+")


# ---------------------------------------------------------------------------
# Step shaping (shared with evaluation.trajectory.writer)
# ---------------------------------------------------------------------------

def infer_action_type(step: Dict[str, Any]) -> str:
    """Return a normalised action type string for any agent step format."""
    # ReAct family — explicit action_type
    if "action_type" in step:
        return str(step["action_type"]).lower()

    # AgentCPM — explicit action field
    if "action" in step:
        return str(step["action"]).lower()

    # Infer from keys (SearchR1, ReSearch, StepSearch, SelfAsk…)
    if step.get("search_query") or step.get("query"):
        return "search"
    if "prediction" in step:
        return "predict"
    if "conclusion" in step:
        return "finish"
    if "follow_up" in step:
        return "followup"

    return "unknown"


def seen_doc_ids(step: Dict[str, Any]) -> List[str]:
    """Return the *seen* (top-k, injected-into-prompt) doc ids for a step.

    Reasoning agents (GLM, OSS, Tongyi, WebWeaver) store these under
    ``component_doc_ids``; AgentCPM under ``output.doc_ids``.  Already-cleaned
    trajectories loaded from disk expose them under ``seen_docs``.  The full
    surfaced ranking (``docs``/``all_docs``) is deliberately *not* written to
    the trajectory — it lives only in ``retrieval/surfaced/{qid}.trec``.
    """
    ids = step.get("component_doc_ids") or step.get("seen_docs")
    if not ids:
        output = step.get("output")
        if isinstance(output, dict):
            ids = output.get("doc_ids")
    return [did for did in (ids or []) if did]


def iter_label(step: Dict[str, Any], fallback_iter: int) -> str:
    """Build the per-line iteration label, e.g. ``"1"`` or ``"1.2"``.

    A single-query iteration is labelled by its iteration number alone; an
    iteration with multiple subqueries becomes ``{iteration}.{subquery}``
    (1-based subquery index), so the three searches of iteration 1 read
    ``1.1``, ``1.2``, ``1.3``.  Steps without an ``iteration`` field (terminal
    answer / force-answer steps) fall back to *fallback_iter*.
    """
    it = step.get("iteration")
    if it is None:
        return str(fallback_iter)
    sub = step.get("sub_iter")
    if sub is None:
        return str(it)
    return f"{it}.{int(sub) + 1}"


def step_to_line(step: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Reduce one trajectory step to a compact JSONL line.

    Search steps carry ``search_query`` + ``seen_docs`` (+ ``think`` only when
    present, i.e. on the first subquery of an iteration).  Terminal steps carry
    ``action_type`` + ``generation``.  The full surfaced doc list is dropped.

    ``action_type`` is emitted on the first subquery of an iteration only
    (``sub_iter`` is ``None`` or ``0``), mirroring how ``think`` appears only on
    the first subquery — the later subqueries (``1.2``, ``1.3``, …) share the
    same action and are left unannotated.
    """
    search_query = step.get("search_query")
    if search_query:
        line: Dict[str, Any] = {"iter": label}
        atype = step.get("action_type") or step.get("action")
        if atype and step.get("sub_iter") in (None, 0):
            line["action_type"] = atype
        if step.get("think"):
            line["think"] = step["think"]
        line["search_query"] = search_query
        seen = seen_doc_ids(step)
        if seen:
            line["seen_docs"] = seen
        if step.get("tokens") is not None:
            line["tokens"] = step["tokens"]
        if step.get("certainty"):
            line["certainty"] = step["certainty"]
        return line

    # Terminal / non-search step (answer, context_limit, max_iter_force, …).
    line = {"iter": label}
    atype = step.get("action_type") or step.get("action")
    if atype:
        line["action_type"] = atype
    if step.get("think"):
        line["think"] = step["think"]
    if step.get("generation"):
        line["generation"] = step["generation"]
    return line


def build_meta_line(query_id: str, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``{"record": "meta", ...}`` JSONL header for a finished query.

    Carries what the trajectory steps themselves don't: the final generation,
    the step/search/iteration counts, and the agent-specific resume payloads in
    :data:`utils.config.AGENT_META_KEYS` that are persisted nowhere else.
    """
    meta: Dict[str, Any] = {
        "record": "meta",
        "qid": query_id,
        "question": question,
        "generation": result.get("generation", ""),
        "num_steps": result.get("num_steps", 0),
        "num_searches": result.get("num_searches", 0),
        "num_iterations": result.get("num_iterations"),
    }
    for optional_key in AGENT_META_KEYS:
        if result.get(optional_key):
            meta[optional_key] = result[optional_key]
    return meta


def dump_line(obj: Dict[str, Any]) -> str:
    """Serialise one JSONL record (compact, never fails on odd value types)."""
    return json.dumps(obj, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

# Keys the step renderer gives a dedicated section to; everything else that is
# scalar falls through to the "Details" bullet list, so a step type added later
# still shows up in the log without touching this module.
_RENDERED_KEYS = frozenset({
    "think", "search_query", "query", "generation", "prediction", "conclusion",
    "observation", "docs", "all_docs", "component_doc_ids", "seen_docs",
    "action_type", "action", "phase", "iteration", "sub_iter", "tokens",
    "output", "input", "certainty",
})


#: Characters of document text shown per doc line.  Enough to recognise the
#: passage; the full text lives in ``retrieval/surfaced/{qid}.trec``.
DOC_SNIPPET_CHARS = 100


def _fence(text: str, lang: str = "") -> str:
    """Wrap *text* in a fenced code block long enough to survive its content."""
    text = str(text).strip()
    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _snippet(doc: Dict[str, Any], limit: int) -> str:
    """One-line preview of a document's text, collapsed and truncated."""
    text = doc_text(doc, max_length=None)
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def render_step_md(
    step: Dict[str, Any],
    *,
    step_no: int,
    label: str,
    doc_snippet_chars: int = DOC_SNIPPET_CHARS,
) -> str:
    """Render one trajectory step as a markdown block.

    Think text and search queries are written in full — this file is where you
    go when a run misbehaved, and a truncated thought is exactly the part you
    would have wanted.  Documents get ``doc_id``, title and a short snippet;
    their full text lives in ``retrieval/surfaced/{qid}.trec``.
    """
    action = infer_action_type(step)
    phase = step.get("phase")
    heading = f"## Step {step_no} · iter {label} · {action}"
    if phase and str(phase) != action:
        heading += f" · {phase}"
    parts: List[str] = [heading]

    think = step.get("think")
    if think:
        parts.append(f"**Think**\n\n{str(think).strip()}")

    query = step.get("search_query") or step.get("query")
    if query:
        parts.append(f"**Query**\n\n{_fence(query)}")

    # Documents: list the seen (prompt-visible) ids with titles, and say how
    # many were surfaced behind them.
    docs = step.get("docs") or step.get("all_docs") or []
    if isinstance(docs, list) and docs and isinstance(docs[0], dict):
        by_id = {(d.get("doc_id") or d.get("id") or ""): d for d in docs}
        seen = seen_doc_ids(step)
        if not seen:
            seen = [d.get("doc_id") or d.get("id") or "?" for d in docs[:10]]
        lines = []
        for rank, did in enumerate(seen, 1):
            doc = by_id.get(did) or {}
            # Two trailing spaces = a hard break, so the snippet stays inside
            # the list item instead of merging into the title's paragraph.
            lines.append(f"{rank}. `{did}` — {doc_title(doc, 'Untitled')}  ")
            snippet = _snippet(doc, doc_snippet_chars)
            lines.append(f"   {snippet}" if snippet else "   *(no text)*")
        parts.append(
            f"**Docs** ({len(seen)} seen / {len(docs)} surfaced)\n\n"
            + "\n".join(lines)
        )
    elif seen_doc_ids(step):
        seen = seen_doc_ids(step)
        lines = [f"{rank}. `{did}`" for rank, did in enumerate(seen, 1)]
        parts.append(f"**Docs** ({len(seen)} seen)\n\n" + "\n".join(lines))

    observation = step.get("observation")
    if observation:
        parts.append(f"**Observation**\n\n{str(observation).strip()}")

    certainty = step.get("certainty")
    if certainty:
        parts.append(f"**Certainty** (injected)\n\n{_fence(certainty, 'xml')}")

    for key, heading_text in (("prediction", "Prediction"),
                              ("conclusion", "Conclusion"),
                              ("generation", "Generation")):
        value = step.get(key)
        if value:
            parts.append(f"**{heading_text}**\n\n{str(value).strip()}")

    details = [
        f"- {key}: {value}"
        for key, value in step.items()
        if key not in _RENDERED_KEYS and not isinstance(value, (dict, list))
    ]
    if details:
        parts.append("**Details**\n\n" + "\n".join(details))

    tokens = step.get("tokens")
    if tokens:
        parts.append(f"*tokens: {tokens}*")

    return "\n\n".join(parts) + "\n\n"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """Stream one query's trajectory to ``{qid}.jsonl`` and ``{qid}.md``.

    Both files are truncated on :meth:`start`, so a query re-run after an
    interrupted attempt replaces its own partial log.  Every write is flushed,
    which is the whole point: whatever the agent had done before it was killed
    is on disk and readable.

    Typical use, from ``run_single``::

        logger = TrajectoryLogger(traj_dir, qid, question, agent_name="UncertaintyAware")
        logger.start()
        try:
            ...                      # agent calls log_step()/log_block()
            logger.finalize(result)
        finally:
            logger.close()

    A logger whose files cannot be opened degrades to a no-op rather than
    taking the run down with it — losing the log is not worth losing the run.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        query_id: str,
        question: str,
        *,
        agent_name: str = "",
        model: str = "",
        answer: Optional[str] = None,
        doc_snippet_chars: int = DOC_SNIPPET_CHARS,
    ) -> None:
        self.output_dir = Path(str(output_dir))
        self.query_id = query_id
        self.question = question
        self.agent_name = agent_name
        self.model = model
        self.answer = answer
        self.doc_snippet_chars = doc_snippet_chars

        self._jsonl = None
        self._md = None
        self._last_iter = 0
        self._step_no = 0
        self._t0 = time.time()
        self._finalized = False
        self._disabled = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def md_path(self) -> Path:
        return self.output_dir / f"{self.query_id}.md"

    @property
    def jsonl_path(self) -> Path:
        return self.output_dir / f"{self.query_id}.jsonl"

    def start(self) -> "TrajectoryLogger":
        """Open both files (truncating) and write the markdown header."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(self.jsonl_path, "w", encoding="utf-8")
            self._md = open(self.md_path, "w", encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Trajectory logging disabled for {self.query_id}: {exc}")
            self._disable()
            return self

        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = [f"# {self.query_id}", "", "**Question**", "", str(self.question).strip(), ""]
        if self.answer:
            header += ["**Gold answer**", "", str(self.answer).strip(), ""]
        meta_bits = [b for b in (self.agent_name, self.model, f"started {started}") if b]
        header += [" · ".join(meta_bits), "", "---", "", ""]
        self._write_md("\n".join(header))
        return self

    def close(self, error: Optional[BaseException] = None) -> None:
        """Close both files, stamping the markdown when the run never finished."""
        if self._disabled:
            return
        if not self._finalized:
            reason = f"Error: {error}" if error is not None else "No result was produced."
            self._write_md(
                "---\n\n"
                f"## ⚠ Incomplete\n\n"
                f"Ended after {self._step_no} step(s) in {_fmt_elapsed(time.time() - self._t0)} "
                f"without a result. {reason}\n"
            )
        for handle in (self._jsonl, self._md):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self._jsonl = self._md = None

    def __enter__(self) -> "TrajectoryLogger":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close(error=exc)
        return False

    # -- writing -------------------------------------------------------------

    def log_step(self, step: Dict[str, Any]) -> None:
        """Record one trajectory step to both files and flush.

        Called the moment the step dict is complete — i.e. wherever the agent
        appends to its ``reasoning_path`` — so a hang inside a long pass still
        leaves every step that preceded it on disk.
        """
        if self._disabled or self._md is None:
            return
        it = step.get("iteration")
        if it is not None:
            self._last_iter = int(it)
        else:
            self._last_iter += 1
        label = iter_label(step, self._last_iter)
        self._step_no += 1

        self._write_jsonl(step_to_line(step, label))
        self._write_md(render_step_md(
            step, step_no=self._step_no, label=label,
            doc_snippet_chars=self.doc_snippet_chars,
        ))

    def log_block(self, text: str, *, title: Optional[str] = None) -> None:
        """Write a markdown-only note between steps.

        For things that never enter the trajectory but explain it: phase
        banners, parse errors and retries.  Deliberately does not
        touch the JSONL, which stays byte-identical to what the end-of-run
        writer produces.
        """
        if self._disabled or self._md is None:
            return
        block = f"### {title}\n\n{text}\n\n" if title else f"{text}\n\n"
        self._write_md(block)

    def finalize(self, result: Dict[str, Any]) -> None:
        """Append the JSONL ``meta`` line and the markdown footer.

        The meta line goes last because its contents (generation, counts) are
        only known now; ``load_result_from_saved_files`` dispatches on ``record``
        per line, so its position does not matter.
        """
        if self._disabled or self._md is None:
            return
        self._write_jsonl(build_meta_line(self.query_id, self.question, result))

        counts = [
            f"**Steps** {result.get('num_steps', 0)}",
            f"**Searches** {result.get('num_searches', 0)}",
        ]
        if result.get("num_iterations") is not None:
            counts.append(f"**Iterations** {result['num_iterations']}")
        counts.append(f"**Elapsed** {_fmt_elapsed(time.time() - self._t0)}")

        footer = ["---", "", "## Result", "", " · ".join(counts), ""]

        usage = result.get("token_usage") or {}
        if usage:
            footer += [
                "**Tokens** "
                f"in {usage.get('input_tokens', 0):,} · "
                f"out {usage.get('output_tokens', 0):,} · "
                f"total {usage.get('total_tokens', 0):,} · "
                f"calls {usage.get('num_calls', 0)}",
                "",
            ]

        generation = result.get("generation") or ""
        if generation:
            footer += ["**Generation**", "", str(generation).strip(), ""]

        self._write_md("\n".join(footer))
        self._finalized = True

    # -- internals -----------------------------------------------------------

    def _write_jsonl(self, obj: Dict[str, Any]) -> None:
        if self._jsonl is None:
            return
        try:
            self._jsonl.write(dump_line(obj) + "\n")
            self._jsonl.flush()
        except Exception as exc:
            logger.warning(f"Trajectory JSONL write failed for {self.query_id}: {exc}")
            self._disable()

    def _write_md(self, text: str) -> None:
        if self._md is None:
            return
        try:
            self._md.write(text)
            self._md.flush()
        except Exception as exc:
            logger.warning(f"Trajectory MD write failed for {self.query_id}: {exc}")
            self._disable()

    def _disable(self) -> None:
        """Stop logging for this query; the run itself carries on regardless."""
        self._disabled = True
        for handle in (self._jsonl, self._md):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self._jsonl = self._md = None
