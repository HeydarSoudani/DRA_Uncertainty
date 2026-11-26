"""The facts of a question: what its gold passages must state.  The judge
(``judge.py``) pairs each fact with the passages of its gold pages and
labels each pair; a passage that states a fact is gold.

A fact holds only what the dataset states: no LLM names an entity or
writes a value.  Its value is known when the dataset states it (the judge
checks it, TRQA's prompt), else None (the judge extracts it).

Fact::

    {"id": "F1",
     "entity": str,             whose property the fact states (a name, or
                                a description of an entity without one)
     "property": str | None,    the property (None: the statement says it)
     "value": str | None,       its value; None: unknown (the statement has
                                a blank, ``BLANK``)
     "statement": str,          the fact as the judge reads it
     "pages": [page key, ...],  the gold pages that should state it (empty:
                                every gold page of the question)
     "refs": [...],             the source ids it is about (InfoSeek: the
                                claim's node ids; WebShaper: the triple's
                                variables, ``?`` the answer; FRAMES: none)
     "value_of": str | None}    WebShaper: the variable whose value an
                                unknown value is (else None)

Each dataset makes its own facts (``datasets/{dataset}/facts.py``):
infoseek_hard from the claims of its research tree, with no LLM; frames and
webshaper with one LLM call per question, the dataset's prompts and reply
parser (``FactSpec``), asked once more with the problems named
(``prompts/facts_retry_note.txt``) when the reply is invalid.  frames: the
LLM writes the property each gold page gives (the value stays unknown).
webshaper: the facts are its formalization's triples, written by code; the
LLM only maps each fact to the gold pages likely to state it (from the
pages' titles).

Memory (frames, webshaper): ``facts.jsonl`` gets one row per question whose
facts are final (``{"id", "structure", "facts", "ok", "errors", "model",
"prompt"}``), appended as the LLM finishes them.  A run reuses them for a
question whose user prompt is unchanged (``structure``: its hash, so the
question, the answer, the pages with their titles and, for webshaper, the
facts written by code); ``ok`` false
is final (the question is dropped), an API error is not saved (asked
again next run).  A reply that is not a JSON object is saved as
``{"id", "structure", "failed": true, "error", "model", "prompt"}``; after
``MAX_REPLY_FAILURES`` of them (one per run) the facts fail for good.  An
LLM (``facts.model``, ``--fact-model``) or prompts other than those of the
memory are refused (``check_fact_writer``).
"""

import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.io_utils import drop_partial_line, iter_jsonl

from ..config import DatasetPaths, FactsConfig
from ..llm import MAX_REPLY_FAILURES, JsonWriter, ReplyError, require_api_key
from ..pipeline.progress import CallBar

log = logging.getLogger(__name__)

# Where the value goes in the statement of a fact without a value.
BLANK = "___"

RETRY_NOTE = "\n\n" + (Path(__file__).resolve().parent / "prompts" / "facts_retry_note.txt").read_text(
    encoding="utf-8").strip()

# A reply parser: (record, pages, reply object) -> (facts, problems); the
# facts are valid only when the problems are empty.
Parse = Callable[[dict, List[dict], Dict[str, Any]], Tuple[List[dict], List[str]]]


@dataclass(frozen=True)
class FactSpec:
    """The fact LLM of a dataset: its name (``source``, in the memory's
    ``structure``), its prompts, its reply parser and the fields of its
    user prompt besides the question, the answer and the pages."""
    source: str
    system: str
    user: str
    parse: Parse
    fields: Callable[[dict], Dict[str, str]] = lambda record: {}

    def prompt_hash(self) -> str:
        text = "\n".join((self.system, self.user, RETRY_NOTE))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def read_prompts(folder: Path) -> Tuple[str, str]:
    """``(system, user)``: ``facts_system.txt`` and ``facts_user.txt`` of
    *folder* (a dataset's ``prompts/``)."""
    return tuple((folder / f"facts_{part}.txt").read_text(encoding="utf-8").strip() for part in ("system", "user"))


def fact(k: int, entity: str, prop: Optional[str], value: Optional[str], statement: str,
         pages: List[str], refs: List[str], value_of: Optional[str] = None) -> dict:
    return {"id": f"F{k}", "entity": entity, "property": prop, "value": value, "statement": statement,
            "pages": pages, "refs": refs, "value_of": value_of}


def property_statement(entity: str, prop: str, value: Optional[str]) -> str:
    """TRQA's statement of a property's value: "the [p] of [e] is [v]"."""
    return f"the [{prop}] of [{entity}] is [{value if value is not None else BLANK}]"


def _pages_block(pages: List[dict]) -> str:
    """The numbered gold pages of a prompt by their title (a web page
    without a fetched title: its URL)."""
    return "\n".join(f"{k}. " + (f"Wikipedia: {p['title']}" if p["kind"] == "wiki"
                                  else f"Web: {p.get('title') or p['url']}")
                     for k, p in enumerate(pages, 1))


def user_prompt(spec: FactSpec, record: dict, pages: List[dict]) -> str:
    """The user prompt of a question with its available gold pages *pages*
    (``{"key", "kind", "title", "url"}``, a wiki page without ``url``),
    with a ``{retry}`` slot for the retry note."""
    return spec.user.format(question=record["question"], answer=record["answer"], pages=_pages_block(pages),
                            **spec.fields(record), retry="{retry}")


def write_facts(spec: FactSpec, record: dict, pages: List[dict], writer: JsonWriter) -> dict:
    """The facts of a question with its available gold pages *pages*:
    ``{"facts", "ok", "errors"}``; ``ok`` false after a second invalid
    reply."""
    prompt = user_prompt(spec, record, pages)
    errors: List[str] = []
    retry = ""
    for _ in range(2):
        data = writer.ask(spec.system, prompt.replace("{retry}", retry))
        facts, problems = spec.parse(record, pages, data)
        if not problems and not facts:
            problems = ["no fact"]
        if not problems:
            return {"facts": facts, "ok": True, "errors": []}
        errors.append(f"writer: {'; '.join(problems)}")
        retry = RETRY_NOTE.format(problems="; ".join(problems))
    return {"facts": [], "ok": False, "errors": errors}


def _structure(spec: FactSpec, record: dict, pages: List[dict]) -> str:
    """The hash of what the facts are made from: the dataset and the user
    prompt (the question, the answer, the pages with their titles and the
    dataset's fields, e.g. webshaper's facts written by code)."""
    structure = [spec.source, user_prompt(spec, record, pages)]
    return hashlib.sha1(json.dumps(structure, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _memory_rows(paths: DatasetPaths) -> List[dict]:
    if not paths.facts.exists():
        return []
    drop_partial_line(paths.facts)
    return list(iter_jsonl(paths.facts))


def _read_memory(paths: DatasetPaths) -> Tuple[Dict[str, dict], Counter]:
    """The final rows by question id (the last row wins) and the failed
    replies per (question id, structure)."""
    final: Dict[str, dict] = {}
    failures: Counter = Counter()
    for r in _memory_rows(paths):
        if r.get("failed"):
            failures[(str(r["id"]), r["structure"])] += 1
        else:
            final[str(r["id"])] = r
    return final, failures


def _given_up() -> dict:
    return {"facts": [], "ok": False, "errors": [f"writer: {MAX_REPLY_FAILURES} replies were not a JSON object"]}


def check_fact_writer(gc: FactsConfig, paths: DatasetPaths, spec: FactSpec) -> None:
    """Refuse another writer or other prompts than the facts in the memory."""
    used = {(r["model"], r["prompt"]) for r in _memory_rows(paths)}
    current = (gc.model, spec.prompt_hash())
    other = sorted(used - {current})
    if other:
        raise SystemExit(f"{paths.facts} holds facts of other writers or prompts ((model, prompt hash): {other}; "
                         f"this run: {current}). Pass that --fact-model and restore the "
                         f"prompts, or remove {paths.facts} (and {paths.judgments}) to write every fact again")


def fill_facts(gc: FactsConfig, paths: DatasetPaths, spec: FactSpec,
               todo: List[Tuple[dict, List[dict]]], writer: Any = None) -> Dict[str, Optional[dict]]:
    """The facts of each question of *todo* (record, available gold pages):
    question id -> ``{"facts", "ok", "errors"}``, those of the
    memory for an unchanged input, the writer's for the others (added to the
    memory once final); None for a writer error (asked again next run).
    *writer*: a ``JsonWriter`` (None: one for ``facts.model``, ``--fact-model``)."""
    memory, failures = _read_memory(paths)
    out: Dict[str, Optional[dict]] = {}
    ask = []
    for record, pages in todo:
        old = memory.get(record["id"])
        structure = _structure(spec, record, pages)
        if old is not None and old["structure"] == structure:
            out[record["id"]] = old
        elif failures[(str(record["id"]), structure)] >= MAX_REPLY_FAILURES:
            out[record["id"]] = _given_up()
        else:
            ask.append((record, pages))
    print(f"[facts] {len(todo)} questions: {len(todo) - len(ask)} with final facts, "
          f"{len(ask)} to write with {gc.model}")
    if not ask:
        return out
    require_api_key(gc.model, f"the facts of {len(ask)} questions", "--fact-model")
    writer = writer or JsonWriter(gc.model, gc.max_tokens)

    def one(record: dict, pages: List[dict]) -> Tuple[dict, List[dict], Optional[dict], Optional[str]]:
        """The question, its pages, its facts (None: failed) and the
        error of a reply that is not a JSON object (else None)."""
        try:
            return record, pages, write_facts(spec, record, pages, writer), None
        except Exception as e:  # noqa: BLE001 - asked again next run
            log.warning("facts of %s failed: %s", record["id"], e)
            return record, pages, None, str(e)[:300] if isinstance(e, ReplyError) else None

    paths.facts.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=max(1, gc.workers))
    try:
        with open(paths.facts, "a", encoding="utf-8") as fh, \
                CallBar(len(ask), "[facts]", getattr(writer, "token_meter", None), unit="question") as bar:
            futures = [pool.submit(one, record, pages) for record, pages in ask]
            for future in as_completed(futures):
                record, pages, result, reply_error = future.result()
                bar.update("failed" if result is None else "ok" if result["ok"] else "invalid")
                out[record["id"]] = result
                structure = _structure(spec, record, pages)
                if reply_error is not None:
                    fh.write(json.dumps({"id": record["id"], "structure": structure, "failed": True,
                                         "error": reply_error, "model": gc.model, "prompt": spec.prompt_hash()},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                    if failures[(str(record["id"]), structure)] + 1 >= MAX_REPLY_FAILURES:
                        out[record["id"]] = _given_up()
                elif result is not None:
                    row = {"id": record["id"], "structure": structure, **result,
                           "model": gc.model, "prompt": spec.prompt_hash()}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    out[record["id"]] = row
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return out
