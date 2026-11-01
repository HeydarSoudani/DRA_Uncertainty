"""The criterion texts of a training pool's gold annotation, written at the
end of deep_qa step 2 and of the trqa_validation build (``fill_texts``),
before the split.

The gold annotation (``annotations.py``) gives per query its criteria with
their gold answers and documents, as far as the source states them; an LLM
(``gold_criteria.model``) writes the text of each criterion, in the words
of the question, from the question and the criteria's source structure:

    multi_hop   one text per criterion from its relations (placeholders E1,
                E2, ... for the criteria, the last one the answer):
                InfoSeek's tree hops, WebShaper's formalization variables
    set         the set ("the complete list of ...") and the rest criterion
                ("any other member: <property>") from the property: TRQA

The row keeps its keys and gets ``text`` per criterion and ``texts_ok``,
``errors``, ``model``, ``prompt``.  ``texts_ok`` is true when every
criterion has a text that names none of the gold answers (no leak of an
answer into a criterion; an answer the question names is exempt), or when
the query has no criteria (FRAMES: no LLM call); false when the texts
failed for good (a leak after the retry, no text for a criterion); None
when the writer call failed (an error, a reply that is not a JSON object),
so the next build run asks again.  A list that leaks is asked once more
with the leak named.  The number of criteria is the source's (no cap).

Memory: ``gold/texts.jsonl`` gets one row per query whose texts are final
(``{"id", "structure", "texts", "texts_ok", "errors", "model", "prompt"}``),
appended as the writer finishes them, so a stopped run keeps its texts.  A
run reuses them for a query whose criteria are unchanged (``structure``)
and writes the others.  A writer (``gold_criteria.model``) or prompts other
than those of the memory are refused (``check_writer``).
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from utils.io_utils import drop_partial_line, iter_jsonl
from utils.text_utils import parse_json_object

from .annotations import mentions
from .config import GoldCriteriaConfig, PoolPaths

log = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────

MULTI_HOP_SYSTEM = """\
You are a criteria writer. Given a question and the chain of entities that leads to its answer, your task is to write one criterion for each entity marked with a placeholder (E1, E2, ...): the part of the question that describes that entity.

== Rules ==
1. Write each criterion in the question's own words: keep the entities the question names, and describe each placeholder entity as the question describes it, never by its name. No criterion may contain the name of any placeholder entity.
2. Every criterion is self-contained: it describes its entity through the whole chain that the question gives for it, not by pointing back with words such as "that city" or "the above".
3. Start each criterion with the entity's type in the question's words ("the city where ...", "the botanist who ..."), or "the entity" when the question gives no type.
4. Use only what the question states; the relations listed are there to tell which part of the question describes which entity.

== Output Format ==
Respond with ONLY a raw JSON object, no markdown, no code blocks, with one key:
- "criteria": One object per placeholder, in the given order: {"entity": "E1", "text": "..."}.

Example question: Which lighthouse was designed by an engineer who was born in the town where the Varn Folk Museum was founded?
Example chain:
E1: Varn Folk Museum was founded in E1
E2: E2 was born in E1
E3 (the answer): E3 was designed by E2
Example output:
{"criteria": [
  {"entity": "E1", "text": "the town where the Varn Folk Museum was founded"},
  {"entity": "E2", "text": "the engineer who was born in the town where the Varn Folk Museum was founded"},
  {"entity": "E3", "text": "the lighthouse that was designed by an engineer who was born in the town where the Varn Folk Museum was founded"}
]}"""

MULTI_HOP_USER = """\
Question: {question}

Chain:
{chain}

Write one criterion per placeholder.{retry}"""

SET_SYSTEM = """\
You are a criteria writer. Given a question whose answer is computed over every member of a set (a count, a sum, an average, the earliest or latest value), and the property the question computes, your task is to write the question's set and property in its own words.

== Rules ==
1. "set": the complete set of members, starting with "the complete list of", with the conditions that define the set in the question's words. When the question counts or selects the members that meet a condition, that condition is not part of the set: it is the property.
2. "property": what the question computes or counts for each member, in the question's words, without the aggregation (no "average", "how many", "the two largest").
3. Name no member of the set.

== Output Format ==
Respond with ONLY a raw JSON object, no markdown, no code blocks, with two keys: "set" and "property".

Example question: How many novels by Jules Verne that were published before 1870 were adapted into films?
Example property: adapted into a film
Example output:
{"set": "the complete list of the novels by Jules Verne that were published before 1870", "property": "adapted into a film"}"""

SET_USER = """\
Question: {question}
Property: {property}{retry}"""

RETRY_NOTE = "\n\nYour earlier criteria named {names}; describe these entities as the question does, without their names."


def prompt_hash() -> str:
    """The hash of the prompts, saved with each row."""
    text = "\n".join((MULTI_HOP_SYSTEM, MULTI_HOP_USER, SET_SYSTEM, SET_USER, RETRY_NOTE))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ── The writer ───────────────────────────────────────────────────────────────

class CriteriaWriter:
    """The LLM that writes the criterion texts (native reasoning off)."""

    def __init__(self, model: str, max_tokens: int) -> None:
        from reasoner_component.factory import create_generator, disable_native_thinking

        self.model = model
        self._llm = disable_native_thinking(create_generator(model, backend="api"))
        self._max_tokens = max_tokens

    def ask(self, system: str, user: str) -> Dict[str, Any]:
        """The reply as a JSON object; a reply that is none raises (the
        query is asked again by the next run)."""
        raw = self._llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 max_tokens=self._max_tokens, temperature=0.0)
        data = parse_json_object(raw or "")
        if not isinstance(data, dict):
            raise ValueError(f"reply is not a JSON object: {(raw or '')[:200]!r}")
        return data


def _leaks(texts: List[str], names: List[str], others: List[str]) -> List[str]:
    """The *names* some text of *texts* mentions (``mentions``, outside the
    longer names of *others*)."""
    return [n for n in names if any(mentions(t, n, others) for t in texts)]


def _texts(row: dict, texts: List[str], errors: List[str]) -> Dict[str, Any]:
    criteria = [{**c, "text": t} for c, t in zip(row["criteria"], texts)]
    return {**row, "criteria": criteria, "texts_ok": True, "errors": errors}


def _failed(row: dict, errors: List[str]) -> Dict[str, Any]:
    return {**row, "criteria": [{**c, "text": None} for c in row["criteria"]], "texts_ok": False, "errors": errors}


def _chain(criteria: List[dict]) -> str:
    """The chain lines of the multi-hop prompt: one per criterion, its
    placeholder and relations; the last criterion is the answer."""
    lines = []
    for k, c in enumerate(criteria, 1):
        tag = f"E{k} (the answer)" if k == len(criteria) else f"E{k}"
        lines.append(f"{tag}: {'; '.join(c['relations']) or '(no relation)'}")
    return "\n".join(lines)


def multi_hop_texts(row: dict, writer: "CriteriaWriter") -> Dict[str, Any]:
    """The texts of a multi-hop row (InfoSeek, WebShaper)."""
    question, criteria, names_all = row["query"], row["criteria"], row["entity_names"]
    others = names_all or [a for c in criteria for a in c["answers"]]
    # An answer the question itself mentions (a named answer) cannot be kept out.
    names = [a for c in criteria for a in c["answers"] if not mentions(question, a, others)]
    errors: List[str] = []
    retry = ""
    for _ in range(2):
        data = writer.ask(MULTI_HOP_SYSTEM, MULTI_HOP_USER.format(question=question, chain=_chain(criteria),
                                                                  retry=retry))
        items = data.get("criteria")
        by_label = {str(i.get("entity", "")).strip(): str(i.get("text", "")).strip()
                    for i in items or [] if isinstance(i, dict)}
        texts = [by_label.get(f"E{k}", "") for k in range(1, len(criteria) + 1)]
        if not all(texts):
            errors.append("writer: no text for every criterion")
            continue
        leaked = _leaks(texts, names, others)
        if leaked:
            errors.append(f"writer: texts name {leaked}")
            retry = RETRY_NOTE.format(names=", ".join(f'"{n}"' for n in leaked))
            continue
        return _texts(row, texts, [])
    return _failed(row, errors)


def set_texts(row: dict, writer: "CriteriaWriter") -> Dict[str, Any]:
    """The texts of a set row (TRQA): the set and the rest criterion."""
    question = row["query"]
    members = [m["entity"] for m in row["members"]]
    errors: List[str] = []
    retry = ""
    for _ in range(2):
        data = writer.ask(SET_SYSTEM, SET_USER.format(question=question, property=row["property"] or "",
                                                      retry=retry))
        set_text = str(data.get("set") or "").strip()
        prop_text = str(data.get("property") or "").strip()
        if not set_text or not prop_text:
            errors.append("writer: no set or property")
            continue
        leaked = [m for m in _leaks([set_text, prop_text], members, members) if not mentions(question, m, members)]
        if leaked:
            errors.append(f"writer: texts name {leaked}")
            retry = RETRY_NOTE.format(names=", ".join(f'"{n}"' for n in leaked))
            continue
        return _texts(row, [set_text, f"any other member: {prop_text}"], [])
    return _failed(row, errors)


def criteria_texts(row: dict, writer: "CriteriaWriter") -> Dict[str, Any]:
    """The row with its criterion texts (``texts_ok``), or with
    ``texts_ok`` false and the reasons in ``errors``."""
    if not row["criteria"]:
        return _texts(row, [], [])
    if row["query_shape"] == "set":
        if len(row["criteria"]) != 2:
            return _failed(row, [f"set query with {len(row['criteria'])} criteria, expected 2"])
        return set_texts(row, writer)
    return multi_hop_texts(row, writer)


# ── Build ────────────────────────────────────────────────────────────────────

def _read_memory(paths: PoolPaths) -> Dict[str, dict]:
    """The memory rows by query id (the last row of a query wins)."""
    if not paths.texts.exists():
        return {}
    drop_partial_line(paths.texts)
    return {str(r["id"]): r for r in iter_jsonl(paths.texts)}


def check_writer(gc: GoldCriteriaConfig, paths: PoolPaths) -> None:
    """Refuse to build without the writer's API key (a pool is never written
    without its criterion texts), or with another writer or other prompts
    than the texts in the memory (one pool, one writer)."""
    if gc.model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"the criterion texts need the writer {gc.model}: set OPENROUTER_API_KEY "
                         "(.env) or gold_criteria.model in the config")
    used = {(r["model"], r["prompt"]) for r in _read_memory(paths).values()}
    other = sorted(used - {(gc.model, prompt_hash())})
    if other:
        raise SystemExit(f"{paths.texts} holds texts of other writers or prompts ((model, prompt hash): {other}; "
                         f"this run: ({gc.model!r}, {prompt_hash()!r})). Restore gold_criteria.model and the "
                         f"prompts, or remove {paths.texts} to write every text again")


def _structure(row: dict) -> str:
    """The hash of what the texts are written from: an earlier run's texts
    are kept while it is unchanged."""
    structure = [row["query"], row["query_shape"], row["property"],
                 [[c["id"], c["kind"], c["relations"], c["answers"]] for c in row["criteria"]],
                 [m["entity"] for m in row["members"]]]
    return hashlib.sha1(json.dumps(structure, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def fill_texts(gc: GoldCriteriaConfig, paths: PoolPaths, rows: List[dict]) -> List[dict]:
    """*rows* (the annotation rows of split ``all``) with their criterion
    texts: those of the memory for a query whose criteria are unchanged,
    the writer's for the others (added to the memory once final)."""
    memory = _read_memory(paths)
    out: Dict[str, dict] = {}
    todo = []
    for row in rows:
        old = memory.get(str(row["id"]))
        if old is not None and old["structure"] == _structure(row):
            out[row["id"]] = {**row, "criteria": [{**c, "text": t} for c, t in zip(row["criteria"], old["texts"])],
                              **{k: old[k] for k in ("texts_ok", "errors", "model", "prompt")}}
        elif not row["criteria"]:
            out[row["id"]] = _texts(row, [], [])
        else:
            todo.append(row)
    print(f"[texts] {len(rows)} queries: {len(rows) - len(todo)} with final texts (kept or without criteria), "
          f"{len(todo)} to write with {gc.model}")
    if todo:
        writer = CriteriaWriter(gc.model, gc.max_tokens)
        prompt = prompt_hash()

        def one(row: dict) -> dict:
            try:
                done = criteria_texts(row, writer)
            except Exception as e:  # noqa: BLE001 - recorded; the next run asks again
                log.warning("criterion texts of %s failed: %s", row["id"], e)
                return {**row, "errors": [f"writer: {e}"]}
            return {**done, "model": gc.model, "prompt": prompt}

        paths.texts.parent.mkdir(parents=True, exist_ok=True)
        pool = ThreadPoolExecutor(max_workers=max(1, gc.workers))
        try:
            with open(paths.texts, "a", encoding="utf-8") as fh:
                futures = [pool.submit(one, row) for row in todo]
                for future in tqdm(as_completed(futures), total=len(futures), desc="[texts]", dynamic_ncols=True):
                    row = future.result()
                    out[row["id"]] = row
                    if row["texts_ok"] is not None:
                        fh.write(json.dumps({"id": row["id"], "structure": _structure(row),
                                             "texts": [c["text"] for c in row["criteria"]],
                                             **{k: row[k] for k in ("texts_ok", "errors", "model", "prompt")}},
                                            ensure_ascii=False) + "\n")
                        fh.flush()
        finally:
            # A stopped run (Ctrl-C) does not wait for the calls not started.
            pool.shutdown(wait=True, cancel_futures=True)
    result = [out[row["id"]] for row in rows]
    counts = {k: sum(r["texts_ok"] is v for r in result) for k, v in (("ok", True), ("failed", False),
                                                                      ("retry", None))}
    print(f"[texts] texts ok {counts['ok']}, failed {counts['failed']}, "
          f"retry next run {counts['retry']} (writer errors)")
    return result
