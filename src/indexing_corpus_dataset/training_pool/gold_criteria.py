"""The criterion texts of a training pool's gold annotation, written at the
end of deep_qa step 2 and of the trqa_validation build (``fill_texts``),
before the split.

The gold annotation (``annotations.py``) gives per query its criteria with
their gold answers and documents, as far as the source states them; an LLM
(``gold_criteria.model``) writes the text of each criterion, in the words
of the question, from the question and the criteria's source structure:

    multi_hop   one text per criterion from its relations (placeholders E1,
                E2, ... for the criteria, the last one the answer):
                InfoSeek's tree hops, WebShaper's formalization variables;
                the writer orders the criteria so that each refers only to
                criteria before it (by placeholder, saved as the
                criterion's id: "the town of E1" -> "the town of c1"),
                the answer last; the row's criteria are reordered and
                renumbered in that order (``_order``, ``_with_texts``)
    set         the property phrase and the aggregation from the question,
                its property and its computation (TRQA): the member
                criteria "<member>: <property>" and the rest criterion
                "any other member: <property>" are built from the phrase,
                the aggregate criterion is the aggregation

The prompts are the files of ``prompts/`` (one system and one user prompt
per shape, the retry note).  The row keeps its keys and gets ``text`` per
criterion, ``texts_ok`` and ``errors``.  ``texts_ok`` is true when every
criterion has a text that names none of the gold answers (no leak of an
answer into a criterion; an answer the question names is exempt; in a set
query, no member, value or final answer in the phrase or the aggregation,
the member's name in its own criterion being the point; in a multi-hop
query, no number or capitalized word the question does not state, such as
a year the relations give, placeholders left out), or when
the query has no criteria (FRAMES: no LLM call); false when the texts
failed for good (a leak or an unstated word after the retry, no text for a
criterion); None when the writer call failed (an error, a reply that is not
a JSON object), so the next build run asks again.  A list that leaks or
holds unstated words, or is in an invalid order, is asked once more with
the problem named.  The number of criteria is the source's (no cap).

Memory: ``criterion_texts.jsonl`` gets one row per query whose texts are final
(``{"id", "structure", "texts", "texts_ok", "errors", "model", "prompt"}``,
and a multi-hop row's ``order``: the source's criterion indices in the
writer's order; ``structure`` is the source row's),
appended as the writer finishes them, so a stopped run keeps its texts.  A
run reuses them for a query whose criteria are unchanged (``structure``)
and writes the others.  A writer (``gold_criteria.model``) or prompts other
than those of the memory are refused (``check_writer``); the prompts are
hashed per query shape (``prompt_hash``).
"""

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.io_utils import drop_partial_line, iter_jsonl
from utils.text_utils import parse_json_object

from .annotations import AGGREGATE, CLOSED, REST, mentions, normalize
from .config import GoldCriteriaConfig, PoolPaths

log = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────

# The prompts are the .txt files of prompts/ next to this module, one system
# and one user prompt per query shape, and the notes of the retry.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


MULTI_HOP_SYSTEM = _read("multi_hop_system.txt")
MULTI_HOP_USER = _read("multi_hop_user.txt")
MULTI_MEMBER_SYSTEM = _read("multi_member_system.txt")
MULTI_MEMBER_USER = _read("multi_member_user.txt")
RETRY_NOTE = "\n\n" + _read("retry_note.txt")
UNSTATED_NOTE = "\n\n" + _read("retry_note_unstated.txt")
ORDER_NOTE = "\n\n" + _read("retry_note_order.txt")

# The prompts of each query shape.
PROMPTS = {"multi_hop": (MULTI_HOP_SYSTEM, MULTI_HOP_USER), "multi_member": (MULTI_MEMBER_SYSTEM, MULTI_MEMBER_USER)}


def prompt_hash(shape: str) -> str:
    """The hash of the prompts of query shape *shape*, saved with each row
    of the memory: a change of one shape's prompts leaves the texts of the
    other shapes valid."""
    notes = (RETRY_NOTE, UNSTATED_NOTE, *((ORDER_NOTE,) if shape == "multi_hop" else ()))
    text = "\n".join((*PROMPTS[shape], *notes))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ── The writer ───────────────────────────────────────────────────────────────

class CriteriaWriter:
    """The LLM that writes the criterion texts (native reasoning off)."""

    def __init__(self, model: str, max_tokens: int, llm: Any = None) -> None:
        """*llm*: a client with ``complete(messages, **kwargs) -> str`` for
        *model* (None: an API client, reasoning off)."""
        self.model = model
        if llm is None:
            from reasoner_component.factory import create_generator, disable_native_thinking
            llm = disable_native_thinking(create_generator(model, backend="api"))
        self._llm = llm
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


def _unstated(texts: List[str], question: str) -> List[str]:
    """The words of *texts* with a digit or a capital first letter (a year,
    a name: a detail a criterion holds only when the question states it)
    that are not words of *question* (``normalize``d)."""
    words = set(normalize(question).split())
    found = (w for t in texts for w in re.findall(r"\w+", t) if w[0].isupper() or any(ch.isdigit() for ch in w))
    return list(dict.fromkeys(w for w in found if not set(normalize(w).split()) <= words))


# The key of a row with texts that carries the writer's order to the
# memory (``fill_texts`` takes it out of the row).
_ORDER_KEY = "_order"


def _with_texts(row: dict, texts: List[Optional[str]], ok: Optional[bool], errors: List[str],
                order: Optional[List[int]] = None) -> Dict[str, Any]:
    """*row* with one text per criterion, ``texts_ok`` *ok* and *errors*.
    With *order* (a multi-hop row: the source's criterion indices in the
    writer's order, the *texts* in that order) the criteria are reordered
    and renumbered "c1", "c2", ... in it, the placeholders of their
    relations ("E<k>") renumbered with them."""
    source = row["criteria"]
    if order is None:
        criteria = [{**c, "text": t} for c, t in zip(source, texts)]
    else:
        position = {i + 1: n + 1 for n, i in enumerate(order)}
        renumber = lambda r: _PLACEHOLDER_RE.sub(lambda m: f"E{position.get(int(m.group(1)), m.group(1))}", r)
        criteria = [{**source[i], "id": f"c{n + 1}", "text": t, "relations": [renumber(r) for r in source[i]["relations"]]}
                    for n, (i, t) in enumerate(zip(order, texts))]
    return {**row, "criteria": criteria, "texts_ok": ok, "errors": errors,
            **({_ORDER_KEY: list(order)} if order is not None else {})}


def _failed(row: dict, errors: List[str]) -> Dict[str, Any]:
    return _with_texts(row, [None] * len(row["criteria"]), False, errors)


# A placeholder in a criterion text ("E2"): the writer refers to another
# entity of the chain by it.
_PLACEHOLDER_RE = re.compile(r"\bE(\d+)\b")


def _criterion_refs(text: str, n: int) -> str:
    """*text* with each placeholder "E<k>" (1 <= k <= *n*) written as the id
    of its criterion, "c<k>" (the belief model's criterion ids)."""
    return _PLACEHOLDER_RE.sub(lambda m: f"c{m.group(1)}" if 1 <= int(m.group(1)) <= n else m.group(0), text)


def _chain(criteria: List[dict]) -> str:
    """The chain lines of the multi-hop prompt: one per criterion, its
    placeholder and relations; the last criterion is the answer."""
    lines = []
    for k, c in enumerate(criteria, 1):
        tag = f"E{k} (the answer)" if k == len(criteria) else f"E{k}"
        lines.append(f"{tag}: {'; '.join(c['relations']) or '(no relation)'}")
    return "\n".join(lines)


def _order(entities: List[str], texts: List[str], n: int) -> List[str]:
    """The problems of the writer's order (*entities*, its placeholders in
    order, with their *texts*) for a chain of *n* criteria: each
    placeholder exactly once, the answer ("E<n>") last, and a text refers
    only to placeholders before it."""
    problems = []
    if sorted(entities) != sorted(f"E{k}" for k in range(1, n + 1)):
        problems.append(f"the placeholders were {', '.join(entities) or 'none'}, not E1 to E{n} once each")
        return problems
    if entities[-1] != f"E{n}":
        problems.append(f"the answer E{n} was not last")
    for k, (e, t) in enumerate(zip(entities, texts)):
        for ref in dict.fromkeys(f"E{m}" for m in _PLACEHOLDER_RE.findall(t)):
            if ref == e or ref in entities[k + 1:]:
                problems.append(f"{e} refers to {ref}" + (", itself" if ref == e else ", which comes after it"))
    return problems


def multi_hop_texts(row: dict, writer: "CriteriaWriter") -> Dict[str, Any]:
    """The texts of a multi-hop row (InfoSeek, WebShaper), in the
    writer's order (``_order``: each criterion refers only to criteria
    before it, the answer last)."""
    question, criteria, names_all = row["text"], row["criteria"], row["entity_names"]
    others = names_all or [a for c in criteria for a in c["answers"]]
    # An answer the question itself mentions (a named answer) cannot be kept out.
    names = [a for c in criteria for a in c["answers"] if not mentions(question, a, others)]
    errors: List[str] = []
    retry = ""
    for _ in range(2):
        data = writer.ask(MULTI_HOP_SYSTEM, MULTI_HOP_USER.format(question=question, chain=_chain(criteria),
                                                                  retry=retry))
        items = [i for i in data.get("criteria") or [] if isinstance(i, dict)]
        entities = [str(i.get("entity", "")).strip() for i in items]
        texts = [str(i.get("text", "")).strip() for i in items]
        if len(texts) != len(criteria) or not all(texts):
            errors.append("writer: no text for every criterion")
            retry = ""
            continue
        problems = _order(entities, texts, len(criteria))
        if problems:
            errors.append(f"writer: order: {'; '.join(problems)}")
            retry = ORDER_NOTE.format(problems="; ".join(problems))
            continue
        # The checks read the texts without their placeholders (a reference
        # to another entity, not a detail of the question).
        bare = [_PLACEHOLDER_RE.sub(" ", t) for t in texts]
        leaked = _leaks(bare, names, others)
        unstated = _unstated(bare, question)
        if leaked or unstated:
            retry = ""
            if leaked:
                errors.append(f"writer: texts name {leaked}")
                retry += RETRY_NOTE.format(names=", ".join(f'"{n}"' for n in leaked))
            if unstated:
                errors.append(f"writer: texts state {unstated}, the question does not")
                retry += UNSTATED_NOTE.format(words=", ".join(f'"{w}"' for w in unstated))
            continue
        order = [int(e[1:]) - 1 for e in entities]
        position = {f"E{i + 1}": f"E{n + 1}" for n, i in enumerate(order)}
        texts = [_PLACEHOLDER_RE.sub(lambda m: position.get(m.group(0), m.group(0)), t) for t in texts]
        return _with_texts(row, [_criterion_refs(t, len(criteria)) for t in texts], True, [], order)
    return _failed(row, errors)


def multi_member_texts(row: dict, writer: "CriteriaWriter") -> Dict[str, Any]:
    """The texts of a set row (TRQA): the property phrase and the
    aggregation from the writer, then "<member>: <phrase>" per member, "any
    other member: <phrase>" and the aggregation."""
    question = row["text"]
    members = [m["entity"] for m in row["members"]]
    # What the phrase and the aggregation must not name: the members (their
    # titles and TRQA's labels), their values and the final answer, unless
    # the question names them.
    values = [t for c in row["criteria"] if c["kind"] != REST for t in c["answers"]]
    others = row["entity_names"] + values
    names = [n for n in dict.fromkeys(others) if not mentions(question, n, others)]
    errors: List[str] = []
    retry = ""
    for _ in range(2):
        data = writer.ask(MULTI_MEMBER_SYSTEM, MULTI_MEMBER_USER.format(question=question, property=row["property"] or "",
                                                      aggregation=row["aggregation"] or "", retry=retry))
        prop_text = str(data.get("property") or "").strip()
        agg_text = str(data.get("aggregation") or "").strip()
        if not prop_text or not agg_text:
            errors.append("writer: no property or aggregation")
            continue
        leaked = _leaks([prop_text, agg_text], names, others)
        if leaked:
            errors.append(f"writer: texts name {leaked}")
            retry = RETRY_NOTE.format(names=", ".join(f'"{n}"' for n in leaked))
            continue
        texts = [f"{m}: {prop_text}" for m in members] + [f"any other member: {prop_text}", agg_text]
        return _with_texts(row, texts, True, [])
    return _failed(row, errors)


def _multi_member_layout(row: dict) -> bool:
    """A set row has one criterion per member, then the rest and the
    aggregate criterion."""
    kinds = [c["kind"] for c in row["criteria"]]
    return kinds == [CLOSED] * len(row["members"]) + [REST, AGGREGATE]


def criteria_texts(row: dict, writer: "CriteriaWriter") -> Dict[str, Any]:
    """The row with its criterion texts (``texts_ok``), or with
    ``texts_ok`` false and the reasons in ``errors``."""
    if not row["criteria"]:
        return _with_texts(row, [], True, [])
    if row["query_shape"] not in PROMPTS:
        return _failed(row, [f"query shape {row['query_shape']!r} has no writer prompts (one of {sorted(PROMPTS)}; "
                             "a row of 'set' predates the rename to 'multi_member': rebuild its annotations)"])
    if row["query_shape"] == "multi_member":
        if not _multi_member_layout(row):
            return _failed(row, ["multi-member query: expected one criterion per member, then the rest and the "
                                 "aggregate"])
        return multi_member_texts(row, writer)
    return multi_hop_texts(row, writer)


# ── Build ────────────────────────────────────────────────────────────────────

def _read_memory(paths: PoolPaths) -> Dict[str, dict]:
    """The memory rows by query id (the last row of a query wins)."""
    if not paths.texts.exists():
        return {}
    drop_partial_line(paths.texts)
    return {str(r["id"]): r for r in iter_jsonl(paths.texts)}


def require_api_key(gc: GoldCriteriaConfig, what: str) -> None:
    """Refuse to go on without the writer's API key when criterion texts
    are to be written (*what*: which ones); a pool is never written without
    its criterion texts."""
    if gc.model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"{what} need the writer {gc.model}: set OPENROUTER_API_KEY "
                         "(.env) or gold_criteria.model in the config")


def check_writer(gc: GoldCriteriaConfig, paths: PoolPaths) -> None:
    """Refuse another writer or other prompts than the texts in the memory
    (one pool, one writer); the API key is checked when texts are left to
    write (``fill_texts``)."""
    used ={(r["model"], r["prompt"]) for r in _read_memory(paths).values()}
    current = {(gc.model, prompt_hash(shape)) for shape in PROMPTS}
    other = sorted(used - current)
    if other:
        raise SystemExit(f"{paths.texts} holds texts of other writers or prompts ((model, prompt hash): {other}; "
                         f"this run: {sorted(current)}). Restore gold_criteria.model and the "
                         f"prompts, or remove {paths.texts} to write every text again")


def _structure(row: dict) -> str:
    """The hash of what the texts are written from: an earlier run's texts
    are kept while it is unchanged."""
    structure = [row["text"], row["query_shape"], row["property"], row["aggregation"],
                 [[c["id"], c["kind"], c["relations"], c["answers"]] for c in row["criteria"]],
                 [m["entity"] for m in row["members"]]]
    return hashlib.sha1(json.dumps(structure, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def fill_texts(gc: GoldCriteriaConfig, paths: PoolPaths, rows: List[dict]) -> List[dict]:
    """*rows* (the rows of the pool, split ``all``) with their criterion
    texts: those of the memory for a query whose criteria are unchanged,
    the writer's for the others (added to the memory once final)."""
    memory = _read_memory(paths)
    out: Dict[str, dict] = {}
    todo = []
    for row in rows:
        old = memory.get(str(row["id"]))
        if old is not None and old["structure"] == _structure(row):
            out[row["id"]] = _with_texts(row, old["texts"], old["texts_ok"], old["errors"], old.get("order"))
            out[row["id"]].pop(_ORDER_KEY, None)
        elif not row["criteria"]:
            out[row["id"]] = _with_texts(row, [], True, [])
        else:
            todo.append(row)
    print(f"[texts] {len(rows)} queries: {len(rows) - len(todo)} with final texts (kept or without criteria), "
          f"{len(todo)} to write with {gc.model}")
    if todo:
        require_api_key(gc, f"the criterion texts of {len(todo)} queries")
        writer = CriteriaWriter(gc.model, gc.max_tokens)

        def one(row: dict) -> Tuple[dict, dict]:
            try:
                done = criteria_texts(row, writer)
            except Exception as e:  # noqa: BLE001 - recorded; the next run asks again
                log.warning("criterion texts of %s failed: %s", row["id"], e)
                done = {**row, "errors": [f"writer: {e}"]}
            return row, done

        paths.texts.parent.mkdir(parents=True, exist_ok=True)
        pool = ThreadPoolExecutor(max_workers=max(1, gc.workers))
        try:
            with open(paths.texts, "a", encoding="utf-8") as fh:
                futures = [pool.submit(one, row) for row in todo]
                for future in tqdm(as_completed(futures), total=len(futures), desc="[texts]", dynamic_ncols=True):
                    # The structure is the source row's (the texts may
                    # reorder its criteria).
                    source, row = future.result()
                    order = row.pop(_ORDER_KEY, None)
                    out[row["id"]] = row
                    if row["texts_ok"] is not None:
                        fh.write(json.dumps({"id": row["id"], "structure": _structure(source),
                                             "texts": [c["text"] for c in row["criteria"]],
                                             "texts_ok": row["texts_ok"], "errors": row["errors"],
                                             "model": gc.model, "prompt": prompt_hash(row["query_shape"]),
                                             **({"order": order} if order is not None else {})},
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
