"""The gold passages of a question: an LLM judge (``gold_passages.model``,
TRQA's judge gpt-4o-mini at temperature 0) reads one (fact, passage) pair
per call (``facts.py``: a fact is an entity, a property, a statement and a
value) and labels it, TRQA's property check (mahta-r/total-recall-qa,
``prompt_combined.py``):

    YES-SAME       the passage states the fact's value for the entity
    YES-DIFFERENT  it states another value
    NO-RELATED     it states no value, its topic fits the fact
    NO-UNRELATED   it states no value, its topic does not fit

A fact without a value (its statement has a blank, ``facts.BLANK``) has its
own prompt with the labels YES (the passage states the value, whatever it
is), NO-RELATED and NO-UNRELATED.  The prompts are
``prompts/judge_{value,blank}_system.txt`` and ``prompts/judge_user.txt``.

Grade of a pair (``grade``): 2 for YES-SAME or YES (the passage is gold
for the fact), 1 for NO-RELATED, 0 otherwise (``NO_LABEL`` included).

Memory: ``gold_passages.jsonl`` gets one row per judged pair (``{"fact":
fact_key, "doc_id", "label", "model", "prompt"}``), appended as the judge
answers; a pair is judged once, whatever question its fact belongs to.  A
reply without a label is saved with ``label`` null and the reply: the
judge answers at temperature 0, so after ``MAX_REPLY_FAILURES`` such
replies (one per run) the pair's label is final, ``NO_LABEL``.  A failed
call (an API error) is not saved (asked again next run).  A judge or
prompts other than those of the memory are refused (``check_judge``).
"""

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from utils.io_utils import drop_partial_line, iter_jsonl

from ..annotation.belief_texts import MAX_REPLY_FAILURES
from ..config import DatasetPaths, GoldPassagesConfig
from ..corpus.texts import passage_texts

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


VALUE_SYSTEM = _read("judge_value_system.txt")
BLANK_SYSTEM = _read("judge_blank_system.txt")
JUDGE_USER = _read("judge_user.txt")

VALUE_LABELS = ("YES-SAME", "YES-DIFFERENT", "NO-RELATED", "NO-UNRELATED")
BLANK_LABELS = ("YES", "NO-RELATED", "NO-UNRELATED")
GOLD_LABELS = ("YES-SAME", "YES")
# The final label of a pair whose replies held no label MAX_REPLY_FAILURES
# times (grade 0).
NO_LABEL = "NO-LABEL"
RELATED_LABELS = ("NO-RELATED",)
# A label in a reply, the longer ones first ("NO-UNRELATED" holds "RELATED").
_LABEL_RE = re.compile(r"\b(YES-SAME|YES-DIFFERENT|NO-UNRELATED|NO-RELATED|YES)\b")


def judge_prompt_hash() -> str:
    text = "\n".join((VALUE_SYSTEM, BLANK_SYSTEM, JUDGE_USER))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def grade(label: Optional[str]) -> int:
    """2: the passage is gold for the fact; 1: on its topic; 0: neither."""
    if label in GOLD_LABELS:
        return 2
    return 1 if label in RELATED_LABELS else 0


def fact_key(fact: dict) -> str:
    """A fact as the judge reads it: two questions' equal facts share their
    judgments."""
    text = json.dumps([fact["entity"], fact["property"], fact["value"], fact["statement"]], ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def fact_block(fact: dict) -> str:
    """The fact lines of the judge's input (the examples' format)."""
    lines = [f"ENTITY: [{fact['entity']}]"]
    if fact["property"]:
        lines.append(f"PROPERTY: [{fact['property']}]")
    lines.append(f"STATEMENT: {fact['statement']}")
    if fact["value"] is not None:
        lines.append(f"VALUE: [{fact['value']}]")
    return "\n".join(lines)


def parse_label(reply: str, labels: Tuple[str, ...]) -> Optional[str]:
    """The label of a reply: the reply itself, else the first label of
    *labels* in it; None without one.  For the blank-value prompt
    (*labels* with YES), a YES-SAME or YES-DIFFERENT of the value prompt
    is a YES: the passage states a value."""
    text = (reply or "").strip().strip("\"'`.").upper()
    for m in ([text] if text in labels else []) + [m.group(1) for m in _LABEL_RE.finditer(text)]:
        if m in labels:
            return m
        if m.startswith("YES-") and "YES" in labels:
            return "YES"
    return None


class Judge:
    """The LLM that labels (fact, passage) pairs."""

    def __init__(self, model: str, max_tokens: int, llm: Any = None) -> None:
        """*llm*: a client with ``complete(messages, **kwargs) -> str`` for
        *model* (None: an API client)."""
        if llm is None:
            from reasoner_component.factory import create_generator
            llm = create_generator(model, backend="api")
        self._llm = llm
        self._max_tokens = max_tokens

    def label(self, fact: dict, passage: Dict[str, str]) -> Tuple[Optional[str], str]:
        """``(label, reply)``; the label None when the reply holds none."""
        blank = fact["value"] is None
        system, labels = (BLANK_SYSTEM, BLANK_LABELS) if blank else (VALUE_SYSTEM, VALUE_LABELS)
        user = JUDGE_USER.format(fact=fact_block(fact), title=passage["title"], text=passage["text"])
        reply = self._llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   max_tokens=self._max_tokens, temperature=0.0)
        return parse_label(reply, labels), reply or ""


def _memory_rows(paths: DatasetPaths) -> List[dict]:
    if not paths.judgments.exists():
        return []
    drop_partial_line(paths.judgments)
    return list(iter_jsonl(paths.judgments))


def _read_memory(paths: DatasetPaths) -> Tuple[Dict[Tuple[str, str], str], Counter]:
    """The final label of each judged pair (``NO_LABEL`` after
    ``MAX_REPLY_FAILURES`` replies without one) and the replies without a
    label of the others."""
    labels: Dict[Tuple[str, str], str] = {}
    failures: Counter = Counter()
    for r in _memory_rows(paths):
        key = (r["fact"], r["doc_id"])
        if r["label"] is not None:
            labels[key] = r["label"]
        else:
            failures[key] += 1
    for key, n in failures.items():
        if key not in labels and n >= MAX_REPLY_FAILURES:
            labels[key] = NO_LABEL
    return labels, failures


def check_judge(gc: GoldPassagesConfig, paths: DatasetPaths) -> None:
    """Refuse another judge or other prompts than the judgments in the
    memory (one dataset, one judge)."""
    used = {(r["model"], r["prompt"]) for r in _memory_rows(paths)}
    other = sorted(used - {(gc.model, judge_prompt_hash())})
    if other:
        raise SystemExit(f"{paths.judgments} holds judgments of other judges or prompts ((model, prompt hash): "
                         f"{other}; this run: {(gc.model, judge_prompt_hash())}). Restore gold_passages.model "
                         f"and the prompts, or remove {paths.judgments} to judge every pair again")


def require_judge_key(gc: GoldPassagesConfig, what: str) -> None:
    if gc.model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"{what} need the judge {gc.model}: set OPENROUTER_API_KEY "
                         "(.env) or gold_passages.model in the config")


def judge_pairs(gc: GoldPassagesConfig, paths: DatasetPaths, pairs: List[Tuple[dict, str]],
                judge: Optional[Judge] = None, texts=passage_texts) -> Dict[Tuple[str, str], Optional[str]]:
    """The label of every (fact, doc id) pair of *pairs*, by
    ``(fact_key, doc_id)``: the memory's, the judge's for the others (added
    to the memory); None for a pair not labelled yet (a failed call, a
    reply without a label before the last allowed one: asked again next
    run) or whose passage is not in the corpus.  *judge*: None, a
    ``Judge`` for ``gold_passages.model``; *texts*: doc ids -> passages."""
    memory, failures = _read_memory(paths)
    wanted: Dict[Tuple[str, str], dict] = {}
    for fact, doc_id in pairs:
        wanted.setdefault((fact_key(fact), doc_id), fact)
    out: Dict[Tuple[str, str], Optional[str]] = {k: memory[k] for k in wanted if k in memory}
    todo = [k for k in wanted if k not in memory]
    print(f"[judge] {len(wanted)} (fact, passage) pairs: {len(out)} judged before, "
          f"{len(todo)} to judge with {gc.model}")
    if not todo:
        return out
    require_judge_key(gc, f"{len(todo)} (fact, passage) pairs")
    passages = texts({doc_id for _, doc_id in todo})
    missing = sorted({doc_id for _, doc_id in todo if doc_id not in passages})
    if missing:
        log.warning("%d passages not in the corpus (e.g. %s): their pairs stay unjudged", len(missing), missing[:3])
    judge = judge or Judge(gc.model, gc.max_tokens)

    def one(key: Tuple[str, str]) -> Tuple[Tuple[str, str], Optional[str], Optional[str]]:
        try:
            return (key, *judge.label(wanted[key], passages[key[1]]))
        except Exception as e:  # noqa: BLE001 - asked again next run
            log.warning("judge of %s on %s failed: %s", wanted[key]["statement"][:60], key[1], e)
            return key, None, None

    asked = [k for k in todo if k[1] in passages]
    out.update({k: None for k in todo if k[1] not in passages})
    paths.judgments.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=max(1, gc.workers))
    prompt = judge_prompt_hash()
    try:
        with open(paths.judgments, "a", encoding="utf-8") as fh:
            futures = [pool.submit(one, k) for k in asked]
            for future in tqdm(as_completed(futures), total=len(futures), desc="[judge]", dynamic_ncols=True,
                               mininterval=10):
                key, label, reply = future.result()
                if reply is None:
                    out[key] = None
                    continue
                row = {"fact": key[0], "doc_id": key[1], "label": label, "model": gc.model, "prompt": prompt}
                if label is None:
                    row["reply"] = reply[:200]
                    label = NO_LABEL if failures[key] + 1 >= MAX_REPLY_FAILURES else None
                out[key] = label
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    failed = sum(out[k] is None for k in todo)
    if failed:
        print(f"[judge] {failed} pairs without a label (retried next run)")
    return out


def label_counts(labels: Iterable[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label or "none"] = counts.get(label or "none", 0) + 1
    return dict(sorted(counts.items()))
