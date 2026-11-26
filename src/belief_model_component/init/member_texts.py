"""The texts of TRQA's gold beliefs (``trqa_gold.py``): a writer LLM (the
run's ``belief_model``, native reasoning off) writes the property phrase
and the aggregation of a multi-member row (``training_datasets/annotation/rows.py``,
``member_row``, with its beliefs, ``trqa_gold.with_member_beliefs``) in the
words of the question; the member beliefs are "<member>: <phrase>", the
aggregate belief is the aggregation.

The prompts are the files of ``prompts/trqa_gold/`` (system, user and the
retry note; ``retry_note_unstated.txt`` is part of the prompt hash only, so
the gold banks written before the hash moved here stay valid).  The texts
are kept when the phrase and the aggregation name no member (its page
title or TRQA's label), no value and not the final answer, unless the
question names it; else they are asked once more with the names, then
fail for good.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from indexing_corpus_dataset.training_datasets.annotation.names import mentions
from indexing_corpus_dataset.training_datasets.annotation.rows import AGGREGATE, CLOSED
from indexing_corpus_dataset.training_datasets.llm import JsonWriter

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "trqa_gold"


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


SYSTEM = _read("system.txt")
USER = _read("user.txt")
RETRY_NOTE = "\n\n" + _read("retry_note.txt")
_HASH_ONLY_NOTE = "\n\n" + _read("retry_note_unstated.txt")

#: The hash of the prompts, in the key of the gold bank.
PROMPT_HASH = hashlib.sha1("\n".join((SYSTEM, USER, RETRY_NOTE, _HASH_ONLY_NOTE)).encode("utf-8")).hexdigest()[:12]


def _leaks(texts: List[str], names: List[str], others: List[str]) -> List[str]:
    """The *names* some text of *texts* mentions (``mentions``, outside the
    longer names of *others*)."""
    return [n for n in names if any(mentions(t, n, others) for t in texts)]


def member_layout(row: dict) -> bool:
    """A multi-member row has one belief per member, then the aggregate belief."""
    kinds = [c["kind"] for c in row["beliefs"]]
    return bool(row["members"]) and kinds == [CLOSED] * len(row["members"]) + [AGGREGATE]


def member_texts(row: dict, writer: JsonWriter) -> Tuple[Optional[List[str]], List[str]]:
    """``(texts, errors)``: one text per belief of the multi-member *row*
    ("<member>: <phrase>" per member, then the aggregation), or None and
    why when they failed for good.  A reply that is not a JSON object
    raises (``ReplyError``; the query is asked again by the next run)."""
    if not member_layout(row):
        return None, ["multi-member query: expected one belief per member, then the aggregate"]
    question = row["text"]
    members = [m["entity"] for m in row["members"]]
    # What the phrase and the aggregation must not name: the members (their
    # titles and TRQA's labels), their values and the final answer, unless
    # the question names them.
    values = [t for c in row["beliefs"] for t in c["answers"]]
    others = row["entity_names"] + values
    names = [n for n in dict.fromkeys(others) if not mentions(question, n, others)]
    errors: List[str] = []
    retry = ""
    for _ in range(2):
        data: Dict[str, Any] = writer.ask(SYSTEM, USER.format(question=question, property=row["property"] or "",
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
        return [f"{m}: {prop_text}" for m in members] + [agg_text], []
    return None, errors
