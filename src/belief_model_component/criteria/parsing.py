"""Parsing of the criteria extractor's reply: the items of a malformed
list, the kind of each criterion from the query shape, a multi-member
query's layout, a multi-clue query's layout check, the cap, and
references to later criteria."""

import json
import re
from typing import Any, Dict, List, Optional

from ..core.types import (
    AGGREGATE, ASPECT, CLOSED, CRITERION_REF_RE, HOP, MULTI_ASPECT, MULTI_CLUE, MULTI_HOP, MULTI_MEMBER, REST, TARGET,
)

# One complete string item of the criteria list, for output that is not
# valid JSON (cut at the token limit, or a stray character between items).
_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*(?=[,\]])')
# One complete object item (``{"text": ...}``), likewise.
_OBJECT_RE = re.compile(r'\{[^{}]*\}')


def salvage_items(body: str) -> List[Any]:
    """The complete items of a malformed or truncated criteria list: its
    objects when it has any, else its strings."""
    objects = []
    for chunk in _OBJECT_RE.findall(body):
        try:
            objects.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return objects or [json.loads(f'"{t}"') for t in _ITEM_RE.findall(body)]


# Subject of the rest criterion of a multi-member query ("any other member: <property>").
REST_SUBJECT = "any other member"


def _is_rest(text: str) -> bool:
    subject = text.rsplit(":", 1)[0] if ":" in text else text
    return " ".join(re.findall(r"[a-z]+", subject.lower())) in (REST_SUBJECT, REST_SUBJECT + "s")


# The target criterion of a multi-clue query: "the <type> that meets c1 to c<k>".
TARGET_RE = re.compile(r"\bmeets\s+c1\s+to\s+c(\d+)\b", re.IGNORECASE)


def target_index(texts: List[str]) -> Optional[int]:
    """The position of the target criterion among multi-clue criterion
    *texts*: the first one after at least one clue that reads "meets c1 to
    c<k>"; None without one."""
    return next((k for k, t in enumerate(texts) if k >= 1 and TARGET_RE.search(t)), None)


def clue_layout(texts: List[str]) -> List[str]:
    """What the multi-clue criterion *texts* get wrong against the layout
    the prompt asks for (clues, the target "meets c1 to c<k>" with k the
    number of clues, at most the asked answer after it); [] when none."""
    at = target_index(texts)
    if at is None:
        return ["no target criterion (\"the <type> that meets c1 to c<k>\")"]
    out = []
    k = int(TARGET_RE.search(texts[at]).group(1))
    if k != at:
        out.append(f"the target c{at + 1} meets c1 to c{k}, but {at} clues come before it")
    if len(texts) - at - 1 > 1:
        out.append(f"{len(texts) - at - 1} criteria after the target c{at + 1} (at most the asked answer)")
    return out


def cap_chain(items: List[str], cap: int, info: Dict[str, Any]) -> List[str]:
    """Criterion texts *items* within *cap*: the last *cap* kept (the answer
    and the criteria closest to it, as the prompt asks), the references
    renumbered; a reference to a left-out criterion is replaced by that
    criterion's text in parentheses (its own references expanded likewise).
    A cut is noted in ``info["warnings"]`` and sets ``info["over_cap"]``."""
    if len(items) <= cap:
        return items
    drop = len(items) - cap
    expanded: List[str] = []

    def _ref(m: "re.Match") -> str:
        n = int(m.group(1))
        if 1 <= n <= drop and n <= len(expanded):
            return f"({expanded[n - 1]})"
        if drop < n <= len(items):
            return f"c{n - drop}"
        return m.group(0)

    for t in items[:drop]:
        expanded.append(CRITERION_REF_RE.sub(_ref, t))
    kept = [CRITERION_REF_RE.sub(_ref, t) for t in items[drop:]]
    info.setdefault("warnings", []).append(f"cap: kept the last {cap} of {len(items)} criteria, dropped "
                                           f"{items[:drop]}")
    info["over_cap"] = True
    return kept


def criterion_kinds(shape: str, texts: List[str]) -> List[str]:
    """Kind of each criterion of a *shape* query: a multi-clue query's
    clues and asked answer are closed, its target ("the <type> that meets
    c1 to c<k>", ``target_index``) a target; a multi-hop query's hops
    before its answer are hops (several answers), its answer closed
    (``shape_kinds``); a multi-member query's list (``member_layout``) is its
    members (closed), the rest criterion and the aggregate; every report
    criterion is an aspect."""
    if shape == MULTI_ASPECT:
        return [ASPECT] * len(texts)
    if shape == MULTI_CLUE:
        kinds = [CLOSED] * len(texts)
        at = target_index(texts)
        if at is not None:
            kinds[at] = TARGET
        return kinds
    if shape == MULTI_MEMBER and len(texts) >= 2:
        return [CLOSED] * (len(texts) - 2) + [REST, AGGREGATE]
    return shape_kinds(shape, [CLOSED] * len(texts))


def shape_kinds(shape: str, kinds: List[str]) -> List[str]:
    """The *kinds* of a *shape* query's criteria as the belief reads them: in
    a multi-hop query every closed criterion but the last (the answer) is a
    hop, which may have several answers; any other shape is unchanged.
    Gold rows record every hop as closed, so their kinds are mapped too."""
    if shape != MULTI_HOP:
        return list(kinds)
    return [HOP if kind == CLOSED and k < len(kinds) - 1 else kind for k, kind in enumerate(kinds)]


def member_layout(items: List[str], cap: Optional[int], info: Dict[str, Any]) -> List[str]:
    """A multi-member query's criteria as the prompt asks: the members, one rest
    criterion, then the aggregate, within *cap* (at least 3; None:
    uncapped) by leaving out members.  The aggregate is the first item after the rest criterion or,
    without one, the last item when it is not "<member>: <property>".  A
    missing rest criterion is added with the members' property, a missing
    aggregate with a generic text.  Every change is noted in
    ``info["warnings"]``."""
    if not items:
        return items
    warnings = info.setdefault("warnings", [])
    rest_at = next((k for k, t in enumerate(items) if _is_rest(t)), None)
    if rest_at is not None:
        members, after = items[:rest_at], items[rest_at + 1:]
        rest_item = items[rest_at]
        extra_rest = [t for t in after if _is_rest(t)]
        if extra_rest:
            warnings.append(f"rest: kept the first of {len(extra_rest) + 1} rest criteria, dropped {extra_rest}")
        after = [t for t in after if not _is_rest(t)]
        aggregate = after[0] if after else None
        if len(after) > 1:
            warnings.append(f"aggregate: kept the first item after the rest criterion, dropped {after[1:]}")
    else:
        aggregate = items[-1] if ":" not in items[-1] else None
        members = items[:-1] if aggregate is not None else list(items)
        rest_item = None
    prop = next((t.rsplit(":", 1)[1].strip() for t in members if ":" in t), "")
    if rest_item is None:
        rest_item = f"{REST_SUBJECT}: {prop or 'the property the query asks about'}"
        warnings.append("rest: added the missing rest criterion")
    if aggregate is None:
        aggregate = f"the answer the query computes from every member's {prop or 'value'}"
        warnings.append("aggregate: added the missing aggregate criterion")
    keep = len(members) if cap is None else cap - 2
    if len(members) > keep:
        warnings.append(f"cap: kept {keep} of {len(members)} members, dropped {members[keep:]}")
    if not warnings:
        del info["warnings"]
    return members[:keep] + [rest_item, aggregate]


def forward_refs(texts: List[str]) -> List[str]:
    """The references of criterion *texts* (ids "c1", "c2", ... in order)
    to the criterion itself or to one after it, as "c<k> refers to c<m>":
    a criterion refers only to criteria before it."""
    return [f"c{k} refers to c{m}" for k, t in enumerate(texts, 1)
            for m in dict.fromkeys(int(x) for x in CRITERION_REF_RE.findall(t)) if k <= m <= len(texts)]
