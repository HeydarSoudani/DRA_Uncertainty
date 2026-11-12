"""Parsing of the criteria extractor's reply: the items of a malformed
list, the kind of each criterion from the query shape, a multi-member
query's layout, a multi-clue query's layout check, the cap, and
references to later criteria."""

import json
import re
from typing import Any, Dict, List, Optional

from ..core.types import AGGREGATE, ASPECT, CLOSED, CRITERION_REF_RE, MULTI_ASPECT, MULTI_CLUE, MULTI_MEMBER, TARGET_RE

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


# The subject of a criterion for the members a multi-member list did not
# name ("any other member: <property>"): no longer a criterion, left out of
# a list that has one (``member_layout``).
_OTHER_MEMBERS = "any other member"


def _is_other_members(text: str) -> bool:
    subject = text.rsplit(":", 1)[0] if ":" in text else text
    return " ".join(re.findall(r"[a-z]+", subject.lower())) in (_OTHER_MEMBERS, _OTHER_MEMBERS + "s")


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
    """Kind of each criterion of a *shape* query: every report criterion is
    an aspect; a multi-clue query's target ("the <type> that meets c1 to
    c<k>", ``target_index``) and a multi-member query's last criterion (the
    computation, ``member_layout``) are aggregates; every other criterion
    is closed (a clue, the asked answer, a hop, a member)."""
    if shape == MULTI_ASPECT:
        return [ASPECT] * len(texts)
    kinds = [CLOSED] * len(texts)
    if shape == MULTI_CLUE:
        at = target_index(texts)
        if at is not None:
            kinds[at] = AGGREGATE
    elif shape == MULTI_MEMBER and len(texts) >= 2:
        kinds[-1] = AGGREGATE
    return kinds


def member_layout(items: List[str], cap: Optional[int], info: Dict[str, Any]) -> List[str]:
    """A multi-member query's criteria as the prompt asks: the members, then
    the aggregate, within *cap* (at least 2; None: uncapped) by leaving out
    members.  An item for the members the list did not name ("any other
    member: <property>") is left out; the aggregate is the last item when
    it is not "<member>: <property>", else one with a generic text is
    added.  Every change is noted in ``info["warnings"]``."""
    if not items:
        return items
    warnings = info.setdefault("warnings", [])
    others = [t for t in items if _is_other_members(t)]
    if others:
        warnings.append(f"other members: left out {others}")
        items = [t for t in items if not _is_other_members(t)]
    aggregate = items[-1] if items and ":" not in items[-1] else None
    members = items[:-1] if aggregate is not None else list(items)
    prop = next((t.rsplit(":", 1)[1].strip() for t in members if ":" in t), "")
    if aggregate is None:
        aggregate = f"the answer the query computes from every member's {prop or 'value'}"
        warnings.append("aggregate: added the missing aggregate criterion")
    keep = len(members) if cap is None else cap - 1
    if len(members) > keep:
        warnings.append(f"cap: kept {keep} of {len(members)} members, dropped {members[keep:]}")
    if not warnings:
        del info["warnings"]
    return members[:keep] + [aggregate]


def forward_refs(texts: List[str]) -> List[str]:
    """The references of criterion *texts* (ids "c1", "c2", ... in order)
    to the criterion itself or to one after it, as "c<k> refers to c<m>":
    a criterion refers only to criteria before it."""
    return [f"c{k} refers to c{m}" for k, t in enumerate(texts, 1)
            for m in dict.fromkeys(int(x) for x in CRITERION_REF_RE.findall(t)) if k <= m <= len(texts)]
