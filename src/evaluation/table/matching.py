"""Matching a predicted grid's rows and columns onto a gold table.

Two problems, both approximate, both stdlib-only (no embeddings, no judge):

**Rows.**  An agent writes "New York (state)" where the gold says "New York".
:func:`match_rows` normalises both sides and pairs them one-to-one.  Two tiers
are offered because the number should not be hostage to the matcher:
``strict`` accepts only an exact match after normalisation, ``lenient`` adds a
fuzzy tier.  Reporting both shows how much of the score rests on fuzz.

**Columns.**  The gold names its property in Wikidata's vocabulary
("coordinates of geographic center"); the agent names it in its own
("Geographic center (lat/lon)").  :func:`match_column` scores the gold label and
description against each column's name and spec by token overlap and takes the
best above a threshold.

Both thresholds are module constants, tuned against the round-trip check in
``matching_test.py`` rather than guessed.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A fuzzy row pair must reach this character-level similarity.  High on purpose:
# the tier exists for "New York (state)" vs "New York", not for guessing between
# neighbouring entities, and entity sets routinely contain near-identical names
# ("Virginia" / "West Virginia") that a loose threshold would happily confuse.
# The affix tier below now carries the cases this used to be stretched to reach,
# so it can sit a little lower without taking on their risk.
ROW_FUZZY_THRESHOLD = 0.85

# A column matches when the gold property's tokens overlap the column this well
# (token F1).  Lower, because the two sides are written in different vocabularies
# and only need to be recognisably the same property.
# Set from the observed score distribution over 45 real (gold property, agent
# column) pairs rather than guessed: every correct pair scored >= 0.50 and the
# two false ones -- "Human Development Index" against a column establishing the
# *Democracy* Index, "start of work period" against tenure start -- scored 0.40
# and 0.33.  Both share exactly one generic token with the gold, which is what a
# near-miss property looks like.
COLUMN_TOKEN_F1_THRESHOLD = 0.45

# Property words that name the same thing in Wikidata's vocabulary and an
# agent's.  Folded to one token on BOTH sides before a column is scored, so
# "date of birth" and "Year of birth" are the same property -- TRQA's single
# largest property family, and one an unfolded token overlap misses outright.
#
# Folding alone would also make "date of birth" and "year of death" share a
# token, so a column match additionally requires overlap on a token that is not
# one of these: agreeing only on the folded word is not agreeing on a property.
# Deliberately only the temporal family.  It is the one where the two
# vocabularies reliably disagree -- Wikidata pins "date of birth" and an agent
# writes "Year of birth", 11 of the 45 columns observed -- and where folding is
# safe because the aspect word beside it ("birth", "death", "inception") still
# has to agree.  Wider maps were tried and removed: folding number/total/
# population together broke "population" vs "total population", which the plain
# overlap already gets right.  A synonym is a repair for a systematic mismatch,
# not a general-purpose thesaurus.
_PROPERTY_SYNONYMS = {
    "year": "date", "years": "date", "dates": "date",
}

# Tokens carrying no discriminative weight in a property label.
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "and", "or", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "its", "this",
    "that", "which", "use", "used", "given", "per",
})

# Leading articles to drop from an entity name before comparing.
_LEADING_ARTICLES = ("the ", "a ", "an ")

_PARENTHETICAL_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def normalize_entity(name: Any) -> str:
    """Fold an entity name to its comparable core.

    Casefolds, strips diacritics, drops parenthetical qualifiers and a leading
    article, and collapses everything non-alphanumeric to single spaces, so
    "New York (state)" and "the New  York state" both become "new york state".

    Separators are decided by ``str.isalnum``, not by an ASCII class, so a name
    in a script that NFKD cannot reduce to ASCII -- Cyrillic, Greek, CJK --
    keeps its characters instead of folding away to the empty string and
    matching nothing.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    text = _PARENTHETICAL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    text = " ".join(text.split())
    for article in _LEADING_ARTICLES:
        if text.startswith(article):
            text = text[len(article):]
            break
    return " ".join(text.split())


def _tokens(text: Any) -> List[str]:
    """Content tokens of a phrase, normalised and stop-worded."""
    return [t for t in normalize_entity(text).split() if t not in _STOPWORDS]


def token_f1(left: Any, right: Any) -> float:
    """Token-level F1 between two phrases (0.0 when either has no content)."""
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(b), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def _content_tokens(text: Any) -> frozenset:
    """The normalised, stop-worded token *set* of a name."""
    return frozenset(_tokens(text))


def _affix_pairs(
    gold_tokens: Sequence[frozenset],
    pred_tokens: Sequence[frozenset],
    matched: Dict[int, int],
    used_pred: set,
) -> List[Tuple[float, int, int]]:
    """Candidate pairs where one name's tokens are wholly inside the other's.

    The gold is Wikidata's label and the agent writes the common name, so the
    two differ by a type affix far more often than by a spelling: "Barnet" for
    "London Borough of Barnet", "1990-91 season" for "1990-91 NBA season",
    "New Railway Station" for "New Railway Station metro station".  Those are
    the same entity, not a fuzzy guess -- a character-similarity tier scores
    them 0.4 and would need a threshold loose enough to confuse real neighbours.

    A pair is offered only when it is **unambiguous**: a name contained in two
    different names on the other side ("Carolina" inside both "North Carolina"
    and "South Carolina") names neither, so it is dropped rather than guessed.
    """
    contains: Dict[int, List[int]] = {}
    for i, g in enumerate(gold_tokens):
        if i in matched or not g:
            continue
        for j, p in enumerate(pred_tokens):
            if j in used_pred or not p:
                continue
            if g <= p or p <= g:
                contains.setdefault(i, []).append(j)

    # Ambiguity is symmetric: drop a gold that swallows several rows, and a row
    # that several golds would take.
    claims: Dict[int, int] = {}
    for i, js in contains.items():
        for j in js:
            claims[j] = claims.get(j, 0) + 1

    pairs: List[Tuple[float, int, int]] = []
    for i, js in contains.items():
        if len(js) != 1:
            continue
        j = js[0]
        if claims.get(j, 0) != 1:
            continue
        g, p = gold_tokens[i], pred_tokens[j]
        pairs.append((len(g & p) / max(len(g), len(p)), i, j))
    return pairs


def match_rows(
    gold_entities: Sequence[str],
    pred_rows: Sequence[Dict[str, Any]],
    *,
    lenient: bool = False,
) -> Dict[int, int]:
    """Pair gold entities with predicted rows, one-to-one.

    Three tiers, tightest first, so a looser pair can never steal an entity a
    tighter one matches: exact after normalisation, then affix-insensitive
    containment (both count as *strict*), then -- under ``lenient`` only -- a
    character-similarity tier.  Every tier pairs one-to-one -- without that
    constraint a single predicted row could satisfy several gold entities at
    once and inflate recall.

    Args:
        gold_entities: Gold entity names, in order.
        pred_rows:     The grid's rows (dicts carrying ``entity``).
        lenient:       Add the fuzzy tier after the two exact ones.

    Returns:
        ``{gold_index: pred_row_index}`` for matched pairs only.
    """
    gold_norm = [normalize_entity(e) for e in gold_entities]
    pred_norm = [normalize_entity(r.get("entity")) for r in pred_rows]

    matched: Dict[int, int] = {}
    used_pred: set = set()

    # ── tier 1: exact after normalisation ────────────────────────────────────
    by_norm: Dict[str, List[int]] = {}
    for j, norm in enumerate(pred_norm):
        if norm:
            by_norm.setdefault(norm, []).append(j)
    for i, norm in enumerate(gold_norm):
        for j in by_norm.get(norm, []):
            if j not in used_pred:
                matched[i] = j
                used_pred.add(j)
                break

    # ── tier 2: affix-insensitive, still exact ───────────────────────────────
    # Part of normalisation rather than of fuzz: the two sides name the same
    # entity and differ by a type affix Wikidata carries and an agent does not.
    # Unambiguous by construction (see _affix_pairs), so it runs in both tiers
    # -- holding it back for `lenient` would make the strict number a measure of
    # whether the gold happens to spell its labels the way the agent does.
    gold_tok = [_content_tokens(e) for e in gold_entities]
    pred_tok = [_content_tokens(r.get("entity")) for r in pred_rows]
    for _score, i, j in sorted(_affix_pairs(gold_tok, pred_tok, matched, used_pred),
                               key=lambda c: (-c[0], c[1], c[2])):
        if i in matched or j in used_pred:
            continue
        matched[i] = j
        used_pred.add(j)

    if not lenient:
        return matched

    # ── tier 3: fuzzy, greedy by descending similarity ───────────────────────
    candidates: List[Tuple[float, int, int]] = []
    for i, g in enumerate(gold_norm):
        if i in matched or not g:
            continue
        for j, p in enumerate(pred_norm):
            if j in used_pred or not p:
                continue
            score = SequenceMatcher(None, g, p).ratio()
            if score >= ROW_FUZZY_THRESHOLD:
                candidates.append((score, i, j))

    for _score, i, j in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if i in matched or j in used_pred:
            continue
        matched[i] = j
        used_pred.add(j)

    return matched


def _property_tokens(text: Any) -> frozenset:
    """Content tokens of a property phrase, with synonyms folded together."""
    return frozenset(_PROPERTY_SYNONYMS.get(t, t) for t in _tokens(text))


def _property_f1(left: Any, right: Any) -> float:
    """:func:`token_f1` over synonym-folded property tokens."""
    a, b = _property_tokens(left), _property_tokens(right)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(b), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def match_column(
    property_label: str,
    property_description: str,
    columns: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], float]:
    """Find the column that establishes the gold property.

    Scored against the column's **name** first, with name+spec and the gold
    description as weaker fallbacks: a long spec otherwise drowns a two-word
    label in incidental token overlap, and did so badly enough to reject columns
    that named the property outright.  Property synonyms are folded first
    (year/date, place/location), and a match must then agree on a token the
    folding did not manufacture.

    Returns:
        ``(col_id, score)``, or ``(None, best_score)`` when nothing clears
        :data:`COLUMN_TOKEN_F1_THRESHOLD`.
    """
    best_id: Optional[str] = None
    best_score = 0.0
    label_tokens = _property_tokens(property_label)

    for col in columns or []:
        name = str(col.get("name", "") or "")
        both = f"{name} {col.get('spec', '') or ''}"

        # The label against the *name alone* is the signal.  Scoring it against
        # name+spec instead -- as this used to -- lets a 20-token spec halve the
        # F1 of a name that matched the property exactly: "Religion or Worldview"
        # against "religion or worldview" is a perfect 1.0 on the name and 0.36
        # once the spec is concatenated on.  The spec is kept as a weaker echo,
        # for the column that establishes the property without naming it.
        score = _property_f1(property_label, name)
        score = max(score, 0.6 * _property_f1(property_label, both))
        if property_description:
            score = max(score, 0.4 * _property_f1(property_description, both))

        # Folding year->date (etc.) is what lets "Year of birth" reach "date of
        # birth"; it also puts a shared token between "date of birth" and "year
        # of death", which are different properties.  So a match must agree on
        # something the folding did not manufacture.
        if not (label_tokens & _property_tokens(both)) - set(_PROPERTY_SYNONYMS.values()):
            continue

        if score > best_score:
            best_score, best_id = score, col.get("col_id")

    if best_score < COLUMN_TOKEN_F1_THRESHOLD:
        return None, best_score
    return best_id, best_score
