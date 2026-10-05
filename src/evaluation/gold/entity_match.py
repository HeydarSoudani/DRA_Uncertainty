"""Match gold entity names against texts that name entities (criteria).

Two passes:

1. :func:`match_entities`: string match on normalized names (accents,
   case, punctuation and a parenthetical qualifier such as "(2017 film)"
   removed) against the text's subject (the part before the last ":" of
   "member: property"): first equal names, then entities whose tokens all
   occur in the subject (a subject with a middle name).  A subject shorter
   than the entity name ("United Kingdom" for "United Kingdom of Great
   Britain and Ireland", "1991 NBA season" for "1991-92 NBA season") is left
   to the LLM pass.
2. :class:`LLMEntityMatcher`: one LLM call per query for the entities left
   unmatched, for aliases, other spellings and transliterations.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import parse_json_object

logger = logging.getLogger(__name__)


def normalize_name(text: str) -> List[str]:
    """Tokens of *text* without accents, case, punctuation, a trailing
    parenthetical qualifier or a leading "the"."""
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    return tokens


def _subject(text: str) -> str:
    """The member part of a "member: property" criterion, else the text."""
    return text.rsplit(":", 1)[0] if ":" in text else text


def match_entities(entities: List[str], texts: List[str]) -> Dict[int, int]:
    """String pass: ``{entity index: text index}``; each entity and each text
    is matched at most once, equal names first."""
    names = [normalize_name(e) for e in entities]
    subjects = [normalize_name(_subject(t)) for t in texts]
    used: set = set()
    matches: Dict[int, int] = {}
    for test in (lambda e, s: e == s, lambda e, s: set(e) <= set(s)):
        for i, name in enumerate(names):
            if i in matches or not name:
                continue
            for j, subject in enumerate(subjects):
                if j not in used and subject and test(name, subject):
                    matches[i] = j
                    used.add(j)
                    break
    return matches


ENTITY_MATCH_SYSTEM = """\
You match entity names. You will receive a numbered list of entities and a numbered list of texts, each of which may name an entity. For each text, decide which listed entity it names, if any: the same real-world entity under another name, spelling, transliteration, or with or without a qualifier (for example "Iosif Stalin" names "Joseph Stalin", and "Mary Kom" names "Mary Kom (film)").

Rules:
1. Match only when the text clearly names that entity; a related entity (a sequel, a namesake, a member of the same family) is not a match.
2. Each entity is matched by at most one text and each text matches at most one entity.

Respond with ONLY a raw JSON object, no markdown, no code blocks:
{"matches": [{"text": text number, "entity": entity number}]}
An empty list means no text names a listed entity."""

ENTITY_MATCH_USER_TEMPLATE = """\
Entities:
{entities}

Texts:
{texts}"""


class LLMEntityMatcher:
    """LLM pass for the entities the string pass left unmatched.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_tokens / temperature: LLM call settings.
    """

    def __init__(self, llm_client: Any, max_tokens: int = 2048, temperature: float = 0.0) -> None:
        self._llm = llm_client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def match(self, entities: List[str], texts: List[str]) -> Tuple[Dict[int, int], Optional[str]]:
        """``({entity index: text index}, error)``; error is None on success."""
        if not entities or not texts:
            return {}, None
        messages = [
            {"role": "system", "content": ENTITY_MATCH_SYSTEM},
            {"role": "user", "content": ENTITY_MATCH_USER_TEMPLATE.format(
                entities="\n".join(f"{i + 1}. {e}" for i, e in enumerate(entities)),
                texts="\n".join(f"{j + 1}. {t}" for j, t in enumerate(texts)),
            )},
        ]
        try:
            raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        except Exception as e:
            logger.warning("LLMEntityMatcher: LLM call failed", exc_info=True)
            return {}, f"llm_call: {e}"
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("matches"), list):
            return {}, "parse: no JSON object with a 'matches' list"
        matches: Dict[int, int] = {}
        used: set = set()
        for m in data["matches"]:
            try:
                i, j = int(m["entity"]) - 1, int(m["text"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(entities) and 0 <= j < len(texts) and i not in matches and j not in used:
                matches[i] = j
                used.add(j)
        return matches, None
