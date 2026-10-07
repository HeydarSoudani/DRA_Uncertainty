"""Gold entities and the matcher that compares criteria against them.

Used by the entity reachability (``analysis/criteria_reachability.py``,
TRQA):

* :mod:`.loaders`: the entities of each query's set.
* :mod:`.entity_match`: entity names, by normalized string match with an LLM
  fallback for aliases.
"""

from .entity_match import ENTITY_MATCH_PROMPT_HASH, LLMEntityMatcher, match_entities
from .loaders import load_gold_entities

__all__ = [
    "ENTITY_MATCH_PROMPT_HASH",
    "load_gold_entities",
    "match_entities",
    "LLMEntityMatcher",
]
