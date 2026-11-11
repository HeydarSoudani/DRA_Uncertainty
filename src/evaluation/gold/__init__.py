"""Gold of the evaluation beyond the qrels and answers.

* :mod:`.criteria`: the gold answers and documents per criterion of a
  query (TRQA's gold bank, the training pools), for the belief update
  evaluation.
* :mod:`.loaders` and :mod:`.entity_match`: the entities of each TRQA set
  and their matcher, for the entity reachability of the criteria
  (``analysis/criteria_reachability.py``).
"""

from .criteria import GOLD_DATASETS, load_gold_rows, match_row
from .entity_match import ENTITY_MATCH_PROMPT_HASH, LLMEntityMatcher, match_entities
from .loaders import load_gold_entities

__all__ = [
    "ENTITY_MATCH_PROMPT_HASH",
    "GOLD_DATASETS",
    "load_gold_entities",
    "load_gold_rows",
    "match_entities",
    "match_row",
    "LLMEntityMatcher",
]
