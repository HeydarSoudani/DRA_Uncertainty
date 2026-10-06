"""Gold information units and the matchers that compare text against them.

Used by the criteria evaluation (``evaluation.criteria``):

* :mod:`.loaders`: per-query gold units: TRQA entities, report nugget
  questions, or a reference criteria list (``layout.DATASET_SPECS``
  ``criteria_gold``).
* :mod:`.entity_match`: entity names, by normalized string match with an LLM
  fallback for aliases.
* :mod:`.info_match`: the items of a reference criteria list, by an LLM that
  judges whether a criterion asks for the item's information.
* :mod:`.nugget_ask`: nugget question-answer pairs, judged YES/NO per
  criterion as Auto-ARGUE judges report sentences.
"""

from .entity_match import LLMEntityMatcher, match_entities, normalize_name
from .info_match import LLMInfoMatcher
from .nugget_ask import NuggetAskMatcher
from .loaders import GoldUnit, load_gold_units, load_reference_criteria

__all__ = [
    "GoldUnit",
    "load_gold_units",
    "load_reference_criteria",
    "normalize_name",
    "match_entities",
    "LLMEntityMatcher",
    "LLMInfoMatcher",
    "NuggetAskMatcher",
]
