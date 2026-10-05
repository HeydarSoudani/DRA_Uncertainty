"""Gold information units and the matchers that compare text against them.

Shared by the criteria evaluation (``evaluation.criteria``) and, later, the
generation evaluation:

* :mod:`.loaders`: per-query gold units: TRQA entities, report nugget
  questions, or a reference criteria list (``layout.DATASET_SPECS``
  ``criteria_gold``).
* :mod:`.entity_match`: entity names, by normalized string match with an LLM
  fallback for aliases.
* :mod:`.info_match`: units of information (a nugget question, a clue), by an
  LLM that judges whether a criterion asks for the unit's information.
"""

from .entity_match import LLMEntityMatcher, match_entities, normalize_name
from .info_match import LLMInfoMatcher
from .loaders import GoldUnit, load_gold_units, load_reference_criteria

__all__ = [
    "GoldUnit",
    "load_gold_units",
    "load_reference_criteria",
    "normalize_name",
    "match_entities",
    "LLMEntityMatcher",
    "LLMInfoMatcher",
]
