"""Generation statistics (the ``generation`` group of ``summary.json``).

* :class:`GenerationEvaluator`: length, words and citation markers of the
  generation, and the ``generation/{query_id}.md`` writer.
"""

from .stats import GenerationEvaluator

__all__ = ["GenerationEvaluator"]
