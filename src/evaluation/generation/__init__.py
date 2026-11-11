"""The per-query generation file.

* :class:`GenerationEvaluator`: the ``generation/{query_id}.md`` writer.
"""

from .stats import GenerationEvaluator

__all__ = ["GenerationEvaluator"]
