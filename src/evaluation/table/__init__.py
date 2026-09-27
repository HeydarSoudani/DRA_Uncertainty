"""Table evaluation: persist and score the grid a grid-shaped agent builds.

The evaluator gates on the *shape* of a result (does it carry a grid?) rather
than on the agent's name, so nothing in this package is GridFill-specific.
"""

from .evaluator import TableEvaluator

__all__ = ["TableEvaluator"]
