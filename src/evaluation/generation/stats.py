"""The per-query generation file ``generation/{query_id}.md``.

The generation (the report or short answer, with its References block) is
saved as is; it is scored by the answer evaluators (``evaluation.answer``),
not here.
"""

from pathlib import Path
from typing import Any, Dict


class GenerationEvaluator:
    """Writer of the per-query generation file."""

    @staticmethod
    def save_item(query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Write the generation text to ``{output_dir}/{query_id}.md``."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"{query_id}.md", "w") as f:
            f.write(result.get("generation", ""))
