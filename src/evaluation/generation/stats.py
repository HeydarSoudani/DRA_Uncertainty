"""Surface statistics of the generations, and their per-query files.

Reads ``result["generation"]`` (the report or short answer) of the unified
result format ``{query_id: {"generation": str, "citation_to_doc_id": {...}?}}``.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..common import mean_or_none
from ..judge import strip_references


def _count_words(text: str) -> int:
    return len(text.split()) if text else 0


def _count_citations(text: str) -> int:
    """Count citation markers of the form [N] in the text, not in its
    appended References block (whose ``[N] doc_id`` lines are not citations)."""
    return len(re.findall(r"\[\d+\]", strip_references(text)))


class GenerationEvaluator:
    """Length, word and citation statistics of the generations.

    Usage::

        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate(results)
    """

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """``num_queries``, ``avg_generation_length`` (chars),
        ``avg_generation_words`` and, when any generation cites,
        ``avg_citations`` (over the generations that cite or carry a
        citation map)."""
        char_lengths: List[int] = []
        word_counts: List[int] = []
        citation_counts: List[int] = []
        for result in results.values():
            generation = result.get("generation", "")
            char_lengths.append(len(generation))
            word_counts.append(_count_words(generation))
            n_cit = _count_citations(generation)
            if n_cit > 0 or result.get("citation_to_doc_id"):
                citation_counts.append(n_cit)

        metrics: Dict[str, Any] = {
            "num_queries": len(results),
            "avg_generation_length": mean_or_none(char_lengths) or 0.0,
            "avg_generation_words": mean_or_none(word_counts) or 0.0,
        }
        if citation_counts:
            metrics["avg_citations"] = mean_or_none(citation_counts)
        return metrics

    @staticmethod
    def save_item(query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Write the generation text to ``{output_dir}/{query_id}.md``."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"{query_id}.md", "w") as f:
            f.write(result.get("generation", ""))
