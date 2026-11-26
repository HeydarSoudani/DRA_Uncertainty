"""Fusion methods for combining multiple retrieval results.

One class per method (``BaseFusion.fuse``); ``fuse_results`` runs the one the
search tool is configured with (interleaving, rrf, concatenation).
"""

from typing import Any, Dict, List

from .base import BaseFusion
from .interleaving import InterleavingFusion
from .rrf import ReciprocalRankFusion
from .concatenation import SimpleConcatenation


def fuse_results(
    ranked_lists: List[List[Dict[str, Any]]],
    fusion_method: str = "interleaving",
    rrf_k: int = 60,
    interleaving_window: int = 3,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists with the named method (one list: returned as is)."""
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0]
    if fusion_method == "interleaving":
        return InterleavingFusion(window=interleaving_window).fuse(ranked_lists)
    if fusion_method == "rrf":
        return ReciprocalRankFusion(k=rrf_k).fuse(ranked_lists)
    if fusion_method == "concatenation":
        return SimpleConcatenation().fuse(ranked_lists)
    raise ValueError(f"Unknown fusion method: {fusion_method!r}")


__all__ = [
    "fuse_results",
    "BaseFusion",
    "InterleavingFusion",
    "ReciprocalRankFusion",
    "SimpleConcatenation",
]
