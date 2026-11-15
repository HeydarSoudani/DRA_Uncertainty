"""Searcher component: local retrievers, fusion, and the unified search tool.

The retrievers are imported on first use (``__getattr__``): their module loads
torch, FAISS and sentence-transformers, which the belief and eval stages,
importing only ``normalize_retrieval_response``, do not need (they run on CPU).
"""

from .fusion import (
    interleaving_fusion,
    reciprocal_rank_fusion,
    simple_concatenation,
    fuse_results,
    fuse_retrieval_results,
    FUSION_METHODS,
)
from .normalize import normalize_retrieval_response

from .searcher import RetrievalSearchTool

_RETRIEVER_NAMES = ("BaseRetriever", "BM25Retriever", "SPLADERetriever",
                    "RerankRetriever", "DenseRetriever", "Encoder")


def __getattr__(name):
    if name in _RETRIEVER_NAMES:
        from . import retriever
        return getattr(retriever, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "SPLADERetriever",
    "RerankRetriever",
    "DenseRetriever",
    "Encoder",
    "interleaving_fusion",
    "reciprocal_rank_fusion",
    "simple_concatenation",
    "fuse_results",
    "fuse_retrieval_results",
    "FUSION_METHODS",
    "normalize_retrieval_response",
    "RetrievalSearchTool",
]
