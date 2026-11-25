"""Searcher component: local retrievers, fusion, and the unified search tool.

The retrievers are imported on first use (``__getattr__``): their module loads
torch, FAISS and sentence-transformers, which the belief and eval stages,
importing only ``normalize_retrieval_response``, do not need (they run on CPU).
"""

from .fusion_methods import fuse_results
from .shared.normalize import normalize_retrieval_response

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
    "fuse_results",
    "normalize_retrieval_response",
    "RetrievalSearchTool",
]
