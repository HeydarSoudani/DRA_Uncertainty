"""Base class for all fusion methods."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseFusion(ABC):
    """Abstract base class for fusion methods.

    All fusion methods must implement `fuse()`; documents are identified by
    ``utils.documents.doc_id`` ('doc_id', else 'id').
    """

    @abstractmethod
    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fuse multiple ranked lists into a single deduplicated ranked list.

        Args:
            ranked_lists: List of ranked result lists. Each result dict must
                          contain at least a 'doc_id' or 'id' key.

        Returns:
            Fused and deduplicated list of results.
        """
        ...
