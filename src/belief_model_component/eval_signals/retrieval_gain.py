"""Supervised retrieval gain of each step against the qrels (evaluation
only, never shown to the agent)."""

from typing import Any, Dict, List, Optional, Set

from utils.text_utils import doc_id as _doc_id


class RetrievalGainSignal:
    """Supervised retrieval gain of each step against the qrels.

    Binary (``qrels``, thresholded by the caller: relevant = grade > 0),
    over the step's unique doc ids:

    - ``new_item_precision``: newly seen relevant docs / the step's docs
      (0, 0.2, ..., 1 for 5 docs).
    - ``num_new_relevant``, ``num_repeated_relevant``, ``num_irrelevant``.
    - ``recall_so_far``: ``num_relevant_seen / num_relevant``, the relevant
      docs seen up to and including this step.

    All fields are null when the query has no relevant doc in the qrels or
    the step has no docs.  Call :meth:`reset` at the start of each query.
    """

    NULL = {
        "new_item_precision": None,
        "num_new_relevant": None,
        "num_repeated_relevant": None,
        "num_irrelevant": None,
        "recall_so_far": None,
        "num_relevant_seen": None,
        "num_relevant": None,
    }

    def __init__(self, qrels: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._qrels = qrels or {}
        self._relevant_ids: Set[str] = set()
        self._relevant_seen: Set[str] = set()

    @property
    def num_relevant(self) -> Optional[int]:
        """Relevant docs of the current query; None without qrels for it."""
        return len(self._relevant_ids) or None

    def reset(self, query_id: Optional[str] = None) -> None:
        """Reset per-query state and load the relevant ids of ``query_id``."""
        self._relevant_seen.clear()
        judged = self._qrels.get(query_id, {}) if query_id else {}
        self._relevant_ids = {d for d, r in judged.items() if float(r) > 0}

    def score(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the step's gain statistics (see the class docstring)."""
        ids = {_doc_id(doc) for doc in docs}
        ids.discard("")
        result = dict(self.NULL)
        if not ids or not self._relevant_ids:
            return result
        relevant = ids & self._relevant_ids
        new = relevant - self._relevant_seen
        self._relevant_seen.update(new)
        result.update({
            "new_item_precision": round(len(new) / len(ids), 4),
            "num_new_relevant": len(new),
            "num_repeated_relevant": len(relevant) - len(new),
            "num_irrelevant": len(ids) - len(relevant),
            "recall_so_far": round(len(self._relevant_seen) / self.num_relevant, 4),
            "num_relevant_seen": len(self._relevant_seen),
            "num_relevant": self.num_relevant,
        })
        return result
