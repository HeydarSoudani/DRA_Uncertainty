"""Tool environment — the single retriever the agent searches against.

Thin adapter over the finalized ``searcher_component.RetrievalSearchTool`` (used
READ-ONLY).  A ``MockToolEnv`` returns deterministic fake docs so rollouts run on
CPU with no index.

One tool instance is SHARED by all concurrent rollouts, so ``search`` must not
depend on per-rollout state.  Environments call it from a worker thread
(``asyncio.to_thread``); the retrieval adapter serialises calls with a lock.
"""

from __future__ import annotations

import abc
import hashlib
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class ToolEnv(abc.ABC):
    """A search tool: ``search(query) -> ranked docs``."""

    @abc.abstractmethod
    def search(self, query: str, *, original_query: Optional[str] = None,
               reasoning: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        ...


class MockToolEnv(ToolEnv):
    """Returns fake docs derived from the query, so the same query returns the
    same docs (novelty-style signals then see repeats)."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def search(self, query: str, *, original_query=None, reasoning=None, top_k=None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        h = int(hashlib.md5(query.encode()).hexdigest()[:6], 16) % 1000
        return [
            {"doc_id": f"doc-{h}-{j}", "title": f"Mock {h}-{j}",
             "text": f"[mock] result {j} for '{query[:40]}'"}
            for j in range(k)
        ]


class RetrievalToolEnv(ToolEnv):
    """Adapter over ``searcher_component.RetrievalSearchTool`` (built by the caller).

    ``top_k`` is left to the tool, as in inference (``BasicAgent.retrieve_documents``
    calls ``execute`` without it); environments slice the shown passages.
    """

    def __init__(self, search_tool: Any) -> None:
        self._tool = search_tool
        self._lock = threading.Lock()

    def search(self, query: str, *, original_query=None, reasoning=None, top_k=None) -> List[Dict[str, Any]]:
        with self._lock:
            return self._tool.execute(query, original_query=original_query, reasoning=reasoning,
                                      top_k=top_k)


def build_retrieval_tool(data_cfg, retrieval_cfg, seen_top_k: int) -> RetrievalToolEnv:
    """Build the retriever and search tool exactly as ``dra_inference.py`` does.

    Dataset paths left empty are filled by the inference defaults
    (``utils.cli_setup.resolve_dataset_defaults``).  Rerankers and
    ``ensure_novel_seen_docs`` are not supported: the tool is shared by
    concurrent rollouts, and the latter keeps per-run state.
    """
    from orchestration import setup_retriever_from_args
    from searcher_component.searcher import RetrievalSearchTool

    args = resolve_data_args(data_cfg, retrieval_cfg)
    retriever = setup_retriever_from_args(args)
    tool = RetrievalSearchTool(
        retriever=retriever,
        top_k=retrieval_cfg.top_k,
        retrieval_input=retrieval_cfg.retrieval_input,
        seen_top_k=seen_top_k,
    )
    return RetrievalToolEnv(tool)


def resolve_data_args(data_cfg, retrieval_cfg=None) -> SimpleNamespace:
    """The inference CLI namespace for *data_cfg*, with its defaults filled in."""
    from utils.cli_setup import resolve_dataset_defaults

    args = SimpleNamespace(
        dataset=data_cfg.dataset,
        dataset_year=data_cfg.dataset_year,
        subset=data_cfg.subset,
        query_key=data_cfg.query_key,
        data_path=data_cfg.data_path,
        qrels_data_path=data_cfg.qrels_data_path,
        min_relevance_score=data_cfg.min_relevance_score,
        retriever=getattr(retrieval_cfg, "retriever", None),
        index_dir=getattr(retrieval_cfg, "index_dir", None),
        corpus_path=getattr(retrieval_cfg, "corpus_path", None),
        top_k=getattr(retrieval_cfg, "top_k", 100),
    )
    resolve_dataset_defaults(args)
    return args
