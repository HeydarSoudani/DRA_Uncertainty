"""Shared base class of the three document levels of a run.

The levels differ only in **which** documents they take from a result:

* :class:`~evaluation.retrieval.surfaced.SurfacedDocEvaluator`: every doc the
  retriever returned in each step (top-k surfaced).
* :class:`~evaluation.retrieval.seen.SeenDocEvaluator`: the subset actually
  passed to the agent/LLM (seen-top-k); the only level that is scored
  (Recall@N and Precision@N).
* :class:`~evaluation.retrieval.cited.CitedDocEvaluator`: the docs cited by the
  final report.

Subclasses implement a single hook, :meth:`doc_iterations`, returning one
ranked doc list per retrieval step; :meth:`save_item` writes the per-query
``retrieval/{level}/{qid}.trec``.  Both are classmethods, so the per-query
files are written without an evaluator.
"""

from pathlib import Path
from typing import Any, Dict, List

from utils.text_utils import doc_id as _doc_id


class BaseDocRetrievalEvaluator:
    """One document level of a run: its per-step doc lists and their TREC file."""

    @classmethod
    def doc_iterations(cls, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Return one ranked doc list per retrieval step of *result*.

        Each inner list is a list of doc dicts (or doc-id strings).  Return an
        empty list to skip the query.  Subclasses must implement this.
        """
        raise NotImplementedError

    @staticmethod
    def _score(doc: Dict[str, Any], rank: int) -> float:
        """Return the rank score written for *doc* at 1-based *rank*."""
        return 1.0 / rank

    @classmethod
    def doc_ids(cls, result: Dict[str, Any]) -> List[str]:
        """The distinct doc ids of *result* at this level, in first-seen order."""
        return list(dict.fromkeys(did for docs in cls.doc_iterations(result) for doc in docs
                                  if (did := _doc_id(doc))))

    @classmethod
    def save_item(cls, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query retrieval results as a TREC file.

        Docs are written per retrieval step; the 6th column records the step
        number (``iter_1``, ``iter_2``, ...).

        Standard TREC columns: ``qid Q0 doc_id rank score run_tag``
        """
        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        iterations = cls.doc_iterations(result)
        lines: List[str] = []
        for iter_idx, iteration_docs in enumerate(iterations, 1):
            for rank, doc in enumerate(iteration_docs, 1):
                did = _doc_id(doc)
                if not did:
                    continue
                score = cls._score(doc, rank)
                lines.append(f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_idx}")

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with open(trec_path, "w") as f:
            f.write(content)
