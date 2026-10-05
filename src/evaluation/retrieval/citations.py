"""The documents a report cites, for :class:`evaluation.retrieval.cited.CitedDocEvaluator`."""

import re
from typing import Any, Dict, List, Set

from utils.text_utils import doc_id as _doc_id


def resolve_cited_docs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The docs cited by a result's report, in citation order, without repeats.

    Each doc is ``{"doc_id", "iter"}`` (the step that surfaced it, 1 when
    unknown).  Sources, in priority order:

    1. ``cited_docs_ranked_list``: the cited list built by the pipeline
       (``utils.text_utils._build_cited_docs_ranked_list``).
    2. ``memory_bank``: doc-id keys in insertion order (WebWeaver).
    3. ``[N]`` markers parsed from ``generation`` / ``final_report`` and
       mapped through ``citation_to_doc_id``, so any report agent works.
    """
    cited_ranked = result.get("cited_docs_ranked_list")
    if cited_ranked:
        docs = [{"doc_id": _doc_id(d), "iter": d.get("iter", 1) if isinstance(d, dict) else 1}
                for d in cited_ranked]
    elif result.get("memory_bank"):
        docs = [{"doc_id": did, "iter": 1} for did in result["memory_bank"]]
    else:
        report = result.get("generation") or result.get("final_report") or ""
        citation_to_doc_id = result.get("citation_to_doc_id") or {}
        docs = []
        if report and citation_to_doc_id:
            for cit in re.findall(r"\[(\d+)\]", report):  # in order; repeats dropped below
                did = citation_to_doc_id.get(int(cit)) or citation_to_doc_id.get(cit)
                docs.append({"doc_id": did, "iter": 1})

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for doc in docs:
        if doc["doc_id"] and doc["doc_id"] not in seen:
            seen.add(doc["doc_id"])
            out.append(doc)
    return out
