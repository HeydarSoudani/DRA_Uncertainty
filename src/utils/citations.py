"""The documents an agent result cites: the ranked cited-doc list
(``retrieval/cited``) and the References section built from it."""

import re
from typing import Any


def _citation_number(key: Any) -> Any:
    """A ``citation_to_doc_id`` key as a number: int keys live, str keys once
    the mapping is read back from JSON (``"10"`` must sort after ``"2"``)."""
    return int(key) if isinstance(key, str) and key.isdigit() else key


def with_references_section(result: dict) -> str:
    """The generation of an agent result with a References section appended
    (``\n\n## References`` then ``[N] doc_id`` lines); *result* is not changed.

    The generation is returned as it is when it is empty, already has a
    ``## References`` section, or no cited document is found.  An agent that
    numbers its own citations (``citation_to_doc_id``: WebWeaver, OSS, GLM)
    gets its own numbering, so every [N] in the text resolves to the document
    it names.  Otherwise the docs are those of :func:`ranked_cited_docs`
    (numbered in that order), and the ``[doc_id]`` citations in the text are
    renumbered to ``[N]``.
    """
    generation = result.get("generation", "")
    if not generation.strip() or "## References" in generation:
        return generation

    # See ``ranked_cited_docs``: an agent that records no citations
    # gets no reference list invented from the documents it merely saw.
    if result.get("records_citations") is False:
        return generation

    citation_to_doc_id = result.get("citation_to_doc_id")
    if citation_to_doc_id:
        numbered = sorted((int(n), did) for n, did in citation_to_doc_id.items() if did and str(n).isdigit())
        if not numbered:
            return generation
        return generation.rstrip() + "\n".join(["\n\n## References\n"] + [f"[{n}] {did}" for n, did in numbered])

    cited_ids = [d["doc_id"] for d in ranked_cited_docs(result)]
    if not cited_ids:
        return generation
    for ref_num, did in enumerate(cited_ids, 1):
        generation = generation.replace(f"[{did}]", f"[{ref_num}]")
    return generation.rstrip() + "\n".join(["\n\n## References\n"] + [f"[{n}] {did}" for n, did in enumerate(cited_ids, 1)])


def extract_citations_from_text(text: str) -> set[int]:
    """Extract citation IDs from text (e.g., [1], [22], [6]).

    Args:
        text: Text containing citation markers like [1], [22], etc.

    Returns:
        Set of citation IDs found in the text
    """
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return set(int(m) for m in matches)


def ranked_cited_docs(result: dict) -> list:
    """Build a deduplicated, ranked cited-doc list from an agent result.

    Extraction strategies, applied in priority order:

    1. **CPMReport / WebWeaver** — uses ``result["citation_to_doc_id"]``.
    2. **Reasoning agents** — walk trajectory steps for ``component_doc_ids``.
    3. **CPMReport trajectory fallback** — ``step["output"]["doc_ids"]``.
    4. **WebWeaver memory_bank fallback** — ``result["memory_bank"]`` keys.

    Returns:
        List of ``{"doc_id": str, "iter": int}`` dicts: by citation number
        (strategy 1), else in first-citation order.
        Empty list when no cited-doc information is available.
    """
    # An agent that records no citations gets an empty cited list rather than
    # strategy 2's, which would silently promote every *seen* doc to a citation
    # and report "what it looked at" under a column that means "what it cited".
    # Opt-in by the agent (``records_citations: False`` in its result); agents
    # that say nothing keep the previous behaviour.
    if result.get("records_citations") is False:
        return []

    citation_to_doc_id = result.get("citation_to_doc_id")
    if citation_to_doc_id:
        final_report = result.get("final_report") or result.get("generation") or ""
        if final_report:
            cited_ids_in_report = extract_citations_from_text(final_report)
        else:
            cited_ids_in_report = set()

        if not cited_ids_in_report:
            cited_ids_in_report = {_citation_number(k) for k in citation_to_doc_id}

        seen: set = set()
        doc_ids: list = []
        for cit_num, did in sorted(citation_to_doc_id.items(), key=lambda x: _citation_number(x[0])):
            cit_int = _citation_number(cit_num)
            if cit_int not in cited_ids_in_report:
                continue
            if did and did not in seen:
                seen.add(did)
                doc_ids.append(did)
        return [{"doc_id": d, "iter": 1} for d in doc_ids]

    trajectory = result.get("trajectory") or []
    use_component = any(step.get("component_doc_ids") for step in trajectory)

    seen_ids: set = set()
    ranked: list = []
    for step_idx, step in enumerate(trajectory, 1):
        if use_component:
            for did in step.get("component_doc_ids") or []:
                if did and did not in seen_ids:
                    seen_ids.add(did)
                    ranked.append({"doc_id": did, "iter": step_idx})
        else:
            output = step.get("output")
            if isinstance(output, dict):
                for did in output.get("doc_ids") or []:
                    if did and did not in seen_ids:
                        seen_ids.add(did)
                        ranked.append({"doc_id": did, "iter": step_idx})

    if ranked:
        return ranked

    memory_bank = result.get("memory_bank")
    if memory_bank and isinstance(memory_bank, dict):
        return [{"doc_id": did, "iter": 1} for did in memory_bank.keys() if did]

    return []
