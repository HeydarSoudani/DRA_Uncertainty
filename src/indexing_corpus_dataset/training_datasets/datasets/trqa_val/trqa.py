"""TRQA's files as the gold annotation reads them: the relevant documents
of the qrels and the page title of each member's first document.  Shared
by the trqa_val build (``relevance.py``) and the TRQA gold beliefs of the
belief model (``belief_model_component/init/trqa_gold.py``)."""

from typing import Dict, Iterable, List

from ....doc_lookup import CorpusLookup


def docs_of(lines: Iterable[str]) -> Dict[str, List[str]]:
    """The relevant documents of each query in the qrels *lines*."""
    docs: Dict[str, List[str]] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 4 and float(parts[3]) > 0:
            docs.setdefault(parts[0], []).append(parts[2])
    return docs


def member_titles(lookup: CorpusLookup, infos: Iterable[dict]) -> Dict[str, str]:
    """The page title of each member's first document in the intermediate
    *infos* (doc id -> title), read from the corpus of *lookup*."""
    firsts = list(dict.fromkeys(v["passage_ids"][0] for info in infos
                                for v in info.get("entity_values") or [] if v.get("passage_ids")))
    return {doc_id: doc["title"].strip() for doc_id, doc in lookup.get(firsts).items() if doc["title"].strip()}
