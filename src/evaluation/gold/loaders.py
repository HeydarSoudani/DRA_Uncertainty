"""Per-query gold entities (TRQA).

The entities of each query's set, from the intermediate information
(``queries_{split}_intermediate_info.jsonl``): the gold of the entity
reachability (``analysis/criteria_reachability.py``).
"""

from pathlib import Path
from typing import Dict, List

from indexing_corpus_dataset.dataset_loaders import load_query_intermediate_info


def load_gold_entities(data_path: Path | str, split: str) -> Dict[str, List[str]]:
    """``{query_id: [entity name, ...]}``, each name once, in file order;
    empty for a dataset without intermediate information."""
    entities = {}
    for qid, rec in load_query_intermediate_info(data_path, split).items():
        names = []
        for ev in rec.get("entity_values") or []:
            name = str(ev.get("entity", "")).strip()
            if name and name not in names:
                names.append(name)
        entities[qid] = names
    return entities
