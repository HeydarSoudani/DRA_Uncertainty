"""trqa_val, stage data_relevance: the gold annotation of the prepared TRQA
wiki1 + wiki2 validation questions (stage data_prepare: ``prepare.py``;
each record holds TRQA's intermediate info and qrels rows, ``meta``), in
the folder of the training datasets:

    queries/trqa_val.jsonl     one row per question (``annotation/rows.py``:
                               id, text, answer, source ``trqa_wiki1`` /
                               ``trqa_wiki2``, split, gold passages; the
                               intermediate info's members, values and
                               documents as ``members``, its property label
                               and aggregation)
    qrels/trqa_val.txt         TRQA's qrels rows, unchanged (the query ids
                               of the two collections do not collide)

The rows (``rows.member_row``) come from the intermediate info: the set
with the members and their values and each member's documents (``entity_values[].passage_ids``), each member named by
the page title (the first line) of its first document in ``wiki_partial``
(``trqa.member_titles``, read by id with ``CorpusLookup``).  No judge: the
qrels are TRQA's.  No corpus and no index: the dataset searches TRQA's
``wiki_partial`` (``layout._SHARED_CORPUS``).  No LLM call: the gold belief texts are made
by the belief model when it first needs them (stage teacher,
``belief_model_component/init/trqa_gold.py``, as TRQA's at inference).
The split: ``pipeline/relevance.finish``.
"""

from typing import List

from indexing_corpus_dataset.layout import default_corpus_path

from ....doc_lookup import CorpusLookup
from ...annotation import rows as annotation_rows
from ...config import BuildConfig, DatasetPaths
from ...pipeline.relevance import Built
from .trqa import docs_of, member_titles

FACTS = None
JUDGE = False


def build(cfg: BuildConfig, paths: DatasetPaths, processed: List[dict]) -> Built:
    titles = member_titles(CorpusLookup(default_corpus_path("trqa_val")), [r["meta"]["info"] for r in processed])
    rows = [annotation_rows.member_row(r["id"], r["source"], r["question"], r["answer"], r["meta"]["info"],
                                       docs_of(r["meta"]["qrels"]).get(r["id"], []), titles)
            for r in processed]
    return Built(rows=rows, qrels={r["id"]: list(r["meta"]["qrels"]) for r in processed}, kept=processed)
