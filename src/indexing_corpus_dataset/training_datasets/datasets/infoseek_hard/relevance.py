"""infoseek_hard, stage data_relevance: the gold chunks, qrels and gold
annotation of the prepared InfoSeek-Hard questions (stage data_prepare:
``prepare.py``; each record holds the chunk ids of its gold pages in
corpus_wiki, InfoSeek's wiki-25-512, ``features.page_chunks``).

    gold       InfoSeek writes every claim from the first 1024 tokens of
               the page of the node it expands: chunk 1 or chunk 2 of that
               page, no rule says which.  One fact per claim
               (``facts.claim_facts``) and one judge call per claim, on
               chunk 2 (``gold_passages/judge.judge_pairs``): the claim's
               gold chunk is chunk 2 when the judge finds the claim there
               (YES-SAME), else chunk 1 (a page of one chunk: chunk 1, no
               call).  A page's gold chunks are those of its claims, chunk 1
               for a page without claims.  A claim without a label leaves
               its question for the next run
    rows       the qrels (binary) are the gold chunks of every gold page;
               a hop's documents (``annotation/rows.row``) those of its own
               page and of the claims it is an end of

The belief texts and the split: ``pipeline/relevance.finish``.
"""

from typing import Dict, List, Optional, Tuple

from ...annotation import rows as annotation_rows
from ...config import BuildConfig, DatasetPaths
from ...corpus.pages import wiki_gold
from ...gold_passages.judge import fact_key, grade, judge_pairs, label_counts
from ...pipeline.relevance import Built
from .facts import claim_facts
from .tree import tree_nodes

FACTS = None
JUDGE = True


def _gold(record: dict) -> Tuple[List[dict], Dict[str, List[str]]]:
    """The claim facts of a prepared question, each with its gold chunk
    candidates (``chunks``: chunk 1 and chunk 2 of its page; one for a page
    of one chunk), and the chunks of each gold page (page key -> chunk ids)."""
    chunks = record["features"]["page_chunks"]
    facts = claim_facts(record, lambda href: wiki_gold(href)["key"])
    return [{**f, "chunks": chunks[f["pages"][0]][:2]} for f in facts], chunks


def _resolve(record: dict, facts: List[dict], chunks: Dict[str, List[str]],
             labels: Dict[Tuple[str, str], Optional[str]]):
    """The gold chunks of a question: (qrels in page order, ref -> gold
    chunks, per fact the chunk chosen); None when a claim has no label yet."""
    chosen: Dict[str, str] = {}
    for f in facts:
        if len(f["chunks"]) < 2:
            chosen[f["id"]] = f["chunks"][0]
            continue
        label = labels.get((fact_key(f), f["chunks"][1]))
        if label is None:
            return None
        chosen[f["id"]] = f["chunks"][1] if grade(label) == 2 else f["chunks"][0]
    page_gold: Dict[str, List[str]] = {}
    for f in facts:
        page_gold.setdefault(f["pages"][0], []).append(chosen[f["id"]])
    page_gold = {key: list(dict.fromkeys(page_gold.get(key) or ids[:1])) for key, ids in chunks.items()}
    ref_docs: Dict[str, List[str]] = {}
    root = record["meta"]["root"]
    for node in tree_nodes(root):
        if node.get("href"):
            ref_docs.setdefault(node["id"], []).extend(page_gold[wiki_gold(node["href"])["key"]])
    for f in facts:
        for ref in f["refs"]:
            ref_docs.setdefault(ref, []).append(chosen[f["id"]])
    doc_ids = [d for key in dict.fromkeys(g["key"] for g in record["gold"]) for d in page_gold[key]]
    return doc_ids, ref_docs, chosen


def build(cfg: BuildConfig, paths: DatasetPaths, processed: List[dict]) -> Built:
    # The gold chunks: one judge call per claim on chunk 2 of its page.
    gold = {r["id"]: _gold(r) for r in processed}
    pairs = [(f, f["chunks"][1]) for facts, _ in gold.values() for f in facts if len(f["chunks"]) > 1]
    labels = judge_pairs(cfg.gold_passages, paths, pairs) if pairs else {}
    kept, rows, pending, chosen_of = [], [], [], {}
    for record in processed:
        facts, chunks = gold[record["id"]]
        resolved = _resolve(record, facts, chunks, labels)
        if resolved is None:
            pending.append(record["id"])
            continue
        doc_ids, ref_docs, chosen = resolved
        kept.append(record)
        chosen_of[record["id"]] = [(f, chosen[f["id"]]) for f in facts]
        rows.append(annotation_rows.row(record, doc_ids, ref_docs))

    claims = [(f, c) for pairs_of in chosen_of.values() for f, c in pairs_of]
    section = {
        "judge": {"model": cfg.gold_passages.model},
        "claims": len(claims),
        "judged": sum(len(f["chunks"]) > 1 for f, _ in claims),
        "labels": label_counts(labels.get((fact_key(f), f["chunks"][1])) for f, _ in claims if len(f["chunks"]) > 1),
        "gold_chunk": {"chunk_1": sum(c == f["chunks"][0] for f, c in claims),
                       "chunk_2": sum(len(f["chunks"]) > 1 and c == f["chunks"][1] for f, c in claims)},
        "unfinished_questions": len(pending)}
    return Built(rows=rows, qrels={row["id"]: [f"{row['id']} 0 {d} 1\n" for d in row["doc_ids"]] for row in rows},
                 kept=kept, pending=pending, pending_reason="with a claim the judge gave no label",
                 sections={"gold_passages": section})
