"""The gold passages of the frames and webshaper questions from their facts
(``facts.py``) and the judge (``judge.py``):

    facts   the dataset's fact writer (``facts.fill_facts``); a question
            whose writer failed for good is dropped (``facts_failed``), one
            whose writer call failed is left for the next run
    pairs   each fact with every passage of its gold pages (stage
            data_prepare keeps only the pages in the corpus); a fact
            without a page (webshaper: the writer gave its triple no page)
            with every passage of the question's pages
    judge   one call per pair (``judge.judge_pairs``)
    keep    a question is kept when every fact has a gold passage (grade 2);
            else dropped (``fact_without_gold``, with the facts); a pair
            without a label leaves the question unfinished (next run)
    qrels   per passage the highest grade over the question's facts: 2 (it
            states a fact, the gold) and 1 (on a fact's topic) are written,
            graded; the row's ``doc_ids`` are the grade-2 passages, a
            belief's those of the facts about its ref (``refs``)
"""

from typing import Dict, List, Optional, Tuple

from ..config import BuildConfig, DatasetPaths, GoldPassagesConfig
from .facts import FactSpec, fill_facts
from .judge import fact_key, grade, judge_pairs, label_counts

# Why a question is dropped when a fact has no gold passage.
NO_GOLD = "fact_without_gold"
# Why a question is dropped when its facts failed for good.
FACTS_FAILED = "facts_failed"


def fact_docs(fact: dict, chunks: Dict[str, List[str]]) -> List[str]:
    """The passages a fact is judged on: those of its pages (*chunks*: page
    key -> passage ids), or of every page of the question when it has
    none."""
    own = [d for key in fact["pages"] for d in chunks.get(key, [])]
    return list(dict.fromkeys(own or [d for docs in chunks.values() for d in docs]))


def judge_questions(gc: GoldPassagesConfig, paths: DatasetPaths, items: List[dict]
                    ) -> Tuple[List[dict], List[dict], List[str], dict]:
    """*items*: one per question, ``{"record", "chunks": {page key: passage
    ids}, "facts": [...]}``.  Returns the kept items (each with ``grades``:
    passage -> highest grade, and ``gold``: fact id -> its grade-2
    passages), the dropped questions (``{"id", "source", "reasons"}``), the
    ids of the unfinished ones and the stats of the judgments."""
    pairs = [(fact, d) for item in items for fact in item["facts"] for d in fact_docs(fact, item["chunks"])]
    labels = judge_pairs(gc, paths, pairs) if pairs else {}
    kept, dropped, pending = [], [], []
    per_fact: List[int] = []
    for item in items:
        record = item["record"]
        gold: Dict[str, List[str]] = {}
        grades: Dict[str, int] = {}
        unfinished = False
        for fact in item["facts"]:
            got: List[Tuple[str, Optional[str]]] = [(d, labels.get((fact_key(fact), d)))
                                                    for d in fact_docs(fact, item["chunks"])]
            if any(label is None for _, label in got):
                unfinished = True
                break
            gold[fact["id"]] = [d for d, label in got if grade(label) == 2]
            for d, label in got:
                grades[d] = max(grades.get(d, 0), grade(label))
        if unfinished:
            pending.append(record["id"])
            continue
        per_fact += [len(v) for v in gold.values()]
        missing = [f for f in item["facts"] if not gold[f["id"]]]
        if missing:
            dropped.append({"id": record["id"], "source": record["source"],
                            "reasons": [{"status": NO_GOLD, "fact": f["statement"]} for f in missing]})
        else:
            kept.append({**item, "grades": grades, "gold": gold})
    unique = {(fact_key(f), d) for f, d in pairs}
    stats = {"pairs": len(unique), "labels": label_counts(labels.get(k) for k in unique),
             "facts": sum(len(i["facts"]) for i in items),
             "facts_with_gold": sum(n > 0 for n in per_fact),
             "gold_per_fact": _mean(per_fact),
             "unfinished_questions": len(pending)}
    return kept, dropped, pending, stats


def _mean(values: List[int]) -> Optional[float]:
    return round(sum(values) / len(values), 2) if values else None


def qrels_lines(qid: str, grades: Dict[str, int]) -> List[str]:
    """The graded qrels of a question: every passage of grade 1 or 2."""
    return [f"{qid} 0 {d} {g}\n" for d, g in grades.items() if g > 0]


def ref_docs(item: dict) -> Dict[str, List[str]]:
    """The gold passages of each source ref of a kept item: those of the
    facts about it."""
    out: Dict[str, List[str]] = {}
    for fact in item["facts"]:
        for ref in fact["refs"]:
            out.setdefault(ref, []).extend(item["gold"][fact["id"]])
    return {ref: list(dict.fromkeys(docs)) for ref, docs in out.items()}


def gold_docs(item: dict) -> List[str]:
    """The gold passages of a kept item (grade 2), in fact order."""
    return list(dict.fromkeys(d for fact in item["facts"] for d in item["gold"][fact["id"]]))


def qrels_of(kept: List[dict]) -> Dict[str, List[str]]:
    """The graded qrels lines of each kept item, by question id."""
    return {item["record"]["id"]: qrels_lines(item["record"]["id"], item["grades"]) for item in kept}


def facts_and_gold(cfg: BuildConfig, paths: DatasetPaths, spec: FactSpec, found: List[Tuple[dict, List[dict]]]
                   ) -> Tuple[List[dict], List[dict], List[str], dict]:
    """The facts and gold passages of the questions of *found* (record, its
    gold pages as the fact writer sees them; each record holds the passage
    ids of its pages, ``features.page_chunks``).  Returns the kept items
    (``judge_questions``, with ``names``: the variable names of the fact
    writer), the dropped questions, the ids of the unfinished ones and the
    ``gold_passages`` section of ``stats.json``."""
    facts = fill_facts(cfg.gold_beliefs, paths, spec, found)
    items, pending, dropped = [], [], []
    for record, _ in found:
        got = facts.get(record["id"])
        if got is None:
            pending.append(record["id"])
        elif not got["ok"]:
            dropped.append({"id": record["id"], "source": record["source"],
                            "reasons": [{"status": FACTS_FAILED, "errors": got["errors"]}]})
        else:
            items.append({"record": record, "chunks": record["features"]["page_chunks"], "facts": got["facts"],
                          "names": got["names"]})
    kept, no_gold, unjudged, judge_stats = judge_questions(cfg.gold_passages, paths, items)
    grades = [item["grades"] for item in kept]
    section = {"judge": {"model": cfg.gold_passages.model}, "fact_writer": {"model": cfg.gold_beliefs.model},
               **judge_stats, "qrels": {g: sum(v == g for gr in grades for v in gr.values()) for g in (2, 1)}}
    return kept, dropped + no_gold, pending + unjudged, section
