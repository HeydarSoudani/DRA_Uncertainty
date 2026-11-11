"""The report stage of webshaper_frames: stats of the dataset as written
(queries, qrels, corpus) per source, the hard negatives, the gold annotation
(levels, criteria with answers and documents), consistency checks between
the files, and the duplicate documents of the corpus.
"""

from pathlib import Path
from typing import Dict, List

from utils.io_utils import iter_jsonl, read_jsonl

from ..config import DatasetPaths
from ..stats import annotation_stats, dataset_stats, format_annotation_table, format_dataset_table, write_stats
from .corpus import read_negatives
from .duplicates import find_duplicates, format_duplicates


def _read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def run_report(paths: DatasetPaths, dedup_threshold: float) -> None:
    """Stats of the dataset as written (queries, qrels, corpus, hard
    negatives): printed as a table and saved as the ``report`` section of
    ``stats.json``; the gold annotation of the rows is printed only (the
    pages stage saves it, in the ``annotations`` section)."""
    for path in (paths.questions, paths.queries, paths.qrels, paths.corpus):
        if not path.exists():
            print(f"[report] {path} not found (no question processed yet): no report")
            return
    qrels: Dict[str, List[str]] = {}
    with open(paths.qrels, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                qid, _, doc_id, _ = line.split()
                qrels.setdefault(qid, []).append(doc_id)
    doc_words = {row["id"]: len(row["contents"].split()) for row in iter_jsonl(paths.corpus)}
    dropped = [r for r in iter_jsonl(paths.dropped) if r["stage"] == "pages"] if paths.dropped.exists() else []
    queries = read_jsonl(paths.queries)
    stats = dataset_stats(read_jsonl(paths.questions), queries, qrels, doc_words, dropped,
                          _read_ids(paths.ids_hard_negative), read_negatives(paths))
    annotation = annotation_stats(queries)
    stats["duplicates"] = find_duplicates(paths, dedup_threshold)
    write_stats(paths.stats, "report", stats)
    print(f"[report] dataset {paths.root}\n\n{format_dataset_table(stats)}\n\n"
          f"{format_duplicates(stats['duplicates'])}\n(details: {paths.duplicates})\n\n"
          f"gold annotation\n\n{format_annotation_table(annotation)}\n")
