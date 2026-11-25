"""Stage data_relevance: the gold annotation, qrels and split of the
prepared questions of a training dataset (``DatasetPaths.prepared``, stage
data_prepare), with the LLM calls: the judge of the gold passages, the fact
writer (frames, webshaper) and the writer of the belief texts.

    preflight  the memories of the writer, the fact writer and the judge
               hold no other model or prompts, the split shares are the
               recorded ones (the dataset's ``FACTS`` and ``JUDGE`` say
               which it uses)
    build      the dataset's rows and qrels (``datasets/{dataset}/relevance.py``,
               ``build``: a ``Built``)
    finish     the belief texts (``annotation/belief_texts.py``), the split
               (``splits.py``), the queries and qrels files, the
               ``annotations``, ``relevance`` and dataset sections of
               ``stats.json`` and the questions dropped here

Incremental: with a *limit* it processes the next N prepared questions not
processed yet (kept or dropped here), in file order (0: none, the dataset
files are rewritten); without one, every prepared question.  A question
whose LLM calls are not final stays unprocessed and is taken first by the
next run.  A dataset whose queries file holds a question the prepared file
no longer has (data_prepare was rerun with other checks) is refused: its
relevance files must be removed first.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.io import atomic_write_text, iter_jsonl, read_jsonl

from .. import registry
from ..annotation import rows as annotation_rows
from ..annotation.belief_texts import check_writer, fill_texts, require_api_key
from ..annotation.stats import format_annotation_table, write_annotation_stats
from ..config import BuildConfig, DatasetPaths
from ..gold_passages.facts import check_fact_writer
from ..gold_passages.judge import check_judge, require_judge_key
from .splits import check_shares, split_rows
from .stats import drop_stats, processed_ids, question_stats, write_dropped, write_stats

# The stage of the questions dropped here, in ``dropped.jsonl``.
STAGE = "relevance"


@dataclass
class Built:
    """What a dataset's build gives for the processed questions."""
    # The annotation rows of the kept questions (``annotation_rows.row``),
    # without belief texts or split.
    rows: List[dict]
    # The qrels lines of each kept question, by id.
    qrels: Dict[str, List[str]]
    # The kept records (for the ``relevance`` stats).
    kept: List[dict]
    # The dropped questions: ``{"id", "source", "reasons"}``.
    dropped: List[dict] = field(default_factory=list)
    # The ids of the questions left for the next run, and why.
    pending: List[str] = field(default_factory=list)
    pending_reason: str = "with a failed call"
    # Other sections of ``stats.json`` (e.g. ``gold_passages``).
    sections: Dict[str, dict] = field(default_factory=dict)


def prepared_records(paths: DatasetPaths) -> List[dict]:
    """The prepared questions, after checking that the dataset holds no
    query outside them."""
    if not paths.prepared.exists():
        raise SystemExit(f"{paths.prepared} does not exist: run --stage data_prepare --dataset {paths.dataset} first")
    records = read_jsonl(paths.prepared)
    ids = {str(r["id"]) for r in records}
    if paths.queries.exists():
        stale = [str(r["id"]) for r in iter_jsonl(paths.queries) if str(r["id"]) not in ids]
        if stale:
            raise SystemExit(f"{len(stale)} queries of {paths.queries} are not prepared questions any more "
                             f"(e.g. {stale[:3]}; data_prepare was rerun with other checks).  Remove "
                             f"{paths.queries}, {paths.qrels} and the '{STAGE}', 'annotations', 'split' sections "
                             f"of {paths.stats}, then run --stage data_relevance again (the LLM caches "
                             "belief_texts.jsonl, facts.jsonl and gold_passages.jsonl are reused)")
    return records


def pick(paths: DatasetPaths, records: List[dict], limit: Optional[int]) -> Tuple[List[dict], List[dict]]:
    """``(todo, processed)``: the prepared questions to process now (the
    next *limit* not processed yet) and every one processed after this run,
    in file order."""
    done = processed_ids(paths, STAGE)
    remaining = [r for r in records if str(r["id"]) not in done]
    todo = remaining if limit is None else remaining[:limit]
    print(f"[relevance] {len(records)} prepared questions: {len(records) - len(remaining)} already processed, "
          f"{len(todo)} to process" + (f" (--limit {limit})" if limit is not None else ""))
    done |= {str(r["id"]) for r in todo}
    return todo, [r for r in records if str(r["id"]) in done]


def write_relevance(paths: DatasetPaths, records: List[dict], n_processed: int, kept: List[dict],
                    dropped: List[dict]) -> None:
    """The ``relevance`` section of ``stats.json`` and the questions
    dropped here."""
    write_dropped(paths.dropped, STAGE, dropped)
    write_stats(paths.stats, STAGE, {"prepared": len(records), "processed": n_processed,
                                     "remaining": len(records) - n_processed,
                                     "kept": question_stats(kept), "dropped": drop_stats(dropped)})


def finish(cfg: BuildConfig, paths: DatasetPaths, records: List[dict], processed: List[dict], built: Built) -> None:
    """The dataset files of *built*: belief texts, split, queries and
    qrels, then the stats.  The texts first: a stopped run leaves the files
    of the last full one."""
    if built.pending:
        print(f"[relevance] {len(built.pending)} questions {built.pending_reason}: left for the next run")
    annotation_rows.check(built.rows)
    rows = split_rows(paths, cfg.split, fill_texts(cfg.gold_beliefs, paths, built.rows))
    annotation_rows.write(paths.queries, rows)
    atomic_write_text(paths.qrels, "".join(line for row in rows for line in built.qrels[row["id"]]))

    stats = write_annotation_stats(paths, cfg.gold_beliefs, rows)
    n_processed = len(processed) - len(built.pending)
    write_relevance(paths, records, n_processed, built.kept, built.dropped)
    for name, section in built.sections.items():
        write_stats(paths.stats, name, section)
    print(f"[{paths.dataset}] {len(built.kept)} kept, {len(built.dropped)} dropped of {n_processed} processed "
          f"questions ({len(records) - n_processed} remaining); "
          f"{sum(map(len, built.qrels.values()))} qrels -> {paths.queries}")
    print(f"[{paths.dataset}] gold annotation\n\n{format_annotation_table(stats)}\n")


def build_relevance(cfg: BuildConfig, paths: DatasetPaths, limit: Optional[int] = None) -> None:
    records = prepared_records(paths)
    dataset = registry.relevance(paths.dataset)
    check_writer(cfg.gold_beliefs, paths)
    if dataset.FACTS is not None:
        check_fact_writer(cfg.gold_beliefs, paths, dataset.FACTS)
    if dataset.JUDGE:
        check_judge(cfg.gold_passages, paths)
    check_shares(paths, cfg.split)
    todo, processed = pick(paths, records, limit)
    if todo:
        require_api_key(cfg.gold_beliefs, "the facts and belief texts of the new questions"
                        if dataset.FACTS is not None else "the belief texts of the new questions")
        if dataset.JUDGE:
            require_judge_key(cfg.gold_passages, "the gold passages of the new questions")
    finish(cfg, paths, records, processed, dataset.build(cfg, paths, processed))
