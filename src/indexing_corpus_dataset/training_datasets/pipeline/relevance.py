"""Stage data_relevance: the gold passages (qrels), rows and split of the
prepared questions of a training dataset (``DatasetPaths.prepared``, stage
data_prepare), with the LLM calls that locate the gold passages: the fact
writer (frames, webshaper; ``--fact-model``) and the judge of the gold
passages (infoseek_hard, frames, webshaper; ``--judge-model``).  trqa_val
makes no call (TRQA's qrels): the stage writes its rows and split only.
No beliefs here: the stage finds the relevant passages of each question;
the belief model makes its gold beliefs when it first needs them (stage
teacher).

    preflight  the memories of the fact LLM and the judge hold no other
               model or prompts, the split shares are the recorded ones
               (the dataset's ``FACTS`` and ``JUDGE`` say which it uses)
    build      the dataset's rows and qrels (``datasets/{dataset}/relevance.py``,
               ``build``: a ``Built``)
    finish     the split (``splits.py``), the queries and qrels files, the
               ``relevance`` and dataset sections of ``stats.json`` and the
               questions dropped here

Incremental: with a *limit* it processes the next N prepared questions not
processed yet (kept or dropped here), in file order (0: none, the dataset
files are rewritten); without one, every prepared question.  A question
whose LLM calls are not final stays unprocessed and is taken first by the
next run.  A dataset whose queries file holds a question the prepared file
no longer has (data_prepare was rerun with other checks) is refused: its
relevance files must be removed first.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.io import atomic_write_text, iter_jsonl, read_jsonl

from .. import registry
from ..annotation import rows as annotation_rows
from ..config import BuildConfig, DatasetPaths
from ..gold_passages.facts import check_fact_writer
from ..llm import require_api_key
from ..gold_passages.judge import check_judge, require_judge_key
from .splits import check_shares, split_rows
from .stats import drop_stats, processed_ids, question_stats, write_dropped, write_stats

# The stage of the questions dropped here, in ``dropped.jsonl``.
STAGE = "relevance"


@dataclass
class Built:
    """What a dataset's build gives for the processed questions."""
    # The rows of the kept questions (``annotation_rows.row``), without
    # their split.
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
                             f"{paths.queries}, {paths.qrels} and the '{STAGE}', 'split' sections "
                             f"of {paths.stats}, then run --stage data_relevance again (the LLM memories "
                             "facts.jsonl and gold_passages.jsonl are reused)")
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


def _summary(paths: DatasetPaths, records: List[dict], n_processed: int, built: Built) -> None:
    """The outcome over every processed question: kept, dropped by reason,
    pending, the qrels (binary) and where the relevant passages of the gold
    pages come from."""
    reasons = Counter(r["status"] for d in built.dropped for r in {x["status"]: x for x in d["reasons"]}.values())
    n_qrels = sum(len(lines) for lines in built.qrels.values())
    gold = built.sections.get("gold_passages", {})
    print(f"[{paths.dataset}] done (all runs so far): {len(built.kept)} kept, {len(built.dropped)} dropped"
          + (f" ({', '.join(f'{k} {v}' for k, v in reasons.most_common())})" if reasons else "")
          + f", {len(built.pending)} pending (next run); {len(records) - n_processed} of {len(records)} "
          "prepared questions not processed yet")
    pages = gold.get("pages")
    print(f"[{paths.dataset}] qrels: {n_qrels} relevant passages"
          + (f"; gold pages: {', '.join(f'{k} {v}' for k, v in pages.items())}, "
             f"{gold['questions_all_chunk_1']} questions with every page on chunk 1"
             if isinstance(pages, dict) else "")
          + f" -> {paths.qrels}")


def finish(cfg: BuildConfig, paths: DatasetPaths, records: List[dict], processed: List[dict], built: Built) -> None:
    """The dataset files of *built*: split, queries and qrels, then the
    stats."""
    if built.pending:
        print(f"[relevance] {len(built.pending)} questions {built.pending_reason}: left for the next run")
    annotation_rows.check(built.rows)
    rows = split_rows(paths, cfg.split, built.rows)
    annotation_rows.write(paths.queries, rows)
    atomic_write_text(paths.qrels, "".join(line for row in rows for line in built.qrels[row["id"]]))

    n_processed = len(processed) - len(built.pending)
    write_relevance(paths, records, n_processed, built.kept, built.dropped)
    for name, section in built.sections.items():
        write_stats(paths.stats, name, section)
    _summary(paths, records, n_processed, built)


def _phases(cfg: BuildConfig, dataset, name: str, n_todo: int) -> None:
    """The phases of the run, before they start."""
    if not dataset.JUDGE:
        print(f"[relevance] {name}: no LLM call (the source's qrels); rows, split and files")
        return
    steps = []
    if dataset.FACTS is not None:
        steps.append(f"facts ({cfg.facts.model})")
    steps.append(f"judge ({cfg.gold_passages.model}, {cfg.gold_passages.workers} workers"
                 + (", every passage of a fact's pages" if dataset.FACTS is not None else "") + ")")
    steps.append("rows, split and files")
    print(f"[relevance] {name}: {n_todo} new questions; phases: " + " -> ".join(steps))


def build_relevance(cfg: BuildConfig, paths: DatasetPaths, limit: Optional[int] = None) -> None:
    records = prepared_records(paths)
    dataset = registry.relevance(paths.dataset)
    if dataset.FACTS is not None:
        check_fact_writer(cfg.facts, paths, dataset.FACTS)
    if dataset.JUDGE:
        check_judge(cfg.gold_passages, paths)
    check_shares(paths, cfg.split)
    todo, processed = pick(paths, records, limit)
    _phases(cfg, dataset, paths.dataset, len(todo))
    if todo:
        if dataset.FACTS is not None:
            require_api_key(cfg.facts.model, "the facts of the new questions", "--fact-model")
        if dataset.JUDGE:
            require_judge_key(cfg.gold_passages, "the gold passages of the new questions")
    finish(cfg, paths, records, processed, dataset.build(cfg, paths, processed))
