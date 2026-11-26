"""The episodes of training: the finished queries of the generated runs (stage generate).

A run is dra_inference's run directory of one training dataset split and
one agent (``{output}/{dataset}_{split}_{retriever}/{agent}_{backend}_{model}/``,
found by the ``policy_model`` of its ``run_config.json``).  An episode is
one finished query of a run with a judged answer and gold beliefs:

* what the belief model reads of it, as the belief stage reads it
  (``belief_model_component.read_saved_query``): the search iterations
  (queries, documents, agent turn) and the answer turn's reasoning and
  final answer (a query without a final answer is not an episode);
* the correctness y of the agent's final answer
  (``eval/generation/accuracy.jsonl``; a query without a verdict, or whose
  judge failed, is not an episode);
* its gold beliefs (``read_gold_rows``), banked by stage teacher
  (``teacher.run.bank_teacher_beliefs``): the teacher's init beliefs
  where they come from the question (infoseek_hard, webshaper:
  multi-clue, as BrowseComp-Plus), the gold beliefs made from the
  dataset's rows with the teacher's texts where they are the annotation's
  (trqa_val, as TRQA's at inference);
* the run's settings the belief model needs: corpus, retriever (the
  document shape), the passage cut.

The documents' text is read from the run's corpus by id (``DocumentStore``),
as the belief stage reads it.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from indexing_corpus_dataset.shared.layout import (
    ALL_SPLIT, DATASET_SPECS, GOLD_BELIEFS, QUESTION_GOLD, belief_bank_path, build_dataset_dir_name, extraction_kind,
    gold_beliefs_path,
)
from utils.io import iter_jsonl
from utils.run_files import scan_queries
from utils.run_files.run_config import read_run_config

logger = logging.getLogger(__name__)

ACCURACY_FILE = Path("eval") / "generation" / "accuracy.jsonl"


@dataclass
class Episode:
    dataset: str
    split: str
    policy_model: str
    run_dir: Path
    qid: str
    question: str
    query_shape: str
    # belief_model_component.SavedQuery: iterations, answer reasoning and answer.
    saved: Any
    # Correctness of the agent's final answer.
    y: bool
    # Gold beliefs: [{id, text, kind, ...}] (the dataset row's or the teacher's banked ones).
    gold_beliefs: List[Dict[str, Any]]
    corpus_path: str
    retriever_name: str
    max_passage_tokens: int
    # The search iterations that showed a document (an upper bound on the update calls).
    doc_steps: int = 0

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.policy_model}/{self.qid}"


def find_run(output_root: Path, dataset: str, split: str, retriever: str, policy_model: str) -> Optional[Path]:
    """The run directory of *policy_model* on *split* of *dataset*, None when
    there is none; stops when several runs of the agent are there."""
    spec = DATASET_SPECS[dataset]
    dataset_dir = output_root / build_dataset_dir_name(dataset, split, spec.query_key, retriever)
    if not dataset_dir.is_dir():
        return None
    found = []
    for run_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if not (run_dir / "run_config.json").exists():
            continue
        if read_run_config(run_dir).get("agent", {}).get("policy_model") == policy_model:
            found.append(run_dir)
    if len(found) > 1:
        raise SystemExit(f"{dataset_dir}: several runs of {policy_model} ({[p.name for p in found]}); keep one")
    return found[0] if found else None


def read_accuracy(run_dir: Path) -> Dict[str, Optional[bool]]:
    """``{qid: correct}`` of the run's judged answers (None: the judge failed)."""
    path = run_dir / ACCURACY_FILE
    out: Dict[str, Optional[bool]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["query_id"])] = None if row.get("judge_error") else row.get("correct")
    return out


def teacher_bank(dataset: str, teacher: str):
    """The bank of *dataset*'s gold beliefs as the belief stage of
    ``--belief-model {teacher}`` banks them (``build_belief_source``), read
    only: the teacher's init beliefs of a dataset whose beliefs come from
    the question (model, shape, cap and prompt in the key); else the gold
    beliefs made from the dataset's rows with the teacher's texts
    (``init.trqa_gold.DatasetGoldBeliefs``: writer and prompt in the key)."""
    from belief_model_component import BankedBeliefSource, LLMBeliefSource, bank_model_id
    from belief_model_component.init.trqa_gold import DatasetGoldBeliefs
    from utils.models import BELIEF_MODEL_TO_LLM

    spec = DATASET_SPECS[dataset]
    model_id = bank_model_id(BELIEF_MODEL_TO_LLM[teacher])
    if spec.gold_beliefs == QUESTION_GOLD:
        extractor = LLMBeliefSource(llm_client=None, max_beliefs=spec.max_beliefs, model_name=model_id,
                                    query_shape=spec.query_shape, kind=extraction_kind(dataset))
        return BankedBeliefSource(extractor, belief_bank_path(dataset, ALL_SPLIT))
    return BankedBeliefSource(DatasetGoldBeliefs(gold_beliefs_path(dataset), writer=model_id),
                              belief_bank_path(dataset, ALL_SPLIT, GOLD_BELIEFS))


def read_gold_rows(dataset: str, teacher: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """``{qid: row}`` of the dataset's rows (the queries file) with usable
    gold beliefs in ``beliefs`` (id, text, kind): the *teacher*'s banked
    ones (``teacher_bank``); a query not banked yet, or without gold
    beliefs (FRAMES), is left out.  *teacher* None (stage generate's
    statistics, before stage teacher): every row, its ``beliefs`` empty."""
    bank = teacher_bank(dataset, teacher) if teacher is not None else None
    rows = {}
    for row in iter_jsonl(gold_beliefs_path(dataset)):
        qid = str(row["id"])
        if bank is None:
            rows[qid] = {**row, "beliefs": []}
        else:
            beliefs = bank.lookup(qid, row["text"])
            if beliefs:
                rows[qid] = {**row, "beliefs": [b.to_dict() for b in beliefs]}
    return rows


def load_run_episodes(run_dir: Path, dataset: str, split: str, policy_model: str,
                      gold_rows: Dict[str, Dict[str, Any]], counts: Optional[Dict[str, int]] = None) -> List[Episode]:
    """The episodes of one run (see the module docstring); *counts* gets
    the queries left out, by reason."""
    from belief_model_component import read_saved_query

    counts = counts if counts is not None else {}

    def drop(reason: str) -> None:
        counts[reason] = counts.get(reason, 0) + 1

    config = read_run_config(run_dir)
    corpus_path = config["dataset"]["corpus_path"]
    retriever = config["searcher"]["retriever_name"]
    passage_tokens = config["agent"].get("max_passage_tokens")
    if passage_tokens is None:
        raise SystemExit(f"{run_dir}: no max_passage_tokens in run_config.json (a run before the token cut); "
                         "generate it again")
    accuracy = read_accuracy(run_dir)
    finished = scan_queries(run_dir)[0]
    episodes = []
    for qid in sorted(finished):
        y = accuracy.get(qid)
        if y is None:
            drop("no_verdict")
            continue
        row = gold_rows.get(qid)
        if row is None:
            drop("no_gold_beliefs")
            continue
        saved = read_saved_query(run_dir, qid, config["agent"]["policy_model"])
        if saved is None:
            drop("unfinished")
            continue
        if not (saved.answer or "").strip():
            drop("no_answer")
            continue
        episodes.append(Episode(
            dataset=dataset, split=split, policy_model=policy_model, run_dir=run_dir, qid=qid,
            question=saved.question, query_shape=DATASET_SPECS[dataset].query_shape, saved=saved, y=bool(y),
            gold_beliefs=row["beliefs"], corpus_path=corpus_path, retriever_name=retriever,
            max_passage_tokens=int(passage_tokens),
            doc_steps=sum(1 for it in saved.iterations if it.doc_ids),
        ))
    return episodes


def load_episodes(output_root: Path, datasets: List[str], split: str, policy_models: List[str], retriever: str,
                  teacher: str, tag: str = "") -> List[Episode]:
    """The episodes of *split* over every dataset and agent (missing runs
    are reported and skipped); *teacher*: whose banked beliefs are the gold
    of a dataset whose beliefs come from the question (``read_gold_rows``)."""
    out: List[Episode] = []
    for dataset in datasets:
        gold_rows = read_gold_rows(dataset, teacher)
        for policy in policy_models:
            run_dir = find_run(output_root, dataset, split, retriever, policy)
            if run_dir is None:
                print(f"{tag} {dataset}/{split}/{policy}: no run (stage generate not run)")
                continue
            counts: Dict[str, int] = {}
            episodes = load_run_episodes(run_dir, dataset, split, policy, gold_rows, counts)
            print(f"{tag} {dataset}/{split}/{policy}: {len(episodes)} episodes"
                  + (f" (left out: {counts})" if counts else ""))
            out.extend(episodes)
    return out


class DocumentStore:
    """The documents of the episodes, read from each run's corpus by id and
    shaped as the run's retriever returns them
    (``belief_model_component.corpus_doc``); one corpus index per
    corpus file, shared by threads."""

    def __init__(self) -> None:
        self._lookups: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _lookup(self, corpus_path: str):
        from indexing_corpus_dataset.shared.doc_lookup import CorpusLookup

        with self._lock:
            lookup = self._lookups.get(corpus_path)
            if lookup is None:
                lookup = CorpusLookup(corpus_path)
                lookup.ensure_index()
                self._lookups[corpus_path] = lookup
            return lookup

    def docs(self, episode: Episode, doc_ids: List[str]) -> List[Dict[str, Any]]:
        from belief_model_component import corpus_doc

        contents = self._lookup(episode.corpus_path).contents(doc_ids)
        return [corpus_doc(i, contents.get(i, ""), episode.retriever_name) for i in doc_ids]
