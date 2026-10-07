"""Report evaluation with Auto-ARGUE, the scorer of NeuCLIR 2024 and RAGTIME 2025.

Auto-ARGUE (Walden et al., 2025, arXiv:2509.26184; https://github.com/hltcoe/auto-argue,
installed at commit :data:`AUTO_ARGUE_COMMIT`) judges a report sentence by
sentence against the request's nuggets, each nugget a question with gold
answers and, per answer, the documents that support it:

* a sentence with citations: is each cited document relevant (it supports a
  nugget answer), does it attest the sentence, and which nugget answers of
  the relevant cited documents does the sentence state;
* a sentence without citations: does it need one, and is it the first
  instance of its claim.

Per report it scores:

* ``nugget_coverage``: nuggets answered / nuggets; ``_weighted`` counts vital
  nuggets 2 and okay ones 1, and equals the plain one without importance
  labels (RAGTIME).
* ``sentence_support``: fully supported sentences / sentences that cite or
  need a citation (first instances only).
* ``citation_support`` / ``citation_relevance``: citations that attest their
  sentence / that point to a nugget-supporting document, over all citations.
* ``f1`` / ``f1_weighted``: of ``sentence_support`` and the (weighted) coverage.

The package runs unmodified; this module supplies its inputs and its judge.

* Reports: ``generation/{qid}.md`` split into cited sentences
  (:mod:`.report_sentences`), cut to the dataset's ``report_chars`` with the
  package's ``truncate_report`` (the tracks' 2000-character limit).
* Nuggets: the dataset's nuggets file as one v3 nugget bank per request:
  questions grouped, ``vital`` when any copy is, AND/OR when given (else OR),
  each answer with its own supporting documents when given, else its
  question's.  The package drops answers without documents and questions
  without answers; a request with no nugget left is not scored.
* Cited document texts: from the corpus (``indexing_corpus_dataset.doc_lookup``).
* Judge: :data:`evaluation.judge.DEFAULT_JUDGE_MODEL` (Qwen3-32B) instead of
  the package's Llama-3.3-70B, as :class:`evaluation.judge.YesNoJudge`:
  temperature 0 with the model's reasoning off (the package reads YES/NO
  from at most 10 tokens, and a reasoning model spends them before
  answering), a reply without YES/NO asked again twice.  The package takes
  the check's default answer for one that still has none
  (``judge_malformed``).

Everything is cached under ``{run_dir}/report_eval/``: the cited documents and
each report's judgments, keyed by a hash of what was judged, so evaluating an
unchanged run again makes no LLM call.  Scores are always recomputed from the
judgments by the package's ``score``.  A request whose report has no sentence
scores 0 on every metric; a report whose judging fails is left out, counted in
``num_judge_failures``, and judged again on the next evaluation.
"""

import asyncio
import csv
import functools
import hashlib
import importlib
import json
import logging
import pickle
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from indexing_corpus_dataset.doc_lookup import CorpusLookup

from ..common import mean_or_none
from ..judge import DEFAULT_JUDGE_MODEL, YesNoJudge
from .report_sentences import report_sentences

logger = logging.getLogger(__name__)

AUTO_ARGUE_COMMIT = "81dda60"

#: Per-report metrics, averaged over the evaluated requests.
METRICS = (
    "nugget_coverage", "nugget_coverage_weighted", "sentence_support",
    "citation_support", "citation_relevance", "f1", "f1_weighted",
)
#: Per-report counts kept in the per-query records.
COUNTS = ("sentences", "character_count", "citations", "nuggets", "correct_nuggets")

# Collection name of the cited-document lookup handed to the package.
_DOC_COLLECTION = "dra_cited_docs"
_RUN_ID = "run"
_IMPORTANCE_RANK = {"vital": 2, "okay": 1}


# ---------------------------------------------------------------------------
# The package, imported and patched
# ---------------------------------------------------------------------------

@functools.cache
def _import_auto_argue():
    """Import the package without its import-time side effects: ``score.py``
    calls ``logging.basicConfig`` and seeds ``random`` and ``numpy``."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    py_state, np_state = random.getstate(), np.random.get_state()
    # importlib: ``auto_argue.score`` the module, which the package's
    # ``__init__`` shadows with its ``score`` function.
    judge, score, utils = (importlib.import_module(f"auto_argue.{m}") for m in ("judge", "score", "utils"))
    root.handlers[:], root.level = handlers, level
    random.setstate(py_state)
    np.random.set_state(np_state)
    return judge, score, utils


class _QuietGather:
    """Stands in for the package's per-batch ``tqdm_asyncio`` progress bars."""

    @staticmethod
    async def gather(*coros, **_):
        return await asyncio.gather(*coros)


class _JudgeChat(YesNoJudge):
    """The chat model the package calls (``ainvoke`` of a LangChain chat model),
    answered by our YES/NO judge."""

    _ROLES = {"system": "system", "human": "user", "ai": "assistant"}

    async def ainvoke(self, messages, **_):
        chat = [{"role": self._ROLES.get(m.type, "user"), "content": m.content} for m in messages]
        return SimpleNamespace(content=await self.ask(chat))


@contextmanager
def _patched(utils, judge, chat: _JudgeChat):
    """Route the package's model calls to *chat* and silence its progress bars
    and per-lookup logging for the duration."""
    saved = (utils.get_model, utils.tqdm_asyncio, judge.tqdm_asyncio)
    argue_logger = logging.getLogger("auto_argue")
    saved_level = argue_logger.level
    utils.get_model = lambda *args, **kwargs: chat
    utils.tqdm_asyncio = judge.tqdm_asyncio = _QuietGather
    argue_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        utils.get_model, utils.tqdm_asyncio, judge.tqdm_asyncio = saved
        argue_logger.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def build_nugget_banks(nuggets: Dict[str, List[Dict[str, Any]]],
                       questions: Dict[str, str]) -> Dict[str, Any]:
    """``{query_id: NuggetBank}`` (v3) from the dataset's nugget records, for
    the requests with at least one answer that has supporting documents."""
    _import_auto_argue()
    from auto_argue.validation.nugget_data import AggregatorType, Answer, NuggetBank, NuggetQuestion

    banks = {}
    for qid, items in nuggets.items():
        grouped: Dict[str, Dict[str, Any]] = {}
        for item in items:
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            g = grouped.setdefault(question, {"answers": {}, "pooled": set(),
                                              "importance": None, "aggregator": None})
            g["pooled"].update(item.get("support_docs") or [])
            importance = item.get("importance")
            if _IMPORTANCE_RANK.get(importance, 0) > _IMPORTANCE_RANK.get(g["importance"], 0):
                g["importance"] = importance
            g["aggregator"] = g["aggregator"] or item.get("aggregator")
            answer_docs = item.get("answer_docs") or {}
            for raw in item.get("answers") or []:
                answer = str(raw).strip()
                if answer:
                    g["answers"].setdefault(answer, set()).update(answer_docs.get(raw) or [])

        bank = NuggetBank(query_id=qid, title_query=questions.get(qid, qid),
                          full_query=questions.get(qid), test_collection=_DOC_COLLECTION)
        for question, g in grouped.items():
            answers = [Answer.from_lazy(answer=a, references=sorted(docs or g["pooled"]))
                       for a, docs in g["answers"].items() if docs or g["pooled"]]
            if answers:
                bank.add_nuggets(NuggetQuestion.from_lazy(
                    query_id=qid, question=question, gold_answers=answers,
                    aggregator_type=AggregatorType(g["aggregator"] or "OR"),
                    importance=g["importance"]))
        if bank.nugget_bank:
            banks[qid] = bank
    return banks


def build_report(query_id: str, generation: str, max_chars: Optional[int]) -> Tuple[Any, int]:
    """``(Report, unmapped_markers)``: the report as Auto-ARGUE reads it."""
    _, _, utils = _import_auto_argue()
    sentences, unmapped = report_sentences(generation)
    report = utils.Report(
        metadata=utils.ReportMetadata(team_id="dra", run_id=_RUN_ID, topic_id=query_id),
        responses=[utils.ReportResponse(text=text, citations=set(cited)) for text, cited in sentences],
        references=sorted({d for _, cited in sentences for d in cited}),
    )
    if max_chars and report.responses:
        report = utils.truncate_report(report, max_chars=max_chars)
    return report, unmapped


def _report_key(report, bank_json: str, docs: Dict[str, Dict[str, str]], judge_model: str) -> str:
    """Hash of everything a report's judgments depend on."""
    payload = {
        "sentences": [[r.text, sorted(r.citations)] for r in report.responses],
        "nuggets": bank_json,
        "docs": {d: docs.get(d) for d in sorted(report.references)},
        "judge": judge_model,
        "auto_argue": AUTO_ARGUE_COMMIT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ArgueReportEvaluator:
    """Score reports against the dataset's nuggets with Auto-ARGUE.

    Usage::

        evaluator = ArgueReportEvaluator(nuggets, questions, corpus_path, max_chars=2000)
        metrics = evaluator.evaluate(results, run_dir)
        print("\n".join(evaluator.summary_lines(metrics)))

    Args:
        nuggets: ``query_id -> [nugget record]`` (``dataset_loaders.load_nuggets``).
        questions: ``query_id -> request text``.
        corpus_path: The corpus JSONL the cited document ids come from.
        max_chars: Report length limit; longer reports are cut at a sentence.
        judge_model: Judge model, resolved by ``reasoner_component.api``.
        max_concurrent_judges: Max in-flight judge requests.
    """

    def __init__(
        self,
        nuggets: Dict[str, List[Dict[str, Any]]],
        questions: Dict[str, str],
        corpus_path,
        max_chars: Optional[int] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        max_concurrent_judges: int = 64,
    ) -> None:
        self.nuggets = nuggets
        self.questions = questions
        self.corpus_path = Path(corpus_path)
        self.max_chars = max_chars
        self.judge_model = judge_model
        self.max_concurrent_judges = max_concurrent_judges

    # -- cited documents ------------------------------------------------

    def _cited_docs(self, reports: Dict[str, Any], out_dir: Path) -> None:
        """Write ``doc_mapping_{collection}.pkl`` (the package's lookup) with
        the text of every cited document, fetching only the ones not cached."""
        path = out_dir / f"doc_mapping_{_DOC_COLLECTION}.pkl"
        docs: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with open(path, "rb") as f:
                docs = pickle.load(f)
        missing = {d for r in reports.values() for d in r.references} - set(docs)
        if missing:
            docs.update(CorpusLookup(self.corpus_path).get(missing))
            with open(path, "wb") as f:
                pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._docs = docs

    # -- judging --------------------------------------------------------

    async def _judge(self, judge, utils, todo: List[Tuple[str, Any, Path]], doc_dir: Path) -> Dict[str, Any]:
        """``{query_id: JudgedReport or Exception}`` for the reports of *todo*."""
        utils._singleton_semaphore = asyncio.Semaphore(self.max_concurrent_judges)

        async def one(qid, report, bank_path):
            try:
                judged = await judge.evaluate_report(
                    report, bank_path, [_DOC_COLLECTION], collection_dir=doc_dir,
                    provider=self.judge_model)
            except Exception as e:  # the package retries; a failure here is final
                return qid, e
            # The package turns a failed sentence into one without judgments.
            if any(not s.judgments for s in judged.sentence_judgments):
                return qid, RuntimeError("a sentence was left unjudged")
            return qid, judged

        out: Dict[str, Any] = {}
        tasks = [asyncio.ensure_future(one(*item)) for item in todo]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="[ARGUE]", dynamic_ncols=True):
            qid, judged = await fut
            out[qid] = judged
        return out

    # -- scoring --------------------------------------------------------

    @staticmethod
    def _score(score_module, judged: Dict[str, Dict], banks_dir: Path, out_dir: Path) -> Dict[str, Dict[str, float]]:
        """Per-request metrics of the package's ``score`` over *judged*."""
        judgments_path, tsv_path = out_dir / "judgments.jsonl", out_dir / "scores.tsv"
        with open(judgments_path, "w", encoding="utf-8") as f:
            for qid in sorted(judged):
                f.write(json.dumps(judged[qid], ensure_ascii=False) + "\n")
        argue_logger = logging.getLogger("auto_argue")
        level = argue_logger.level
        argue_logger.setLevel(logging.ERROR)  # one "no importance weights" warning per RAGTIME request
        try:
            score_module.score(judgments_jsonl=judgments_path, nuggets_path=banks_dir, output_tsv=tsv_path,
                               topics=[], run_ids=[], validate=True, penalize_missing_topics=False)
        finally:
            argue_logger.setLevel(level)
        per_topic: Dict[str, Dict[str, float]] = {}
        with open(tsv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["request_id"] != "all":
                    per_topic.setdefault(row["request_id"], {})[row["metric"]] = float(row["value"])
        return per_topic

    # -- public API -----------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]], run_dir=None) -> Dict[str, Any]:
        """Judge (or reuse the cached judgments of) every report with nuggets
        and score them.

        Returns ``num_evaluated`` (requests with nuggets), ``num_without_nuggets``,
        ``num_empty_reports``, ``num_judge_failures``, ``metrics`` (mean of each
        of :data:`METRICS` over the evaluated requests but the failures),
        ``judge_calls`` / ``judge_malformed`` (this evaluation's calls) and
        ``per_query``; ``{}`` when no request has nuggets.
        """
        judge, score_module, utils = _import_auto_argue()
        out_dir = Path(run_dir) / "report_eval" if run_dir else Path(tempfile.mkdtemp(prefix="report_eval_"))
        banks_dir, cache_dir = out_dir / "nuggets", out_dir / "judgments"
        banks_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        banks = build_nugget_banks({q: n for q, n in self.nuggets.items() if q in results}, self.questions)
        if not banks:
            logger.warning("No report with nuggets to evaluate")
            return {}
        bank_json: Dict[str, str] = {}
        for qid, bank in banks.items():
            bank_json[qid] = bank.model_dump_json(exclude_none=True, indent=1)
            (banks_dir / f"nuggets_{qid}.v3.json").write_text(bank_json[qid], encoding="utf-8")
        for stale in banks_dir.glob("nuggets_*.v3.json"):
            if stale.name.split("_", 1)[1].split(".")[0] not in banks:
                stale.unlink()

        reports, unmapped = {}, {}
        for qid in banks:
            reports[qid], unmapped[qid] = build_report(qid, results[qid].get("generation", ""), self.max_chars)
        empty = {qid for qid, r in reports.items() if not r.responses}
        to_score = {qid: r for qid, r in reports.items() if qid not in empty}
        self._cited_docs(to_score, out_dir)

        judged: Dict[str, Dict] = {}
        todo = []
        for qid, report in to_score.items():
            key = _report_key(report, bank_json[qid], self._docs, self.judge_model)
            cache = cache_dir / f"{qid}.json"
            if cache.exists():
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("key") == key:
                    judged[qid] = cached["judged"]
                    continue
            todo.append((qid, report, banks_dir / f"nuggets_{qid}.v3.json", key))

        chat = _JudgeChat(self.judge_model)
        failures: Dict[str, str] = {}
        if todo:
            print(f"  Auto-ARGUE: judging {len(todo)} report(s) ({len(judged)} cached) with {self.judge_model}")
            utils.get_text_from_id_fast.__dict__.get("cache", {}).pop(_DOC_COLLECTION, None)
            keys = {qid: key for qid, _, _, key in todo}
            with _patched(utils, judge, chat):
                fresh = asyncio.run(self._judge(judge, utils, [t[:3] for t in todo], out_dir))
            for qid, outcome in fresh.items():
                if isinstance(outcome, Exception):
                    failures[qid] = str(outcome)
                    logger.error(f"Auto-ARGUE failed for {qid}: {outcome}")
                    continue
                judged[qid] = outcome.model_dump(mode="json")
                (cache_dir / f"{qid}.json").write_text(
                    json.dumps({"key": keys[qid], "judged": judged[qid]}, ensure_ascii=False), encoding="utf-8")

        per_topic = self._score(score_module, judged, banks_dir, out_dir) if judged else {}

        per_query: List[Dict[str, Any]] = []
        for qid in sorted(banks):
            if qid in failures:
                continue
            record: Dict[str, Any] = {"query_id": qid, "empty_report": qid in empty,
                                      "unmapped_markers": unmapped[qid]}
            scores = per_topic.get(qid, {})
            for m in METRICS + COUNTS:
                record[m] = round(scores.get(m, 0.0), 5)
            if qid in empty:
                record["nuggets"] = len(banks[qid].nugget_bank)
            per_query.append(record)

        return {
            "num_evaluated": len(per_query),
            "num_without_nuggets": len([q for q in results if q not in banks]),
            "num_empty_reports": len(empty),
            "num_judge_failures": len(failures),
            "metrics": {m: mean_or_none([r[m] for r in per_query], 5) for m in METRICS},
            "judge_calls": chat.calls,
            "judge_malformed": chat.malformed,
            "per_query": per_query,
        }

    def summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """The ``generation.nuggets`` group of ``summary.json``."""
        return {
            "method": "auto_argue",
            **{k: metrics[k] for k in ("num_evaluated", "num_without_nuggets", "num_empty_reports",
                                       "num_judge_failures", "metrics")},
            "judge_model": self.judge_model,
            "auto_argue_commit": AUTO_ARGUE_COMMIT,
            "max_chars": self.max_chars,
        }

    def summary_lines(self, metrics: Dict[str, Any]) -> List[str]:
        """The terminal lines of the run summary for :meth:`evaluate`'s *metrics*."""
        def f(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        m = metrics["metrics"]
        return [
            f"  Nuggets (Auto-ARGUE, {self.judge_model}): {metrics['num_evaluated']} reports"
            f" (without nuggets {metrics['num_without_nuggets']},"
            f" empty {metrics['num_empty_reports']},"
            f" judge failures {metrics['num_judge_failures']})",
            f"    coverage {f(m.get('nugget_coverage'))} (weighted {f(m.get('nugget_coverage_weighted'))})"
            f" | sentence support {f(m.get('sentence_support'))}"
            f" | F1 {f(m.get('f1'))} (weighted {f(m.get('f1_weighted'))})",
            f"    citation support {f(m.get('citation_support'))}"
            f" | citation relevance {f(m.get('citation_relevance'))}",
        ]
