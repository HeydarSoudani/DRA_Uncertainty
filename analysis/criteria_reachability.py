"""Criteria reachability: are the gold documents reachable through the criteria list?

The agent searches for documents that establish its criteria, so a gold
document that gives information for no criterion is one the criteria do
not lead to.  Per request, an LLM judge reads every gold document of
``--grades`` in full against the request's criteria and labels every
criterion:

* ``support``: the document gives specific facts for the criterion,
  backed by a verbatim span; it is reachable through the criterion;
* ``related``: it is on the criterion's topic without specific facts; it
  is not reachable through the criterion;
* ``none``: it is not about the criterion; it is not reachable through
  the criterion.

Labels follow TREC: ``support`` as in TREC RAG nugget assignment,
``related`` as in the TREC Deep Learning relevance grades.  A document is
reachable when at least one criterion has ``support``; reachability is the
share of reachable gold documents.  The share with at least ``related`` is
a secondary diagnostic of topical closeness.  A ``support`` whose span is
not in the document is lowered to ``related``; a span joined with an
ellipsis, which the prompt forbids, still counts when every piece is in the
document.

Per request:

1. Criteria: from the criteria bank of the split
   (:func:`indexing_corpus_dataset.layout.criteria_bank_path`, shared with
   ``dra_inference``); a request missing from it gets its criteria from
   :class:`uncertainty_estimator.criteria.LLMCriteriaSource` with the
   dataset's init prompt and ``max_criteria``, and they are added to it.
   ``llm_criteria`` and ``max_criteria`` default to the run config
   (``--config``, ``dra_inference.yaml``), so the analysis reads the
   criteria the runs use.  A request whose extraction failed or was only
   salvaged (not banked) is left out, so every judgment is of banked
   criteria.
2. Gold documents: the qrel documents whose grade is in ``--grades``, from
   the corpus, whole: the qrels judge the whole document.
3. Judge: one call per (request, document) with all criteria and
   ``prompts/criteria_reachability_{system,user}.txt`` (temperature 0, reasoning
   off), asked once more when the reply cannot be parsed or leaves a
   criterion without a label.  The judge sees the request's topic title
   (``DatasetSpec.title_key``), not the request: the criteria already carry
   the request's requirements and limits, and the title only says what they
   are about.

Calls per request: one for the criteria when missing from the bank, plus
one per gold document.  Requests: those with a qrel of ``--grades``, in
query id order.  The judgments are cached and a request is done once every
gold document is judged with its banked criteria; a run does the requests
not done yet, ``--limit N`` the first N of them, so repeated runs with
``--limit`` go through the split N at a time.  The outputs cover every done
request.  A document missing from the corpus or whose judgment failed is
counted and left out of the rates.

Output in the run outputs, in
``OUTPUT_ROOT/criteria_reachability/{dataset}_{split}/``
(:func:`indexing_corpus_dataset.layout.criteria_reachability_dir`;
``--output`` overrides it):

- ``judgments.jsonl``: one line per (request, document, criteria, topic,
  judge, prompt) (cache, shared by every ``--grades`` and ``--tag``).
- ``grades-{g}_{judge}[_{tag}]/summary.json``: settings, counts,
  reachability (micro over documents, macro over requests, per grade) and
  the criterion stats.
- ``grades-{g}_{judge}[_{tag}]/per_doc.jsonl``: per (request, document),
  the grade, the document's length, the reach and every criterion's label,
  reason and span.

Usage::

    python analysis/criteria_reachability.py --dataset neuclir --limit 2
    python analysis/criteria_reachability.py --dataset ragtime --grades 2 3
    python analysis/criteria_reachability.py --dataset neuclir --max-criteria 10
"""

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# Repo root (utils) and src/ (component packages), as for experiments/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tqdm import tqdm

from evaluation.common import mean_or_none, read_jsonl, summary_stats, write_json, write_jsonl
from indexing_corpus_dataset.dataset_loaders import load_split
from indexing_corpus_dataset.doc_lookup import CorpusLookup
from indexing_corpus_dataset.layout import (
    DATA_ROOT, DATASET_SPECS, apply_dataset_defaults, criteria_bank_path, criteria_reachability_dir,
    default_corpus_path, resolve_split_id,
)
from reasoner_component.factory import create_generator, disable_native_thinking
from uncertainty_estimator.criteria import BankedCriteriaSource, LLMCriteriaSource
from uncertainty_estimator.judges import span_in_passage, split_span
from utils.cli_setup import load_run_config
from utils.text_utils import doc_text, doc_title, parse_json_object

logger = logging.getLogger(__name__)

REPORT_DATASETS = [name for name, spec in DATASET_SPECS.items() if spec.report_eval == "argue"]
CONFIG_DEFAULT = _REPO_ROOT / "experiments" / "configs" / "dra_inference.yaml"
JUDGE_MAX_TOKENS = 4096

PROMPTS = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "criteria_reachability_system.txt").read_text().strip()
USER_TEMPLATE = (PROMPTS / "criteria_reachability_user.txt").read_text().strip()
# Part of the judgment cache key: a changed prompt judges again.
PROMPT_HASH = hashlib.sha1((SYSTEM_PROMPT + USER_TEMPLATE).encode()).hexdigest()[:12]

LABELS = ("none", "related", "support")
RANK = {label: k for k, label in enumerate(LABELS)}
# The headline level first.
LEVELS = ("support", "related")


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def criteria_source(args, bank_path: Path) -> BankedCriteriaSource:
    """The criteria bank of the split, extracting with ``llm_criteria``."""
    return BankedCriteriaSource(
        LLMCriteriaSource(
            llm_client=disable_native_thinking(create_generator(args.llm_criteria, backend="api")),
            max_criteria=args.max_criteria,
            model_name=args.llm_criteria,
            query_shape=DATASET_SPECS[args.dataset].query_shape,
        ),
        bank_path,
    )


def bank_criteria(args, source: BankedCriteriaSource, queries: Dict[str, str]) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """``({qid: criteria}, {bank outcome: count})``, the criteria read from
    the bank or extracted and added to it; a salvaged extraction, which is
    not banked, gives no criteria."""
    def one(qid: str):
        criteria, info = source.get(qid, queries[qid])
        if not criteria:
            logger.error("Criteria extraction failed for %s: %s", qid, info.get("errors"))
        elif info["bank"] == "salvaged":
            logger.error("Criteria extraction salvaged (not banked) for %s, left out: %s", qid, info.get("errors"))
            criteria = []
        return qid, [c.to_dict() for c in criteria], info["bank"]

    criteria, outcomes = {}, {"hit": 0, "added": 0, "salvaged": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for qid, crit, outcome in tqdm(pool.map(one, queries), total=len(queries), desc="criteria (bank)"):
            criteria[qid] = crit
            outcomes[outcome] += 1
    return criteria, outcomes


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judgment_key(qid: str, doc_id: str, criteria: List[Dict], topic: str, judge: str) -> str:
    texts = json.dumps([c["text"] for c in criteria], ensure_ascii=False)
    return hashlib.sha1(f"{qid}\t{doc_id}\t{texts}\t{topic}\t{judge}\t{PROMPT_HASH}".encode()).hexdigest()


def render_passage(doc: Dict[str, str]) -> str:
    """The whole document: title and full text."""
    return f"{doc_title(doc)}\n{doc_text(doc, max_length=None)}"


def parse_labels(raw: str, criteria: List[Dict], passage: str) -> Optional[Dict[str, Dict]]:
    """``{criterion id: {"label", "reason", "span", "downgraded"}}`` with
    every criterion, or None when the reply has no ``criteria`` list or
    leaves a criterion without a valid label.  A ``support`` whose span is
    not in *passage* is lowered to ``related`` (``downgraded``)."""
    data = parse_json_object(raw or "")
    items = data.get("criteria") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    ids = {c["id"] for c in criteria}
    labels = {}
    for item in items:
        if not isinstance(item, dict) or item.get("label") not in LABELS:
            continue
        cid = str(item.get("id"))
        if cid not in ids or cid in labels:
            continue
        label, span = item["label"], str(item.get("span") or "")
        # A span joined with an ellipsis counts when every piece is in the passage.
        pieces = [span_in_passage(passage, p) for p in split_span(span)] if label == "support" else []
        downgraded = label == "support" and not (pieces and all(pieces))
        labels[cid] = {"label": "related" if downgraded else label, "reason": str(item.get("reason", "")),
                       "span": " ... ".join(pieces) if pieces and all(pieces) else span, "downgraded": downgraded}
    if len(labels) < len(ids):
        return None
    return {c["id"]: labels[c["id"]] for c in criteria}


def judge_one(client, topic: str, criteria: List[Dict], passage: str) -> Tuple[Optional[Dict], Optional[str]]:
    """``(labels, error)`` of one document; asked once more on a bad reply."""
    user = USER_TEMPLATE.format(
        topic=topic, criteria="\n".join(f"{c['id']}: {c['text']}" for c in criteria), passage=passage)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    error = None
    for _ in range(2):
        try:
            raw = client.complete(messages, max_tokens=JUDGE_MAX_TOKENS, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - recorded as the judgment's failure
            error = f"llm_call: {exc}"
            continue
        labels = parse_labels(raw, criteria, passage)
        if labels is not None:
            return labels, None
        error = f"parse: no 'criteria' list with a label for every criterion: {(raw or '')[:300]}"
    return None, error


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def reach(labels: Dict[str, Dict]) -> str:
    """The highest label of a document over the criteria."""
    return max((v["label"] for v in labels.values()), key=RANK.get, default="none")


def at_level(label: str, level: str) -> bool:
    return RANK[label] >= RANK[level]


def rate(labels: List[str], level: str) -> Optional[float]:
    return round(sum(at_level(x, level) for x in labels) / len(labels), 4) if labels else None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    spec = DATASET_SPECS[args.dataset]
    config = load_run_config(args.config)
    for field in ("llm_criteria", "max_criteria"):
        if getattr(args, field) is None:
            setattr(args, field, config[field])
    args.judge_model = args.judge_model or args.llm_criteria
    apply_dataset_defaults(args, ("dataset_year", "subset", "query_key", "max_criteria"))
    split = resolve_split_id(args.dataset, args.dataset_year, args.subset)
    data_path = Path(args.data_path or DATA_ROOT / args.dataset)
    corpus_path = Path(args.corpus_path or default_corpus_path(args.dataset, args.subset))
    bank_path = criteria_bank_path(args.dataset, split)
    grades = sorted(set(args.grades or [max(spec.relevance_gains)]))

    tag = args.tag
    if tag is None and args.max_criteria != spec.max_criteria:
        tag = f"maxc{args.max_criteria}"
    name = f"grades-{'-'.join(map(str, grades))}_{args.judge_model.rstrip('/').split('/')[-1]}" + (f"_{tag}" if tag else "")
    split_dir = Path(args.output or criteria_reachability_dir(args.dataset, split))
    out_dir = split_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    queries, qrels, _ = load_split(data_path, split, query_key=args.query_key,
                                   min_relevance_score=min(grades), only_with_qrels=True)
    topics, _, _ = load_split(data_path, split, query_key=spec.title_key)
    gold = {q: sorted((d, g) for d, g in qrels.get(q, {}).items() if g in grades) for q in queries}
    qids = sorted(q for q in queries if gold[q])
    print(f"{args.dataset} {split}: {len(qids)} requests with a gold document of grades {grades}")
    texts = CorpusLookup(corpus_path).get({d for q in qids for d, _ in gold[q]})
    cache_path = split_dir / "judgments.jsonl"
    cache = {r["key"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}

    # ── Done and to do ─────────────────────────────────────────────────────
    # Done: criteria in the bank and every gold document in the corpus
    # judged with them.  --limit caps the requests to do in this run.
    source = criteria_source(args, bank_path)
    criteria: Dict[str, List[Dict]] = {}
    todo = []
    for q in qids:
        banked = source.lookup(q, queries[q])
        crit = [c.to_dict() for c in banked] if banked else None
        if crit and all(judgment_key(q, d, crit, topics[q], args.judge_model) in cache
                        for d, _ in gold[q] if d in texts):
            criteria[q] = crit
        else:
            todo.append(q)
    pending = len(todo)
    if args.limit:
        todo = todo[:args.limit]
    pending -= len(todo)
    print(f"Done {len(criteria)}, to do in this run {len(todo)}, left for later {pending}")

    # ── Criteria ───────────────────────────────────────────────────────────
    todo_criteria, bank = bank_criteria(args, source, {q: queries[q] for q in todo})
    print(f"Criteria bank {bank_path}: {bank['hit']} read, {bank['added']} added, "
          f"{bank['salvaged']} salvaged (not added), {bank['failed']} failed")
    no_criteria = [q for q in todo if not todo_criteria.get(q)]
    if no_criteria:
        logger.warning("%d request(s) without criteria, left out: %s", len(no_criteria), no_criteria)
    criteria.update({q: c for q, c in todo_criteria.items() if c})
    todo = [q for q in todo if q in criteria]
    qids = [q for q in qids if q in criteria]

    # ── Gold documents ─────────────────────────────────────────────────────
    missing = sum(1 for q in qids for d, _ in gold[q] if d not in texts)
    if missing:
        logger.warning("%d gold document(s) not in the corpus %s, left out", missing, corpus_path)

    # ── Judge ──────────────────────────────────────────────────────────────
    tasks = {}
    for q in todo:
        for d, _ in gold[q]:
            if d in texts:
                key = judgment_key(q, d, criteria[q], topics[q], args.judge_model)
                if key not in cache:
                    tasks[key] = (q, d)
    errors: Dict[Tuple[str, str], str] = {}
    if tasks:
        client = disable_native_thinking(create_generator(args.judge_model, backend="api"))

        def one(item):
            key, (q, d) = item
            labels, error = judge_one(client, topics[q], criteria[q], render_passage(texts[d]))
            return key, q, d, labels, error

        with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key, q, d, labels, error in tqdm(pool.map(one, tasks.items()), total=len(tasks), desc="judging"):
                if labels is None:
                    errors[(q, d)] = error
                    logger.error("Judgment failed for %s / %s: %s", q, d, error)
                    continue
                rec = {"key": key, "query_id": q, "doc_id": d, "labels": labels}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                cache[key] = rec
    else:
        print("All judgments cached; no LLM call")

    # ── Per request ────────────────────────────────────────────────────────
    records, all_docs = [], []
    downgraded = 0
    for q in qids:
        crit = criteria[q]
        docs = []
        for d, g in gold[q]:
            rec = cache.get(judgment_key(q, d, crit, topics[q], args.judge_model))
            if d not in texts or rec is None:
                continue
            downgraded += sum(v["downgraded"] for v in rec["labels"].values())
            docs.append({"query_id": q, "doc_id": d, "grade": g,
                         "doc_chars": len(doc_text(texts[d], max_length=None)),
                         "reach": reach(rec["labels"]), "labels": rec["labels"]})
        all_docs.extend(docs)
        load = {c["id"]: {lvl: [x["doc_id"] for x in docs if at_level(x["labels"][c["id"]]["label"], lvl)]
                          for lvl in LEVELS}
                for c in crit}
        reaches = [x["reach"] for x in docs]
        records.append({
            "query_id": q,
            "criteria": [{**c, "docs_support": load[c["id"]]["support"], "docs_related": load[c["id"]]["related"]}
                         for c in crit],
            "num_gold": len(gold[q]),
            "num_judged": len(docs),
            "reachability_support": rate(reaches, "support"),
            "reachability_related": rate(reaches, "related"),
        })
    write_jsonl(out_dir / "per_doc.jsonl", all_docs)

    # ── Summary ────────────────────────────────────────────────────────────
    all_crit = [c for r in records for c in r["criteria"]]
    metrics = {}
    for lvl in LEVELS:
        metrics[f"reachability_{lvl}"] = {
            "micro": rate([x["reach"] for x in all_docs], lvl),
            "macro": mean_or_none([r[f"reachability_{lvl}"] for r in records if r["num_judged"]], 4),
            "per_grade": {g: rate([x["reach"] for x in all_docs if x["grade"] == g], lvl) for g in grades},
        }
    summary = {
        "settings": {
            "dataset": args.dataset, "split": split, "grades": grades, "config": args.config,
            "llm_criteria": args.llm_criteria, "max_criteria": args.max_criteria, "criteria_bank": str(bank_path),
            "judge_model": args.judge_model, "judge_prompt": PROMPT_HASH,
            "topic_key": spec.title_key, "passage": "full document",
        },
        "counts": {
            "requests": len(records),
            "requests_left_for_later": pending,
            "requests_without_criteria": len(no_criteria),
            "this_run": {
                "requests_done": len(todo),
                "criteria_from_bank": bank["hit"],
                "criteria_added_to_bank": bank["added"],
                "criteria_salvaged": bank["salvaged"],
                "criteria_failed": bank["failed"],
            },
            "criteria": len(all_crit),
            "criteria_per_request": mean_or_none([len(r["criteria"]) for r in records], 2),
            "gold_docs": sum(r["num_gold"] for r in records),
            "gold_docs_per_grade": {g: sum(1 for r in records for d, gg in gold[r["query_id"]] if gg == g) for g in grades},
            "judged_docs": len(all_docs),
            "missing_in_corpus": missing,
            "failed_judgments": len(errors),
            "span_downgrades": downgraded,
        },
        "metrics": metrics,
        "stats": {
            "criterion_support": {lvl: mean_or_none([float(bool(c[f"docs_{lvl}"])) for c in all_crit], 4)
                                  for lvl in LEVELS},
            "docs_support_per_criterion": summary_stats([len(c["docs_support"]) for c in all_crit]),
        },
    }
    write_json(out_dir / "summary.json", summary)

    c, m = summary["counts"], summary["metrics"]
    print(f"\nRequests {c['requests']}, criteria {c['criteria']} ({c['criteria_per_request']} per request), "
          f"gold docs {c['gold_docs']} judged {c['judged_docs']} "
          f"(missing {c['missing_in_corpus']}, failed {c['failed_judgments']}, span downgrades {c['span_downgrades']})")
    for lvl in LEVELS:
        r = m[f"reachability_{lvl}"]
        print(f"  reachability ({lvl:<8}) micro {r['micro']}  macro {r['macro']}  per grade {r['per_grade']}")
    print(f"  criteria supported: {summary['stats']['criterion_support']}")
    print(f"Saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI arguments.

    A default of None is resolved at run time: the dataset fields from
    ``layout.DATASET_SPECS``, ``llm_criteria`` and ``max_criteria`` from the
    run config (``--config``), so the analysis reads the criteria the runs
    use.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="neuclir", choices=REPORT_DATASETS, help="Report dataset.")
    parser.add_argument("--subset", type=str, default=None, help="Subset (None = the dataset's default). neuclir: news|technical; ragtime: unused.")
    parser.add_argument("--dataset-year", type=str, default=None, help="Split year (None = the dataset's default).")
    parser.add_argument("--query-key", type=str, default=None, help="Key of the query text in the query records (None = the dataset's default).")
    parser.add_argument("--grades", type=int, nargs="+", default=None, help="Qrel grades of the gold documents (None = the top grade). neuclir: 1 3; ragtime: 2 3.")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset folder (None = DATA_ROOT/{dataset}).")
    parser.add_argument("--corpus-path", type=str, default=None, help="Corpus JSONL (None = the dataset's corpus under DATA_ROOT).")

    # ── Criteria and judge ──────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=str(CONFIG_DEFAULT), help="Run config that llm_criteria and max_criteria are read from.")
    parser.add_argument("--llm-criteria", type=str, default=None, help="Criteria-extraction LLM (None = the config's llm_criteria).")
    parser.add_argument("--max-criteria", type=int, default=None, help="Criteria the extractor may add (None = the config's max_criteria, else the dataset's).")
    parser.add_argument("--judge-model", type=str, default=None, help="Reachability judge LLM (None = the criteria-extraction LLM).")

    # ── Run control ─────────────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="Requests not done yet to do in this run (None = all).")
    parser.add_argument("--workers", type=int, default=16, help="Parallel LLM calls.")
    parser.add_argument("--tag", type=str, default=None, help="Run folder suffix (None = maxc{N} when --max-criteria differs from the dataset's, else none).")
    parser.add_argument("--output", type=str, default=None, help="Output folder (None = OUTPUT_ROOT/criteria_reachability/{dataset}_{split}).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#   python analysis/criteria_reachability.py --dataset neuclir --limit 1
#   python analysis/criteria_reachability.py --dataset neuclir
#   python analysis/criteria_reachability.py --dataset neuclir --grades 1 3 --limit 1
#   python analysis/criteria_reachability.py --dataset ragtime --grades 2 3 --limit 5
#   python analysis/criteria_reachability.py --dataset ragtime --grades 2 3 --workers 32
#   python analysis/criteria_reachability.py --dataset neuclir --max-criteria 10
#   python analysis/criteria_reachability.py --dataset neuclir --judge-model openrouter/qwen/qwen3-32b --tag qwen3-32b