"""Derive (or read) the criteria of a dataset split and score them.

    python -m evaluation.criteria --dataset trqa --subset wiki2 --sample 100
    python -m evaluation.criteria --dataset neuclir --prompt-file old_report_prompt.txt --tag old
    python -m evaluation.criteria --dataset browsecomp_plus --prompt-file old_prompt.txt --tag ref   # no gold: extract only
    python -m evaluation.criteria --dataset browsecomp_plus --reference .../ref/criteria.jsonl
    python -m evaluation.criteria --dataset ragtime --run-dir RUN_DIR

The queries are the ones a run would use (the dataset's query key, and only
queries with a qrel at ``min_relevance_score``; ``--all-queries`` drops the
qrels filter) that have gold units.  Unset dataset arguments take the
dataset's defaults (``layout.DATASET_SPECS``).

Outputs, in ``--output-dir`` (default
``{DRA_OUTPUT_ROOT}/criteria_eval/{dataset}_{split}/{tag}``):

* ``criteria.jsonl``: one ``{"query_id", "question", "criteria",
  "criteria_info"}`` per query.  Reused on a rerun: only the missing queries
  are extracted (``--overwrite`` starts over).
* ``eval.jsonl``: the per-query scores and match details.
* ``summary.json``: the settings and the macro-averaged scores.
* ``criteria_eval_judgments.jsonl``: the cached YES/NO nugget judgments
  (report datasets), reused on a rerun.
"""

import argparse
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from tqdm import tqdm

from indexing_corpus_dataset.dataset_loaders import load_split
from indexing_corpus_dataset.layout import DATA_ROOT, DATASET_SPECS, OUTPUT_ROOT, apply_dataset_defaults, resolve_split_id
from reasoner_component.factory import create_generator, disable_native_thinking
from uncertainty_estimator.criteria import LLMCriteriaSource

from ..common import read_jsonl, write_json, write_jsonl
from ..gold import load_gold_units
from ..judge import DEFAULT_JUDGE_MODEL
from ..uncertainty import load_uncertainty_meta
from .evaluator import CRITERIA_JUDGMENTS_FILE, build_criteria_evaluator

logger = logging.getLogger(__name__)

DEFAULT_CRITERIA_MODEL = "openrouter/qwen/qwen3.6-27b"


def _none_if_null(value: str):
    return None if value in ("", "null", "None") else value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    p.add_argument("--subset", type=_none_if_null, default=None, help="null = the dataset's default")
    p.add_argument("--dataset-year", type=_none_if_null, default=None, help="null = the dataset's default")
    p.add_argument("--data-path", default=None, help="default: DATA_ROOT/{dataset}")
    p.add_argument("--query-key", default=None, help="default: the dataset's query key")
    p.add_argument("--min-relevance-score", type=int, default=None, help="default: the dataset's threshold")
    p.add_argument("--all-queries", action="store_true", help="do not require a qrel at min_relevance_score")
    p.add_argument("--limit", type=int, default=None, help="score only the first N queries")
    p.add_argument("--sample", type=int, default=None, help="score N queries drawn at random (seed 0)")
    p.add_argument("--llm-criteria", default=DEFAULT_CRITERIA_MODEL, help="criteria-extraction LLM")
    p.add_argument("--max-criteria", type=int, default=None, help="default: the dataset's max_criteria")
    p.add_argument("--prompt-file", default=None, help="replace the criteria-extraction system prompt")
    p.add_argument("--run-dir", default=None, help="score the criteria of this run (its uncertainty/ files) instead of extracting")
    p.add_argument("--reference", default=None, help="score against this criteria list (criteria.jsonl or uncertainty/ dir) instead of the dataset's gold")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="LLM of the matchers")
    p.add_argument("--no-llm-match", action="store_true", help="string matching only (no judge calls)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tag", default=None, help="output subdirectory name")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--overwrite", action="store_true", help="re-extract every query")
    return p.parse_args()


def _extract(args, queries: Dict[str, str], path: Path) -> Dict[str, Dict]:
    """Criteria of every query, extracting the ones missing from *path*."""
    if args.overwrite and path.exists():
        path.unlink()
    done = {str(r["query_id"]): r for r in read_jsonl(path)}
    todo = [q for q in queries if q not in done]
    if todo:
        system_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else None
        source = LLMCriteriaSource(
            llm_client=disable_native_thinking(create_generator(args.llm_criteria, backend="api")),
            max_criteria=args.max_criteria,
            model_name=args.llm_criteria,
            query_shape=DATASET_SPECS[args.dataset].query_shape,
            system_prompt=system_prompt,
        )

        def one(qid: str) -> Dict:
            criteria, info = source.get(qid, queries[qid])
            return {"query_id": qid, "question": queries[qid],
                    "criteria": [c.to_dict() for c in criteria], "criteria_info": info}

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in tqdm(pool.map(one, todo), total=len(todo), desc="criteria extraction"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["query_id"]] = rec
    return {q: done[q] for q in queries if q in done}


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    apply_dataset_defaults(args, ("dataset_year", "subset", "query_key", "min_relevance_score", "max_criteria"))
    split = resolve_split_id(args.dataset, args.dataset_year, args.subset)
    data_path = Path(args.data_path or DATA_ROOT / args.dataset)

    queries, _, _ = load_split(
        data_path, split, query_key=args.query_key,
        min_relevance_score=args.min_relevance_score, only_with_qrels=not args.all_queries,
    )
    # A dataset without gold and no reference: extract the criteria only.
    gold = None
    if args.reference is not None or spec.criteria_gold is not None:
        gold = load_gold_units(args.dataset, data_path, split, reference=args.reference)
        queries = {q: t for q, t in queries.items() if gold.get(q)}
    if args.limit:
        queries = dict(list(queries.items())[:args.limit])
    if args.sample and args.sample < len(queries):
        keep = set(random.Random(0).sample(sorted(queries), args.sample))
        queries = {q: t for q, t in queries.items() if q in keep}
    print(f"{args.dataset} {split}: {len(queries)} queries"
          + (" with gold units" if gold is not None else " (no gold: extraction only)"))

    if args.tag is None:
        if args.run_dir:
            args.tag = f"run_{Path(args.run_dir).resolve().name}"
        else:
            prompt = Path(args.prompt_file).stem if args.prompt_file else "default"
            args.tag = f"{prompt}_{args.llm_criteria.split('/')[-1]}"
    out_dir = Path(args.output_dir or OUTPUT_ROOT / "criteria_eval" / f"{args.dataset}_{split}" / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        records = {q: {"criteria": m.get("criteria") or [], "criteria_info": m.get("criteria_info") or {}}
                   for q, m in load_uncertainty_meta(args.run_dir).items() if q in queries}
    else:
        records = _extract(args, queries, out_dir / "criteria.jsonl")

    settings = {
        "dataset": args.dataset, "split": split, "query_key": args.query_key,
        "all_queries": args.all_queries, "limit": args.limit, "sample": args.sample,
        "criteria_source": f"run:{args.run_dir}" if args.run_dir else args.llm_criteria,
        "prompt_file": args.prompt_file, "max_criteria": args.max_criteria,
        "reference": args.reference, "judge_model": None if args.no_llm_match or gold is None else args.judge_model,
    }
    if gold is None:
        counts = [len(r["criteria"]) for r in records.values()]
        summary = {"settings": settings, "num_queries": len(records),
                   "mean_num_criteria": round(sum(counts) / len(counts), 4) if counts else None}
        write_json(out_dir / "summary.json", summary)
        print(f"Saved criteria (not scored) to {out_dir}")
        return

    evaluator = build_criteria_evaluator(
        args.dataset, gold, args.judge_model, max_workers=args.workers,
        mode="info" if args.reference is not None else None, use_llm=not args.no_llm_match,
    )
    evaluator.use_judgment_cache(out_dir / CRITERIA_JUDGMENTS_FILE)
    result = evaluator.evaluate(
        {q: queries[q] for q in records},
        {q: r["criteria"] for q, r in records.items()},
        infos={q: r.get("criteria_info") or {} for q, r in records.items()},
    )

    write_jsonl(out_dir / "eval.jsonl", result["per_query"])
    write_json(out_dir / "summary.json", {"settings": settings, **result["summary"]})
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
