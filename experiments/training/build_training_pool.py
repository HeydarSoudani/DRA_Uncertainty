"""Build a training pool (``--pool``) under its dataset directory,
data/training_pool/{pool} of the repo.

Every pool has one row per query in ``queries/queries_all.jsonl``: the
question and answer, its source, its split (``sft``, ``rl`` or ``val``)
and its gold annotation (criteria with their gold answers, documents and
texts; ``training_pool/annotations.py``), and its gold documents in
``qrels/qrels_all.txt``.  The splits are views of that file
(``dataset_loaders``: ``--subset sft`` of dra_inference reads the rows of
split ``sft``).

trqa_validation: the TRQA wiki1 + wiki2 validation queries, intermediate
info and qrels merged into those two files (one call, no ``--step``;
``--limit N``: incremental, the next N queries not yet in the pool, wiki1
then wiki2; 0 = rewrite the files only); it searches TRQA's wiki_partial
corpus and indices; its gold annotation comes from the intermediate info,
with the criterion texts written by an LLM (``gold_criteria`` section).

deep_qa: deep-research questions from InfoSeek-Hard, WebShaper and FRAMES
with their gold pages fetched from the source, one step per call
(``--step N``):

    1  merge the sources, drop duplicates and the     questions.jsonl
       fixed question types (filters.py); each
       record has the skeleton of its gold
       annotation (annotations.py)
    2  fetch the gold pages (Wikipedia API, web,      pages/index.jsonl, pages/text.jsonl,
       Wayback fallback), Filter A (all gold pages,   queries/queries_all.jsonl,
       >= min_page_share with web gold pages), the    qrels/qrels_all.txt,
       gold annotation (criteria, answers, fetched    corpus/corpus.jsonl,
       documents, level; the criterion texts by an    criterion_texts.jsonl
       LLM, ``gold_criteria`` section), the split
       (full queries with the shares of the
       ``split`` section; the others: val)
    3  hard negatives of the Wikipedia gold pages     negatives/index.jsonl,
       (Wikipedia API: morelike, category siblings;   pages/text.jsonl,
       at most max_per_gold each; ``negatives``       corpus/corpus.jsonl,
       section); web gold pages get none; no          corpus/ids_gold.txt,
       negative duplicates a corpus document          corpus/ids_hard_negative.txt
       (containment >= dedup_threshold)
    4  analysis: stats of the dataset per source      (log tables),
       (queries, docs per query, lengths, hard        duplicates.jsonl
       negatives, gold annotation levels and
       texts, checks); duplicate documents of the
       corpus (same id, url or text, near-identical
       text; duplicates.py), report only

The corpus holds the gold pages and the hard negatives with their text;
``ids_gold.txt`` and ``ids_hard_negative.txt`` list the ids of each kind.
Every step writes its section of ``stats.json`` (steps 2 and 3 with their
settings, the split its shares and counts in ``split``); steps 1 and 2 write
their dropped questions to ``dropped.jsonl``.

Settings: ``--config`` (YAML, sections ``shared``, ``pages``, ``negatives``,
``split`` and ``gold_criteria``).  The contact of the User-Agent comes from ``DRA_CONTACT`` in
``.env``.  Steps 2 and 3 are incremental, like dra_inference.py: step 2
skips the questions already processed and ``--limit N`` takes the next N of
the others (``--source`` keeps the questions of the given sources only;
without it, step 1 order; ``--limit 0`` fetches nothing and rewrites the
dataset files, annotation and split); step 3 skips the gold pages with negatives and
``--limit N`` takes the next N gold pages (``--source``: gold pages of
questions of those sources; qrels order).  The dataset files cover every
processed question and gold page, and a query keeps its split when the pool
grows.  The criterion texts of an earlier run are kept; a run writes the
texts of the new queries and retries those whose writer call failed.  Step
2 and the trqa_validation build need the writer's API key
(``OPENROUTER_API_KEY``) when they have texts to write.  Other page
settings than the earlier runs' (``steps.PAGE_SETTINGS``),
other step 3 settings, or other split shares than the recorded ones are
refused.  One build of a pool runs at a time (a lock file in its directory).
"""

import argparse
import fcntl
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from indexing_corpus_dataset.training_pool import trqa_validation
from indexing_corpus_dataset.training_pool.config import POOLS, PoolPaths, load_config
from indexing_corpus_dataset.training_pool.negatives import run_step3
from indexing_corpus_dataset.training_pool.sources import LOADERS
from indexing_corpus_dataset.training_pool.steps import run_step1, run_step2, run_step4

_HERE = Path(__file__).resolve().parent
_CONFIG_DEFAULT = str(_HERE.parent / "configs" / "build_training_pool.yaml")


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the training pool (questions, gold pages, hard negatives).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--pool", type=str, default="deep_qa", choices=POOLS, help="Training pool to build.")
    parser.add_argument("--step", type=int, default=None, choices=[1, 2, 3, 4], help="deep_qa (required): 1 questions, 2 gold pages + Filter A + gold annotation with criterion texts + sft / rl / val split, 3 hard negatives, 4 dataset stats; trqa_validation: none (one build call).")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings, sections 'shared', 'pages', 'negatives', 'split' and 'gold_criteria'.")
    parser.add_argument("--limit", type=_non_negative_int, default=None, help="Step 2: the next N unprocessed questions (0 = rewrite the dataset files only); step 3: the next N >= 1 unprocessed wiki gold pages; trqa_validation: the next N queries not yet in the pool (0 = rewrite the files only) (None = all).")
    parser.add_argument("--source", type=str, default=None, help="Steps 2 and 3: only these sources, comma-separated (webshaper, frames, infoseek; None = all).")
    return parser.parse_args()


def _lock(root: Path):
    """An exclusive lock on ``{root}/.build.lock``, held until the process
    exits; exits when another build of the pool holds it."""
    root.mkdir(parents=True, exist_ok=True)
    fh = open(root / ".build.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another build of {root} is running (lock {root / '.build.lock'}); wait for it to finish")
    return fh


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # LiteLLM and its HTTP client log every call of the criterion texts' writer at INFO.
    for name in ("LiteLLM", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)
    args = _parse_args()
    if args.pool == "deep_qa" and args.step is None:
        raise SystemExit("--step is required for deep_qa")
    if args.pool == "trqa_validation" and args.step is not None:
        raise SystemExit("trqa_validation takes no --step (one build call)")
    if args.pool == "deep_qa" and args.step not in (2, 3) and args.limit is not None:
        raise SystemExit("--limit applies to steps 2 and 3 only")
    if args.step not in (2, 3) and args.source is not None:
        raise SystemExit("--source applies to steps 2 and 3 only")
    if args.step == 3 and args.limit == 0:
        raise SystemExit("--limit 0 applies to step 2 only (step 3: N >= 1)")
    sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else None
    unknown = sorted(set(sources or ()) - set(LOADERS))
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; expected some of {list(LOADERS)}")

    cfg = load_config(args.config)
    paths = PoolPaths(args.pool)
    lock = _lock(paths.root)  # noqa: F841 - held until exit

    if args.pool == "trqa_validation":
        print(f"[build_training_pool] trqa_validation, dataset {paths.root}")
        trqa_validation.build(paths, cfg.gold_criteria, cfg.split, args.limit)
        return
    print(f"[build_training_pool] deep_qa step {args.step}, dataset {paths.root}")

    if args.step == 1:
        run_step1(paths)
    elif args.step in (2, 3):
        (run_step2 if args.step == 2 else run_step3)(cfg, paths, args.limit, sources)
    else:
        run_step4(paths, cfg.negatives.dedup_threshold)


if __name__ == "__main__":
    main()

# python experiments/training/build_training_pool.py --pool trqa_validation
# python experiments/training/build_training_pool.py --pool trqa_validation --limit 10
# python experiments/training/build_training_pool.py --step 1
# python experiments/training/build_training_pool.py --step 2 --source webshaper --limit 10
# python experiments/training/build_training_pool.py --step 2 --source infoseek --limit 10
# python experiments/training/build_training_pool.py --step 2 --source frames --limit 10
# python experiments/training/build_training_pool.py --step 3 --limit 100
# python experiments/training/build_training_pool.py --step 4
