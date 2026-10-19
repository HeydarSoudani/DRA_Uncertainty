"""Build a training pool (``--pool``) under its dataset directory,
data/training_pool/{pool} of the repo.

trqa_validation: the TRQA wiki1 + wiki2 validation queries, intermediate
info and qrels merged into split ``all``, then split into ``sft``, ``rl``
and ``val`` (one call, no ``--step``); it searches TRQA's wiki_partial
corpus and indices.

deep_qa: deep-research questions from InfoSeek-Hard, WebShaper and FRAMES
with their gold pages fetched from the source, one step per call
(``--step N``):

    1  merge the sources, drop duplicates and the     questions.jsonl
       fixed question types (filters.py)
    2  fetch the gold pages (Wikipedia API, web,      pages/index.jsonl, pages/text.jsonl,
       Wayback fallback), Filter A (all gold pages,   queries/queries_all.jsonl,
       >= min_page_share with web gold pages), then   qrels/qrels_all.txt,
       split ``all`` into sft / rl / val by a hash    corpus/corpus.jsonl,
       of the query id (shares: ``split`` section)    queries/queries_{sft,rl,val}.jsonl,
                                                      qrels/qrels_{sft,rl,val}.txt
    3  analysis: stats of the dataset per source      (log table)
       (queries, docs per query, lengths, checks)

Every step writes its section of ``stats.json`` (step 2 with its settings,
the split its shares and counts in ``split``); steps 1 and 2 write their
dropped questions to ``dropped.jsonl``.

Settings: ``--config`` (YAML, sections ``shared``, ``pages`` and ``split``).
The contact of the User-Agent comes from ``DRA_CONTACT`` in ``.env``.  Step 2
is incremental, like dra_inference.py: it skips the questions already
processed and ``--limit N`` takes the next N of the others (``--source``
keeps the questions of the given sources only; without it, step 1 order);
the dataset files cover every processed question, and a query keeps its
split when the pool grows.  Other page settings than the earlier runs'
(``steps.PAGE_SETTINGS``) or other split shares than the recorded ones are
refused.  One build of a pool runs at a time (a lock
file in its directory).
"""

import argparse
import fcntl
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from indexing_corpus_dataset.training_pool import trqa_validation
from indexing_corpus_dataset.training_pool.splits import write_splits
from indexing_corpus_dataset.training_pool.config import POOLS, PoolPaths, load_config
from indexing_corpus_dataset.training_pool.steps import run_step1, run_step2, run_step3

_HERE = Path(__file__).resolve().parent
_CONFIG_DEFAULT = str(_HERE.parent / "configs" / "build_training_pool.yaml")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the training pool (questions + gold pages).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--pool", type=str, default="deep_qa", choices=POOLS, help="Training pool to build.")
    parser.add_argument("--step", type=int, default=None, choices=[1, 2, 3], help="deep_qa only (required): 1 questions, 2 gold pages + Filter A + sft / rl / val split, 3 dataset stats.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings, sections 'shared', 'pages' and 'split'.")
    parser.add_argument("--limit", type=_positive_int, default=None, help="Step 2: the next N unprocessed questions (None = all).")
    parser.add_argument("--source", type=str, default=None, help="Step 2: only these sources, comma-separated (webshaper, frames, infoseek; None = all).")
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
    args = _parse_args()
    if (args.pool == "deep_qa") != (args.step is not None):
        raise SystemExit("--step is required for deep_qa and not used for trqa_validation")
    if args.step != 2 and (args.limit is not None or args.source is not None):
        raise SystemExit("--limit and --source apply to step 2 only")

    cfg = load_config(args.config)
    paths = PoolPaths(args.pool)
    lock = _lock(paths.root)  # noqa: F841 - held until exit

    if args.pool == "trqa_validation":
        print(f"[build_training_pool] trqa_validation, dataset {paths.root}")
        trqa_validation.build(paths)
        write_splits(paths, cfg.split)
        return
    print(f"[build_training_pool] deep_qa step {args.step}, dataset {paths.root}")

    if args.step == 1:
        run_step1(paths)
    elif args.step == 2:
        sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else None
        run_step2(cfg, paths, args.limit, sources)
    else:
        run_step3(paths)


if __name__ == "__main__":
    main()

# python experiments/training/build_training_pool.py --pool trqa_validation
# python experiments/training/build_training_pool.py --step 1
# python experiments/training/build_training_pool.py --step 2 --limit 10
# python experiments/training/build_training_pool.py --step 2 --source infoseek --limit 10
# python experiments/training/build_training_pool.py --step 2 --source frames --limit 10
# python experiments/training/build_training_pool.py --step 3
