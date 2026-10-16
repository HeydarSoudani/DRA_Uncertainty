"""Build the training pool: deep-research questions from InfoSeek-Hard,
WebShaper and FRAMES with their gold pages fetched from the source.

One step per call (``--step N``) under the dataset directory
(``shared.output_dir``, default data/training_pool of the repo):

    1  merge the sources, drop duplicates and the     questions.jsonl
       fixed question types (filters.py)
    2  fetch the gold pages (Wikipedia API, web,      pages/index.jsonl, pages/text.jsonl,
       Wayback fallback), Filter A (all gold pages,   queries/queries_train.jsonl,
       >= min_page_share with web gold pages)         qrels/qrels_train.txt,
                                                      corpus/corpus.jsonl
    3  analysis: stats of the dataset per source      (log table)
       (queries, docs per query, lengths, checks)

Every step writes its section of ``stats.json`` (step 2 with its settings);
steps 1 and 2 write their dropped questions to ``dropped.jsonl``.

Settings: ``--config`` (YAML, sections ``shared`` and ``pages``), then the
flags below.  The contact of the User-Agent comes from ``DRA_CONTACT`` in
``.env``.  Step 2 is incremental, like dra_inference.py: it skips the
questions already processed and ``--limit N`` takes the next N of the others
(``--source`` keeps the questions of the given sources only; without it, step
1 order); the dataset files cover every processed question.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from indexing_corpus_dataset.training_pool.config import PoolPaths, load_config
from indexing_corpus_dataset.training_pool.steps import run_step1, run_step2, run_step3

_HERE = Path(__file__).resolve().parent
_CONFIG_DEFAULT = str(_HERE / "configs" / "build_training_pool.yaml")

# CLI flag (dest) -> config key.
_FLAGS = {
    "output_dir": "shared.output_dir",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the training pool (questions + gold pages).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3], help="1 questions, 2 gold pages + Filter A, 3 dataset stats.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings, sections 'shared' and 'pages'.")
    parser.add_argument("--output-dir", type=str, default=None, help="Dataset directory (None = data/training_pool of the repo).")
    parser.add_argument("--limit", type=int, default=None, help="Step 2: the next N unprocessed questions (None = all).")
    parser.add_argument("--source", type=str, default=None, help="Step 2: only these sources, comma-separated (webshaper, frames, infoseek; None = all).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if args.step != 2 and (args.limit is not None or args.source is not None):
        raise SystemExit("--limit and --source apply to step 2 only")

    # Only the flags given on the command line override the --config file.
    overrides = {key: getattr(args, dest) for dest, key in _FLAGS.items() if getattr(args, dest) is not None}
    cfg = load_config(args.config, overrides)
    paths = PoolPaths(cfg)

    paths.root.mkdir(parents=True, exist_ok=True)
    print(f"[build_training_pool] step {args.step}, dataset {paths.root}")

    if args.step == 1:
        run_step1(paths)
    elif args.step == 2:
        sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else None
        run_step2(cfg, paths, args.limit, sources)
    else:
        run_step3(paths)


if __name__ == "__main__":
    main()

# python experiments/build_training_pool.py --step 1
# python experiments/build_training_pool.py --step 2 --limit 10
# python experiments/build_training_pool.py --step 2 --source infoseek --limit 10
# python experiments/build_training_pool.py --step 2 --source frames --limit 10
# python experiments/build_training_pool.py --step 3
