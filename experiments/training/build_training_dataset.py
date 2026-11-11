"""Build a training dataset (``--dataset``) under its folder,
data/training_datasets/{dataset} of the repo, one call per build.

Every training dataset has one row per query in ``queries/queries_all.jsonl``:
the question and answer, its source, its split (``sft``, ``rl`` or ``val``)
and its gold annotation (criteria with their gold answers, documents and
texts; ``training_datasets/gold/annotations.py``), and its gold documents in
``qrels/qrels_all.txt``.  The splits are views of that file
(``dataset_loaders``: ``--subset sft`` of dra_inference reads the rows of
split ``sft``).

infoseek: InfoSeek-Hard questions over the corpus of the original release,
``wiki-25-512.jsonl`` (Wikipedia 2025 in chunks of 512 tokens), downloaded
once into DATA_ROOT/infoseek/corpus with its indices next to it; the qrels
are the first chunk of each gold page (matched by page title), where
InfoSeek's claims are written from; a question
with a gold page missing from the corpus is dropped
(``training_datasets/infoseek.py``).  Split sft / rl / val with the shares
of the ``split`` section.

webshaper_frames: WebShaper and FRAMES questions with their gold pages
fetched from the source (Wikipedia API, web, Wayback fallback), Filter A,
the hard negatives of the Wikipedia gold pages, and a report of the dataset
and the corpus duplicates (``training_datasets/webshaper_frames/``,
settings ``pages`` and ``negatives``; the User-Agent contact from
``DRA_CONTACT`` in ``.env``).  Validation only: every query in ``val``.

trqa_validation: the TRQA wiki1 + wiki2 validation files merged
(``training_datasets/trqa_validation.py``), searching TRQA's corpus.  Split
sft / rl / val with the shares of the ``split`` section.

Every build is incremental, like dra_inference.py: ``--limit N`` processes
the next N questions not processed yet (0: none, the dataset files are
rewritten; none: all of them), ``--source`` (webshaper_frames only) keeps
the questions of the given sources.  The dataset files cover every
processed question, and a query keeps its split as the dataset grows.  The
criterion texts of earlier builds are kept; a build writes the texts of the
new queries and retries those whose writer call failed, with the writer's
API key (``OPENROUTER_API_KEY``) when it has texts to write.  Other page or
negatives settings than the earlier builds', or other split shares than the
recorded ones, are refused.  One build of a dataset runs at a time (a lock
file in its folder).

Settings: ``--config`` (YAML, sections ``shared``, ``pages``, ``negatives``,
``split`` and ``gold_criteria``).
"""

import argparse
import fcntl
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from indexing_corpus_dataset.layout import BUILT_DATASETS
from indexing_corpus_dataset.training_datasets.config import DatasetPaths, load_config
from indexing_corpus_dataset.training_datasets.sources import LOADERS

_HERE = Path(__file__).resolve().parent
_CONFIG_DEFAULT = str(_HERE.parent / "configs" / "build_training_dataset.yaml")


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training dataset (questions, gold annotation, qrels, split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(BUILT_DATASETS), help="Training dataset to build.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings, sections 'shared', 'pages', 'negatives', 'split' and 'gold_criteria'.")
    parser.add_argument("--limit", type=_non_negative_int, default=None, help="The next N questions not processed yet (0 = rewrite the dataset files only; None = all).")
    parser.add_argument("--source", type=str, default=None, help="webshaper_frames only: these sources, comma-separated (webshaper, frames; None = both).")
    return parser.parse_args()


def _lock(root: Path):
    """An exclusive lock on ``{root}/.build.lock``, held until the process
    exits; exits when another build of the dataset holds it."""
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
    if args.source is not None and args.dataset != "webshaper_frames":
        raise SystemExit("--source applies to webshaper_frames only")
    sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else None
    unknown = sorted(set(sources or ()) - set(LOADERS))
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; expected some of {list(LOADERS)}")

    cfg = load_config(args.config)
    paths = DatasetPaths(args.dataset)
    lock = _lock(paths.root)  # noqa: F841 - held until exit
    print(f"[build_training_dataset] {args.dataset}, dataset {paths.root}")

    # Each dataset's build imported alone: webshaper_frames needs the web
    # page dependencies (trafilatura), the others do not.
    if args.dataset == "infoseek":
        from indexing_corpus_dataset.training_datasets import infoseek
        infoseek.build(cfg, paths, args.limit)
    elif args.dataset == "webshaper_frames":
        from indexing_corpus_dataset.training_datasets import webshaper_frames
        webshaper_frames.build(cfg, paths, args.limit, sources)
    else:
        from indexing_corpus_dataset.training_datasets import trqa_validation
        trqa_validation.build(paths, cfg.gold_criteria, cfg.split, args.limit)


if __name__ == "__main__":
    main()

# python experiments/training/build_training_dataset.py --dataset infoseek --limit 100
# python experiments/training/build_training_dataset.py --dataset webshaper_frames --limit 100
# python experiments/training/build_training_dataset.py --dataset webshaper_frames --source frames --limit 50
# python experiments/training/build_training_dataset.py --dataset trqa_validation --limit 100
