"""File-backed configuration for index building.

The frequently-varied knobs (``--retriever`` and ``--dataset``) stay on
the CLI; everything else lives in a YAML config file and is merged onto the
parsed args.  This mirrors the inference pipeline's config pattern
(:mod:`utils.cli_setup` + ``experiments/configs/dra_inference.yaml``) but is scoped to
the index-building entry points so it stays import-light.

Two entry points share this module:
  - ``index_builder.py``       uses :data:`BUILD_DEFAULTS` + ``index_build.yaml``
  - ``index_builder_test.py``  uses :data:`TEST_DEFAULTS`  + ``index_build_test.yaml``

Precedence (lowest -> highest):
    *_DEFAULTS  <  YAML config file  <  explicit CLI override (--key value)

Any config key can be overridden ad-hoc on the command line without being a
declared argparse flag, via :func:`utils.cli_setup.parse_cli_overrides`.
"""

from pathlib import Path

import yaml

# Reuse the generic (config-dict-driven) merge helpers from the inference CLI.
from utils.cli_setup import parse_cli_overrides, apply_config_to_args
from indexing_corpus_dataset.layout import apply_dataset_defaults, doc_max_length

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Default config file paths for each entry point.
BUILD_CONFIG_DEFAULT = str(_CONFIG_DIR / "index_build.yaml")
TEST_CONFIG_DEFAULT = str(_CONFIG_DIR / "index_build_test.yaml")


# ---------------------------------------------------------------------------
# Built-in fallbacks (used only when a key is absent from the YAML file, so
# downstream ``args.<attr>`` access never raises).  Keep in sync with the
# shipped YAML files under configs/.
# ---------------------------------------------------------------------------

# Production index builder (index_builder.py).
BUILD_DEFAULTS = {
    # Dataset selection (--retriever and --dataset stay on the CLI).
    "dataset_year": None,        # null = per dataset (layout.DATASET_SPECS)
    "subset": None,              # null = per dataset (layout.DATASET_SPECS)
    "trqa_partial": True,        # use wiki_partial vs full wiki corpus (trqa only)
    # Path overrides (null = auto-derive from dataset/subset).
    "corpus_path": None,
    "save_dir": None,
    "index_path": None,
    "embedding_path": None,
    # Index-build knobs.
    "max_length": None,          # null = per dataset + retriever (layout.doc_max_length)
    "batch_size": 16,
    "length_sorted": True,       # encode in length-sorted batches (same embeddings, less padding)
    "faiss_type": "Flat",
    "save_embedding": True,
    "use_fp16": True,
    "faiss_gpu": False,
    "device_id": None,
}

# Index builder test (index_builder_test.py): build keys + test-only extras.
TEST_DEFAULTS = {
    **BUILD_DEFAULTS,
    "data_path": None,            # root for queries/qrels (local or s3://); null = canonical
    "query_key": None,            # JSONL key for query text; null = per dataset
    "min_relevance_score": None,  # minimum qrel score treated as relevant; null = per dataset
    "query_limit": 10,            # max queries to use
    "sub_corpus_max_size": 3000,  # gold + distractors up to this many; null = gold only
}


def load_index_config(config_path, defaults: dict) -> dict:
    """Load a YAML index-build config, overlaid on ``defaults``.

    Built-in ``defaults`` overlaid with whatever the YAML file provides; unknown
    keys in the file are ignored with a warning.  A missing file falls back to
    ``defaults`` so the scripts still run with no config present.
    """
    merged = dict(defaults)
    if not config_path:
        return merged
    path = Path(config_path)
    if not path.exists():
        print(f"WARNING: config file not found: {config_path} — using built-in defaults")
        return merged
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - set(defaults)
    if unknown:
        print(f"WARNING: ignoring unknown keys in {config_path}: {sorted(unknown)}")
    merged.update({k: v for k, v in data.items() if k in defaults})
    return merged


def resolve_index_config(args, defaults: dict, extras: list) -> None:
    """Merge the YAML config (+ CLI overrides) onto the parsed ``args`` namespace.

    ``args.config`` selects the YAML file; ``extras`` are the ``parse_known_args``
    leftovers (ad-hoc ``--key value`` overrides for any config key).
    """
    config = load_index_config(args.config, defaults)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)


def resolve_max_length(args) -> None:
    """Auto-select ``max_length`` left null in config from dataset + retriever."""
    if args.max_length is not None:
        return
    args.max_length = doc_max_length(args.dataset, args.retriever)
    print(f"Auto-selected max_length: {args.max_length} ({args.dataset} + {args.retriever})")


def resolve_split_defaults(args) -> None:
    """Auto-select ``dataset_year`` / ``subset`` left null in config from the dataset spec."""
    apply_dataset_defaults(args, ("dataset_year", "subset"))
