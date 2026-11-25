"""The roots of the project's files: the downloaded data, the run outputs and
the data this repo builds.  Each root can be moved with a ``DRA_*`` env var,
set in the shell, an sbatch script or the project-root ``.env`` (read here on
import, a real environment value winning).  Standard library only.
"""

import os
from pathlib import Path

#: The repository checkout (this file is ``src/utils/paths.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dra_env_defaults() -> None:
    """Populate ``DRA_*`` env vars from the project-root ``.env`` if unset.

    Every entry point imports this module (inference CLI, downloaders, index
    builder), but the standalone scripts never call ``python-dotenv``, and
    ~/.bashrc is not sourced in non-interactive SLURM jobs.  This tiny,
    dependency-free reader makes ``DRA_DATA_ROOT`` / ``DRA_OUTPUT_ROOT`` set in
    ``.env`` take effect everywhere.  A real environment value always wins, so
    exporting the var (shell or sbatch) still overrides ``.env``.
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("DRA_") and key not in os.environ:
            value = value.split("#", 1)[0].strip().strip('"').strip("'")
            if value:
                os.environ[key] = value


_load_dra_env_defaults()

# The one canonical dataset root.  Defaults to the cluster project location but
# can be overridden with the ``DRA_DATA_ROOT`` env var (no code edit needed) —
# set it in the shell, in an sbatch script, or in the project-root ``.env``.
# Hardcoded default (not derived from ``__file__`` depth) so it is stable
# regardless of where the package is imported from.
DATA_ROOT = Path(os.environ.get(
    "DRA_DATA_ROOT", "/projects/0/prjs0834/heydars/DRA_training/data"
))

# The one canonical root of the run outputs: the checkout's ``run_outputs``
# (as ``scripts/_activate.sh`` sets it); ``DRA_OUTPUT_ROOT`` overrides it
# like ``DRA_DATA_ROOT`` does for the data.
OUTPUT_ROOT = Path(os.environ.get("DRA_OUTPUT_ROOT", str(REPO_ROOT / "run_outputs")))

# The repo's data folder: the data built by this project (not DATA_ROOT,
# which only holds downloaded inputs); it follows the checkout.
REPO_DATA_ROOT = REPO_ROOT / "data"

# The training datasets (``BUILT_DATASETS``), all in one folder like an
# inference dataset's: ``TRAINING_DATASETS_ROOT/queries/{dataset}.jsonl``,
# ``qrels/{dataset}.txt``, ``belief_bank/silver/{dataset}.jsonl`` and the
# shared ``corpus/`` (corpus_web).  Each is built in its own folder of
# ``TRAINING_RAW_ROOT/{dataset}``: the source as loaded, the caches, the
# prepared questions (stage data_prepare), the LLM memories (stage
# data_relevance), the dropped questions and the build stats.
TRAINING_DATASETS_ROOT = REPO_DATA_ROOT / "training_datasets"
TRAINING_RAW_ROOT = REPO_DATA_ROOT / "training_datasets_raw"

# Root of the belief banks, in the repo's data folder; ``DRA_BELIEF_ROOT``
# overrides it.
BELIEF_ROOT = Path(os.environ.get("DRA_BELIEF_ROOT", str(REPO_DATA_ROOT)))
