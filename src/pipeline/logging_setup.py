"""The ``--quiet`` logging setup of a process (the main one and each GPU worker)."""

import logging
import os
import sys


def _silence_hf_progress_bars() -> None:
    """Turn off HuggingFace/datasets tqdm bars and downgrade their loggers.

    Called from :func:`quiet_logging` so model/corpus loading stays silent in
    non-interactive runs.
    Each import is guarded: the helper must never be the reason a worker dies.
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for mod, attr in (
        ("transformers.utils.logging", "disable_progress_bar"),
        ("datasets.utils.logging", "disable_progress_bar"),
        ("huggingface_hub.utils", "disable_progress_bars"),
    ):
        try:
            __import__(mod)
            getattr(sys.modules[mod], attr)()
        except Exception:
            pass
    for name in ("transformers", "datasets", "sentence_transformers", "faiss"):
        logging.getLogger(name).setLevel(logging.ERROR)


def quiet_logging() -> None:
    """The ``--quiet`` setup of a process (the main one and each GPU worker):
    the component packages' and utils loggers at ERROR (their modules log
    under ``__name__``), no HF/datasets progress bars."""
    for name in ("deep_research_agents", "searcher_component", "reasoner_component",
                 "belief_model_component", "utils"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # The retriever's from_pretrained() draws a 398-step "Loading weights"
    # tqdm bar per process.  Under sbatch stdout is not a TTY, so every
    # redraw is appended verbatim to the .out file.
    _silence_hf_progress_bars()
