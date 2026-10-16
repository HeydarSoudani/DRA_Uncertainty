"""SFT backend — TODO (veRL multi-turn SFT).

Cold-start supervised fine-tuning on the per-turn chat samples written by
``data.sft_dataset.generate_sft_data``.  Delegate to veRL's SFT trainer (the
RL backend's sibling) rather than hand-roll.

TODO(sft-backend):
  * feed the JSONL samples (loss on the final assistant message only),
  * emit the reference/init checkpoint consumed by the RL stage.
"""

from __future__ import annotations

from ..config import SFTConfig


def run_sft_backend(cfg: SFTConfig, data_path: str) -> str:
    raise NotImplementedError(
        "SFT backend is a TODO (veRL SFT). The data is ready at "
        f"{data_path}; set sft.backend: none to stop after building it."
    )
