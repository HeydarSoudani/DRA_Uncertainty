"""Loss-mask helpers and trajectory validation.

CORRECTNESS-CRITICAL.  The single most important agentic-RL detail is that only
the policy's own generated tokens contribute to the loss; the context, and with
it every injected tool observation, is masked out (Search-R1
``state_masking``).  A silent misalignment here trains the model on retrieved
text and quietly wrecks RL, so everything is asserted.
"""

from __future__ import annotations

from typing import List

from ..data.schema import Trajectory, Turn


def make_observation_text(docs: List[dict], *, max_docs: int = 5, max_chars: int = 500) -> str:
    """Render retrieved docs into the observation string of the reference env."""
    lines = []
    for d in docs[:max_docs]:
        did = d.get("doc_id") or d.get("id") or ""
        txt = d.get("text") or d.get("relevant_text") or ""
        lines.append(f"[{did}] {txt[:max_chars]}")
    return "<information>\n" + "\n".join(lines) + "\n</information>"


def validate_turn(turn: Turn) -> None:
    ids = turn.token_ids
    mask = turn.loss_mask
    assert len(ids) == len(mask), (
        f"Turn {turn.index}: token/mask length mismatch ({len(ids)} != {len(mask)})"
    )
    n_prompt = len(turn.prompt_token_ids)
    # context tokens masked (0), generated tokens trained on (1)
    assert mask[:n_prompt] == [0] * n_prompt, \
        f"Turn {turn.index}: prompt tokens must have loss_mask == 0"
    assert mask[n_prompt:] == [1] * len(turn.gen_token_ids), \
        f"Turn {turn.index}: generated tokens must have loss_mask == 1"
    if turn.gen_logprobs is not None:
        assert len(turn.gen_logprobs) == len(turn.gen_token_ids), \
            f"Turn {turn.index}: logprob/token length mismatch"


def validate_trajectory(traj: Trajectory) -> None:
    for t in traj.turns:
        validate_turn(t)
