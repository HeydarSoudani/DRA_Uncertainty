"""Process reward — DESIGNABLE, stubbed.

Per-search-turn credit.  The plan is to
wrap the finalized ``uncertainty_estimator.signals`` (READ-ONLY) — DocNovelty,
QueryNovelty, RetrievalGain (needs qrels), the criteria signals — and pair information-gain with a
redundancy penalty (StepSearch-style) to resist novelty-farming.

Only the INTERFACE is fixed here; which signals and how they map to a scalar per
turn is designable.  The stub returns zeros so the composite reward runs.
"""

from __future__ import annotations

import logging
from typing import List

from .base import ProcessComponent
from ..config import RewardConfig
from ..data.schema import PromptRecord, Trajectory

logger = logging.getLogger(__name__)


class ProcessReward(ProcessComponent):
    """Wraps uncertainty-estimator signals into per-turn credit. TODO(reward-process)."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        # TODO(reward-process): lazily construct the requested signals, e.g.
        #   from uncertainty_estimator.signals import DocNoveltySignal, RetrievalGainSignal
        #   self._novelty = DocNoveltySignal(); self._gain = RetrievalGainSignal(qrels=...)
        # NOTE: RetrievalGain needs qrels at RL time -> thread prompt.qrels through.

    def per_turn(self, traj: Trajectory, prompt: PromptRecord) -> List[float]:
        # TODO(reward-process): for each search turn, run signals over
        # turn.obs_doc_ids (novelty vs seen, new-item recall vs prompt.qrels,
        # coverage gain vs prompt.criteria) and combine into one scalar.
        # Returning zeros keeps the pipeline runnable until designed.
        return [0.0] * traj.num_search_turns


class MockProcessReward(ProcessComponent):
    """Smoke-test signal: +1 for each search turn that surfaces the prompt's gold
    doc, -0.5 for a fully-redundant turn (repeats the previous turn's docs).

    A minimal stand-in for 'information gain minus redundancy' so the loop shows
    non-trivial process credit on CPU — NOT the designed reward."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def per_turn(self, traj: Trajectory, prompt: PromptRecord) -> List[float]:
        gold = set((prompt.qrels or {}).keys())
        seen: set = set()
        out: List[float] = []
        prev: set = set()
        for t in traj.turns:
            if t.action_kind != "search":
                continue
            ids = set(t.obs_doc_ids)
            gain = 1.0 if (ids & gold) - seen else 0.0
            redundant = -0.5 if ids and ids == prev else 0.0
            out.append(gain + redundant)
            seen |= ids
            prev = ids
        return out
