"""Outcome reward — is the final answer correct?  1.0 or 0.0.

Scored exactly as the inference evaluation scores it (``evaluation``, used
READ-ONLY), so a training reward of 1.0 means an inference run would count the
answer correct:

    trqa_exact  TRQA: numeric exact match of the extracted prediction
                (``NumericMatchEvaluator``'s ``exact_match``).
    llm_judge   other datasets (BrowseComp-Plus): the BrowseComp-Plus LLM judge
                (``AccuracyEvaluator``: same prompt, model and parsing).
    auto        trqa_exact on TRQA prompts, llm_judge otherwise.

A run that never answers (max turns, format failure, context overflow) has no
answer and scores 0.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict

from .base import OutcomeComponent
from ..config import RewardConfig
from ..data.schema import PromptRecord, Trajectory

logger = logging.getLogger(__name__)

METRICS = ("auto", "trqa_exact", "llm_judge", "mock")


class OutcomeReward(OutcomeComponent):
    def __init__(self, cfg: RewardConfig) -> None:
        if cfg.outcome_metric not in METRICS:
            raise ValueError(f"reward.outcome_metric must be one of {METRICS}, got {cfg.outcome_metric!r}")
        self.cfg = cfg
        self._judge = None
        self._judge_lock = threading.Lock()

    def metric_for(self, prompt: PromptRecord) -> str:
        if self.cfg.outcome_metric != "auto":
            return self.cfg.outcome_metric
        return "trqa_exact" if prompt.meta.get("dataset") == "trqa" else "llm_judge"

    def outcome(self, traj: Trajectory, prompt: PromptRecord) -> float:
        return self.judge(traj, prompt)["correct"]

    def judge(self, traj: Trajectory, prompt: PromptRecord) -> Dict[str, Any]:
        """``{"correct": 0/1, "metric": ..., ...}`` for one trajectory."""
        metric = self.metric_for(prompt)
        golds = prompt.answers or []
        if not traj.answered or not traj.final_answer or not golds:
            return {"correct": 0.0, "metric": metric, "reason": "no answer" if golds else "no gold"}
        if metric == "trqa_exact":
            return self._trqa_exact(traj.final_answer, golds, metric)
        if metric == "llm_judge":
            return self._llm_judge(traj, prompt, golds[0], metric)
        raise ValueError(f"unsupported outcome metric {metric!r}")

    @staticmethod
    def _trqa_exact(answer: str, golds, metric: str) -> Dict[str, Any]:
        from evaluation.answer.numeric_match import extract_prediction, soft_exact_match

        prediction = extract_prediction(answer)
        correct = any(soft_exact_match(prediction, g, decimals=3)["exact_match"] for g in golds)
        return {"correct": float(correct), "metric": metric, "prediction": prediction}

    def _llm_judge(self, traj: Trajectory, prompt: PromptRecord, gold: str, metric: str) -> Dict[str, Any]:
        from evaluation.answer import AccuracyEvaluator

        with self._judge_lock:
            if self._judge is None:
                self._judge = AccuracyEvaluator(answers={}, judge_model=self.cfg.judge_model)
                self._judge.client  # created once, before the threads share it
        verdict = self._judge.judge(prompt.question, traj.final_answer, gold)
        return {"correct": 1.0 if verdict.get("correct") else 0.0, "metric": metric,
                "extracted": verdict.get("extracted_final_answer"),
                "parse_error": verdict.get("parse_error", False)}


class MockOutcomeReward(OutcomeComponent):
    """Deterministic non-zero signal for smoke tests: 1.0 iff the (mock) answer
    matches a gold answer, else 0.0.  Lets the loop show reward variance on CPU."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def outcome(self, traj: Trajectory, prompt: PromptRecord) -> float:
        golds = prompt.answers or []
        return 1.0 if traj.final_answer and traj.final_answer in golds else 0.0
