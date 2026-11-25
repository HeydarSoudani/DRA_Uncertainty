"""The reward of RL: the Brier score of the final score, behind a format gate.

There is no reward on the belief transitions.  Per final-score reply j of a
chain, with p_j the probability of its WEP (``status_probability``) and y
the correctness of the agent's final answer:

    r_j = gate_j * (1 - (p_j - y)^2)

gate_j is 1 when the reply parses (``parse_final_score``) and was not cut
off.  A chain gets ``scorer_samples`` such replies on its own final belief
(its own call and extra samples of the same prompt); its reward is

    R = gate_chain * mean_j r_j

gate_chain is Agent-BRACE's format gate over the chain's belief calls: 1
when every call parsed at its first attempt, was not cut off and has no
issue of ``GATE_ISSUES`` (the init call, with own beliefs: a belief list).
"""

from typing import Any, Dict, Iterable, Optional

from .chain import SCORE, ChainResult

#: A reply cut at its token budget (vLLM's finish reason).
TRUNCATED = "length"


def brier_reward(p: float, y: bool) -> float:
    return 1.0 - (p - float(y)) ** 2


def score_probability(reply: str, finish_reason: Optional[str] = None) -> Optional[float]:
    """The probability of a final-score reply, None when it fails the gate."""
    from belief_model_component import parse_final_score

    if finish_reason == TRUNCATED:
        return None
    score = parse_final_score(reply or "")
    return score.probability if score is not None else None


def score_reward(reply: str, y: bool, finish_reason: Optional[str] = None) -> float:
    """r_j of one final-score reply."""
    p = score_probability(reply, finish_reason)
    return 0.0 if p is None else brier_reward(p, y)


def step_passes(record: Dict[str, Any]) -> bool:
    """A step record's update call (if any) passed the gate."""
    from belief_model_component import GATE_ISSUES

    if not record.get("called"):
        return True
    if any(str(e).startswith(("parse", "belief_update")) for e in record.get("errors") or []):
        return False
    return not any(i.get("code") in GATE_ISSUES for i in record.get("issues") or [])


def chain_gate(result: ChainResult) -> bool:
    """gate_chain of the module docstring."""
    if result.error is not None or not result.beliefs:
        return False
    if any(c.finish_reason == TRUNCATED for c in result.calls if c.kind != SCORE):
        return False
    return all(step_passes(r) for r in result.steps)


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None
