"""The belief model's tokens, once per belief model of the run.

Per query: ``belief_tokens``, the model's total tokens (the meta line of
``belief/{model}/{qid}.jsonl``).  Row (``trajectory`` group, the model's
``belief_model`` cell): ``belief tokens``, their mean over the queries that
record them.  No LLM call.
"""

from typing import Any, Dict, Optional

from ..shared.context import BELIEF_MODEL, EvalContext, EvalSetup
from ..shared.measures import mean
from ..shared.result import GroupResult
from .metrics import BELIEF_TOKENS, GROUP


def belief_tokens(files: Dict[str, Any]) -> GroupResult:
    """The row and per-query values of the belief *files* (``load_belief_files``)."""
    tokens = {q: (f[0].get("tokens") or {}).get("total_tokens") for q, f in files.items()}
    defined = [v for v in tokens.values() if v is not None]
    res = GroupResult(per_query={q: {"belief_tokens": v} for q, v in tokens.items()})
    if defined:
        res.add(GROUP, BELIEF_TOKENS, mean(defined), n=len(defined))
    return res


class BeliefTokenEvaluator:
    """:func:`belief_tokens` of each belief model; every dataset."""

    name = "belief tokens"
    scope = BELIEF_MODEL
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["BeliefTokenEvaluator"]:
        return cls()

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        return belief_tokens(ctx.belief_files)
