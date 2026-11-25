"""Evidence uptake: does the belief's certainty s_t rise when the agent finds
a gold document, and stay put when it finds none?

Reads one belief model's step records (``belief/{model}/{qid}.jsonl``) and
the run's qrels.  QA datasets only: the qrels of a report dataset are
pooled, so an unjudged relevant document would count as a no-evidence
document.

Definitions:
    step t            one intermediate step of the trajectory (an agent turn
                      and its belief update); the final update on the answer
                      turn is left out
    s_t               the step's ``certainty`` (the mean status probability
                      of the beliefs); s_0 = ``STATUS_PROBABILITY["unknown"]``
    new document      a document at the first step that shows it
    gold document     a document in the query's qrels (at the run's
                      ``min_relevance_score``)
    evidence step     a step with at least one new gold document
    no-evidence step  a step with new documents, none of them gold

A step with no new document is neither.  A step is not scored when s_{t-1}
is already the top (``STATUS_PROBABILITY["certain"]``: s_t cannot rise) or
when its update call failed (called, no reply parsed: the belief did not
read the step); its documents still count as seen.

Rows (``belief`` group):
    evidence uptake   per query, the share of its evidence steps where s_t
                      rises; the mean over the queries with one (bootstrap
                      CI over queries, n = those queries)
    false uptake      the same over the no-evidence steps

Per query: ``evidence_uptake``, ``false_uptake``.  No LLM call.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from belief_model_component import UNKNOWN, status_probability

from ..shared.context import BELIEF_MODEL, EvalContext, EvalSetup
from ..shared.measures import bootstrap_ci, mean
from ..shared.result import GroupResult
from .metrics import EVIDENCE_UPTAKE, FALSE_UPTAKE, GROUP

#: s_0: the certainty of a belief with no evidence.
START = status_probability(UNKNOWN)
#: The top of s_t: every belief "certain".
TOP = status_probability("certain")
#: A change of s_t below this is no change (rounding of the saved certainty).
EPS = 1e-9


def _query_uptake(steps: List[Dict[str, Any]], relevant: Set[str]) -> Tuple[List[bool], List[bool]]:
    """``(evidence, no_evidence)``: per scored step of each kind, whether s_t rose."""
    evidence: List[bool] = []
    no_evidence: List[bool] = []
    seen: Set[str] = set()
    prev = START
    for step in steps:
        if step.get("final_update"):
            continue
        ids = [d["doc_id"] for d in step.get("docs") or [] if d.get("doc_id")]
        new = {i for i in ids if i not in seen}
        seen.update(ids)
        cur = step.get("certainty")
        failed = bool(step.get("called")) and step.get("observations") is None
        if new and cur is not None and not failed and prev < TOP:
            (evidence if new & relevant else no_evidence).append(cur - prev > EPS)
        if cur is not None:
            prev = cur
    return evidence, no_evidence


def evidence_uptake(files: Dict[str, Tuple[Any, List[Dict], Any]],
                    qrels: Dict[str, Dict[str, Any]]) -> GroupResult:
    """The uptake rows of the belief *files* (``load_belief_files``) against
    *qrels* (``{query_id: {doc_id: grade}}``, already thresholded); empty
    when no query has a scored step."""
    per_query: Dict[str, Dict[str, Optional[float]]] = {}
    for qid, (_meta, steps, _final) in files.items():
        relevant = {d for d, grade in (qrels.get(qid) or {}).items() if float(grade) > 0}
        evidence, no_evidence = _query_uptake(steps, relevant)
        per_query[qid] = {"evidence_uptake": mean(evidence) if evidence else None,
                          "false_uptake": mean(no_evidence) if no_evidence else None}
    res = GroupResult(per_query=per_query)
    for metric, key in ((EVIDENCE_UPTAKE, "evidence_uptake"), (FALSE_UPTAKE, "false_uptake")):
        values = [r[key] for r in per_query.values() if r[key] is not None]
        if values:
            res.add(GROUP, metric, mean(values), *bootstrap_ci(values, mean), len(values))
    return res if res.rows else GroupResult()


class EvidenceUptakeEvaluator:
    """:func:`evidence_uptake` of each belief model; QA datasets."""

    name = "belief"
    scope = BELIEF_MODEL
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["EvidenceUptakeEvaluator"]:
        return cls() if setup.task == "qa" else None

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        if not ctx.qrels:
            return GroupResult()
        return evidence_uptake(ctx.belief_files, ctx.qrels)
