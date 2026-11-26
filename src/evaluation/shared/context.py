"""What the evaluators are built from (:class:`EvalSetup`) and what they read
when they run (:class:`EvalContext`).

Every evaluator class has the same shape (``evaluation/README.md``)::

    class XEvaluator:
        name = "..."              # its label in the timing line
        scope = GENERATION        # or BELIEF_MODEL: once per belief model
        provides = None           # or the EvalContext field its raw output fills

        @classmethod
        def from_setup(cls, setup: EvalSetup) -> Optional["XEvaluator"]: ...
        def evaluate(self, ctx: EvalContext) -> GroupResult: ...
        def pending(self, ctx: EvalContext) -> int: ...   # with ``provides`` only

``from_setup`` returns None when the evaluator does not apply to the
dataset; ``evaluate`` returns an empty ``GroupResult`` when its inputs are
missing (no belief file, no answer verdicts).  An evaluator with
``provides`` also has ``pending``: how many of its verdicts are not saved
and would be judged now (no LLM call).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

#: Scopes: scored once per run (the generation), or once per belief model.
GENERATION = "generation"
BELIEF_MODEL = "belief_model"


@dataclass
class EvalSetup:
    """The dataset side of a run, fixed before any query is scored.

    Attributes:
        spec:        The dataset's ``layout.DATASET_SPECS`` entry.
        qrels:       ``{query_id: {doc_id: grade}}`` at the run's ``min_relevance_score``.
        answers:     ``query_id -> ground-truth answer`` (QA datasets).
        questions:   ``query_id -> question / request text``.
        judge_model: The LLM judge; None: ``shared.judge.DEFAULT_JUDGE_MODEL``.
        corpus_path: The corpus JSONL (the cited documents of a report).
        data_path:   The dataset directory (the nuggets of a report dataset).
        split:       The split's name in *data_path* (the nuggets file).
    """
    spec: Any
    qrels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    answers: Optional[Dict[str, str]] = None
    questions: Optional[Dict[str, str]] = None
    judge_model: Optional[str] = None
    corpus_path: Optional[Path] = None
    data_path: Optional[Path] = None
    split: Optional[str] = None

    @property
    def task(self) -> str:
        """``"qa"`` or ``"report"``."""
        return self.spec.task

    @property
    def answer_eval(self) -> Optional[str]:
        """How answers are scored: ``"llm_judge"``, ``"numeric_match"`` or None."""
        return self.spec.answer_eval

    @property
    def has_accuracy(self) -> bool:
        """Answers are scored (ground-truth answers and an ``answer_eval``)."""
        return bool(self.answers) and self.answer_eval in ("llm_judge", "numeric_match")

    @property
    def has_report_eval(self) -> bool:
        """Reports are scored against nuggets (``report_eval == "argue"``)."""
        return self.spec.report_eval == "argue"

    @property
    def judge_kwargs(self) -> Dict[str, str]:
        """``{"judge_model": ...}`` when one is set, for the judged evaluators."""
        return {"judge_model": self.judge_model} if self.judge_model else {}


@dataclass
class EvalContext:
    """What an evaluator reads.

    Attributes:
        results:      ``{query_id: result}`` read back from the run's files
                      (``runner.load_run_results``).
        run_dir:      The run directory; None: no file is read or written.
        qrels:        As in :class:`EvalSetup`.
        model:        The belief model being scored (scope ``BELIEF_MODEL``), else None.
        belief_files: ``{query_id: (meta, steps, final)}`` of *model*
                      (``utils.run_files.load_belief_files``).
        accuracy:     The raw output of the accuracy evaluator (``per_query``
                      verdicts with ``correct``, ``judge_error``,
                      ``verbal_confidence``); ``{}`` when answers are not scored.
        report:       The raw output of the report evaluator (Auto-ARGUE
                      ``per_query`` records); ``{}`` when reports are not scored.
    """
    results: Dict[str, Dict[str, Any]]
    run_dir: Optional[Path] = None
    qrels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model: Optional[str] = None
    belief_files: Dict[str, Any] = field(default_factory=dict)
    accuracy: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def finals(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """``{query_id: final belief line}`` of *belief_files*."""
        return {q: f[2] for q, f in self.belief_files.items()}
