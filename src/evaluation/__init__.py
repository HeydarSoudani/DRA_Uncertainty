"""Evaluation of deep-research agent runs: scoring only.

The run's files are written and read by ``utils.run_files``; the evaluation reads
them back and writes its own under ``eval/`` (``metrics_file``).  One
module or subpackage per group of the run's ``metrics.md``, in order:

retrieval
    Recall@N, Precision@N of the seen docs (``SeenDocEvaluator``).
generation
    Answer correctness (``AccuracyEvaluator`` by LLM judge,
    ``NumericMatchEvaluator`` by numeric match) and the report evaluation
    against nuggets (``ArgueReportEvaluator``, Auto-ARGUE).
uncertainty
    The agent's verbal confidence against correctness
    (``VerbalConfidenceEvaluator``, once per run); the belief rule-based
    score and the belief LLM score against correctness
    (``UncertaintyEvaluator``), or s_T against nugget coverage
    (``ReportUncertaintyEvaluator``), once per belief model.
belief
    Evidence uptake and false uptake of the belief's certainty against the
    qrels (``evidence_uptake``), and the update's issues (``issue_summary``).
trajectory
    Per-query searches, tokens and forced answers (``TrajectoryEvaluator``).

Beside them: ``runner`` (evaluation of a run), ``metrics_file`` (its files
and log), ``measures`` (AUROC, Brier, ECE, tau-b, bootstrap CIs) and
``judge`` (the LLM-judge client).
"""
