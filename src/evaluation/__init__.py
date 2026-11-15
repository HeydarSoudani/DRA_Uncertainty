"""Evaluation of deep-research agent runs: scoring only.

The run's files are written and read by ``run_files``; the evaluation reads
them back and writes its own under ``eval/`` (``metrics_file``).  One
module or subpackage per group of the run's ``metrics.md``, in order:

retrieval
    Recall@N, Precision@N of the seen docs (``SeenDocEvaluator``).
generation
    Answer correctness (``AccuracyEvaluator`` by LLM judge,
    ``NumericMatchEvaluator`` by numeric match), the report evaluation
    against nuggets (``ArgueReportEvaluator``, Auto-ARGUE) and the agent's
    verbal confidence.
uncertainty
    s_T and its baselines against correctness (``UncertaintyEvaluator``)
    or against nugget coverage (``ReportUncertaintyEvaluator``).
belief
    The belief update evaluation (``BeliefUpdateEvaluator``) with its
    per-criterion gold, and the update's issues (``issue_summary``).
trajectory
    Per-query searches, tokens and forced answers (``TrajectoryEvaluator``).

Beside them: ``runner`` (evaluation of a run), ``metrics_file`` (its files
and log), ``measures`` (AUROC, Brier, ECE, tau-b, bootstrap CIs) and
``judge`` (the LLM-judge client).
"""
