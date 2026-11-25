"""What every evaluation group uses.

    result     Row, Section, GroupResult, NO_MODEL: what an evaluator returns
    context    EvalSetup (what evaluators are built from), EvalContext (what they read)
    files      the eval/ files: metrics.md, per_query.csv, their paths and readers
    log        the terminal block and the helpers the groups' log lines use
    measures   AUROC, Brier, ECE, tau-b, means, bootstrap CIs
    judge      the LLM judge (free-text grader, YES/NO judge, verdict keys)

Nothing here knows a group: the groups import from here, never the reverse.
"""
