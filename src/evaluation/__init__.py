"""Evaluation of deep-research agent runs: scoring only.

The run's files are written and read by ``utils.run_files``; the evaluation
reads them back and writes its own under ``eval/``.  One package per group
of the run's ``metrics.md``, in order:

    retrieval     Recall@N, Precision@N of the seen documents
    generation    answer accuracy (LLM judge, numeric match) or report nugget
                  coverage (Auto-ARGUE)
    uncertainty   the agent's verbal confidence and the belief scores against
                  correctness, or s_T against nugget coverage
    belief        evidence uptake, false uptake and update issues of a belief model
    trajectory    searches, tokens, forced answers

Beside them: ``shared`` (what every group uses) and ``runner`` (the
evaluation of a run: which evaluators, in which order).  The overview, the
evaluator interface and how to add one: ``README.md`` in this directory.
"""
