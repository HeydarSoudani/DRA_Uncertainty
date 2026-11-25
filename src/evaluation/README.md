# `evaluation`: scoring a run

The evaluation reads a run back from its saved files (`utils.run_files`) and writes `eval/metrics.md` and
`eval/per_query.csv`. It only scores: it never runs the agent or the belief model.
`experiments/inference/dra_inference.py` calls it at the end of every stage through `runner.py`.

## Layout

There is one directory per **group** of `metrics.md`, so a row's `group` cell tells you which directory produced
it.

```
evaluation/
├── runner.py          EVALUATORS (the registry, in run order), build_evaluators, load_run_results, evaluate_and_save
├── shared/            what every group uses; it imports no group
│   ├── result.py      Row, Section, GroupResult, NO_MODEL
│   ├── context.py     EvalSetup (what evaluators are built from), EvalContext (what they read), scopes
│   ├── files.py       eval/ paths; write/read metrics.md and per_query.csv
│   ├── log.py         the terminal block, and the get/val/n/ci helpers the log lines use
│   ├── measures.py    auroc, brier, tau_b, mean, mean_or_none, bootstrap_ci
│   └── judge.py       the LLM judge: free-text grader, YesNoJudge, verdict_key
├── retrieval/         seen_docs.py      SeenDocEvaluator
├── generation/        llm_judge.py      AccuracyEvaluator          (answer_eval "llm_judge")
│                      numeric_match.py  NumericMatchEvaluator      (answer_eval "numeric_match", TRQA)
│                      accuracy.py       the accuracy rows both produce
│                      argue.py          ArgueReportEvaluator       (report_eval "argue": NeuCLIR, RAGTIME)
│                      confidence.py     verbal_confidence (parsed from the generation)
│                      references.py     the References block of a generation
│                      report_sentences.py  a report as cited sentences (Auto-ARGUE's input)
├── uncertainty/       verbal.py         VerbalConfidenceEvaluator  (QA, once per run)
│                      qa.py             UncertaintyEvaluator       (QA, per belief model)
│                      report.py         ReportUncertaintyEvaluator (reports, per belief model)
├── belief/            uptake.py         EvidenceUptakeEvaluator    (QA, per belief model)
│                      issues.py         IssueEvaluator             (per belief model)
└── trajectory/        agent.py          TrajectoryEvaluator        (once per run)
                       belief_tokens.py  BeliefTokenEvaluator       (per belief model)
```

Every group directory has the same files:

- `__init__.py` lists the group's modules and exports its evaluators.
- `metrics.py` holds the group's `GROUP` name, its metric names, and its log line (`log_line` for the
  generation's scope, `model_log_line` for a belief model's). No other file writes a metric name of the group as
  a literal.
- One file per evaluator. Its module docstring defines what it measures: the definitions, the **Rows** it adds
  to `metrics.md`, its **per-query** columns, and whether it calls an LLM.

## The evaluator interface

Every evaluator class has the same shape (`shared/context.py`):

```python
class XEvaluator:
    name = "x"                 # its label in the timing line
    scope = GENERATION         # or BELIEF_MODEL: scored once per belief model of the run
    provides = None            # or "accuracy" / "report": the EvalContext field its raw output fills

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["XEvaluator"]:
        ...                    # None when it does not apply to the dataset

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        ...                    # empty GroupResult when its inputs are missing

    def pending(self, ctx: EvalContext) -> int:
        ...                    # only with `provides`: verdicts not saved yet, which
                               # evaluate would judge now (no LLM call)
```

Each evaluator module also has a plain function with explicit inputs that does the work, so the evaluator can be
used outside a run (for example in a notebook): `seen_doc_scores(results, qrels)`, `trajectory_stats(results)`,
`belief_tokens(files)`, `evidence_uptake(files, qrels)`, `issue_summary(files)`,
`verbal_confidence_scores(accuracy_per_query)`, `belief_scores(finals, accuracy_per_query)` and
`report_scores(finals, report_per_query)`. The judged evaluators expose `score(results, run_dir)` instead, which
returns their raw verdicts.

A `GroupResult` carries:

- `rows`: rows of `metrics.md`.
- `per_query`: columns of `per_query.csv`.
- `sections`: extra titled tables, such as the belief issue table.
- `raw`: output that a later evaluator reads.

The runner tags a belief model's rows with the model, prefixes its per-query columns with `{model}/`, and adds
` ({model})` to its section titles.

## A run's evaluation

`evaluate_and_save` follows the registry order:

1. **The generation** (`scope == GENERATION`) is scored once: retrieval, accuracy or the report evaluation,
   verbal confidence, then trajectory. The evaluators with `provides` fill `ctx.accuracy` / `ctx.report`.
2. **Each belief model** (`scope == BELIEF_MODEL`) is scored with its belief files loaded once into
   `ctx.belief_files`: uncertainty, belief uptake and issues, then belief tokens.
3. **Rows, columns and sections out of the scope are kept** from the existing `metrics.md` and
   `per_query.csv`. The generate stage scores only the generation; the belief stage adds only its own model, and
   re-reads the answer verdicts from `eval/generation/` without making a judge call. The belief stage scores the
   generation too when its saved rows cannot be kept (no `metrics.md`, another query count, a query missing from
   `per_query.csv`) or when a verdict is pending (`pending > 0`: a judge failure of the generation's evaluation,
   judged again now), so the belief models are always scored against the verdicts of the generation's rows. A
   belief model without any belief file to read keeps its saved rows.

The judged evaluators cache their verdicts under `eval/generation/`, keyed by `shared.judge.verdict_key` over
everything a verdict depends on. Re-evaluating an unchanged run makes no LLM call; a failed judgment is retried
on the next evaluation.

| File | Contents |
|---|---|
| `eval/metrics.md` | header, then one table `group \| belief_model \| metric \| value \| ci_low \| ci_high \| n` (groups in `shared.files.GROUPS` order; `belief_model` "-" for the generation), then the extra sections |
| `eval/per_query.csv` | one row per query, every per-query value of every group |
| `eval/generation/accuracy.jsonl` | the answer verdicts (QA) |
| `eval/generation/{judgments/, nuggets/, scores.tsv, ...}` | the Auto-ARGUE cache and scores (reports) |

The measures and their definitions per dataset are described in `experiments/inference/README.md` ("Evaluation").

## Adding an evaluator

1. Put the file in the directory of the group its rows belong to (`metrics.md`'s `group` cell). Add its metric
   names to that group's `metrics.py`, and its log text there too if it has any.
2. Write the plain function, then the class with `name`, `scope`, `provides`, `from_setup` and `evaluate`
   (and `pending` when it has `provides`).
   The module docstring lists its rows and per-query columns.
3. Export it from the group's `__init__.py`, and add it to `runner.EVALUATORS` after any evaluator whose raw
   output it reads. Its position fixes its columns' place in `per_query.csv`.
4. For a new group, also add it to `shared.files.GROUPS` and its log line to `runner.GENERATION_LOG_LINES` or
   `runner.MODEL_LOG_LINES`.

Renaming a metric changes `metrics.md`: older runs keep the old name until they are evaluated again.
