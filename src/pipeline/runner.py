"""The run of the stages of ``--stage``, then the evaluation of the stage's
scope."""

from typing import Optional

from evaluation.runner import build_evaluators, evaluate_and_save, load_run_results
from utils.run_files import scan_queries

from .belief_stage import run_belief
from .generate_stage import run_generate_stage
from .run_plan import BELIEVES, GENERATES, RunPlan


def run_pipeline(plan: RunPlan, data_path: str, policy_model: str = "cpm_report", verbose: bool = True,
                 workers: int = 1, worker_config: Optional[dict] = None, **kwargs):
    """Run the stages of ``kwargs["stage"]`` (see
    ``experiments/inference/dra_inference.py``), then evaluate the stage's
    scope (generate: the generation; belief: its belief model; eval, all:
    everything), keeping the rest of the evaluation files.

    The evaluation scores every finished query of the run directory:
    ``--limit`` and ``--query-ids-file`` only choose which queries the
    generate and belief stages run.

    Args:
        plan:          From ``pipeline.run_plan.prepare_run``.
        data_path:     Path to dataset directory (nuggets of the report datasets).
        policy_model:  Agent to use (internal name); one of ALL_AGENTS.
        verbose:       Whether to print detailed logs.
        workers:       Query-level worker processes of the generate stage.
                       1 = sequential (default). >1 = one process per worker
                       (one GPU each on a GPU machine, CPU otherwise), queries
                       split across them; each saves its queries' files
                       straight to the run directory.
        worker_config: Serialisable config dict passed to each worker so it
                       can rebuild the agent stack.  Required when workers > 1.
        **kwargs:      Agent-specific, belief and evaluation parameters.
    """
    dataset, run_dir, answers = plan.dataset, plan.run_dir, plan.answers
    stage = kwargs["stage"]

    evaluators = build_evaluators(plan.qrels, kwargs, answers=answers, questions=plan.all_questions,
                                  dataset=dataset,
                                  data_path=data_path, split=plan.split)

    if stage in GENERATES and plan.queries:
        run_generate_stage(plan, policy_model, kwargs["llm_model"], verbose, workers, worker_config, kwargs)
    if stage in BELIEVES:
        run_belief(plan, verbose, kwargs)

    # ==================== Read the run back from disk ====================
    # Every query of the run is evaluated from its saved files; each stage
    # scores its own scope of eval/metrics.md and eval/per_query.csv.
    finished, unfinished = scan_queries(run_dir)
    if unfinished:
        print(f"\n{len(unfinished)} unfinished queries (no meta line in trajectory/{{qid}}.jsonl); "
              f"they are generated again on resume: "
              f"{', '.join(sorted(unfinished)[:20])}{' ...' if len(unfinished) > 20 else ''}")
    if not finished:
        print(f"No finished query in {run_dir.rstrip('/')}/trajectory. Run --stage generate first.")
        return
    if kwargs.get("query_ids_file") and set(finished) - set(plan.all_questions):
        print(f"Evaluating the {len(finished)} finished queries of the run "
              f"(--query-ids-file selected {len(plan.all_questions)}; it does not filter the evaluation)")
    results = load_run_results(run_dir, finished)
    if not results:
        print(f"Could not load any results from {run_dir}.")
        return

    # The LLM-as-judge evaluators (BrowseComp-Plus accuracy, Auto-ARGUE reports)
    # call the OpenRouter-hosted judge directly and reuse the verdicts saved in
    # the run directory; no local server to start.
    scope = {"generate": dict(generation=True, models=[]),
             "belief": dict(generation=False, models=[kwargs["belief_model_name"]])}.get(stage, {})
    evaluate_and_save(results, evaluators, run_dir, num_unfinished=len(unfinished), **scope)

    # ==================== Final status ==========================================
    print(f"\n{'=' * 80}")
    print(f"[OUTPUT] All results saved to: {run_dir}")
    print(f"{'=' * 80}")
