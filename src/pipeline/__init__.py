"""The inference pipeline: the stages of a run and the wiring of its components.

This is the layer that assembles the component packages
(``deep_research_agents``, ``searcher_component``, ``reasoner_component``,
``belief_model_component``, ``evaluation``) into a run; the components never
import each other through it, and ``utils`` stays below it.  Its entry point
is ``experiments/inference/dra_inference.py`` (the CLI); the belief training
runs that script as a child process.

Modules:
    run_plan        the stages (STAGES), RunPlan and prepare_run: the split,
                    the run directory, resume; no model loaded
    runner          run_pipeline: the stages of --stage, then the evaluation
    generate_stage  the agent on the queries, sequential or one process per GPU
    workers         run_query (one query, its files saved) and the GPU workers
    belief_stage    the belief model over the saved trajectories
    factories       retriever, rerankers, search tool, agent, belief model
    resources       the generate stage's GPUs: worker plan, vLLM start, release
    logging_setup   the --quiet logging of a process
    cli             the CLI's run config (dra_inference.yaml), dataset defaults
                    and the run_pipeline kwargs
    run_config      the run directory's name and run_config.json (built,
                    checked against a resumed run, updated)

The package exports what the CLI uses; the rest is internal to it.
"""

from .cli import (apply_run_config, assemble_pipeline_kwargs, cleanup_event_loop, load_run_config,
                  resolve_dataset_defaults)
from .factories import build_rerankers, setup_retriever_from_args
from .logging_setup import quiet_logging
from .resources import WorkerPlan, plan_generate_workers
from .run_plan import STAGES, prepare_run
from .runner import run_pipeline

__all__ = [
    "apply_run_config", "assemble_pipeline_kwargs", "cleanup_event_loop", "load_run_config",
    "resolve_dataset_defaults",
    "build_rerankers", "setup_retriever_from_args",
    "quiet_logging", "WorkerPlan", "plan_generate_workers",
    "STAGES", "prepare_run",
    "run_pipeline",
]
