"""Run deep research agents pipeline.

Supported agents via --policy-model (the LLM is selected automatically per agent):
    cpm_report      Writing-as-Reasoning agent (CPMReport)         → openbmb/AgentCPM-Report (vLLM)
    searchr1        SearchR1 reasoning agent                       → PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3 (vLLM)
    research        ReSearch reasoning agent                       → agentrl/ReSearch-Qwen-7B-Instruct (vLLM)
    stepsearch      StepSearch reasoning agent                     → Zill1/StepSearch-7B-Instruct (vLLM)
    react           ReAct reasoning agent (--use-plan optional)    → claude-sonnet-4-6 (API)
    selfask         SelfAsk reasoning agent                        → claude-sonnet-4-6 (API)
    searcho1        SearchO1 reasoning agent                       → claude-sonnet-4-6 (API)
    webweaver       WebWeaver outline agent                        → Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (vLLM)
    drtulu          DR-Tulu reasoning agent                        → rl-research/DR-Tulu-8B (vLLM)
    glm             GLM reasoning agent                            → zai-org/GLM-4.7-Flash (vLLM)
    oss_20b         GPT-OSS-20B reasoning agent                    → gpt-oss-20b (vLLM)
    oss_120b        GPT-OSS-120B reasoning agent                   → gpt-oss-120b (vLLM)
    tongyi          Tongyi-DeepResearch ReAct agent                → Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (vLLM)
    cpm_explore     AgentCPM-Explore deep search agent             → openbmb/AgentCPM-Explore (vLLM)

Agentic workflows:
    ReAct-style (react, selfask, searcho1, research, searchr1, stepsearch, drtulu, glm, oss_20b, oss_120b, tongyi, cpm_explore):
        Query → [Think → Search → Observe]* → Report → Evaluate
        Instruction-tuned : react, selfask, searcho1
        RL-trained        : research, searchr1, stepsearch, drtulu, tongyi, cpm_explore, glm, oss_20b, oss_120b

    Outline-style (webweaver):
        Query → [Think → Search → Write_outline]* → Outline → [Think → Retrieve → Write_section]* → Report → Evaluate

    Report-style (cpm_report):
        Query → Search → Init Plan → [Search → Write]* → [Extend Plan → [Search → Write]*]* → Report → Evaluate


Stages (--stage):
    generate   Run the agent on the queries without a finished trajectory, save
               its files, then evaluate the generation (retrieval, generation,
               verbal confidence, trajectory).
    belief     Run the belief model --belief-model over the finished
               trajectories without a belief file of its own, in its folder
               belief/{model}/ (pipeline.belief_stage: no agent, no
               retrieval; document text from the corpus by id), then evaluate
               that belief model and add it to the run's evaluation files.  One
               generation, many belief models: each has its own folder and its
               own rows; other settings of a model replace that model's beliefs only.
    eval       Evaluate the run from its saved files (judge verdicts reused),
               every belief model of the run.
    all        generate, then belief, then evaluate everything.

This file is the CLI: the settings (YAML config + flags) and the setup of
the run.  The stages themselves are the ``pipeline`` package (src/pipeline/:
run_plan, runner, generate_stage, belief_stage, workers, factories).

Output structure: see OUTPUT STRUCTURE at the end of this file.
"""

import argparse
import time
import warnings
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Suppress async cleanup noise
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("asyncio.sslproto").setLevel(logging.CRITICAL)

from indexing_corpus_dataset.layout import BELIEF_SOURCES, DATASETS, OUTPUT_ROOT

from utils.config import BELIEF_MODEL_TO_LLM, DEFAULT_BELIEF_MODEL, POLICY_MODEL_TO_LLM, POLICY_MODEL_ALIAS, resolve_temperature
from pipeline import (
    STAGES,
    build_rerankers,
    plan_generate_workers,
    prepare_run,
    quiet_logging,
    run_pipeline,
    setup_retriever_from_args,
)
from reasoner_component import setup_llm
from utils.cli_setup import (
    resolve_dataset_defaults,
    assemble_pipeline_kwargs,
    cleanup_event_loop,
    load_run_config,
    parse_cli_overrides,
    apply_config_to_args,
    _sm_bool,
)

_OUTPUT_PREFIX = str(OUTPUT_ROOT)
_CONFIG_DEFAULT = str(Path(__file__).resolve().parents[1] / "configs" / "dra_inference.yaml")


# ============================================================================
# Run settings
# ============================================================================
def _resolve_args(args, extras):
    """Merge the YAML config and CLI overrides onto ``args`` and derive the agent settings."""

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    cli_subset = args.subset
    cli_belief_source = args.belief_source
    config = load_run_config(args.config)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset
    if cli_belief_source is not None:
        args.belief_source = cli_belief_source

    # ── Map --belief-model to its LLM (the full id is what the bank and run_config record);
    # the short name names the model's folder (belief/{name}/) and run_config entry ──
    args.belief_model_name = args.belief_model
    args.belief_model = BELIEF_MODEL_TO_LLM[args.belief_model]

    # ── Derive --llm-model from --policy-model ────────────────────────────-
    args.llm_model = POLICY_MODEL_TO_LLM[args.policy_model]
    args.policy_model_cli = args.policy_model
    args.policy_model = POLICY_MODEL_ALIAS.get(args.policy_model, args.policy_model)

    # ── Resolve the sampling temperature from the agent ────────────────────-
    # Left on auto (null in the YAML), each agent gets the value it was tuned
    # at; an explicit --llm-temperature / YAML value wins for every agent.
    # Resolved here, before assemble_pipeline_kwargs, so the generator, the
    # worker config and the recorded run_config all see the one number.
    _explicit_temp = args.llm_temperature
    args.llm_temperature = resolve_temperature(args.policy_model, _explicit_temp)
    if _explicit_temp is None:
        print(f"Auto-selected temperature: {args.llm_temperature} (from --policy-model {args.policy_model_cli})")

    if args.output is None:
        args.output = _OUTPUT_PREFIX

    return args


def main():
    """Main entry point: resolve the settings, plan the run, set up the
    generate stage's GPUs and components when there are queries to
    generate, then run the stages (``pipeline.run_pipeline``)."""
    args    = _resolve_args(*parse_args())

    verbose = args.verbose and not args.quiet

    if args.quiet:
        # Same treatment the workers get in _init_worker: no HF/datasets tqdm
        # bars, which single-GPU runs would otherwise draw in this process.
        quiet_logging()

    resolve_dataset_defaults(args)

    from reasoner_component.vllm_server import VLLMServerManager
    vllm_manager = VLLMServerManager()

    # Workers and GPUs of the generate stage, set below when there are
    # queries to generate: the belief and eval stages use no GPU (they run
    # on a CPU machine), so nothing here touches torch or CUDA for them.
    workers = 1
    gpu_ids = None
    total_gpus_on_machine = 0
    visible_before = None

    def _pipeline_kwargs(llm_client=None, retriever=None) -> dict:
        kwargs = assemble_pipeline_kwargs(args, llm_client, retriever, workers, verbose, gpu_ids=gpu_ids)
        # ── Deep-research-pipeline-specific kwargs ────────────────────────
        kwargs.update({
            "stage": args.stage,
            "vllm_manager": vllm_manager,
            "total_gpus_on_machine": total_gpus_on_machine,
            "cuda_visible_devices_before": visible_before,
            "judge_model": args.judge_model,
            "query_ids_file": args.query_ids_file,
            "rerank_top_k": args.rerank_top_k,
            "retrieval_input": args.retrieval_input,
            "post_fusion_reranker_input": args.post_fusion_reranker_input,
            "post_retrieval_reranker_name": args.post_retrieval_reranker,
            "post_fusion_reranker_name": args.post_fusion_reranker,
        })
        return kwargs

    try:
        # ── Queries left to run, before any model is set up ──────────────
        pipeline_kwargs = _pipeline_kwargs()
        plan = prepare_run(
            data_path=args.data_path,
            output_path=args.output,
            policy_model=args.policy_model,
            subset=args.subset,
            dataset_year=args.dataset_year,
            query_key=args.query_key,
            limit=args.limit,
            **pipeline_kwargs,
        )

        # With nothing to generate (the belief or eval stage, or every query
        # generated) no vLLM server, LLM client, retriever or reranker is set
        # up.  The belief stage reads the documents from the corpus and calls
        # its own model (belief_workers threads); the LLM-as-judge evaluators
        # use the OpenRouter-hosted judge, so no GPU is needed for them.
        if not plan.queries and (args.workers != 1 or args.gpu_ids is not None):
            print("[Workers] Nothing to generate: --workers and --gpu-ids are ignored "
                  "(the belief stage runs --belief-workers threads)")
        if plan.queries:
            worker_plan = plan_generate_workers(args, vllm_manager, len(plan.queries))
            workers, gpu_ids = worker_plan.workers, worker_plan.gpu_ids
            total_gpus_on_machine, visible_before = worker_plan.total_gpus, worker_plan.visible_before
            if workers > 1:
                # Several workers: each builds its own LLM client, retriever and
                # rerankers on its assigned GPU (or the CPU).  Skip them in the
                # main process to avoid OOM on shared GPUs.
                pipeline_kwargs = _pipeline_kwargs()
            else:
                pipeline_kwargs = _pipeline_kwargs(setup_llm(args), setup_retriever_from_args(args))
                # ── Optional post-retrieval & post-fusion rerankers ─────────────────
                post_retrieval, post_fusion = build_rerankers(
                    args.post_retrieval_reranker, args.post_fusion_reranker, args.rerank_top_k,
                )
                if post_retrieval is not None:
                    pipeline_kwargs["post_retrieval_reranker"] = post_retrieval
                if post_fusion is not None:
                    pipeline_kwargs["post_fusion_reranker"] = post_fusion

        run_pipeline(
            plan,
            data_path=args.data_path,
            policy_model=args.policy_model,
            verbose=verbose,
            **pipeline_kwargs,
        )
        print("Pipeline execution completed successfully")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise
    finally:
        time.sleep(0.5)
        cleanup_event_loop()
        vllm_manager.shutdown()

# ============================================================================
# CLI arguments
# ============================================================================
def _none_if_null(value: str) -> Optional[str]:
    """Read the literal ``null`` on the CLI as None (auto-select), like the YAML."""
    return None if value == "null" else value


def parse_args():
    """Build the CLI argument parser and parse the command line.

    Only the frequently-varied knobs are declared as CLI arguments.  The
    mostly-fixed variables live in a YAML config file (``--config``, default
    experiments/configs/dra_inference.yaml) and are merged onto ``args`` in
    ``_resolve_args``; any of them can still be overridden by passing the
    matching ``--flag`` on the command line (returned in ``extras``).
    """
    parser = argparse.ArgumentParser(
        description="Run deep research agents pipeline (unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,  # only full flag names, e.g. --belief-source
    )

    # ── Stage ───────────────────────────────────────────────────────────────
    parser.add_argument("--stage", type=str, default="all", choices=list(STAGES), help="What to run. 'generate': the agent on the queries without a finished trajectory (trajectory/, retrieval/, generation/), then the evaluation of the generation (retrieval, generation, verbal confidence, trajectory). 'belief': the belief model --belief-model over the finished trajectories without a belief file of its own (belief/{model}/{qid}.jsonl and .md, one folder per belief model; no agent and no retrieval: the documents' text is read from the corpus by id), then the evaluation of that belief model, added to the run's evaluation files (the generation's rows and other belief models' rows kept); other settings (--belief-source, --max-beliefs) than the model's saved ones replace that model's beliefs only. 'eval': the evaluation only, from the saved files (every evaluator of the dataset, every belief model of the run; judge verdicts reused from eval/generation/, so an unchanged run makes no LLM call; the belief evaluation makes none). 'all': generate, then belief, then the evaluation of everything. Every stage writes eval/metrics.md and eval/per_query.csv, keeping the rows it does not score.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="Path to the YAML file holding the mostly-fixed pipeline variables. Any value in it can be overridden by passing the matching --flag on the CLI.")
    # ── Generation (the belief and eval stages need the same values to find the run directory) ──
    parser.add_argument("--dataset", type=str, default="browsecomp_plus", choices=list(DATASETS), help="Dataset; all use local indices.")
    parser.add_argument("--subset", type=_none_if_null, default=None, help="Dataset subset/collection (unset or null = the dataset's default in layout.DATASET_SPECS). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical; browsecomp_plus: test; ragtime: unused.")
    parser.add_argument("--policy-model", type=str, default="glm", choices=list(POLICY_MODEL_TO_LLM), help="Agent to run; the LLM is selected automatically from the agent. cpm_report = Writing-as-Reasoning (report generation); searchr1/research/stepsearch/react/selfask/searcho1 = Reasoning-augmented retrieval; glm/oss_20b/oss_120b/tongyi = vendor-specific ReAct agents.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever; its index must be built for --dataset.")
    parser.add_argument("--workers", type=int, default=1, help="Query-level worker processes of the generate stage; each loads its own retriever and agent. On a GPU machine one worker per GPU (capped at the GPUs, or at --gpu-ids); on a CPU machine CPU processes (the retriever on CPU, one copy in RAM each). 0 = auto: the GPU count, 1 without GPU. Ignored by the belief and eval stages (no GPU; the belief stage runs --belief-workers threads).")
    # ── Belief ──────────────────────────────────────────────────────────────
    parser.add_argument("--belief-model", type=str, default=DEFAULT_BELIEF_MODEL, choices=list(BELIEF_MODEL_TO_LLM), help="Belief model (its folder belief/{name}/ of the run, one per model); its LLM is selected from utils/config.py BELIEF_MODEL_TO_LLM: qwen3.5_4b (Qwen/Qwen3.5-4B on local vLLM: the belief stage starts its server on the first visible GPU, port 7010, and stops it after; needs a GPU), qwen3.5_9b (openrouter/qwen/qwen3.5-9b) or qwen3.6_27b (openrouter/qwen/qwen3.6-27b). One LLM for its three calls: belief init (banked under the LLM id), belief update and final score.")
    parser.add_argument("--belief-source", type=str, default=None, choices=list(BELIEF_SOURCES), help="Beliefs of the belief model: gold (TRQA: made from the intermediate info; trqa_val: the dataset rows) or silver (extracted by --belief-model). A choice only on TRQA and trqa_val; elsewhere the dataset's own source is used. Unset = belief_source in --config.")
    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="Cap the queries each stage runs (for quick tests): the generate stage's queries left to generate, the belief stage's trajectories left without a belief file. The evaluation still scores every finished query of the run (as with --query-ids-file).")
    parser.add_argument("--quiet", type=_sm_bool, nargs="?", const=True, default=False, help="Print minimal logs (overrides verbose)")

    args, extras = parser.parse_known_args()
    return args, extras


if __name__ == "__main__":
    main()


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
#   {output}/{dataset}_{split}[_{query_key}]_{retriever}/{agent}_{backend}_{model}/
#   ({output} = DRA_OUTPUT_ROOT; _{query_key} only when not the dataset's default)
#   e.g. run_outputs/neuclir_2024_news_qwen3_emb_4b/oss_20b_vllm_gpt-oss-20b/
#   ({agent} = the --policy-model name)
#
#   One home per fact: each file below holds its facts only.
#     ├── run_config.json       every resolved setting: dataset, agent, searcher (generate
#     │                         stage; a resume with other settings is refused), belief_models
#     │                         ({model: settings}, belief stage; other settings of a model
#     │                         replace that model's beliefs), evaluation
#     ├── retrieval/
#     │   ├── surfaced/{qid}.trec   surfaced-doc TREC (raw retriever output, col 6 = iter_N)
#     │   ├── seen/{qid}.trec       seen-doc TREC (docs shown to the LLM)
#     │   └── cited/{qid}.trec      cited-doc TREC (docs cited by the LLM)
#     ├── generation/{qid}.md   the final generation (with its References section)
#     ├── trajectory/
#     │   ├── {qid}.jsonl       one line per step (queries, seen doc ids, search_step of a
#     │   │                     search iteration), then the meta line (counts, agent data,
#     │   │                     agent_generation); written last: the meta line marks the
#     │   │                     query as generated
#     │   └── {qid}.md          the same trajectory, human-readable, written live
#     ├── belief/{model}/       belief stage (pipeline.belief_stage), one folder per
#     │   │                     belief model (--belief-model short name)
#     │   ├── {qid}.jsonl       meta line + one line per search iteration + final line
#     │   │                     (utils.run_files.belief); written last: marks the query as done
#     │   └── {qid}.md          the same belief, human-readable
#     └── eval/                 the evaluation (evaluation.runner): generate scores the
#         │                     generation's rows, belief its model's rows, eval and all every row
#         ├── metrics.md        one table: group | belief_model | metric | value | ci_low |
#         │                     ci_high | n (retrieval, generation, uncertainty, belief,
#         │                     trajectory); belief_model "-" on the rows of the generation
#         │                     (verbal confidence included), one row per belief model on
#         │                     the belief score and belief rows
#         ├── per_query.csv     every per-query value of every group (a belief model's
#         │                     columns prefixed "{model}/")
#         └── generation/       accuracy.jsonl: per-query answer correctness (datasets with
#                               answers); or Auto-ARGUE inputs, cached judgments and
#                               per-query scores.tsv (report datasets)
#
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#   python experiments/inference/dra_inference.py --stage generate --dataset browsecomp_plus --limit 1 --workers 1 
#   python experiments/inference/dra_inference.py --stage generate --dataset trqa --subset wiki2 --limit 3 --workers 1 --quiet
#   python experiments/inference/dra_inference.py --stage belief --dataset browsecomp_plus --limit 1 --workers 1 --belief_model qwen3.5_9b
#   python experiments/inference/dra_inference.py --stage belief --dataset trqa --subset wiki2 --limit 1
#   python experiments/inference/dra_inference.py --stage eval --dataset trqa --subset wiki2 --workers 1 --quiet

#   python experiments/inference/dra_inference.py --stage generate --dataset browsecomp_plus --limit 2 --workers 2 --quiet
#   python experiments/inference/dra_inference.py --stage generate --dataset trqa --subset wiki2 --limit 2 --workers 2 --quiet
#   python experiments/inference/dra_inference.py --stage belief --dataset browsecomp_plus --limit 2 --workers 2 --quiet
#   python experiments/inference/dra_inference.py --stage belief --dataset trqa --subset wiki2 --limit 1 --quiet
