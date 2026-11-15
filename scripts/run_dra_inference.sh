#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=01:15:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/%x_%j.out
# GPU plan for THIS config: `oss_20b` is listed in experiments/configs/openrouter_registry.yaml
# (openai/gpt-oss-20b) and OPENROUTER_API_KEY is set in .env, so resolve_agent_backend()
# returns "api": NO local vLLM server is started and no GPU is reserved for the LLM.
# All 4 GPUs therefore become generate workers, one process per GPU, each holding its
# own Qwen3-Embedding-4B encoder (~14 GB) and mmap'ing the shared FAISS index
# (BrowseComp-Plus: ~1 GB).  The queries are split round-robin over the 4 workers.
# SBU is billed on actual runtime, not --time.
#
# Time estimate, BrowseComp-Plus generate stage, 50 queries (from queries 769-772 of this run):
#   ~1.5-2.5 min/query (32-48 agent steps) -> 46 new queries / 4 workers ~ 25-35 min,
#   plus worker startup and the eval (LLM judge) ~ 5 min; --time 1:15 leaves margin
#   for long trajectories (max_iteration 100).  The belief stage needs no GPU: run it
#   afterwards on a CPU node with STAGE=belief (~7-10 min/query, 16 threads ~ 30-40 min).

# sbatch runs non-interactively; scripts/_activate.sh loads the Python/3.13.5 module
# stack, activates the project venv (torch/faiss/vllm), and exports the project
# env vars (HF_HOME, DRA_DATA_ROOT, DRA_OUTPUT_ROOT, ...) — same env as interactive.
# Sourced CWD-relative: sbatch preserves the submission dir (the repo root), which
# is also why the python/log paths below are relative.
source scripts/_activate.sh
mkdir -p script_logging

# 4 worker processes share the node's 64 cores.  Without this each process lets
# FAISS/OpenMP spawn 64 threads, so the flat-index scans oversubscribe the node 4x.
export OMP_NUM_THREADS=16

# DATASET + RETRIEVER must match the built index (see scripts/run_index_builder.sh).
DATASET=browsecomp_plus            # trqa | neuclir | browsecomp_plus | ragtime
SUBSET=                            # empty = the dataset's default (layout.DATASET_SPECS); trqa: wiki1|wiki2|ecommerce; neuclir: news|technical
RETRIEVER=qwen3_emb_4b
POLICY_MODEL=oss_20b               # glm | oss_20b | oss_120b | tongyi | react | cpm_report | ...
STAGE=${STAGE:-generate}                # generate | belief | eval | all (= generate, then belief)
LIMIT=${LIMIT:-48}                 # generate stage only: new queries to generate (4 already done -> 50 total)
WORKERS=${WORKERS:-4}              # generate stage: one query-level worker per GPU
BELIEF_WORKERS=${BELIEF_WORKERS:-16}  # belief stage: queries at once (threads, API calls only)

COMMON=(--dataset "$DATASET" ${SUBSET:+--subset "$SUBSET"} --retriever "$RETRIEVER"
        --policy-model "$POLICY_MODEL" --quiet)

# --limit caps each stage on its own, so `--stage all --limit 46` would leave one finished
# trajectory without a belief file.  `all` therefore runs two calls: generate with the
# limit, then belief over every finished trajectory without a belief file (no limit).
if [[ "$STAGE" == "generate" || "$STAGE" == "all" ]]; then
    python experiments/inference/dra_inference.py "${COMMON[@]}" \
        --stage generate --limit "$LIMIT" --workers "$WORKERS"
fi
if [[ "$STAGE" == "belief" || "$STAGE" == "all" ]]; then
    python experiments/inference/dra_inference.py "${COMMON[@]}" \
        --stage belief --belief-workers "$BELIEF_WORKERS"
fi
if [[ "$STAGE" == "eval" ]]; then
    python experiments/inference/dra_inference.py "${COMMON[@]}" --stage eval
fi

# --quiet: silences the per-iteration agent logs, the HuggingFace "Loading weights" bars
# and the live per-worker tqdm bars (each redraw is appended verbatim to a non-TTY .out
# file — that alone was ~1 MB of the 2.3 MB GLM log).  What remains is the GPU plan, the
# worker split, one "[Worker i] n/m done" line per finished query, and the eval summary.
#
# Output: run_outputs/browsecomp_plus_test_qwen3_emb_4b/oss_20b_api_gpt-oss-20b/
# Resume is automatic: the generate stage skips queries whose trajectory/{qid}.jsonl ends
# with its meta line, the belief stage those with belief/{qid}.jsonl, and --limit is
# applied to what REMAINS, so a re-submit continues rather than restarts.
#
# PRE-FLIGHT (optional): same code path, 1 query per worker, ~15 min of node time:
#     sbatch --time=00:30:00 --export=ALL,STAGE=generate,LIMIT=4 scripts/run_dra_inference.sh
# Its queries are kept (resume skips them); lower LIMIT of the full run by the same count.
#
# Smoke test: python experiments/inference/dra_inference.py --dataset browsecomp_plus --policy-model oss_20b --limit 1 --workers 1
# Belief:     python experiments/inference/dra_inference.py --dataset browsecomp_plus --policy-model oss_20b --stage belief
# Eval only:  python experiments/inference/dra_inference.py --dataset browsecomp_plus --policy-model oss_20b --stage eval
