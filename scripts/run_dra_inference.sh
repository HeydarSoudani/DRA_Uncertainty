#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=03:00:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/%x_%j.out
# GPU plan for THIS config: `uncertainty_aware` is listed in experiments/configs/openrouter_registry.yaml
# (qwen/qwen3.6-27b) and OPENROUTER_API_KEY is set in .env, so resolve_agent_backend()
# returns "api": NO local vLLM server is started and no GPU is reserved for the LLM.
# All 4 GPUs therefore become pipeline workers — one process per GPU, each holding its
# own Qwen3-Embedding-4B encoder (~14 GB) and mmap'ing the shared 57 GB FAISS index.
# The 50 queries are split round-robin over the 4 workers.
# SBU is billed on actual runtime, not --time.

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
DATASET=trqa                       # trqa | neuclir | browsecomp_plus
SUBSET=wiki2                       # trqa: wiki1|wiki2|ecommerce (eval split defaults to 'test')
RETRIEVER=qwen3_emb_4b
AGENT=uncertainty_aware          # glm | oss_20b | oss_120b | tongyi | react | cpm_report | ...
UNCERTAINTY_ESTIMATOR=off          # off | monitor
LIMIT=${LIMIT:-50}                 # overridable for a pre-flight, see the smoke line below
NUM_GPUS=${NUM_GPUS:-4}            # one query-level worker per GPU

python experiments/dra_inference.py \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --retriever "$RETRIEVER" \
    --agentic-model "$AGENT" \
    --uncertainty-estimator "$UNCERTAINTY_ESTIMATOR" \
    --limit "$LIMIT" \
    --num-gpus "$NUM_GPUS" \
    --quiet

# --quiet: silences the per-iteration agent logs, the HuggingFace "Loading weights" bars
# and the live per-worker tqdm bars (each redraw is appended verbatim to a non-TTY .out
# file — that alone was ~1 MB of the 2.3 MB GLM log).  What remains is the GPU plan, the
# worker split, one "[Worker i] n/m done" line per finished query, and the eval summary.
#
# Output: run_outputs/trqa_wiki2_test_qwen3_emb_4b/uncertainty_aware_api_qwen3.6-27b/ue-off/
# Resume is automatic: queries that already have retrieval/surfaced/{qid}.trec are skipped,
# and --limit is applied to what REMAINS, so a re-submit continues rather than restarts.
#
# PRE-FLIGHT (recommended before the full run): same code path, 1 query per worker,
# ~15 min of node time.  It exercises exactly what the full run does — worker spawn,
# retriever load, OpenRouter routing, per-query save, eval, summary.json:
#     sbatch --time=00:30:00 --export=ALL,LIMIT=4 scripts/run_dra_inference.sh
# Its 4 queries are real results and are kept: resume skips them, so the follow-up
# full run wants --export=ALL,LIMIT=46 to land on exactly 50 queries total.
#
# Smoke test: python experiments/dra_inference.py --dataset trqa --subset wiki2 --agentic-model uncertainty_aware --limit 1 --num-gpus 1
# Eval only:  python experiments/dra_inference.py --dataset trqa --subset wiki2 --agentic-model uncertainty_aware --eval-only --num-gpus 0
