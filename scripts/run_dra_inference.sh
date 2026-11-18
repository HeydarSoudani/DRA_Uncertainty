#!/bin/bash
#SBATCH --job-name=run_dra_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpu_h100
#SBATCH --time=01:30:00
#SBATCH --mem=360GB
#SBATCH --output=script_logging/%x_%j.out

# Generate-stage time (N = queries, 3 workers; per-query rates scaled 4/3 from 4-worker runs):
#   | agent   | dataset         | total time        | measured on (4 workers) |
#   |---------|-----------------|-------------------|-------------------------|
#   | oss_20b | browsecomp_plus | N x 1.2 + 3 min   | 48 queries, 42 min      |
#   | glm     | browsecomp_plus | N x 0.87 + 3 min  | 1 query, 3.1 min        |
#   | oss_20b | trqa wiki2      | N x 0.8 + 3 min   | 1 query, 3.0 min        |
#   | glm     | trqa wiki2      | N x 0.2 + 3 min   | 1 query, 1.1 min        |

# Loads modules, venv and project env vars; run from the repo root (paths are relative).
source scripts/_activate.sh
mkdir -p script_logging

# 3 worker processes share the job's 48 cores.  Without this each process lets
# FAISS/OpenMP spawn one thread per core, so the flat-index scans oversubscribe the job 3x.
export OMP_NUM_THREADS=16

# DATASET + RETRIEVER must match the built index (see scripts/run_index_builder.sh).
DATASET=trqa                       # trqa | neuclir | browsecomp_plus | ragtime
SUBSET=wiki2                       # empty = the dataset's default (layout.DATASET_SPECS); trqa: wiki1|wiki2|ecommerce; neuclir: news|technical
RETRIEVER=qwen3_emb_4b
POLICY_MODEL=oss_20b               # glm | oss_20b | oss_120b | tongyi | react | cpm_report | ...
STAGE=${STAGE:-generate}                # generate | belief | eval | all (= generate, then belief)
LIMIT=${LIMIT:-49}                 # generate stage only: new queries to generate (1 already done -> 50 total)
WORKERS=${WORKERS:-3}              # generate stage: one query-level worker per GPU
# Belief model (each has its own folder belief/{model}/ in the run, the others are kept):
#   qwen3.5_4b   Qwen/Qwen3.5-4B, local vLLM: the belief stage starts its server on GPU 0 (port 7010)
#                and stops it after; a belief-only job needs 1 GPU
#   qwen3.5_9b   OpenRouter (or scripts/run_dra_inference_cpu.sh)
#   qwen3.6_27b  OpenRouter (or scripts/run_dra_inference_cpu.sh)
BELIEF_MODEL=${BELIEF_MODEL:-qwen3.6_27b}
if [[ "$BELIEF_MODEL" == "qwen3.5_4b" ]]; then
    BELIEF_WORKERS=${BELIEF_WORKERS:-64}  # belief stage: queries at once; the local server batches them
else
    BELIEF_WORKERS=${BELIEF_WORKERS:-16}  # belief stage: queries at once (threads, API calls)
fi

COMMON=(--dataset "$DATASET" ${SUBSET:+--subset "$SUBSET"} --retriever "$RETRIEVER"
        --policy-model "$POLICY_MODEL" --quiet)

if [[ "$STAGE" == "generate" || "$STAGE" == "all" ]]; then
    python experiments/inference/dra_inference.py "${COMMON[@]}" \
        --stage generate --limit "$LIMIT" --workers "$WORKERS"
fi
if [[ "$STAGE" == "belief" || "$STAGE" == "all" ]]; then
    python experiments/inference/dra_inference.py "${COMMON[@]}" \
        --stage belief --belief-model "$BELIEF_MODEL" --belief-workers "$BELIEF_WORKERS"
fi
if [[ "$STAGE" == "eval" ]]; then
    python experiments/inference/dra_inference.py "${COMMON[@]}" --stage eval
fi

