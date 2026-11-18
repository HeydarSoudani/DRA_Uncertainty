#!/bin/bash
#SBATCH --job-name=run_dra_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=01:15:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/%x_%j.out

# Generate-stage time (N = queries, 4 workers):
#   | agent   | dataset         | total time        | measured on        |
#   |---------|-----------------|-------------------|--------------------|
#   | oss_20b | browsecomp_plus | N x 0.9 + 3 min   | 48 queries, 42 min |

# Loads modules, venv and project env vars; run from the repo root (paths are relative).
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

