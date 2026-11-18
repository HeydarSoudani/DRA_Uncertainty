#!/bin/bash
#SBATCH --job-name=run_dra_inference_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=staging
#SBATCH --time=02:00:00
#SBATCH --mem=112GB
#SBATCH --output=script_logging/%x_%j.out

# CPU-only stages of dra_inference: belief (replays saved trajectories, API calls only)
# and eval. The generate stage needs GPUs: use scripts/run_dra_inference.sh for it.
# Memory = CPUS x 7 GB (Snellius bills more memory as extra CPUs).

# Loads modules, venv and project env vars; run from the repo root (paths are relative).
source scripts/_activate.sh
mkdir -p script_logging

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Must match the generate run whose trajectories are replayed.
DATASET=browsecomp_plus            # trqa | neuclir | browsecomp_plus | ragtime
SUBSET=                            # empty = the dataset's default (layout.DATASET_SPECS); trqa: wiki1|wiki2|ecommerce; neuclir: news|technical
RETRIEVER=qwen3_emb_4b
POLICY_MODEL=oss_20b               # glm | oss_20b | oss_120b | tongyi | react | cpm_report | ...
STAGE=${STAGE:-belief}             # belief | eval
BELIEF_WORKERS=${BELIEF_WORKERS:-4}  # belief stage: queries at once (threads, API calls only)
LIMIT=${LIMIT:-48}                   # belief stage only: trajectories left without a belief file to process; empty = all

COMMON=(--dataset "$DATASET" ${SUBSET:+--subset "$SUBSET"} --retriever "$RETRIEVER"
        --policy-model "$POLICY_MODEL" --quiet)

case "$STAGE" in
    belief)
        python experiments/inference/dra_inference.py "${COMMON[@]}" \
            --stage belief --belief-workers "$BELIEF_WORKERS" ${LIMIT:+--limit "$LIMIT"} ;;
    eval)
        python experiments/inference/dra_inference.py "${COMMON[@]}" --stage eval ;;
    *)
        echo "[run_dra_inference_cpu] STAGE must be belief or eval (got '$STAGE'); generate needs GPUs" >&2
        exit 1 ;;
esac
