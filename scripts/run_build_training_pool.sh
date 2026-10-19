#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=script_logging/slurm_%A.out
# Build a training pool, one step per job (CPU only; step 2 is network-bound).
#   sbatch --export=ALL,POOL=trqa_validation scripts/run_build_training_pool.sh   (no steps)
#   sbatch --export=ALL,STEP=1 scripts/run_build_training_pool.sh
#   sbatch --export=ALL,STEP=2 scripts/run_build_training_pool.sh
#   sbatch --export=ALL,STEP=2,LIMIT=200 scripts/run_build_training_pool.sh   (the next 200 questions)
#   sbatch --export=ALL,STEP=2,SOURCE=infoseek,LIMIT=200 scripts/run_build_training_pool.sh   (the next 200 InfoSeek ones)
#   sbatch --export=ALL,STEP=4 scripts/run_build_training_pool.sh   (sft / rl / val split)
# Step 2 reads the User-Agent contact from DRA_CONTACT in .env.
# Step 2 (~73k Wikipedia pages at 180/min, ~7 h, plus web pages) is
# incremental: a rerun skips the processed questions and retries the ones with
# transient failures.  Off a terminal it prints a status line every 1% / minute.

source scripts/_activate.sh
mkdir -p script_logging

POOL=${POOL:-deep_qa}
ARGS=(--pool "$POOL")
[ "$POOL" = deep_qa ] && ARGS+=(--step "${STEP:-1}")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$SOURCE" ] && ARGS+=(--source "$SOURCE")

python experiments/training/build_training_pool.py "${ARGS[@]}"
