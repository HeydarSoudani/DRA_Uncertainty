#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=script_logging/slurm_%A.out
# Build a training dataset, one call per job (CPU only).
#   sbatch --export=ALL,DATASET=infoseek,LIMIT=200 scripts/run_build_training_dataset.sh
#   sbatch --export=ALL,DATASET=webshaper_frames,LIMIT=200 scripts/run_build_training_dataset.sh
#   sbatch --export=ALL,DATASET=webshaper_frames,SOURCE=frames,LIMIT=50 scripts/run_build_training_dataset.sh
#   sbatch --export=ALL,DATASET=trqa_validation scripts/run_build_training_dataset.sh
# LIMIT: the next N questions not processed yet (0 = rewrite the files only;
# unset = all).  infoseek downloads its corpus (18.5 GB) into
# DATA_ROOT/infoseek/corpus on its first build and reads it once per build
# with new gold titles; webshaper_frames reads the User-Agent contact from
# DRA_CONTACT in .env and is network-bound (Wikipedia at 180 requests/min).
# Every build is incremental and ends with the split; one build of a dataset
# runs at a time (lock file).  Off a terminal it prints a status line every
# 1% / minute.

source scripts/_activate.sh
mkdir -p script_logging

ARGS=(--dataset "${DATASET:?set DATASET=infoseek|webshaper_frames|trqa_validation}")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$SOURCE" ] && ARGS+=(--source "$SOURCE")

python experiments/training/build_training_dataset.py "${ARGS[@]}"
