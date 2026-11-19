#!/bin/bash
#SBATCH --job-name=train_belief_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/%x_%j.out
# Belief model training, one phase per job (experiments/training/train_belief_model.py).
# The resources above suit phase 4 (one GPU shared by vLLM and the trainer).
# Override them per phase on the sbatch line:
#   phase 1 (CPU; infoseek reads its 18.5 GB corpus):
#     sbatch --partition=rome --gpus=0 --time=12:00:00 --mem=32GB \
#       --export=ALL,PHASE=1,DATASET=infoseek,LIMIT=500 scripts/run_train_belief_model.sh
#     sbatch --partition=rome --gpus=0 --export=ALL,PHASE=1,DATASET=webshaper_frames,SOURCE=frames,LIMIT=50 scripts/run_train_belief_model.sh
#   phase 2 (agents on OpenRouter, teacher on OpenRouter; the retriever wants a GPU):
#     sbatch --export=ALL,PHASE=2,DATASET=infoseek,SUBSET=sft,LIMIT=50 scripts/run_train_belief_model.sh
#     sbatch --export=ALL,PHASE=2,DATASET=infoseek,SUBSET=rl,POLICY_MODEL=glm scripts/run_train_belief_model.sh
#   phase 3 (data parallel over the job's GPUs):
#     sbatch --gpus=4 --export=ALL,PHASE=3 scripts/run_train_belief_model.sh
#   phase 4:
#     sbatch --export=ALL,PHASE=4,BELIEFS=gold scripts/run_train_belief_model.sh
#   phase 5 (a trained model is served on local vLLM: 1 GPU):
#     sbatch --time=04:00:00 --export=ALL,PHASE=5,BELIEF_MODEL=qwen3.5_4b_rl,DATASET=infoseek,SUBSET=val scripts/run_train_belief_model.sh
# SET: extra config overrides, space-separated, e.g. SET="rl.group_size=4 rl.max_chain_steps=40".

source scripts/_activate.sh
mkdir -p script_logging

ARGS=(--phase "${PHASE:?set PHASE=1..5}")
[ -n "$DATASET" ] && ARGS+=(--dataset "$DATASET")
[ -n "$SUBSET" ] && ARGS+=(--subset "$SUBSET")
[ -n "$POLICY_MODEL" ] && ARGS+=(--policy-model "$POLICY_MODEL")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$SOURCE" ] && ARGS+=(--source "$SOURCE")
[ -n "$BELIEF_MODEL" ] && ARGS+=(--belief-model "$BELIEF_MODEL")
[ -n "$BELIEFS" ] && ARGS+=(--beliefs "$BELIEFS")
for kv in $SET; do ARGS+=(--set "$kv"); done

if [ "$PHASE" = "3" ]; then
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    torchrun --standalone --nproc_per_node "${NGPU:-1}" experiments/training/train_belief_model.py "${ARGS[@]}"
else
    python experiments/training/train_belief_model.py "${ARGS[@]}"
fi
