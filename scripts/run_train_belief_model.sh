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
# Belief model training, one stage per job (experiments/training/train_belief_model.py).
# The resources above suit stage rl (one GPU shared by vLLM and the trainer).
# Override them per stage on the sbatch line:
#   data_prepare (CPU, no LLM; infoseek_hard and frames read their 18.5 GB corpus once for new titles):
#     sbatch --partition=rome --gpus=0 --time=04:00:00 --mem=32GB \
#       --export=ALL,STAGE=data_prepare,DATASET=infoseek_hard scripts/run_train_belief_model.sh
#     (trqa_val and frames the same; webshaper also fetches its web gold pages: DRA_CONTACT in .env,
#      and a rerun fetches the pages still pending)
#   data_relevance (CPU; the judge and the writers on OpenRouter):
#     sbatch --partition=rome --gpus=0 --time=12:00:00 --mem=32GB \
#       --export=ALL,STAGE=data_relevance,DATASET=infoseek_hard,LIMIT=500 scripts/run_train_belief_model.sh
#   generate (agents on OpenRouter; the retriever wants a GPU):
#     sbatch --export=ALL,STAGE=generate,DATASET=infoseek_hard,SUBSET=sft,LIMIT=50 scripts/run_train_belief_model.sh
#     sbatch --export=ALL,STAGE=generate,DATASET=infoseek_hard,SUBSET=rl,POLICY_MODEL=glm scripts/run_train_belief_model.sh
#   teacher (teacher on OpenRouter; CPU, documents read from the corpus by id):
#     sbatch --partition=rome --gpus=0 --export=ALL,STAGE=teacher,DATASET=infoseek_hard,SUBSET=sft scripts/run_train_belief_model.sh
#   filter (CPU):
#     sbatch --partition=rome --gpus=0 --time=02:00:00 --mem=32GB --export=ALL,STAGE=filter scripts/run_train_belief_model.sh
#   sft (data parallel over the job's GPUs):
#     sbatch --gpus=4 --export=ALL,STAGE=sft scripts/run_train_belief_model.sh
#   rl:
#     sbatch --export=ALL,STAGE=rl scripts/run_train_belief_model.sh
#     sbatch --export=ALL,STAGE=rl,BELIEFS=own scripts/run_train_belief_model.sh
#   evaluate (a trained model is served on local vLLM: 1 GPU):
#     sbatch --time=04:00:00 --export=ALL,STAGE=evaluate,BELIEF_MODEL=qwen3.5_4b_rl,DATASET=infoseek_hard,SUBSET=val scripts/run_train_belief_model.sh

source scripts/_activate.sh
mkdir -p script_logging

ARGS=(--stage "${STAGE:?set STAGE=data_prepare|data_relevance|generate|teacher|filter|sft|rl|evaluate}")
[ -n "$DATASET" ] && ARGS+=(--dataset "$DATASET")
[ -n "$SUBSET" ] && ARGS+=(--subset "$SUBSET")
[ -n "$POLICY_MODEL" ] && ARGS+=(--policy-model "$POLICY_MODEL")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ -n "$TEACHER" ] && ARGS+=(--teacher "$TEACHER")
[ -n "$STUDENT" ] && ARGS+=(--student "$STUDENT")
[ -n "$INIT_MODEL" ] && ARGS+=(--init-model "$INIT_MODEL")
[ -n "$BELIEFS" ] && ARGS+=(--beliefs "$BELIEFS")
[ -n "$BELIEF_MODEL" ] && ARGS+=(--belief-model "$BELIEF_MODEL")

if [ "$STAGE" = "sft" ]; then
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    torchrun --standalone --nproc_per_node "${NGPU:-1}" experiments/training/train_belief_model.py "${ARGS[@]}"
else
    python experiments/training/train_belief_model.py "${ARGS[@]}"
fi
