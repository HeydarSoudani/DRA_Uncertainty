#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/%x_%j.out
# One phase of belief model training per job (experiments/training/belief_model_training.py):
#   sbatch --export=ALL,PHASE=1,SPLIT=val scripts/run_belief_model_training.sh
#   sbatch --export=ALL,PHASE=1,SPLIT=sft,EXTRA="--limit 200" scripts/run_belief_model_training.sh
#   sbatch --export=ALL,PHASE=1,SPLIT=rl,EXTRA="--limit 500" scripts/run_belief_model_training.sh
#   sbatch --export=ALL,PHASE=2,POLICY_MODEL=glm scripts/run_belief_model_training.sh
#
# GPU use per phase:
#   1  dra_inference workers on all 4 GPUs (agent and teacher via OpenRouter);
#      sft and val in monitor mode, rl in off mode
#   2  torchrun, data parallel on all 4 GPUs
#   3  1 GPU, shared by the vLLM rollout engine and the trainer
#
# Settings: experiments/configs/belief_model_training.yaml; anything in EXTRA
# is passed through, e.g. EXTRA="--seed 1".

source scripts/_activate.sh
mkdir -p script_logging
export OMP_NUM_THREADS=16

PHASE=${PHASE:-1}
NUM_GPUS=${NUM_GPUS:-4}
TRAINING_DATA=${TRAINING_DATA:-trqa_validation}
POLICY_MODEL=${POLICY_MODEL:-glm}

ARGS=(--phase "$PHASE" --training-data "$TRAINING_DATA" --policy-model "$POLICY_MODEL")

if [[ "$PHASE" == "2" ]]; then
    torchrun --standalone --nproc_per_node "$NUM_GPUS" experiments/training/belief_model_training.py "${ARGS[@]}" ${EXTRA:-}
elif [[ "$PHASE" == "3" ]]; then
    CUDA_VISIBLE_DEVICES=${RL_GPU:-0} python experiments/training/belief_model_training.py "${ARGS[@]}" ${EXTRA:-}
else
    python experiments/training/belief_model_training.py "${ARGS[@]}" --split "${SPLIT:-sft}" --num-gpus "$NUM_GPUS" ${EXTRA:-}
fi
