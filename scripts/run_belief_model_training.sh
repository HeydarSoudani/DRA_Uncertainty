#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/%x_%j.out
# One phase of belief model training per job (experiments/belief_model_training.py):
#   sbatch --export=ALL,PHASE=1 scripts/run_belief_model_training.sh
#
# GPU use per phase:
#   1  dra_inference workers on all 4 GPUs (agent and teacher via OpenRouter)
#   2  torchrun, data parallel on all 4 GPUs
#   3  1 GPU, shared by the vLLM rollout engine and the trainer (the others are
#      free for a local reward judge, TODO)
#   4  the student vLLM server on GPU 0, dra_inference workers on GPUs 1-3
#
# Settings: experiments/configs/belief_model_training.yaml; anything in EXTRA
# is passed through, e.g. EXTRA="--set rl.num_iterations=50".

source scripts/_activate.sh
mkdir -p script_logging
export OMP_NUM_THREADS=16

PHASE=${PHASE:-1}
RUN_NAME=${RUN_NAME:-default}
NUM_GPUS=${NUM_GPUS:-4}

ARGS=(--phase "$PHASE" --run-name "$RUN_NAME")

if [[ "$PHASE" == "2" ]]; then
    torchrun --standalone --nproc_per_node "$NUM_GPUS" experiments/belief_model_training.py "${ARGS[@]}" ${EXTRA:-}
elif [[ "$PHASE" == "3" ]]; then
    CUDA_VISIBLE_DEVICES=${RL_GPU:-0} python experiments/belief_model_training.py "${ARGS[@]}" ${EXTRA:-}
else
    python experiments/belief_model_training.py "${ARGS[@]}" --num-gpus "$NUM_GPUS" ${EXTRA:-}
fi
