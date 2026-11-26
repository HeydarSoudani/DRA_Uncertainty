#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for a Qwen3-Reranker (pointwise yes/no reranker, port 8000).
#
# Unlike serve_qwen3_reranker.sh, which serves the general Qwen3-8B for
# *listwise* reranking, this serves the dedicated Qwen3-Reranker family.
#
# Usage:
#   bash vllm_server_scripts/rerankers/serve_qwen3_pointwise_reranker.sh           # 8B
#   MODEL_SIZE=4B bash vllm_server_scripts/rerankers/serve_qwen3_pointwise_reranker.sh
#   PORT=8001 MODEL_SIZE=0.6B bash vllm_server_scripts/rerankers/serve_qwen3_pointwise_reranker.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "Qwen/Qwen3-Reranker-${MODEL_SIZE:-8B}" "$@"
