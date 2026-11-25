#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Qwen3-8B as a listwise reranker (port 8000).
#
# Usage:
#   bash vllm_server_scripts/rerankers/serve_qwen3_reranker.sh
#   PORT=8001 bash vllm_server_scripts/rerankers/serve_qwen3_reranker.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "Qwen/Qwen3-8B" "$@"
