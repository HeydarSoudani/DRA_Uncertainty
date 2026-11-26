#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for RankZephyr-7B, a listwise reranker (port 8000).
#
# Usage:
#   bash vllm_server_scripts/rerankers/serve_rankzephyr.sh
#   PORT=8001 bash vllm_server_scripts/rerankers/serve_rankzephyr.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "castorini/rank_zephyr_7b_v1_full" "$@"
