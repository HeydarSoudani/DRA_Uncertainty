#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Rank-R1-7B (ielab), a setwise reranker: LoRA adapters
# on Qwen/Qwen2.5-7B-Instruct, served under the alias "rank-r1" (port 8001,
# the reranker's api_url in src/searcher_component/shared/config.py).
#
# Usage:
#   bash vllm_server_scripts/rerankers/serve_rank_r1.sh
#   PORT=8000 bash vllm_server_scripts/rerankers/serve_rank_r1.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "rank-r1" "$@"
