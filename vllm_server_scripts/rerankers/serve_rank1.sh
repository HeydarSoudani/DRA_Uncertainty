#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Rank1-7B (JHU CLSP), a pointwise reranker that scores
# (query, passage) pairs via P(true)/P(false) logprobs on /v1/completions
# (port 8000).
#
# Usage:
#   bash vllm_server_scripts/rerankers/serve_rank1.sh
#   PORT=8001 bash vllm_server_scripts/rerankers/serve_rank1.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "jhu-clsp/rank1-7b" "$@"
