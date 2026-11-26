#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for DR-Tulu-8B (port 6008).
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_drtulu.sh
#   PORT=8000 bash vllm_server_scripts/reasoners/serve_drtulu.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "rl-research/DR-Tulu-8B" "$@"
