#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for AgentCPM-Explore (port 6008).
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_cpm_explore.sh
#   PORT=8000 bash vllm_server_scripts/reasoners/serve_cpm_explore.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "openbmb/AgentCPM-Explore" "$@"
