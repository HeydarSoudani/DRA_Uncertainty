#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for GLM-4.7-Flash (port 6008, glm47 tool parser).
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_glm.sh
#   PORT=8000 bash vllm_server_scripts/reasoners/serve_glm.sh
#   bash vllm_server_scripts/reasoners/serve_glm.sh --pipeline-parallel-size 2
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "zai-org/GLM-4.7-Flash" "$@"
