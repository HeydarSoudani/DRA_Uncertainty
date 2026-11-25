#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for gpt-oss (port 6008).
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_gpt_oss.sh              # gpt-oss-20b
#   bash vllm_server_scripts/reasoners/serve_gpt_oss.sh 120b         # gpt-oss-120b
#   PORT=8000 bash vllm_server_scripts/reasoners/serve_gpt_oss.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

VARIANT="${1:-20b}"                         # "20b" or "120b"
shift $(( $# > 0 ? 1 : 0 ))
vllm_serve "openai/gpt-oss-${VARIANT}" "$@"
