#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Tongyi-DeepResearch-30B-A3B (port 6008, hermes tool
# parser).  Also the server of WebWeaver when it runs on the local model.
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_tongyi.sh
#   PORT=8000 bash vllm_server_scripts/reasoners/serve_tongyi.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B" "$@"
