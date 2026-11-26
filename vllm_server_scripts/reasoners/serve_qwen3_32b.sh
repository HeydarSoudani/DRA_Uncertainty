#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Qwen3-32B as an accuracy-evaluation judge (port 6009).
#
# The pipeline's judge runs on OpenRouter (openrouter/qwen/qwen3-32b); this
# server is for running the judge locally by hand.  Parameters match the
# AgentIR evaluation setup (texttron/AgentIR evaluation/evaluate_bcp.py).
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_qwen3_32b.sh
#   PORT=6010 MAX_MODEL_LEN=32768 bash vllm_server_scripts/reasoners/serve_qwen3_32b.sh
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

vllm_serve "Qwen/Qwen3-32B" \
    ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
    ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
    "$@"
