#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for one of the Qwen2.5-7B search agents (port 6008).
#
# Usage:
#   bash vllm_server_scripts/reasoners/serve_qwen7b_search_agents.sh searchr1     # default
#   bash vllm_server_scripts/reasoners/serve_qwen7b_search_agents.sh research
#   bash vllm_server_scripts/reasoners/serve_qwen7b_search_agents.sh stepsearch
#
# Settings: the model's spec in src/reasoner_component/vllm_server/registry.py.  PORT, TP_SIZE
# and CUDA_VISIBLE_DEVICES override them; extra arguments go to vllm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

AGENT="${1:-searchr1}"
shift $(( $# > 0 ? 1 : 0 ))
case "$AGENT" in
    searchr1)   MODEL="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" ;;
    research)   MODEL="agentrl/ReSearch-Qwen-7B-Instruct" ;;
    stepsearch) MODEL="Zill1/StepSearch-7B-Instruct" ;;
    *) echo "unknown agent: $AGENT (searchr1 | research | stepsearch)" >&2; exit 2 ;;
esac
vllm_serve "$MODEL" "$@"
