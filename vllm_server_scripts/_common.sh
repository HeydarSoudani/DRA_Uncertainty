# shellcheck shell=bash
# ──────────────────────────────────────────────────────────────────────────────
# Shared setup for the serve_*.sh scripts.
#
# Every server's settings live in one registry, src/reasoner_component/vllm_server/registry.py
# (SERVERS); these scripts only name a model and pass overrides through:
#
#   python -m reasoner_component.vllm_server list          # the registered models
#   python -m reasoner_component.vllm_server serve <model> [--port N] [--tp N]
#       [--max-model-len N] [--max-num-seqs N] [--dry-run] [any vllm flag ...]
#
# Source it near the top of each serve script, then call vllm_serve:
#   source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
#   vllm_serve "<model>" "$@"
#
# Overrides read from the environment (all optional):
#   PORT                  the server's port
#   TP_SIZE               tensor parallel size (default: fitted to the GPU memory)
#   CUDA_VISIBLE_DEVICES  the GPUs to use (default: the first TP_SIZE GPUs)
# ──────────────────────────────────────────────────────────────────────────────

# ── Cache / weights paths (env first; Snellius defaults) ──────────────────────
export HF_HOME="${HF_HOME:-/projects/0/prjs0834/heydars/.cache/huggingface}"
# Pre-staged weights (<DRA_MODELS_DIR>/<org>--<model>/) are served when present.
export DRA_MODELS_DIR="${DRA_MODELS_DIR:-/projects/0/prjs0834/heydars/DRA_training/models}"

# ── Make the project packages importable (as scripts/_activate.sh does) ───────
_DRA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_DRA_REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# ── vllm_serve <model> [extra args...]: run the registered server in the foreground
vllm_serve() {
    local model="$1"; shift
    exec python -m reasoner_component.vllm_server serve "$model" \
        ${PORT:+--port "$PORT"} \
        ${TP_SIZE:+--tp "$TP_SIZE"} \
        "$@"
}
