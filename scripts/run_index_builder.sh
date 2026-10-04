#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/slurm_%A.out
# Sizing (H100, qwen3_emb_4b, 4 GPUs; --time is ~1.5x wall):
#   dataset          docs   max_length  batch  docs/s/GPU  wall    --time    peak RAM
#   browsecomp_plus  100K   4096        16     ~5.6        ~1.3 h  02:00:00
#   trqa             5.9M   512         32     ~131        ~3.3 h  05:00:00  ~120GB
#   ragtime          4.0M   1024        32     ~55         ~6 h    10:00:00  ~85GB
#   neuclir          10.0M  1024        32     ~55-61      ~14 h   24:00:00  ~206GB

# torch / faiss / vllm live in the project venv on the Python/3.13.5 module
# stack. scripts/_activate.sh loads the modules, activates the venv, and exports the
# project env vars (PYTHONUNBUFFERED, HF_HOME, HF_DATASETS_CACHE, DRA_DATA_ROOT,
# ...) — same env as interactive runs. Sourced CWD-relative: sbatch preserves
# the submission dir (the repo root), which is why the paths below are relative.
source scripts/_activate.sh
mkdir -p script_logging

RETRIEVER=qwen3_emb_4b        # bm25 | spladepp | bge | qwen3_emb_4b
DATASET=neuclir               # trqa | neuclir | browsecomp_plus | ragtime

case "$RETRIEVER" in
    bm25)              ARGS=() ;;
    spladepp|spladev3) ARGS=(--use_fp16 --max_length 256 --batch_size 512 --save_embedding) ;;
    # qwen3 embedding models (4B): max_length is auto-selected per dataset
    # (layout.DATASET_SPECS: browsecomp_plus 4096, neuclir/ragtime
    # 1024, trqa 512). batch_size is corpus-dependent (measured on H100, see
    # sizing notes above): long-doc corpora want a small batch (compute-bound);
    # short-doc corpora too (padding-bound, smaller batch = less wasted compute).
    qwen3_emb_*)
        case "$DATASET" in
            browsecomp_plus) ARGS=(--use_fp16 --batch_size 16 --faiss_type Flat --save_embedding) ;;
            *)               ARGS=(--use_fp16 --batch_size 32 --faiss_type Flat --save_embedding) ;;
        esac
        ;;
    *)                 ARGS=(--use_fp16 --max_length 512 --batch_size 512 --faiss_type Flat --save_embedding) ;;
esac

python -m indexing_corpus_dataset.index_builder \
    --retriever "$RETRIEVER" \
    --dataset "$DATASET" \
    "${ARGS[@]}"
