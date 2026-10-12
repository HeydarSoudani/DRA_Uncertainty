#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=06:00:00
#SBATCH --mem=480GB
#SBATCH --output=script_logging/slurm_%A.out
# Sizing (H100, qwen3_emb_4b, 4 GPUs; --time is ~1.5x wall):
#   dataset          docs   max_length  batch  docs/s/GPU  wall    --time    peak RAM
#   browsecomp_plus  100K   4096        16     ~5.6        ~1.3 h  02:00:00
#   trqa             5.9M   512         32     ~131        ~3.3 h  05:00:00  ~120GB
#   ragtime          4.0M   1024        32     ~55         ~6 h    10:00:00  ~85GB
#   neuclir          10.0M  1024        32     ~55-61      ~14 h   24:00:00  ~206GB
# Other retrievers on trqa (5.9M docs, 4 GPUs; docs/s/GPU estimated from a 20K-doc
# H100 benchmark, scaled by the qwen3 benchmark-to-run ratio):
#   retriever        max_length  batch  docs/s/GPU  wall      --time
#   bge              512         512    ~1050       ~0.7 h    01:30:00
#   contriever       512         512    ~1700       ~0.5 h    01:00:00
#   spladepp         256         512    ~1300 (*)   ~3.5 h    06:00:00
#   (*) spladepp encodes in one process (DataParallel) and serialises each doc's
#   vector on CPU, so encoding is CPU-bound (~1 h). The Lucene impact index then
#   takes ~2.5 h (~650 docs/s): weights are quantized x100, each term is indexed
#   as many times as its weight, and more input files or threads do not help.

# torch / faiss / vllm live in the project venv on the Python/3.13.5 module
# stack. scripts/_activate.sh loads the modules, activates the venv, and exports the
# project env vars (PYTHONUNBUFFERED, HF_HOME, HF_DATASETS_CACHE, DRA_DATA_ROOT,
# ...) — same env as interactive runs. Sourced CWD-relative: sbatch preserves
# the submission dir (the repo root), which is why the paths below are relative.
source scripts/_activate.sh
mkdir -p script_logging

RETRIEVER=spladepp            # bm25 | spladepp | bge | qwen3_emb_4b
DATASET=trqa                  # trqa | neuclir | browsecomp_plus | ragtime

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
