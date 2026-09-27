"""Prompt loading — adapts the finalized dataset loaders into PromptRecords.

READ-ONLY reuse of ``indexing_corpus_dataset`` (the same loaders the inference
pipeline uses).  A ``synthetic`` source is provided so the whole training loop
is runnable on CPU with no dataset / index present.
"""

from __future__ import annotations

import logging
import random
from typing import List

from .schema import PromptRecord
from ..config import DataConfig

logger = logging.getLogger(__name__)


def synthetic_prompts(n: int = 8) -> List[PromptRecord]:
    """Tiny in-memory dataset for smoke tests (no disk / index required)."""
    out: List[PromptRecord] = []
    for i in range(n):
        out.append(
            PromptRecord(
                id=f"syn-{i:03d}",
                question=f"synthetic question {i}: who won event number {i}?",
                answers=[f"answer-{i}"],
                qrels={f"doc-{i}-gold": 1},
                criteria=[f"identify entity {i}", f"verify date {i}"],
                meta={"synthetic": True, "dataset": "synthetic"},
            )
        )
    return out


def load_prompts(cfg: DataConfig, seed: int = 0) -> List[PromptRecord]:
    """Build the list of training prompts from ``cfg``.

    ``source == "synthetic"``  -> in-memory smoke dataset.
    ``source == "dataset"``    -> real queries/qrels/answers via the finalized
                                  ``indexing_corpus_dataset`` loaders, with the
                                  inference defaults for empty paths/splits.

    With ``cfg.shuffle`` the order is shuffled with *seed* before ``limit``.
    """
    if cfg.source == "synthetic":
        prompts = synthetic_prompts(cfg.num_synthetic)
    elif cfg.source == "dataset":
        prompts = _dataset_prompts(cfg)
    else:
        raise ValueError(f"Unknown DataConfig.source '{cfg.source}' (expected 'synthetic' or 'dataset')")

    if cfg.shuffle:
        random.Random(seed).shuffle(prompts)
    if cfg.limit:
        prompts = prompts[: cfg.limit]
    return prompts


def _dataset_prompts(cfg: DataConfig) -> List[PromptRecord]:
    # READ-ONLY reuse of the inference dataset loaders and defaults.
    from indexing_corpus_dataset.dataset_loaders import load_split, load_qrels, resolve_split_id
    from ..rollout.tool_env import resolve_data_args

    args = resolve_data_args(cfg)
    file_data_set = resolve_split_id(args.dataset, args.dataset_year, args.subset)
    qrels_path = args.qrels_data_path or args.data_path
    same_qrels = qrels_path == args.data_path

    queries, qrels, answers = load_split(
        args.data_path, file_data_set, query_key=args.query_key,
        min_relevance_score=args.min_relevance_score, only_with_qrels=same_qrels,
    )
    if not same_qrels:
        qrels = load_qrels(qrels_path, file_data_set, min_relevance_score=args.min_relevance_score)
        queries = {qid: q for qid, q in queries.items() if qid in qrels}

    prompts: List[PromptRecord] = []
    for qid, text in queries.items():
        ans = answers.get(qid) if answers else None
        if isinstance(ans, str):
            ans = [ans]
        prompts.append(
            PromptRecord(
                id=qid,
                question=text,
                answers=ans,
                qrels=qrels.get(qid) if qrels else None,
                criteria=None,
                meta={"dataset": args.dataset, "split": file_data_set},
            )
        )
    logger.info("Loaded %d training prompts from dataset '%s/%s'", len(prompts), args.dataset, file_data_set)
    return prompts
