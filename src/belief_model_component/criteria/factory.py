"""Criteria factory of the belief model: the criteria source of each query.

The belief model makes its criteria list once per sample
(``BeliefModel.reset`` calls ``CriteriaSource.get``).  This module decides
which source that is, from the dataset and the run's ``criteria_source``;
the pipelines (dra_inference, belief model training through it,
``analysis/criteria_reachability.py``) only pass these settings.

Per query:

1. The kind of its criteria, gold or silver
   (``layout.effective_criteria_source``): the requested kind on a dataset
   with a choice (TRQA, trqa_validation), else gold.  A query without
   gold criteria gets silver ones.
2. Its criteria of that kind: read from the bank when there, else made,
   added to the bank and used (``BankedCriteriaSource``):

   - the extractor's criteria (``LLMCriteriaSource``): gold where they are
     the question's own (BrowseComp-Plus, NeuCLIR, RAGTIME), silver
     elsewhere (``layout.extraction_kind``); bank
     ``layout.criteria_bank_path``;
   - TRQA's gold criteria, made from the intermediate info
     (``indexing_corpus_dataset.training_datasets.gold.trqa``); bank
     ``layout.criteria_bank_path(..., "gold")``;
   - a training dataset's gold criteria (``GoldCriteriaSource``): the dataset's
     queries file is their bank, written by the dataset build only
     (``build_training_dataset.py``); a query without them gets silver ones.

Functions:

- ``check_max_criteria``: the cap is for silver criteria only.
- ``check_criteria_inputs``: before any model is loaded (no LLM call).
- ``build_criteria_source``: the source of a split, as each worker builds
  it.
- ``prepare_criteria``: makes the missing criteria of the run's queries
  before the workers start, so they read them from the bank.
- ``belief_llm``: the client of a belief model LLM.
"""

import json
import logging
import os
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from tqdm import tqdm

from indexing_corpus_dataset.layout import (
    ANNOTATION_GOLD, BUILT_DATASETS, DATASET_SPECS, GOLD_CRITERIA, SILVER_CRITERIA, criteria_bank_path,
    effective_criteria_source, extraction_kind, gold_criteria_path,
)
from reasoner_component import create_generator, disable_native_thinking

from .sources import (
    GOLD_CRITERIA_MODEL, BankedCriteriaSource, CriteriaSource, FallbackCriteriaSource, GoldCriteriaSource,
    LLMCriteriaSource,
)

logger = logging.getLogger(__name__)

# Queries whose criteria ``prepare_criteria`` makes at once (LLM calls).
PREPARE_WORKERS = 16


def belief_llm(model: str):
    """LLM client of a belief model call, native reasoning off.  A ``vllm/``
    model (e.g. a trained belief model) is served locally at
    ``$BELIEF_VLLM_API_BASE`` (default ``$VLLM_API_BASE``, so it can sit
    next to a locally served agent); any other model goes through the API."""
    if model.startswith("vllm/"):
        return disable_native_thinking(create_generator(model, api_base=os.getenv("BELIEF_VLLM_API_BASE")))
    return disable_native_thinking(create_generator(model, backend="api"))


def bank_model_id(model: str) -> str:
    """The name of *model* in the bank key.  A ``vllm/`` model is named
    with the checkpoint its server loaded (``{model}@{path}``, the ``root``
    of vLLM's ``/models``): every checkpoint of a trained belief model is
    served under the same name, and must not read another's criteria."""
    if not model.startswith("vllm/"):
        return model
    from reasoner_component.vllm import DEFAULT_VLLM_API_BASE

    base = (os.getenv("BELIEF_VLLM_API_BASE") or os.getenv("VLLM_API_BASE") or DEFAULT_VLLM_API_BASE).rstrip("/")
    bare = model.removeprefix("vllm/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=30) as reply:
            served = json.load(reply).get("data") or []
    except Exception as e:  # noqa: BLE001 - the key needs the checkpoint
        raise RuntimeError(f"criteria bank: cannot read the checkpoint of {model} from {base}/models ({e})") from e
    for entry in served:
        if entry.get("id") == bare:
            return f"{model}@{entry.get('root') or bare}"
    raise RuntimeError(f"criteria bank: {bare} is not served at {base} "
                       f"(served: {[e.get('id') for e in served]})")


def check_max_criteria(dataset: str, max_criteria: Optional[int]) -> None:
    """*max_criteria* (after the dataset's default) caps silver criteria
    only: TRQA's and trqa_validation's members, webshaper_frames' FRAMES hops.  Stops
    when it is set on a dataset whose criteria are all gold (BrowseComp-Plus,
    NeuCLIR, RAGTIME: the question's own, never capped)."""
    if max_criteria is not None and extraction_kind(dataset) != SILVER_CRITERIA:
        raise SystemExit(f"max_criteria {max_criteria}: the cap is for silver criteria only (TRQA, "
                         f"trqa_validation, webshaper_frames' FRAMES); the criteria of {dataset} are gold, never capped. "
                         "Set max_criteria to null")


def check_criteria_inputs(dataset: str, criteria_source: str, llm_criteria: Optional[str],
                          max_criteria: Optional[int] = None) -> None:
    """Check the criteria settings of a run of *dataset* before any model is
    loaded, and say where its criteria come from.  Stops when *max_criteria*
    is set on a dataset without silver criteria (``check_max_criteria``) or
    a training dataset's gold criteria file is missing."""
    check_max_criteria(dataset, max_criteria)
    spec = DATASET_SPECS[dataset]
    source = effective_criteria_source(criteria_source, dataset)
    if not spec.criteria_choice and criteria_source != source:
        print(f"[INFO] criteria_source {criteria_source} has no effect on {dataset}: its queries use their gold "
              f"criteria" + (", silver for a query without them" if spec.gold_criteria == ANNOTATION_GOLD else ""))
    if not llm_criteria:
        print("[INFO] criteria: none (no llm_criteria); the criteria-based signals are null")
        return
    if source != GOLD_CRITERIA or spec.gold_criteria != ANNOTATION_GOLD:
        print(f"[INFO] criteria: {extraction_kind(dataset)}, extracted from each query by {llm_criteria}")
    elif dataset in BUILT_DATASETS:
        path = gold_criteria_path(dataset)
        if not path.exists():
            raise SystemExit(f"criteria_source gold: {path} not found; build the dataset with "
                             f"experiments/training/build_training_dataset.py --dataset {dataset}"
                             + (", or run with --criteria-source silver" if spec.criteria_choice else ""))
        print(f"[INFO] criteria: gold from {path}, silver for a query without them (extracted by {llm_criteria})")
    else:
        print(f"[INFO] criteria: gold, made from the intermediate info (texts by {llm_criteria}), silver for a "
              f"query without them")


def build_criteria_source(
    dataset: str,
    split: str,
    *,
    llm_criteria: Optional[str],
    max_criteria: Optional[int],
    criteria_source: str = GOLD_CRITERIA,
    fallback: bool = True,
) -> Optional[CriteriaSource]:
    """The criteria source of the queries of *split* of *dataset* (see the
    module docstring); None without *llm_criteria*.

    *llm_criteria* extracts the criteria from the question and writes
    TRQA's gold criterion texts; *max_criteria* caps the extraction (None:
    uncapped).  *fallback* false: a query without gold criteria gets none
    (``analysis/criteria_reachability.py``)."""
    if not llm_criteria:
        logger.warning("No --llm-criteria model; criteria-based signals disabled")
        return None
    spec = DATASET_SPECS[dataset]
    kind = extraction_kind(dataset)
    llm = belief_llm(llm_criteria)
    model_id = bank_model_id(llm_criteria)
    bank = criteria_bank_path(dataset, split, kind)
    source: CriteriaSource = BankedCriteriaSource(
        LLMCriteriaSource(llm_client=llm, max_criteria=max_criteria, model_name=model_id,
                          query_shape=spec.query_shape, kind=kind),
        bank)
    cap = "uncapped" if max_criteria is None else f"at most {max_criteria}"
    print(f"Belief model: {kind} criteria ({cap}) from the bank {bank}, missing ones from {model_id}")
    if effective_criteria_source(criteria_source, dataset) != GOLD_CRITERIA or spec.gold_criteria != ANNOTATION_GOLD:
        return source

    if dataset in BUILT_DATASETS:
        path = gold_criteria_path(dataset)
        gold: CriteriaSource = GoldCriteriaSource(path, spec.query_shape)
        print(f"Belief model: gold criteria from {path}")
    else:
        from indexing_corpus_dataset.training_datasets.gold.trqa import TRQAGoldCriteria

        gold_bank = criteria_bank_path(dataset, split, GOLD_CRITERIA)
        gold = BankedCriteriaSource(TRQAGoldCriteria(split, writer=model_id, llm_client=llm), gold_bank)
        print(f"Belief model: gold criteria from the bank {gold_bank}, missing ones made from the "
              f"intermediate info (texts by {model_id})")
    if not fallback:
        return gold
    print(f"Belief model: a query without gold criteria gets the {kind} ones above")
    return FallbackCriteriaSource(gold, source)


def prepare_criteria(source: CriteriaSource, queries: Dict[str, str]) -> Dict[str, Any]:
    """Get the criteria of every query of *queries* (``{qid: text}``) from
    *source*, so the missing ones are made and banked here, before the
    workers start (they then read them from the bank; a query still
    missing is made by its worker).  Prints and returns the counts: per
    kind, per origin (``bank_hit``, ``bank_added``, ``dataset``: a
    training dataset's gold) and the queries ``failed`` (no criteria)."""
    source.prepare()

    def one(item):
        qid, query = item
        try:
            return qid, source.get(qid, query)
        except Exception as e:  # noqa: BLE001 - its worker tries again
            logger.warning("Criteria of %s failed", qid, exc_info=True)
            return qid, ([], {"errors": [f"criteria_source: {e}"]})

    counts: Counter = Counter()
    failed = []
    with ThreadPoolExecutor(max_workers=PREPARE_WORKERS) as pool:
        for qid, (criteria, info) in tqdm(pool.map(one, queries.items()), total=len(queries), desc="criteria"):
            if not criteria:
                failed.append(qid)
                logger.warning("No criteria for %s: %s", qid, info.get("errors"))
                continue
            counts[info.get("criteria_kind") or "unknown"] += 1
            origin = info.get("bank")
            counts["dataset" if origin is None and info.get("model") == GOLD_CRITERIA_MODEL else f"bank_{origin}"] += 1
            counts["gold_missing"] += "gold_missing" in info
    result = {**counts, "failed": failed}
    print(f"[INFO] criteria of {len(queries)} queries: {counts['gold']} gold, {counts['silver']} silver "
          f"({counts['gold_missing']} without gold), {len(failed)} without criteria; "
          f"{counts['bank_hit']} from the bank, {counts['bank_added']} made and added, "
          f"{counts['dataset']} from the dataset's gold")
    return result
