"""Belief init factory of the belief model: the belief source of each query.

The belief model makes its belief list once per sample
(``BeliefModel.reset`` calls ``BeliefSource.get``).  This module decides
which source that is, from the dataset and the run's ``belief_source``;
the callers (the belief stage of dra_inference,
``analysis/criteria_reachability.py``) only pass these settings.

Per query:

1. The kind of its beliefs, gold or silver
   (``layout.effective_belief_source``): the requested kind on a dataset
   with a choice (TRQA, trqa_val), else gold.  A query without
   gold beliefs gets silver ones.
2. Its beliefs of that kind: read from the bank when there, else made,
   added to the bank and used (``BankedBeliefSource``):

   - the extractor's beliefs (``LLMBeliefSource``): gold where they are
     the question's own (BrowseComp-Plus, NeuCLIR, RAGTIME), silver
     elsewhere (``layout.extraction_kind``); bank
     ``layout.belief_bank_path``;
   - TRQA's gold beliefs, made from the intermediate info
     (``trqa_gold.py``); bank
     ``layout.belief_bank_path(..., "gold")``;
   - a training dataset's gold beliefs (``GoldBeliefSource``): the dataset's
     queries file is their bank, written by the dataset build only
     (``train_belief_model.py --stage data``); a query without them gets silver ones.

Functions:

- ``check_max_beliefs``: the cap is for silver beliefs only.
- ``check_belief_inputs``: before any model is loaded (no LLM call).
- ``build_belief_source``: the source of a split, built once and shared
  by the belief stage's workers.
- ``prepare_beliefs``: makes the missing beliefs of the run's queries
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

from indexing_corpus_dataset.layout import (
    ANNOTATION_GOLD, BUILT_DATASETS, DATASET_SPECS, GOLD_BELIEFS, SILVER_BELIEFS, belief_bank_path,
    effective_belief_source, extraction_kind, gold_beliefs_path,
)
from reasoner_component import create_generator, disable_native_thinking
from utils.progress import step_iter

from .sources import (
    GOLD_BELIEF_MODEL, BankedBeliefSource, BeliefSource, FallbackBeliefSource, GoldBeliefSource,
    LLMBeliefSource,
)

logger = logging.getLogger(__name__)

# Queries whose beliefs ``prepare_beliefs`` makes at once (LLM calls).
PREPARE_WORKERS = 16


def belief_llm(model: str):
    """LLM client of a belief model call, native reasoning off.  A ``vllm/``
    model is served locally at ``$BELIEF_VLLM_API_BASE`` (set by the belief
    stage to the server it starts, ``utils.vllm_manager.belief_server``);
    any other model goes through the API, routed to the hosts of its
    provider pin (:func:`belief_provider_pin`)."""
    if model.startswith("vllm/"):
        return disable_native_thinking(create_generator(model, api_base=belief_vllm_api_base()))
    generator = create_generator(model, backend="api")
    pin = belief_provider_pin(model)
    client = getattr(generator, "_client", generator)
    if pin and isinstance(getattr(client, "config", None), dict):
        # Before the reasoning switch, which copies this body.
        client.config["extra_body"] = {**(client.config.get("extra_body") or {}), "provider": pin}
    return disable_native_thinking(generator)


def belief_provider_pin(model: str):
    """The OpenRouter ``provider`` routing of the belief model LLM *model*
    (``utils.config.BELIEF_OPENROUTER_PROVIDER``), None when it has none."""
    from utils.config import BELIEF_OPENROUTER_PROVIDER

    pin = BELIEF_OPENROUTER_PROVIDER.get(model)
    return dict(pin) if pin else None


def belief_vllm_api_base() -> str:
    """The endpoint of a ``vllm/`` belief model: ``$BELIEF_VLLM_API_BASE``,
    else ``$VLLM_API_BASE``, else the default local server."""
    from reasoner_component.vllm import DEFAULT_VLLM_API_BASE

    return (os.getenv("BELIEF_VLLM_API_BASE") or os.getenv("VLLM_API_BASE") or DEFAULT_VLLM_API_BASE).rstrip("/")


def bank_model_id(model: str) -> str:
    """The name of *model* in the bank key.  A ``vllm/`` model is named
    with the checkpoint its server loaded (``{model}@{path}``, the ``root``
    of vLLM's ``/models``): every checkpoint of a trained belief model is
    served under the same name, and must not read another's beliefs."""
    if not model.startswith("vllm/"):
        return model
    base = belief_vllm_api_base()
    bare = model.removeprefix("vllm/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=30) as reply:
            served = json.load(reply).get("data") or []
    except Exception as e:  # noqa: BLE001 - the key needs the checkpoint
        raise RuntimeError(f"belief bank: cannot read the checkpoint of {model} from {base}/models ({e}); "
                           "start its vLLM server ($BELIEF_VLLM_API_BASE) or use an API belief model") from e
    for entry in served:
        if entry.get("id") == bare:
            return f"{model}@{entry.get('root') or bare}"
    raise RuntimeError(f"belief bank: {bare} is not served at {base} "
                       f"(served: {[e.get('id') for e in served]})")


def check_max_beliefs(dataset: str, max_beliefs: Optional[int]) -> None:
    """*max_beliefs* (after the dataset's default) caps silver beliefs
    only: TRQA's and trqa_val's members.  Stops
    when it is set on a dataset whose beliefs are all gold (BrowseComp-Plus,
    NeuCLIR, RAGTIME: the question's own, never capped)."""
    if max_beliefs is not None and extraction_kind(dataset) != SILVER_BELIEFS:
        raise SystemExit(f"max_beliefs {max_beliefs}: the cap is for silver beliefs only (TRQA, "
                         f"trqa_val); the beliefs of {dataset} are gold, never capped. "
                         "Set max_beliefs to null")


def check_belief_inputs(dataset: str, belief_source: str, belief_model: Optional[str],
                        max_beliefs: Optional[int] = None) -> None:
    """Check the belief settings of a run of *dataset* before any model is
    loaded, and say where its beliefs come from.  Stops when there is no
    *belief_model* (it makes the beliefs), when *max_beliefs* is set on a
    dataset without silver beliefs (``check_max_beliefs``) or when a
    training dataset's gold belief file is missing."""
    if not belief_model:
        raise SystemExit("The belief stage needs a belief model: pass --belief-model")
    check_max_beliefs(dataset, max_beliefs)
    spec = DATASET_SPECS[dataset]
    source = effective_belief_source(belief_source, dataset)
    if not spec.belief_choice and belief_source != source:
        print(f"[INFO] belief_source {belief_source} has no effect on {dataset}: its queries use their gold "
              f"beliefs" + (", silver for a query without them" if spec.gold_beliefs == ANNOTATION_GOLD else ""))
    if source != GOLD_BELIEFS or spec.gold_beliefs != ANNOTATION_GOLD:
        print(f"[INFO] beliefs: {extraction_kind(dataset)}, extracted from each query by {belief_model}")
    elif dataset in BUILT_DATASETS:
        path = gold_beliefs_path(dataset)
        if not path.exists():
            raise SystemExit(f"belief_source gold: {path} not found; build the dataset with "
                             f"experiments/training/train_belief_model.py --stage data --dataset {dataset}"
                             + (", or run with --belief-source silver" if spec.belief_choice else ""))
        print(f"[INFO] beliefs: gold from {path}, silver for a query without them (extracted by {belief_model})")
    else:
        print(f"[INFO] beliefs: gold, made from the intermediate info (texts by {belief_model}), silver for a "
              f"query without them")


def build_belief_source(
    dataset: str,
    split: str,
    *,
    belief_model: str,
    max_beliefs: Optional[int],
    belief_source: str = GOLD_BELIEFS,
    fallback: bool = True,
) -> BeliefSource:
    """The belief source of the queries of *split* of *dataset* (see the
    module docstring).

    *belief_model* extracts the beliefs from the question and writes
    TRQA's gold belief texts; *max_beliefs* caps the extraction (None:
    uncapped).  *fallback* false: a query without gold beliefs gets none
    (``analysis/criteria_reachability.py``)."""
    if not belief_model:
        raise ValueError("the belief source needs a belief model")
    spec = DATASET_SPECS[dataset]
    kind = extraction_kind(dataset)
    llm = belief_llm(belief_model)
    model_id = bank_model_id(belief_model)
    bank = belief_bank_path(dataset, split, kind)
    source: BeliefSource = BankedBeliefSource(
        LLMBeliefSource(llm_client=llm, max_beliefs=max_beliefs, model_name=model_id,
                        query_shape=spec.query_shape, kind=kind),
        bank)
    cap = "uncapped" if max_beliefs is None else f"at most {max_beliefs}"
    print(f"Belief model: {kind} beliefs ({cap}) from the bank {bank}, missing ones from {model_id}")
    if effective_belief_source(belief_source, dataset) != GOLD_BELIEFS or spec.gold_beliefs != ANNOTATION_GOLD:
        return source

    if dataset in BUILT_DATASETS:
        path = gold_beliefs_path(dataset)
        gold: BeliefSource = GoldBeliefSource(path, spec.query_shape)
        print(f"Belief model: gold beliefs from {path}")
    else:
        from .trqa_gold import TRQAGoldBeliefs

        gold_bank = belief_bank_path(dataset, split, GOLD_BELIEFS)
        gold = BankedBeliefSource(TRQAGoldBeliefs(split, writer=model_id, llm_client=llm), gold_bank)
        print(f"Belief model: gold beliefs from the bank {gold_bank}, missing ones made from the "
              f"intermediate info (texts by {model_id})")
    if not fallback:
        return gold
    print(f"Belief model: a query without gold beliefs gets the {kind} ones above")
    return FallbackBeliefSource(gold, source)


def prepare_beliefs(source: BeliefSource, queries: Dict[str, str], workers: int = PREPARE_WORKERS) -> Dict[str, Any]:
    """Get the beliefs of every query of *queries* (``{qid: text}``) from
    *source*, so the missing ones are made and banked here, before the
    workers start (they then read them from the bank; a query still
    missing is made by its worker).  Prints and returns the counts: per
    kind, per origin (``bank_hit``, ``bank_added``, ``dataset``: a
    training dataset's gold) and the queries ``failed`` (no beliefs).
    *workers*: queries made at once (a locally served model takes more)."""
    source.prepare()

    def one(item):
        qid, query = item
        try:
            return qid, source.get(qid, query)
        except Exception as e:  # noqa: BLE001 - its worker tries again
            logger.warning("Belief init of %s failed", qid, exc_info=True)
            return qid, ([], {"errors": [f"belief_source: {e}"]})

    counts: Counter = Counter()
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for qid, (beliefs, info) in step_iter(pool.map(one, queries.items()), total=len(queries), desc="Beliefs", unit="query"):
            if not beliefs:
                failed.append(qid)
                logger.warning("No beliefs for %s: %s", qid, info.get("errors"))
                continue
            counts[info.get("belief_source") or "unknown"] += 1
            origin = info.get("bank")
            counts["dataset" if origin is None and info.get("model") == GOLD_BELIEF_MODEL else f"bank_{origin}"] += 1
            counts["gold_missing"] += "gold_missing" in info
    result = {**counts, "failed": failed}
    print(f"[INFO] beliefs of {len(queries)} queries: {counts['gold']} gold, {counts['silver']} silver "
          f"({counts['gold_missing']} without gold), {len(failed)} without beliefs; "
          f"{counts['bank_hit']} from the bank, {counts['bank_added']} made and added, "
          f"{counts['dataset']} from the dataset's gold")
    return result
