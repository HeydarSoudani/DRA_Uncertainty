"""The SFT samples (stage filter): the teacher's calls on the sft (train) and
val (validation) splits, as (messages, reply) pairs.

**Update and final score.**  The teacher's beliefs of an episode
(``belief/{teacher}/{qid}.jsonl`` of the generated run, written by stage
teacher) are replayed through
the belief model (``chain.run_chain`` with a ``ReplayClient``): each call
gets the teacher's saved reply, so the prompts are the belief model's own,
byte for byte.  The replay is checked against the saved file (the same
calls, the same belief after every step, the same s_T); an episode whose
replay differs gives no sample (``replay_mismatch``).  A call is a sample
when it passes the format gate: an update call that parsed with no issue
of ``GATE_ISSUES``, a final score that parsed (``reward``).  Then the
filter rule (``filtering.rules.keep_sample``).

**Init.**  On the datasets of ``filter.sft_init_datasets``, one sample per query
of the split: the init prompt of the dataset (its shape and cap) and the
gold beliefs as the reply, in the prompt example's layout.  A query with
more gold beliefs than the prompt's cap gives none.

Each sample: ``{set, task, dataset, split, policy_model, qid, step,
messages, completion, y}`` (``set``: train or val; ``task``: init, update
or score).
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from utils.io_utils import write_jsonl

from ..chain import INIT, SCORE, UPDATE, ReplayClient, init_source, run_chain
from ..config import RunPaths, TrainingConfig
from ..reward import score_probability, step_passes
from ..runs import DocumentStore, Episode, read_gold_rows
from .rules import keep_sample

#: The split of each sample set.
SETS = {"train": "sft", "val": "val"}


def _same(a: Any, b: Any) -> bool:
    from utils.run_files.belief import clean_record

    return json.dumps(clean_record(a), sort_keys=True) == json.dumps(clean_record(b), sort_keys=True)


def replay_teacher(episode: Episode, store: DocumentStore, teacher: str) -> Tuple[List[Dict[str, Any]], str]:
    """The samples of one episode and its outcome (``ok`` or why it gave none)."""
    from utils.run_files import belief_dir
    from utils.run_files.belief import SCHEMA_VERSION, read_belief_file

    meta, steps, final = read_belief_file(belief_dir(episode.run_dir, teacher) / f"{episode.qid}.jsonl")
    if meta is None:
        return [], "no_teacher_belief"
    if meta.get("schema_version") != SCHEMA_VERSION:
        return [], "teacher_belief_old_schema"
    replies = {(UPDATE, s["iteration"]): s["reply"] for s in steps if s.get("called") and s.get("reply") is not None}
    if final is not None and final.get("reply") is not None:
        replies[(SCORE, final["iteration"])] = final["reply"]
    client = ReplayClient(replies)
    result = run_chain(episode, client, store)
    if result.error is not None or client.unexpected or len(result.steps) != len(steps):
        return [], "replay_mismatch"
    for got, saved in zip(result.steps, steps):
        if got["called"] != saved.get("called") or not _same(got["beliefs"], saved.get("beliefs")):
            return [], "replay_mismatch"
    if (final or {}).get("certainty") != (result.final or {}).get("certainty"):
        return [], "replay_mismatch"

    base = {"dataset": episode.dataset, "split": episode.split, "policy_model": episode.policy_model,
            "qid": episode.qid, "y": episode.y}
    samples = []
    for call in client.calls:
        if call.attempt != 1:
            continue
        if call.kind == UPDATE:
            passed = step_passes(result.steps[call.step - 1])
        elif call.kind == SCORE:
            passed = score_probability(call.reply) is not None
        else:
            continue
        if passed:
            samples.append({**base, "task": call.kind, "step": call.step, "messages": call.messages,
                            "completion": call.reply})
        else:
            samples.append({**base, "task": call.kind, "gate_failed": True})
    return samples, "ok"


def init_completion(beliefs: List[Dict[str, Any]]) -> str:
    """The gold beliefs as the init reply, in the layout of the init
    prompt's example (one belief per line)."""
    items = ",\n".join("    " + json.dumps({"id": c["id"], "text": c["text"]}, ensure_ascii=False) for c in beliefs)
    return "{\n  \"beliefs\": [\n" + items + "\n  ]\n}"


def init_samples(dataset: str, split: str, counts: Counter) -> List[Dict[str, Any]]:
    from indexing_corpus_dataset.layout import DATASET_SPECS

    cap = DATASET_SPECS[dataset].max_beliefs
    source = init_source(dataset, None)
    out = []
    for qid, row in read_gold_rows(dataset).items():
        if row.get("split") != split:
            continue
        if cap is not None and len(row["beliefs"]) > cap:
            counts["init_over_cap"] += 1
            continue
        out.append({"dataset": dataset, "split": split, "policy_model": None, "qid": qid, "y": None,
                    "task": INIT, "step": None, "messages": source.messages(row["text"]),
                    "completion": init_completion(row["beliefs"])})
    return out


def build_samples(cfg: TrainingConfig, paths: RunPaths,
                  episodes: Dict[str, List[Episode]]) -> Dict[str, Dict[str, int]]:
    """``filter/sft_samples.jsonl``: the samples of the train and val sets
    from *episodes* (by split; the teacher ``shared.teacher``, replayed on
    ``filter.workers`` threads); returns the counts of each set."""
    teacher = cfg.shared.teacher
    store = DocumentStore()
    rows: List[Dict[str, Any]] = []
    all_counts: Dict[str, Dict[str, int]] = {}
    for set_name, split in SETS.items():
        counts: Counter = Counter()
        candidates = []
        with ThreadPoolExecutor(max_workers=cfg.filter.workers) as pool:
            replayed = list(pool.map(lambda ep: replay_teacher(ep, store, teacher), episodes[split]))
        for samples, outcome in replayed:
            counts[f"episodes_{outcome}"] += 1
            for s in samples:
                if s.pop("gate_failed", False):
                    counts[f"{s['task']}_gate_failed"] += 1
                else:
                    candidates.append(s)
        for dataset in cfg.filter.sft_init_datasets:
            candidates += init_samples(dataset, split, counts)
        for s in candidates:
            if keep_sample(s):
                counts[f"{s['task']}_samples"] += 1
                rows.append({"set": set_name, **s})
            else:
                counts[f"{s['task']}_filtered"] += 1
        y_counts = Counter(s["y"] for s in rows if s["set"] == set_name and s["task"] == SCORE)
        counts["score_samples_y1"] = y_counts.get(True, 0)
        counts["score_samples_y0"] = y_counts.get(False, 0)
        all_counts[set_name] = dict(sorted(counts.items()))
        print(f"[stage filter] sft {set_name} ({split}): {all_counts[set_name]}")
    paths.filter_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.sft_samples, rows)
    print(f"[stage filter] wrote {len(rows)} SFT samples to {paths.sft_samples}")
    return all_counts
