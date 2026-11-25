"""Stage filter: what the training stages train on (CPU only).

Writes under ``{BELIEF_TRAINING_ROOT}/filter/``, for every student:

* ``sft_samples.jsonl``: the SFT samples of the train (sft) and val sets,
  from the teacher's replies (``filtering.samples``, the teacher
  ``shared.teacher`` or --teacher; its stage teacher must have run);
* ``rl_episodes.jsonl``: the RL episodes of the train (rl) and val sets,
  one row per episode ``{set, dataset, split, policy_model, qid, y}``
  (``filtering.rules.episode_drop_reason`` on the train set);
* ``stats.json``: the teacher, the pool (training datasets, agents,
  retriever), the ``filter`` settings and the counts of each set (kept,
  left out by reason).

Each split's episodes are loaded once for both.  Stage sft reads the
samples, stage rl the episodes (``load_selected_episodes``); both stop when
the pool of ``stats.json`` is not the config's (``read_filter_stats``) and
record the stats in their run.  Each call rewrites the three files.
"""

import json
from collections import Counter
from typing import Any, Dict, List

from utils.io import read_jsonl, write_jsonl

from ..shared.config import RunPaths, TrainingConfig
from ..shared.runs import Episode, load_episodes
from .rules import episode_drop_reason
from .samples import SETS, build_samples

#: The split of each RL episode set.
RL_SETS = {"train": "rl", "val": "val"}


def _pool(cfg: TrainingConfig) -> Dict[str, Any]:
    """The episode pool of *cfg*: what the filtered files were drawn from."""
    sh = cfg.shared
    return {"training_datasets": list(sh.training_datasets), "policy_models": list(sh.policy_models),
            "retriever": sh.retriever}


def select_rl_episodes(cfg: TrainingConfig, paths: RunPaths,
                       episodes: Dict[str, List[Episode]]) -> Dict[str, Dict[str, int]]:
    """``filter/rl_episodes.jsonl``: the RL episodes of the train and val
    sets from *episodes* (by split); returns the counts of each set."""
    rows: List[Dict[str, Any]] = []
    all_counts: Dict[str, Dict[str, int]] = {}
    for set_name, split in RL_SETS.items():
        counts: Counter = Counter()
        for ep in episodes[split]:
            reason = episode_drop_reason(ep, cfg.filter.rl_max_chain_steps) if set_name == "train" else None
            if reason is not None:
                counts[reason] += 1
                continue
            counts["episodes"] += 1
            counts["episodes_y1" if ep.y else "episodes_y0"] += 1
            rows.append({"set": set_name, "dataset": ep.dataset, "split": ep.split,
                         "policy_model": ep.policy_model, "qid": ep.qid, "y": ep.y})
        all_counts[set_name] = dict(sorted(counts.items()))
        print(f"[stage filter] rl {set_name} ({split}): {all_counts[set_name]}")
    paths.filter_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.rl_episodes, rows)
    print(f"[stage filter] wrote {len(rows)} RL episodes to {paths.rl_episodes}")
    return all_counts


def run_filter(cfg: TrainingConfig) -> None:
    sh = cfg.shared
    paths = RunPaths()
    settings = cfg.to_dict()["filter"]
    print(f"[stage filter] teacher {sh.teacher}; settings {json.dumps(settings)}")
    splits = sorted(set(SETS.values()) | set(RL_SETS.values()))
    episodes = {split: load_episodes(cfg.inference_output, sh.training_datasets, split, sh.policy_models,
                                     sh.retriever, tag="[stage filter]")
                for split in splits}
    sft_counts = build_samples(cfg, paths, episodes)
    rl_counts = select_rl_episodes(cfg, paths, episodes)
    stats = {"teacher": sh.teacher, **_pool(cfg), "filter": settings, "sft": sft_counts, "rl": rl_counts}
    paths.filter_stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[stage filter] counts in {paths.filter_stats}")


def read_filter_stats(cfg: TrainingConfig, paths: RunPaths, tag: str) -> Dict[str, Any]:
    """``filter/stats.json``; stops when stage filter has not run, or ran
    on another pool than *cfg*'s (training datasets, agents, retriever)."""
    if not paths.filter_stats.exists():
        raise SystemExit(f"{tag} {paths.filter_stats} not found: run --stage filter first")
    stats = json.loads(paths.filter_stats.read_text())
    pool = _pool(cfg)
    changed = {k: (stats.get(k), v) for k, v in pool.items() if stats.get(k) != v}
    if changed:
        details = "; ".join(f"{k}: filter {old}, config {new}" for k, (old, new) in changed.items())
        raise SystemExit(f"{tag} the filtered files are of another pool ({details}): run --stage filter again")
    return stats


def load_selected_episodes(cfg: TrainingConfig, paths: RunPaths, set_name: str, tag: str) -> List[Episode]:
    """The RL episodes of *set_name* (train or val) that stage filter kept;
    the ones whose run no longer has them, or whose answer was judged again
    since (another y), are reported and left out."""
    wanted = {f"{r['dataset']}/{r['split']}/{r['policy_model']}/{r['qid']}": r["y"]
              for r in read_jsonl(paths.rl_episodes) if r["set"] == set_name}
    sh = cfg.shared
    loaded = [ep for ep in load_episodes(cfg.inference_output, sh.training_datasets, RL_SETS[set_name],
                                         sh.policy_models, sh.retriever, tag=tag)
              if ep.key in wanted]
    episodes = [ep for ep in loaded if ep.y == wanted[ep.key]]
    missing, rejudged = len(wanted) - len(loaded), len(loaded) - len(episodes)
    if missing or rejudged:
        print(f"{tag} left out {missing} filtered episodes no longer in their runs and {rejudged} judged again "
              "since; run --stage filter again")
    return episodes
