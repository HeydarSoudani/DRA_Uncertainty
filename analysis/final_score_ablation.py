"""Final score ablation: does the belief help the final scorer judge the
agent's answer?

Rescores saved monitor runs of a short-answer dataset without rerunning
the agent.  Per query, the belief before the answer turn (b_{T-1}) is
rebuilt by replaying the run's saved belief update replies
(``BeliefUpdateSignal`` with a client that returns them; every replayed
belief must equal the saved one, else the query is left out), then s_T is
computed four ways against the correctness y of the agent's answer
(``accuracy.jsonl``):

- ``scorer``: the final scorer (``belief_model_component.update.final_score``)
  on b_{T-1} and the answer turn, as at inference;
- ``scorer_no_belief``: the same prompt with the criteria only (no
  candidate, support, proposal or removed name): the query and the answer;
- ``formula``: the fixed function of the statuses of b_{T-1}
  (``core.certainty.certainty``);
- ``answer_formula``: its form anchored on the scorer's answer's candidates
  (``core.certainty.answer_certainty``).

Measures per variant (the queries every variant scored): AUROC, Kendall's
tau-b, Brier score and ECE (10 bins), as in the belief model training
scores.  Runs of schema 37 (a final update instead of the scorer) are
replayed up to their last step; the scorer is always called anew.

Usage::

    python analysis/final_score_ablation.py --run-dir RUN [--run-dir RUN2 ...]
        [--model openrouter/qwen/qwen3.6-27b] [--subset SUBSET] [--limit N] [--out-dir DIR]

``--model`` defaults to the run's ``belief_judge_model``; ``--subset``
(the corpus subset the run read, not recorded in its ``run_config``) to
the dataset's default subset, e.g. ``--subset ecommerce`` for a TRQA
ecommerce run.  Outputs in
``--out-dir`` (default ``RUN/final_score_ablation`` for one run):
``scores.jsonl`` (per query: y, each variant's s_T and label, the replies)
and ``summary.json`` (the measures).
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from belief_model_component import (  # noqa: E402
    BeliefState, BeliefUpdateSignal, Criterion, answer_certainty, belief_llm, build_belief_updater,
    build_final_scorer, certainty,
)
from belief_model_training.runs import read_correctness  # noqa: E402
from belief_model_training.scoring import _final_measures  # noqa: E402
from evaluation.belief import clean_record, passage_tokens  # noqa: E402
from indexing_corpus_dataset.doc_lookup import CorpusLookup  # noqa: E402
from indexing_corpus_dataset.layout import DATASET_SPECS, default_corpus_path  # noqa: E402
from utils.io_utils import read_jsonl  # noqa: E402

logger = logging.getLogger(__name__)

VARIANTS = ("scorer", "scorer_no_belief", "formula", "answer_formula")


class _SavedReplies:
    """An LLM client that returns the run's saved replies in order (a saved
    call that raised raises again)."""

    def __init__(self) -> None:
        self.replies: List[Optional[str]] = []

    def complete(self, messages, **kwargs) -> str:
        reply = self.replies.pop(0)
        if reply is None:
            raise RuntimeError("the saved call failed")
        return reply


def _canonical(value: Any) -> str:
    return json.dumps(clean_record(value), sort_keys=True, ensure_ascii=False)


def replay_belief(lines: List[Dict[str, Any]], shape: str, max_tokens: Optional[int],
                  corpus: CorpusLookup) -> BeliefState:
    """b_{T-1} of one saved belief file (its meta and step lines), rebuilt
    from the saved replies; raises on a step whose belief differs."""
    meta = lines[0]
    steps = [r for r in lines if r.get("record") == "step"]
    llm = _SavedReplies()
    signal = BeliefUpdateSignal(build_belief_updater(llm, "saved", shape, max_passage_tokens=max_tokens))
    signal.reset(meta["question"], [Criterion(**c) for c in meta["criteria"]])
    texts = corpus.contents({d["doc_id"] for r in steps for d in r.get("docs") or []})
    for r in steps:
        if r.get("called"):
            llm.replies.append(r.get("reply"))
        docs = [{"doc_id": d["doc_id"], "relevant_text": texts.get(d["doc_id"], "")} for d in r.get("docs") or []]
        out = signal.step(docs, r["iteration"], r.get("agent_turn"))
        if _canonical(out["beliefs"]) != _canonical(r.get("beliefs")):
            raise ValueError(f"iteration {r['iteration']}: replayed beliefs differ from the saved ones")
    return signal.state


def _score(scorer, query: str, state: BeliefState, answer_turn: str) -> Dict[str, Any]:
    score, raw, errors = scorer.score(query, state, answer_turn)
    return {"s": score.probability if score else None, "label": score.status if score else None,
            "candidates": score.candidates if score else [], "reply": raw, "errors": errors}


def ablate_run(run_dir: Path, model: Optional[str], limit: Optional[int],
               subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """The per-query rows of one run."""
    config = json.loads((run_dir / "run_config.json").read_text())
    dataset = config["dataset"]["dataset"]
    spec = DATASET_SPECS[dataset]
    if spec.task != "qa":
        raise SystemExit(f"{run_dir}: {dataset} is not a short-answer dataset")
    shape = spec.query_shape
    try:
        max_tokens = passage_tokens(config)
    except ValueError as exc:
        raise SystemExit(f"{run_dir}: {exc}") from None
    model = model or config["belief_model"].get("belief_judge_model")
    scorer = build_final_scorer(belief_llm(model), model=model, query_shape=shape)
    corpus = CorpusLookup(default_corpus_path(dataset, subset or spec.subset))
    correct = read_correctness(run_dir)
    rows = []
    paths = sorted((run_dir / "belief").glob("*.jsonl"))
    for path in paths[:limit] if limit else paths:
        lines = read_jsonl(path)
        meta = lines[0]
        final = next((r for r in lines if r.get("record") == "final"), None)
        qid = str(meta["query_id"])
        y = correct.get(qid)
        if final is None or not final.get("agent_turn") or y is None:
            logger.info("%s: no answer turn or no correctness; skipped", qid)
            continue
        try:
            state = replay_belief(lines, shape, max_tokens, corpus)
        except Exception as e:
            logger.warning("%s: replay failed: %s; skipped", qid, e)
            continue
        kinds = [c.kind for c in state.criteria]
        beliefs = state.belief_lists()
        with_belief = _score(scorer, meta["question"], state, final["agent_turn"])
        without = _score(scorer, meta["question"], BeliefState(state.criteria), final["agent_turn"])
        rows.append({
            "run": str(run_dir), "query_id": qid, "y": y,
            "scorer": with_belief["s"], "scorer_label": with_belief["label"],
            "scorer_no_belief": without["s"], "scorer_no_belief_label": without["label"],
            "formula": certainty(beliefs, kinds),
            "answer_formula": answer_certainty(beliefs, kinds, with_belief["candidates"], state.ruled_out_lists())
            if with_belief["s"] is not None
            else None,
            "answer_candidates": with_belief["candidates"],
            "replies": {"scorer": with_belief["reply"], "scorer_no_belief": without["reply"]},
            "errors": with_belief["errors"] + without["errors"],
        })
        print(f"{qid}: y={int(y)} " + " ".join(f"{v}={rows[-1][v]:.3f}" if rows[-1][v] is not None else f"{v}=null"
                                               for v in VARIANTS), flush=True)
    return rows


def summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The measures of every variant on the queries all of them scored."""
    full = [r for r in rows if all(r[v] is not None for v in VARIANTS)]
    y = np.array([float(r["y"]) for r in full])
    out: Dict[str, Any] = {"queries": len(rows), "scored_by_all": len(full), "correct_rate": float(y.mean())
                           if len(y) else math.nan}
    for v in VARIANTS:
        out.update(_final_measures(v, np.array([r[v] for r in full], dtype=float), y))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--model", default=None, help="final scorer model (default: the run's belief judge)")
    parser.add_argument("--subset", default=None, help="corpus subset of the runs (default: the dataset's)")
    parser.add_argument("--limit", type=int, default=None, help="queries per run")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = args.out_dir or (args.run_dir[0] / "final_score_ablation" if len(args.run_dir) == 1 else None)
    if out_dir is None:
        raise SystemExit("--out-dir is needed with several runs")
    rows = [row for run in args.run_dir for row in ablate_run(run, args.model, args.limit, args.subset)]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(clean_record(r), ensure_ascii=False) + "\n")
    result = summary(rows)
    (out_dir / "summary.json").write_text(json.dumps(clean_record(result), indent=2) + "\n")
    print(json.dumps(clean_record(result), indent=2))


if __name__ == "__main__":
    main()
