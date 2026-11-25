"""The belief stage: the belief model over the saved trajectories of a run.

The generate stage runs the agent and saves each query's trajectory
(``trajectory/{qid}.jsonl``) and generation (``generation/{qid}.md``); the
belief model is not part of that run and never changes what the agent does.
This stage reads the saved files back (``belief_model_component.read_saved_query``:
per search iteration its queries, the documents it showed and the agent's
turn, then the answer turn) and gives them to the belief model, the
documents' text read from the corpus by id (``CorpusLookup``) and shaped as
the run's retriever returns a document (``belief_model_component.corpus_doc``;
nothing is retrieved again).

:func:`run_belief` is the stage of one belief model (``--belief-model``):
its settings in ``run_config.json``, its beliefs (``prepare_beliefs``), a
local vLLM server for a ``vllm/`` model, then :func:`run_belief_stage`.

Each query's belief goes to ``belief/{model}/{qid}.jsonl``, one folder per
belief model (``utils.run_files.save_belief``), and its readable view to
``belief/{model}/{qid}.md`` (``belief_model_component.views.log_view``).
A query whose belief file exists is done: the stage resumes.  A query
whose beliefs or a belief call failed before any reply (the API, not the
model) stops there and is not saved, so the next call runs it again; the
stage lists these queries at its end.  The queries are split round-robin
across parallel threads, each with its own belief model (its own client and
token meter, so each query's tokens are its own) and its own progress bar
(``utils.progress.WorkerBars``: the query and its step).
"""

import contextlib
import io
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from utils.io import atomic_write_text
from utils.progress import WorkerBars
from utils.run_files import belief_dir, believed_queries, save_belief, scan_queries
from utils.run_files.run_config import read_run_config

from .factories import build_belief_model
from .run_config import belief_config_diffs, build_belief_config, set_belief_config
from .run_plan import RunPlan

if TYPE_CHECKING:
    from belief_model_component import SavedQuery

logger = logging.getLogger(__name__)


def run_belief(plan: RunPlan, verbose: bool, kwargs: dict) -> None:
    """The belief stage of the belief model ``--belief-model``: the model
    over the run's finished trajectories without a belief file of its own
    (``belief/{model}/``); the other belief models of the run are left as
    they are.

    Its settings are its entry of the ``belief_models`` section of
    ``run_config.json`` (``build_belief_config``; the passage cut, retriever
    and agent are the generate stage's, read from the file; the corpus is
    this call's, of the run's dataset and split, ``check_run_dataset``);
    when they differ from the saved ones, the model's beliefs
    (``belief/{model}/``) are removed, then the
    new settings saved, and every query is run again.  A belief file of an
    older schema is made again too.  With ``--limit`` (``--stage all``),
    the queries this call generated come first."""
    run_dir = Path(plan.run_dir)
    name = kwargs["belief_model_name"]
    saved = read_run_config(run_dir)
    max_passage_tokens = saved["agent"]["max_passage_tokens"]
    section = build_belief_config(
        plan.dataset, max_passage_tokens=max_passage_tokens,
        **{k: kwargs[k] for k in ("belief_model", "max_beliefs", "belief_source")},
    )
    diffs = belief_config_diffs(run_dir, name, section)
    if diffs:
        print(f"\n[Belief] Other {name} settings than the run's ({'; '.join(diffs)}): removing its beliefs "
              f"({belief_dir(run_dir, name).relative_to(run_dir)}/) "
              "and running every query again")
        # Removed before the new settings are saved: an interrupted call
        # never leaves old beliefs under the new settings.
        shutil.rmtree(belief_dir(run_dir, name), ignore_errors=True)
    set_belief_config(run_dir, name, section)

    finished = scan_queries(run_dir)[0]
    done, stale = believed_queries(run_dir, name)
    todo = [qid for qid in plan.all_questions if qid in finished and qid not in done]
    todo.sort(key=lambda qid: qid not in plan.queries)   # this call's generated queries first
    if done:
        print(f"[Belief] Found {len(done & finished)} queries with a belief file — skipping them")
    if stale & finished:
        print(f"[Belief] {len(stale & finished)} belief files of an older schema: running those queries again")
    if plan.limit is not None and plan.limit > 0:
        todo = todo[:plan.limit]
    if not todo:
        print("[Belief] No finished trajectory without a belief file")
        return

    if not kwargs["belief_model"].startswith("vllm/"):
        _believe(plan, todo, saved, max_passage_tokens, verbose, kwargs)
        return
    # A locally served belief model: its own vLLM server for this stage only
    # (started after the generate stage released its GPUs), more queries at
    # once for belief init (the server batches them).
    from reasoner_component.vllm_server import require_spec, serve
    previous = os.environ.get("BELIEF_VLLM_API_BASE")
    with serve(require_spec(kwargs["belief_model"])) as api_base:
        os.environ["BELIEF_VLLM_API_BASE"] = api_base
        try:
            _believe(plan, todo, saved, max_passage_tokens, verbose, kwargs,
                     prepare_workers=kwargs["belief_workers"])
        finally:
            # The server stops with the block: no later call may reach for it.
            if previous is None:
                os.environ.pop("BELIEF_VLLM_API_BASE", None)
            else:
                os.environ["BELIEF_VLLM_API_BASE"] = previous


def _believe(plan: RunPlan, todo: List[str], saved: dict, max_passage_tokens: int, verbose: bool, kwargs: dict,
             prepare_workers: Optional[int] = None) -> None:
    """Make the beliefs of *todo* (``prepare_beliefs``, *prepare_workers*
    queries at once, None = its default), then run the belief model over
    their saved trajectories (:func:`run_belief_stage`)."""
    from belief_model_component import build_belief_source, prepare_beliefs

    run_dir = Path(plan.run_dir)
    name = kwargs["belief_model_name"]
    # Beliefs, made and banked before the workers start; the workers'
    # belief models share this source and read every query's from the bank.
    source = build_belief_source(plan.dataset, plan.split, belief_model=kwargs["belief_model"],
                                 max_beliefs=kwargs["max_beliefs"], belief_source=kwargs["belief_source"])
    questions = {qid: plan.all_questions[qid] for qid in todo}
    if prepare_workers:
        prepare_beliefs(source, questions, workers=prepare_workers)
    else:
        prepare_beliefs(source, questions)

    def build_model():
        return build_belief_model(
            dataset=plan.dataset, belief_model=kwargs["belief_model"], belief_source=source,
            qrels=plan.qrels, max_passage_tokens=max_passage_tokens,
        )

    print(f"\n[Belief] Running the belief model {name} on {len(todo)} queries, "
          f"saving to {belief_dir(run_dir, name)}/...")
    n_done = run_belief_stage(
        run_dir, name, todo, build_model=build_model, corpus_path=kwargs["corpus_path"],
        retriever_name=saved["searcher"]["retriever_name"], policy_model=saved["agent"]["policy_model"],
        workers=kwargs["belief_workers"], verbose=verbose,
    )
    print(f"[Belief] Completed {n_done}/{len(todo)} queries")


def _belief_md(summary: Dict[str, Any], steps: List[Dict[str, Any]], final: Optional[Dict[str, Any]],
               qid: str, question: str) -> str:
    """The readable belief of one query: its beliefs, the belief after each
    search iteration, then the final belief."""
    from belief_model_component import belief_md, init_md
    beliefs = summary.get("beliefs") or []
    parts = [f"# Belief · {qid}", "", question, "", init_md(summary)]
    parts += [belief_md(step, beliefs) for step in steps]
    if final is not None:
        parts.append(belief_md(final, beliefs, final=True))
    return "\n\n".join(parts) + "\n"


def _call_failed(record: Optional[Dict[str, Any]]) -> bool:
    """A record whose LLM call raised before any reply (the API or the
    server, not the model: a reply that did not parse is the model's)."""
    return bool(record) and bool(record.get("called")) and record.get("reply") is None


def _belief_of_query(belief_model, saved: "SavedQuery", docs_of: Callable[[List[str]], List[Dict[str, Any]]],
                     verbose: bool, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run *belief_model* over one saved query: reset, one observe per search
    iteration, finish on the answer turn (short-answer queries: the final
    update on its reasoning, then the final scorer on its answer; a report
    makes no call).  ``failed`` names what failed for
    a reason other than the model's reply (the beliefs, a call that raised
    before any reply, an exception); the query then stops there, is not
    saved, and the next belief stage runs it again.  *progress* gets the
    step being run: ``beliefs``, ``step k/N`` (N: the search iterations
    plus the final step), ``final``."""
    from belief_model_component import beliefs_call_failed, verbose_print_belief, verbose_print_init

    qid = saved.qid
    summary: Dict[str, Any] = {"beliefs": []}
    n_steps = len(saved.iterations) + 1

    def tick(text: str) -> None:
        if progress is not None:
            progress(text)

    def out(failed: Optional[str] = None) -> Dict[str, Any]:
        return {"meta": belief_model.meta(), "steps": list(belief_model.steps), "final": belief_model.final,
                "summary": summary, "failed": failed}

    tick("beliefs")
    try:
        belief_model.reset(query_id=qid, query=saved.question)
        summary = belief_model.init_summary()
    except Exception:
        logger.warning("Belief model reset failed for %s", qid, exc_info=True)
        return out("reset")
    if beliefs_call_failed(summary.get("belief_info") or {}):
        return out("beliefs")
    if verbose:
        verbose_print_init(summary, agent_name=f"Belief {qid}")
    beliefs = summary.get("beliefs") or []
    for k, it in enumerate(saved.iterations, 1):
        tick(f"step {k}/{n_steps}")
        try:
            record = belief_model.observe(
                subqueries=it.subqueries, docs=docs_of(it.doc_ids),
                iter_num=it.agent_iteration, agent_turn=it.agent_turn,
            )
        except Exception:
            logger.warning("Belief model failed at iteration %s of %s", it.agent_iteration, qid, exc_info=True)
            return out(f"iteration {it.agent_iteration}")
        if _call_failed(record):
            return out(f"belief update call at iteration {it.agent_iteration}")
        if verbose:
            verbose_print_belief(it.agent_iteration, record, beliefs, agent_name=f"Belief {qid}")
    tick(f"step {n_steps}/{n_steps}, final")
    n_before = len(belief_model.steps)
    try:
        belief_model.finish_answer(saved.answer_reasoning, saved.answer)
    except Exception:
        logger.warning("Belief model final update or score failed for %s", qid, exc_info=True)
        return out("final")
    final_update = belief_model.steps[n_before] if len(belief_model.steps) > n_before else None
    if _call_failed(final_update):
        return out("final update call")
    if _call_failed(belief_model.final):
        return out("final score call")
    if verbose and final_update is not None:
        verbose_print_belief(None, final_update, beliefs, agent_name=f"Belief {qid}")
    if verbose and belief_model.final is not None:
        verbose_print_belief(None, belief_model.final, beliefs, agent_name=f"Belief {qid}", final=True)
    return out()


def run_belief_stage(
    run_dir: Path,
    model_name: str,
    qids: List[str],
    *,
    build_model: Callable[[], Any],
    corpus_path: str,
    retriever_name: str,
    policy_model: str,
    workers: int,
    verbose: bool,
) -> int:
    """Run the belief model over the saved runs of *qids* and save each
    query's belief files; returns the number of queries done.

    Args:
        run_dir: The run directory (its ``trajectory/`` and ``generation/``).
        model_name: The belief model's short name (its folder ``belief/{model_name}/``).
        qids: The finished queries without a belief file.
        build_model: Builds one belief model (``pipeline.factories.build_belief_model``,
            on the run's shared belief source); one per worker, built here
            before the workers start (only the first prints its settings).
        corpus_path: The run's corpus JSONL (the documents' text).
        retriever_name: The run's retriever (``corpus_doc``).
        policy_model: The run's agent (``belief_model_component.NO_TURN_AGENTS``).
        workers: Belief threads (queries run at once), one bar each.
        verbose: Print each query's belief.
    """
    from belief_model_component import corpus_doc, read_saved_query
    from indexing_corpus_dataset.shared.doc_lookup import CorpusLookup

    run_dir = Path(run_dir)
    model_dir = belief_dir(run_dir, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    lookup = CorpusLookup(corpus_path)
    lookup.ensure_index()
    workers = max(1, min(workers, len(qids)))
    # Round-robin, as the generate stage splits its queries across workers.
    chunks = [qids[i::workers] for i in range(workers)]
    models = [build_model()]
    with contextlib.redirect_stdout(io.StringIO()):
        models += [build_model() for _ in range(workers - 1)]

    def docs_of(doc_ids: List[str]) -> List[Dict[str, Any]]:
        contents = lookup.contents(doc_ids)
        return [corpus_doc(i, contents.get(i, ""), retriever_name) for i in doc_ids]

    def run_one(model, qid: str, progress: Callable[[str], None]) -> Optional[str]:
        """None when the query's belief is saved, else what failed."""
        saved = read_saved_query(run_dir, qid, policy_model)
        if saved is None:
            return "no finished trajectory"
        out = _belief_of_query(model, saved, docs_of, verbose, progress)
        if out["failed"]:
            logger.warning("Belief of %s not saved (%s failed); the next belief stage runs it again",
                           qid, out["failed"])
            return out["failed"]
        # The .jsonl last: it marks the query as done.
        atomic_write_text(model_dir / f"{qid}.md",
                          _belief_md(out["summary"], out["steps"], out["final"], qid, saved.question))
        save_belief(run_dir, model_name, qid, saved.question, out["meta"], out["steps"], out["final"])
        return None

    # Verbose prints each query's belief, which live bars would scramble:
    # one line per finished query instead.
    bars = WorkerBars([len(c) for c in chunks], "Belief", live=False if verbose else None)
    not_saved: Dict[str, str] = {}

    def run_chunk(worker: int) -> None:
        for qid in chunks[worker]:
            bars.status(worker, f"{qid}, starting")
            try:
                failed = run_one(models[worker], qid, lambda text, q=qid: bars.status(worker, f"{q}, {text}"))
            except Exception as exc:
                logger.error("Belief stage failed for %s: %s", qid, exc, exc_info=True)
                failed = f"{type(exc).__name__}: {exc}"
            if failed:
                not_saved[qid] = failed
            bars.done(worker, qid, f"not saved: {failed}" if failed else "saved")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(run_chunk, i) for i in range(workers)]:
            future.result()
    bars.close()
    if not_saved:
        shown = sorted(not_saved)[:20]
        print(f"[Belief] {len(not_saved)} queries not saved (run again by the next belief stage): "
              + "; ".join(f"{q} ({not_saved[q]})" for q in shown) + (" ..." if len(not_saved) > 20 else ""))
    return len(qids) - len(not_saved)
