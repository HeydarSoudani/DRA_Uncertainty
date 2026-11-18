"""The belief stage: the belief model over the saved trajectories of a run.

The generate stage runs the agent and saves each query's trajectory
(``trajectory/{qid}.jsonl``) and generation (``generation/{qid}.md``); the
belief model is not part of that run and never changes what the agent does.
This stage reads the saved files back and gives the belief model, per query,
what the agent's turns showed:

* per search iteration (the trajectory lines that share a ``search_step``,
  ``BasicAgent._record_iteration``): its queries, the documents it showed
  (the lines' ``seen_docs``, in order; their text is read from the corpus by
  id, ``CorpusLookup``, and shaped as the run's retriever returns a
  document, ``corpus_doc``; nothing is retrieved again) and the agent's
  turn (``agent_turn_text`` of the lines' reasoning; ``cpm_report`` reads
  none, ``NO_TURN_AGENTS``);
* the answer turn (``answer_turn``): the reasoning after the last search and
  the generation as the agent returned it (the meta line's
  ``agent_generation``; ``generation/{qid}.md`` has the References section
  the pipeline appends, and its docid citations renumbered).

Each query's belief goes to ``belief/{qid}.jsonl`` (``utils.run_files.save_belief``)
and its readable view to ``belief/{qid}.md`` (``belief_model_component.views.log_view``).
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.io_utils import atomic_write_text
from utils.progress import WorkerBars

logger = logging.getLogger(__name__)

#: Agents that passed no turn to the belief model (their reasoning is a
#: plan or a section, not a turn on the search results).
NO_TURN_AGENTS = ("cpm_report",)

#: Retrievers that return a stored document with its first line as the
#: ``title`` and the rest as ``text`` (``searcher_component.retriever``:
#: BM25 on an index with stored documents); the others return the corpus
#: row (``id``, ``contents``).
TITLED_RETRIEVERS = ("bm25",)


@dataclass
class SearchIteration:
    """One search iteration of a saved trajectory."""
    agent_iteration: int
    subqueries: List[str]
    doc_ids: List[str]
    agent_turn: Optional[str]


@dataclass
class SavedQuery:
    """What the belief model reads of one query's saved run."""
    qid: str
    question: str
    iterations: List[SearchIteration]
    answer_turn: Optional[str]


def corpus_doc(doc_id: str, contents: str, retriever_name: str) -> Dict[str, Any]:
    """The document *doc_id* as the run's retriever returned it
    (``searcher_component.retriever``), normalized as the agents read it
    (``normalize_retrieval_response``): the corpus ``contents``, and for a
    titled retriever (``TITLED_RETRIEVERS``) its first line as ``title``."""
    from searcher_component import normalize_retrieval_response
    doc: Dict[str, Any] = {"id": doc_id, "contents": contents}
    if retriever_name in TITLED_RETRIEVERS:
        lines = contents.split("\n") if contents else []
        doc = {
            "id": doc_id,
            "title": lines[0].strip("\"") if lines else "",
            "text": "\n".join(lines[1:]),
            "contents": contents,
        }
    return normalize_retrieval_response([doc])[0]


def read_saved_query(run_dir: Path, qid: str, policy_model: str) -> Optional[SavedQuery]:
    """The belief model's input of one query (see the module docstring), or
    None when its trajectory is not finished (no meta line).  A trajectory
    written before the stages (no ``agent_generation`` in its meta line)
    raises ValueError: its search iterations are not marked."""
    from belief_model_component import agent_turn_text, answer_turn

    from utils.run_files import read_trajectory

    meta, steps = read_trajectory(run_dir, qid)
    if not meta:
        return None
    if "agent_generation" not in meta:
        raise ValueError(f"trajectory/{qid}.jsonl was written before the generate stage "
                         "(no agent_generation in its meta line); generate it again")
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for line in steps:
        if line.get("search_step") is not None:
            groups.setdefault(int(line["search_step"]), []).append(line)
    reads_turn = policy_model not in NO_TURN_AGENTS
    iterations = []
    for _, group in sorted(groups.items()):
        iterations.append(SearchIteration(
            agent_iteration=int(str(group[0]["iter"]).split(".", 1)[0]),
            subqueries=[q for q in (line.get("search_query") or line.get("query") for line in group) if q],
            doc_ids=[d for line in group for d in line.get("seen_docs") or []],
            agent_turn=agent_turn_text([line.get("think") for line in group]) if reads_turn else None,
        ))
    return SavedQuery(
        qid=qid, question=meta["question"], iterations=iterations,
        answer_turn=answer_turn(steps, meta["agent_generation"] or ""),
    )


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


#: Belief init errors of a call that raised before any reply (nothing banked,
#: so the next call makes them again): the belief source itself, the
#: extractor's call, TRQA's gold belief writer (whose failure would
#: otherwise leave the query on its silver beliefs).
INIT_CALL_ERRORS = ("belief_source:", "llm_call:", "gold beliefs: writer:")


def _init_failed(info: Dict[str, Any]) -> bool:
    """The query's beliefs (``belief_info``) failed for the API or the
    server, not for the model's reply: an error of ``INIT_CALL_ERRORS``,
    also behind a silver fallback (``gold_missing``, the gold errors joined
    by "; ")."""
    texts = [str(e) for e in info.get("errors") or []] + [str(info.get("gold_missing") or "")]
    return any(part.startswith(INIT_CALL_ERRORS) for text in texts for part in text.split("; "))


def _call_failed(record: Optional[Dict[str, Any]]) -> bool:
    """A record whose LLM call raised before any reply (the API or the
    server, not the model: a reply that did not parse is the model's)."""
    return bool(record) and bool(record.get("called")) and record.get("reply") is None


def _belief_of_query(belief_model, saved: SavedQuery, docs_of: Callable[[List[str]], List[Dict[str, Any]]],
                     verbose: bool, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run *belief_model* over one saved query: reset, one observe per search
    iteration, finish on the answer turn.  ``failed`` names what failed for
    a reason other than the model's reply (the beliefs, a call that raised
    before any reply, an exception); the query then stops there, is not
    saved, and the next belief stage runs it again.  *progress* gets the
    step being run: ``beliefs``, ``step k/N`` (N: the search iterations
    plus the final update), ``final``."""
    from utils.text_utils import verbose_print_belief, verbose_print_init

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
    if _init_failed(summary.get("belief_info") or {}):
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
    try:
        belief_model.finish(saved.answer_turn)
    except Exception:
        logger.warning("Belief model final update failed for %s", qid, exc_info=True)
        return out("final")
    if _call_failed(belief_model.final):
        return out("final call")
    if verbose and belief_model.final is not None:
        verbose_print_belief(None, belief_model.final, beliefs, agent_name=f"Belief {qid}", final=True)
    return out()


def run_belief_stage(
    run_dir: Path,
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
        qids: The finished queries without a belief file.
        build_model: Builds one belief model (``orchestration.build_belief_model``,
            on the run's shared belief source); one per worker, built here
            before the workers start (only the first prints its settings).
        corpus_path: The run's corpus JSONL (the documents' text).
        retriever_name: The run's retriever (``corpus_doc``).
        policy_model: The run's agent (``NO_TURN_AGENTS``).
        workers: Belief threads (queries run at once), one bar each.
        verbose: Print each query's belief.
    """
    from utils.run_files import BELIEF_DIR, save_belief
    from indexing_corpus_dataset.doc_lookup import CorpusLookup

    run_dir = Path(run_dir)
    belief_dir = run_dir / BELIEF_DIR
    belief_dir.mkdir(parents=True, exist_ok=True)
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
        atomic_write_text(belief_dir / f"{qid}.md",
                          _belief_md(out["summary"], out["steps"], out["final"], qid, saved.question))
        save_belief(run_dir, qid, saved.question, out["meta"], out["steps"], out["final"])
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
