"""A query's saved run as the belief model reads it.

The generate stage saves each query's trajectory (``trajectory/{qid}.jsonl``)
and generation (``generation/{qid}.md``).  The belief stage
(``pipeline.belief_stage``) and the belief training
(``belief_model_training.shared.runs``) read them back with
:func:`read_saved_query`, which gives, per query, what the agent's turns
showed:

* per search iteration (the trajectory lines that share a ``search_step``,
  ``BasicAgent._record_iteration``): its queries, the documents it showed
  (the lines' ``seen_docs``, in order; their text is read from the corpus by
  id, ``CorpusLookup``, and shaped as the run's retriever returns a
  document, :func:`corpus_doc`; nothing is retrieved again) and the agent's
  turn (``agent_turn_text`` of the lines' reasoning; ``cpm_report`` reads
  none, :data:`NO_TURN_AGENTS`);
* the answer turn (``answer_turn``): the reasoning after the last search and
  the generation as the agent returned it (the meta line's
  ``agent_generation``; ``generation/{qid}.md`` has the References section
  the pipeline appends, and its docid citations renumbered).  The final
  update reads the reasoning (``answer_reasoning``; none for
  :data:`NO_TURN_AGENTS`) and the final scorer the generation, passed apart
  (``BeliefModel.finish_answer``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.run_files.trajectory import read_trajectory

from .agent_turn import agent_turn_text, answer_reasoning, answer_turn

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
    # The answer turn apart: the reasoning after the last search ("" for
    # ``NO_TURN_AGENTS``) and the agent's generation.
    answer_reasoning: str = ""
    answer: Optional[str] = None


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
        answer_reasoning=answer_reasoning(steps) if reads_turn else "",
        answer=meta["agent_generation"] or None,
    )
