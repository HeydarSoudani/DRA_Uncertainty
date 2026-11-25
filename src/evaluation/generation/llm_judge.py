"""Answer accuracy by LLM-as-judge (BrowseComp-Plus methodology).

The judge compares the agent's answer with the ground-truth answer, with the
grading prompt of BrowseComp-Plus (Chen et al., 2025, arXiv:2508.06600),
itself OpenAI's BrowseComp simple-evals prompt, kept verbatim
(:data:`GRADER_TEMPLATE`).  Its ``confidence`` field is kept so saved verdicts
stay valid, but not read: the agent's own confidence is read from the
generation (``confidence.py``).  The judge model is
``shared.judge.DEFAULT_JUDGE_MODEL``.

Scoring: binary; accuracy = correct / evaluated (rows: ``accuracy.py``).
An empty generation is incorrect without a judge call.  A reply without a
``correct:`` field is asked again :data:`PARSE_RETRIES` times; a query whose
judging still fails is counted incorrect, in ``judge errors``, and judged
again on the next evaluation.

Cache: every verdict is saved in ``eval/generation/accuracy.jsonl`` with a
``key`` hashing what was judged (question, cleaned response, gold answer,
judge model, grader prompt); evaluating the run again reuses it.

Datasets: ``answer_eval == "llm_judge"`` (BrowseComp-Plus, infoseek_hard, ...).
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.io import read_jsonl, write_jsonl
from reasoner_component import LiteLLMClient

from ..shared.context import GENERATION, EvalContext, EvalSetup
from ..shared.files import ACCURACY_FILE
from ..shared.judge import DEFAULT_JUDGE_MODEL, complete_within, judge_all, make_judge_client, verdict_key
from ..shared.result import GroupResult
from .accuracy import accuracy_result
from .confidence import verbal_confidence
from .references import strip_references

logger = logging.getLogger(__name__)

# Asks again of a judge reply without a parsable ``correct:`` field.
PARSE_RETRIES = 2

# ---------------------------------------------------------------------------
# Grader prompt, verbatim from BrowseComp-Plus Appendix F / search_agent/prompts.py
# ---------------------------------------------------------------------------

GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise \
and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on \
[correct_answer], in the context of this [question]. You should judge whether the \
extracted_final_answer is semantically equivalent to [correct_answer], allowing the \
extracted_final_answer to be string variations of [correct_answer]. You should also allow \
the extracted_final_answer to be more precise or verbose than [correct_answer], as long as \
its additional details are correct. Do not comment on any background to the problem, do not \
attempt to solve the problem, do not argue for any answer different than [correct_answer], \
focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, \
or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if \
there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is \
incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put \
100 if there is no confidence score available. /no_think"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the structured fields from the judge LLM's response.

    Returns a dict with keys:
        extracted_final_answer, reasoning, correct (bool), parse_error,
        raw_response.  The judge's ``confidence`` field is not read (the
        agent's own confidence is, from the generation).
    """
    result: Dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "parse_error": False,
        "raw_response": text,
    }

    # Field prefix pattern: handles "field:", "**field:**", and "**field**:"
    def _field_re(name: str) -> str:
        return rf"\*{{0,2}}{name}\*{{0,2}}:\*{{0,2}}"

    # Extract "correct:" field (yes/no)
    correct_match = re.search(rf"(?:^|\n)\s*{_field_re('correct')}\s*(yes|no)", text, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).strip().lower() == "yes"

    # Extract "extracted_final_answer:" field
    answer_match = re.search(
        rf"(?:^|\n)\s*{_field_re('extracted_final_answer')}\s*(.+?)(?=\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract "reasoning:" field
    reasoning_match = re.search(
        rf"(?:^|\n)\s*{_field_re('reasoning')}\s*(.+?)(?=\n\s*(?:{_field_re('correct')}|{_field_re('confidence')})|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    if result["correct"] is None:
        result["parse_error"] = True

    return result



# ---------------------------------------------------------------------------
# AccuracyEvaluator
# ---------------------------------------------------------------------------

def _clean_response_for_judge(text: str) -> str:
    """The agent generation without what is irrelevant to correctness.

    Drops the appended ``## References`` block (document snippets, often
    5-10x longer than the answer) and the lenticular-bracket citation markers
    (``【12345】``) of tool-calling agents, which name internal document IDs.
    """
    return re.sub(r"【[^】]*】", "", strip_references(text)).strip()


def _error_record(reasoning: str, judge_input: str = "", judge_error: bool = True) -> Dict[str, Any]:
    """The verdict of a response that was not judged: incorrect."""
    return {
        "extracted_final_answer": None,
        "judge_input": judge_input,
        "reasoning": reasoning,
        "correct": False,
        "judge_error": judge_error,
        "raw_response": "",
    }


def _verdict_key(question: str, judge_input: str, correct_answer: str, judge_model: str) -> str:
    """Hash of everything a verdict depends on."""
    return verdict_key([question, judge_input, correct_answer, judge_model, GRADER_TEMPLATE])


class AccuracyEvaluator:
    """Answer correctness by LLM judge (module docstring).

    Usage::

        evaluator = AccuracyEvaluator(answers={"q1": "gold answer", ...},
                                      questions={"q1": "question", ...})
        raw = evaluator.score(results, run_dir)        # the verdicts (accuracy.py)
        group = evaluator.evaluate(EvalContext(results, run_dir))   # rows, saved verdicts

    Args:
        answers: ``query_id -> ground-truth answer``.
        questions: ``query_id -> question text``; the query id stands in for a
            missing question.
        judge_model: Judge model, resolved by ``reasoner_component.providers.routing``.
        max_concurrent_judges: Max in-flight judge requests.
    """

    name = "accuracy"
    scope = GENERATION
    provides = "accuracy"

    def __init__(
        self,
        answers: Dict[str, str],
        questions: Optional[Dict[str, str]] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        max_concurrent_judges: int = 64,
    ) -> None:
        self.answers = answers
        self.questions = questions or {}
        self.judge_model = judge_model
        self.max_concurrent_judges = max_concurrent_judges
        self._client: Optional[LiteLLMClient] = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["AccuracyEvaluator"]:
        if not (setup.answers and setup.answer_eval == "llm_judge"):
            return None
        return cls(answers=setup.answers, questions=setup.questions, **setup.judge_kwargs)

    @property
    def client(self) -> LiteLLMClient:
        """The judge client, created on first use."""
        if self._client is None:
            self._client = make_judge_client(self.judge_model)
        return self._client

    def judge(self, question: str, response: str, correct_answer: str) -> Dict[str, Any]:
        """Judge one response: the parsed verdict plus ``judge_input``, the
        cleaned response the judge saw, and ``judge_error``.  A reply without
        a verdict is asked again :data:`PARSE_RETRIES` times; a failed call or
        a reply that never parses is judged incorrect."""
        response = _clean_response_for_judge(response)
        prompt = GRADER_TEMPLATE.format(
            question=question, response=response, correct_answer=correct_answer,
        )
        for _ in range(PARSE_RETRIES + 1):
            try:
                raw_response = complete_within(self.client, [{"role": "user", "content": prompt}])
            except Exception as e:
                logger.error(f"Judge failed: {e}")
                return _error_record(f"Judge error: {e}", response)
            parsed = _parse_judge_response(raw_response)
            if not parsed["parse_error"]:
                break
        parsed["judge_input"] = response
        parsed["judge_error"] = parsed["parse_error"]
        if parsed["correct"] is None:
            parsed["correct"] = False
        return parsed

    def _judge_query(self, item) -> Dict[str, Any]:
        query_id, generation, key = item
        verdict = self.judge(
            self.questions.get(query_id, query_id), generation, self.answers[query_id],
        )
        return {"query_id": query_id, "key": key, **verdict}

    def _saved_verdicts(self, run_dir) -> Dict[str, Dict[str, Any]]:
        """The verdicts of a previous evaluation (``accuracy.jsonl``) that
        were judged without error, by query id."""
        if not run_dir:
            return {}
        path = Path(run_dir) / ACCURACY_FILE
        if not path.exists():
            return {}
        return {r["query_id"]: r for r in read_jsonl(path)
                if r.get("key") and not r.get("judge_error")}

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        """The rows and per-query values of :meth:`score` (``accuracy.py``),
        its verdicts saved to ``{run_dir}/eval/generation/accuracy.jsonl``."""
        return accuracy_result(self.score(ctx.results, ctx.run_dir), ctx.run_dir, self.save_results)

    def pending(self, ctx: EvalContext) -> int:
        """The queries :meth:`score` would judge now (no saved verdict for
        their current input, or a saved judge error); no LLM call."""
        return len(self._plan(ctx.results, ctx.run_dir)[1])

    def _plan(self, results: Dict[str, Dict[str, Any]], run_dir) -> Tuple[List[Dict[str, Any]], List[Tuple]]:
        """``(records, to_judge)``: the records known without a judge call
        (saved verdicts of unchanged queries, empty generations) and the
        ``(query_id, generation, key)`` items left to judge."""
        saved = self._saved_verdicts(run_dir)
        per_query: List[Dict[str, Any]] = []
        to_judge: List[Tuple] = []
        for query_id in (qid for qid in results if self.answers.get(qid)):
            generation = results[query_id].get("generation", "")
            if not generation:
                per_query.append({"query_id": query_id, **_error_record("Empty generation", judge_error=False)})
                continue
            key = _verdict_key(self.questions.get(query_id, query_id), _clean_response_for_judge(generation),
                               self.answers[query_id], self.judge_model)
            if saved.get(query_id, {}).get("key") == key:
                per_query.append(saved[query_id])
            else:
                to_judge.append((query_id, generation, key))
        return per_query, to_judge

    def score(self, results: Dict[str, Dict[str, Any]], run_dir=None) -> Dict[str, Any]:
        """Judge every query with a ground-truth answer, reusing the verdicts
        saved in ``{run_dir}/eval/generation/accuracy.jsonl`` for unchanged queries.

        Returns ``accuracy``, ``num_correct``, ``num_evaluated``,
        ``num_judge_errors`` and ``per_query`` (the judge records, with the
        agent's ``verbal_confidence``); ``{}`` when no query has an answer.
        An empty generation is incorrect without a judge call.
        """
        if not any(self.answers.get(qid) for qid in results):
            logger.warning("No queries with ground-truth answers to evaluate")
            return {}

        per_query, to_judge = self._plan(results, run_dir)
        empty = [r["query_id"] for r in per_query if r.get("reasoning") == "Empty generation"]
        if empty:
            logger.warning(f"Empty generation for {len(empty)} query(ies), marked incorrect: {', '.join(empty[:10])}")
        if to_judge:
            print(f"  Judging {len(to_judge)} answer(s) ({len(per_query)} reused or empty) with {self.judge_model}")
            per_query += judge_all(
                to_judge, self._judge_query,
                on_error=lambda item, e: {"query_id": item[0], **_error_record(f"Judge error: {e}")},
                max_workers=self.max_concurrent_judges, desc="[Judge]",
            )

        # The agent's own confidence, read from the generation (never the
        # judge's extraction, which defaults a missing one to 100).
        for r in per_query:
            r["verbal_confidence"] = verbal_confidence(results[r["query_id"]].get("generation", ""))
        num_correct = sum(1 for r in per_query if r["correct"])
        num_evaluated = len(per_query)
        return {
            "accuracy": round(num_correct / num_evaluated, 5),
            "num_correct": num_correct,
            "num_evaluated": num_evaluated,
            "num_judge_errors": sum(1 for r in per_query if r.get("judge_error")),
            "per_query": sorted(per_query, key=lambda r: r["query_id"]),
        }

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Write ``accuracy.jsonl``: one line per query, its verdict and the
        cache ``key`` (without the raw judge response or the judged text,
        which is ``generation/{qid}.md``).  The aggregates are in
        ``eval/metrics.md``, the judge model in ``run_config.json``."""
        if not metrics:
            return
        fields = ("query_id", "key", "correct", "judge_error", "extracted_final_answer",
                  "reasoning", "verbal_confidence")
        write_jsonl(output_path, [{k: r.get(k) for k in fields} for r in metrics.get("per_query", [])])

