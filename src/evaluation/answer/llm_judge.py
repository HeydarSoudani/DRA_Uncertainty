"""Answer accuracy by LLM-as-judge (BrowseComp-Plus methodology).

The judge compares the agent's answer with the ground-truth answer.  The
grading prompt is the one of BrowseComp-Plus (Chen et al., 2025,
arXiv:2508.06600), which adopts the prompt of OpenAI's BrowseComp
simple-evals.  The judge model is :data:`evaluation.judge.DEFAULT_JUDGE_MODEL`.

Scoring: binary (correct / incorrect).  Accuracy = #correct / #evaluated.

A verdict is saved in ``accuracy.jsonl`` with a ``key`` hashing what was
judged (question, cleaned response, gold answer, judge model, grader prompt);
evaluating the run again reuses it.  A reply without a ``correct:`` field is
asked again twice; a query whose judging still fails is counted incorrect, in
``num_judge_errors``, and judged again on the next evaluation.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm_client import LiteLLMClient

from ..common import print_header, write_jsonl
from ..judge import DEFAULT_JUDGE_MODEL, complete_within, judge_all, make_judge_client, strip_references

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
        extracted_final_answer, reasoning, correct (bool), confidence (int).
    """
    result: Dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
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

    # Extract "confidence:" field
    confidence_match = re.search(rf"(?:^|\n)\s*{_field_re('confidence')}\s*(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

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
        "confidence": 0,
        "judge_error": judge_error,
        "raw_response": "",
    }


def _verdict_key(question: str, judge_input: str, correct_answer: str, judge_model: str) -> str:
    """Hash of everything a verdict depends on."""
    payload = [question, judge_input, correct_answer, judge_model, GRADER_TEMPLATE]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class AccuracyEvaluator:
    """Evaluate answer correctness with an LLM judge (BrowseComp-Plus method).

    Usage::

        evaluator = AccuracyEvaluator(answers={"q1": "gold answer", ...},
                                      questions={"q1": "question", ...})
        metrics = evaluator.evaluate(results)   # {query_id: {"generation": str, ...}}
        evaluator.print_results(metrics)

    Args:
        answers: ``query_id -> ground-truth answer``.
        questions: ``query_id -> question text``; the query id stands in for a
            missing question.
        judge_model: Judge model, resolved by ``reasoner_component.api``.
        max_concurrent_judges: Max in-flight judge requests.
    """

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
        path = Path(run_dir) / "accuracy.jsonl" if run_dir else None
        if path is None or not path.exists():
            return {}
        saved = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("record") != "meta" and record.get("key") and not record.get("judge_error"):
                    saved[record["query_id"]] = record
        return saved

    def evaluate(self, results: Dict[str, Dict[str, Any]], run_dir=None) -> Dict[str, Any]:
        """Judge every query with a ground-truth answer, reusing the verdicts
        saved in ``{run_dir}/accuracy.jsonl`` for unchanged queries.

        Returns ``accuracy``, ``num_correct``, ``num_evaluated``,
        ``num_judge_errors``, ``num_judged`` (judge calls made now) and
        ``per_query`` (the judge records); ``{}`` when no query has an answer.
        An empty generation is incorrect without a judge call.
        """
        evaluable_ids = [qid for qid in results if self.answers.get(qid)]
        if not evaluable_ids:
            logger.warning("No queries with ground-truth answers to evaluate")
            return {}

        saved = self._saved_verdicts(run_dir)
        per_query: List[Dict[str, Any]] = []
        to_judge = []
        for query_id in evaluable_ids:
            generation = results[query_id].get("generation", "")
            if not generation:
                logger.warning(f"Empty generation for {query_id}, marking incorrect")
                per_query.append({"query_id": query_id, **_error_record("Empty generation", judge_error=False)})
                continue
            key = _verdict_key(self.questions.get(query_id, query_id), _clean_response_for_judge(generation),
                               self.answers[query_id], self.judge_model)
            if saved.get(query_id, {}).get("key") == key:
                per_query.append(saved[query_id])
            else:
                to_judge.append((query_id, generation, key))

        if to_judge:
            print(f"  Judging {len(to_judge)} answer(s) ({len(per_query)} reused or empty) with {self.judge_model}")
            per_query += judge_all(
                to_judge, self._judge_query,
                on_error=lambda item, e: {"query_id": item[0], **_error_record(f"Judge error: {e}")},
                max_workers=self.max_concurrent_judges, desc="[Judge]",
            )

        num_correct = sum(1 for r in per_query if r["correct"])
        num_evaluated = len(per_query)
        return {
            "accuracy": round(num_correct / num_evaluated, 5),
            "num_correct": num_correct,
            "num_evaluated": num_evaluated,
            "num_judge_errors": sum(1 for r in per_query if r.get("judge_error")),
            "num_judged": len(to_judge),
            "per_query": sorted(per_query, key=lambda r: r["query_id"]),
        }

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "ACCURACY EVALUATION RESULTS (LLM-as-Judge)",
    ) -> None:
        """Pretty-print the accuracy metrics."""
        if not metrics:
            print("  No accuracy metrics available (no ground-truth answers)")
            return
        print_header(header)
        print(f"  Judge model:        {self.judge_model}")
        print(f"  Queries evaluated:  {metrics.get('num_evaluated', 0)}")
        print(f"  Correct:            {metrics.get('num_correct', 0)}")
        print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"  Judge errors:       {metrics.get('num_judge_errors', 0)}"
              f"  (judged now: {metrics.get('num_judged', 0)}, reused: "
              f"{metrics.get('num_evaluated', 0) - metrics.get('num_judged', 0)})")
        print("=" * 80)

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Write ``accuracy.jsonl``: a ``{"record": "meta", ...}`` line with the
        run-level aggregates and the judge model, then one line per query
        (without the raw judge response)."""
        if not metrics:
            return
        meta = {
            "record": "meta",
            "accuracy": metrics["accuracy"],
            "num_correct": metrics["num_correct"],
            "num_evaluated": metrics["num_evaluated"],
            "num_judge_errors": metrics["num_judge_errors"],
            "judge_model": self.judge_model,
        }
        fields = ("query_id", "key", "correct", "judge_error", "extracted_final_answer", "judge_input",
                  "reasoning", "confidence")
        write_jsonl(output_path, [meta] + [
            {k: r.get(k) for k in fields} for r in metrics.get("per_query", [])
        ])
        print(f"  Saved accuracy results: {output_path}")
