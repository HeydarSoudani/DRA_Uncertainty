# Uncertainty-Aware Agent

`--agentic-model uncertainty_aware` is a SearchR1-style search agent that, after every retrieval, reads the
`<certainty>` tag of the shared uncertainty estimator. The tag is computed by code, never written by the policy, and
is exactly the one every other agent reads under `--uncertainty-estimator-mode inform` (criteria states, retrieval
signals `doc_novelty` / `criteria_delta`, reasoning signals `query_novelty` / `criteria_targeting`; no gold-based
signal). What sets this agent apart is only its system prompt, which explains the tag.

The agent always runs the estimator in `inform` mode: `--uncertainty-estimator-mode` is ignored for it (the run
directory is `ue-inform_{criteria_judge}`). Criteria extraction, the criteria judge, the signals, the per-step
intermediate answers and `uncertainty/{qid}.jsonl` all come from the estimator settings (`llm_criteria`,
`max_criteria`, `criteria_judge`, `criteria_judge_model`).

Code: `src/deep_research_agents/agents/uncertainty_aware_agent.py` (loop, turn parsing, passage labels). Prompts:
`src/deep_research_agents/prompts/uncertainty_aware/` (`system.txt`, `certainty_section.txt`, `user.txt`,
`format_error.txt`). The tag layout is in `src/uncertainty_estimator/certainty.py`.

## Loop

```
iteration t  policy:     <think> <search>q</search>
             code:       retrieve ─▶ <information> [d1]…
             estimator:  observe the iteration ─▶ <certainty step="t">
end          policy:     <think> <answer>…</answer>      (or ua_max_turns runs out = failure)
```

The run is one growing transcript in a single user message.

## What the policy sees

```
Question: …

<think>…</think>
<search>…</search>
<information>
[d1] Title
text…
</information>
<certainty step="1">
  <criteria covered="1" partial="1" not_covered="1">
    <k1 status="covered">its conservation status is 'endangered'</k1>
    <k2 status="partial">it is the only representative of its genus in the country</k2>
    <k3 status="not_covered">one of its local names roughly translates to 'devil's basket'</k3>
  </criteria>
  <retrieval_signals doc_novelty="0.42" criteria_delta="+1"/>
  <reasoning_signals query_novelty="0.81" criteria_targeting="0.60"/>
</certainty>
<think>…</think>
…
```

- Passage labels `d1, d2, …` are stable within a run: the same `doc_id` keeps its label.
- The tag's `step` counts searches from 1. A null signal is left out of the tag.
- If the estimator fails on an iteration, no tag is appended and the run continues (counted in
  `ua_outcome.missing_certainty`).
- The policy writes only `<think>` + one `<search>` or `<answer>`. A turn with no action is re-asked up to
  `ua_max_format_retries` times, then the run ends as `format_failure`. A `<certainty>`/`<information>` written by
  the policy is dropped (warning).

## Configuration (`experiments/configs/dra_inference.yaml`)

| key | default | meaning |
|---|---|---|
| `ua_max_turns` | 8 | policy turns before a run with no answer fails |
| `ua_max_passage_chars` | 1500 | chars per passage in `<information>` |
| `ua_max_format_retries` | 2 | re-asks after a turn with no action |
| `ua_max_tokens_per_call` | 4096 | output cap for one policy turn (and one intermediate answer) |
| `ua_disable_native_thinking` | true | backbone reasoning mode off for policy turns and intermediate answers |

`seen_top_k` and the run temperature apply; `max_iteration` does not.

## Outputs

The result (and the trajectory meta line, via `AGENT_META_KEYS`) carries:

- `ua_records`: one per policy turn: `turn`, `action`, `action_text`, `valid`, `errors`, `warnings`,
  `format_retries`, `native_reasoning_chars`; for searches also `search` (index), `labels` and the `certainty` tag.
- `ua_doc_labels`: `dN → doc_id`.
- `ua_outcome`: `end` (`answered` / `max_turns` / `format_failure` / `llm_error`), `answer`, `num_turns`,
  `num_searches`, `format_retries`, `missing_certainty`, `config`.

The tag is also saved as `certainty` on the search step in `trajectory/{qid}.jsonl`, and the full step record in
`uncertainty/{qid}.jsonl`. A run that does not answer ends on a trajectory step whose `action_type` is its end
(`max_turns`, `format_failure` or `llm_error`, the last with the error message). The run is scored by the shared
evaluators.
