# Belief-Aware Agent

`--policy-model belief_aware` is a SearchR1-style search agent that, after every retrieval, reads the
`<belief_state>` tag of the shared belief model in `inform` mode. The tag is computed by code, never written by
the policy, and is exactly the one every other agent reads under `--belief-mode inform` (belief
status and attempts per criterion, then the signals `doc_novelty`, `criteria_delta`, `query_novelty`; no gold-based
signal). What sets this agent apart is only its system prompt, which introduces the tag: one section
(`belief_section.txt`) shared by the `qa` and `report` tasks, describing the fields only, with no advice on how
to use them.

`--belief-mode` applies as for every other agent:

| Mode | Trajectory | System prompt | Run directory |
|---|---|---|---|
| `inform` | `<belief_state>` after each `<information>` | includes `belief_section.txt` | `belief-inform` |
| `monitor` | no tag (belief and signals still saved to `belief/{qid}.jsonl`) | never mentions the tag | `belief-monitor` |
| `off` | no tag, no belief model (plain SearchR1-style agent) | never mentions the tag | `belief-off` |

Criteria extraction, the belief updater, the signals, the per-step intermediate answers and
`belief/{qid}.jsonl` all come from the belief model settings (`llm_criteria`, `max_criteria`,
`belief_judge_model`, `query_scorer_model`).

Code: `src/deep_research_agents/agents/belief_aware_agent.py` (loop, turn parsing, passage labels). Prompts:
`src/deep_research_agents/prompts/belief_aware/` (`system.txt`, `belief_section.txt`, `user.txt`,
`format_error.txt`). The tag layout is in `src/belief_model_component/belief_state.py`.

## Loop

```
iteration t  policy:     <reasoning> <search>q</search>
             code:       retrieve ─▶ <information> [d1]…
             belief model:  observe the iteration ─▶ <belief_state step="t">   (inform only)
end          policy:     <reasoning> <answer>…</answer>
forced       (max_iteration runs out first) one more call asks for <reasoning> <answer>
```

The run is one growing transcript in a single user message.

## What the policy sees

In `inform` mode (in `monitor` and `off` modes the `<belief_state>` blocks are absent):

```
Question: …

<reasoning>…</reasoning>
<search>…</search>
<information>
[d1] Title
text…
</information>
<belief_state step="1">
  <criteria>
    <k1 status="almost certain" attempts="1">the plant whose conservation status is 'endangered'</k1>
    <k2 status="possible" attempts="2">the plant that is the only representative of its genus in the country</k2>
    <k3 status="unknown" attempts="0">the plant one of whose local names roughly translates to 'devil's basket'</k3>
  </criteria>
  <signals doc_novelty="0.40" criteria_delta="+2" query_novelty="0.81"/>
</belief_state>
<reasoning>…</reasoning>
…
```

- Passage labels `d1, d2, …` are stable within a run: the same `doc_id` keeps its label. A search shows
  its first `seen_top_k` distinct documents; a repeated id or a document without an id is not shown.
- The tag's `step` counts searches from 1. A null signal is left out of the tag.
- If the belief model fails on an iteration, no tag is appended and the run continues (counted in
  `ba_outcome.missing_belief_state`, which is null outside `inform` mode).
- The policy writes only `<reasoning>` + one `<search>` or `<answer>`. A turn with no action is re-asked up to
  `max_retries` times, then the run ends as `format_failure`. A `<belief_state>`/`<information>` written by
  the policy is dropped (warning). Reasoning written without tags before the action is kept as the
  turn's think (warning).

## Configuration (`experiments/configs/dra_inference.yaml`)

The agent uses the shared keys; it has none of its own.

| key | default | meaning for this agent |
|---|---|---|
| `max_iteration` | 100 | policy turns before the forced answer; enforced in code, never stated in the prompt |
| `max_retries` | 3 | re-asks after a turn with no action |
| `max_passage_chars` | 4000 | chars per passage in `<information>` |
| `llm_max_tokens_per_call` | 10000 | output cap for one policy turn (and one intermediate answer) |

`seen_top_k` and the run temperature apply as well. The backbone's own reasoning mode is always off
(not configurable), as in the Search-R1 family: the protocol's `<reasoning>` is the only reasoning.
The tag is `<reasoning>` rather than Search-R1's `<think>` because `<think>` is Qwen3.x's native
reasoning tag: with native thinking off, the chat template pre-fills an empty `<think></think>`, and the
model then skips reasoning altogether instead of opening `<think>` again.

The intermediate answer (asked by the belief model after each search) is appended to the transcript's user
message, uses the `<reasoning>` + `<answer>` format, and stops at `</answer>` or at a `<search>` the model
starts instead (then the intermediate answer is None). Its prompt is `INTERMEDIATE_ANSWER_INSTRUCTION` in `src/belief_model_component/prompts/user_prompts.py`.

The forced answer works the same way: its instruction plus the `<reasoning>` + `<answer>` format is
appended to the transcript's user message, and one call is made, with no re-ask. It is asked in two cases:

| case | instruction (`prompts/answer_prompts.py`) | transcript | trajectory step | end without `<answer>` |
|---|---|---|---|---|
| `max_iteration` runs out | `MAX_TURNS_ANSWER_INSTRUCTION` | full; windowed if it does not fit | `max_iter_force` | `max_turns` |
| a policy turn exceeds the context window | `FINAL_ANSWER_INSTRUCTION` | windowed | `context_limit` | `context_limit` |

The windowed transcript keeps every turn and `<belief_state>` tag but only the passages of the last 3
searches; older `<information>` blocks read `(earlier search results omitted)`. Both step types are in the
trajectory evaluator's force-answer set, as for the other agents.

## Outputs

The result (and the trajectory meta line, via `AGENT_META_KEYS`) carries:

- `ba_records`: one per policy turn: `turn`, `action`, `action_text`, `valid`, `errors`, `warnings`,
  `format_retries`, `native_reasoning_chars`; for searches also `search` (index), `labels` and the `belief_state` tag.
- `ba_doc_labels`: `dN → doc_id`.
- `ba_outcome`: `end` (`answered` / `forced_answer` / `max_turns` / `context_limit` / `format_failure` /
  `llm_error`), `forced_by` (`max_turns` / `context_limit` when a forced answer was asked, else null), `answer`,
  `num_turns`, `num_searches`, `format_retries`, `missing_belief_state`, `config` (`max_iteration`, `max_retries`,
  `max_passage_chars`).

The tag is also saved as `belief_state` on the search step in `trajectory/{qid}.jsonl`, and the full step record in
`belief/{qid}.jsonl`. A run that does not answer ends on a trajectory step whose `action_type` is its end
(`max_iter_force` or `context_limit` with `errors`, `format_failure` or `llm_error`, the last with the error message). The run is scored by the shared
evaluators.
