# Uncertainty-Aware Agent

`--agentic-model uncertainty_aware` — a SearchR1-style search agent that, after every retrieval, reads a **belief**: a
progress report computed by code, not written by the policy. The belief carries two things:

- **doc novelty** — how many of the passages just shown are new to the run;
- **criteria progress** — the query's criteria, each with a status `covered` / `partial` / `not_covered`.

Code: `src/deep_research_agents/agents/uncertainty_aware_agent.py` (loop, turn parsing, novelty, belief rendering) and
`src/deep_research_agents/agent_tools/uncertainty_aware_criteria.py` (criteria updater). Prompts:
`src/deep_research_agents/prompts/uncertainty_aware/` (policy) and `.../uncertainty_aware/criteria/` (updater). The criteria updater
and its prompts began as copies of the former controller's criteria-coverage signal (prompt fixes are listed under
[Criteria updater](#criteria-updater)). As in every agent, an uncertainty estimator attached with
`--uncertainty-estimator monitor` observes each search iteration and never changes the run.

## Loop

```
start        updater: init prompt ─▶ criteria k1…kn, all not_covered
iteration t  policy:  <think> <search>q</search>
             code:    retrieve ─▶ <information> [d1]… ─▶ novelty new/shown
             updater: update prompt(question, current list, q, shown passages) ─▶ delta ─▶ apply
             code:    append <belief step="t">
end          policy:  <think> <answer>…</answer>      (or ua_max_turns runs out = failure)
```

The run is one growing transcript in a single user message. Per search there is one updater call, plus one init
call per query.

## What the policy sees

```
Question: …

<think>…</think>
<search>…</search>
<information>
[d1] Title
text…
</information>
<belief step="0">
  <novelty new="3" shown="5"/>
  <criteria covered="1" partial="1" not_covered="1" total="3">
    <k1 status="covered">its conservation status is 'endangered'</k1>
    <k2 status="partial">it is the only representative of its genus in the country</k2>
    <k3 status="not_covered">one of its local names roughly translates to 'devil's basket'</k3>
  </criteria>
</belief>
<think>…</think>
…
```

- Passage labels `d1, d2, …` are stable within a run: the same `doc_id` keeps its label.
- Novelty counts shown passages (`seen_top_k`) whose `doc_id` was never shown earlier in the run.
- Only statuses are shown. The updater's evidence notes and reasoning are logged, never shown to the policy.
- The criteria are not shown before the first search; they first appear in `<belief step="0">`.
- The policy writes only `<think>` + one `<search>` or `<answer>`. A turn with no action is re-asked up to
  `ua_max_format_retries` times, then the run ends as `format_failure`. A `<belief>`/`<information>` written by the
  policy is dropped (warning).

## Criteria updater

Same logic as the former controller's criteria-coverage signal (the controller has been removed; the
`CriteriaCoverageSignal` in `uncertainty_estimator` is a different signal), with two modes:

| mode | init prompt | update prompt | list |
|---|---|---|---|
| `static` | copy each condition of the query verbatim | `tick` only | fixed |
| `dynamic` | decompose the query into 2…`max_criteria` criteria | `tick`, `add`, `remove` | frozen after `stabilization_window` iterations with no add/remove |

`ua_criteria_mode: auto` picks `static` on `browsecomp_plus` and `dynamic` otherwise.

- The updater answers in JSON with only the criteria that changed; code applies the delta (case-insensitive name
  match, status aliases such as `partially covered` → `partial`).
- It sees the passages the policy was shown: `ua_evidence_top_k` passages × `ua_evidence_chars` characters
  (the former controller used 10 × 200).
- Model: `ua_criteria_model`, or the policy backbone when empty. Calls are greedy (temperature 0) with the
  model's reasoning switched off when `ua_disable_native_thinking` (on OpenRouter; for a separate non-OpenRouter
  model the switch is not sent).
- A failed or unparsable call leaves the list unchanged; the failure is recorded on the step and in the outcome. If
  init fails, the run continues with no criteria (the belief then shows novelty only). JSON wrapped in prose with no
  code fence is still parsed (the outermost `{…}` is taken). The `critical_gaps` / `minor_gaps` keys the prompts ask
  for are ignored.
- Prompt differences from the former controller's prompts: the dynamic update prompt states whether the list is frozen
  (the controller's prompt dropped the `{frozen_instruction}` placeholder), its example JSON has no trailing comma, and a few
  wording slips are fixed ("a criterion", "the criterion's information need").

## Configuration (`experiments/configs/dra_inference.yaml`)

| key | default | meaning |
|---|---|---|
| `ua_max_turns` | 8 | policy turns before a run with no answer fails |
| `ua_max_passage_chars` | 1500 | chars per passage in `<information>` |
| `ua_max_format_retries` | 2 | re-asks after a turn with no action |
| `ua_max_tokens_per_call` | 4096 | output cap for one policy turn |
| `ua_disable_native_thinking` | true | reasoning mode off for policy and updater |
| `ua_show_novelty` | true | show `<novelty>` (ablation) |
| `ua_show_criteria` | true | show `<criteria>` (ablation); both off = no belief and no belief section in the system prompt |
| `ua_criteria_mode` | auto | auto / static / dynamic |
| `ua_criteria_model` | "" | updater model; empty = backbone |
| `ua_max_criteria` | 8 | cap on the criteria list |
| `ua_stabilization_window` | 15 | dynamic mode freeze window |
| `ua_criteria_max_tokens` | 1024 | output cap for one updater call |
| `ua_evidence_top_k` | 5 | passages per update shown to the updater |
| `ua_evidence_chars` | 1500 | chars per passage shown to the updater |

`seen_top_k` and the run temperature apply; `max_iteration` does not.

## Outputs

The result (and the trajectory meta line, via `AGENT_META_KEYS`) carries:

- `ua_criteria` — the initial list; `ua_criteria_raw` — the init call's raw output.
- `ua_records` — one per policy turn: `turn`, `action`, `action_text`, `valid`, `errors`, `warnings`,
  `format_retries`; for searches also `step`, `novelty`, `novel_docs`, `shown_docs`, `search_novelty` (labels),
  `criteria` (full summary: statuses, evidence, changed/new/removed, reasoning, raw, error), `criteria_error`, and the
  rendered `belief`.
- `ua_doc_labels` — `dN → doc_id`.
- `ua_outcome` — `end` (`answered` / `max_turns` / `format_failure` / `llm_error`), `answer`, `num_turns`,
  `num_searches`, `format_retries`, `criteria_mode`, `criteria_init_error`, `criteria_errors`,
  `criteria_update_failures`, `final_criteria` and `final_covered/partial/not_covered/total`, `mean_novelty`, `config`.

The trajectory log shows the criteria table (with evidence; `*` marks a status that changed this step) and the
novelty after every search. A run that does not answer ends on a trajectory step whose `action_type` is its end
(`max_turns`, `format_failure` or `llm_error`, the last with the error message). The run is scored by the shared
evaluators; the belief is saved, not scored.
