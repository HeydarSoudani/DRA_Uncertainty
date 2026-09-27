# Uncertainty-Aware Agent Training

`experiments/dra_uncertainty_aware_train.py` trains the [uncertainty-aware agent](uncertainty_aware_agent.md): an SFT cold start on teacher
trajectories, then GRPO. The common pipeline lives in `src/training`; the script holds only what is specific to the
uncertainty-aware agent and plugs it in. This page covers the split between the two, the environment, the samples and reward,
the stages, and what is still missing.

## Layout

| where | what |
|---|---|
| `src/training/rollout/env.py` | `AgentEnv` interface: `reset(prompt)`, `messages()`, `gen_params()`, `step(text)`, `summary()` |
| `src/training/rollout/agent_rollout.py` | agent-agnostic driver: policy call, env step, one sample per call |
| `src/training/rollout/policy_client.py` | `MockPolicyClient`, `ServerPolicyClient` (vLLM, token ids + logprobs), `ApiPolicyClient` (SFT teacher, text only) |
| `src/training/rollout/tool_env.py` | shared search tool: mock, or `RetrievalSearchTool` built as in `dra_inference.py` |
| `src/training/rollout/env_llm.py` | `TextLLM`: auxiliary model calls through `reasoner_component`, reasoning switched off as in the uncertainty-aware agent |
| `src/training/reward/outcome_reward.py` | outcome reward, scored as the inference evaluation scores it |
| `src/training/data/sft_dataset.py` | teacher rollouts, reject sampling, per-turn chat samples |
| `experiments/dra_uncertainty_aware_train.py` | `UncertaintyAwareEnv`, criteria updater, `agent:` config parsing, `--selftest` |
| `experiments/configs/dra_uncertainty_aware_train.yaml` | all settings; unknown keys are rejected |

`src/training` imports nothing from the uncertainty-aware agent. `UncertaintyAwareEnv` imports the inference agent's own pieces
(`parse_turn`, `DocRegistry`, `render_information`, `render_belief`, prompts, `CriteriaTracker`), so the policy is
trained on the context it reads at inference.

## Environment

`UncertaintyAwareEnv` is one uncertainty-aware agent run, step for step as `UncertaintyAwareAgent.inference`:

- `reset`: criteria init call (all `not_covered`), system prompt, `Question: …` transcript.
- `messages`: `[system, user(transcript)]`, plus `[assistant(bad turn), user(format error)]` after a malformed turn.
- `step`: a malformed turn is re-asked up to `max_format_retries` times, then the run ends as `format_failure`; an
  answer ends it as `answered`; a search retrieves, shows `seen_top_k` passages, updates novelty and criteria, and
  appends `<information>` and `<belief>`; after `max_turns` turns the run ends as `max_turns`.

`--selftest` runs the same scripted policy through `UncertaintyAwareAgent.inference` and `UncertaintyAwareEnv` (answered, format failure,
max turns; every novelty/criteria display setting) and checks that every policy call and every updater call receives
identical messages.

The criteria updater is `agent.criteria_model` (default `openrouter/qwen/qwen3.6-27b`, the inference backbone): greedy,
reasoning off, frozen. Inference falls back to the policy backbone when the key is empty; training does not, since the
policy changes with every update.

## Samples and reward

The transcript is one growing user message, so turn t+1's context is not turn t's context plus its output. Each policy
call is therefore its own sample, `prompt_token_ids` (masked) followed by `gen_token_ids` (trained), and every sample of
a trajectory gets the trajectory's GRPO advantage. Re-asked turns are samples too. `ServerPolicyClient` applies the
chat template locally (`enable_thinking: false`), sends token ids to vLLM, and keeps stop strings in the output so text
and ids align. A context that no longer fits ends the run as `context_overflow`.

The reward is the outcome only: 1 if the answer is correct, else 0 (also for runs that never answer). TRQA uses the
numeric exact match of `TRQAGenerationEvaluator`; other datasets use the BrowseComp-Plus judge of
`AccuracyEvaluator` (`reward.judge_model`).

## Stages

| stage | what runs | output (`output_dir`) |
|---|---|---|
| `sft` | teacher (`sft.teacher_model`) through `UncertaintyAwareEnv`; keep correct trajectories with no re-asked turn; one chat sample per turn | `sft/sft_data.jsonl`, `sft/rollouts/` |
| `rl` | policy on vLLM through `UncertaintyAwareEnv`; score, advantages, update | `rollouts/step*/`, per-step stats in the log |

An SFT sample is `{"messages": [system, user, assistant], "prompt_id", "turn", "action", "outcome"}`; the assistant
message is the normalised turn (`<think>…</think>\n<search>…</search>`), and the loss belongs on it alone.

```
python experiments/dra_uncertainty_aware_train.py --selftest                 # parity with the inference agent (CPU)
python experiments/dra_uncertainty_aware_train.py --smoke [--mode async] [--stage sft]   # CPU, all mocks
python experiments/dra_uncertainty_aware_train.py --stage sft --limit 200    # teacher data (retriever on GPU)
python experiments/dra_uncertainty_aware_train.py --stage rl --server-url http://127.0.0.1:8000
```

Data defaults to TRQA `wiki1_validation` (91 queries) so `test` stays held out; `--subset wiki2` has 1083.

## Not yet implemented

- GRPO update and weight sync (`trainer.backend: grpo`, veRL). Until then `trainer.backend: mock` runs rollouts,
  rewards and advantages without changing weights.
- SFT backend (`sft.backend`, veRL SFT). `none` stops after writing the data.
- Process rewards (`reward.lambda_process` is 0).
