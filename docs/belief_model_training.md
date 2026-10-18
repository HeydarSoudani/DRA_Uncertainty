# Belief Model Training

This page describes how the belief model is trained with the agent frozen: what is trained, the three phases of
`experiments/training/belief_model_training.py`, the files each phase writes, and how the trained model is evaluated.

The method follows Agent-BRACE ([arXiv 2605.11436](https://arxiv.org/abs/2605.11436),
[code](https://github.com/joykirat18/Agent-BRACE)) adapted to retrieval: the environment is the retriever, an
observation is the passages of one search, and the belief is one status per fixed criterion of the query.
Agent-BRACE trains the belief model and the policy together; here only the belief model is trained, and the
agent (glm, oss, ...) stays frozen.

## What is trained

The belief model of `src/belief_model_component`: one model with two calls, as Agent-BRACE's single belief model
writes both the content of the belief and its certainty.

- **Init** (`LLMCriteriaSource`), once per query: f(query) -> the criteria C, each with the status `unknown`.
  Its prompt is the criteria extraction prompt of the query's shape; its reply is the JSON list of criteria.
  C stays fixed for the whole search.
- **Update** (`LLMBeliefUpdater`), every search iteration: b_t = f(query, b_{t-1}, o_t). Its prompt holds the
  query, each criterion with its status and evidence, and the passages of the step not judged before; its reply
  is the JSON list of updates.

The query targeting scorer (criteria attempts) is not part of the belief model: it stays a frozen prompted model
(`shared.query_scorer_model`, default the teacher), never the student. A trained belief model is used at
inference with `--llm-criteria vllm/<served name> --belief-judge-model vllm/<served name>`, a separate
`--query-scorer-model`, and `BELIEF_VLLM_API_BASE=http://host:port/v1`.

## Phases

One phase per call; each reads the previous phase's output from `{output_root}/{run_name}/`.

| Phase | Does | Output |
|---|---|---|
| 1 | frozen agent + teacher belief model, `monitor` mode, on the training pool; episodes, SFT samples, teacher scores | `phase1_trajectories/` |
| 2 | SFT warm-up of the student (LoRA, then merged) | `phase2_sft/merged/` |
| 3 | RL of the student (GRPO on teacher-forced belief updates) | `phase3_rl/final/` |

```bash
python experiments/training/belief_model_training.py --phase 1 --run-name pool_glm --policy-model glm
torchrun --nproc_per_node 4 experiments/training/belief_model_training.py --phase 2 --run-name pool_glm
CUDA_VISIBLE_DEVICES=0 python experiments/training/belief_model_training.py --phase 3 --run-name pool_glm
# SLURM: sbatch --export=ALL,PHASE=1,RUN_NAME=pool_glm scripts/run_belief_model_training.sh
```

### Training data

The training data is the training pool built by `experiments/training/build_training_pool.py` (`data/training_pool`,
dataset `training_pool`, split `train`). `dra_inference.py` reads its queries, qrels and corpus from there and
its index from `data/training_pool/indices`; the criteria bank of the pool is written to
`data/training_pool/criteria_bank/`.

### Phase 1: teacher trajectories

`dra_inference.py` runs the agent with the teacher belief model (`--teacher-belief-model`, both calls; its
criteria are read from the criteria bank when banked) in `monitor` mode on every training pool query with qrels,
or the next `--limit` of them; inference resume applies, so a call after the pool has grown runs only the new
questions. In monitor mode the agent
never sees the belief, so each run gives a fixed sequence of observations: the documents of every search, in the
order the belief model saw them.

Each `belief/{qid}.jsonl` becomes an episode. It carries the gold of the RL rewards, so phase 3 reads no qrels
or run files: per step the gold documents first seen in it (`new_gold_ids`, from the qrels, checked against the
saved `recall_so_far`), the passages of its replayed updater call (`shown_doc_ids`) and `recall_so_far`; per episode the correctness of the final answer (`correct`, from the
run's `accuracy.jsonl`; null when the judge failed). Each query gets a use, `sft`, `rl` or `val`, from a hash of
its id (`trajectories.sft_share` 0.30, `trajectories.val_share` 0.05, the rest RL), so it keeps its use when the
pool grows. Only `sft` episodes give SFT samples (the `val` episodes give the validation samples), of both calls:

- **Init**, one per episode whose criteria the teacher wrote: the criteria extraction prompt, rebuilt from the
  query, its shape and the criteria cap, and the criteria as the reply (`{"criteria": [...]}`, the format the
  prompt asks for). The bank keeps the parsed list, not the raw reply, so the target is that list; a sample is
  kept only when the target parses back to the same criteria.
- **Update.** Document text is read back from the corpus (`CorpusLookup`), and the teacher's saved replies are
  replayed through the inference updater code. This gives the exact prompt of every teacher call and checks the
  replay against the saved `criteria_updates` (statuses, cited passages, span checks); an episode is kept up to
  its first step that differs. Every teacher reply that parsed, and whose cited spans all occur in their passage
  (`require_verified_spans`), is an SFT sample.

- `episodes.jsonl`: per query its criteria, the model that wrote them and the criteria cap, the agent, the pool
  source, `num_relevant`, `correct` and `use`; per step the document ids, the teacher's reply, `recall_so_far`
  and `new_gold_ids`; and how many steps the replay matched. Every use.
- `sft.jsonl`: `{qid, dataset, split, call, iteration, messages, completion}`; `call` is `init` (iteration 0) or
  `update`. `sft` episodes only.
- `sft_val.jsonl`: the same, `val` episodes only (SFT validation).
- `teacher_scores.json`: the teacher on all episodes (`scoring.assess`): precision / recall / F1 of the documents
  cited as support against the new gold documents; coverage C_t against `recall_so_far` (mean signed error,
  RMSE, Kendall tau-b); final coverage against correctness (AUROC, Brier, ECE).

Phase 1 prints, per use, the episodes, the share with a gold document found, correct / wrong / not judged, steps
and teacher calls per episode, and the agents and sources.

### Phase 2: SFT

As Agent-BRACE's SFT script: the input is the chat-template prompt the inference server builds (native reasoning
off), then the teacher's reply and the end-of-turn token; the loss is on the reply only, averaged over the reply
tokens of the batch. Init and update samples form one set, unweighted (about one init sample per episode against
one update sample per step). LoRA (r=64, alpha=128, all projections), lr 1e-4 (the usual LoRA rate; Agent-BRACE
uses 2e-5), batch 1 x 32 accumulation per GPU (Agent-BRACE's 4 x 8, one sample per forward pass so that samples
up to `shared.max_seq_len` = 32768 tokens fit), at most 3 epochs. After every epoch a checkpoint is saved and the
loss on `sft_val.jsonl` computed (all samples, and init and update apart); the checkpoint with the lowest loss is
merged into full weights (the last one without validation samples).

### Phase 3: RL

GRPO on teacher-forced belief updates, with the PPO update of Agent-BRACE's `BeliefStateLMTrainer`, outside
veRL. The init call is not trained; the criteria stay as learned in SFT.

- **Prompts.** A step item is one updater call of an `rl` episode, within its matched replay. Its prompt is
  built from the teacher's belief b_{t-1} (the teacher's replies replayed up to the step); the agent ran in
  monitor mode, so the documents do not depend on the belief. Each batch draws `rl.prompts_per_batch` items:
  `rl.gold_step_share` from calls that show gold documents, at most `rl.no_gold_share` from episodes without
  gold, the rest from the other steps; the last updater calls of episodes are balanced on correctness.
- **Groups.** `rl.group_size` replies per prompt from vLLM at temperature 1.0 (one request with `n`), at most
  `rl.max_new_tokens` = 8192 tokens (the inference updater's budget).
  Each reply is applied to a copy of b_{t-1}, which gives its statuses and cited support.
- **Reward** (`rewards.py`): R_t = f_t times the weighted mean of the terms available at the step. f_t: the
  reply was not cut off, parses, gives valid statuses to known criteria once, and all cited spans occur in their
  passage. r_st: F1 of the documents cited as support against the gold documents among the call's passages
  (`Episode.shown_gold`: a document of a failed or unparsed call stays unjudged and counts in the call that shows
  it again; only when either is non-empty). r_sc: 1 - (C_t - recall_so_far)^2. r_ts: 1 - (C_T - y)^2 on the episode's last
  updater call, when y is known and the whole episode replayed. Equal weights (`rl.reward_weight_*`).
- **Update.** GRPO advantage: each reward normalized within its group (0 when the group's rewards are equal),
  given to every reply token; log pi_old recomputed by the trainer before the update (vLLM's log-probs only
  checked against it, `rollout_logprob_gap`), at the sampling temperature; dual-clip PPO (clip 0.2, clip_c 3.0),
  token-mean over each mini-batch's reply tokens, mini-batches of 32 (8 optimizer steps per iteration), one epoch,
  grad clip 1.0, lr 1e-5, full parameters in bf16 with an AdamW of fp32 moments and stochastic rounding into the
  bf16 weights (plain bf16 AdamW drops updates below bf16's resolution). No critic, no KL to a reference (KL can
  be switched on).
- **One GPU.** vLLM sleeps during the update; the new weights go through `/dev/shm` and are reloaded in place.
- **Validation.** Before training and every `rl.val_every` iterations: one greedy reply per step of the `val`
  episodes (at most `rl.val_max_prompts`), its mean reward and terms and the measures of `scoring.measures`.
  The teacher's replies to the same steps are scored once, as the bar. `final/` is the checkpoint with the best
  validation reward.

`metrics.jsonl` has one line per iteration (parse and truncation rates, reply lengths, reward and each term,
groups with zero spread, PPO statistics); `validation.jsonl` the teacher, then one line per validation.

### Evaluation

There is no evaluation phase: the trained model is evaluated with `dra_inference.py` (agent frozen, no / teacher /
student belief model in `inform` mode; the student served by vLLM, `--criteria-bank false`, the phase 1 query
scorer). The commands are in `experiments/training/README.md`.

## Settings

`experiments/configs/belief_model_training.yaml` has one section per phase plus `shared`. Precedence: defaults <
YAML < CLI flags < `--set section.key=value`. The CLI lists the shared flags first, then per phase the models
(query scorer; agent, teacher belief model, student). The agent and retriever settings of phase 1 come from `experiments/configs/dra_inference.yaml` and are never repeated. Each phase saves its merged
settings as `phase{N}_config.yaml`.

## Code

`src/belief_model_training/`: `config`, `episodes` (episodes, `BeliefTracker`, teacher
replay), `trajectories` (phase 1), `scoring` (teacher scores, validation; the rewards' measures), `sft` (phase 2), `rollout` (vLLM), `env`, `rewards`, `ppo` (losses from
veRL), `trainer`, `rl` (phase 3).

Phase 2 needs `peft`.
