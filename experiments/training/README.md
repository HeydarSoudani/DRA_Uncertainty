# Belief Model Training

Training plan of the belief model, with the deep research agent frozen.

## Goal: uncertainty of the deep research agent

The goal is to report an uncertainty score for the deep research agent (DRA), both during its search and for its
final answer. The idea is to distill this uncertainty into a belief state kept by a separate belief model next to
the agent: a list of criteria that a complete answer must satisfy, each with a status that reflects the evidence
found so far.

The uncertainty score is the coverage of the criteria after search step t, the mean status level scaled to
$[0, 1]$ with negative levels counted as 0:

$$
C_t = \frac{1}{|\mathcal{C}|} \sum_{k \in \mathcal{C}} \frac{\max\left(\ell_t(k), 0\right)}{4} \in [0, 1].
$$

In practice, the belief model is trained so that $C_t$ follows the recall $\rho_t$ of relevant documents during
the search, and the final coverage $C_T$ predicts whether the agent's final answer is correct ($y \in \{0, 1\}$).
Only the belief model is trained; the agent and the query targeting scorer stay frozen.

## Model

One model, two calls:

| Call | Input | Output |
|---|---|---|
| init (once per query) | query $q$ | criteria $\mathcal{C}$, each `unknown` |
| update (every search step $t$) | $q$, belief $b_{t-1}$, new documents $D_t$ | belief $b_t$: a status per criterion, with cited spans |

Each status has a signed level $\ell_t(k)$:

| Status | doubtful | unlikely | unknown | possible | probable | almost certain | certain |
|---|---|---|---|---|---|---|---|
| $\ell$ | −2 | −1 | 0 | 1 | 2 | 3 | 4 |

## Phases

| Phase | Script | Does | Output |
|---|---|---|---|
| 0 | `build_training_pool.py --pool P` | builds a training pool, deep_qa (questions, gold pages, qrels, `--step N`) or trqa_validation (merged TRQA validation files), and splits it into `sft` / `rl` / `val` | `data/training_pool/{pool}/` |
| 1 | `belief_model_training.py --phase 1 --split S` | runs dra_inference with the frozen agent on one split: `sft` and `val` in monitor mode with the teacher belief model, `rl` in off mode | `run_outputs/` (dra_inference layout) |
| 2 | `belief_model_training.py --phase 2` | SFT of the student on the teacher's replies of the `sft` run (LoRA, then merged), validated on the `val` run | `phase2_sft/merged/` |
| 3 | `belief_model_training.py --phase 3` | RL of the student (GRPO on belief updates from its own belief chain) on the `rl` run, validated on the `val` run | `phase3_rl/final/` |

`--training-data` picks the pool (`deep_qa`, the default, or `trqa_validation`) and `--policy-model` the agent.
Phase 1 writes only the inference runs; phases 2 and 3 read the runs of that agent and write to
`{output_root}/{training_data}/{policy_model}/`. Evaluation of the trained model: `dra_inference.py` (see
[Evaluation](#evaluation-dra_inference)).

```bash
python experiments/training/build_training_pool.py --pool trqa_validation
T="--training-data trqa_validation --policy-model glm"
python experiments/training/belief_model_training.py --phase 1 $T --split val
python experiments/training/belief_model_training.py --phase 1 $T --split sft --limit 200
python experiments/training/belief_model_training.py --phase 1 $T --split rl --limit 500
torchrun --nproc_per_node 4 experiments/training/belief_model_training.py --phase 2 $T
CUDA_VISIBLE_DEVICES=0 python experiments/training/belief_model_training.py --phase 3 $T
```

## Phase 0: Training pools

Each pool is a dataset of `dra_inference.py` under `data/training_pool/{pool}`. Split `all` holds every query of
the pool. The build (the end of deep_qa step 2, the trqa_validation call) splits it by a hash of the query id into three disjoint splits, with the shares of
the `split` section of `experiments/configs/build_training_pool.yaml`:

| Split | Share | Phase 1 mode | Used by |
|---|---|---|---|
| `sft` | 30% | monitor (teacher) | phase 2: SFT samples |
| `rl` | 60% | off | phase 3: RL prompts |
| `val` | 10% | monitor (teacher) | phase 2: validation loss; phase 3: validation |

A query's split depends on its id only, so a query keeps its split when the pool grows, and `--limit` on one split
never reaches the queries of another.

### deep_qa

The deep_qa pool holds deep-research questions whose gold evidence spans
several documents, from InfoSeek-Hard, WebShaper and FRAMES. Classic QA sets (NQ, HotpotQA, 2WikiMultiHopQA,
MuSiQue, TriviaQA) are not used: they need too little search.

| Source | Questions | Gold pages / question | Gold pages: wiki / web | Question words | Answer words |
|---|---|---|---|---|---|
| InfoSeek-Hard | 18,110 | 3.9 | all wiki | 42 | 2.9 |
| WebShaper | 500 | 8.1 | 22% / 78% | 65 | 3.4 |
| FRAMES | 514 | 3.1 | all wiki | 26 | 5.1 |
| all | 19,124 | 4.0 | 96% / 4% | 43 | 3.0 |

Values are means, after step 1.

```bash
python experiments/training/build_training_pool.py --step 1                               # merge the sources -> questions.jsonl
python experiments/training/build_training_pool.py --step 2 --source infoseek --limit 500 # fetch gold pages of the next 500 InfoSeek questions, Filter A, split all -> sft / rl / val
python experiments/training/build_training_pool.py --step 3                               # print the statistics per source
```

Step 2 ends with the split, so the split files always match the pool. The shares and per-split counts are recorded
in the `split` section of `stats.json`. Other shares (they would move queries between splits) and other page
settings than the earlier step 2 runs' (`wiki_lang`, `min_words`, `min_page_share`, `wayback`, `wayback_timestamp`)
are refused unless `--force` (`--step 2 --force` re-splits). One build of a pool runs at a time (`.build.lock` in
its directory).

### trqa_validation

The trqa_validation pool merges the TRQA wiki1 and wiki2 validation files into
`queries/queries_all.jsonl`, `queries/queries_all_intermediate_info.jsonl` and `qrels/qrels_all.txt`, then splits
them (one call); it searches TRQA's `wiki_partial` corpus.

Left out: the one query whose text is also a TRQA test query, and the 14 wiki2 ids that each name two different questions (28 queries). 1,145 queries remain (91 wiki1, 1,054 wiki2), all with qrels: 346 sft, 688 rl and 111 val.

```bash
python experiments/training/build_training_pool.py --pool trqa_validation
```

## Phase 1: Trajectory collection

Phase 1 runs `dra_inference.py` with the frozen agent on one split of the training pool, every query of the split
or the next `--limit` not run yet (inference resume). The mode follows the split:

| Split | Mode | Belief model |
|---|---|---|
| `sft`, `val` | monitor: the agent never sees the belief | teacher (`openrouter/qwen/qwen3.6-27b`): criteria and belief updates; frozen query scorer |
| `rl` | off | none |

The query scorer (criteria attempts) runs only in monitor mode; training does not read the attempts. The belief
does not change what the agent does, so the documents of every step are fixed, and phase 3 can replay
the steps of an off-mode run with the student. The runs go to dra_inference's output root in its usual layout
(`run_outputs/{pool}_{split}_{retriever}/{agent}/belief-{mode}/`), and phase 1 writes nothing else.

**Episodes.** Phases 2 and 3 read a split's run at their start (`belief_model_training/runs.py`), one episode per
finished query:

| Level | Field | Source | Used by |
|---|---|---|---|
| step | `doc_ids`: documents shown in the search iteration (all its searches; none when they returned nothing); `iteration`, the belief model's step number (from 1), and `agent_iteration`, the agent's own (`iter`) | `trajectory/{qid}.jsonl` (`seen_docs`) | SFT replay, RL replay |
| step | `new_gold_ids`: relevant documents first seen in the step; `recall_so_far` ($\rho_t$) | qrels | $r_{\mathrm{st}}$, $r_{\mathrm{sc}}$ |
| step | `teacher_output`: the teacher's reply (monitor runs) | `belief/{qid}.jsonl` | SFT |
| episode | `correct` ($y$; null when the judge failed) | `accuracy.jsonl` | $r_{\mathrm{ts}}$ |
| episode | criteria and the model that wrote them (monitor runs) | `belief/{qid}.jsonl` | SFT init samples |
| episode | the criteria cap of the init prompt (`belief_model.max_criteria` of the teacher's run: the split's own, the `val` run's for `rl`; the dataset's default when absent) | `run_config.json` | SFT init samples, RL |

Only finished queries are episodes, as dra_inference counts them: the trajectory ends with its meta line and, in a
monitor run, the belief file exists. The qrels are read with the run's `dataset.min_relevance_score`. The runs a phase reads must share
their agent and searcher settings (`run_config.json`, the temperature included): phase 2 checks the sft
and val runs, phase 3 the rl and val runs, and either stops with the keys that differ. All runs take these
settings from the same inference config (`shared.inference_config`), so keep it unchanged across the phase 1
runs. For monitor runs, the loader also checks that the run's belief model is the teacher, that the belief file
(schema 16) has one step per search iteration of the trajectory, joined on `agent_iteration`, with the same step
number and documents, and that the saved `recall_so_far` matches the qrels. Episodes without a verdict in
`accuracy.jsonl` (a run that stopped before its evaluation) are counted and reported; `--eval-only` judges them. It prints per split: episodes, share with gold, correct / wrong, steps (and teacher calls) per episode,
sources.

The cap on episodes without gold (about 15%) and the correct / wrong balance are applied when phase 3 samples its
batches.

## Phase 2: SFT

A sample is one teacher call of the `sft` monitor run: its prompt $x$ (masked from the loss) and the teacher's
reply $z$ (trained). A trajectory with K search steps gives at most K + 1 samples: one init call
($q \to \mathcal{C}$) and one per step whose new documents the teacher judged. Each teacher reply is replayed
through the inference updater to rebuild its exact prompt and check it against the saved updates. The main
process writes the samples to `phase2_sft/samples.jsonl` (`train`: sft split, `val`: val split). The agent's queries and final answer are never targets. The student learns the
output format and the teacher's transitions, errors included; RL then corrects the transitions.

**Loss.** Cross-entropy on the reply tokens and the end-of-turn token, averaged over all reply tokens of the batch
$\mathcal{B}$ (token-mean), with init and update samples in one unweighted set:

$$
\mathcal{L}_{\mathrm{SFT}}(\phi) = -\frac{1}{\sum_{(x,z)\in\mathcal{B}} |z|} \sum_{(x,z)\in\mathcal{B}} \sum_{j=1}^{|z|} \log \pi_\phi\left(z_j \mid x, z_{<j}\right)
$$

| Setting | Value | Why |
|---|---|---|
| student | `Qwen/Qwen3-4B-Instruct-2507` | |
| LoRA r / alpha / dropout | 64 / 128 / 0.05, all projections | Agent-BRACE |
| learning rate | 1e-4 | usual LoRA rate (Agent-BRACE: 2e-5) |
| epochs | at most 3 | a checkpoint per epoch; the one with the lowest validation loss is merged. A short SFT keeps sampling diverse for RL |
| batch | 1 × 32 accumulation per GPU | same size as Agent-BRACE's 4 × 8, one sample per pass so samples up to 32,768 tokens fit |
| validation | the `val` samples: loss on all / init / update samples after each epoch | picks the epoch and shows whether the few init samples are learned |

**Sizes.** On trqa_validation, about 346 sft episodes × (1 + teacher calls per episode) samples, i.e. about 3k at
8 calls per episode.

### Teacher scores

SFT copies the teacher, errors included, so phase 2 scores the teacher on the sft and val episodes before training
(`phase2_sft/teacher_scores.json`). A belief update is scored on three parts, each with a standard measure:

| Part | Measure |
|---|---|
| which documents count as evidence | precision / recall / F1 of the documents cited as support (with a verified span) against $N_t$, pooled over steps |
| how far the coverage moves | coverage $C_t$ against $\rho_t$: mean signed error (> 0 = ahead of recall), RMSE, Kendall τ-b over all steps |
| where the coverage ends | final coverage $C_T$ against $y$: AUROC, Brier score, ECE (10 bins) |

The phase 3 rewards are the per-sample forms of these measures (`belief_model_training/scoring.py`). Not measured
yet: whether the right criterion moved by the right amount (this needs labels linking gold documents to criteria).

Left for later, depending on the teacher scores: keep only teacher replies that score well (e.g. support F1 ≥ τ),
or use a gold-guided teacher (the qrels labels of the new documents as a hint in the teacher's prompt, not in the
student's).

## Phase 3: RL (GRPO)

**Notation.** At updater step t of an episode:

- $N_t$: relevant documents among the passages of the updater call (a document of a failed or unparsed call is
  shown again, and counted, in a later call);
- $S_t$: documents the reply cites as support, with a verified span;
- $\rho_t$: `recall_so_far`, the share of the query's relevant documents seen up to step t;
- $C_t$: coverage after the reply (see [Goal](#goal-uncertainty-of-the-deep-research-agent));
- $T$: the last updater call of the student's chain; $y \in \{0, 1\}$: the agent's final answer is correct.

**Prompts.** A prompt $x_t$ is one updater step of an `rl` episode: a step with a document not shown in an
earlier step. The agent is frozen, so the documents of every step are fixed, and $b_{t-1}$ is the student's own:
its criteria (greedy init call) and its greedy replies to steps $< t$, as at inference. Training on teacher states
would leave the student unprepared for its own earlier errors at inference (exposure bias; DAgger, Ross et al.
2011), and the `rl` run has no teacher. A step without a prompt (no criteria parsed, or no call there in the
student's chain) is left out. $r_{\mathrm{ts}}$ applies on the last updater call of the student's chain: no later step
shows a document not judged yet once the reply parses (a reply that does not parse scores 0 anyway).

**Groups.** The student samples $G = 8$ replies $o_1, \dots, o_G \sim \pi_{\phi_{\text{old}}}(\cdot \mid x_t)$ at
temperature 1, and each is applied to a copy of $b_{t-1}$. A group whose replies all score alike gives no
gradient and is replaced by new prompts, with up to two more rollouts per iteration (DAPO's dynamic sampling).
The init call is not trained in RL.

**Reward.** The reward of a reply (`belief_model_training/rewards.py`) is a format gate times the weighted mean of
the terms available at the step, $\mathcal{T}_t \subseteq \{\mathrm{st}, \mathrm{sc}, \mathrm{ts}\}$:

$$
R_t = f_t \cdot \frac{\sum_{k \in \mathcal{T}_t} w_k\, r_k}{\sum_{k \in \mathcal{T}_t} w_k}, \qquad w_{\mathrm{st}} = w_{\mathrm{sc}} = w_{\mathrm{ts}} = 1,
$$

with

$$
r_{\mathrm{st}} = \mathrm{F1}(S_t, N_t) = \frac{2\,|S_t \cap N_t|}{|S_t| + |N_t|} \ \ (1 \text{ when } S_t = N_t = \emptyset), \qquad
r_{\mathrm{sc}} = 1 - (C_t - \rho_t)^2, \qquad
r_{\mathrm{ts}} = 1 - (C_T - y)^2 .
$$

| Term | Measures | Available when |
|---|---|---|
| format $f_t \in \{0, 1\}$ | 1 if the reply is not cut off, parses, gives each criterion it lists a valid status, names only known criteria, each once, and every cited passage has spans that all occur verbatim in it (a citation without spans fails) | always |
| state tracking $r_{\mathrm{st}}$ | the cited support matches the relevant documents of the step; citing nothing on a step without gold scores 1 | always |
| state correctness $r_{\mathrm{sc}}$ | the coverage follows recall | $\rho_t$ known |
| task success $r_{\mathrm{ts}}$ | the final coverage predicts correctness | the last updater call of the chain, $y$ known |

The squared errors are minimized by the expected target given the input, so they reward calibrated coverage.
$R_t$ is the logged reward and the validation measure.

**Advantage.** Each term is normalized within the group (GDPO, arXiv 2601.05242), so a term with a small
spread, such as the squared errors near 1, is not drowned by F1, and a tiny spread is not blown up by the floor
$\sigma_{\min} = 0.05$. A reply that fails the format gate scores 0 on each available term:

$$
\hat{A}_i = \frac{\sum_{k} w_k\, \dfrac{r_{k,i} - \operatorname{mean}_k}{\max(\operatorname{std}_k, \sigma_{\min})}}{\operatorname{std}_{\text{batch}}\left(\sum_{k} w_k\, \cdots\right)},
$$

with mean and std over the $G$ replies of the group, a term with zero spread counting 0. A reply is thus judged
against the other replies to the same step, not against easier or harder steps.

**Objective.** The PPO update of Agent-BRACE with dual clipping, no critic and no KL term, with DAPO's clip-higher.
Replies with a zero advantage (no gradient) are left out of the update; the loss of a mini-batch $\mathcal{M}$ is
averaged over the reply tokens of its replies (`loss_agg_mode: token-mean`). With the importance ratio
$\eta_{i,j} = \pi_\phi(o_{i,j} \mid x, o_{i,<j}) / \pi_{\phi_{\text{old}}}(o_{i,j} \mid x, o_{i,<j})$:

$$
\mathcal{L}_{\mathrm{RL}}(\phi) = -\frac{1}{\sum_{i \in \mathcal{M}} |o_i|} \sum_{i \in \mathcal{M}} \sum_{j=1}^{|o_i|} \ell_{i,j}, \qquad
\ell_{i,j} =
\begin{cases}
\min\left(\eta_{i,j}\hat{A}_i,\ \operatorname{clip}(\eta_{i,j}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\,\hat{A}_i\right) & \hat{A}_i \ge 0 \\
\max\left(\min\left(\eta_{i,j}\hat{A}_i,\ \operatorname{clip}(\eta_{i,j}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\,\hat{A}_i\right),\ c\,\hat{A}_i\right) & \hat{A}_i < 0
\end{cases}
$$

with $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$, $c = 3$, learning rate 1e-5 and one PPO epoch in
mini-batches of 32 replies with a nonzero advantage. An iteration with no group whose replies differ makes no
update. The mean token entropy of the replies is logged, to watch for entropy collapse.

**Batch.** 32 prompts × 8 replies = 256 replies per iteration: half of the prompts from steps with new gold, at
most 15% from episodes without gold, the rest from other steps; last calls are balanced on $y$. 200 iterations ≈
6,400 prompts.

**Validation.** Greedy, on a fixed sample of at most 50 episodes of the `val` run (their trajectories only),
before training and every 20 iterations: the student's own chain over each whole episode, its criteria, then
every call, with $r_{\mathrm{ts}}$ on its last call. Each validation gives the mean reward, each term and the
teacher-score measures (compare with `phase2_sft/teacher_scores.json`). `final/` is the checkpoint with the best
validation reward.

**Open.** $r_{\mathrm{sc}}$ counts relevant documents ($\rho_t$), while a status means covered content; check
`coverage_minus_recall` in the pilot's teacher scores before changing the weights. The training data is
short-answer QA only (the training pool), with no hard negatives yet.

## Evaluation (dra_inference)

The trained belief model is evaluated with `experiments/inference/dra_inference.py`: the frozen agent with no,
teacher or student belief model in `inform` mode (in monitor mode the scores would not depend on the belief).
Scores are the run's `summary.json`: answer accuracy, Recall@N, GradedRecall@N. Belief quality itself is measured
in phase 3 validation.

```bash
# serve the student on a GPU the inference workers do not use
CUDA_VISIBLE_DEVICES=0 vllm serve $RUN/phase3_rl/final --served-model-name belief-student --port 6020 &
export BELIEF_VLLM_API_BASE=http://127.0.0.1:6020/v1

COMMON="--dataset browsecomp_plus --policy-model glm --query-scorer-model openrouter/qwen/qwen3.6-27b --gpu-ids 1,2,3"
python experiments/inference/dra_inference.py $COMMON --belief-mode off
python experiments/inference/dra_inference.py $COMMON --belief-mode inform \
    --llm-criteria openrouter/qwen/qwen3.6-27b --belief-judge-model openrouter/qwen/qwen3.6-27b
python experiments/inference/dra_inference.py $COMMON --belief-mode inform --criteria-bank false \
    --llm-criteria vllm/belief-student --belief-judge-model vllm/belief-student --output <student output dir>
```

- `--criteria-bank false` for the student: the bank keys on the model name, which is the same for every served
  checkpoint, so a new checkpoint would reuse an older one's criteria.
- `--query-scorer-model` is the phase 1 scorer in every run; its default is the belief model, which would make the
  student score queries too.
- Give the student run its own `--output`: teacher and student runs share the agent, dataset and belief mode.
