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
| 0 | `build_training_pool.py --step N` | builds the training pool: questions, gold pages, qrels | `data/training_pool/` |
| 1 | `belief_model_training.py --phase 1` | runs the frozen agent with the teacher belief model on the training pool; writes episodes, SFT samples and teacher scores | `phase1_trajectories/` |
| 2 | `belief_model_training.py --phase 2` | SFT of the student on the teacher's replies (LoRA, then merged) | `phase2_sft/merged/` |
| 3 | `belief_model_training.py --phase 3` | RL of the student (GRPO on belief updates replayed from the teacher's trajectories) | `phase3_rl/final/` |

The outputs of phases 1 to 3 go to `{output_root}/{run_name}/`. Evaluation of the trained model:
`dra_inference.py` (see [Evaluation](#evaluation-dra_inference)).

```bash
python experiments/training/build_training_pool.py --step 1
python experiments/training/build_training_pool.py --step 2
python experiments/training/build_training_pool.py --step 3
python experiments/training/belief_model_training.py --phase 1 --run-name pool_glm --policy-model glm
torchrun --nproc_per_node 4 experiments/training/belief_model_training.py --phase 2 --run-name pool_glm
CUDA_VISIBLE_DEVICES=0 python experiments/training/belief_model_training.py --phase 3 --run-name pool_glm
```

## Phase 0: Training pool

The training pool (dataset `training_pool`, split `train`) holds deep-research questions whose gold evidence spans
several documents, from InfoSeek-Hard, WebShaper and FRAMES. Classic QA sets (NQ, HotpotQA, 2WikiMultiHopQA,
MuSiQue, TriviaQA) are not used: they need too little search.

| Step | Does | Output |
|---|---|---|
| 1 | merges the sources, drops duplicates and two FRAMES question types (tabular reasoning, post processing); 19,124 questions kept | `questions.jsonl` |
| 2 | fetches the gold pages (Wikipedia API; other web pages with a Wayback Machine fallback) and applies Filter A: a question is kept when all its gold pages are fetched, or at least 80% of them when it has web gold pages | `queries/`, `qrels/`, `corpus/`, `pages/` |
| 3 | prints the statistics per source (queries, documents per query, lengths, checks) | log table |

Every step writes its section of `stats.json`; steps 1 and 2 write their dropped questions to `dropped.jsonl`.
Step 2 is incremental: it skips the questions already processed, and `--limit N` takes the next N. The corpus
holds the gold pages only (no distractors yet).

## Phase 1: Trajectory collection

The frozen agent runs on the training pool with the teacher belief model (`openrouter/qwen/qwen3.6-27b`) in
monitor mode, i.e. the agent never sees the belief. Each query gives one episode. Because the belief does not
change what the agent does, the documents of every step are fixed, so one set of trajectories serves both later
phases: SFT trains on the teacher's replies, and RL replays each step with the student in place of the teacher.

Each episode in `episodes.jsonl` stores:

| Level | Field | Source | Used by |
|---|---|---|---|
| step | `doc_ids`: documents observed in the step; `teacher_output`: the teacher's reply | belief file | SFT, RL replay |
| step | `new_gold_ids`: relevant documents first seen in the step | qrels | recall check |
| step | `shown_doc_ids`: passages shown in the step's updater call; their relevant ones form $N_t$ | replay | $r_{\mathrm{st}}$ |
| step | `recall_so_far` ($\rho_t$) | belief file | $r_{\mathrm{sc}}$ |
| episode | `correct` ($y$; null when the judge failed) | run's `accuracy.jsonl` | $r_{\mathrm{ts}}$ |
| episode | `num_relevant`, `policy_model`, `source` | belief file, query id | checks, batch mixing |
| episode | `use`: `sft` / `rl` / `val` | hash of the query id (30% / 65% / 5%) | phases 2, 3 |

The `new_gold_ids` are checked against the saved `recall_so_far`, and each teacher reply is replayed through the
inference updater to rebuild its exact prompt and check it against the saved updates. The `sft` episodes give the
SFT samples (`sft.jsonl`) and the `val` episodes the SFT validation samples (`sft_val.jsonl`). Phase 1 prints,
per use: episodes, share with gold, correct / wrong, steps and teacher calls per episode, agents and sources.

### Teacher scores

SFT copies the teacher, errors included, so phase 1 scores the teacher on all episodes before any training
(`teacher_scores.json`). A belief update is scored on three parts, each with a standard measure:

| Part | Measure |
|---|---|
| which documents count as evidence | precision / recall / F1 of the documents cited as support (with a verified span) against $N_t$, pooled over steps |
| how far the coverage moves | coverage $C_t$ against $\rho_t$: mean signed error (> 0 = ahead of recall), RMSE, Kendall τ-b over all steps |
| where the coverage ends | final coverage $C_T$ against $y$: AUROC, Brier score, ECE (10 bins) |

The phase 3 rewards are the per-sample forms of these measures (`belief_model_training/scoring.py`). Not measured
yet: whether the right criterion moved by the right amount (this needs labels linking gold documents to criteria).

**Sizes.** About 1,000 sft, 2,000 rl and 200 val episodes, so about 4,500 pool queries (pilot with `--limit 500`
first). With several agents, there is one phase 1 run per agent.

The cap on episodes without gold (about 15%) and the correct / wrong balance are applied when phase 3 samples its
batches, not in phase 1.

## Phase 2: SFT

A sample is one teacher call: its prompt $x$ (masked from the loss) and the teacher's reply $z$ (trained). A
trajectory with K search steps gives at most K + 1 samples: one init call ($q \to \mathcal{C}$) and one per step whose
new documents the teacher judged. The agent's queries and final answer are never targets. The student learns the
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
| validation | `sft_val.jsonl`: loss on all / init / update samples after each epoch | picks the epoch and shows whether the few init samples are learned |

**Sizes.** About 1,000 sft episodes × (1 + teacher calls per episode) samples, i.e. about 9k at 8 calls per
episode.

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
- $T$: the episode's last updater call; $y \in \{0, 1\}$: the agent's final answer is correct.

**Prompts and groups.** A prompt $x_t$ is one updater step of an `rl` episode, built from the **teacher's**
belief $b_{t-1}$. The student samples $G = 8$ replies $o_1, \dots, o_G \sim \pi_{\phi_{\text{old}}}(\cdot \mid x_t)$,
and each is applied to a copy of $b_{t-1}$. The advantage of a reply compares it with the other replies to the
same prompt:

$$
\hat{A}_i = \frac{R_i - \operatorname{mean}\left(\{R_k\}_{k=1}^{G}\right)}{\operatorname{std}\left(\{R_k\}_{k=1}^{G}\right)}, \qquad \hat{A}_i = 0 \text{ when all } R_k \text{ are equal}.
$$

A reply is thus judged against other replies to the same step, not against easier or harder steps. The init call
is not trained in RL.

**Reward.** The reward of a reply (`belief_model_training/rewards.py`) is a format gate times the weighted mean of
the terms available at the step, $\mathcal{T}_t \subseteq \{\mathrm{st}, \mathrm{sc}, \mathrm{ts}\}$:

$$
R_t = f_t \cdot \frac{\sum_{k \in \mathcal{T}_t} w_k\, r_k}{\sum_{k \in \mathcal{T}_t} w_k}, \qquad w_{\mathrm{st}} = w_{\mathrm{sc}} = w_{\mathrm{ts}} = 1,
$$

with $R_t = f_t$ when no term is available. The terms are

$$
r_{\mathrm{st}} = \mathrm{F1}(S_t, N_t) = \frac{2\,|S_t \cap N_t|}{|S_t| + |N_t|}, \qquad
r_{\mathrm{sc}} = 1 - (C_t - \rho_t)^2, \qquad
r_{\mathrm{ts}} = 1 - (C_T - y)^2 .
$$

| Term | Measures | Available when |
|---|---|---|
| format $f_t \in \{0, 1\}$ | 1 if the reply is not cut off, parses, gives each known criterion one valid status and all cited spans occur verbatim in their passage | always |
| state tracking $r_{\mathrm{st}}$ | the cited support matches the relevant documents of the step | $N_t$ or $S_t$ non-empty (an empty step does not pay) |
| state correctness $r_{\mathrm{sc}}$ | the coverage follows recall | $\rho_t$ known |
| task success $r_{\mathrm{ts}}$ | the final coverage predicts correctness | last updater call, $y$ known and the whole episode replayed |

The squared errors are minimized by the expected target given the input, so they reward calibrated coverage.

**Objective.** The PPO update of Agent-BRACE with dual clipping, no critic and no KL term, averaged over all reply
tokens of the batch. With the importance ratio
$\eta_{i,j} = \pi_\phi(o_{i,j} \mid x, o_{i,<j}) / \pi_{\phi_{\text{old}}}(o_{i,j} \mid x, o_{i,<j})$:

$$
\mathcal{L}_{\mathrm{RL}}(\phi) = -\frac{1}{\sum_i |o_i|} \sum_i \sum_{j=1}^{|o_i|} \ell_{i,j}, \qquad
\ell_{i,j} =
\begin{cases}
\min\left(\eta_{i,j}\hat{A}_i,\ \operatorname{clip}(\eta_{i,j}, 1-\epsilon, 1+\epsilon)\,\hat{A}_i\right) & \hat{A}_i \ge 0 \\
\max\left(\min\left(\eta_{i,j}\hat{A}_i,\ \operatorname{clip}(\eta_{i,j}, 1-\epsilon, 1+\epsilon)\,\hat{A}_i\right),\ c\,\hat{A}_i\right) & \hat{A}_i < 0
\end{cases}
$$

with $\epsilon = 0.2$, $c = 3$, learning rate 1e-5 and one PPO epoch in mini-batches of 32 replies.

**Batch.** 32 prompts × 8 replies = 256 replies per iteration: half of the prompts from steps with new gold, at
most 15% from episodes without gold, the rest from other steps; last calls are balanced on $y$. 200 iterations ≈
6,400 prompts.

**Validation.** Greedy replies on the `val` steps (at most 400) before training and every 20 iterations: mean
reward, each term and the teacher-score measures, next to the teacher's replies on the same steps. `final/` is the
checkpoint with the best validation reward.

**Open.** $r_{\mathrm{sc}}$ counts relevant documents ($\rho_t$), while a status means covered content; check
`coverage_minus_recall` in the pilot's teacher scores before changing the weights. The training data is
short-answer QA only (the training pool).

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
