# Belief Model Training

Training plan of the belief model, with the deep research agent frozen.

## Goal: uncertainty of the deep research agent

The goal is to report an uncertainty score for the deep research agent (DRA), both during its search and for its
final answer. The idea is to distill this uncertainty into a belief state kept by a separate belief model next to
the agent: a list of criteria that a complete answer must satisfy, and per criterion two values, one per source of
uncertainty (the split of RAGChecker, Ru et al. 2024: claim recall for the retriever, context utilization for the
generator):

| Side | Value per criterion $k$ | Question it answers |
|---|---|---|
| retriever | status, signed level $\ell_t(k)$ | how much of $k$ do the documents found so far give? |
| reasoner | utilized gain $u_t(k)$ | how much of what raised $k$ has the agent used? |

The retriever side is the coverage after search step t, the mean status level scaled to $[0, 1]$ with negative
levels counted as 0:

$$
C_t = \frac{1}{|\mathcal{C}|} \sum_{k \in \mathcal{C}} \frac{\max\left(\ell_t(k), 0\right)}{4} \in [0, 1].
$$

The reasoner side is judged per rise. A rise of $k$ at step $t-1$ is a status change that gains clipped level,
$g = \max(\ell_{t-1}(k), 0) - \max(\ell_{t-2}(k), 0) > 0$. At step $t$ the belief model reads the agent's next turn
(its reasoning and queries, written after it read the passages of $t-1$) and judges whether the turn uses the
information that raised $k$. $u_t(k)$ sums the gains of $k$'s rises judged used, and the utilization is the used
share of all judged gains:

$$
U_t = \frac{\sum_k u_t(k)}{\sum_k \sum_{\text{judged rises of } k} g} \ (1 \text{ before any rise}), \qquad
\hat{y}_T = C_T \cdot U_T .
$$

$u$ only grows: a later contradiction lowers the status, not what the agent used. In practice, the belief model is
trained so that $C_t$ follows the recall $\rho_t$ of relevant documents during the search, its judgments of use
follow whether the agent's turn states the gold intermediate answer, and the final confidence $\hat{y}_T$ predicts
whether the agent's final answer is correct ($y \in \{0, 1\}$). Only the belief model is trained; the agent and the
query targeting scorer stay frozen.

## Model

One model, two calls:

| Call | Input | Output |
|---|---|---|
| init (once per query) | query $q$ | criteria $\mathcal{C}$, each `unknown` |
| update (every search step $t$) | $q$, belief $b_{t-1}$, new documents $D_t$; when criteria rose at $t-1$: those rises and the agent's turn $a_t$ | belief $b_t$: a status per criterion, with cited spans; per rise of $t-1$: utilized or not, with a span of $a_t$ |

A step with rises to judge calls the updater even without a new document, and after the last step one final call
judges the last rises against the agent's answer turn (reasoning and final answer). The judgments only use the
agent's turn and the statuses only the passages (rule 6 of the updater prompt), so the definition works online,
in `inform` mode too.

Each status has a signed level $\ell_t(k)$:

| Status | doubtful | unlikely | unknown | possible | probable | almost certain | certain |
|---|---|---|---|---|---|---|---|
| $\ell$ | −2 | −1 | 0 | 1 | 2 | 3 | 4 |

The criteria follow the query shape of the dataset: `multi_hop` for deep_qa (one criterion per entity on the chain
to the answer, the answer last), `set` for trqa_validation (the complete list of members, then
"any other member: <property>").

## Phases

| Phase | Script | Does | Output |
|---|---|---|---|
| 0 | `build_training_pool.py --pool P` | builds a training pool, deep_qa (questions, gold pages, qrels, `--step N`) or trqa_validation (merged TRQA validation files), with its gold annotation and criterion texts, and splits it into `sft` / `rl` / `val` | `data/training_pool/{pool}/` |
| 1 | `belief_model_training.py --phase 1 --split S` | runs dra_inference with the frozen agent on the queries of one split that have gold criteria: `sft` and `val` in monitor mode with the teacher belief model on the gold criteria, `rl` in off mode | `run_outputs/` (dra_inference layout) |
| 2 | `belief_model_training.py --phase 2` | SFT of the student on the gold criteria (init) and the teacher's replies (update) of the `sft` run (LoRA, then merged), validated on the `val` run | `phase2_sft/merged/` |
| 3 | `belief_model_training.py --phase 3` | RL of the student (GRPO on whole belief chains) on the `rl` run, validated on the `val` run | `phase3_rl/final/` |

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
python experiments/training/build_training_pool.py --step 3 --limit 500                   # hard negatives of the next 500 wiki gold pages
python experiments/training/build_training_pool.py --step 4                               # print the statistics per source
```

Step 3 adds hard negatives to the corpus, which otherwise holds only gold pages. For each Wikipedia gold page it
asks the Wikipedia API for candidates: first `morelike` (the articles with the most similar text), then category
siblings (articles in the page's most specific categories). It keeps at most `negatives.max_per_gold` (10) per gold
page, skipping pages that are gold for any query, disambiguation and list pages, pages titled with the answer, and
pages under `min_words`. Web gold pages get none. Negatives are fetched like the gold pages (Parsoid HTML, same text
extraction). `negatives/index.jsonl` remembers the processed gold pages: a run continues with the next ones in qrels
order, and `--limit N` counts gold pages. `corpus/corpus.jsonl` holds gold pages and hard negatives with their text;
`corpus/ids_gold.txt` and `corpus/ids_hard_negative.txt` list the ids of each. qrels are unchanged. Steps 2 and 3
write the corpus the same way, so a later step 2 run keeps the negatives.

Step 2 ends with the split, so the split files always match the pool. The shares and per-split counts are recorded
in the `split` section of `stats.json`. Other shares (they would move queries between splits) and other page
settings than the earlier step 2 runs' (`wiki_lang`, `min_words`, `min_page_share`, `wayback`, `wayback_timestamp`)
are refused: to change the shares, remove the `split` section of `stats.json` and the next build re-splits; to
change the page settings, rebuild the pool from step 1. Other step 3 settings than the earlier runs' are refused too:
to change them, remove `negatives/index.jsonl` and the `step3` section of `stats.json`. One build of a pool runs at a time (`.build.lock` in
its directory).

### trqa_validation

The trqa_validation pool merges the TRQA wiki1 and wiki2 validation files into
`queries/queries_all.jsonl`, `queries/queries_all_intermediate_info.jsonl` and `qrels/qrels_all.txt`, then splits
them (one call); it searches TRQA's `wiki_partial` corpus.

Left out: the one query whose text is also a TRQA test query, and the 14 wiki2 ids that each name two different questions (28 queries). 1,145 queries remain (91 wiki1, 1,054 wiki2), all with qrels: 346 sft, 688 rl and 111 val.

```bash
python experiments/training/build_training_pool.py --pool trqa_validation
```

### Gold annotation (deep_qa steps 1-2, trqa_validation build)

Each pool has a gold annotation per query: its criteria, each with its gold answers and gold documents, as far as
the source states them (`training_pool/annotations.py`). Every row has the same keys whatever the source; what the
source does not state is an empty value (`[]` or `null`), never a missing key.

| Source | Criteria | `answers` | `doc_ids` |
|---|---|---|---|
| InfoSeek | one per entity of the research tree that the question does not name, children before parents, the answer last (`multi_hop`) | the entity, and its title without a parenthetical | its Wikipedia page |
| WebShaper | one per variable of the formalization, the farthest from the answer first, then the answer (`multi_hop`) | the answer only (final answer) | none (the URLs are the query's) |
| FRAMES | none | | |
| TRQA wiki2 | "the complete list of ..." and "any other member: <property>" (`set`) | the members / their values (`entity_values`) | the members' documents (`entity_values[].passage_ids`) |
| TRQA wiki1 | the same | the same | the same |

`level` follows from the row and selects a group of queries:

| `level` | The row has | Training can use |
|---|---|---|
| `full` | criteria, each with gold answers and gold documents | per criterion: its answers and its documents; the final answer and the qrels |
| `partial` | criteria, some without answers or without documents | the criteria that have them; the final answer and the qrels |
| `query` | no criteria | the final answer and the qrels only |

Queries per level, with the pools as built on 2026-11-01 (deep_qa: the 998 questions step 2 has processed so far,
910 kept; `level` does not depend on the criterion texts):

| Pool | Source | Queries | full | partial | query | Criteria / query (mean) |
|---|---|---|---|---|---|---|
| deep_qa | InfoSeek | 400 | 400 | 0 | 0 | 2.5 |
| deep_qa | WebShaper | 106 | 0 | 106 | 0 | 5.3 |
| deep_qa | FRAMES | 404 | 0 | 0 | 404 | 0 |
| trqa_validation | TRQA wiki1 | 91 | 91 | 0 | 0 | 2 |
| trqa_validation | TRQA wiki2 | 1,054 | 1,054 | 0 | 0 | 2 |

An InfoSeek query is `partial` when an entity of its criteria has no Wikipedia page in the tree, or that page was not
fetched (Filter A keeps an InfoSeek query only when all its pages were fetched, so none is so far). A WebShaper
criterion has no documents, and only the answer criterion has answers. The numbers come from the annotation tables of
step 4 (deep_qa, `step4` section of `stats.json`) and of the trqa_validation build (`build` section). Step 1 builds the criteria from the source (`annotation` of each record in
`questions.jsonl`, with the page keys of their documents); step 2 resolves the keys to the fetched documents (a page
that was not fetched gives none, so a query can fall from full to partial) and writes `gold/annotations_all.jsonl`,
split with the queries into `gold/annotations_{sft,rl,val}.jsonl`. The trqa_validation build writes the same files
from the intermediate info. Query row: `id`, `source`, `query`, `answer`, `query_shape`, `level`, `doc_ids` (the
qrels), `criteria`, `members` (set queries: `entity`, `value`, `doc_ids`), `entity_names` (InfoSeek: every entity of
the tree), `property` (set queries); criterion: `id`, `kind`, `relations` (the source structure the writer reads),
`answers`, `doc_ids`, `text`; then `texts_ok`, `errors`, `model`, `prompt` of the texts. Steps 2 and 4 and the trqa_validation build count the rows per source and
level and the criteria with answers and documents, and check the row format.

### Criterion texts (end of deep_qa step 2 and of the trqa_validation build)

An LLM (`gold_criteria.model`, the teacher) writes the text of each criterion from the question and the criterion's
relations (multi-hop: InfoSeek, WebShaper) or the set's property (TRQA), in the question's words; answers, documents
and level come from the source (`training_pool/gold_criteria.py`). It runs before the split, so every annotation file
holds its texts; step 2 and the trqa_validation build need `OPENROUTER_API_KEY`. `texts_ok` is true when every text
names none of the gold answers (a name inside a longer entity name, such as Groningen in "University of Groningen",
does not count), or when the query has no criteria (FRAMES, no LLM call); false when the texts failed for good (a leak
after one retry with the names given, no text for a criterion); null when the writer call failed (an error, a reply
that is not JSON), and the next run asks again. The number of criteria is the source's (no cap). Final texts are
appended to `gold/texts.jsonl` as they are written, so a stopped run keeps them; a run reuses them for unchanged
criteria and writes only the new queries' texts. Another `gold_criteria.model` or other prompts than those of
`gold/texts.jsonl` are refused (remove the file to write every text again). `--step 2 --limit 0` fetches nothing and
rewrites the dataset files, annotation and split from the questions already processed. `gold/ids_{sft,rl,val}.txt` lists the queries with `texts_ok` and at least one
criterion; phase 1 runs these only. The annotation tables of steps 2 and 4 count the texts per source (ok, failed,
retry).

## Phase 1: Trajectory collection

Phase 1 runs `dra_inference.py` with the frozen agent on the queries of one split that have gold criteria
(`--query-ids-file` = `gold/ids_{split}.txt` of the pool), every such query or the next `--limit` not run yet (inference
resume). The mode follows the split:

| Split | Mode | Belief model |
|---|---|---|
| `sft`, `val` | monitor: the agent never sees the belief | the query's gold criteria (`--gold-criteria true`) and the teacher (`openrouter/qwen/qwen3.6-27b`) for the belief updates (statuses and use); frozen query scorer |
| `rl` | off | none |

The query scorer (criteria attempts) runs only in monitor mode; training does not read the attempts. The belief
does not change what the agent does, so the documents and agent turns of every step are fixed, and phase 3 can run
the student over the steps of an off-mode run. The runs go to dra_inference's output root in its usual layout
(`run_outputs/{pool}_{split}_{retriever}/{agent}/belief-{mode}/`), and phase 1 writes nothing else.

**Episodes.** Phases 2 and 3 read a split's run at their start (`belief_model_training/runs.py`), one episode per
finished query with gold criteria (`texts_ok`, at least one criterion):

| Level | Field | Source | Used by |
|---|---|---|---|
| step | `doc_ids`: documents shown in the search iteration (all its searches; none when they returned nothing); `iteration`, the belief model's step number (from 1), and `agent_iteration`, the agent's own (`iter`) | `trajectory/{qid}.jsonl` (`seen_docs`) | SFT replay, RL chains |
| step | `agent_turn`: the agent's reasoning and queries of the iteration (as the teacher read it in monitor runs) | `trajectory/{qid}.jsonl`, `belief/{qid}.jsonl` | the updater prompt |
| step | `new_gold_ids`: relevant documents first seen in the step; `recall_so_far` ($\rho_t$) | qrels | $r_{\mathrm{ret}}$ |
| step | `stated`: the criteria whose gold answer the step's agent turn states | gold criteria, `utilized_gold.py` | $r_{\mathrm{reas}}$ |
| step | `teacher_output`, `teacher_used`: the teacher's reply and its judgments (monitor runs) | `belief/{qid}.jsonl` | SFT |
| episode | `answer_turn`: reasoning after the last search and the final answer; the teacher's final call on it (monitor runs) | `trajectory/{qid}.jsonl`, `generation/{qid}.md`, `belief/{qid}.jsonl` | final call |
| episode | `correct` ($y$; null when the judge failed) | `accuracy.jsonl` | $r_{\mathrm{final}}$ |
| episode | `gold_criteria`: criteria with `answers` and `doc_ids` (empty when the source states none) | `gold/annotations_{split}.jsonl` | init (SFT target, RL chains), gold of use |

**Gold of use.** A turn states a criterion's gold answer when the answer occurs in it as whole words (case, accents
and punctuation ignored; not inside a longer entity name of the query's gold structure) or, for a number, when a
number of the turn equals it. A (turn, criterion) pair without a match, whose gold documents were shown before the
turn, is asked of the correctness judge (`rl.utilized_judge_model`, YES/NO); the verdicts are cached in
`{output_root}/{training_data}/{policy_model}/utilized_gold.jsonl`. `rl.utilized_judge_model: null` uses the match
only.

Only finished queries are episodes, as dra_inference counts them: the trajectory ends with its meta line and, in a
monitor run, the belief file exists. The qrels are read with the run's `dataset.min_relevance_score`. The runs a phase reads must share
their agent and searcher settings (`run_config.json`, the temperature included): phase 2 checks the sft
and val runs, phase 3 the rl and val runs, and either stops with the keys that differ. All runs take these
settings from the same inference config (`shared.inference_config`), so keep it unchanged across the phase 1
runs. For monitor runs, the loader also checks that the run's updater is the teacher on the gold criteria (and
that the run's criteria are the pool's), that the belief file (schema 17) has one step per search iteration of the
trajectory, joined on `agent_iteration`, with the same step number and documents, and that the saved
`recall_so_far` matches the qrels. Episodes without a verdict in `accuracy.jsonl` (a run that stopped before its
evaluation) are counted and reported; `--eval-only` judges them. It prints per split: episodes, share with gold,
correct / wrong, steps (and teacher calls) per episode, sources, and the counts of the gold of use.

## Phase 2: SFT

A sample is one call of the `sft` monitor run: its prompt $x$ (masked from the loss) and its target $z$ (trained).
A trajectory with K search steps gives at most K + 2 samples: one init call ($q \to \mathcal{C}$, the gold
criteria as the target), one per step the teacher's updater call judged, and the final call on the answer turn.
Each teacher reply is replayed through the inference updater to rebuild its exact prompt and check it against the
saved updates and judgments of use. The main
process writes the samples to `phase2_sft/samples.jsonl` (`train`: sft split, `val`: val split). The agent's queries and final answer are never targets. The student learns the
output format and the teacher's transitions, errors included; RL then corrects the transitions.

**Loss.** Cross-entropy on the reply tokens and the end-of-turn token, averaged over all reply tokens of the batch
$\mathcal{B}$ (token-mean), with init and update samples in one unweighted set:

$$
\mathcal{L}_{\mathrm{SFT}}(\phi) = -\frac{1}{\sum_{(x,z)\in\mathcal{B}} |z|} \sum_{(x,z)\in\mathcal{B}} \sum_{j=1}^{|z|} \log \pi_\phi\left(z_j \mid x, z_{<j}\right)
$$

| Setting | Value | Why |
|---|---|---|
| student | `Qwen/Qwen3.5-4B`, non-thinking | same generation, architecture and tokenizer as the teacher (Qwen3.6-27B); Agent-BRACE's `Qwen/Qwen3-4B-Instruct-2507` stays as a baseline. The vision encoder is kept frozen so the saved weights keep the layout vLLM loads |
| LoRA r / alpha / dropout | 64 / 128 / 0.05, attention, Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `out_proj`) and MLP projections | Agent-BRACE; Qwen3.5's Gated DeltaNet layers are 3 of every 4 layers |
| learning rate | 1e-4 | usual LoRA rate (Agent-BRACE: 2e-5) |
| epochs | at most 3 | a checkpoint per epoch; the one with the lowest validation loss is merged. A short SFT keeps sampling diverse for RL |
| batch | 1 × 32 accumulation per GPU | same size as Agent-BRACE's 4 × 8, one sample per pass so samples up to 32,768 tokens fit |
| validation | the `val` samples: loss on all / init / update samples after each epoch | picks the epoch and shows whether the few init samples are learned |

**Sizes.** On trqa_validation, about 346 sft episodes × (1 + teacher calls per episode) samples, i.e. about 3k at
8 calls per episode.

### Teacher scores

SFT copies the teacher, errors included, so phase 2 scores the teacher on the sft and val episodes before training
(`phase2_sft/teacher_scores.json`, `belief_model_training/scoring.py`):

| Part | Measure |
|---|---|
| retriever | coverage $C_t$ against $\rho_t$: mean signed error (> 0 = ahead of recall), RMSE, Kendall τ-b over all calls; logged: precision / recall / F1 of the documents cited as support against $N_t$ |
| reasoner | the judgments of use against the gold: accuracy, precision / recall / F1 of "utilized", and the gold rate |
| final | $\hat{y}_T = C_T \cdot U_T$ against $y$: AUROC, Brier score, ECE (10 bins); the same for $C_T$ alone, to see what the reasoner side adds |

The phase 3 rewards are the per-sample forms of these measures.

## Phase 3: RL (GRPO)

**Notation.** In a chain over an episode, at updater call k (step $t$):

- $\rho_t$: `recall_so_far`, the share of the query's relevant documents seen up to step t;
- $C_k$: coverage after the call; $U_T$: utilization after the chain's last call;
- $J_k$: the rises the call judges, each with the student's judgment and the gold (the agent turn states the
  criterion's gold answer);
- $y \in \{0, 1\}$: the agent's final answer is correct.

**Chains.** A chain is the student's belief updater run over a whole `rl` episode from its gold criteria (the init
call is not trained in RL): one call per step with a document not judged yet or rises to judge, then the final call
on the answer turn, each prompt built from the chain's own $b_{t-1}$, as at inference. The agent is frozen, so the
documents and agent turns of every step are fixed. Training on the student's own states, not the teacher's, keeps
it ready for its own earlier errors (exposure bias; DAgger, Ross et al. 2011).

**Groups.** The student runs $G = 8$ chains per episode at temperature 1. A group whose chains all score alike
gives no gradient and is replaced by new episodes, with up to two more rollouts per iteration (DAPO's dynamic
sampling).

**Reward.** One reward per chain (`belief_model_training/rewards.py`), over every step and the final belief:

$$
R = w \sum_k r_{\mathrm{step},k} + r_{\mathrm{final}}, \qquad
r_{\mathrm{step},k} = f_k \left(r_{\mathrm{ret},k} + r_{\mathrm{reas},k}\right),
$$

$$
r_{\mathrm{ret},k} = 1 - (C_k - \rho_t)^2, \qquad
r_{\mathrm{reas},k} = \frac{1}{|J_k|} \sum_{j \in J_k} \mathbb{1}[\text{judged}_j = \text{gold}_j] \ \ (1 \text{ when } J_k = \emptyset), \qquad
r_{\mathrm{final}} = f_T \left(1 - (C_T U_T - y)^2\right),
$$

with $w$ = `rl.reward_weight_step` (1).

| Term | Measures | Available when |
|---|---|---|
| format $f_k \in \{0, 1\}$ | 1 if the reply is not cut off, parses, gives each criterion it lists a valid status, names only known criteria, each once, every cited passage has spans that all occur verbatim in it, and every rise it judges has a valid value, with a span that occurs in the agent's turn when used | every call |
| $r_{\mathrm{ret}}$ (retriever) | the coverage follows recall | step calls (not the final call, which has no passage) |
| $r_{\mathrm{reas}}$ (reasoner) | the judgments of use match the gold; a call with nothing to judge cannot get one wrong | every call |
| $r_{\mathrm{final}}$ | the final confidence predicts correctness (Brier) | $y$ known (else 0) |

The squared errors are minimized by the expected target given the input, so they reward calibrated coverage and
confidence. $r_{\mathrm{reas}} = 1$ on a call with nothing to judge keeps a chain from gaining by raising more
statuses. $R$ is the logged reward and the validation measure.

**Advantage.** Each of the three terms (the sums of $f_k r_{\mathrm{ret},k}$ and of $f_k r_{\mathrm{reas},k}$, and
$r_{\mathrm{final}}$) is normalized within the episode's group (GDPO, arXiv 2601.05242) and weighted $w$, $w$, 1,
so a term with a small spread is not drowned and a tiny spread is not blown up by the floor
$\sigma_{\min} = 0.05$:

$$
\hat{A}_i = \frac{\sum_{m} w_m\, \dfrac{r_{m,i} - \operatorname{mean}_m}{\max(\operatorname{std}_m, \sigma_{\min})}}{\operatorname{std}_{\text{batch}}\left(\sum_{m} w_m\, \cdots\right)},
$$

with mean and std over the $G$ chains of the group, a term with zero spread counting 0. Every call of chain $i$
takes $\hat{A}_i$: a chain is judged against the other chains over the same episode.

**Objective.** The PPO update of Agent-BRACE with dual clipping, no critic and no KL term, with DAPO's clip-higher,
over the calls of the chains. Calls with a zero advantage are left out of the update; the loss of a mini-batch
$\mathcal{M}$ of calls is averaged over their reply tokens (`loss_agg_mode: token-mean`). With the importance ratio
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
mini-batches of 32 calls with a nonzero advantage. An iteration with no group whose chains differ makes no update.
The mean token entropy of the replies is logged, to watch for entropy collapse.

**Batch.** 16 episodes × 8 chains per iteration, about 1,000 calls at 8 calls per chain; the episodes are drawn
without replacement, the correct and wrong final answers weighing the same.

**Validation.** Greedy, on a fixed sample of at most 50 episodes of the `val` run (their trajectories only),
before training and every 20 iterations: one chain per episode, scored as in training. Each validation gives the
mean reward, each term and the teacher-score measures (compare with `phase2_sft/teacher_scores.json`). `final/` is
the checkpoint with the best validation reward.

**Open.** $r_{\mathrm{ret}}$ counts relevant documents ($\rho_t$), while a status means covered content; check
`coverage_minus_recall` in the pilot's teacher scores before changing the weights. The gold of use covers InfoSeek
and TRQA only (WebShaper and FRAMES have no gold intermediate answers).

## Evaluation (dra_inference)

The trained belief model is evaluated with `experiments/inference/dra_inference.py`: the frozen agent with no,
teacher or student belief model in `inform` mode (in monitor mode the scores would not depend on the belief).
Scores are the run's `summary.json`: answer accuracy, Recall@N, GradedRecall@N. Belief quality itself is measured
in phase 3 validation. The student is trained on the `multi_hop` (deep_qa) or `set` (trqa_validation) prompts; a
dataset of another query shape (browsecomp_plus: `single_target`) gives it prompts it was not trained on.

```bash
# serve the student on a GPU the inference workers do not use
CUDA_VISIBLE_DEVICES=0 vllm serve $RUN/phase3_rl/final --served-model-name belief-student --port 6020 --language-model-only &
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
