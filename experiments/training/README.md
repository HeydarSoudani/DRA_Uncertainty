# Belief Model Training

This file describes how the belief model is trained: the setup, the seven stages with their commands, then each
stage in turn (training datasets, trajectories, teacher beliefs, filter, SFT, RL, evaluation). The belief model itself (its calls, prompts and
belief state) is described in `src/belief_model_component/README.md`.

## Setup

The deep research agent is frozen; only the belief model is trained. The recipe follows Agent-BRACE
(arXiv 2605.11436): SFT on a teacher's transitions, then RL of the belief model with a group-normalized advantage and
the PPO clip.

| | |
|---|---|
| Student | `qwen3.5_4b`, Qwen3.5-4B (`shared.student` or `--student`, a local base `--belief-model`), one model for every call |
| Teacher | `qwen3.6_27b` (`shared.teacher` or `--teacher`, a `--belief-model` of dra_inference) |
| Calls trained | belief update and final score; belief init in the own-init RL run (and from gold beliefs in SFT) |
| Prompts | the inference prompts, built by the inference belief model itself |
| Training data | `infoseek_hard` (multi_hop) and `trqa_val` (multi_member), pooled |
| Agents | `oss_20b` and `glm` (`shared.policy_models`) |
| RL reward | gate x Brier of the final score against the correctness of the agent's answer; no reward on belief transitions |

Every training call runs the inference `BeliefModel` (`belief_model_component`) over a saved trajectory
(`src/belief_model_training/chain.py`), driven as the belief stage drives it (reset, one `observe` per search
iteration, `finish_answer` on the saved reasoning and answer); only its LLM client is swapped. A replay client returns
the teacher's saved replies (SFT), a sampling client the student's replies (RL). The init source, the updater and the
final scorer each get their own view of that client, so every call is recorded with the kind of the component that
made it (the final update is an update call of step N + 1, the final score a score call of the same step). So the
prompts of training are the prompts of inference, and neither the belief model, the agents nor dra_inference has
training code.

## Stages

`experiments/training/train_belief_model.py --stage NAME`, one stage per call; settings in
`experiments/configs/belief_training.yaml` (sections `shared`, `data`, `generate`, `belief_stage`, `filter`, `sft`,
`rl`); to change a setting, edit the YAML or pass another file with `--config`. The flags `--teacher`
(`shared.teacher`), `--student` (`shared.student`), `--init-model` (`rl.init_model`) and `--beliefs` (`rl.beliefs`)
overwrite their YAML keys. Launcher: `scripts/run_train_belief_model.sh` (one job per stage, `STAGE=...`).

| Stage | Does | Flags | Output |
|---|---|---|---|
| `data` | builds a training dataset with its gold annotation and its sft / rl / val split | `--dataset` (required), `--limit` | `data/training_datasets/{dataset}/` |
| `generate` | dra_inference `--stage generate` on a split for each agent; then the run statistics | `--dataset`, `--subset` (required; sft, rl, val), `--policy-model`, `--limit` | the runs (dra_inference layout), `stats/` |
| `teacher` | dra_inference `--stage belief --belief-model {teacher}` on the sft or val split for each agent | `--dataset`, `--subset` (required; sft, val), `--policy-model`, `--limit`, `--teacher` | the runs' `belief/{teacher}/`, `eval/metrics.md` |
| `filter` | what is trained on: SFT samples from the teacher's replies, RL episodes (CPU) | `--teacher` | `filter/` |
| `sft` | LoRA SFT of the student on the filtered samples, merged | `--student` | `{student}/sft/final` |
| `rl` | GRPO over belief chains of the filtered episodes, from the student's `sft/final` | `--student`, `--init-model`, `--beliefs` | `{student}/rl/final` or `{student}/rl_own_init/final` |
| `evaluate` | dra_inference `--stage belief --belief-model M` on a run | `--dataset`, `--belief-model` (required), `--subset`, `--policy-model`, `--limit` | the run's `eval/metrics.md` |

A stage refuses the flags it does not take. `--limit` means the next N questions in `data` and the queries the
dra_inference stage runs in `generate`, `teacher` and `evaluate`. `sft` and `rl` take no dataset flag: they pool every
training dataset and agent of `shared`, on fixed splits.

The stages generate, filter, sft and rl write under `BELIEF_TRAINING_ROOT` (`utils/config.py`:
`OUTPUT_ROOT/belief_training`). The `final/` of each training run is served by dra_inference on local vLLM as a
`--belief-model` named `{student}_{run}`: `qwen3.5_4b_sft`, `qwen3.5_4b_rl`, `qwen3.5_4b_rl_own_init`
(`utils.config.TRAINED_BELIEF_RUNS`, the students in `STUDENT_BELIEF_MODELS`).

```bash
T=experiments/training/train_belief_model.py
python $T --stage data --dataset infoseek_hard --limit 500
python $T --stage data --dataset trqa_val
python $T --stage generate --dataset infoseek_hard --subset sft     # each agent
python $T --stage generate --dataset infoseek_hard --subset val
python $T --stage generate --dataset infoseek_hard --subset rl
python $T --stage teacher --dataset infoseek_hard --subset sft      # each agent; teacher beliefs
python $T --stage teacher --dataset infoseek_hard --subset val
python $T --stage filter
torchrun --nproc_per_node 4 $T --stage sft
python $T --stage rl                                          # rl.beliefs gold: run rl
python $T --stage rl --beliefs own                            # run rl_own_init
python $T --stage evaluate --belief-model qwen3.5_4b_rl --dataset infoseek_hard --subset val
python $T --stage evaluate --belief-model qwen3.5_4b --dataset browsecomp_plus   # a baseline
```

## Stage data: Training datasets

Each training dataset is a dataset of `dra_inference.py` under `data/training_datasets/{dataset}`, with one row per
query:

| File | Contents |
|---|---|
| `queries/queries_all.jsonl` | one row per query: the question and answer, its source, its split, its gold annotation (see [Gold annotation](#gold-annotation)) |
| `qrels/qrels_all.txt` | the gold documents of each query (TREC qrels, binary) |
| `belief_texts.jsonl`, `stats.json`, `dropped.jsonl` | build files: the writer's saved belief texts, the counts and settings of each build stage, the dropped questions with their stage and reasons |

| Dataset | Questions | Corpus | Splits |
|---|---|---|---|
| `infoseek_hard` | InfoSeek-Hard | the original release's `wiki-25-512.jsonl` (Wikipedia 2025 in chunks of 512 tokens), under `DATA_ROOT/infoseek/corpus` with its indices | sft / rl / val |
| `trqa_val` | TRQA wiki1 + wiki2 validation | TRQA's `wiki_partial` | sft / rl / val |

One build call per dataset; every build is incremental, like dra_inference: `--limit N` processes the next N
questions not processed yet, `--limit 0` processes none and rewrites the dataset files, no `--limit` processes all.

```bash
python experiments/training/train_belief_model.py --stage data --dataset infoseek_hard --limit 500
python experiments/training/train_belief_model.py --stage data --dataset trqa_val
```

The splits `sft`, `rl` and `val` are views of split `all`: the queries loader reads `--subset sft` as the rows with
`split` `sft` and their qrels (`layout.is_split_view`), so dra_inference runs a split as on any dataset and its run
folders are named by the split. Every build sets each row's `split`. Training uses the queries with the full gold annotation
(`level` `full`: every belief with gold answers and gold documents), so the full queries of `infoseek_hard` and `trqa_val` are split with the shares of the
`data.split` section of `experiments/configs/belief_training.yaml`, and every other query goes to `val` as its
out-of-domain (OOD) part:

| Split | Queries | Stages | Used by |
|---|---|---|---|
| `sft` | 30% of the full ones | generate, teacher | sft: SFT samples |
| `rl` | 60% of the full ones | generate | rl: RL episodes |
| `val` | in domain: 10% of the full ones | generate, teacher | sft: validation loss; rl: validation; evaluate |

A full query without belief texts (`texts_ok` not true: not written yet, or failed) has `split` null and is in no
split until a later build writes its texts. A query keeps the split it has in the queries file, so no query moves
when the dataset grows, and `--limit` on one split never reaches the queries of another. The new full queries of a
build are taken in the order of a hash of their id (not the source order), and each goes to the split furthest below
its share, so the shares hold at every dataset size (20 full queries: 6 sft, 12 rl, 2 val; 30: 9, 18, 3). The
shares and per-split counts are recorded in the `split` section of `stats.json`; other shares are refused (remove
the `split` section to re-split every query). One build of a dataset runs at a time (`.build.lock` in its folder).

Classic QA sets (NQ, HotpotQA, 2WikiMultiHopQA, MuSiQue, TriviaQA) are not used: they need too little search.

| Source | Questions | Gold pages / question | Gold pages: wiki / web | Question words | Answer words |
|---|---|---|---|---|---|
| InfoSeek-Hard | 18,110 | 3.9 | all wiki | 42 | 2.9 |

### infoseek_hard

The InfoSeek-Hard questions (`Lk123/InfoSeek`, `data/InfoSeek-Hard-18K.jsonl`, pinned), run over the corpus their
authors search (VectorSpaceLab/InfoSeek, `research/InfoSeek/rl/InfoSeek/Retrieve`): `wiki-25-512.jsonl` of
`Lk123/wiki-25-512` (pinned), Wikipedia 2025 split into chunks of 512 tokens, each chunk with its page title on the
first line. The first build downloads it (18.5 GB) into `DATA_ROOT/infoseek/corpus/`; it is used unchanged, a doc id
is the chunk's `id` (its line, the row of the released index) as a string. The indices go next to it
(`DATA_ROOT/infoseek/indices`).

InfoSeek records each hop's gold page as a Wikipedia title (`href` of a research tree node), with no corpus id and no
position in the page. The build matches each title exactly against the first line of every chunk (Unicode NFC,
underscores as spaces, first letter upper case on both sides) and keeps the page's **first chunk** (lowest id) as
its gold document. InfoSeek's construction code (`ConstructTreeData.py`) writes every claim from the first 1024
tokens of the page of the node it expands and picks the link "from the front part of the wiki page"; on the first
30 questions, all 91 claimed links (B named on A's page) are in the first chunk of A's page, and none in a later
chunk, so the later chunks hold no evidence and would only make long pages weigh more in recall. The query's qrels
are the first chunk of every gold page. A hop's gold documents are the first chunk of its own page and of the
pages whose claims link to it: the link to a hop is stated on the linking node's page. A question with a gold page that is not in the corpus (renamed,
redirected or missing in the 2025 dump) is dropped (`title_not_in_corpus` in `dropped.jsonl`). The titles looked up
are kept in `corpus_titles.jsonl`, so a build reads the corpus once, and only when it has new titles. The `corpus`
section of `stats.json` records the corpus revision (another revision is refused) and the titles found.

### webshaper_frames (not registered)

`src/indexing_corpus_dataset/training_datasets/webshaper_frames/` builds WebShaper and FRAMES questions with their
gold pages fetched from the source and hard negatives of the Wikipedia gold pages (validation only). The code is kept
but not registered: it has no `layout.DATASET_SPECS` entry and is not a stage data choice, and its settings (`shared`,
`pages`, `negatives` of the build config) keep their defaults and are not in the YAML.

### trqa_val

The trqa_val dataset merges the TRQA wiki1 and wiki2 validation files (queries, intermediate info, qrels)
into `queries/queries_all.jsonl` and `qrels/qrels_all.txt`: a query and its intermediate info become one row, the
members with their values and documents in `members`, with `property` (the label) and `aggregation`. It searches
TRQA's `wiki_partial` corpus. The gold loaders (`load_query_intermediate_info`) read the intermediate info of a
training dataset from its rows, in TRQA's record format.

Left out: the one query whose text is also a TRQA test query, and the 14 wiki2 ids that each name two different
questions (28 queries). 1,145 queries remain (91 wiki1, 1,054 wiki2), all with qrels and all full, so `val` has no
OOD part: 343 sft, 687 rl and 115 val. `--limit N` adds the next N kept queries not yet in the dataset (file order,
wiki1 then wiki2). The `build` section of `stats.json` records the queries in the dataset and the ones remaining.
Its rows have the keys of the other training datasets' (source `trqa_wiki1` / `trqa_wiki2`).

### Gold annotation

Each training dataset has a gold annotation per query: its beliefs, each with its gold answers and gold documents,
as far as the source states them (`training_datasets/gold/annotations.py`). Every row has the same keys whatever the
source; what the source does not state is an empty value (`[]` or `null`), never a missing key.

| Source | Beliefs | `answers` | `doc_ids` |
|---|---|---|---|
| InfoSeek | one per entity of the research tree that the question does not name, children before parents, the answer last (`multi_hop`) | the entity, and its title without a parenthetical | the first corpus chunk of its Wikipedia page and of the pages whose claims link to it |
| WebShaper | one per variable of the formalization, the farthest from the answer first, then the answer (`multi_hop`) | the answer only (final answer) | none (the URLs are the query's) |
| FRAMES | none | | |
| TRQA wiki2 | one per member, "<member>: <property>", the member named by the Wikipedia title of its first document ("San Fernando, La Union", not TRQA's label "San Fernando") (`closed`); then the aggregation, how the answer follows from the values (`aggregate`) (`multi_member`) | the member's value (`entity_values`); aggregate: the final answer | the member's documents (`entity_values[].passage_ids`); aggregate: none |
| TRQA wiki1 | the same | the same | the same |

Every TRQA member has its own belief. The aggregate belief has no documents since no document states the
computed value; a member the extractor did not list reaches the belief as a passage observation on the aggregate.

`level` follows from the row and selects a group of queries:

| `level` | The row has | Training can use |
|---|---|---|
| `full` | beliefs, each with its gold: gold answers and gold documents (an `aggregate` one answers only) | per belief: its answers and its documents; the final answer and the qrels |
| `partial` | beliefs, some without answers or without documents | the beliefs that have them; the final answer and the qrels |
| `query` | no beliefs | the final answer and the qrels only |

An InfoSeek query is `partial` when an entity of its beliefs has no Wikipedia page in the tree (a question with a
page missing from the corpus is dropped). A WebShaper belief has no documents, and only the answer belief has
answers. The `annotations` section of `stats.json`, the same in every dataset, counts the rows per source and level,
the beliefs per kind and with their answers, documents and gold, the texts per source (ok, failed, retry), and
records the writer of the texts. The loaders build the beliefs from the source (with the page keys of their
documents); the build resolves the keys to documents (infoseek_hard: the first chunk of the page) and writes the rows
to `queries/queries_all.jsonl`. The trqa_val build
writes the same rows from the intermediate info. Query row: `id`, `source`, `split`, `text` (the question),
`answer`, `query_shape`, `level`, `doc_ids` (the qrels), `beliefs`, `members` (multi-member queries: `entity`,
`value`, `doc_ids`), `entity_names` (InfoSeek: every entity of the tree; TRQA: the members), `property` and
`aggregation` (multi-member queries); belief: `id`, `text`, `answers`, `doc_ids`, `kind` (`closed`,
`aggregate`), `relations` (the source structure the writer reads); then `texts_ok`, `errors` of the texts (the writer
and its prompt hash are in `stats.json` and `belief_texts.jsonl`, not per row).

### Belief texts (end of every build)

An LLM (`gold_beliefs.model`, the teacher) writes the belief texts in the question's words, one call per query;
answers, documents and level come from the source (`training_datasets/gold/belief_texts.py`). The prompts are the files
of `training_datasets/gold/prompts/`, one system and one user prompt per query shape:

- multi-hop (InfoSeek, WebShaper): one text per belief, from its relations, with only the details the question
  states (a relation may hold more, such as a year). The writer orders the beliefs so that each refers only to
  beliefs before it ("the town of B1"), the answer last; the row's beliefs are reordered and renumbered in that
  order (saved as `order` in `belief_texts.jsonl`). An invalid order is asked once more with the problem named.
- set (TRQA): the property phrase (what is looked up per member, with its year and unit, no condition) and the
  aggregation (what the answer computes over "the members": the property, the conditions on its values and how many
  members it takes, not what selects the members, which the member beliefs carry), from the question, TRQA's property label
  and its computation code. The member beliefs "<member>: <phrase>" are built from the phrase; the aggregate
  belief is the aggregation.

The writer runs before the split, so the rows are written with their texts; a build with texts to write needs
`OPENROUTER_API_KEY`. `texts_ok` is true when every text names none of the gold answers (a name inside a longer
entity name, such as Groningen in "University of Groningen", does not count; multi-member queries: neither the phrase
nor the aggregation names a member, a value or the final answer that the question does not name; multi-hop queries:
no number or capitalized word the question does not state), or when the query has no beliefs (FRAMES, no LLM
call); false when the texts failed for good (a leak, an unstated word or an invalid order after one retry with them
given, no text for a belief); null when the writer call failed (an error, a reply that is not JSON), and the next
build asks again. The number of beliefs is the source's (no cap). Final texts are appended to
`belief_texts.jsonl` as they are written, so a stopped build keeps them; a build reuses them for unchanged
beliefs (the texts do not depend on the documents) and writes only the new queries' texts. Another
`gold_beliefs.model` or other prompts than those of `belief_texts.jsonl` are refused (remove the file to write
every text again). The prompts are hashed per query shape, so editing the set prompts leaves the multi-hop texts
valid. The queries of a split (its generate run) are the full ones with `texts_ok` and at least one belief, and in
`val` also the OOD ones. The `split` section of `stats.json` counts the queries per split, source and level, and the
full ones in no split (`no_split`).


## Stage generate: Trajectories

Stage generate runs dra_inference `--stage generate` on one split of a training dataset (`--subset sft | rl | val`) for
each agent of `shared.policy_models` (or `--policy-model`), as a child process with the inference config
(`shared.inference_config`): the agent's trajectories, and the evaluation of its answers
(`eval/generation/accuracy.jsonl`, the correctness y of training). Settings: `generate.workers`, `quiet`.

The stage resumes, and `--limit N` caps the queries it runs. The runs are dra_inference's
(`{output}/{dataset}_{split}_{retriever}/{agent}_{backend}_{model}/`). The belief does not change what the agent does,
so the documents and agent turns of every step are fixed, and the student's chains are run over the saved
trajectories in the later stages.

**Episodes** (`runs.py`). An episode is a finished query of a run with a judged answer (a query whose judge failed is
left out), a final answer and gold beliefs (`texts_ok`, at least one belief): its search iterations and the
answer turn's reasoning and answer as the belief stage reads them (`orchestration.belief_stage.read_saved_query`), y, its gold beliefs, and the run's corpus,
retriever and passage cut. Documents are read from the corpus by id, as in the belief stage.

**Statistics** (`trajectories/stats.py`), printed and written to `stats/{dataset}_{split}_{agent}.json`: episodes
and the queries left out, the agent's accuracy, the search iterations with a document per episode (mean, median,
90th percentile, max; they set `filter.rl_max_chain_steps`).

## Stage teacher: Teacher beliefs

Stage teacher runs dra_inference `--stage belief --belief-model {teacher}` (`shared.teacher` or `--teacher`) on the
`sft` or `val` split (`--subset`; `rl` needs no teacher) for each agent, over the generated trajectories: the
teacher's beliefs in the run's `belief/{teacher}/` (the SFT samples of stage filter), then their evaluation in the run's `eval/metrics.md` (the teacher's
s_T against y, the baseline of the trained model). Only finished trajectories get beliefs; the stage prints how many
queries of the split have none yet. The stage resumes, and `--limit N` caps the queries it runs. Settings:
`belief_stage.belief_workers`, `quiet` (shared with stage evaluate). The stage needs no GPU: the documents are read
from the corpus by id.

## Stage filter: What is trained on

Stage filter decides what the training stages train on, on CPU, and writes it under `filter/` for every student:
`sft_samples.jsonl` (SFT samples), `rl_episodes.jsonl` (one row per RL episode: set, dataset, split, agent, qid, y)
and `stats.json` (the teacher, the pool: `shared.training_datasets`, `policy_models` and `retriever`, the `filter`
settings and the counts of each set, kept and left out by reason). Each split's episodes are loaded once for both
files. Stage sft and stage rl read these files, stop when the pool of `stats.json` is not the config's, and record
`stats.json` in their run's `config.json`; stage rl also leaves out (and reports) an episode no longer in its run or
judged again since (another y). Each call rewrites the three files; run it again after more trajectories or teacher
beliefs.

**SFT samples** (`filtering/samples.py`), from the replies of the teacher (`shared.teacher` or `--teacher`; its stage
teacher must have run). The teacher's beliefs of each sft (train set) and val (validation set) episode are
replayed through the belief model on `filter.workers` threads: every call gets the teacher's saved reply, so its
prompt is rebuilt by the belief model itself. The replay is checked against the saved file (the same calls, the same belief after every step, the
same s_T); an episode that differs gives no sample. A call is a sample when it passes the format gate:

| Task | Sample | Gate |
|---|---|---|
| update | each update call (first attempt; the final update too) and the teacher's reply | the reply parsed, no issue of `GATE_ISSUES` |
| score | the final score call and the teacher's reply | the reply parsed (`parse_final_score`) |
| init | on `filter.sft_init_datasets` (infoseek_hard), each query of the split: the init prompt and the gold beliefs, in the layout of the prompt's example | at most the prompt's cap of beliefs |

Then the sample rule (`filtering/rules.py`, `keep_sample`): every gated sample is kept for now (TODO: the rule is
open; Agent-BRACE does not filter). TRQA's gold beliefs are not init samples: its members are not in the question.

**RL episodes** (`filtering/run.py`): the episodes of the `rl` split (train set) and the `val` split (validation set).
A train episode with more search iterations with a document than `filter.rl_max_chain_steps` is left out
(`filtering/rules.py`, `episode_drop_reason`; not cut: the final score always reads a complete belief).

## Stage sft: SFT

**Training** (`sft/train.py`): LoRA (r 64, alpha 128, dropout 0.05; attention, Gated DeltaNet and MLP projections),
lr 2e-5, at most 5 epochs, effective batch 32 per GPU (1 x 32 accumulation), as Agent-BRACE. The loss is the
token-mean cross entropy on the reply and the end-of-turn token; the prompt is the chat template the inference server
builds (native reasoning off). Samples longer than `shared.max_seq_len` are dropped (in the student's tokens, so
here and not in stage filter). A checkpoint every epoch; the one with the lowest validation loss is merged into
`{student}/sft/final`.

## Stage rl: RL

**Chains.** A chain is the inference belief model over one filtered rl episode with the student's sampled replies
(`rl/rollout.py`): from the gold beliefs (`rl.beliefs: gold`, run `rl`) or from the student's own init call
(`rl.beliefs: own`, run `rl_own_init`), one update call per search iteration with a passage, then the final score.
Each iteration draws `episodes_per_batch` episodes (a new shuffled order every epoch) and runs `group_size` chains per
episode (the GRPO group), all on the same fixed trajectory, at temperature 0.7. All chains run at once, each in its own
thread; their pending calls are generated together, one vLLM batch per round. Each chain's final-score call gets
`scorer_samples - 1` more replies on the same prompt. The episodes are stage filter's (`filter/rl_episodes.jsonl`).

**Reward** (`reward.py`). Per final-score reply $j$, with $p_j$ the probability of its WEP and $y$ the correctness of
the agent's answer:

$$
r_j = g_j \left(1 - (p_j - y)^2\right), \qquad R = g_{\mathrm{chain}} \cdot \frac{1}{K} \sum_{j=1}^{K} r_j
$$

$g_j$ is 1 when the reply parses and was not cut off. $g_{\mathrm{chain}}$ is Agent-BRACE's format gate over the
chain's belief calls: 1 when every call parsed at its first attempt, was not cut off and has no issue of
`GATE_ISSUES`, and the chain has beliefs. $K$ = `scorer_samples` (the chain's own reply and the extra ones).

**Advantage.** Within the group (the chains of one episode): every init and update call of chain $c$ gets
$(R_c - \mathrm{mean}\,R) / \mathrm{std}\,R$; every final-score reply gets $(r_j - \mathrm{mean}\,r) / \mathrm{std}\,r$
over the group's replies. Since the trajectory is the same for the whole group, differences in reward come from the
belief model only. A group whose rewards are all equal gives no gradient and is replaced by a new episode, up to
`refill_rounds` more rollouts.

**Update** (`rl/trainer.py`, Agent-BRACE's BeliefStateLMTrainer): PPO clip 0.2, dual-clip 3.0, no critic, no KL, lr
1e-5, one epoch over the batch in mini-batches of `ppo_mini_batch_chains` chains (all the calls of a chain together),
token-mean loss, full parameters in bf16 with stochastic-rounding AdamW. vLLM and the trainer share one GPU: the
engine sleeps during the update and reloads the weights from `sync_dir` after it; between updates the trainer's
weights and optimizer state wait in pinned CPU memory (allocated once, reused).

**Validation**, before training and every `val_every` iterations: one greedy chain per val episode; s_T against y
(AUROC, Brier), the mean reward and the gate rate (`validation.jsonl`). `final/` holds the checkpoint with the best
AUROC (Brier on a tie). Outputs: `metrics.jsonl` (one line per iteration: reward, gate rate, groups with spread, PPO
statistics, the log-prob gap to vLLM, entropy), `validation.jsonl`, `step_{n}/`, `final/`, `config.json`.

## Stage evaluate: Evaluation

Stage evaluate runs dra_inference `--stage belief --belief-model M` on the generated run of each agent: the belief model over
the saved trajectories, then its evaluation, added to the run's `eval/metrics.md` next to the other belief models
(one `belief_model` row each). `M` is a trained model (`{student}_sft`, `{student}_rl`, `{student}_rl_own_init`,
served from its checkpoint on local vLLM, one GPU) or a prompted baseline (`qwen3.5_4b`, `qwen3.5_9b`,
`qwen3.6_27b`). The datasets: the val splits and the test sets
(BrowseComp-Plus, TRQA, the report sets; multi_clue and multi_aspect queries are not in the training data).
Headline: s_T AUROC and Brier against y.

Not coded yet: the check that the final score uses the belief (s_T with and without the belief).
