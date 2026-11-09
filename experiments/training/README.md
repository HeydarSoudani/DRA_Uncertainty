# Belief Model Training

Training plan of the belief model, with the deep research agent frozen.

## Goal: uncertainty of the deep research agent

The goal is to report an uncertainty score for the deep research agent (DRA), both during its search and for its
final answer. The uncertainty is kept in a belief state, maintained by a separate belief model next to the agent:
a fixed list of criteria that a complete answer must satisfy, and per criterion one statement: its status (how
likely it is covered, seven words of Agent-BRACE's verbal scale), its selected candidate, the sources this rests on
(documents [dN], the retriever side, and the agent's turns [tN], the reasoner side) and the other candidates with why
they lose or tie, all in one text with a fixed opening. The status rests on what the passages show; the agent's turns
propose candidates and mark doubts, but alone never make one more than "possible" (code lowers such a status,
`turn_only`). Only the statement text is shown (to the next update and to the agent); the observations and every
source assigned to a criterion are saved. Each criterion is its own sequence of statements over the steps, and is scored on its own.

The belief is turned into one number, the certainty score $s_t \in [0, 1]$ (`belief_model_component/certainty.py`),
a fixed function of the statuses only:

1. each status becomes a probability $q_k = (\ell + 2) / 6$ from its signed level $\ell$ (doubtful 1/12 instead of 0,
   unknown 1/3, certain 1);
2. $s_t$ is the geometric mean of the $q_k$, the rest criterion of a set query left out.

$$
s_t = \Big(\prod_{k \in \mathcal{C}'} q_k\Big)^{1/|\mathcal{C}'|}
$$

The geometric mean lets one weak or contradicted criterion pull the score down, but unlike a product it does not
push a query with many criteria (a set query's members) to 0. The final score $s_T$, after the answer turn, is
compared with the correctness $y \in \{0, 1\}$ of the agent's final answer.

The belief model is trained to learn two skills, rewarded per criterion over whole trajectories:

- **source assignment**: which documents and which agent turns bear on each criterion, and what they state;
- **WEP labelling**: the status of each criterion from its sources, calibrated at every step, so that $s_T$
  predicts $y$.

Only the belief model is trained; the agent stays frozen.

## Model

One call per step, the same prompt for the prompted teacher and the trained student
(`belief_model_component/prompts/belief_update_system.txt`):

| Call | Input | Output |
|---|---|---|
| update (every search step $t$) | $q$, the criteria with $b_{t-1}$ (one statement text each), the agent's turn $t$ (its reasoning), the new passages of step $t$ | `think_observations` (spans of the turn per criterion), `passage_observations` (spans of each passage per criterion), `statements` (`{id, statement}`, the new statement text of every criterion whose belief changes) |
| final (answer turn) | the same, the answer turn in place of the turn, no passage | the same, no document observation |

A think observation needs a span that occurs in the turn, and a document observation one that occurs in its passage
(checked in code); other observations do not enter the belief. Observations have no stance and no candidate; the
statement judges them. Code reads the statement's opening (status, selected candidates) and flags a cited source
without an observation on the criterion. The criteria are the pool's
gold criteria; the criteria init call is not trained.

Each status has a signed level $\ell$:

| Status | doubtful | unlikely | unknown | possible | probable | almost certain | certain |
|---|---|---|---|---|---|---|---|
| $\ell$ | −2 | −1 | 0 | 1 | 2 | 3 | 4 |

The criteria follow the query shape of the dataset: `multi_hop` for deep_qa (one criterion per entity on the chain
to the answer, the answer last), `set` for trqa_validation (one criterion per member, "<member>: <property>",
then "any other member: <property>", then the aggregation, how the answer follows from the members' values).
Training uses the queries with the full gold annotation (InfoSeek and TRQA): per criterion its gold answers and
gold documents.

## Phases

| Phase | Script | Does | Output |
|---|---|---|---|
| 0 | `build_training_pool.py --pool P` | builds a training pool, deep_qa (questions, gold pages, qrels, `--step N`) or trqa_validation (merged TRQA validation files), with its gold annotation and criterion texts, and splits it into `sft` / `rl` / `val` | `data/training_pool/{pool}/` |
| 1 | `belief_model_training.py --phase 1 --split S` | runs dra_inference with the frozen agent on the queries of one split (`--subset S`: the pool's rows with that `split`): `sft` and `val` in monitor mode with the teacher belief model on the criteria of `shared.criteria_source` (default gold; a query without gold criteria: on its own silver ones), `rl` in off mode. deep_qa runs go to `belief-monitor/`, trqa_validation runs to `belief-monitor-{source}/`; phases 2 and 3 need gold | `run_outputs/` (dra_inference layout) |
| 2 | `belief_model_training.py --phase 2` | teacher scores, then SFT of the student on the teacher's belief update replies of the `sft` run (LoRA, then merged), validated on the `val` run | `phase2_sft/merged/` |
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

Each pool is a dataset of `dra_inference.py` under `data/training_pool/{pool}`, with one row per query:

| File | Contents |
|---|---|
| `queries/queries_all.jsonl` | one row per query: the question and answer, its source, its split, its gold annotation (see [Gold annotation](#gold-annotation-deep_qa-steps-1-2-trqa_validation-build)) |
| `qrels/qrels_all.txt` | the gold documents of each query (TREC qrels) |
| `corpus/` | deep_qa only: gold pages and hard negatives (trqa_validation searches TRQA's corpus) |
| `criterion_texts.jsonl`, `stats.json` | build files: the writer's saved criterion texts, the counts and settings of each step |

deep_qa also keeps its step files next to them (`questions.jsonl`, `pages/`, `negatives/`, `dropped.jsonl`). The
splits `sft`, `rl` and `val` are views of split `all`: the queries loader reads `--subset sft` as the rows with
`split` `sft` and their qrels (`layout.is_pool_view`), so dra_inference runs a split as on any dataset and its run
folders are named by the split. The build (the end of deep_qa step 2, the trqa_validation call) sets each row's
`split`. SFT
and RL need the full gold annotation (`level` `full`: every criterion with gold answers and gold documents, for
$r_{\mathrm{ret}}$ and $r_{\mathrm{reas}}$), so the full queries are split with the shares of the `split` section of
`experiments/configs/build_training_pool.yaml`, and every other query goes to `val` as its out-of-domain (OOD) part:

| Split | Queries | Phase 1 mode | Used by |
|---|---|---|---|
| `sft` | 30% of the full ones | monitor (teacher, gold criteria) | phase 2: SFT samples |
| `rl` | 60% of the full ones | off | phase 3: RL prompts |
| `val` | in domain: 10% of the full ones | monitor (teacher, gold criteria) | phase 2: validation loss; phase 3: validation |
| `val` | OOD: the others (deep_qa: WebShaper, FRAMES) | monitor (teacher; WebShaper on its gold criteria, FRAMES on silver ones) | evaluation only |

A full query without criterion texts (`texts_ok` not true: not written yet, or failed) has `split` null and is in no
split until a later build writes its texts. A query keeps the split it has in the queries file, so no query moves
when the pool grows, and `--limit` on one split never reaches the queries of another. The new full queries of a build
are taken in the order of a hash of their id (not the source order), and each goes to the split furthest below its
share, so the shares hold at every pool size (20 full queries: 6 sft, 12 rl, 2 val; 30: 9, 18, 3).
In the OOD part WebShaper keeps its gold criteria (they have texts, but no gold intermediate answers: `level`
partial), and FRAMES, which has no structure, gets silver criteria from the belief model's init call, as on any new
dataset; its queries have qrels and a judged final answer, so the certainty score against correctness can be
measured, but not the observations and beliefs (no gold answers and documents per criterion). Since deep_qa has one
query shape (`multi_hop`), FRAMES gets the multi-hop criteria prompt (at most 8 criteria).

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
python experiments/training/build_training_pool.py --step 2 --source webshaper,frames     # the OOD part of val (all of them; or with --limit)
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

Step 2 sets the split of every row it writes, so the splits always match the pool. The shares and per-split counts are recorded
in the `split` section of `stats.json`. Other shares (they would mix two splits in one pool) and other page
settings than the earlier step 2 runs' (`wiki_lang`, `min_words`, `min_page_share`, `wayback`, `wayback_timestamp`)
are refused: to change the shares, remove the `split` section of `stats.json` and the next build re-splits every query; to
change the page settings, rebuild the pool from step 1. Other step 3 settings than the earlier runs' are refused too:
to change them, remove `negatives/index.jsonl` and the `step3` section of `stats.json`. One build of a pool runs at a time (`.build.lock` in
its directory).

### trqa_validation

The trqa_validation pool merges the TRQA wiki1 and wiki2 validation files (queries, intermediate info, qrels) into
`queries/queries_all.jsonl` and `qrels/qrels_all.txt` (one call): a query and its intermediate info become one row,
the members with their values and documents in `members`, with `property` (the label) and `aggregation`. It searches
TRQA's `wiki_partial` corpus. The gold loaders (`load_query_intermediate_info`) read the intermediate info of a pool
from its rows, in TRQA's record format.

Left out: the one query whose text is also a TRQA test query, and the 14 wiki2 ids that each name two different questions (28 queries). 1,145 queries remain (91 wiki1, 1,054 wiki2), all with qrels and all full, so `val` has no OOD part: 343 sft, 687 rl and 115 val.

```bash
python experiments/training/build_training_pool.py --pool trqa_validation             # every query
python experiments/training/build_training_pool.py --pool trqa_validation --limit 100 # the next 100 not yet in the pool
```

`--limit N` is incremental, like deep_qa step 2: it adds the next N kept queries not yet in the pool (file order,
wiki1 then wiki2) to those already there; `--limit 0` adds none and rewrites the files. The criterion texts of
earlier calls are reused, so only the new queries call the writer, and every earlier query keeps its split. The
`build` section of `stats.json` records the queries in the pool and the ones remaining.

Its rows have the keys of deep_qa's (source `trqa_wiki1` / `trqa_wiki2`).

### Gold annotation (deep_qa steps 1-2, trqa_validation build)

Each pool has a gold annotation per query: its criteria, each with its gold answers and gold documents, as far as
the source states them (`training_pool/annotations.py`). Every row has the same keys whatever the source; what the
source does not state is an empty value (`[]` or `null`), never a missing key.

| Source | Criteria | `answers` | `doc_ids` |
|---|---|---|---|
| InfoSeek | one per entity of the research tree that the question does not name, children before parents, the answer last (`multi_hop`) | the entity, and its title without a parenthetical | its Wikipedia page |
| WebShaper | one per variable of the formalization, the farthest from the answer first, then the answer (`multi_hop`) | the answer only (final answer) | none (the URLs are the query's) |
| FRAMES | none | | |
| TRQA wiki2 | one per member, "<member>: <property>", the member named by the Wikipedia title of its first document ("San Fernando, La Union", not TRQA's label "San Fernando") (`closed`); then "any other member: <property>" (`rest`); then the aggregation, how the answer follows from the values (`aggregate`) (`set`) | the member's value (`entity_values`); rest: none; aggregate: the final answer | the member's documents (`entity_values[].passage_ids`); rest and aggregate: none |
| TRQA wiki1 | the same | the same | the same |

Every TRQA member has its own criterion, so the rest criterion has no gold in the pool: training can hide some
members under it (their values and documents then become its gold), as at inference, where the belief model lists
only the members it knows. The aggregate criterion has no documents since no document states the computed value.

`level` follows from the row and selects a group of queries:

| `level` | The row has | Training can use |
|---|---|---|
| `full` | criteria, each with its gold: gold answers and gold documents (a `rest` criterion none, an `aggregate` one answers only) | per criterion: its answers and its documents; the final answer and the qrels |
| `partial` | criteria, some without answers or without documents | the criteria that have them; the final answer and the qrels |
| `query` | no criteria | the final answer and the qrels only |

Queries per level, with the pools as built on 2026-11-01 (deep_qa: the 998 questions step 2 has processed so far,
910 kept; `level` does not depend on the criterion texts):

| Pool | Source | Queries | full | partial | query | Criteria / query (mean) |
|---|---|---|---|---|---|---|
| deep_qa | InfoSeek | 400 | 400 | 0 | 0 | 2.5 |
| deep_qa | WebShaper | 106 | 0 | 106 | 0 | 5.3 |
| deep_qa | FRAMES | 404 | 0 | 0 | 404 | 0 |
| trqa_validation | TRQA wiki1 | 91 | 91 | 0 | 0 | 15.4 |
| trqa_validation | TRQA wiki2 | 1,054 | 1,054 | 0 | 0 | 11.7 |

TRQA's criteria per query are the members plus the rest and the aggregate (the members per query of the TRQA files,
before the build leaves out 29 queries).

An InfoSeek query is `partial` when an entity of its criteria has no Wikipedia page in the tree, or that page was not
fetched (Filter A keeps an InfoSeek query only when all its pages were fetched, so none is so far). A WebShaper
criterion has no documents, and only the answer criterion has answers. The numbers come from the annotation tables of
the `annotations` section of `stats.json`, the same in both pools (written by deep_qa steps 2 and 4 and by the trqa_validation build, with the writer of the texts). Step 1 builds the criteria from the source (`annotation` of each record in
`questions.jsonl`, with the page keys of their documents); step 2 resolves the keys to the fetched documents (a page
that was not fetched gives none, so a query can fall from full to partial) and writes the rows to
`queries/queries_all.jsonl`. The trqa_validation build writes the same rows from the intermediate info. Query row:
`id`, `source`, `split`, `text` (the question), `answer`, `query_shape`, `level`, `doc_ids` (the qrels), `criteria`,
`members` (set queries: `entity`, `value`, `doc_ids`), `entity_names` (InfoSeek: every entity of the tree; TRQA: the
members), `property` and `aggregation` (set queries); criterion: `id`, `text`, `answers`, `doc_ids`, `kind`
(`closed`, `rest`, `aggregate`), `relations` (the source structure the writer reads); then `texts_ok`, `errors` of the texts (the writer and its prompt hash
are in `stats.json` and `criterion_texts.jsonl`, not per row). Steps 2 and 4 and the trqa_validation build count the rows per source and
level, the criteria per kind and with their answers, documents and gold, and check the row format.

### Criterion texts (end of deep_qa step 2 and of the trqa_validation build)

An LLM (`gold_criteria.model`, the teacher) writes the criterion texts in the question's words, one call per query;
answers, documents and level come from the source (`training_pool/gold_criteria.py`). The prompts are the files of
`training_pool/prompts/`, one system and one user prompt per query shape:

- multi-hop (InfoSeek, WebShaper): one text per criterion, from its relations, with only the details the question
  states (a relation may hold more, such as a year).
- set (TRQA): the property phrase (what is looked up per member, with its year and unit, no condition) and the
  aggregation (what the answer computes over "the members": the property, the conditions on its values and how many
  members it takes, not what selects the members, which the member criteria carry), from the question, TRQA's property label
  and its computation code. The member criteria "<member>: <phrase>" and the rest criterion "any other member:
  <phrase>" are built from the phrase; the aggregate criterion is the aggregation.

The writer runs before the split, so the rows are written
with their texts; step 2 and the trqa_validation build need `OPENROUTER_API_KEY`. `texts_ok` is true when every text
names none of the gold answers (a name inside a longer entity name, such as Groningen in "University of Groningen",
does not count; set queries: neither the phrase nor the aggregation names a member, a value or the final answer that
the question does not name; multi-hop queries: no number or capitalized word the question does not state), or when
the query has no criteria (FRAMES, no LLM call); false when the texts failed for good (a leak or an unstated word
after one retry with them given, no text for a criterion); null when the writer call failed (an error, a reply
that is not JSON), and the next run asks again. The number of criteria is the source's (no cap). Final texts are
appended to `criterion_texts.jsonl` as they are written, so a stopped run keeps them; a run reuses them for unchanged
criteria and writes only the new queries' texts. Another `gold_criteria.model` or other prompts than those of
`criterion_texts.jsonl` are refused (remove the file to write every text again). The prompts are hashed per query
shape, so editing the set prompts leaves the multi-hop texts valid. `--step 2 --limit 0` fetches nothing
and rewrites the dataset files from the questions already processed. The queries of a split (its phase 1 run) are
the full ones with `texts_ok` and at least one criterion, and in `val` also the OOD ones. The `split` section of
`stats.json` counts the queries per split, source and level, and the full ones in no split (`no_split`). The annotation tables of steps 2 and 4 count the texts per source (ok, failed,
retry).

## Phase 1: Trajectory collection

Phase 1 runs `dra_inference.py` with the frozen agent on the queries of one split (`--subset {split}`: the
pool's rows with that `split`), every such query or the next `--limit` not run yet (inference resume). The mode
follows the split:

| Split | Mode | Belief model |
|---|---|---|
| `sft`, `val` | monitor: the agent never sees the belief | the query's gold criteria (`--criteria-source`, `shared.criteria_source`: gold, or silver on trqa_validation; a query without gold criteria: the teacher's silver ones, through the criteria bank, `gold_missing` in its `criteria_info`) and the teacher (`openrouter/qwen/qwen3.6-27b`) for the belief updates |
| `rl` | off | none |

The belief does not change what the agent does, so the documents and agent turns of every step are fixed, and
phase 3 can run the student over the steps of an off-mode run. The runs go to dra_inference's output root in its
usual layout (`run_outputs/{pool}_{split}_{retriever}/{agent}/belief-monitor-gold/` for `sft` and `val`,
`belief-off/` for `rl`), and phase 1 writes nothing else.

**Episodes.** Phases 2 and 3 read a split's run at their start (`belief_model_training/runs.py`), one episode per
finished in-domain query (`level` full, `texts_ok`, at least one criterion; the OOD queries of `val` are counted as
`out_of_domain` and left out):

| Level | Field | Source | Used by |
|---|---|---|---|
| step | `doc_ids`: documents shown in the search iteration (all its searches; none when they returned nothing); `iteration`, the belief model's step number (from 1), and `agent_iteration`, the agent's own (`iter`) | `trajectory/{qid}.jsonl` (`seen_docs`) | SFT replay, RL chains |
| step | `agent_turn`: the agent's reasoning and queries of the iteration (as the teacher read it in monitor runs) | `trajectory/{qid}.jsonl`, `belief/{qid}.jsonl` | the update prompt |
| step | `new_gold_ids`, `recall_so_far`: relevant documents first seen in the step, share seen so far | qrels | statistics, check of the saved run |
| step | `stated`: the criteria whose gold answer the step's agent turn states | gold criteria, `nugget_gold.py` | assign (expected think observations) |
| step | `teacher_output`, `teacher_records`: the teacher's reply and its saved observations, changed criteria, issues and statements (monitor runs) | `belief/{qid}.jsonl` | SFT, replay check |
| episode | `answer_turn`: reasoning after the last search and the final answer; the teacher's final call on it (monitor runs, the `final` line) | `trajectory/{qid}.jsonl`, `generation/{qid}.md`, `belief/{qid}.jsonl` | final call |
| episode | `correct` ($y$; null when the judge failed) | `accuracy.jsonl` | $r_{\mathrm{final}}$ |
| episode | `gold_criteria`: criteria with `answers` and `doc_ids` | `queries/queries_all.jsonl` of the pool | the criteria of the chains, all rewards |

**Gold of a candidate or spans.** A selected candidate is correct ($g = 1$), and an observation's spans state the
gold, when one of the criterion's gold answers occurs in it as whole words (case, accents and punctuation ignored; not
inside a longer entity name of the query's gold structure) or, for a number, when a number of the text equals it. A
text without a match is asked of the correctness judge (`rl.answer_judge_model`, YES/NO: is the candidate the gold
answer? / does the text state it?). A turn states a
criterion's gold answer by the same match; a (turn, criterion) pair without a match, whose gold documents were
shown before the turn, is asked of the same judge. The verdicts are cached in
`{output_root}/{training_data}/{policy_model}/answer_gold.jsonl`; `rl.answer_judge_model: null` uses the match only.
The rest criterion of a set query has no gold yet (the gold lists every member as its own criterion) and is not
scored.

Only finished queries are episodes, as dra_inference counts them: the trajectory ends with its meta line and, in a
monitor run, the belief file exists. The qrels are read with the run's `dataset.min_relevance_score`. The runs a phase reads must share
their agent and searcher settings (`run_config.json`, the temperature included): phase 2 checks the sft
and val runs, phase 3 the rl and val runs, and either stops with the keys that differ. All runs take these
settings from the same inference config (`shared.inference_config`), so keep it unchanged across the phase 1
runs. For monitor runs, the loader also checks that the run's updater is the teacher on the gold criteria (and
that the run's criteria are the pool's), that the belief file (schema 32) has one step per search iteration of the
trajectory, joined on `agent_iteration`, with the same step number and documents, and that the saved
`recall_so_far` matches the qrels. Episodes without a verdict in `accuracy.jsonl` (a run that stopped before its
evaluation) are counted and reported; `--eval-only` judges them.

## Scores and rewards

The same scores rate the teacher (phase 2), reward the student's chains (phase 3) and validate them
(`belief_model_training/scoring.py`, `chain_score`). Each criterion $k$ with gold answers (not the rest criterion)
is its own sequence of statements, one per step of the episode: a step without a call, or whose call failed, keeps
the statement of the step before, and the answer turn is the last step. Each criterion is scored on its own, then
the scores are averaged over the criteria, so neither a long episode nor a criterion with many updates weighs more.

**Gate** $f \in \{0, 1\}$ per call: 1 if the reply is not cut off and parses, names known criteria and new passages
only, and writes each statement with an opening code can read (`criteria.GATE_ISSUES`). The chain's `format` is the share of its
calls that pass.

**Source assignment** $\mathrm{assign}_k$: an F1, $2\,TP / (2\,TP + FP + FN)$, of criterion $k$'s observations
over the chain's calls, documents and turns together:

| Case | TP | FN | FP |
|---|---|---|---|
| passage $d$ shown, $d$ a gold document of $k$ (not the aggregate) whose text as the call showed it (cut at 3000 characters for a short-answer query) has a gold answer of $k$ | an observation whose spans state it | none | |
| the turn states the gold answer of $k$ | a think observation whose spans state it | none | |
| an observation that did not enter the belief (no verified span of its passage or turn, a document observation on a computed criterion) | | | yes |
| any other observation | neutral | | |

**WEP calibration** $\mathrm{cal}_k$: the mean over the steps of $1 - (p_{k,t} - y_{k,t})^2$, $p_{k,t}$ the
probability of the status and $y_{k,t}$ whether the statement covers the criterion correctly: a one-answer
criterion, whether its selected candidate is a gold answer (0 without one); a several-answers criterion, the share
of its gold answers among its selected candidates.

**Final** $r_{\mathrm{final}} = 1 - (s_T - y)^2$, the certainty score of the final statements against the
correctness of the agent's final answer; 0 when the final call fails the gate.

A part with nothing to score (no gold criterion, no expected or written observation, $y$ unknown) is None, not 1.

Two more per-criterion measures are reported, not rewarded: the recall of $k$'s gold documents (those that should
give an observation) among its document observations (`doc_recall_obs`) and among the documents its final statement
rests on, its support part (`doc_recall_statement`); and the share of steps whose selected candidate is correct
(`value_accuracy`).

### Teacher scores

SFT copies the teacher, errors included, so phase 2 scores the teacher on the sft and val episodes before training
(`phase2_sft/teacher_scores.json`), on its replayed calls:

| Part | Measures |
|---|---|
| gate | share of calls that pass; the rate of each issue code per call |
| per criterion | mean over the chains of the mean over their criteria: `crit_assign_f1`, `crit_doc_recall_obs`, `crit_doc_recall_statement`, `crit_value_accuracy`, `crit_cal` |
| observation | precision, recall and F1 of document and think observations, pooled |
| calibration | Brier score, ECE and AUROC (on the items with a binary $y$) of the statuses of every (criterion, step) against $y_{k,t}$; gold rate and count per status (reliability) |
| final | $s_T$ against $y$: AUROC, Kendall τ-b, Brier score, ECE (10 bins); the same for the mean, product and min of the criterion scores (information only) |
| rewards | mean `format`, `assign`, `cal`, $r_{\mathrm{final}}$ |

## Phase 2: SFT

A sample is one belief update call of the `sft` monitor run: its prompt $x$ (masked from the loss) and the
teacher's reply $z$ (trained). A trajectory with K search steps gives at most K + 1 samples: one per step the
teacher's call judged and the final call on the answer turn. Each teacher reply is replayed through the inference
belief update to rebuild its exact prompt and check it against the saved records; a call is kept when it passes
the gate and, with `sft.require_verified_spans`, when it is clean (`criteria.UNCLEAN_REPLY`: all its observations
entered the trace, no status corrected by code, no `bad_ref`, `extra_selected` or `no_selected`). The main process
writes the samples to `phase2_sft/samples.jsonl` (`train`: sft split, `val`: val split). The student learns the
output format and the teacher's transitions, errors included; RL then corrects the transitions.

**Loss.** Cross-entropy on the reply tokens and the end-of-turn token, averaged over all reply tokens of the batch
$\mathcal{B}$ (token-mean):

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
| validation | the `val` samples: loss after each epoch | picks the epoch |

## Phase 3: RL (GRPO)

**Chains.** A chain is the student's belief update run over a whole `rl` episode from its gold criteria: one call
per step with a document not judged yet or an agent turn, then the final call on the answer turn, each prompt built
from the chain's own $b_{t-1}$, as at inference. The agent is frozen, so the documents and agent turns of every
step are fixed. Training on the student's own states, not the teacher's, keeps it ready for its own earlier errors
(exposure bias; DAgger, Ross et al. 2011). A call's reply budget is the inference updater's (12,288 tokens, 16,384
for a set query).

**Groups.** The student runs $G = 8$ chains per episode at temperature 1. A group whose chains all score alike
gives no gradient and is replaced by new episodes, with up to two more rollouts per iteration (DAPO's dynamic
sampling).

**Reward.** One reward per trajectory (`belief_model_training/rewards.py`), over every step and the final belief:

$$
R = w_{\mathrm{format}}\,\mathrm{format} + w_{\mathrm{assign}}\,\overline{\mathrm{assign}_k}
  + w_{\mathrm{cal}}\,\overline{\mathrm{cal}_k} + w_{\mathrm{final}}\, r_{\mathrm{final}},
$$

where $\overline{x_k}$ is the mean over the chain's criteria that have the part. All weights are 1
(`rl.reward_weight_{format,assign,cal,final}`). $R$ is the logged reward and the validation measure.

**Advantage.** Each of the four terms is normalized within the episode's group (GDPO, arXiv 2601.05242), a chain
without the term counting as the group mean, then weighted and summed, so a term with a small spread is not
drowned and a tiny spread is not blown up by the floor $\sigma_{\min} = 0.05$:

$$
\hat{A}_i = \frac{\sum_{m} w_m\, \dfrac{r_{m,i} - \operatorname{mean}_m}{\max(\operatorname{std}_m, \sigma_{\min})}}{\operatorname{std}_{\text{batch}}\left(\sum_{m} w_m\, \cdots\right)},
$$

with mean and std over the $G$ chains of the group, a term with zero spread counting 0. Every call of chain $i$
takes $\hat{A}_i$: a chain is judged against the other chains over the same episode, so the belief terms also reach
the observation tokens of the reply.

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
mean reward, each term and the teacher-score measures, $s_T$ against $y$ included (compare with
`phase2_sft/teacher_scores.json`). `final/` is the checkpoint with the best validation reward.

**Open.** One pool per run (`shared.training_data`): InfoSeek (deep_qa) and TRQA (trqa_validation) are not yet
mixed in one RL run. The non-gold passages are neutral in the source assignment; a judged sample of them will show
whether a precision term is needed. The rest criterion joins the rewards once members are masked into it.

## Evaluation (dra_inference)

The trained belief model is evaluated with `experiments/inference/dra_inference.py`: the frozen agent with no,
teacher or student belief model in `inform` mode (in monitor mode the scores would not depend on the belief).
Scores are the run's `summary.json`: answer accuracy, Recall@N, GradedRecall@N. The belief files carry the
certainty score per step (`certainty`) and after the final call (`final_certainty`), to compare with the
correctness in `accuracy.jsonl`. Belief quality itself is measured in phase 3 validation. The student is trained on the `multi_hop` (deep_qa) or `set` (trqa_validation) prompts; a
dataset of another query shape (browsecomp_plus: `single_target`) gives it prompts it was not trained on.

```bash
# serve the student on a GPU the inference workers do not use
CUDA_VISIBLE_DEVICES=0 vllm serve $RUN/phase3_rl/final --served-model-name belief-student --port 6020 --language-model-only &
export BELIEF_VLLM_API_BASE=http://127.0.0.1:6020/v1

COMMON="--dataset browsecomp_plus --policy-model glm --query-scorer-model openrouter/qwen/qwen3.6-27b --gpu-ids 1,2,3"
python experiments/inference/dra_inference.py $COMMON --belief-mode off
python experiments/inference/dra_inference.py $COMMON --belief-mode inform \
    --llm-criteria openrouter/qwen/qwen3.6-27b --belief-judge-model openrouter/qwen/qwen3.6-27b
python experiments/inference/dra_inference.py $COMMON --belief-mode inform \
    --llm-criteria openrouter/qwen/qwen3.6-27b --belief-judge-model vllm/belief-student --output <student output dir>
```

- The student is trained on the belief update only, so its runs take the teacher's criteria (`--llm-criteria`,
  read from the criteria bank) and only the updates come from the student.
- `--query-scorer-model` is the phase 1 scorer in every run; its default is the belief model, which would make the
  student score queries too.
- Give the student run its own `--output`: teacher and student runs share the agent, dataset and belief mode.
