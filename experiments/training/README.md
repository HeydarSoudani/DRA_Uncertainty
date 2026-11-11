# Belief Model Training

Training plan of the belief model, with the deep research agent frozen.

## Goal: uncertainty of the deep research agent

The goal is to report an uncertainty score for the deep research agent (DRA), both during its search and for its
final answer. The uncertainty is kept in a belief state, maintained by a separate belief model next to the agent:
a fixed list of criteria that a complete answer must satisfy, and per criterion its candidates (possible answers),
each with a status (how likely it is an answer, Agent-BRACE's verbal scale) and its support: the passage observations
(documents [dN], the retriever side) that support it or speak against it. The agent's turns ([tN], the reasoner side)
only propose names (proposals, no status, not rivals) and make claims, saved but never shown to the update; a
passage that supports a proposal makes it a candidate. The model writes the status from the support and, for a
one-answer criterion, from the other candidates' support (per criterion type in
`prompts/belief_update/criterion_statuses.txt`); code enforces only hard limits (no support: at most "possible",
`no_support`; no new support: the status kept, `no_new_support`). The candidates with their support (quoted spans
and passage titles) and the proposals are shown to the next update; the agent sees the candidates and proposals. A
candidate the update does not name keeps its status; one a passage states fails a part is removed. A step makes a call
only when it has a new passage; otherwise the belief is carried forward. A
criterion's status is the highest of its candidates. Each criterion is its own sequence of beliefs over the steps, and is scored on its own.

The belief is turned into one number, the certainty score $s_t \in [0, 1]$ (`belief_model_component/core/certainty.py`),
a fixed function of the statuses:

1. each status becomes a probability $p = (\ell + 2) / 6$ from its signed level $\ell$ (unlikely 1/6, unknown 1/3,
   possible 1/2, certain 1); $q_k$ is the criterion status's $p$ (a one-answer status already counts the competing
   candidates);
2. $s_t$ is the geometric mean of the $q_k$, the rest criterion of a multi-member query left out.

$$
s_t = \Big(\prod_{k \in \mathcal{C}'} q_k\Big)^{1/|\mathcal{C}'|}
$$

The geometric mean lets one weak or contradicted criterion pull the score down, but unlike a product it does not
push a query with many criteria (a multi-member query's members) to 0. The final score $s_T$ is compared with the
correctness $y \in \{0, 1\}$ of the agent's final answer. On a short-answer query it is the final scorer's
(`belief_model_component/update/final_score.py`): one call on $b_{T-1}$ and the answer turn judges how likely the
answer is correct and writes the status of the final answer on the belief's scale, read as a probability as a
candidate's status is; on a report it is $s_T$ of the final
belief.

The belief model is trained on two tasks (two prompts, one model):

- **belief update**, rewarded per criterion over whole trajectories: **source assignment** (which documents and
  which agent turns bear on each criterion, and what they state) and **WEP labelling** (the status of each
  criterion from its sources, calibrated at every step);
- **final scoring**: the status of the agent's answer (how likely it is correct), from the belief, so that $s_T$ predicts $y$.

Only the belief model is trained; the agent stays frozen.

## Model

One call per step with a new passage, then the final call, the same prompts for
the prompted teacher and the trained student (`belief_model_component/prompts/belief_update/system.txt`,
`prompts/final_score/system.txt`):

| Call | Input | Output |
|---|---|---|
| update (a search step $t$ with a new passage) | $q$, the criteria with $b_{t-1}$ (their candidates, each with status and support, and their proposals), the agent's turns since the last call (its reasoning), the new passages of step $t$ | `reasoning_observations` (claims: `{id, turn, candidate, span, claim}`), `passage_observations` (spans of each passage per criterion), `candidates` (`{id, candidate, support, against, reason, status}`, every candidate whose support or status changes) |
| final scorer (answer turn, short-answer query) | $q$, the criteria with $b_{T-1}$ (as above, plus the names a passage ruled out), the answer turn (last 2000 characters of reasoning, final answer) | `candidates` (`{id, candidate}` per criterion, the answer's candidate), `reason`, `status` (one of the five statuses) |
| final update (answer turn, report) | the update's input, with the answer turn and no passage | the update's output |

A claim needs a span that occurs in its turn, and a document observation one that occurs in its passage (checked in
code); other observations do not enter the belief. Passage observations have no stance and no candidate; the
candidate updates say which passages support or speak against which candidate. Code checks each reason's opening against the candidate's current status and flags a
cited source without an observation on the criterion. The criteria are the dataset's
gold criteria; the criteria init call is not trained.

Each status has a signed level $\ell$:

| Status | unlikely | unknown | possible | probable | almost certain | certain |
|---|---|---|---|---|---|---|
| $\ell$ | −1 | 0 | 1 | 2 | 3 | 4 |

The criteria follow the query shape of the dataset: `multi_hop` for infoseek and webshaper_frames (one criterion per entity on the chain
to the answer, the answer last), `multi_member` for trqa_validation (one criterion per member, "<member>: <property>",
then "any other member: <property>", then the aggregation, how the answer follows from the members' values).
Training uses the queries with the full gold annotation (InfoSeek and TRQA): per criterion its gold answers and
gold documents.

## Phases

| Phase | Script | Does | Output |
|---|---|---|---|
| 0 | `build_training_dataset.py --dataset D` | builds a training dataset, infoseek (InfoSeek-Hard over the original wiki-25-512 corpus), webshaper_frames (fetched gold pages and hard negatives, val only) or trqa_validation (merged TRQA validation files), with its gold annotation and criterion texts, and splits it into `sft` / `rl` / `val` | `data/training_datasets/{dataset}/` |
| 1 | `belief_model_training.py --phase 1 --split S` | runs dra_inference with the frozen agent on the queries of one split (`--subset S`: the dataset's rows with that `split`): `sft` and `val` in monitor mode with the teacher belief model on the criteria of `shared.criteria_source` (default gold; a query without gold criteria: on its own silver ones), `rl` in off mode. infoseek and webshaper_frames runs go to `belief-monitor/`, trqa_validation runs to `belief-monitor-{source}/`; phases 2 and 3 need gold | `run_outputs/` (dra_inference layout) |
| 2 | `belief_model_training.py --phase 2` | teacher scores, then SFT of the student on the teacher's belief update replies of the `sft` run (LoRA, then merged), validated on the `val` run | `phase2_sft/merged/` |
| 3 | `belief_model_training.py --phase 3` | RL of the student (GRPO on whole belief chains) on the `rl` run, validated on the `val` run | `phase3_rl/final/` |

`--training-data` picks the training dataset (`infoseek`, the default, `trqa_validation`, or `webshaper_frames` for val) and `--policy-model` the agent.
Phase 1 writes only the inference runs; phases 2 and 3 read the runs of that agent and write to
`{output_root}/{training_data}/{policy_model}/`. Evaluation of the trained model: `dra_inference.py` (see
[Evaluation](#evaluation-dra_inference)).

```bash
python experiments/training/build_training_dataset.py --dataset trqa_validation
T="--training-data trqa_validation --policy-model glm"
python experiments/training/belief_model_training.py --phase 1 $T --split val
python experiments/training/belief_model_training.py --phase 1 $T --split sft --limit 200
python experiments/training/belief_model_training.py --phase 1 $T --split rl --limit 500
torchrun --nproc_per_node 4 experiments/training/belief_model_training.py --phase 2 $T
CUDA_VISIBLE_DEVICES=0 python experiments/training/belief_model_training.py --phase 3 $T
```

## Phase 0: Training datasets

Each training dataset is a dataset of `dra_inference.py` under `data/training_datasets/{dataset}`, with one row per
query:

| File | Contents |
|---|---|
| `queries/queries_all.jsonl` | one row per query: the question and answer, its source, its split, its gold annotation (see [Gold annotation](#gold-annotation)) |
| `qrels/qrels_all.txt` | the gold documents of each query (TREC qrels, binary) |
| `criterion_texts.jsonl`, `stats.json`, `dropped.jsonl` | build files: the writer's saved criterion texts, the counts and settings of each build stage, the dropped questions with their stage and reasons |

| Dataset | Questions | Corpus | Splits |
|---|---|---|---|
| `infoseek` | InfoSeek-Hard | the original release's `wiki-25-512.jsonl` (Wikipedia 2025 in chunks of 512 tokens), under `DATA_ROOT/infoseek/corpus` with its indices | sft / rl / val |
| `webshaper_frames` | WebShaper, FRAMES | its own `corpus/`: the fetched gold pages and their hard negatives | val only |
| `trqa_validation` | TRQA wiki1 + wiki2 validation | TRQA's `wiki_partial` | sft / rl / val |

One build call per dataset; every build is incremental, like dra_inference: `--limit N` processes the next N
questions not processed yet, `--limit 0` processes none and rewrites the dataset files, no `--limit` processes all.

```bash
python experiments/training/build_training_dataset.py --dataset infoseek --limit 500
python experiments/training/build_training_dataset.py --dataset webshaper_frames --limit 200
python experiments/training/build_training_dataset.py --dataset webshaper_frames --source frames --limit 50
python experiments/training/build_training_dataset.py --dataset trqa_validation
```

The splits `sft`, `rl` and `val` are views of split `all`: the queries loader reads `--subset sft` as the rows with
`split` `sft` and their qrels (`layout.is_split_view`), so dra_inference runs a split as on any dataset and its run
folders are named by the split. Every build sets each row's `split`. SFT and RL need the full gold annotation
(`level` `full`: every criterion with gold answers and gold documents, for $r_{\mathrm{ret}}$ and
$r_{\mathrm{reas}}$), so the full queries of `infoseek` and `trqa_validation` are split with the shares of the
`split` section of `experiments/configs/build_training_dataset.yaml`, and every other query goes to `val` as its
out-of-domain (OOD) part; `webshaper_frames` is validation only:

| Split | Queries | Phase 1 mode | Used by |
|---|---|---|---|
| `sft` | 30% of the full ones | monitor (teacher, gold criteria) | phase 2: SFT samples |
| `rl` | 60% of the full ones | off | phase 3: RL prompts |
| `val` | in domain: 10% of the full ones | monitor (teacher, gold criteria) | phase 2: validation loss; phase 3: validation |
| `val` | OOD: `webshaper_frames` (every query) | monitor (teacher; WebShaper on its gold criteria, FRAMES on silver ones) | evaluation only |

A full query without criterion texts (`texts_ok` not true: not written yet, or failed) has `split` null and is in no
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
| WebShaper | 500 | 8.1 | 22% / 78% | 65 | 3.4 |
| FRAMES | 514 | 3.1 | all wiki | 26 | 5.1 |

### infoseek

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

### webshaper_frames

WebShaper and FRAMES questions with their gold pages fetched from the source, in stages run by every build:

| Stage | Does | Files |
|---|---|---|
| questions | merge the sources, drop duplicates and the FRAMES reasoning types Tabular reasoning and Post processing | `questions.jsonl` |
| pages | fetch the gold pages of the next `--limit` questions (`--source`: of those sources): Wikipedia API (Parsoid HTML), web pages (Trafilatura, pypdf), Wayback Machine fallback; Filter A (a question with only Wikipedia gold pages needs all of them, one with web gold pages `min_page_share` of all its pages); the rows, qrels and corpus of every processed question | `pages/`, `queries/`, `qrels/`, `corpus/` |
| negatives | hard negatives of the new Wikipedia gold pages, then the corpus again (none with `--limit 0`) | `negatives/index.jsonl`, `corpus/` |
| report | the dataset per source, the hard negatives, the gold annotation, consistency checks and the duplicate documents of the corpus (report only) | `stats.json`, `duplicates.jsonl` |

For each Wikipedia gold page the negatives stage asks the Wikipedia API for candidates: first `morelike` (the
articles with the most similar text), then category siblings (articles in the page's most specific categories). It
keeps at most `negatives.max_per_gold` (10) per gold page, skipping pages that are gold for any query, disambiguation
and list pages, pages titled with the answer, pages under `min_words`, and duplicates of a corpus document. Web gold
pages get none. `corpus/corpus.jsonl` holds gold pages and hard negatives with their text; `corpus/ids_gold.txt` and
`corpus/ids_hard_negative.txt` list the ids of each. Other page settings than the earlier builds' (`wiki_lang`,
`min_words`, `min_page_share`, `wayback`, `wayback_timestamp`) are refused (remove the dataset to change them), and
other negatives settings too (remove `negatives/index.jsonl` and the `negatives` section of `stats.json`). The
User-Agent contact comes from `DRA_CONTACT` in `.env`.

Every query goes to `val`: WebShaper keeps its gold criteria (they have texts, but no gold intermediate answers:
`level` partial), and FRAMES, which has no structure, gets silver criteria from the belief model's init call (the
multi-hop prompt, at most 8 criteria), as on any new dataset; its queries have qrels and a judged final answer, so
the certainty score against correctness can be measured, but not the observations and beliefs.

### trqa_validation

The trqa_validation dataset merges the TRQA wiki1 and wiki2 validation files (queries, intermediate info, qrels)
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

Each training dataset has a gold annotation per query: its criteria, each with its gold answers and gold documents,
as far as the source states them (`training_datasets/gold/annotations.py`). Every row has the same keys whatever the
source; what the source does not state is an empty value (`[]` or `null`), never a missing key.

| Source | Criteria | `answers` | `doc_ids` |
|---|---|---|---|
| InfoSeek | one per entity of the research tree that the question does not name, children before parents, the answer last (`multi_hop`) | the entity, and its title without a parenthetical | the first corpus chunk of its Wikipedia page and of the pages whose claims link to it |
| WebShaper | one per variable of the formalization, the farthest from the answer first, then the answer (`multi_hop`) | the answer only (final answer) | none (the URLs are the query's) |
| FRAMES | none | | |
| TRQA wiki2 | one per member, "<member>: <property>", the member named by the Wikipedia title of its first document ("San Fernando, La Union", not TRQA's label "San Fernando") (`closed`); then "any other member: <property>" (`rest`); then the aggregation, how the answer follows from the values (`aggregate`) (`multi_member`) | the member's value (`entity_values`); rest: none; aggregate: the final answer | the member's documents (`entity_values[].passage_ids`); rest and aggregate: none |
| TRQA wiki1 | the same | the same | the same |

Every TRQA member has its own criterion, so the rest criterion has no gold in the dataset: training can hide some
members under it (their values and documents then become its gold), as at inference, where the belief model lists
only the members it knows. The aggregate criterion has no documents since no document states the computed value.

`level` follows from the row and selects a group of queries:

| `level` | The row has | Training can use |
|---|---|---|
| `full` | criteria, each with its gold: gold answers and gold documents (a `rest` criterion none, an `aggregate` one answers only) | per criterion: its answers and its documents; the final answer and the qrels |
| `partial` | criteria, some without answers or without documents | the criteria that have them; the final answer and the qrels |
| `query` | no criteria | the final answer and the qrels only |

An InfoSeek query is `partial` when an entity of its criteria has no Wikipedia page in the tree (a question with a
page missing from the corpus is dropped). A WebShaper criterion has no documents, and only the answer criterion has
answers. The `annotations` section of `stats.json`, the same in every dataset, counts the rows per source and level,
the criteria per kind and with their answers, documents and gold, the texts per source (ok, failed, retry), and
records the writer of the texts. The loaders build the criteria from the source (with the page keys of their
documents); the build resolves the keys to documents (webshaper_frames: the fetched page, a page not fetched gives
none; infoseek: the first chunk of the page) and writes the rows to `queries/queries_all.jsonl`. The trqa_validation build
writes the same rows from the intermediate info. Query row: `id`, `source`, `split`, `text` (the question),
`answer`, `query_shape`, `level`, `doc_ids` (the qrels), `criteria`, `members` (multi-member queries: `entity`,
`value`, `doc_ids`), `entity_names` (InfoSeek: every entity of the tree; TRQA: the members), `property` and
`aggregation` (multi-member queries); criterion: `id`, `text`, `answers`, `doc_ids`, `kind` (`closed`, `rest`,
`aggregate`), `relations` (the source structure the writer reads); then `texts_ok`, `errors` of the texts (the writer
and its prompt hash are in `stats.json` and `criterion_texts.jsonl`, not per row).

### Criterion texts (end of every build)

An LLM (`gold_criteria.model`, the teacher) writes the criterion texts in the question's words, one call per query;
answers, documents and level come from the source (`training_datasets/gold/criterion_texts.py`). The prompts are the files
of `training_datasets/gold/prompts/`, one system and one user prompt per query shape:

- multi-hop (InfoSeek, WebShaper): one text per criterion, from its relations, with only the details the question
  states (a relation may hold more, such as a year). The writer orders the criteria so that each refers only to
  criteria before it ("the town of c1"), the answer last; the row's criteria are reordered and renumbered in that
  order (saved as `order` in `criterion_texts.jsonl`). An invalid order is asked once more with the problem named.
- set (TRQA): the property phrase (what is looked up per member, with its year and unit, no condition) and the
  aggregation (what the answer computes over "the members": the property, the conditions on its values and how many
  members it takes, not what selects the members, which the member criteria carry), from the question, TRQA's property label
  and its computation code. The member criteria "<member>: <phrase>" and the rest criterion "any other member:
  <phrase>" are built from the phrase; the aggregate criterion is the aggregation.

The writer runs before the split, so the rows are written with their texts; a build with texts to write needs
`OPENROUTER_API_KEY`. `texts_ok` is true when every text names none of the gold answers (a name inside a longer
entity name, such as Groningen in "University of Groningen", does not count; multi-member queries: neither the phrase
nor the aggregation names a member, a value or the final answer that the question does not name; multi-hop queries:
no number or capitalized word the question does not state), or when the query has no criteria (FRAMES, no LLM
call); false when the texts failed for good (a leak, an unstated word or an invalid order after one retry with them
given, no text for a criterion); null when the writer call failed (an error, a reply that is not JSON), and the next
build asks again. The number of criteria is the source's (no cap). Final texts are appended to
`criterion_texts.jsonl` as they are written, so a stopped build keeps them; a build reuses them for unchanged
criteria (the texts do not depend on the documents) and writes only the new queries' texts. Another
`gold_criteria.model` or other prompts than those of `criterion_texts.jsonl` are refused (remove the file to write
every text again). The prompts are hashed per query shape, so editing the set prompts leaves the multi-hop texts
valid. The queries of a split (its phase 1 run) are the full ones with `texts_ok` and at least one criterion, and in
`val` also the OOD ones. The `split` section of `stats.json` counts the queries per split, source and level, and the
full ones in no split (`no_split`).

## Phase 1: Trajectory collection

Phase 1 runs `dra_inference.py` with the frozen agent on the queries of one split (`--subset {split}`: the
dataset's rows with that `split`), every such query or the next `--limit` not run yet (inference resume). The mode
follows the split:

| Split | Mode | Belief model |
|---|---|---|
| `sft`, `val` | monitor: the agent never sees the belief | the query's gold criteria (`--criteria-source`, `shared.criteria_source`: gold, or silver on trqa_validation; a query without gold criteria: the teacher's silver ones, through the criteria bank, `gold_missing` in its `criteria_info`) and the teacher (`openrouter/qwen/qwen3.6-27b`) for the belief updates |
| `rl` | off | none |

The belief does not change what the agent does, so the documents and agent turns of every step are fixed, and
phase 3 can run the student over the steps of an off-mode run. The runs go to dra_inference's output root in its
usual layout (`run_outputs/{dataset}_{split}_{retriever}/{agent}/belief-monitor/` for `sft` and `val`,
`belief-off/` for `rl`), and phase 1 writes nothing else.

**Episodes.** Phases 2 and 3 read a split's run at their start (`belief_model_training/runs.py`), one episode per
finished in-domain query (`level` full, `texts_ok`, at least one criterion; the OOD queries of `val` are counted as
`out_of_domain` and left out):

| Level | Field | Source | Used by |
|---|---|---|---|
| step | `doc_ids`: documents shown in the search iteration (all its searches; none when they returned nothing); `iteration`, the belief model's step number (from 1), and `agent_iteration`, the agent's own (`iter`) | `trajectory/{qid}.jsonl` (`seen_docs`) | SFT replay, RL chains |
| step | `agent_turn`: the agent's reasoning and queries of the iteration (as the teacher read it in monitor runs) | `trajectory/{qid}.jsonl`, `belief/{qid}.jsonl` | the update prompt |
| step | `new_gold_ids`, `recall_so_far`: relevant documents first seen in the step, share seen so far | qrels | statistics, check of the saved run |
| step | `stated`: the criteria whose gold answer the step's agent turn states | gold criteria, `nugget_gold.py` | assign (expected claims) |
| step | `teacher_output`, `teacher_records`: the teacher's reply and its saved observations, candidate updates, changed criteria, issues and beliefs (monitor runs) | `belief/{qid}.jsonl` | SFT, replay check |
| episode | `answer_turn`: reasoning after the last search and the final answer; the teacher's final call on it (monitor runs, the `final` line) | `trajectory/{qid}.jsonl`, `generation/{qid}.md`, `belief/{qid}.jsonl` | final call |
| episode | `correct` ($y$; null when the judge failed) | `accuracy.jsonl` | $r_{\mathrm{final}}$ |
| episode | `gold_criteria`: criteria with `answers` and `doc_ids` | `queries/queries_all.jsonl` of the dataset | the criteria of the chains, all rewards |

**Gold of a candidate or spans.** A selected candidate is correct ($g = 1$), and an observation's spans state the
gold, when one of the criterion's gold answers occurs in it as whole words (case, accents and punctuation ignored; not
inside a longer entity name of the query's gold structure) or, for a number, when a number of the text equals it. A
text without a match is asked of the correctness judge (`rl.answer_judge_model`, YES/NO: is the candidate the gold
answer? / does the text state it?). A turn states a
criterion's gold answer by the same match; a (turn, criterion) pair without a match, whose gold documents were
shown before the turn, is asked of the same judge. The verdicts are cached in
`{output_root}/{training_data}/{policy_model}/answer_gold.jsonl`; `rl.answer_judge_model: null` uses the match only.
The rest criterion of a multi-member query has no gold yet (the gold lists every member as its own criterion) and is not
scored.

Only finished queries are episodes, as dra_inference counts them: the trajectory ends with its meta line and, in a
monitor run, the belief file exists. The qrels are read with the run's `dataset.min_relevance_score`. The runs a phase reads must share
their agent and searcher settings (`run_config.json`, the temperature included): phase 2 checks the sft
and val runs, phase 3 the rl and val runs, and either stops with the keys that differ. All runs take these
settings from the same inference config (`shared.inference_config`), so keep it unchanged across the phase 1
runs. For monitor runs, the loader also checks that the run's updater is the teacher on the gold criteria (and
that the run's criteria are the dataset's), that the belief file (schema 40, `SCHEMA_VERSION`) has one step per search iteration of the
trajectory, joined on `agent_iteration`, with the same step number and documents, and that the saved
`recall_so_far` matches the qrels. Episodes without a verdict in `accuracy.jsonl` (a run that stopped before its
evaluation) are counted and reported; `--eval-only` judges them.

## Scores and rewards

The same scores rate the teacher (phase 2), reward the student's chains (phase 3) and validate them
(`belief_model_training/scoring.py`, `chain_score`). Each criterion $k$ with gold answers (not the rest criterion)
is its own sequence of beliefs, one per step of the episode: a step without a call, or whose call failed, keeps
the belief of the step before, and the answer turn is the last step. Each criterion is scored on its own, then
the scores are averaged over the criteria, so neither a long episode nor a criterion with many updates weighs more.

**Gate** $f \in \{0, 1\}$ per call: 1 if the reply is not cut off and parses, names known criteria and passages of the call
only, and gives each candidate update a name and a candidate status, once per candidate (`criteria.GATE_ISSUES`). The chain's `format` is the share of its
update calls that pass. The final scorer's gate: not cut off, and a status parses.

**Source assignment** $\mathrm{assign}_k$: an F1, $2\,TP / (2\,TP + FP + FN)$, of criterion $k$'s observations
over the chain's calls, documents and turns together:

| Case | TP | FN | FP |
|---|---|---|---|
| passage $d$ new in the call (a judged passage is never shown again), $d$ a gold document of $k$ (not the aggregate) whose text as the call showed it (cut at the run's `max_passage_tokens`; training needs runs made with the token cut, the evaluation also reads older runs' `max_passage_chars`, and 3000 characters, reports whole, before either was recorded) has a gold answer of $k$ | an observation whose spans state it | none | |
| a turn of the call states the gold answer of $k$ (a call reads the turns of the steps before it that made none) | a claim whose span states it | none | |
| an observation that did not enter the belief (no verified span of its passage or turn, a claim without a candidate or valid claim, a document observation on a computed criterion) | | | yes |
| any other observation | neutral | | |

**WEP calibration** $\mathrm{cal}_k$: the mean over the steps of $1 - (p_{k,t} - y_{k,t})^2$, $p_{k,t}$ the
probability of the criterion's status and $y_{k,t}$ whether its selected candidate(s) cover the criterion correctly: a one-answer
criterion, whether its selected candidate is a gold answer (0 without one); a several-answers criterion, the share
of its gold answers among its selected candidates. A hop of a multi-hop query is a several-answers criterion with one
gold answer, so $y$ is whether its selected candidates include it; another valid answer (two entities that both fit
the hop) is not counted against it.

**Final** $r_{\mathrm{final}} = 1 - (s_T - y)^2$, $s_T$ against the correctness of the agent's final answer; 0
when the final call fails its gate. On a short-answer query $s_T$ is the final scorer's, so $r_{\mathrm{final}}$
rewards the scorer's reply only (its own GRPO stream, phase 3).

A part with nothing to score (no gold criterion, no expected or written observation, $y$ unknown) is None, not 1.

Two more per-criterion measures are reported, not rewarded: the recall of $k$'s gold documents (those that should
give an observation) among its document observations (`doc_recall_obs`); and the share of steps whose selected candidate is correct
(`value_accuracy`).

### Teacher scores

SFT copies the teacher, errors included, so phase 2 scores the teacher on the sft and val episodes before training
(`phase2_sft/teacher_scores.json`), on its replayed calls:

| Part | Measures |
|---|---|
| gate | share of calls that pass; the rate of each issue code per call |
| per criterion | mean over the chains of the mean over their criteria: `crit_assign_f1`, `crit_doc_recall_obs`, `crit_value_accuracy`, `crit_cal` |
| observation | precision, recall and F1 of document observations and claims, pooled |
| calibration | Brier score, ECE and AUROC (on the items with a binary $y$) of the statuses of every (criterion, step) against $y_{k,t}$; gold rate and count per status (reliability) |
| final | $s_T$ against $y$: AUROC, Kendall τ-b, Brier score, ECE (10 bins): the final scorer's (`certainty_scorer_*`), the fixed function of the final beliefs (`certainty_*`, and its mean, product and min) and its form anchored on the scorer's answer's candidates (`certainty_answer_*`), the last two baselines |
| rewards | mean `format`, `assign`, `cal`, $r_{\mathrm{final}}$ |

## Phase 2: SFT

A sample is one belief update call of the `sft` monitor run: its prompt $x$ (masked from the loss) and the
teacher's reply $z$ (trained). A trajectory with K search steps gives at most K + 1 samples: one per step that made
a call (a new passage) and the final call on the answer turn (task `score` on a short-answer query: kept when it
parses, its answer's candidates have no issue, and its probability is within 0.5 of $y$, inclusive so "possible" is kept for both $y$; rejection sampling so the student does
not copy the teacher's miscalibration; the counts per $y$ are printed). Each teacher reply is replayed through the inference
belief update to rebuild its exact prompt and check it against the saved records; a call is kept when it passes
the gate and, with `sft.require_verified_spans`, when it is clean (`criteria.UNCLEAN_REPLY`: all its observations
entered the trace, no status corrected by code (`no_support`, `no_new_support`), no `no_reason_opening`,
`wrong_previous`, `not_listed`, `bad_ref`, `remove_unlisted`, `bad_removal`, `unexpected_candidate`, `against_only`,
`certain_with_rival` or `unequal_tie`). The main process
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
from the chain's own $b_{t-1}$, as at inference. On a short-answer query the final call is the scorer's: it gets
$G$ replies on the chain's own $b_{T-1}$, the first one the chain's $s_T$; each reply takes its own advantage from
its $r_{\mathrm{final}}$, normalized within those $G$ replies, and the update calls take the chain's advantage
from format, assign and cal only, so neither task is credited with the other's reply. The agent is frozen, so the documents and agent turns of every
step are fixed. Training on the student's own states, not the teacher's, keeps it ready for its own earlier errors
(exposure bias; DAgger, Ross et al. 2011). A call's reply budget is the inference updater's (12,288 tokens, 16,384
for a multi-member query).

**Groups.** The student runs $G = 8$ chains per episode at temperature 1. A group whose chains all score alike
gives no gradient and is replaced by new episodes, with up to two more rollouts per iteration (DAPO's dynamic
sampling).

**Reward.** One reward per trajectory (`belief_model_training/rewards.py`), over every step and the final belief:

$$
R = w_{\mathrm{format}}\,\mathrm{format} + w_{\mathrm{assign}}\,\overline{\mathrm{assign}_k}
  + w_{\mathrm{cal}}\,\overline{\mathrm{cal}_k} + w_{\mathrm{final}}\, r_{\mathrm{final}},
$$

where $\overline{x_k}$ is the mean over the chain's criteria that have the part. All weights are 1
(`rl.reward_weight_{format,assign,cal,final}`). $R$ is the logged reward and the validation measure. On a
short-answer query $r_{\mathrm{final}}$ is the final scorer's, trained as its own stream (below).

**Advantage.** Each of the four terms is normalized within the episode's group (GDPO, arXiv 2601.05242), a chain
without the term counting as the group mean, then weighted and summed, so a term with a small spread is not
drowned and a tiny spread is not blown up by the floor $\sigma_{\min} = 0.05$:

$$
\hat{A}_i = \frac{\sum_{m} w_m\, \dfrac{r_{m,i} - \operatorname{mean}_m}{\max(\operatorname{std}_m, \sigma_{\min})}}{\operatorname{std}_{\text{batch}}\left(\sum_{m} w_m\, \cdots\right)},
$$

with mean and std over the $G$ chains of the group, a term with zero spread counting 0. There are two streams.
Every belief update call of chain $i$ takes $\hat{A}_i$ over the terms of the chain (format, assign, cal, and on a
report $r_{\mathrm{final}}$): a chain is judged against the other chains over the same episode, so the belief terms
also reach the observation tokens of the reply. On a short-answer query the final scorer gets $G$ replies on the
chain's own $b_{T-1}$, and each reply takes its advantage from its $r_{\mathrm{final}}$ alone, normalized within
those $G$ replies: the update calls are not credited with the scorer's reply, nor the scorer with the chain's belief.
Each stream is normalized on its own, so the two weigh 1:1 and $w_{\mathrm{final}}$ changes only the logged and
validation $R$ there. The two streams' rewards are logged apart (`update_reward_mean`, `scorer_reward_mean`).

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

**Open.** One training dataset per run (`shared.training_data`): InfoSeek (infoseek) and TRQA (trqa_validation) are not yet
mixed in one RL run. The non-gold passages are neutral in the source assignment; a judged sample of them will show
whether a precision term is needed. The rest criterion joins the rewards once members are masked into it.

## Evaluation (dra_inference)

The trained belief model is evaluated with `experiments/inference/dra_inference.py`: the frozen agent with no,
teacher or student belief model in `inform` mode (in monitor mode the scores would not depend on the belief).
Scores are the run's `metrics.md` (the inference evaluation, `experiments/inference/README.md#evaluation`; phase 1
runs get it too, being dra_inference runs): Recall@N, answer accuracy, the uncertainty group (s_T against the
correctness, with the formula and verbal-confidence baselines) and, on the training datasets and TRQA, the belief update group
(what the belief does with each gold document). Phases 2 and 3 score with `scoring.py`, which shares its measures
and gold match with the evaluation (`evaluation.metrics`, `evaluation.belief.gold`). The student is trained on the `multi_hop` (infoseek) or `multi_member` (trqa_validation) prompts; a
dataset of another query shape (browsecomp_plus: `multi_clue`) gives it prompts it was not trained on.

```bash
# serve the student on a GPU the inference workers do not use
CUDA_VISIBLE_DEVICES=0 vllm serve $RUN/phase3_rl/final --served-model-name belief-student --port 6020 --language-model-only &
export BELIEF_VLLM_API_BASE=http://127.0.0.1:6020/v1

COMMON="--dataset browsecomp_plus --policy-model glm --gpu-ids 1,2,3"
python experiments/inference/dra_inference.py $COMMON --belief-mode off
python experiments/inference/dra_inference.py $COMMON --belief-mode inform \
    --llm-criteria openrouter/qwen/qwen3.6-27b --belief-judge-model openrouter/qwen/qwen3.6-27b
python experiments/inference/dra_inference.py $COMMON --belief-mode inform \
    --llm-criteria openrouter/qwen/qwen3.6-27b --belief-judge-model vllm/belief-student --output <student output dir>
```

- The student is trained on the belief update and the final scorer, not the criteria init, so its runs take the
  teacher's criteria (`--llm-criteria`, read from the criteria bank) and only the updates and the final score
  come from the student.
- Give the student run its own `--output`: teacher and student runs share the agent, dataset and belief mode.
