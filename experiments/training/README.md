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
| Training data | `infoseek_hard` (multi_clue, as BrowseComp-Plus) and `trqa_val` (multi_member), pooled |
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
| `data_prepare` | the questions of a training dataset worth its gold annotation: the source as loaded, then the checks (no LLM) | `--dataset` (required) | `data/training_datasets_raw/{dataset}/` |
| `data_relevance` | the qrels, gold annotation and sft / rl / val split of the prepared questions (LLM calls: the fact LLM and the judge of the gold passages; none for trqa_val) | `--dataset` (required), `--limit`, `--judge-model`, `--fact-model` | `data/training_datasets/` (`queries/{dataset}.jsonl`, `qrels/{dataset}.txt`), build files in `data/training_datasets_raw/{dataset}/` |
| `generate` | dra_inference `--stage generate` on a split for each agent; then the run statistics | `--dataset`, `--subset` (required; sft, rl, val), `--policy-model`, `--limit` | the runs (dra_inference layout), `stats/` |
| `teacher` | dra_inference `--stage belief --belief-model {teacher}` on the sft or val split for each agent | `--dataset`, `--subset` (required; sft, val), `--policy-model`, `--limit`, `--teacher` | the runs' `belief/{teacher}/`, `eval/metrics.md` |
| `filter` | what is trained on: SFT samples from the teacher's replies, RL episodes (CPU) | `--teacher` | `filter/` |
| `sft` | LoRA SFT of the student on the filtered samples, merged | `--student` | `{student}/sft/final` |
| `rl` | GRPO over belief chains of the filtered episodes, from the student's `sft/final` | `--student`, `--init-model`, `--beliefs` | `{student}/rl/final` or `{student}/rl_own_init/final` |
| `evaluate` | dra_inference `--stage belief --belief-model M` on a run | `--dataset`, `--belief-model` (required), `--subset`, `--policy-model`, `--limit` | the run's `eval/metrics.md` |

A stage refuses the flags it does not take. `--limit` means the next N prepared questions in `data_relevance` and the queries the
dra_inference stage runs in `generate`, `teacher` and `evaluate`. `sft` and `rl` take no dataset flag: they pool every
training dataset and agent of `shared`, on fixed splits.

The stages generate, filter, sft and rl write under `BELIEF_TRAINING_ROOT` (`utils/config.py`:
`OUTPUT_ROOT/belief_training`). The `final/` of each training run is served by dra_inference on local vLLM as a
`--belief-model` named `{student}_{run}`: `qwen3.5_4b_sft`, `qwen3.5_4b_rl`, `qwen3.5_4b_rl_own_init`
(`utils.config.TRAINED_BELIEF_RUNS`, the students in `STUDENT_BELIEF_MODELS`).

```bash
T=experiments/training/train_belief_model.py
python $T --stage data_prepare --dataset infoseek_hard
python $T --stage data_relevance --dataset infoseek_hard --limit 500
python $T --stage data_prepare --dataset trqa_val
python $T --stage data_relevance --dataset trqa_val
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

## Stages data_prepare and data_relevance: Training datasets

A training dataset is built in two stages, so the LLM calls are spent only on questions worth them.
`data_prepare` makes no LLM call: it writes the source as loaded to `data/training_datasets_raw/{dataset}` (again only when the code that makes the
records changes: its hash is in `source.json`), then
runs the dataset's checks and writes the questions that pass to `data/training_datasets_raw/{dataset}/prepared.jsonl`.
`data_relevance` finds the gold passages (qrels), writes the rows and splits the prepared questions, with the LLM calls that locate
the gold passages: the fact LLM (frames: the property of each gold page; webshaper: the gold pages of each fact;
`--fact-model`, default `openrouter/qwen/qwen3.6-27b`) and the
judge of the gold passages (infoseek_hard, frames, webshaper; `--judge-model`, default `openrouter/openai/gpt-4o-mini`).
trqa_val makes no call (TRQA's qrels). The stage has no beliefs: the belief model makes its gold beliefs when it
first needs them, as at inference (see [Stage teacher](#stage-teacher-teacher-beliefs)). A rerun of `data_prepare`
costs nothing but the corpus reads. Each LLM phase shows a progress bar (on a terminal; off one, a line every 30 s)
with its outcomes and tokens, and the run ends with a summary: kept, dropped by reason, pending, the qrels count,
where the relevant passages of the gold pages come from (judged or chunk 1).

| Folder | File | Contents |
|---|---|---|
| `data/training_datasets_raw/{dataset}` (build folder, one per dataset) | `questions.jsonl`, `source.json` | every question of the source in one record format (`training_datasets/pipeline/source.py`, loaded by `datasets/{dataset}/source.py`), nothing removed; where it comes from and the hash of the code that made the records (a change rebuilds them) |
| | `corpus_titles.jsonl`, `web_pages/` | caches: the chunk ids of each gold page title looked up in corpus_wiki; webshaper's fetched web pages (and `web_pages/corpus_web.json`: what corpus_web was last written from, so an unchanged build keeps it) |
| | `prepared.jsonl` | data_prepare: the questions that pass the checks, each record with its `features` (the chunk ids of its gold pages; infoseek_hard: depth, branching, nodes) |
| | `facts.jsonl`, `gold_passages.jsonl` | data_relevance memories: the fact LLM's saved facts (frames, webshaper: per question its facts with their pages), the judge's saved labels (and extracted values); a reply that cannot be read (not JSON, no label) is saved too and is final the second time (the LLMs answer at temperature 0): failed facts, judge label `NO-LABEL` (not relevant); an API error is not saved |
| | `gold_passages_report.jsonl` | data_relevance (frames, webshaper), rewritten each run: per kept question its facts, their routed pages, relevant passages (label, extracted value) and YES-DIFFERENT passages, and the source of each gold page's relevant passages |
| | `stats.json`, `dropped.jsonl` | one section per stage (`prepare`: the funnel of the checks, `corpus`, `web`; `relevance`, `gold_passages`, `split`); the dropped questions with their stage, filter and reasons |
| `data/training_datasets` (shared by every training dataset, laid out like an inference dataset) | `queries/{dataset}.jsonl` | data_relevance: one row per query: the question and answer, its source, its split, its gold passages (see [Rows](#rows)) |
| | `qrels/{dataset}.txt` | data_relevance: the gold documents of each query (TREC qrels, binary for every dataset) |
| | `belief_bank/silver/{dataset}.jsonl` | the extractor's silver beliefs (dra_inference) |
| | `corpus/corpus_web.jsonl` | data_prepare (webshaper): WebShaper's web gold pages, until the user merges them with corpus_wiki under `DATA_ROOT` |

| Dataset | Questions | Corpus | Splits |
|---|---|---|---|
| `infoseek_hard` | InfoSeek-Hard | corpus_wiki: the original release's `wiki-25-512.jsonl` (Wikipedia 2025 in chunks of 512 tokens), under `DATA_ROOT/infoseek/corpus` with its indices | sft / rl / val |
| `frames` | FRAMES (no tabular or post-processing questions) | corpus_wiki | sft / rl / val |
| `webshaper` | WebShaper | corpus_wiki + corpus_web (`data/training_datasets/corpus/corpus_web.jsonl`: its web gold pages, cut the same way; merged by the user for the index) | sft / rl / val |
| `trqa_val` | TRQA wiki1 + wiki2 validation | TRQA's `wiki_partial` | sft / rl / val |

`data_relevance` is incremental, like dra_inference: `--limit N` processes the next N prepared questions not processed
yet, `--limit 0` processes none and rewrites the dataset files, no `--limit` processes all. It refuses a dataset whose
queries file holds a question the prepared file no longer has (data_prepare rerun with other checks): remove
`queries/{dataset}.jsonl`, `qrels/{dataset}.txt` and the `relevance` and `split` sections of `stats.json` first (the
LLM memories are reused).

```bash
python experiments/training/train_belief_model.py --stage data_prepare --dataset infoseek_hard
python experiments/training/train_belief_model.py --stage data_relevance --dataset infoseek_hard --limit 500
python experiments/training/train_belief_model.py --stage data_prepare --dataset frames
python experiments/training/train_belief_model.py --stage data_relevance --dataset frames --limit 30
python experiments/training/train_belief_model.py --stage data_prepare --dataset webshaper
python experiments/training/train_belief_model.py --stage data_relevance --dataset webshaper --limit 30
python experiments/training/train_belief_model.py --stage data_prepare --dataset trqa_val
python experiments/training/train_belief_model.py --stage data_relevance --dataset trqa_val
```

Checks of `data_prepare`, in order; a question is dropped at its first failed check (`dropped.jsonl`, stage
`prepare`, with the filter and the reasons):

| Dataset | Checks |
|---|---|
| every one | `duplicate` (a repeated question, its first row kept), `answer` (empty, or stated by the question as whole words; answers shorter than 4 characters or numeric are not looked for) |
| `infoseek_hard` | `hardness` (the tree's (depth, max branching) not in the hardness cells), `tree` (a node without an href, a repeated node id that cannot be resolved, a claim target not in the tree; an attribute node that reuses an entity node's id gets its own id, `{id}#{k}`, when each claim names one child with that id), `hop_named` (the question names an entity node with children), `pages` (a gold page not in corpus_wiki, or whose first chunk has no text), `claim_target` (see below) |
| `frames` | `reasoning_type` (tabular, post processing), `no_gold`, `links` (a link that is not a URL or not an English Wikipedia page), `pages` |
| `webshaper` | `no_gold`, `fetch` (a web gold page whose fetch may still succeed: pending, the next run fetches it again), `pages` (no gold page in the corpus; a page not found is left out, as WebShaper's links often no longer serve their page) |
| `trqa_val` | `ambiguous_id` (an id naming two questions in one TRQA split: both dropped), `test_overlap` (also a TRQA test question), `no_gold` (no relevant qrels), `pages` (a qrels document not in `wiki_partial`, or without text) |

The splits `sft`, `rl` and `val` are views of split `all`: the queries loader reads `--subset sft` as the rows with
`split` `sft` and their qrels (`layout.is_split_view`), so dra_inference runs a split as on any dataset and its run
folders are named by the split. Every data_relevance run sets each row's `split`: every query of every dataset is
split with the shares of the `data.split` section of `experiments/configs/belief_training.yaml`:

| Split | Queries | Stages | Used by |
|---|---|---|---|
| `sft` | 30% | generate, teacher | sft: SFT samples |
| `rl` | 60% | generate | rl: RL episodes |
| `val` | 10% | generate, teacher | sft: validation loss; rl: validation; evaluate |

Training leaves out a query whose gold beliefs the teacher did not make. A query keeps the split it has in the queries file, so no query moves
when the dataset grows, and `--limit` on one split never reaches the queries of another. The new queries of a
run are taken in the order of a hash of their id (not the source order), and each goes to the split furthest below
its share, so the shares hold at every dataset size (20 queries: 6 sft, 12 rl, 2 val; 30: 9, 18, 3). The
shares and per-split counts are recorded in the `split` section of `stats.json`; other shares are refused (remove
the `split` section to re-split every query). One stage of a dataset runs at a time (`.build.lock` in its folder).

Classic QA sets (NQ, HotpotQA, 2WikiMultiHopQA, MuSiQue, TriviaQA) are not used: they need too little search.

| Source | Questions | Gold pages / question | Gold pages: wiki / web | Question words | Answer words |
|---|---|---|---|---|---|
| InfoSeek-Hard | 18,110 | 3.9 | all wiki | 42 | 2.9 |

### infoseek_hard

The InfoSeek-Hard questions (`Lk123/InfoSeek`, `data/InfoSeek-Hard-18K.jsonl`, pinned), run over the corpus their
authors search (VectorSpaceLab/InfoSeek, `research/InfoSeek/rl/InfoSeek/Retrieve`): `wiki-25-512.jsonl` of
`Lk123/wiki-25-512` (pinned), Wikipedia 2025 split into chunks of 512 tokens, each chunk with its page title on the
first line. The first data_prepare downloads it (18.5 GB) into `DATA_ROOT/infoseek/corpus/`; it is used unchanged, a doc id
is the chunk's `id` (its line, the row of the released index) as a string. The indices go next to it
(`DATA_ROOT/infoseek/indices`).

Hardness: of the 18,110 distinct questions, data_prepare keeps those whose research tree has a (depth, max branching)
cell of `datasets/infoseek_hard/prepare.py` `HARD_CELLS`: (6, 1) 304, (5, 1) 921, (5, 2) 675, (5, 3) 6, (4, 3) 1,106, (4, 4) 6,
(4, 5) 1, (3, 6) 2, (2, 6) 1, 3,022 in all. Depth counts the nodes of the longest root-to-leaf path, root included; branching is the most
children of one node. Both count every child: an entity clue (a node with its own page) or an attribute clue (entity
`None`, an exact fact of its parent). The trees have 4 to 6 nodes (48 have 7 or 8), so deep trees are chains and
wide trees are shallow; the cells are all the deep chains and every tree both deep and wide (depth 4 from branching 3). Every question of the
cells that passes the other checks is kept. The `prepare` section of `stats.json` counts each cell in the source,
once repeated questions are removed, and kept.

Claim targets: InfoSeek writes the claims of a node from the first 1024 tokens of its page, so each claim must be
found in the first two chunks of that page in corpus_wiki. An entity target is looked for by its names (the entity, the
title of its page, each without its disambiguation), an attribute target by every number of the claim (thousands
separators removed); a claim on an attribute without a number is not checked. Case, accents and punctuation are
ignored. A question with a claim not found is dropped (`claim_unreachable`, with the claim, the page and the names or
numbers tried); the `claims` counts of the `prepare` section give the claims checked per kind and the ones missed.

InfoSeek records each hop's gold page as a Wikipedia title (`href` of a research tree node), with no corpus id and no
position in the page. data_prepare matches each title exactly against the first line of every chunk (HTML entities
decoded, Unicode NFC, underscores as spaces, first letter upper case on both sides). InfoSeek's construction code
(`ConstructTreeData.py`) writes every claim from the first 1024 tokens of the page of the node it expands, so a claim
sits in chunk 1 or chunk 2 of that page, with no rule for which. One fact per claim (the claim with its node ids
written as entities, its value the target entity) is judged on chunk 2 (see [Gold passages](#gold-passages)): the
claim's gold chunk is chunk 2 when the judge finds it there (YES-SAME), else chunk 1; a page of one chunk needs no
call. A page's gold chunks are those of its claims, chunk 1 for a page without claims. The query's qrels (binary) are
the gold chunks of every gold page, so every gold page has at least one. A question with a gold page that is not in the corpus (renamed, redirected or missing in the 2025 dump) is
dropped by data_prepare (`page_not_in_corpus`). The titles looked up are kept in the raw folder's
`corpus_titles.jsonl`, so data_prepare reads the corpus once, and only when it has new titles. The `corpus` section of `stats.json` records the
corpus revision (another revision is refused) and the titles found; the `gold_passages` section the claims, the
judge's labels and how many gold chunks are chunk 1 or chunk 2.

### frames

The FRAMES questions (`google/frames-benchmark`, `test.tsv`, pinned) without the tabular and post-processing ones
(tables are stripped from corpus_wiki; post-processing failures are not research failures), over corpus_wiki. A
question is kept when every gold page is an English Wikipedia page found in the corpus by its title, as in
infoseek_hard. An LLM (`--fact-model`) writes one fact per gold page from the question, the answer and the
page titles: the entity is the page title, the property is what the question needs from the page. FRAMES does not say
which page states which value (its answers are often computed), so the value is unknown and the judge extracts it.
Each fact is judged on every chunk of its page; see
[Gold passages](#gold-passages) for the relevant passages and the chunk-1 fallback. The queries are split like every dataset's;
frames has no gold beliefs (`gold_beliefs` `annotation` without members), so the teacher makes none and training
leaves its queries out until frames takes its beliefs from the question.

### webshaper

The WebShaper questions (`Alibaba-NLP/WebShaper`, pinned). Its English Wikipedia gold pages are found in corpus_wiki
by title; its other gold pages (web pages, other-language Wikipedia pages) are fetched, live or else from the
Wayback Machine snapshot closest to 2025-06-01 (`data.web`; the User-Agent needs a contact, `DRA_CONTACT` in `.env`),
and cut like corpus_wiki into `data/training_datasets/corpus/corpus_web.jsonl` (passages of at most 512 tokens of the
Qwen3.5-4B tokenizer, whole sentences, the page title on the first line, ids `web_<url hash>_<k>`). corpus_web holds
only WebShaper's gold pages (no web filler, no hard negatives): a web passage is always some question's gold page.
Merge corpus_wiki and corpus_web for the index. The fetches are data_prepare's (kept in the raw folder's
`web_pages/`). A page not found or not fetched has no passages; a question without any page in the corpus is dropped
(`page_not_in_corpus`); a page whose fetch failed transiently is fetched again by the next data_prepare runs (3
fetches in all), its question pending until then. `data.web.refetch_failed: true` fetches every failed page once
more in the next data_prepare run (then set it back to false); a page fetched since keeps its passages. Each triple
of the formalization is one fact, written by code with nothing the dataset does not state: a constant or the answer is
written out (a known value, checked by the judge); a variable object is an unknown value (extracted by the judge); a
variable subject is described by its other triples whose other term is known ("the entity that took place on 16 August
1986"), else "an entity". An identity triple (a variable, a copula such as "has name" or "is", the answer) makes the
variable the answer everywhere and is no fact; a repeated triple counts once; a triple that is not three terms is
skipped, and so is one with no known term (an unnamed subject and an unknown value, which any passage on the property
would fit). A question left without any fact is dropped (`no_facts`, no LLM call). One LLM call per question
only maps each fact to the gold pages likely to state it, from the pages' titles (a web page by its fetched title);
this routing never names an entity or a value. See [Gold passages](#gold-passages) for the routing safeguards, the
relevant passages and the chunk-1 fallback.

### Gold passages

The gold passages of infoseek_hard, frames and webshaper come from an LLM judge (`--judge-model`, default
gpt-4o-mini at temperature 0, as TRQA): one call per (fact, passage) pair, the fact as an entity (a name, or a
description), a property, a statement and a value. A fact holds only what the dataset states. A fact with a known value
gets TRQA's property check (mahta-r/total-recall-qa): YES-SAME (the passage states the fact's value for the entity),
YES-DIFFERENT (another value), NO-RELATED (no value, on the fact's topic) or NO-UNRELATED. A fact with an unknown value
(its statement has a blank) gets the extract prompt: the judge writes the value the passage states, `<value>-RELATED`
(label `VALUE`, the value saved), else NONE-RELATED or NONE-UNRELATED. A passage is relevant for YES-SAME or VALUE;
every other label is not. The qrels are binary for every dataset. The gold pages are given by the dataset; the facts
only help locate their relevant passages, and every gold page ends with at least one relevant passage. On frames and
webshaper a fact is judged on every passage of its pages; a fact without a page is judged on every page, and a page no
fact maps to is judged on every fact; a gold page without a judged relevant passage gets its first passage. No question
is dropped for a lack of judged relevant passages; the `gold_passages` section of `stats.json` counts the pages per
source (`pages`: `judged`, `chunk_1`) and the questions whose every page is on chunk 1 (`questions_all_chunk_1`: a
sign of wrong facts or page assignments), the facts with a known or unknown value, the routing safeguards used,
the distinct values extracted per fact (`extracted`) and per webshaper variable (`variables`), and the facts with a
YES-DIFFERENT passage; `gold_passages_report.jsonl` holds the details per question. The judged pairs are kept in `gold_passages.jsonl` (another judge or prompts are
refused); a pair without a label is judged again by the next data_relevance run, its question left unprocessed until then. The
prompts are `training_datasets/gold_passages/prompts/judge_*.txt`; the facts' prompts `datasets/{frames,webshaper}/prompts/facts_*.txt`
(the retry note `gold_passages/prompts/facts_retry_note.txt`).

### trqa_val

The trqa_val dataset merges the TRQA wiki1 and wiki2 validation files (queries, intermediate info, qrels)
into `queries/trqa_val.jsonl` and `qrels/trqa_val.txt`: a query and its intermediate info become one row, the
members with their values and documents in `members`, with `property` (the label) and `aggregation`. It searches
TRQA's `wiki_partial` corpus. The gold loaders (`load_query_intermediate_info`) read the intermediate info of a
training dataset from its rows, in TRQA's record format; the gold beliefs are made from the members
(`trqa_gold.with_member_beliefs`).

Left out by data_prepare: the one query whose text is also a TRQA test query, and the 14 wiki2 ids that each name two
different questions (28 queries). 1,145 queries remain (91 wiki1, 1,054 wiki2), all with qrels: 343 sft, 687 rl
and 115 val. `--limit N` adds the next N prepared queries not yet in the dataset
(file order, wiki1 then wiki2). The `relevance` section of `stats.json` records the queries processed and remaining.
Its rows have the keys of the other training datasets' (source `trqa_wiki1` / `trqa_wiki2`).

### Rows

Each training dataset has one row per query (`training_datasets/annotation/rows.py`): the question, its answer, its
split and its gold passages. The rows hold no beliefs: the belief model's gold beliefs are made by stage teacher, as
at inference: on infoseek_hard and webshaper (multi-clue) the teacher's init beliefs from the question; on trqa_val
one belief per member of the row, then the aggregate, with the teacher's texts
(`belief_model_component/init/trqa_gold.py`, as TRQA's); frames has none. Every row has the same keys whatever the
source; what the source does not state is an empty value (`[]` or `null`), never a missing key.

Query row: `id`, `source`, `split`, `text` (the question), `answer`, `doc_ids` (the qrels, binary), `members`
(multi-member queries: `entity`, `value`, `doc_ids`; a member named by the Wikipedia title of its first document,
"San Fernando, La Union", not TRQA's label "San Fernando"), `entity_names` (InfoSeek: every entity of the tree;
TRQA: the members and their labels), `property` and `aggregation` (multi-member queries). The query shape is the
dataset's (`layout.DATASET_SPECS`), not the row's. The prepared records keep their source structure (InfoSeek's
tree, WebShaper's formalization, data_prepare's `annotation`); data_relevance uses it only to find the facts.

The `split` section of `stats.json` counts the queries per split and source.


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
left out), a final answer and gold beliefs (`runs.read_gold_rows`: the teacher's banked ones, see stage teacher): its search
iterations and the
answer turn's reasoning and answer as the belief stage reads them (`orchestration.belief_stage.read_saved_query`), y, its gold beliefs, and the run's corpus,
retriever and passage cut. Documents are read from the corpus by id, as in the belief stage.

**Statistics** (`trajectories/stats.py`), printed and written to `stats/{dataset}_{split}_{agent}.json`: episodes
and the queries left out (gold beliefs not counted: they are banked by stage teacher), the agent's accuracy, the search iterations with a document per episode (mean, median,
90th percentile, max; they set `filter.rl_max_chain_steps`).

## Stage teacher: Teacher beliefs

Stage teacher runs dra_inference `--stage belief --belief-model {teacher}` (`shared.teacher` or `--teacher`) on the
`sft` or `val` split (`--subset`; `rl` needs no teacher) for each agent, over the generated trajectories: the
teacher's beliefs in the run's `belief/{teacher}/` (the SFT samples of stage filter), then their evaluation in the run's `eval/metrics.md` (the teacher's
s_T against y, the baseline of the trained model). Only finished trajectories get beliefs; the stage prints how many
queries of the split have none yet. The stage resumes, and `--limit N` caps the queries it runs. Settings:
`belief_stage.belief_workers`, `quiet` (shared with stage evaluate). The stage needs no GPU: the documents are read
from the corpus by id.

Before its runs, the stage banks the gold beliefs of every query of the dataset, every split (`teacher/run.py`
`bank_teacher_beliefs`), in `data/training_datasets/belief_bank/gold/{dataset}.jsonl`, as the belief stage makes
them: on infoseek_hard and webshaper the teacher's init beliefs from the question (multi-clue, as BrowseComp-Plus;
`gold_beliefs: question` in `layout.DATASET_SPECS`); on trqa_val the beliefs of each row with the property phrase and
aggregation written by the teacher (`belief_model_component/init/trqa_gold.py` `DatasetGoldBeliefs`, the prompts in
`belief_model_component/prompts/trqa_gold/`, as TRQA's gold beliefs at inference). They are the gold beliefs of
training (SFT replay, init samples, RL from gold beliefs), the `rl` split's included. Banked ones are kept; a query
whose beliefs failed (a writer error) is asked again by the next run and left out of training until then; texts that
failed for good (an answer leaked after the retry) are banked as failed.

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
| init | on `filter.sft_init_datasets` (infoseek_hard), each query of the split: the init prompt and the gold beliefs (infoseek_hard: the teacher's banked multi-clue init), in the layout of the prompt's example | at most the prompt's cap of beliefs |

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
