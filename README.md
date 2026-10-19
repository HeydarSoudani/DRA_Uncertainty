# DRA Training

Inference and training pipeline for **Deep Research Agents (DRA)** — agentic
retrieval-augmented models that iterate *think → search → observe → report* over
a document index.

**Supported agents** (the LLM is picked automatically per agent):

| Family | Agents |
|---|---|
| Instruction-tuned (API) | `react`, `selfask`, `searcho1` (claude-sonnet-4-6) |
| RL-trained (vLLM) | `searchr1`, `research`, `stepsearch`, `drtulu`, `glm`, `oss_20b`, `oss_120b`, `tongyi`, `cpm_explore` |
| Outline / report (vLLM) | `webweaver`, `cpm_report` |

**Datasets:** `trqa` (Wikipedia / e-commerce), `neuclir` (news + technical, 2022–2024),
`browsecomp_plus`, `ragtime` (multilingual news, 2025).

**Retrievers:** `bm25`, `spladepp`, `spladev3` (sparse); `bge`, `e5`, `dpr`,
`contriever`, `reasonir`, `qwen3_emb_{0.6b,4b,8b}` (dense).

## Installation

```bash
pip install -e .
```

This registers all packages so imports resolve from any working directory.
Set `DRA_DATA_ROOT` (corpus + indices), `DRA_OUTPUT_ROOT` (run outputs) and `DRA_CRITERIA_ROOT`
(criteria banks, default the repo's `data/`) to control where data is read/written.

## 1. Download datasets

```bash
# trqa  (subset: wiki1 | wiki2 | ecommerce)
python src/indexing_corpus_dataset/download_datasets.py trqa --subset wiki1

# neuclir  (subset: news | technical; 2024 news also writes report nuggets)
python src/indexing_corpus_dataset/download_datasets.py neuclir --year 2024 --subset news

# ragtime  (2025; queries, qrels, nuggets, English corpus)
python src/indexing_corpus_dataset/download_datasets.py ragtime

# browsecomp_plus  (corpus only)
python src/indexing_corpus_dataset/download_datasets.py browsecomp_plus --skip-queries-qrels
```

Canonical layout written under `$DRA_DATA_ROOT/<dataset>/`:

```
queries/queries_{split}.jsonl     {"id","text","answer"?}
qrels/qrels_{split}.txt           TREC: qid 0 docid rel
nuggets/nuggets_{split}.jsonl     {"id","nuggets":[{"id","question","answers","importance","support_docs"}]}
corpus/{name}.jsonl               {"id","contents"}
```

## 2. Build index

```bash
python -m indexing_corpus_dataset.index_builder \
    --retriever qwen3_emb_4b --dataset browsecomp_plus \
    --use_fp16 --max_length 4096 --batch_size 16 --faiss_type Flat --save_embedding
```

On Slurm: `sbatch scripts/run_index_builder.sh` (sets per-retriever args
automatically). Build config defaults live in
`src/indexing_corpus_dataset/configs/index_build.yaml`.

**Test the build** — small smoke index over gold + distractor passages, then
checks entity recall:

```bash
python src/indexing_corpus_dataset/index_builder_test.py \
    --retriever bge --dataset neuclir --query-limit 20
```

## 3. Inference

Run via the wrapper script (edit `DATASET` / `RETRIEVER` / `POLICY_MODEL` /
`BELIEF_MODE` at the top):

```bash
sbatch scripts/run_dra_inference.sh
```

Underlying command:

```bash
python experiments/inference/dra_inference.py \
    --dataset browsecomp_plus \
    --retriever qwen3_emb_4b \
    --policy-model glm \
    --belief-mode monitor \
    --num-gpus 0
```

Quick checks:

```bash
# single-query smoke test
python experiments/inference/dra_inference.py --dataset browsecomp_plus --limit 1 --num-gpus 0
# evaluate already-saved runs, no generation
python experiments/inference/dra_inference.py --dataset browsecomp_plus --eval-only --num-gpus 0
```

A run is evaluated from its saved files, both at its end and with `--eval-only`, so the two write the same
`summary.json`. The judge verdicts are kept in the run directory (`accuracy.jsonl`, `report_eval/`) and
reused for queries whose input is unchanged, so `--eval-only` on an unchanged run makes no LLM call and needs no GPU. The terminal log prints one
EVALUATION SUMMARY block in the order of `summary.json`: retrieval (seen docs), generation (correctness or nuggets, and
length), trajectory, then the time of each evaluation stage. Cited-doc and fusion metrics and the full @k
tables are only in `summary.json`. `_eval_results_cache*.pkl.gz` files left by older code are no longer
read and can be deleted.

Run defaults (top_k, rerankers, criteria LLM, eval k-values, …) are in
`experiments/configs/dra_inference.yaml`.

### Belief model

`--belief-mode monitor` plugs a passive belief model (`src/belief_model_component/`) into any agent; `off` (the
default) disables it. It never changes the trajectory. The belief model follows Agent-BRACE (Singh et al., 2026,
arXiv:2605.11436), adapted to search: the environment is the retriever, the observation is the documents of a search
iteration, and the belief state gives each criterion of the query one of the seven statuses of
Agent-BRACE's verbal belief scale. At the start of each sample it derives a fixed list of
criteria from the query (`llm_criteria`, at most `max_criteria`, null = the dataset's default in `layout.DATASET_SPECS`;
for a report request it is the number of criteria the extractor may add, asked in the prompt).
The dataset's `query_shape` (`layout.DATASET_SPECS`) picks the prompt, a shared core plus one block per shape, and the
model only lists the pieces of information a complete response must establish: the clues of a single target (BCP,
copied verbatim), the set, one "member: property" per known member and a last "any other member: property" (for the
members found only by the search) of a set query (TRQA), or, for a report
request (NeuCLIR, RAGTIME), every constraint the request states (no cap) followed by at most `max_criteria` (5)
criteria the extractor adds, as one plain list (the limit on added criteria is asked in the prompt; code never cuts a
report list). A report criterion keeps the request's own limits in its
wording; what the request leaves out is written into the criterion it narrows ("..., not ..."), an exception
("unless ...") is its own criterion, and background on the asker is kept only when it limits what information applies
(country, location, situation); every report criterion reads on its own, naming its subject instead of pointing back
("it", "these"). Criteria are distinct: none asks for what another one asks for. Each criterion is `closed` (one fact), `open` (several parts or answers), `rest` or `aspect`
(one aspect of a report topic), set by code from the shape: all clues are closed, the set (always the first criterion)
is open, its members closed and its "any other member" criterion rest (added by code when the model leaves it out),
all report criteria are aspects. Entity reachability counts the member criteria only.
Criteria bank (`criteria_bank: true`, the default): a query's criteria are read from
`CRITERIA_ROOT/{dataset}/criteria_bank/criteria_{split}.jsonl` and extracted only when missing, then added to it
(`BankedCriteriaSource`). A line is used only when the query id and text, `llm_criteria`, the query shape,
`max_criteria` and a hash of the init prompt all match, so a changed prompt or cap extracts again; a truncated reply is asked once more and fails when truncated again, and
a failed extraction is never added, so every criteria list a run uses is banked. A query extracted by two processes at
once keeps the line banked first. Runs and `analysis/criteria_reachability.py` share the bank, so the analysis assesses the criteria the
runs use; the criteria are never scored against gold during a run. At the end of each search iteration the belief model computes the per-step signals of
the report (`papers/ACL_2027__Uncertainty_Quantification_for_DRAs/report`, Section "Instantiation"):

| field | report | meaning |
|---|---|---|
| `doc_novelty` | ν^D | fraction of the iteration's docs whose id was not seen in an earlier iteration (0, 0.2, ..., 1 for 5 docs) |
| `criteria_delta` | Δ^D | change of the belief state over the signed status levels (doubtful -2, unlikely -1, unknown 0, ..., certain 4), summed over the criteria; positive when the passages cover more of the criteria, negative when they turn against them |
| `query_novelty` | ν^q | novelty of the iteration's queries vs earlier queries: 1 − max cosine similarity |
| `criteria_attempts_after` | a | per criterion, the number of iterations whose queries directly targeted it (a state, like the belief state) |

Extra saved information: `new_item_precision` and `recall_so_far` (against the qrels at the dataset's `min_relevance_score`;
NeuCLIR and RAGTIME use grade 3, the very valuable docs, while the end-of-run `GradedRecall@N`
counts every official level with its gain) and per-step intermediate answers (BrowseComp-Plus and
TRQA, all agents except `cpm_report`): after each iteration the agent's own model is asked, from its trajectory so
far, for its most likely answer(s) in its answer format; it may give one, several or none (`IntermediateAnswerSignal`,
prompt in `src/belief_model_component/prompts/`). They are saved as is, not evaluated. Embeddings come from the retriever's encoder, so with a sparse retriever
`query_novelty` is null. The criteria signals use LLM judges (`src/belief_model_component/judges.py`,
model `belief_judge_model`, default `llm_criteria`; the query scorer's model is `query_scorer_model`, default
`belief_judge_model`):

- Belief state and `criteria_delta`: the belief updater (Agent-BRACE's b_t = f(query, b_{t-1}, o_t)). b_0 gives
  every criterion the status "unknown". One call per iteration sees each criterion with its current status and its
  attached evidence (the last 4 passages, each shown as the spans the updater cited that occur verbatim in the passage
  and the passage's title line), plus the passages of the iteration it has not judged in an earlier iteration (none: no call, belief
  unchanged; the passages of an iteration whose update failed are shown again when retrieved again). It sees neither the search queries nor the signals. It lists only the criteria those passages
  support, contradict or relate to (including passages that point against a criterion): the passages that support
  or contradict each, with one or more verbatim spans each, and the new status, one of the seven
  statuses of Agent-BRACE's verbal belief scale, lowest first:

  | level | status | the passages found so far ... |
  |---|---|---|
  | -2 | doubtful | contradict it, and none supports it |
  | -1 | unlikely | point against it without contradicting it, and none supports it |
  | 0 | unknown | do not relate to it (b_0) |
  | 1 | possible | relate to it but give none of its parts directly, or disagree on it |
  | 2 | probable | give some parts of it, or give it only indirectly or by combining passages |
  | 3 | almost certain | give most parts of it directly, and none contradicts it |
  | 4 | certain | give all parts of it directly, and none contradicts it |

  The parts of a criterion are the pieces of information it asks for. The positive levels rate how much of the
  criterion the passages give, by one passage or several; the negative levels rate evidence against it. Agent-BRACE
  ranks "unknown" lowest (nominal probability 0); here it is the neutral level 0, so a contradiction lowers the level
  and new coverage raises it. "certain" is the paper's "confirmed / certain". A search that finds nothing on a
  criterion never lowers it. The
  status words are Agent-BRACE's; their evidence descriptions adapt them to retrieval and draw on GRADE's certainty
  factors (directness, consistency, completeness). A shape block refines what support means for its queries. The guidance lives in the prompt only:
  `BeliefState.update` takes every status as written, so a status may stay, rise or fall by any number of levels. A
  criterion the updater leaves out, or gives no valid status, keeps its status. Each span is checked on its own (a span joined with "..." is split first);
  only cited passages with a verified span are attached, once per criterion and role. This guards the evidence, never
  the status. `criteria_delta` and the analysis read each status as its signed level (`STATUS_LEVEL`, -2 to 4).
- `criteria_attempts_after`: one more call per iteration scores every (query, criterion) pair as 0, 0.5 or 1. It sees
  each criterion's status and the verified spans of its latest evidence passage, so a query naming
  an entity the evidence already linked to a criterion counts as targeting it; a query that restates most of the user
  query scores at most 0.5. One query may target several criteria. A criterion that some query of the iteration
  scores 1 on is `criteria_targeted` and its attempt count grows by 1 (once per iteration, however many queries target
  it; 0.5 never counts; whatever its status). When the call fails the attempts are unchanged.

#### Inform mode

`--belief-mode inform` computes the same signals and also shows them to the agent: right after each
iteration's search results, one `<belief_state>` tag is appended to the trajectory (`src/belief_model_component/belief_state.py`):

```xml
<belief_state step="3">
  <criteria>
    <k1 status="certain" attempts="1">born in the 1960s</k1>
    <k2 status="possible" attempts="2">won a regional award</k2>
    <k3 status="unknown" attempts="0">the films they directed</k3>
  </criteria>
  <signals doc_novelty="0.40" criteria_delta="+2" query_novelty="0.81"/>
</belief_state>
```

The tag is descriptive only: each criterion's status and attempts, then the step's three signals; gold-based fields (`new_item_precision`,
`recall_so_far`, relevant counts) stay in `belief/{qid}.jsonl` for analysis. A null signal is left out.
The system prompts are unchanged. Where the tag goes:

| agents | place |
|---|---|
| searchr1, research, stepsearch, searcho1, react | appended to the prompt after the search-result block |
| selfask | between `Follow up: …` and the `Intermediate answer:` prefill (the initial retrieval's tag after the prompt) |
| glm, oss, tongyi, drtulu, cpm_explore | appended to the iteration's last tool / result message |
| webweaver | appended to the planner's search observation |
| cpm_report | appended to the current information of the next plan / write prompt |

The tag is saved once, as `belief_state` on the search step in `trajectory/{qid}.jsonl`. Tags the model writes itself are removed in the prompt-string agents. `belief_aware`
follows the flag like every other agent; in `inform` mode its system prompt also introduces the tag (one section,
`belief_section.txt`, the same for every task, with no advice on how to use it), and in `monitor` and
`off` modes the prompt never mentions it.

#### Report evaluation (Auto-ARGUE)

On the report datasets (`report_eval: "argue"` in `layout.DATASET_SPECS`: NeuCLIR, RAGTIME) every run, and every
`--eval-only` pass, scores the reports against the nuggets with the official
[Auto-ARGUE](https://github.com/hltcoe/auto-argue) package (`src/evaluation/answer/argue.py`). There is no flag. The
package runs unmodified with our judge (`judge_model`, Qwen3-32B, temperature 0, reasoning off) instead of its
Llama-3.3-70B.

- Input: `generation/{qid}.md` split into sentences, each with the documents its `[N]` markers cite (resolved through
  the References block), cut at the dataset's `report_chars` (2000) as in the tracks.
- Nuggets: the dataset's nuggets as v3 banks. NeuCLIR uses the NIST QC'd bank with importance (vital counts 2, okay 1),
  the AND/OR aggregation and each answer's supporting documents; only its 22 requests with supporting documents can be
  scored. RAGTIME has no importance labels, so the weighted scores equal the plain ones.
- Scores (mean over requests, an empty report scores 0): `nugget_coverage` (`_weighted`), `sentence_support`,
  `citation_support`, `citation_relevance`, `f1` (`_weighted`). A nugget only counts when its sentence cites one of
  its supporting documents.
- Cache: `report_eval/` keeps the nugget banks, the cited documents and each report's judgments keyed by a hash of
  what was judged; an unchanged run is rescored without LLM calls. Cited document texts come from the corpus through a
  byte-offset index built once next to it (`{corpus}.jsonl.offsets.npz`).

Install (pinned; the provider integrations are not needed): `pip install langchain<1.0 langchain-core langsmith
"rag_run_validator @ git+https://github.com/hltcoe/rag-run-validator.git@v0.1"` and
`pip install --no-deps "auto-argue @ git+https://github.com/hltcoe/auto-argue.git@81dda60"`.

### Evaluation code

`src/evaluation/` follows the groups of `summary.json`; each group also writes its per-query files.

```
src/evaluation/
├── runner.py        save_query_outputs, build_evaluators, load_run_results, evaluate_and_save
├── answer/          AccuracyEvaluator (LLM judge), NumericMatchEvaluator (TRQA), ArgueReportEvaluator (reports);
│                    summary group generation.correctness / generation.nuggets
├── retrieval/       surfaced / seen / cited evaluators, citations, TREC metrics, fusion evaluation
├── trajectory/      TrajectoryEvaluator (statistics), save_trajectory
├── generation/      GenerationEvaluator (length, words, citations, generation/{qid}.md)
├── belief/          save_belief (belief/{qid}.jsonl schema)
├── gold/            TRQA gold entities and the entity matcher (criteria reachability)
├── judge.py         LLM-judge client and DEFAULT_JUDGE_MODEL
└── common.py        file, statistics and terminal helpers
```

### Output format

```
$DRA_OUTPUT_ROOT/{dataset}_{split}[_{query_key}]_{retriever}/{agent}_{backend}_{model}/belief-{mode}/
├── run_config.json                 every resolved setting: dataset, agent (temperature included), searcher,
│                                   belief model, evaluation
├── retrieval/
│   ├── surfaced/{qid}.trec         raw retriever output (all iterations, col 6 = iter_N)
│   ├── seen/{qid}.trec             docs shown to the LLM
│   ├── cited/{qid}.trec            docs cited by the LLM
│   └── fusion_{method}.trec        deduped fusion ranking (aggregate over all queries)
├── generation/{qid}.md             the final generation (with its References section)
├── belief/{qid}.jsonl              belief on only: meta line (criteria, final belief) + one line per search iteration
├── trajectory/{qid}.jsonl          one line per step (queries, seen doc ids, <belief_state> tags, per-turn
│   │                               diagnostics), then the meta line (counts, agent data)
│   └── {qid}.md                    the same trajectory, human-readable, written live
├── accuracy.jsonl                  per-query verdicts (datasets with answers)
├── report_eval/                    Auto-ARGUE inputs (nuggets/, cited docs), cached judgments/, per-query scores.tsv
└── summary.json                    grouped metrics (retrieval / generation / trajectory)
```

`summary.json`:

```
{
  "num_queries": N,
  "retrieval":  {"seen": {...}, "cited": {...}, "fusion": {...}},
  "generation": {"correctness": {...}  (datasets with answers) | "nuggets": {...}  (Auto-ARGUE),
                 "stats": {avg_generation_length, avg_generation_words, avg_citations}},
  "trajectory": {...}
}
```

`_{query_key}` is in the name only when it is not the dataset's default. Each fact has one file: the run settings are
in `run_config.json` only, the final generation in `generation/{qid}.md` only, the aggregates in `summary.json` only.

A query is finished when its `trajectory/{qid}.jsonl` ends with the meta line, which is written after every other file
of the query (and, with the belief model on, its belief file exists); a resumed run skips finished queries and runs
the others again. A resume whose `dataset`, `agent`, `searcher` or `belief_model` settings differ from the run's
`run_config.json` is refused, so every query of a directory ran with the same settings.

#### `belief/{query_id}.jsonl`

Every line has `record` (`meta` or `step`) and `query_id`; join correctness from `accuracy.jsonl` on `query_id`. A
signal that could not be computed is null, never 0; floats are finite and rounded to 4 decimals. The file is written
atomically, and a resumed run redoes a query whose file is missing.

Meta line (one per query):

| field | meaning |
|---|---|
| `schema_version` | 16 |
| `question` | the query text |
| `num_criteria`, `num_iterations`, `num_unique_docs` | per-query counts |
| `num_relevant` | relevant docs of the query in the qrels (null without qrels) |
| `criteria` | `[{id, text, kind}]` |
| `criteria_info` | criteria LLM `model`, `query_shape`, `errors`; `bank: "hit"` when read from the criteria bank |
| `final_criteria_attempts` | last attempts, one per criterion |
| `final_belief`, `criteria_evidence` | last belief state (one belief status per criterion); per criterion, its `status` and attached evidence `[{doc_id, step, role, spans, span_verified}]` (`span_verified`: one bool per span) (`role`: `support` or `contradict`) |

Step line (one per search iteration, flat scalars first):

| field | meaning |
|---|---|
| `iteration` | 1, 2, ... for every agent |
| `agent_iteration` | the agent's own counter (base and meaning differ per agent) |
| `num_subqueries`, `num_docs`, `num_new_docs` | step counts (unique doc ids) |
| `doc_novelty`, `criteria_delta`, `query_novelty` | x_t, see the table above |
| `new_item_precision` | newly seen relevant docs / the step's docs (0, 0.2, ..., 1 for 5 docs) |
| `num_new_relevant`, `num_repeated_relevant`, `num_irrelevant` | qrels counts |
| `recall_so_far`, `num_relevant_seen`, `num_relevant` | relevant docs seen up to and including this step / all the query's relevant docs |
| `intermediate_answers` | list of answers; `[]` for "no candidate", null when off or failed |
| `queries[]`, `docs[]` | per-query (`text`) and per-doc novelty detail (`queries[].target_scores` has one score per criterion) |
| `belief_before`, `belief_after` | the belief state: one belief status per criterion, in `criteria` order |
| `criteria_sources_after` | distinct supporting documents per criterion so far |
| `criteria_targeted`, `criteria_attempts_after` | ids targeted directly this iteration (null when the scorer failed); attempts per criterion, in `criteria` order |
| `criteria_updates` | `[{id, from, to, written, support, contradict, reason, note}]`, one per criterion (then one per unknown or repeated id); `written` is the status as the updater wrote it, `note` says why a status was kept or a citation dropped |
| `belief_updater_output` | the belief updater's raw reply (null when it was not called) |
| `intermediate_answer_reasoning`, `errors` | free text, reasons for nulls |

The run settings (agent, mode, models, criteria cap, encoder) are in `run_config.json`; the iteration's
`<belief_state>` tag (inform mode) is on its trajectory step. `agent_iteration` is the iteration of the step's searches
in `trajectory/{qid}.jsonl` (`iter`, before the dot). `iteration` is not the retrieval `iter_N`: `iter_N` counts retrieval calls, so an iteration with several queries spans
several `iter_N`. Join on `docs[].doc_id` when a doc-level link to `retrieval/*.trec` is needed.

### Criteria reachability

`analysis/criteria_reachability.py` assesses the init component: is the gold reachable through a request's criteria
list? It is the only place the criteria are judged against gold; runs only read and fill the bank. Criteria come from
the criteria bank (extracted and added when missing), with `llm_criteria` and `max_criteria` read from the run config
(`--config`, default `experiments/configs/dra_inference.yaml`); a request whose extraction failed
is left out. The dataset's `reachability` (`layout.DATASET_SPECS`) picks the gold:

- `documents` (NeuCLIR, RAGTIME): an LLM judge (`--judge-model`, default `llm_criteria`, prompts
  `analysis/prompts/criteria_reachability_{system,user}.txt`) reads each whole gold document of `--grades` (default:
  the top grade) with all criteria, in one call per document. It sees the request's topic title
  (`DatasetSpec.title_key`: NeuCLIR `topic_title`, RAGTIME `title`) instead of the request, since the criteria already
  carry the request's requirements and limits. It labels every criterion: `support` (specific facts, backed by a
  verbatim span; lowered to `related` when the span is not in the document), `related` (on the topic, no specific
  facts) or `none` (not about it); only `support` counts as reaching. Reachability is the share of gold documents that
  at least one criterion supports (headline); the share at `related` or above is secondary; both micro, macro and per
  grade. A reply that leaves a criterion without a label is asked again.
- `entities` (TRQA): the gold is the entity set of each query (`queries_{split}_intermediate_info.jsonl`). A gold
  entity is reachable when a member criterion names it: normalized string match, then one LLM call per request for
  the entities and member criteria left (aliases, spellings, transliterations; `src/evaluation/gold/entity_match.py`).
  The set and rest criteria are left out, since they cover every member by construction. Reachability is the share of
  reachable gold entities (headline, micro and macro); `member_precision`, the share of member criteria that name a
  gold entity, is secondary.
- None (BrowseComp-Plus): the criteria are the clues the question states, reachable by construction, so the run only
  fills the criteria bank (no judge call, no output).

Requests are those a run uses; `--limit N` does the first N not done yet, so repeated runs go through the split N at a
time. Output in the run outputs, in `OUTPUT_ROOT/criteria_reachability/{dataset}_{split}/` (`--output` overrides it):
`judgments.jsonl` (the judgment cache, shared by every `--grades` and `--tag`) and, per run folder, `summary.json`
(`settings`, `counts`, `metrics`, `stats`) with `per_doc.jsonl` in `grades-{g}_{judge}[_{tag}]/` (per request and
document: grade, document length, reach, and every criterion's label, reason and span) or `per_query.jsonl` in
`entities_{judge}[_{tag}]/` (per request: every gold entity with the member criterion that names it, and the member
criteria that name none).

```bash
python analysis/criteria_reachability.py --dataset ragtime --grades 2 3
python analysis/criteria_reachability.py --dataset trqa --subset wiki2 --dataset-year test
python analysis/criteria_reachability.py --dataset browsecomp_plus   # criteria bank only
```

## 4. Training

Belief model training with the agent frozen (Agent-BRACE adapted to retrieval), one phase per call of
`experiments/training/belief_model_training.py`.
The trained belief model makes both calls, criteria init and belief update; the query scorer stays frozen.
Phase 1 runs dra_inference on one split of a training pool (`experiments/training/build_training_pool.py`: `deep_qa` or `trqa_validation`, split into `sft` / `rl` / `val`): `sft` and `val` in monitor mode with the teacher, `rl` in off mode; phases 2 and 3 read those runs.
