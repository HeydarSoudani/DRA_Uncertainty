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

Run via the wrapper script (edit `DATASET` / `RETRIEVER` / `AGENT` /
`UNCERTAINTY_ESTIMATOR` at the top):

```bash
sbatch scripts/run_dra_inference.sh
```

Underlying command:

```bash
python experiments/dra_inference.py \
    --dataset browsecomp_plus \
    --retriever qwen3_emb_4b \
    --agentic-model glm \
    --uncertainty-estimator-mode monitor \
    --num-gpus 0
```

Quick checks:

```bash
# single-query smoke test
python experiments/dra_inference.py --dataset browsecomp_plus --limit 1 --num-gpus 0
# evaluate already-saved runs, no generation
python experiments/dra_inference.py --dataset browsecomp_plus --eval-only --num-gpus 0
```

A run is evaluated from its saved files, both at its end and with `--eval-only`, so the two write the same
`summary.json`. The judge verdicts are kept in the run directory (`accuracy.jsonl`, `report_eval/`,
the `criteria_eval` of the uncertainty meta lines and `criteria_eval_judgments.jsonl`) and
reused for queries whose input is unchanged, so `--eval-only` on an unchanged run makes no LLM call and needs no GPU. The terminal log prints one
EVALUATION SUMMARY block in the order of `summary.json`: retrieval (seen docs), generation (correctness or nuggets, and
length), criteria, trajectory, then the time of each evaluation stage. Cited-doc and fusion metrics and the full @k
tables are only in `summary.json`. `_eval_results_cache*.pkl.gz` files left by older code are no longer
read and can be deleted.

Run defaults (top_k, rerankers, criteria LLM, eval k-values, …) are in
`experiments/configs/dra_inference.yaml`.

### Uncertainty estimator

`--uncertainty-estimator-mode monitor` plugs a passive monitor (`src/uncertainty_estimator/`) into any agent; `off` (the
default) disables it. It never changes the trajectory. At the start of each sample it derives a fixed list of
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
is open, its members closed and its "any other member" criterion rest (added by code when the model leaves it out;
fully covered only while the set is), all report criteria are aspects. Entity recall and precision count the member criteria only. The coverage judge's prompt defines only the kinds present in the list.
Criteria bank (`criteria_bank: true`, the default): a query's criteria are read from
`CRITERIA_ROOT/{dataset}/criteria_bank/criteria_{split}.jsonl` and extracted only when missing, then added to it
(`BankedCriteriaSource`). A line is used only when the query id and text, `llm_criteria`, the query shape,
`max_criteria` and a hash of the init prompt all match, so a changed prompt or cap extracts again; a failed extraction
and a salvaged one (the complete items of a truncated reply) are never added. Runs and `analysis/criteria_reachability.py` share the bank, so the analysis scores the criteria the
runs use. At the end of each search iteration the estimator computes the per-step signals of
the report (`papers/ACL_2027__Uncertainty_Quantification_for_DRAs/report`, Section "Instantiation"):

| field | report | meaning |
|---|---|---|
| `doc_novelty` | ν^D | fraction of the iteration's docs whose id was not seen in an earlier iteration (0, 0.2, ..., 1 for 5 docs) |
| `criteria_delta` | Δ^D | change of the criteria state (uncovered 0, partially covered 1, fully covered 2) caused by the novel docs; negative when a contradiction lowered it |
| `query_novelty` | ν^q | novelty of the iteration's queries vs earlier queries: 1 − max cosine similarity |
| `criteria_attempts_after` | a | per criterion, the number of iterations whose queries directly targeted it (a state, like the criteria state) |

Extra saved information: `new_item_precision` and `new_item_graded_recall` (against qrels) and per-step intermediate answers (BrowseComp-Plus and
TRQA, all agents except `cpm_report`): after each iteration the agent's own model is asked, from its trajectory so
far, for its most likely answer(s) in its answer format; it may give one, several or none (`IntermediateAnswerSignal`,
prompt in `src/uncertainty_estimator/prompts/`). They are saved as is, not evaluated. Embeddings come from the retriever's encoder, so with a sparse retriever
`query_novelty` is null. The criteria signals use LLM judges (`src/uncertainty_estimator/judges.py`,
model `criteria_judge_model`, default `llm_criteria`):

- `criteria_delta`: a stateful coverage judge. One call per iteration sees each criterion with its current status,
  its attached evidence (the last 4 passages, each shown as the spans the judge cited that occur verbatim in the
  passage and the first 100 words of the passage; a passage attached to several criteria is shown under each) and,
  when partially covered, what it is still `missing`; plus the iteration's novel passages. It does not see the search
  queries. It lists the criteria a new passage supports or contradicts, alone or combined with the attached evidence
  (partially covered criteria and criteria that refer to each other are re-checked every iteration), with one or
  more single-sentence verbatim spans per cited passage. Each span is checked on its own (a span joined with "..." is
  split first); a span that is not verbatim is not shown, but the citation still counts (`span_verified` is saved per
  span). `CriteriaState.apply` enforces the rules: a raise needs a
  supporting passage, a lowering needs a contradicting passage and moves one level at most per iteration, and an
  update that keeps the status only attaches its evidence. An open criterion is fully covered only once its support
  comes from 3 distinct documents (`OPEN_MIN_SOURCES`); before that it is capped at partially covered, with the
  missing sources as its `missing`. Several partial passages can together make a criterion
  fully covered, and a contradiction lowers it, so `criteria_delta` is negative when coverage was lost.
- `criteria_attempts_after`: one more call per iteration scores every (query, criterion) pair as 0, 0.5 or 1. It sees
  each criterion's status, the verified spans of its latest evidence passage and what is `missing`, so a query naming
  an entity the evidence already linked to a criterion counts as targeting it; a query that restates most of the user
  query scores at most 0.5. One query may target several criteria. A criterion that some query of the iteration
  scores 1 on is `criteria_targeted` and its attempt count grows by 1 (once per iteration, however many queries target
  it; 0.5 never counts; covered criteria are counted too). When the call fails the attempts are unchanged.

#### Inform mode

`--uncertainty-estimator-mode inform` computes the same signals and also shows them to the agent: right after each
iteration's search results, one `<certainty>` tag is appended to the trajectory (`src/uncertainty_estimator/certainty.py`):

```xml
<certainty step="3">
  <criteria covered="1" partial="1" not_covered="1">
    <k1 status="covered" attempts="1">born in the 1960s</k1>
    <k2 status="partial" attempts="2">won a regional award</k2>
    <k3 status="not_covered" kind="open" attempts="0">the films they directed</k3>
  </criteria>
  <retrieval_signals doc_novelty="0.40" criteria_delta="+1"/>
  <reasoning_signals query_novelty="0.81"/>
</certainty>
```

`kind="open"` marks an open criterion; closed criteria carry no kind. Only the criteria state with the attempts and the other three signals above are shown; gold-based fields (`new_item_precision`,
`new_item_graded_recall`, relevant counts) stay in `uncertainty/{qid}.jsonl` for analysis. A null signal is left out.
The system prompts are unchanged. Where the tag goes:

| agents | place |
|---|---|
| searchr1, research, stepsearch, searcho1, react | appended to the prompt after the search-result block |
| selfask | between `Follow up: …` and the `Intermediate answer:` prefill (the initial retrieval's tag after the prompt) |
| glm, oss, tongyi, drtulu, cpm_explore | appended to the iteration's last tool / result message |
| webweaver | appended to the planner's search observation |
| cpm_report | appended to the current information of the next plan / write prompt |

The tag is also saved as `certainty` on the search step in `trajectory/{qid}.jsonl` and as `certainty_tag` in
`uncertainty/{qid}.jsonl`. Tags the model writes itself are removed in the prompt-string agents. `uncertainty_aware`
follows the flag like every other agent; in `inform` mode its system prompt also explains the tag, and in `monitor` and
`off` modes the prompt never mentions it.

#### Criteria evaluation

In a run with the estimator on (`monitor` or `inform`) on a dataset with `criteria_gold`, each query's criteria are
scored against it with the judge `judge_model`, in a background thread while the agent runs, so the judge calls do
not delay the agent. The score is printed when the query ends
(`[Agent] criteria eval (nuggets, 9 criteria, 15 gold): nugget coverage ... `) and saved as `criteria_eval` in the meta
line of `uncertainty/{qid}.jsonl`; it never reaches the agent. The run's evaluation reuses it (or recomputes it when
the criteria or the judge changed, writing the new score back into the meta line) and writes the `criteria` group
of `summary.json`, laid out as the report's: `method`, `judge_model`, the counts (`num_evaluated`,
`num_empty_criteria`, `num_judge_failures`), `metrics` and `stats` (list sizes and diagnostics).
The nugget judgments made at evaluation are cached one by one (`criteria_eval_judgments.jsonl`, keyed by judge,
prompt, criterion, question and answer), so a changed criteria list costs only its new criteria.

`python -m evaluation.criteria` scores the criteria list without an agent run (`src/evaluation/criteria/`, gold and
matchers in `src/evaluation/gold/`). It derives the criteria of a split with `LLMCriteriaSource` (or reads them from a
run with `--run-dir`) and compares them with the dataset's `criteria_gold` (`layout.DATASET_SPECS`):

- `entities` (TRQA): closed criteria are matched to the query's gold entities by name (normalized string match, then an
  LLM for aliases); `recall` over the entities, `precision` over the closed criteria.
- `nuggets` (NeuCLIR, RAGTIME): scored the way Auto-ARGUE scores the reports, so the two evaluations compare nugget
  by nugget. Both use the same nuggets and requests (the Auto-ARGUE nugget banks: questions grouped, answers without
  documents and questions without answers dropped), the same judge and settings (Qwen3-32B, temperature 0, reasoning
  off, YES/NO asked again twice, then NO), and the same covered rule. For every (criterion, nugget question, gold
  answer) the judge says whether the criterion asks for that question-answer pair (`src/evaluation/gold/nugget_ask.py`).
  A nugget is covered when the matched answers meet its aggregator (one for OR, all for AND).
  Metrics: `nugget_coverage`, `nugget_coverage_weighted` (vital 2, okay 1, as Auto-ARGUE); `avg_unmatched_criteria`
  (criteria that ask for no nugget answer) is a diagnostic in `stats`. The request check scores the criteria
  against the request itself: an LLM lists the request's constraints (requirement, limit, exclusion, exception,
  background) and the criteria that carry each, then the criteria that ask for what the request leaves out
  (`constraint_recall`, `exclusion_recall`, `violations_per_query`). In a run's evaluation, each nugget's criteria
  label is crossed with Auto-ARGUE's answered label: `answered_if_asked` / `answered_if_not_asked` in
  `criteria.metrics` are the report's coverage of the nuggets the criteria asked for / did not ask for.
- No gold (BrowseComp-Plus): extraction only, or scored against a reference criteria list (`--reference`): an LLM
  lists each reference unit's qualifiers and scores every (unit, criterion) pair 1, 0.5 or 0; `recall_strict` /
  `recall_lenient`, `precision_strict` / `precision_lenient`.

```bash
python -m evaluation.criteria --dataset trqa --subset wiki2 --sample 100
python -m evaluation.criteria --dataset neuclir --prompt-file other_prompt.txt --tag other
```

`--prompt-file` replaces the criteria-extraction prompt, for comparisons. Outputs (`criteria.jsonl`, reused on a rerun;
`eval.jsonl`; `summary.json`; `criteria_eval_judgments.jsonl`, the cached nugget judgments) go to `{DRA_OUTPUT_ROOT}/criteria_eval/{dataset}_{split}/{tag}/`.

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
├── uncertainty/     save_uncertainty (uncertainty/{qid}.jsonl schema)
├── criteria/, gold/ CriteriaEvaluator (in a run and offline: python -m evaluation.criteria), gold units, matchers
├── judge.py         LLM-judge client and DEFAULT_JUDGE_MODEL
└── common.py        file, statistics and terminal helpers
```

### Output format

```
$DRA_OUTPUT_ROOT/{dataset}_{split}_{query_key}_{retriever}/{agent}_{backend}_{model}/{uncertainty_config}/
├── run_config.json                 full agent/searcher/uncertainty-estimator settings
├── retrieval/
│   ├── surfaced/{qid}.trec         raw retriever output (all iterations, col 6 = iter_N)
│   ├── seen/{qid}.trec             docs shown to the LLM
│   ├── cited/{qid}.trec            docs cited by the LLM
│   └── fusion_{method}.trec        deduped fusion ranking (aggregate over all queries)
├── generation/{qid}.md             per-query report (markdown)
├── trajectory/{qid}.jsonl          meta line + one line per step
├── uncertainty/{qid}.jsonl         meta line (criteria, config) + one line per search iteration (signals)
├── accuracy.jsonl                  per-query answer correctness (datasets with answers)
├── report_eval/                    Auto-ARGUE inputs (nuggets/, cited docs), cached judgments/, per-query scores.tsv
├── criteria_eval_judgments.jsonl   cached YES/NO nugget judgments of the criteria eval (report datasets)
└── summary.json                    grouped metrics (retrieval / generation / criteria / trajectory)
```

`summary.json`:

```
{
  "num_queries": N,
  "retrieval":  {"seen": {...}, "cited": {...}, "fusion": {...}},
  "generation": {"correctness": {...}  (datasets with answers) | "nuggets": {...}  (Auto-ARGUE),
                 "stats": {avg_generation_length, avg_generation_words, avg_citations}},
  "criteria":   {method, judge_model, num_evaluated, num_empty_criteria, num_judge_failures,
                 "metrics": {coverage (or recall / precision) scores, request check, answered_if_asked (report datasets)},
                 "stats": {avg_num_criteria, avg_num_gold, ...}},
  "trajectory": {...}
}
```

With the estimator on, `{uncertainty_config}` is `ue-{mode}` (e.g. `ue-monitor`, `ue-inform`).

#### `uncertainty/{query_id}.jsonl`

Every line has `record` (`meta` or `step`) and `query_id`; join correctness from `accuracy.jsonl` on `query_id`. A
signal that could not be computed is null, never 0; floats are finite and rounded to 4 decimals. The file is written
atomically, and a resumed run redoes a query whose file is missing.

Meta line (one per query):

| field | meaning |
|---|---|
| `schema_version` | 6 |
| `question` | the query text |
| `agent`, `llm_model`, `dataset`, `llm_criteria`, `max_criteria` | run settings |
| `criteria_source`, `criteria_judge`, `query_scorer`, `encoder` | components in use (null when off) |
| `num_criteria`, `num_iterations`, `num_unique_docs` | per-query counts |
| `num_relevant` | relevant docs of the query in the qrels (null without qrels) |
| `total_gain` | summed official gain of the query's relevant docs (null without graded qrels) |
| `criteria` | `[{id, text, kind}]` |
| `criteria_info` | criteria LLM `model`, `query_shape`, `errors` |
| `final_criteria_attempts` | last attempts, one per criterion |
| `criteria_eval` | the criteria scored against the gold during the run (`CriteriaEvaluator` record; null when not scored) |
| `final_criteria_state`, `criteria_evidence` | last criteria state; per criterion, its status, what it is still `missing` (partially covered only) and attached evidence `[{doc_id, step, role, spans, span_verified}]` (`span_verified`: one bool per span) (`role`: `support` or `contradict`) |

Step line (one per search iteration, flat scalars first):

| field | meaning |
|---|---|
| `iteration` | 1, 2, ... for every agent |
| `agent_iteration` | the agent's own counter (base and meaning differ per agent) |
| `num_subqueries`, `num_docs`, `num_new_docs` | step counts (unique doc ids) |
| `doc_novelty`, `criteria_delta`, `query_novelty` | x_t, see the table above |
| `new_item_precision` | newly seen relevant docs / the step's docs (0, 0.2, ..., 1 for 5 docs) |
| `num_new_relevant`, `num_repeated_relevant`, `num_irrelevant` | qrels counts |
| `new_item_graded_recall` | `new_gain` / `total_gain`; sums over the steps to GradedRecall@N |
| `new_gain`, `total_gain` | summed official gain of the relevant docs first seen this step, and of all the query's relevant docs |
| `intermediate_answers` | list of answers; `[]` for "no candidate", null when off or failed |
| `subqueries`, `queries[]`, `docs[]` | per-query and per-doc novelty detail (`queries[].target_scores` has one score per criterion) |
| `criteria_state_before`, `criteria_state_after` | one status per criterion, in `criteria` order |
| `criteria_targeted`, `criteria_attempts_after` | ids targeted directly this iteration (null when the scorer failed); attempts per criterion, in `criteria` order |
| `criteria_updates` | `[{id, from, to, proposed, support, contradict, reason, missing, applied, note}]`, one per update the coverage judge proposed; `applied` is false for a rejected one (`note` says why) |
| `criteria_judge_output` | the coverage judge's raw reply (null when it was not called) |
| `intermediate_answer_reasoning`, `errors` | free text, reasons for nulls |

`iteration` is not the retrieval `iter_N`: `iter_N` counts retrieval calls, so an iteration with several queries spans
several `iter_N`. Join on `docs[].doc_id` when a doc-level link to `retrieval/*.trec` is needed.

### Criteria reachability

`analysis/criteria_reachability.py --dataset {neuclir,ragtime}` asks whether the gold documents are reachable through
a request's criteria list. Criteria come from the criteria bank (extracted and added when missing), with
`llm_criteria` and `max_criteria` read from the run config (`--config`, default
`experiments/configs/dra_inference.yaml`); a request whose extraction failed or was only salvaged is left out. An LLM
judge (`--judge-model`, default `llm_criteria`, prompts `analysis/prompts/criteria_reachability_{system,user}.txt`)
reads each whole gold document of `--grades` (default: the top grade) with all criteria, in one call per document. It
sees the request's topic title (`DatasetSpec.title_key`: NeuCLIR `topic_title`, RAGTIME `title`) instead of the
request, since the criteria already carry the request's requirements and limits. It labels every criterion: `support` (specific facts, backed by a verbatim span; lowered to `related` when the span is not in
the document), `related` (on the topic, no specific facts) or `none` (not about it); only `support` counts as reaching.
Reachability is the share of gold documents that at least one criterion supports (headline); the share at `related` or
above is secondary; both micro, macro and per grade. Calls per request: one for the
criteria when missing from the bank, plus one per gold document; a reply that leaves a criterion without a label is
asked again. Output in the run outputs, in
`OUTPUT_ROOT/criteria_reachability/{dataset}_{split}/` (`--output` overrides it): `judgments.jsonl` (the judgment cache,
shared by every `--grades` and `--tag`) and, per `grades-{g}_{judge}[_{tag}]/`, `summary.json` (`settings`, `counts`,
`metrics`, `stats`) and `per_doc.jsonl` (per request and document: grade, document length, reach, and every
criterion's label, reason and span).

## 4. Training

> 🚧 Under construction.
