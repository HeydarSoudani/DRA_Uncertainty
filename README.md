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
`browsecomp_plus`.

**Retrievers:** `bm25`, `spladepp`, `spladev3` (sparse); `bge`, `e5`, `dpr`,
`contriever`, `reasonir`, `qwen3_emb_{0.6b,4b,8b}` (dense).

## Installation

```bash
pip install -e .
```

This registers all packages so imports resolve from any working directory.
Set `DRA_DATA_ROOT` (corpus + indices) and `DRA_OUTPUT_ROOT` (run outputs) to
control where data is read/written.

## 1. Download datasets

```bash
# trqa  (subset: wiki1 | wiki2 | ecommerce)
python src/indexing_corpus_dataset/download_datasets.py trqa --subset wiki1

# neuclir  (subset: news | technical)
python src/indexing_corpus_dataset/download_datasets.py neuclir --year 2023 --subset news

# browsecomp_plus  (corpus only)
python src/indexing_corpus_dataset/download_datasets.py browsecomp_plus --skip-queries-qrels
```

Canonical layout written under `$DRA_DATA_ROOT/<dataset>/`:

```
queries/queries_{split}.jsonl     {"id","text","answer"?}
qrels/qrels_{split}.txt           TREC: qid 0 docid rel
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

Run defaults (top_k, rerankers, criteria LLM, eval k-values, …) are in
`experiments/configs/dra_inference.yaml`.

### Uncertainty estimator

`--uncertainty-estimator-mode monitor` plugs a passive monitor (`src/uncertainty_estimator/`) into any agent; `off` (the
default) disables it. It never changes the trajectory. At the start of each sample it extracts a fixed list of
criteria from the query (`llm_criteria`); at the end of each search iteration it computes the per-step signals of
the report (`papers/ACL_2027__Uncertainty_Quantification_for_DRAs/report`, Section "Instantiation"):

| field | report | meaning |
|---|---|---|
| `doc_novelty` | ν^D | novelty of the iteration's docs vs earlier docs: 0 for a seen id, else 1 − max cosine similarity |
| `criteria_delta` | Δ^D | change of the criteria state (uncovered 0, partially covered 1, fully covered 2) caused by the novel docs |
| `query_novelty` | ν^q | novelty of the iteration's queries vs earlier queries: 1 − max cosine similarity |
| `criteria_targeting` | τ^q | how strongly the queries target criteria still uncovered or partially covered |

Extra saved information: `marginal_recall` and `new_relevant_frac` (against qrels) and per-step intermediate answers (BrowseComp-Plus and
TRQA, all agents except `cpm_report`): after each iteration the agent's own model is asked, from its trajectory so
far, for its most likely answer(s) in its answer format; it may give one, several or none (`IntermediateAnswerSignal`,
prompt in `src/uncertainty_estimator/prompts/`). They are saved as is, not evaluated. Embeddings come from the retriever's encoder, so with a sparse retriever
`query_novelty` is null and `doc_novelty` uses ids only. The judge behind `criteria_delta` and
`criteria_targeting` is set by `criteria_judge` (`src/uncertainty_estimator/judges.py`):

- `nli`: an NLI cross-encoder (`criteria_judge_model`, default DeBERTa-v3-large NLI) scores all k × t (passage,
  criterion) pairs of an iteration in batches; the entailment probability gives fully (≥ 0.9) or partially (≥ 0.5)
  covered. The probabilities are saved, so the thresholds can be re-tuned offline. Query targeting is the cosine similarity of
  query and criterion embeddings (dense retrievers only; else `criteria_targeting` is null).
- `llm`: one LLM call per iteration labels every (passage, criterion) pair of the novel passages (with an evidence
  quote); the judge does not see the criteria state. One more call per iteration scores every (query, criterion)
  pair as 0, 0.5 or 1. `criteria_judge_model` defaults to `llm_criteria`.
- `none`: both signals are null.

A criterion's status is the highest coverage any doc gave it so far, so `criteria_delta` is never negative.

#### Inform mode

`--uncertainty-estimator-mode inform` computes the same signals and also shows them to the agent: right after each
iteration's search results, one `<certainty>` tag is appended to the trajectory (`src/uncertainty_estimator/certainty.py`):

```xml
<certainty step="3">
  <criteria covered="1" partial="1" not_covered="1">
    <k1 status="covered">born in the 1960s</k1>
    <k2 status="partial">won a regional award</k2>
    <k3 status="not_covered">studied in Lisbon</k3>
  </criteria>
  <retrieval_signals doc_novelty="0.42" criteria_delta="+1"/>
  <reasoning_signals query_novelty="0.81" criteria_targeting="0.60"/>
</certainty>
```

Only the criteria state and the four signals above are shown; gold-based fields (`marginal_recall`,
`new_relevant_frac`, relevant counts) stay in `uncertainty/{qid}.jsonl` for analysis. A null signal is left out.
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
does not read the tag.

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
└── summary.json                    grouped metrics (answer / retrieval / trajectory / generation / uncertainty)
```

With the estimator on, `{uncertainty_config}` is `ue-{mode}_{criteria_judge}` (e.g. `ue-monitor_nli`, `ue-inform_nli`).

#### `uncertainty/{query_id}.jsonl`

Every line has `record` (`meta` or `step`) and `query_id`; join correctness from `accuracy.jsonl` on `query_id`. A
signal that could not be computed is null, never 0; floats are finite and rounded to 4 decimals. The file is written
atomically, and a resumed run redoes a query whose file is missing.

Meta line (one per query):

| field | meaning |
|---|---|
| `schema_version` | 2 |
| `question` | the query text |
| `agent`, `llm_model`, `dataset`, `llm_criteria`, `max_criteria` | run settings |
| `criteria_source`, `criteria_judge`, `query_scorer`, `encoder` | components in use (null when off) |
| `num_criteria`, `num_iterations`, `num_unique_docs` | per-query counts |
| `num_relevant` | relevant docs of the query in the qrels (null without qrels) |
| `criteria` | `[{id, text}]` |
| `criteria_info` | criteria LLM `model`, `reasoning`, `errors` |
| `final_criteria_state`, `criteria_evidence` | last criteria state; per criterion, the ids of the covering docs |

Step line (one per search iteration, flat scalars first):

| field | meaning |
|---|---|
| `iteration` | 1, 2, ... for every agent |
| `agent_iteration` | the agent's own counter (base and meaning differ per agent) |
| `num_subqueries`, `num_docs`, `num_new_docs` | step counts (unique doc ids) |
| `doc_novelty`, `criteria_delta`, `query_novelty`, `criteria_targeting` | x_t, see the table above |
| `marginal_recall` | newly seen relevant docs / all relevant docs of the query |
| `new_relevant_frac` | newly seen relevant docs / the step's docs |
| `num_new_relevant`, `num_repeated_relevant`, `num_irrelevant` | qrels counts |
| `intermediate_answers` | list of answers; `[]` for "no candidate", null when off or failed |
| `intermediate_answer_confidence` | stated confidence in [0, 1] |
| `subqueries`, `queries[]`, `docs[]` | per-query and per-doc novelty detail (`queries[].target_scores` has one score per criterion) |
| `criteria_state_before`, `criteria_state_after` | one status per criterion, in `criteria` order |
| `criteria_judgments` | `[{doc_id, statuses, scores or evidence}]` for the novel docs |
| `intermediate_answer_reasoning`, `errors` | free text, reasons for nulls |

`iteration` is not the retrieval `iter_N`: `iter_N` counts retrieval calls, so an iteration with several queries spans
several `iter_N`. Join on `docs[].doc_id` when a doc-level link to `retrieval/*.trec` is needed.

## 4. Training

> 🚧 Under construction.
