# Inference

`dra_inference.py` runs one deep research agent over one dataset split, optionally with the belief model next to it,
evaluates the run and saves every step. This README covers how to run it, what the belief model computes, how runs
are evaluated, and the output files. Datasets and indices must be built first
(see the [root README](../../README.md#data-and-indices)).

## Contents

- [Running](#running)
- [Belief model](#belief-model)
  - [Criteria](#criteria)
  - [Belief update](#belief-update)
  - [Per-step signals](#per-step-signals)
  - [Inform mode](#inform-mode)
- [Evaluation](#evaluation)
  - [Report evaluation (Auto-ARGUE)](#report-evaluation-auto-argue)
- [Output format](#output-format)

## Running

On Slurm, edit `DATASET` / `SUBSET` / `RETRIEVER` / `POLICY_MODEL` / `BELIEF_MODE` at the top of the wrapper and
submit it:

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

| Flag | Values |
|---|---|
| `--dataset`, `--subset` | `trqa` (`wiki1` / `wiki2` / `ecommerce`), `neuclir` (`news` / `technical`), `browsecomp_plus`, `ragtime`; unset subset = the dataset's default |
| `--retriever` | must have an index built for the dataset |
| `--policy-model` | the agent; its LLM is picked automatically (`utils/config.py`) |
| `--belief-mode` | `off`, `monitor` (default), `inform` |
| `--num-gpus` | query-level workers, one per GPU; `0` = auto-detect |
| `--limit N` | first N queries only |
| `--eval-only` | evaluate the saved run, no generation |

Every other setting (top_k, rerankers, criteria LLM, judges, eval k-values, ...) is in
`experiments/configs/dra_inference.yaml` and can be overridden with the matching `--flag`.

Quick checks:

```bash
# single-query smoke test
python experiments/inference/dra_inference.py --dataset browsecomp_plus --limit 1 --num-gpus 0
# evaluate an already-saved run, no generation
python experiments/inference/dra_inference.py --dataset browsecomp_plus --eval-only --num-gpus 0
```

A resumed run skips the queries already finished and runs the others again. A resume whose `dataset`, `agent`,
`searcher` or `belief_model` settings differ from the run's `run_config.json` is refused, so every query of a
directory ran with the same settings.

## Belief model

The belief model (`src/belief_model_component/`) can be plugged into any agent:

| `--belief-mode` | Effect |
|---|---|
| `off` | disabled |
| `monitor` | computes and saves the belief state and the per-step signals; never changes the trajectory |
| `inform` | as `monitor`, and also shows a `<belief_state>` tag to the agent after each search |

It follows Agent-BRACE (Singh et al., 2026, arXiv:2605.11436), adapted to search: the environment is the retriever,
the observation is the documents of a search iteration, and the belief state gives each criterion of the query one
of the seven statuses of Agent-BRACE's verbal belief scale. The per-step signals are those of the report
(`papers/ACL_2027__Uncertainty_Quantification_for_DRAs/report`, Section "Instantiation"). All criteria and belief
calls are LLM judges (`src/belief_model_component/judges.py`): the criteria extractor uses `llm_criteria`, the belief
updater `belief_judge_model` (default `llm_criteria`) and the query scorer `query_scorer_model` (default
`belief_judge_model`).

### Criteria

At the start of each sample, the belief model derives a fixed list of criteria from the query: the pieces of
information a complete response must establish. The dataset's `query_shape` (`layout.DATASET_SPECS`) picks the
prompt, a shared core plus one block per shape:

| Shape | Datasets | Criteria |
|---|---|---|
| single target | BrowseComp-Plus | the clues of the target, copied verbatim |
| set | TRQA | the set, then one "member: property" per known member, then "any other member: property" for members found only by the search |
| report | NeuCLIR, RAGTIME | every constraint the request states (no cap), then at most `max_criteria` (5) criteria the extractor adds, as one plain list |

`max_criteria` (null = the dataset's default in `layout.DATASET_SPECS`) caps the list; for a report request it is the
number of criteria the extractor may add, asked in the prompt, and code never cuts a report list.

Rules for report criteria:

- A criterion keeps the request's own limits in its wording. What the request leaves out is written into the
  criterion it narrows ("..., not ..."); an exception ("unless ...") is its own criterion.
- Background on the asker is kept only when it limits what information applies (country, location, situation).
- Every criterion reads on its own, naming its subject instead of pointing back ("it", "these").
- Criteria are distinct: none asks for what another one asks for.

Each criterion has a `kind`, set by code from the shape: `closed` (one fact), `open` (several parts or answers),
`rest` or `aspect` (one aspect of a report topic). All clues are closed; the set (always the first criterion) is open,
its members closed and its "any other member" criterion rest (added by code when the model leaves it out); all report
criteria are aspects.

**Criteria bank** (`criteria_bank: true`, the default). A query's criteria are read from
`CRITERIA_ROOT/{dataset}/criteria_bank/criteria_{split}.jsonl` and extracted only when missing, then added to it
(`BankedCriteriaSource`).

- A line is used only when the query id and text, `llm_criteria`, the query shape, `max_criteria` and a hash of the
  init prompt all match, so a changed prompt or cap extracts again.
- A truncated reply is asked once more and fails when truncated again. A failed extraction is never added, so every
  criteria list a run uses is banked.
- A query extracted by two processes at once keeps the line banked first.
- Runs and `analysis/criteria_reachability.py` share the bank, so the analysis assesses the criteria the runs use.
  The criteria are never scored against gold during a run.

### Belief update

The belief updater implements Agent-BRACE's b_t = f(query, b_{t-1}, o_t). b_0 gives every criterion the status
`unknown`. One call per iteration sees:

- each criterion with its current status and its attached evidence (the last 4 passages, each shown as the spans the
  updater cited that occur verbatim in it, plus its title line);
- the passages of the iteration it has not judged in an earlier iteration. With none, there is no call and the belief
  is unchanged; the passages of an iteration whose update failed are shown again when retrieved again.

It sees neither the search queries nor the signals. It lists only the criteria those passages support, contradict or
relate to (including passages that point against a criterion), with the passages that support or contradict each
(one or more verbatim spans each) and the new status, lowest first:

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
criterion the passages give, by one passage or several; the negative levels rate evidence against it. A search that
finds nothing on a criterion never lowers it.

Differences from Agent-BRACE: it ranks "unknown" lowest (nominal probability 0); here it is the neutral level 0, so a
contradiction lowers the level and new coverage raises it. "certain" is the paper's "confirmed / certain". The status
words are Agent-BRACE's; their evidence descriptions adapt them to retrieval and draw on GRADE's certainty factors
(directness, consistency, completeness). A shape block refines what support means for its queries.

The guidance lives in the prompt only:

- `BeliefState.update` takes every status as written, so a status may stay, rise or fall by any number of levels. A
  criterion the updater leaves out, or gives no valid status, keeps its status.
- Each span is checked on its own (a span joined with "..." is split first); only cited passages with a verified span
  are attached, once per criterion and role. This guards the evidence, never the status.
- `criteria_delta` and the analysis read each status as its signed level (`STATUS_LEVEL`, -2 to 4).

**Criteria attempts.** One more call per iteration scores every (query, criterion) pair as 0, 0.5 or 1. It sees each
criterion's status and the verified spans of its latest evidence passage, so a query naming an entity the evidence
already linked to a criterion counts as targeting it; a query that restates most of the user query scores at most
0.5. One query may target several criteria. A criterion that some query of the iteration scores 1 on is
`criteria_targeted`, and its attempt count grows by 1 (once per iteration, however many queries target it; 0.5 never
counts; whatever its status). When the call fails the attempts are unchanged.

### Per-step signals

At the end of each search iteration the belief model computes:

| field | report | meaning |
|---|---|---|
| `doc_novelty` | ν^D | fraction of the iteration's docs whose id was not seen in an earlier iteration (0, 0.2, ..., 1 for 5 docs) |
| `criteria_delta` | Δ^D | change of the belief state over the signed status levels, summed over the criteria; positive when the passages cover more of the criteria, negative when they turn against them |
| `query_novelty` | ν^q | novelty of the iteration's queries vs earlier queries: 1 − max cosine similarity |
| `criteria_attempts_after` | a | per criterion, the number of iterations whose queries directly targeted it (a state, like the belief state) |

Embeddings come from the retriever's encoder, so with a sparse retriever `query_novelty` is null.

Extra saved information:

- `new_item_precision` and `recall_so_far`, against the qrels at the dataset's `min_relevance_score`. NeuCLIR and
  RAGTIME use grade 3 (the very valuable docs), while the end-of-run `GradedRecall@N` counts every official level
  with its gain.
- Intermediate answers (BrowseComp-Plus and TRQA, all agents except `cpm_report`): after each iteration the agent's own
  model is asked, from its trajectory so far, for its most likely answer(s) in its answer format; it may give one,
  several or none (`IntermediateAnswerSignal`, prompt in `src/belief_model_component/prompts/`). They are saved as
  is, not evaluated.

### Inform mode

`--belief-mode inform` computes the same signals and also shows them to the agent: right after each iteration's
search results, one `<belief_state>` tag is appended to the trajectory (`src/belief_model_component/belief_state.py`):

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

The tag is descriptive only: each criterion's status and attempts, then the step's three signals. Gold-based fields
(`new_item_precision`, `recall_so_far`, relevant counts) stay in `belief/{qid}.jsonl` for analysis. A null signal is
left out. Where the tag goes:

| agents | place |
|---|---|
| searchr1, research, stepsearch, searcho1, react | appended to the prompt after the search-result block |
| selfask | between `Follow up: …` and the `Intermediate answer:` prefill (the initial retrieval's tag after the prompt) |
| glm, oss, tongyi, drtulu, cpm_explore | appended to the iteration's last tool / result message |
| webweaver | appended to the planner's search observation |
| cpm_report | appended to the current information of the next plan / write prompt |

The tag is saved once, as `belief_state` on the search step in `trajectory/{qid}.jsonl`. Tags the model writes itself
are removed in the prompt-string agents. The system prompts are unchanged, except for `belief_aware`: in `inform`
mode its system prompt introduces the tag (one section, `belief_section.txt`, the same for every task, with no advice
on how to use it); in `monitor` and `off` modes its prompt never mentions it.

## Evaluation

A run is evaluated from its saved files, both at its end and with `--eval-only`, so the two write the same
`summary.json`. The evaluators depend on the dataset:

| Dataset | Answer evaluation |
|---|---|
| BrowseComp-Plus | LLM judge (`--judge-model`), `accuracy.jsonl` |
| TRQA | numeric match, `accuracy.jsonl` |
| NeuCLIR, RAGTIME | Auto-ARGUE report scores, `report_eval/` |

Every dataset also gets retrieval (surfaced, seen and cited docs, fusion), generation statistics and trajectory
statistics. Judge verdicts are reused for queries whose input is unchanged, so `--eval-only` on an unchanged run makes
no LLM call and needs no GPU.

The terminal log prints one EVALUATION SUMMARY block in the order of `summary.json`: retrieval (seen docs), generation
(correctness or nuggets, and length), trajectory, then the time of each evaluation stage. Cited-doc and fusion
metrics and the full @k tables are only in `summary.json`. `_eval_results_cache*.pkl.gz` files left by older code are
no longer read and can be deleted.

`src/evaluation/` follows the groups of `summary.json`; each group also writes its per-query files:

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

### Report evaluation (Auto-ARGUE)

On the report datasets (`report_eval: "argue"` in `layout.DATASET_SPECS`: NeuCLIR, RAGTIME), every run and every
`--eval-only` pass scores the reports against the nuggets with the official
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

Install (pinned; the provider integrations are not needed):

```bash
pip install "langchain<1.0" langchain-core langsmith \
    "rag_run_validator @ git+https://github.com/hltcoe/rag-run-validator.git@v0.1"
pip install --no-deps "auto-argue @ git+https://github.com/hltcoe/auto-argue.git@81dda60"
```

## Output format

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

`_{query_key}` is in the name only when it is not the dataset's default. Each fact has one file: the run settings are
in `run_config.json` only, the final generation in `generation/{qid}.md` only, the aggregates in `summary.json` only.

A query is finished when its `trajectory/{qid}.jsonl` ends with the meta line, which is written after every other file
of the query (and, with the belief model on, after its belief file).
