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
| `monitor` | computes and saves the belief state; never changes the trajectory |
| `inform` | as `monitor`, and also shows a `<belief_state>` tag to the agent after each search |

It follows Agent-BRACE (Singh et al., 2026, arXiv:2605.11436), adapted to search: the environment is the retriever,
the observation of a search iteration is its documents and the agent's turn, and the belief state gives each
criterion of the query its nuggets: the answers found for it, each with one of the seven statuses of Agent-BRACE's
verbal belief scale and the sources it rests on. All criteria and belief calls are LLM calls: the criteria extractor
uses `llm_criteria`, the belief update (`src/belief_model_component/updater.py`) `belief_judge_model` (default
`llm_criteria`).

### Criteria

At the start of each sample, the belief model derives a fixed list of criteria from the query: the pieces of
information a complete response must establish. The dataset's `query_shape` (`layout.DATASET_SPECS`) picks the
prompt, a shared core plus one block per shape:

| Shape | Datasets | Criteria |
|---|---|---|
| single target | BrowseComp-Plus | every clue of the target, copied verbatim (no cap) |
| multi-hop | deep_qa | one criterion per entity on the chain, the answer last (FRAMES; at most 8) |
| set | TRQA | one "member: property" per known member, then "any other member: property" for members found only by the search, then the aggregate: how the answer follows from the members' values (the layout of the gold criteria; at most 52: 50 members, the rest and the aggregate) |
| report | NeuCLIR, RAGTIME | every requirement the long request (background and problem statement) states, as one plain list (no cap, nothing added) |

`max_criteria` (null = the dataset's default in `layout.DATASET_SPECS`) caps silver criteria only: TRQA and
trqa_validation (52) and deep_qa's FRAMES (8). Gold criteria are never capped: on BrowseComp-Plus, NeuCLIR and RAGTIME,
whose criteria are all gold, a `max_criteria` stops the run (`criteria_init.check_max_criteria`).

Rules for report criteria:

- A criterion keeps the request's own limits in its wording. What the request leaves out is written into the
  criterion it narrows ("..., not ..."); an exception ("unless ...") is its own criterion.
- Background on the asker is kept only when it limits what information applies (country, location, situation).
- Every criterion reads on its own, naming its subject instead of pointing back ("it", "these").
- Criteria are distinct: none asks for what another one asks for.

Each criterion has a `kind`, set by code from the shape: `closed` (one fact), `rest`, `aggregate` or `aspect` (one
aspect of a report topic). All clues and hops are closed; a set query's members are closed, its "any other member"
criterion rest and its last criterion aggregate (each added by code when the model leaves it out); all report
criteria are aspects. A closed or aggregate criterion has one answer, so its nuggets are alternatives; the others
have several answers that hold together (members, values, facts). No document states an aggregate (the agent computes
it), so its nuggets rest on the agent's turns only.

**Gold and silver criteria.** Criteria are gold when they come from the question or from the dataset's annotation,
silver when an LLM extracts structure the question does not state (`DatasetSpec.gold_criteria`, `criteria_choice`):

| Dataset | Gold | Silver |
|---|---|---|
| BrowseComp-Plus, NeuCLIR, RAGTIME | the extractor's criteria (the question's own clues and requirements) | none |
| deep_qa InfoSeek, WebShaper | the pool's rows (`queries/queries_all.jsonl`, any `level`) | none |
| deep_qa FRAMES | none | the extractor's multi-hop criteria |
| TRQA | made from the intermediate info on demand (`trqa_gold_criteria.py`) | the extractor's set criteria |
| trqa_validation | the pool's rows | the extractor's set criteria |

**Criteria source** (`criteria_source`: `gold`, the default, or `silver`). A choice on TRQA and trqa_validation only,
whose queries have both; the run goes to `belief-{mode}-gold/` or `belief-{mode}-silver/`. Elsewhere it has no
effect (an info line says so) and the run goes to `belief-{mode}/`. The belief model always runs: a query without
gold criteria (FRAMES, failed texts) gets silver ones (`gold_missing` in its `criteria_info`). Each query's
`criteria_info` records `criteria_kind` (`gold` or `silver`), and a gold row's `gold_level`. `run_config.json`
records `criteria_source` and the source used, `criteria_source_effective`.

**Criteria init** (`belief_model_component/criteria_init.py`). The criteria are part of the belief model: it makes
each query's list once, at the start of the query (`BeliefModel.reset`). The run only passes the dataset,
`criteria_source` and `llm_criteria`:

1. The kind: the requested `criteria_source` on TRQA and trqa_validation, gold elsewhere; a query without gold
   criteria gets silver ones.
2. The criteria of that kind: read from the bank when there, else made, added to the bank and used. Every kind is
   made the first time a query needs it, in inference, in training (phase 1 is a dra_inference run) and in
   `analysis/criteria_reachability.py`; no build step.

Before the workers start (after the vLLM servers are up), the run makes and banks the missing criteria of its
queries in parallel (`prepare_criteria`, with counts per kind and origin); the workers then read them from the bank,
and a query still missing is made by its worker. The settings are checked before any model is loaded
(`check_criteria_inputs`: e.g. a missing pool gold file stops the run).

The banks:

- The extractor's criteria (`llm_criteria`): `CRITERIA_ROOT/{dataset}/criteria_bank/{gold|silver}/criteria_{split}.jsonl`
  (gold for BrowseComp-Plus, NeuCLIR, RAGTIME; silver for TRQA; a training pool has one bank for all its splits,
  `CRITERIA_ROOT/training_pool/{pool}/criteria_bank/silver/criteria_all.jsonl`), `BankedCriteriaSource`.
- TRQA's gold criteria, in `criteria_bank/gold/` next to its silver ones (key: the writer and the hash of its
  prompts). They are made from the query's intermediate info as the trqa_validation pool makes its own
  (`annotations.set_row`: one criterion per member, the rest and the aggregate; members named by the page title of
  their first document, looked up in `wiki_partial` by id, or by TRQA's label for ecommerce); `llm_criteria` writes
  the property phrase and the aggregation (the pools' prompts, `training_pool/prompts/set_*`). The bank line keeps
  the whole annotation row (gold answers, documents, members) under `bank_only`, never written to a run's files.
  Texts that fail for good (a member or value leaks into the phrase, after one retry) are banked as final with no
  criteria, and the query gets silver ones; a failed writer call is not banked and is asked again next run.
- A training pool's gold criteria: the pool's queries file is their bank, written by its build only
  (`build_training_pool.py`), never during a run, so training scores against the same texts. A query whose gold
  texts failed gets silver ones; rebuild the pool to fix them.

Rules of a bank line:

- It is used only when the query id and text, `llm_criteria`, the query shape, `max_criteria` and a hash of the
  init prompt all match, so a changed prompt or cap extracts again. A `vllm/` model is keyed with the checkpoint
  its server loaded (`{model}@{path}`), so two checkpoints served under one name do not share criteria.
- A truncated reply is asked once more and fails when truncated again. A failed extraction is never added, so every
  criteria list a run uses is banked.
- A query extracted by two processes at once keeps the line banked first. Each process reads only the lines added
  since its last read.
- Runs and `analysis/criteria_reachability.py` share the bank, so the analysis assesses the criteria the runs use.
  The criteria are never scored against gold during a run.

### Belief update

The belief update implements Agent-BRACE's b_t = f(query, b_{t-1}, o_t). b_0 has no nugget. Each search iteration
makes one call (`belief_update_system.txt` in `src/belief_model_component/prompts/`, the per-shape context in
`nugget_shapes.txt`), the same prompt for the prompted teacher and a trained student, so one training sample is one
call. It reads every criterion with its current belief, the agent's turn of the iteration (its reasoning and search
queries, labelled `t<iteration>`; at the end, its reasoning and final answer) and the iteration's documents not judged
before, each labelled `d<n>` when first shown. Its reply has three lists:

1. **Think observations**, from the turn only: the turn gives or rejects one value of one criterion. A value that
   does not occur in the turn is dropped.
2. **Document observations**, from the passages only: a passage supports or contradicts one value of one criterion
   (not the aggregate), with spans copied from the passage. Each span is checked on its own (a span joined with
   "..." is split first); an observation without a verified span is dropped.
3. **Beliefs**: per criterion with a new observation, its whole new nugget list: one per answer (values naming one
   answer in different words share a nugget), each with a status and the labels of its sources. A nugget whose value
   is neither a current nugget's nor a new observation's, or with no valid status, is dropped. Its evidence is, per
   cited label, that source's observations (on its value, else all of the source's observations on the criterion),
   plus every source with an observation on its value.

The belief is self-contained: each nugget keeps its sources' observations (label, stance, value, verified spans), and
those are all the next call reads of b_{t-1} (the latest five per nugget are listed). When the call fails the belief
stays as it was and the documents are shown again when retrieved again.

Criteria are independent: no observation or nugget refers to another criterion. The status of a nugget, lowest
first, is how likely its value is a correct answer:

| level | status | the documents ... |
|---|---|---|
| -2 | doubtful | contradict the value, and none supports it |
| -1 | unlikely | point against it without contradicting it (e.g. support another value of a one-answer criterion), and none supports it |
| 0 | unknown | do not relate to it; only the agent's turns give it |
| 1 | possible | relate to it but do not show it is an answer, or disagree on it |
| 2 | probable | show it only in part, indirectly, or by combining several documents |
| 3 | almost certain | show most of what makes it an answer, and none contradicts it |
| 4 | certain | show it is an answer, and none contradicts it |

The agent's turns never raise a status above `unknown`: the status rates the documents, and the evidence labels show
which source each nugget rests on (document observations from the retriever side, think observations from the
reasoner side). Differences from Agent-BRACE: it ranks "unknown" lowest; here it is the neutral level 0. "certain" is
the paper's "confirmed / certain".

The update prompt and the agent read each nugget as one sentence (`belief_state.nugget_sentence`):

```
It is certain that the answer is "Henrik Dahl", given [d3] [t2].
It is doubtful that the answer is "Nils Berg", given [t1]; against [d2].
```

The certainty score s_t (`belief_model_component/certainty.py`) turns a belief into one number in [0, 1]:

1. Each status becomes a probability, (level + 2) / 6 (doubtful 0, unknown 1/3, certain 1).
2. Each criterion is scored by its most likely nugget, or as unknown without a nugget. After the final call, the last
   criterion of a multi-hop query is scored by the nugget the answer turn supports, i.e. the agent's own answer.
3. s_t is the geometric mean over the criteria. The rest and aggregate criteria of a set query are left out.

s_T, after the final call, is compared with the correctness of the agent's final answer.

Saved per iteration (`belief/{qid}.jsonl`, schema in `src/evaluation/belief/writer.py`): the passages shown, the
think and document observations with their notes, the nugget updates, the nuggets after the iteration, the documents supporting each
criterion (`criteria_sources_after`), the certainty score `certainty` and the raw reply; the meta line has
`final_certainty`, the score after the final call on the answer turn. For evaluation only, never shown to the agent:

- `new_item_precision` and `recall_so_far`, against the qrels at the dataset's `min_relevance_score`. NeuCLIR and
  RAGTIME use grade 3 (the very valuable docs), while the end-of-run `GradedRecall@N` counts every official level
  with its gain.
- Intermediate answers (BrowseComp-Plus and TRQA, all agents except `cpm_report`): after each iteration the agent's own
  model is asked, from its trajectory so far, for its most likely answer(s) in its answer format; it may give one,
  several or none (`IntermediateAnswerSignal`, prompt in `src/belief_model_component/prompts/`). They are saved as
  is, not evaluated.

### Inform mode

`--belief-mode inform` computes the same belief and also shows it to the agent: right after each iteration's search
results, one `<belief_state>` tag is appended to the trajectory (`src/belief_model_component/belief_state.py`):

```xml
<belief_state step="2">
  <criteria>
    <k1>the engineer who was born in Osby
      It is certain that the answer is "Henrik Dahl", given [d1] [t2].
      It is doubtful that the answer is "Nils Berg", given [t1]; against [d2].
    </k1>
    <k2>the lighthouse that the engineer designed</k2>
  </criteria>
  <sources>[d1] Henrik Dahl | [d2] Nils Berg (engineer)</sources>
</belief_state>
```

The tag is descriptive only: each criterion's nuggets, then the titles of the documents they cite. Gold-based fields
(`new_item_precision`, `recall_so_far`, relevant counts) stay in `belief/{qid}.jsonl` for analysis. Where the tag goes:

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
$DRA_OUTPUT_ROOT/{dataset}_{split}[_{query_key}]_{retriever}/{agent}_{backend}_{model}/belief-{mode}[-gold]/
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

`{agent}` is the `--policy-model` name (`oss_20b`, not the internal agent `oss`), as is `agent.policy_model` in
`run_config.json`. `_{query_key}` is in the name only when it is not the dataset's default. Each fact has one file: the run settings are
in `run_config.json` only, the final generation in `generation/{qid}.md` only, the aggregates in `summary.json` only.

A query is finished when its `trajectory/{qid}.jsonl` ends with the meta line, which is written after every other file
of the query (and, with the belief model on, after its belief file).
