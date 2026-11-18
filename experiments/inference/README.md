# Inference

`dra_inference.py` runs one deep research agent over one dataset split and saves every step (the generate stage),
runs the belief model over the saved trajectories (the belief stage) and evaluates the run. This README covers how to run it, what the belief model computes, how runs
are evaluated, and the output files. Datasets and indices must be built first
(see the [root README](../../README.md#data-and-indices)).

## Contents

- [Running](#running)
- [Belief model](#belief-model)
  - [Beliefs](#beliefs)
  - [Belief update](#belief-update)
- [Evaluation](#evaluation)
  - [Report evaluation (Auto-ARGUE)](#report-evaluation-auto-argue)
- [Output format](#output-format)

## Running

On Slurm, edit `DATASET` / `SUBSET` / `RETRIEVER` / `POLICY_MODEL` / `STAGE` at the top of the wrapper and
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
    --stage all \
    --workers 0
```

| Flag | Values |
|---|---|
| `--dataset`, `--subset` | `trqa` (`wiki1` / `wiki2` / `ecommerce`), `neuclir` (`news` / `technical`), `browsecomp_plus`, `ragtime`; unset subset = the dataset's default |
| `--retriever` | must have an index built for the dataset |
| `--policy-model` | the agent; its LLM is picked automatically (`utils/config.py`) |
| `--stage` | `generate`, `belief`, `eval`, `all` (default); see below |
| `--belief-model` | the belief model, one of `BELIEF_MODEL_TO_LLM` (`utils/config.py`): `qwen3.5_4b`, `qwen3.5_9b`, `qwen3.6_27b` (default); see [Belief model](#belief-model) |
| `--belief-source` | `gold` or `silver` beliefs of the belief model (TRQA, trqa_validation) |
| `--workers` | query-level worker processes of the generate stage: one per GPU on a GPU machine, CPU processes otherwise; `0` = auto (GPU count, 1 without GPU) |
| `--belief-workers` | queries the belief stage runs at once (threads; `belief_workers` in the YAML) |
| `--limit N` | each stage runs at most N queries |

Every other setting (top_k, rerankers, judges, eval k-values, ...) is in
`experiments/configs/dra_inference.yaml` and can be overridden with the matching `--flag`.

The stages:

| `--stage` | Runs |
|---|---|
| `generate` | the agent on the queries without a finished trajectory, then the evaluation of the generation |
| `belief` | the belief model over the finished trajectories without a belief file, then the evaluation of that belief model |
| `eval` | the evaluation of everything, from the saved files |
| `all` | `generate`, then `belief`, then the evaluation of everything |

The agent runs once; any number of belief model versions can then run over the same trajectories. The belief stage
needs no agent, retriever or GPU: it reads each search iteration's queries, the ids of the documents the agent was
shown and the agent's turn from `trajectory/{qid}.jsonl`, reads the documents' text from the corpus by id
(this call's corpus of the run's dataset and split; `CorpusLookup`, shaped as the run's retriever returns a
document), and the answer as the agent returned it (`agent_generation` in the trajectory's meta
line; `src/orchestration/belief_stage.py`). The belief model therefore reads what it read inside the agent loop.

```bash
python experiments/inference/dra_inference.py --dataset trqa --subset wiki2 --stage generate --limit 3
python experiments/inference/dra_inference.py --dataset trqa --subset wiki2 --stage belief
python experiments/inference/dra_inference.py --dataset trqa --subset wiki2 --stage eval
```

Every stage resumes: the generate stage skips the queries with a finished trajectory, the belief stage those with a
belief file of the current schema (an older one is made again). A query whose beliefs or a belief call failed before
any reply (the API or the server, not a reply that did not parse) stops there and is not saved, so the next belief
stage runs it again; the stage lists these queries at its end, and `metrics.md` gives the finished queries with a
belief file (`queries with a belief file`, the belief group). With `--stage all --limit N`, the belief stage takes the
queries this call generated first. A generate stage whose `dataset`, `agent` or `searcher` settings differ from the run's `run_config.json` is
refused, so every trajectory of a directory ran with the same settings; the belief and eval stages are refused when
the `dataset` settings (split, query key, relevance threshold) differ. The corpus path is recorded but not
compared (a data root may move); a change is printed.

One generation, many belief models: each belief model (`--belief-model`, its short name) has its own folder
`belief/{model}/` and its own entry in the `belief_models` section of `run_config.json`, so running the belief stage
with another `--belief-model` adds that model's beliefs and keeps the others. A belief stage whose settings for a
model (belief source and cap) differ from that model's saved entry removes that model's beliefs only
(`belief/{model}/`, `eval/belief/{model}/`), then saves the new settings, and runs every query again; copy them first
to keep them. The evaluation reports every belief model of the run (see Evaluation).
A `vllm/` belief model (`qwen3.5_4b`) is served by the belief stage itself: it starts the model's vLLM server
after the generate stage released its GPUs (so `--stage all` works), and stops it when the stage ends.

## Belief model

The belief model (`src/belief_model_component/`) runs in the belief stage, over the saved trajectories of any agent;
it never changes what the agent does. It follows Agent-BRACE (Singh et al., 2026, arXiv:2605.11436), adapted to search: the environment is the retriever,
the observation of a search iteration is its documents and the agent's turn, and the belief state gives each
belief of the query its candidates (possible answers), each with a status of Agent-BRACE's verbal belief scale, for
how likely it is an answer, and its support (the passage observations on it), and the names the agent proposed that no
passage supports yet. All beliefs and belief
calls are LLM calls on one backbone, the LLM of `--belief-model` (`belief_models.{model}.belief_model` in `run_config.json`, the full id): the belief init, the belief update
(`src/belief_model_component/update/updater.py`) and the final score.

The backbone is chosen by `--belief-model`; the models differ in size only (same prompts, temperature, token caps,
passage cut, reasoning off), for the model size study:

| `--belief-model` | LLM | runs on | precision | script |
|---|---|---|---|---|
| `qwen3.5_4b` | `vllm/Qwen/Qwen3.5-4B` | local vLLM, 1 GPU | bf16 | `scripts/run_dra_inference.sh` |
| `qwen3.5_9b` | `openrouter/qwen/qwen3.5-9b` | OpenRouter, bf16 hosts only | bf16 | either script (`run_dra_inference_cpu.sh` needs no GPU) |
| `qwen3.6_27b` (default) | `openrouter/qwen/qwen3.6-27b` | OpenRouter, fp8 hosts only | fp8 | either script |

The OpenRouter models are pinned to the hosts of one precision (`BELIEF_OPENROUTER_PROVIDER` in `utils/config.py`,
OpenRouter's `provider.quantizations`), so every call of a model runs the same weights; qwen3.6-27b has no bf16 host.
The pin is recorded in the model's `belief_models` entry (`openrouter_provider`); a changed pin makes that model's
beliefs again.

The local model's server (`utils/vllm_manager.py`, `_BELIEF_SERVER_SPECS` and `belief_server`) runs on GPU 0, port 7010
(`$BELIEF_VLLM_API_BASE` is set to it), text only (`--language-model-only`), bf16, prefix caching on, up to 128
sequences at once; a server already on port 7010 that serves the model is reused and left running (anything else on the port stops the stage). Weights come from
`$DRA_MODELS_DIR/Qwen--Qwen3.5-4B/` when pre-staged there, else from the Hugging Face cache. The belief stage sends
`--belief-workers` queries at once (64 in `run_dra_inference.sh` for this model; each query's calls run one
after another), and belief init runs as many queries at once. Each model writes its own `belief/{model}/` folder.

### Beliefs

At the start of each sample, the belief model derives a fixed list of beliefs from the query: the pieces of
information a complete response must establish. The dataset's `query_shape` (`layout.DATASET_SPECS`) picks the
prompt, a shared core plus one block per shape:

| Shape | Datasets | Beliefs |
|---|---|---|
| multi-clue | BrowseComp-Plus | every clue of the target (the one entity the clues describe), copied verbatim, then the target ("the institution that meets B1 to B5"), then the asked answer when it is not the target ("the birth year of the wife of c4"; no cap) |
| multi-hop | infoseek, webshaper_frames | one belief per entity on the chain, the answer last, each naming only its own hop and referring only to hops before it by their ids ("the engineer who was born in B1"; an extraction with a forward reference is asked once more; FRAMES; at most 8) |
| multi-member | TRQA | one "member: property" per known member, then the aggregate: how the answer follows from the members' values (the layout of the gold beliefs; at most 52: 51 members and the aggregate) |
| multi-aspect | NeuCLIR, RAGTIME | every requirement the long request (background and problem statement) states, as one plain list (no cap, nothing added) |

`max_beliefs` (null = the dataset's default in `layout.DATASET_SPECS`) caps silver beliefs only: TRQA and
trqa_validation (52) and webshaper_frames' FRAMES (8). Gold beliefs are never capped: on BrowseComp-Plus, NeuCLIR and RAGTIME,
whose beliefs are all gold, a `max_beliefs` stops the run (`init.factory.check_max_beliefs`).

Rules for report beliefs:

- A belief keeps the request's own limits in its wording. What the request leaves out is written into the
  belief it narrows ("..., not ..."); an exception ("unless ...") is its own belief.
- Background on the asker is kept only when it limits what information applies (country, location, situation).
- Every belief reads on its own, naming its subject instead of pointing back ("it", "these").
- Beliefs are distinct: none asks for what another one asks for.

Each belief has a `kind`, set by code from the shape and the texts (`init.parsing.belief_kinds`, also
applied to the training datasets' gold rows): `closed` (one fact with one answer), `aggregate` (it combines the beliefs
before it) or `aspect` (one aspect of a report topic). A multi-clue query's clues and asked answer are closed, its
"meets B1 to B<k>" belief an aggregate, the target (`init.parsing.target_index`); every hop of a multi-hop query
is closed; a multi-member query's members are closed and its last belief an aggregate, the computation (added by
code when the model leaves it out; an "any other member" item is dropped); all report beliefs are aspects. A closed
belief has one answer, so its candidates are alternatives. An aggregate's statuses are code's: a target's
candidates are the names listed on every clue, a computation's are the answers the agent's turns compute; each is at
the lowest status of its inputs, "unlikely" when a passage speaks against it.

**Gold and silver beliefs.** Beliefs are gold when they come from the question or from the dataset's annotation,
silver when an LLM extracts structure the question does not state (`DatasetSpec.gold_beliefs`, `belief_choice`):

| Dataset | Gold | Silver |
|---|---|---|
| BrowseComp-Plus, NeuCLIR, RAGTIME | the extractor's beliefs (the question's own clues and requirements) | none |
| infoseek, webshaper_frames WebShaper | the dataset's rows (`queries/queries_all.jsonl`, any `level`) | none |
| webshaper_frames FRAMES | none | the extractor's multi-hop beliefs |
| TRQA | made from the intermediate info on demand (`training_datasets/gold/trqa.py`) | the extractor's set beliefs |
| trqa_validation | the dataset's rows | the extractor's set beliefs |

**Belief source** (`belief_source`: `gold`, the default, or `silver`). A choice on TRQA and trqa_validation only,
whose queries have both; the run goes to `belief-{mode}-gold/` or `belief-{mode}-silver/`. Elsewhere it has no
effect (an info line says so) and the run goes to `belief-{mode}/`. The belief model always runs: a query without
gold beliefs (FRAMES, failed texts) gets silver ones (`gold_missing` in its `belief_info`). Each query's
`belief_info` records `belief_source` (`gold` or `silver`), and a gold row's `gold_level`. `run_config.json`
records `belief_source` and the source used, `belief_source_effective`.

**Belief init** (`belief_model_component/beliefs/factory.py`). The beliefs are part of the belief model: it makes
each query's list once, at the start of the query (`BeliefModel.reset`). The run only passes the dataset,
`belief_source` and `belief_model`:

1. The kind: the requested `belief_source` on TRQA and trqa_validation, gold elsewhere; a query without gold
   beliefs gets silver ones.
2. The beliefs of that kind: read from the bank when there, else made, added to the bank and used. Every kind is
   made the first time a query needs it, in the belief stage and in `analysis/criteria_reachability.py`; no build
   step.

Before the workers start (after the vLLM servers are up), the run makes and banks the missing beliefs of its
queries in parallel (`prepare_beliefs`, with counts per kind and origin); the workers share that belief source
and read them from the bank, and a query still missing is made by its worker. The settings are checked before any model is loaded
(`check_belief_inputs`: e.g. a missing training dataset gold file stops the run).

The banks:

- The extractor's beliefs (`belief_model`): `BELIEF_ROOT/{dataset}/belief_bank/{gold|silver}/beliefs_{split}.jsonl`
  (gold for BrowseComp-Plus, NeuCLIR, RAGTIME; silver for TRQA; a training dataset has one bank for all its splits,
  `BELIEF_ROOT/training_datasets/{dataset}/belief_bank/silver/beliefs_all.jsonl`), `BankedBeliefSource`.
- TRQA's gold beliefs, in `belief_bank/gold/` next to its silver ones (key: the writer and the hash of its
  prompts). They are made from the query's intermediate info as the trqa_validation dataset makes its own
  (`annotations.multi_member_row`: one belief per member, then the aggregate; members named by the page title of
  their first document, looked up in `wiki_partial` by id, or by TRQA's label for ecommerce); `belief_model` writes
  the property phrase and the aggregation (the training datasets' prompts, `training_datasets/gold/prompts/multi_member_*`). The bank line keeps
  the whole annotation row (gold answers, documents, members) under `bank_only`, never written to a run's files.
  Texts that fail for good (a member or value leaks into the phrase, after one retry) are banked as final with no
  beliefs, and the query gets silver ones; a failed writer call is not banked and is asked again next run.
- A training dataset's gold beliefs: its queries file is their bank, written by its build only
  (`build_training_dataset.py`), never during a run, so training scores against the same texts. A query whose gold
  texts failed gets silver ones; rebuild the dataset to fix them.

Rules of a bank line:

- It is used only when the query id and text, `belief_model`, the query shape, `max_beliefs` and a hash of the
  init prompt all match, so a changed prompt or cap extracts again. A `vllm/` model is keyed with the checkpoint
  its server loaded (`{model}@{path}`), so two checkpoints served under one name do not share beliefs.
- A truncated reply is asked once more and fails when truncated again. A failed extraction is never added, so every
  belief list a run uses is banked.
- A query extracted by two processes at once keeps the line banked first. Each process reads only the lines added
  since its last read.
- Runs and `analysis/criteria_reachability.py` share the bank, so the analysis assesses the beliefs the runs use.
  The beliefs are never scored against gold during a run.

### Belief update

The belief update implements Agent-BRACE's b_t = f(query, b_{t-1}, o_t). In b_0 no belief has a candidate. A call
(`system.txt` in `src/belief_model_component/prompts/belief_update/`, the WEP definitions of the shape's belief
types in `prompts/shared/wep_definitions.txt`, its user template in `user.txt`, and the shape's own `<shape>.txt`: its rules and its example) is the same prompt for the prompted teacher and a trained student, so one training sample is one call.

**When a call is made** (`BeliefUpdateSignal.plan`). A search iteration makes a call when its search returned a
document with text; the call shows every document of the iteration, also one shown before (labelled `d<n>` when first
shown, the same label after; a passage shown again on the same candidate and stance counts once). Otherwise no call
is made (`called: false`): the belief is carried forward and the turn waits; the next call reads every turn since the
last call (the last 6), each labelled `t<iteration>`. After the last iteration, the answer turn (its reasoning and
final answer) makes the final call: on a short-answer dataset the final scorer's (below), on a report one more
update with the turns still waiting.

**Passage length.** The agent and the belief update read the same text of each document: its first
`max_passage_tokens` tokens (one shared setting, default 512 as in BrowseComp-Plus, counted with the `Qwen/Qwen3.5-4B`
tokenizer whatever the agent's model, no marker at the cut; `utils.text_utils.cut_passage`), report documents
included. A passage's title is its metadata title, else its front matter's (`title:`), else its first line.

The call reads every belief (id `B<k>`) with its statement, its passages and its
proposals, the turns and the passages (prompt in `prompts/belief_update/`). The reply has three lists:

1. **Reasoning observations** (claims): per candidate the reasoning names, assesses or rejects, `{id, candidate,
   span, agent_claim}`: the claim is a word of estimative probability (WEP), or "rejected". The span must occur in a turn.
2. **Passage observations**: per passage, belief and candidate, `{id, passage, candidate, stance, spans}`: `+` when the
   spans say the candidate meets a part of the belief, `-` when they say it fails one. Each span is checked on its own
   (a span joined with "..." is split first); an observation without a verified span is dropped.
3. **Beliefs**: per belief with a new passage observation, `{id, reason, most_probable, wep}`: the model chooses the
   most probable candidate and its WEP from all the belief's passages, previous and new (more passages, more parts
   and more direct spans raise the word; a `-` passage, a hedged span or another supported candidate lowers it).

The words, lowest first, with their ranges (`prompts/shared/wep_definitions.txt`, `core.types.STATUS_PROBABILITY`
the midpoints): almost no chance (below 20%, 0.10), unknown (no passage supports a candidate, 0.25), unlikely
(20-40%, 0.30), possible (40-60%, 0.50), probable (60-80%, 0.70), almost certain (80-95%, 0.875), certain (95% or
more, 0.97). The labels are Agent-BRACE's (Kent's words of estimative probability). On a report's aspect each
candidate is one fact; it has no most probable candidate (null), the WEP rates how well the kept facts cover the
aspect, and a `-` removes a fact.

The belief is kept at two levels. **Shown** to the next update (`views.belief_text.format_belief`): per belief its
statement, built by code (`belief_statement`: "It is <wep> that <most probable candidate> addresses <belief>.", "It is
unknown which candidate addresses <belief>.", on an aspect "the kept facts address"), one line per passage on a
candidate (the stance, the candidate, the spans and the passage's title in brackets; per candidate and stance the
latest 5, the earlier ones counted as `+ <candidate>: (N more passages)`), then `proposed:` and the names the agent
proposed that no passage supports yet:

```
B1: It is possible that Osby addresses the town where the Varn Folk Museum was founded.
  + Osby: "the folk museum of the Varn valley opened in Osby in 1921" [Osby]
  + Tirsa: "a crafts museum with objects from the Varn valley opened in Tirsa in 1925" [Tirsa]
B2: It is unknown which candidate addresses the engineer who was born in B1.
  proposed: Nils Berg
```

**Saved**: every observation that entered (verified spans), per belief every source ever observed on it (the
`references` of the `final` line), the agent's latest claim on each candidate, and the belief items as written and as
kept, with their reasons (`updates`); the claims are never shown.

Code applies the reply (`BeliefState.apply`), in order: observations, claims, passage observations, belief items,
aggregates.

- **Reasoning only proposes.** A claim sets the agent's claim on a listed candidate, or adds a proposal (never
  selected; at most 8 per belief), updates it, or takes it off the proposals when the claim is "rejected". A
  rejection never removes a candidate (a computed answer included): it is saved as the candidate's claim. A proposal
  becomes a candidate when a passage observation names it.
- **Passage observations.** The passage must be one of the call (`unknown_passage`), the stance `+` or `-`
  (`stance_invalid`), the candidate given (`candidate_missing`). The observation joins its candidate's support (the
  same passage and stance on a candidate counts once). A new name becomes a candidate (a `-` alone too); on an aspect
  a new fact whose words contain a listed one, or the reverse, joins it with a `+` (`fact_merged`), and a `-` removes
  the fact or proposal it names (never a near one). A one-answer candidate is never removed by a passage. A new
  candidate with a `+` in none of its belief's passage spans and titles is flagged (`candidate_not_in_passages`; a
  figure such as "120 m" is found by its numbers), one whose name contains a listed one, or the reverse
  (`candidate_near_duplicate`).
- **Most probable and WEP.** The most probable candidate written is kept when it is listed with a `+` passage, else
  code takes the first added of those with the most `+` passages (`most_probable_wrong`). The WEP is taken as written;
  one that is no word of the scale keeps the previous WEP (`wep_invalid`); a word above "unknown" with no supported
  candidate is set to "unknown" (`wep_without_support`); "unknown" written with a supported candidate is kept and
  flagged (`unknown_with_support`). A belief has one WEP: the most probable candidate carries it (an aspect: every
  kept fact), any other candidate has no status. A belief with a new passage observation and no item keeps its WEP
  (`belief_missing`).
- **Target.** A multi-clue query's target takes no observation (`observation_on_target`) and no belief item
  (`update_on_target`). After each call code derives its candidate: the name that is the most probable candidate of
  every clue, at the lowest of their WEPs; none when a clue is "unknown" or names another.
- **Computed answer** (multi_member). Its candidates are the answers the reasoning computes (claims); a passage
  observation on it names one (`answer_not_computed` otherwise) or none (null: it states an input). It takes no
  belief item (`update_on_computed`). Code keeps each answer at the lowest status of the inputs, "unlikely" when a
  passage speaks against it, at most "possible" when the reasoning computed another answer later or rejects it; saved,
  never shown:
  the update reads the belief as "unknown" with the passages on its inputs. The member a member belief is about is
  never its candidate (`subject_candidate`).
- **Not taken**: an item of an unknown belief, a second item of one belief (`belief_repeated`), an item of a belief
  without a passage observation in the call (`update_without_observation`); a claim without a candidate or a valid
  claim (`claim_incomplete`); a claim "unknown" (a name the agent only checks) is not taken either, as a note
  (`claim_unknown`). A - passage on a one-answer candidate whose spans share no content word with the belief is
  flagged (`span_off_belief`). Removed facts and dropped proposals (rejected, or past the cap) are saved in the final
  line's `references[].removed`; a name added again gets a note (`candidate_readded`).
- **Issue codes.** Every code is in one table (`belief_model_component/core/issues.py`, `ISSUES`) with its effect
  (dropped, corrected, flagged or note) and its training role (format gate, unclean). The step record's `issues` holds
  the errors, its `notes` what is no error. The trajectory md shows them one line per effect, and `metrics.md` gives
  the share of update calls with each effect, failing the format gate or unclean, and every code that occurs
  (`evaluation/belief/issues.py`).

Derived per belief: its status (its WEP; an aggregate's the highest of its candidates), and its selected
candidate(s): for a one-answer belief its most probable candidate at "possible" or above, for an aspect every fact; a
proposal is never selected. When the call fails the belief stays as it was and its turns' claims are lost.

The certainty score s_t (`belief_model_component/core/certainty.py`) is a fixed function of the statuses:

1. Each belief's status (its WEP) becomes a probability, `core.types.STATUS_PROBABILITY` (the midpoint of each
   word's range, unknown 0.25): q_k. The same table turns every status into a number, the final scorer's included.
   The update judges a one-answer WEP from all the belief's passages, rivals included, so the candidates share
   nothing.
2. s_t is the mean q_k over the beliefs the updater judges, on every query shape: the expected share of beliefs
   that hold, so a change on any belief moves it. An aggregate is left out: its status is derived from its inputs
   (the lowest of them), so it would count its weakest input twice. How much each belief counts for the final
   answer is the final scorer's to judge (s_T below), not the formula's.

s_T is compared with the correctness of the agent's final answer. On a short-answer dataset it is the final
scorer's (`belief_model_component/update/final_score.py`, prompt in `prompts/belief_score/`): one call reads the query,
the belief before the answer turn (statements, passages and proposals, as the update reads it) and
the final turn (`[t<N>]`, the last 2000 characters of its reasoning and the final answer under `Final answer:`, no
citations, no confidence line), and writes a reason and the WEP that the final answer is correct, read as a
probability by `STATUS_PROBABILITY`. It weighs each belief by how much the answer's correctness depends on it; it is
not told which candidate the answer is. The fixed function of the final statuses is kept next to it as a baseline
(`certainty_formula`). On a report, s_T is the fixed function of the belief after the final update.

Saved in `belief/{model}/{qid}.jsonl` (schema in `utils/run_files/belief.py`): a meta line, one line per iteration and
one `final` line for the answer turn. A step line has the retrieval measures, `certainty`, `called`, the documents
(`{doc_id, new, label}`), every belief's state after the step (`beliefs`: `{id, status, most_probable, selected,
candidates, proposals}`, each candidate with its `support` and the agent's `claim`), the belief `updates` of the
reply (with their reasons and passage observations), the beliefs it `changed`, the `observations` that entered (claims and
passage observations, with their verified spans), the reply's `issues`, the agent turn and the raw reply. The `final` line has the same belief fields plus `references` (per
belief every source assigned to it over the sample, and the candidates and proposals dropped from it); its `certainty` is s_T,
with `certainty_label`, `certainty_formula` and `final_score` (status, reason). The same belief is readable in
`belief/{model}/{qid}.md`: the beliefs, a belief table after each iteration (with the labels of its documents) and the
final belief. For evaluation only: `new_item_precision` and `recall_so_far`, against the qrels at the dataset's
`min_relevance_score` (NeuCLIR and RAGTIME: grade 3, the very valuable docs), the same relevant docs as the
end-of-run Recall@N.

## Evaluation

A run is evaluated from its saved files at the end of every stage. Each stage scores its own scope and keeps the
rest of `eval/metrics.md` and `eval/per_query.csv`: `generate` the generation, `belief` its belief model, `eval` and
`all` everything. The belief stage scores the generation too when the saved generation rows cannot be kept (no
`metrics.md`, no generation row in it, or another number of queries than the run's, e.g. after an interrupted
generate stage). Judge verdicts are reused for queries whose input is unchanged, so `--stage eval` on an unchanged run
makes no LLM call and needs no GPU. The s_T and belief rows score the queries with a belief file.

The measures, by group, in the order of the files and the log. The generation's measures (retrieval, generation,
the agent's verbal confidence, the agent's trajectory) have one row; the beliefs' measures (s_T, belief, belief
model tokens) have one row per belief model of the run (`belief/{model}/`), tagged in the `belief_model` column of
`metrics.md` ("-" on the generation's rows), with their per-query columns of `per_query.csv` prefixed `{model}/` and
their sections (status calibration, issues) titled `({model})`. The judge verdicts are cached per model, so adding a
belief model costs only its own judge calls.

| Group | Measures | Datasets |
|---|---|---|
| retrieval | Recall@N, Precision@N of the seen docs (N = the distinct docs shown to the agent), N | all |
| generation | accuracy (LLM judge, `--judge-model`; TRQA: numeric exact match), correct, evaluated, judge errors | QA |
| | Auto-ARGUE nugget coverage and F1 (NeuCLIR: also the weighted ones), reports evaluated, without nuggets, empty, without citations, with unmapped citations, judge failures | reports |
| uncertainty | verbal: queries scored, AUROC [CI], Brier (once per run) | QA, glm and oss |
| | queries scored; s_T: AUROC [CI], Brier; formula: AUROC [CI] (per belief model) | QA, belief on |
| | queries scored; s_T Kendall tau-b [CI] and Brier vs nugget coverage (and weighted) | reports, belief on |
| belief | gold and control event change, belief recall, answer reached, status Brier; status calibration table (below) | TRQA, trqa_validation, infoseek, belief on |
| trajectory | searches, agent tokens, belief model tokens (per query, means), forced answers, max iterations reached | all |

CIs are 95% percentile bootstrap intervals over queries (1000 resamples, `evaluation.measures`). Nothing else is
computed: no @k tables, NDCG, MAP, GradedRecall, cited-doc or fusion metrics, soft match or length statistics (the
raw outputs stay in the run, for `analysis/`).

**Uncertainty** (`src/evaluation/uncertainty/`). y is the correctness of the final answer (a judge error leaves the
query out). The baseline `verbal` (`uncertainty/verbal.py`) is a measure of the generation, scored once per run: the
agent's own `Confidence: X%` (glm and oss; read from `generation/{qid}.md` by a regex,
`evaluation/generation/confidence.py`, "~85 %" and a fraction such as "0.85" accepted; a generation without it has
none), on the judged queries with a verbal confidence. Per belief model (`uncertainty/qa.py`): `s_T` (the final
scorer, the headline) and the ablation `formula` (the fixed function of the final statuses: the mean of the status
probabilities of the beliefs the updater judges, aggregates left out), on the queries with y and s_T. The verbal and
s_T rows can therefore have different n. Rows: AUROC [CI] for every score, Brier for verbal and s_T. On a report dataset (`uncertainty/report.py`)
s_T is the certainty of the final belief (the mean of the aspects' status probabilities; a report has no
final scorer) and y the report's Auto-ARGUE nugget coverage: Kendall's tau-b [CI] (the ranking) and Brier (s_T as the
expected coverage); an empty report is left out. Left-out queries are logged.

**Belief** (`src/evaluation/belief/evaluator.py`), on the queries whose beliefs are the gold beliefs (TRQA members,
InfoSeek hops; not WebShaper or FRAMES, which have no gold per belief). Scored beliefs: a member or hop (gold
answer and documents) and an aggregate (gold answer only).

- A **gold event** is a gold document of belief k, new to the belief, whose text as the update read it (the
  `max_passage_tokens` cut) states k's gold answer (string match, else the judge). A **control event** is a new
  document observed on k that is not one of its gold documents and does not state its answer.
- `gold event: gold candidate change`: the change of the gold candidate's status probability at the belief's
  first gold event (unknown -> certain = 0.25 -> 0.97 = +0.72; a gold answer that is not a candidate is unknown);
  `control event: ...`: the same at its control events, which should be about 0. An event whose gold candidate is
  already "certain" is left out.
- `belief recall`: the share of the belief's gold documents the agent saw.
- `reached answer`: at the end, the belief's selected candidate is a gold answer; `reached answer, gold seen`: the same over members and hops with a gold document seen (the belief's own
  share).
- `status Brier`: the final status probability (the WEP) against whether the most probable
  candidate is a gold answer (whatever the WEP; `reached` needs "possible" or above).

Each row comes twice: per query (each query's mean over its beliefs, then the mean over queries, CI over queries)
and `(c)`, the mean over every belief. Below the table, the status calibration: per final status, how many
beliefs had it and how many have a gold answer as most probable candidate (the Brier's y). A candidate is a gold answer when it matches one, else when the
judge (`judge_model`) says so, cached in `eval/belief/{model}/judge_cache.jsonl`. For diagnosis, `eval/belief/{model}/events.csv`
has every event with a gold event's funnel stage (`call_failed`, `not_assigned`, `no_gold_candidate`,
`not_selected`, `correct_lost`, `correct_kept`), and `per_query.csv` has each query's values and `belief_correct`
(every member or hop reached).

The terminal log prints one block: one line per group of the generation, then per belief model its lines:

```
EVALUATION  trqa / wiki2_test / glm / belief models: qwen3.5_9b, qwen3.6_27b  (n=200; scored now: generation, qwen3.5_9b, qwen3.6_27b)
retrieval    Recall@N 0.412 | Precision@N 0.081 (N 38.5)
generation   accuracy 0.310 (62/200)
uncertainty  verbal AUROC 0.610 [0.53,0.69] Brier 0.310 (n 196)
trajectory   searches 9.2 | agent tok 41234 | forced 4
[qwen3.5_9b]
uncertainty  s_T AUROC 0.710 [0.63,0.78] Brier 0.220 (n 188)
belief       files 200 | update calls 1840 | format gate failed 0.020 | belief tok 18230
belief gold  gold change 0.540 | control change 0.050 | recall 0.710 | reached 0.470 (gold seen 0.680) | status Brier 0.210 (n 200)
[qwen3.6_27b]
uncertainty  s_T AUROC 0.780 [0.71,0.84] Brier 0.190 (n 188)
belief       files 200 | update calls 1840 | format gate failed 0.010 | belief tok 23102
belief gold  gold change 0.620 | control change 0.030 | recall 0.710 | reached 0.520 (gold seen 0.730) | status Brier 0.180 (n 200)
```

Every run can be compared in one table with `analysis/collect_results.py --root run_outputs` (one row per run, the
headline measures as columns; `--all` for every measure, `--group` to keep some groups, `--ci` for the CIs):
`results.md` and `results.csv`.

`utils/run_files/` writes and reads every file of the run directory; `src/evaluation/` only scores, one module or
subpackage per group:

```
utils/run_files/
├── __init__.py      QUERY_OUTPUT_DIRS, save_query_outputs, load_result_from_saved_files
├── retrieval.py     surfaced / seen / cited doc levels and their TREC files
├── generation.py    generation/{qid}.md
├── trajectory.py    trajectory/{qid}.jsonl, scan_queries (finished queries)
├── trajectory_logger.py  streamed trajectory lines and the step shaping both writers share
├── belief.py        belief/{model}/{qid}.jsonl (schema, SCHEMA_VERSION), save/read/load, believed_queries
└── run_config.py    run directory name, run_config.json, passage cut of a run

src/evaluation/
├── runner.py        build_evaluators, load_run_results, evaluate_and_save
├── metrics_file.py  metrics.md, per_query.csv and the log block
├── measures.py      AUROC, Brier, ECE, tau-b, bootstrap CIs (shared with the analyses)
├── judge.py         LLM-judge client and DEFAULT_JUDGE_MODEL
├── retrieval.py     seen-doc Recall@N, Precision@N
├── generation/      AccuracyEvaluator (LLM judge), NumericMatchEvaluator (TRQA), ArgueReportEvaluator (reports),
│                    verbal_confidence
├── uncertainty/     UncertaintyEvaluator (qa.py), ReportUncertaintyEvaluator (report.py)
├── belief/          BeliefUpdateEvaluator, per-belief gold (gold_rows.py), gold match and judge
│                    (gold_match.py), issue_summary
└── trajectory.py    TrajectoryEvaluator (per query)
```

### Report evaluation (Auto-ARGUE)

On the report datasets (`report_eval: "argue"` in `layout.DATASET_SPECS`: NeuCLIR, RAGTIME), every evaluation
scores the reports against the nuggets with the official
[Auto-ARGUE](https://github.com/hltcoe/auto-argue) package (`src/evaluation/generation/argue.py`). There is no flag. The
package runs unmodified with our judge (`judge_model`, Qwen3-32B, temperature 0, reasoning off) instead of its
Llama-3.3-70B.

- Input: `generation/{qid}.md` split into sentences, each with the documents its `[N]` markers cite (resolved through
  the References block), read whole: a report has no length limit (the tracks' 2000-character cut is not applied, so
  scores are comparable across our runs, not to the official track numbers). Agents that cite by docid (oss, glm on a
  report task) have their `[docid]` / `[[docid]]` citations, unique id prefixes included, numbered `[N]` when the run
  ends (`utils.text_utils.number_docid_citations`). `metrics.md` counts the reports without any parsed citation and
  those with markers that map to no reference: their citation and nugget scores are not meaningful.
- Nuggets: the dataset's nuggets as v3 banks. NeuCLIR uses the NIST QC'd bank with importance (vital counts 2, okay 1),
  the AND/OR aggregation and each answer's supporting documents; only its 22 requests with supporting documents can be
  scored. RAGTIME has no importance labels, so the weighted scores equal the plain ones.
- Scores (mean over requests, an empty report scores 0): `nugget_coverage` (`_weighted`), `sentence_support`,
  `citation_support`, `citation_relevance`, `f1` (`_weighted`). A nugget only counts when its sentence cites one of
  its supporting documents.
- Cache: `eval/generation/` keeps the nugget banks, the cited documents and each report's judgments keyed by a hash of
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
$DRA_OUTPUT_ROOT/{dataset}_{split}[_{query_key}]_{retriever}/{agent}_{backend}_{model}/
├── run_config.json                 every resolved setting: dataset, agent (temperature included), searcher
│                                   (generate stage), belief_models ({model: settings}, belief stage), evaluation
├── retrieval/
│   ├── surfaced/{qid}.trec         raw retriever output (all iterations, col 6 = iter_N)
│   ├── seen/{qid}.trec             docs shown to the LLM
│   └── cited/{qid}.trec            docs cited by the LLM
├── generation/{qid}.md             the final generation (with its References section)
├── trajectory/{qid}.jsonl          one line per step (queries, seen doc ids, the search_step of a search
│   │                               iteration), then the meta line (counts, agent data, agent_generation)
│   └── {qid}.md                    the same trajectory, human-readable, written live
├── belief/{model}/                 belief stage, one folder per belief model (--belief-model short name)
│   ├── {qid}.jsonl                 meta line (beliefs) + one line per search iteration + final line
│   └── {qid}.md                    the same belief, human-readable
└── eval/                           the evaluation, written by every stage
    ├── metrics.md                  every measure: group | belief_model | metric | value | ci_low | ci_high | n
    ├── per_query.csv               every per-query value of every group (belief model columns: {model}/...)
    ├── generation/                 accuracy.jsonl: per-query verdicts (datasets with answers); or Auto-ARGUE
    │                               inputs (nuggets/, cited docs), cached judgments/, per-query scores.tsv (reports)
    └── belief/{model}/             belief update evaluation: events.csv (one row per gold or control event),
                                    judge_cache.jsonl
```

`{agent}` is the `--policy-model` name (`oss_20b`, not the internal agent `oss`), as is `agent.policy_model` in
`run_config.json`. `_{query_key}` is in the name only when it is not the dataset's default. Each fact has one file: the run settings are
in `run_config.json` only, the final generation in `generation/{qid}.md` only, the aggregates in `eval/metrics.md` only.
`eval/per_query.csv` repeats per-query values of the other files (correctness, s_T, verbal confidence, tokens): it is
evaluation output, rebuilt by every evaluation and never read by inference or training. An evaluation removes the
`summary.json` of earlier code.

A query is generated when its `trajectory/{qid}.jsonl` ends with the meta line, which is written after every other
file of the query; its belief by a belief model is done when `belief/{model}/{qid}.jsonl` exists (written last, atomically). The meta line's
`agent_generation` is the generation as the agent returned it, the answer the belief stage reads; `generation/{qid}.md`
has the References section the pipeline appends, with docid citations renumbered `[N]`.
