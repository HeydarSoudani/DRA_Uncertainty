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
criterion of the query its candidates (possible answers), each with a status of Agent-BRACE's verbal belief scale, for
how likely it is an answer, and its support (the passage observations on it), and the names the agent proposed that no
passage supports yet. All criteria and belief
calls are LLM calls: the criteria extractor
uses `llm_criteria`, the belief update (`src/belief_model_component/update/updater.py`) `belief_judge_model` (default
`llm_criteria`).

### Criteria

At the start of each sample, the belief model derives a fixed list of criteria from the query: the pieces of
information a complete response must establish. The dataset's `query_shape` (`layout.DATASET_SPECS`) picks the
prompt, a shared core plus one block per shape:

| Shape | Datasets | Criteria |
|---|---|---|
| multi-clue | BrowseComp-Plus | every clue of the target (the one entity the clues describe), copied verbatim, then the target ("the institution that meets c1 to c5"), then the asked answer when it is not the target ("the birth year of the wife of c4"; no cap) |
| multi-hop | infoseek, webshaper_frames | one criterion per entity on the chain, the answer last, each naming only its own hop and referring only to hops before it by their ids ("the engineer who was born in the town of c1"; an extraction with a forward reference is asked once more; FRAMES; at most 8) |
| multi-member | TRQA | one "member: property" per known member, then the aggregate: how the answer follows from the members' values (the layout of the gold criteria; at most 52: 51 members and the aggregate) |
| multi-aspect | NeuCLIR, RAGTIME | every requirement the long request (background and problem statement) states, as one plain list (no cap, nothing added) |

`max_criteria` (null = the dataset's default in `layout.DATASET_SPECS`) caps silver criteria only: TRQA and
trqa_validation (52) and webshaper_frames' FRAMES (8). Gold criteria are never capped: on BrowseComp-Plus, NeuCLIR and RAGTIME,
whose criteria are all gold, a `max_criteria` stops the run (`criteria.factory.check_max_criteria`).

Rules for report criteria:

- A criterion keeps the request's own limits in its wording. What the request leaves out is written into the
  criterion it narrows ("..., not ..."); an exception ("unless ...") is its own criterion.
- Background on the asker is kept only when it limits what information applies (country, location, situation).
- Every criterion reads on its own, naming its subject instead of pointing back ("it", "these").
- Criteria are distinct: none asks for what another one asks for.

Each criterion has a `kind`, set by code from the shape and the texts (`criteria.parsing.criterion_kinds`, also
applied to the training datasets' gold rows): `closed` (one fact with one answer), `aggregate` (it combines the criteria
before it) or `aspect` (one aspect of a report topic). A multi-clue query's clues and asked answer are closed, its
"meets c1 to c<k>" criterion an aggregate, the target (`criteria.parsing.target_index`); every hop of a multi-hop query
is closed; a multi-member query's members are closed and its last criterion an aggregate, the computation (added by
code when the model leaves it out; an "any other member" item is dropped); all report criteria are aspects. A closed
criterion has one answer, so its candidates are alternatives. An aggregate's statuses are code's: a target's
candidates are the names listed on every clue, a computation's are the answers the agent's turns compute; each is at
the lowest status of its inputs, "unlikely" when a passage speaks against it.

**Gold and silver criteria.** Criteria are gold when they come from the question or from the dataset's annotation,
silver when an LLM extracts structure the question does not state (`DatasetSpec.gold_criteria`, `criteria_choice`):

| Dataset | Gold | Silver |
|---|---|---|
| BrowseComp-Plus, NeuCLIR, RAGTIME | the extractor's criteria (the question's own clues and requirements) | none |
| infoseek, webshaper_frames WebShaper | the dataset's rows (`queries/queries_all.jsonl`, any `level`) | none |
| webshaper_frames FRAMES | none | the extractor's multi-hop criteria |
| TRQA | made from the intermediate info on demand (`training_datasets/gold/trqa.py`) | the extractor's set criteria |
| trqa_validation | the dataset's rows | the extractor's set criteria |

**Criteria source** (`criteria_source`: `gold`, the default, or `silver`). A choice on TRQA and trqa_validation only,
whose queries have both; the run goes to `belief-{mode}-gold/` or `belief-{mode}-silver/`. Elsewhere it has no
effect (an info line says so) and the run goes to `belief-{mode}/`. The belief model always runs: a query without
gold criteria (FRAMES, failed texts) gets silver ones (`gold_missing` in its `criteria_info`). Each query's
`criteria_info` records `criteria_kind` (`gold` or `silver`), and a gold row's `gold_level`. `run_config.json`
records `criteria_source` and the source used, `criteria_source_effective`.

**Criteria init** (`belief_model_component/criteria/factory.py`). The criteria are part of the belief model: it makes
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
(`check_criteria_inputs`: e.g. a missing training dataset gold file stops the run).

The banks:

- The extractor's criteria (`llm_criteria`): `CRITERIA_ROOT/{dataset}/criteria_bank/{gold|silver}/criteria_{split}.jsonl`
  (gold for BrowseComp-Plus, NeuCLIR, RAGTIME; silver for TRQA; a training dataset has one bank for all its splits,
  `CRITERIA_ROOT/training_datasets/{dataset}/criteria_bank/silver/criteria_all.jsonl`), `BankedCriteriaSource`.
- TRQA's gold criteria, in `criteria_bank/gold/` next to its silver ones (key: the writer and the hash of its
  prompts). They are made from the query's intermediate info as the trqa_validation dataset makes its own
  (`annotations.multi_member_row`: one criterion per member, then the aggregate; members named by the page title of
  their first document, looked up in `wiki_partial` by id, or by TRQA's label for ecommerce); `llm_criteria` writes
  the property phrase and the aggregation (the training datasets' prompts, `training_datasets/gold/prompts/multi_member_*`). The bank line keeps
  the whole annotation row (gold answers, documents, members) under `bank_only`, never written to a run's files.
  Texts that fail for good (a member or value leaks into the phrase, after one retry) are banked as final with no
  criteria, and the query gets silver ones; a failed writer call is not banked and is asked again next run.
- A training dataset's gold criteria: its queries file is their bank, written by its build only
  (`build_training_dataset.py`), never during a run, so training scores against the same texts. A query whose gold
  texts failed gets silver ones; rebuild the dataset to fix them.

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

The belief update implements Agent-BRACE's b_t = f(query, b_{t-1}, o_t). In b_0 no criterion has a candidate. A call
(`system.txt` in `src/belief_model_component/prompts/belief_update/`, the status definitions of the shape's criterion
types in `statuses.txt`, and the shape's own `<shape>.txt`: its terms, its rules, its output notes and its example) is the same prompt for the prompted teacher and a trained student, so one training sample is one call.

**When a call is made** (`BeliefUpdateSignal.plan`). A search iteration makes a call when it has a document not
judged before (labelled `d<n>` when first shown); a judged passage is never shown again. Otherwise no call
is made (`called: false`): the belief is carried forward and the turn waits; the next call reads every turn since the
last call (the last 6), each labelled `t<iteration>`. After the last iteration, the answer turn (its reasoning and
final answer) makes the final call: on a short-answer dataset the final scorer's (below), on a report one more
update with the turns still waiting.

**Passage length.** The agent and the belief update read the same text of each document: its first
`max_passage_tokens` tokens (one shared setting, default 512 as in BrowseComp-Plus, counted with the `Qwen/Qwen3.5-4B`
tokenizer whatever the agent's model, no marker at the cut; `utils.text_utils.cut_passage`), report documents
included. A passage's title is its metadata title, else its front matter's (`title:`), else its first line.

The call reads every criterion with its current candidates, their support and its proposals, the turns and the
passages. The reply has three lists:

1. **Reasoning observations** (claims): per candidate the reasoning names, assesses or rejects, `{id, turn,
   candidate, span, claim}`: the claim is a status, or "rejected". The span must occur in its turn.
2. **Passage observations**: per passage and criterion (not a computed one), the spans of the passage that state a
   part of the criterion. Each span is checked on its own (a span joined with "..." is split first); an observation
   without a verified span is dropped.
3. **Candidates**: per candidate whose support or status changes, `{id, candidate, support, against, reason,
   status}`: the labels of the call's passages whose observations support it or speak against it, a reason opening
   with `Was <current status>:`, `Was proposed:` or `New:`, and the new status, or "removed".

The belief is kept at two levels. **Shown** to the next update: per criterion its candidates, highest status first,
each with its support, one line per passage observation (`+` supports, `-` against: the spans and the passage's title
in brackets; labels are local to one call, so they are never shown again), then `proposed:` and the names the agent
proposed that no passage supports yet. **Saved**: every observation that entered (verified spans), per criterion every
source ever assigned to it (the `references` of the `final` line), the agent's latest claim on each candidate, and
the candidate updates with their reasons (`updates`); the claims and reasons are never shown.

Code applies the reply (`BeliefState.apply`):

- **Reasoning only proposes.** A claim sets the agent's claim on a listed candidate (never its status), or adds a
  proposal (no status, not a rival, never selected; at most 8 per criterion), updates it, or drops it when the claim
  is "rejected". A proposal becomes a candidate when a passage supports it.
- **Support.** A support or against label must be a passage of the call observed on the criterion (`bad_ref`
  otherwise, the label dropped). A second observation of a passage on the criterion adds its new spans to the first. The basis of a
  candidate follows from its support ("passage", or "reasoning" for a computed criterion's candidates).
- **Hard limits** (not for a computed criterion, whose status the reasoning sets): a candidate no passage supports is
  at most "possible", and a new name without any passage of the call stays a proposal (`no_support`); a name that
  passages of the call only speak against is not added, and a proposal stays one (`against_only`); a listed
  candidate whose update adds no support keeps its status, unless in the same call another of its candidates gained
  support, was added or was removed (one answer: the status depends on the others; a proposal added or dropped does
  not count); otherwise `no_new_support`. A criterion is judged on its own parts: a part that refers to another
  criterion is met when a passage links the candidate to a candidate of that criterion, whatever that candidate's
  status, so a change of the referenced criterion moves nothing.
- **Target.** A multi-clue query's target criterion takes no observation (`derived_criterion`) and no candidate
  update (`derived_update`). After each call code derives its candidates from the clues before it: a name listed on
  every clue, at the lowest of its statuses there, with their support (shown on the clues only). A rival or a passage
  against it on one clue lowers it there.
- **Removal.** A candidate the reply does not name keeps its status. "removed" is taken when a passage of the call
  speaks against the candidate (`against`), or for a proposal a claim of the call rejected; else flagged and kept
  (`bad_removal`; `remove_unlisted` for a name neither listed nor proposed). Removed candidates and dropped proposals
  are saved in the final line's `references[].removed`; a name added again is flagged (`readded`). Names are matched
  on letters and digits only; a new name that contains a listed one, or the reverse, is flagged (`near_duplicate`).
- **Not taken**: an update of an unknown criterion, without a name (`empty_candidate`), with a status that is no
  candidate status (`bad_status`), a second one of the same candidate (`duplicate_candidate`), or of a criterion
  without an observation in the call (`unexpected_candidate`); a claim without a candidate or a valid claim
  (`bad_claim`).
- **Flagged** (statuses kept): a reason without its opening (`no_reason_opening`), a `Was X` that is not the current
  status (`wrong_previous`), `Was` on an unlisted candidate (`not_listed`), `New:` on a listed one (`listed_as_new`),
  a reason citing a source with no observation on the criterion (`bad_ref`), a new candidate in none of its
  criterion's passage spans and titles (`unknown_candidate`), no new support with the current status written again
  (`unchanged`); in a one-answer criterion, a "certain" candidate while another has support (`certain_with_rival`)
  and two candidates with the same support but different statuses (`unequal_tie`).

Derived per criterion: its status, the highest of its candidates ("unknown" without one; an aspect's its own), and
its selected candidate(s): for a one-answer criterion the first one listed at "possible" or above, for an aspect every
fact; a proposal is never selected. `changed` lists the criteria whose candidates or proposals changed. When the call
fails the belief stays as it was, its new documents are shown again when retrieved again and its turns' claims are
lost. Every problem goes into the step's `issues` list (`criteria.ISSUE_*`).

The statuses, lowest first: `unlikely` (-1), `possible` (1), `probable` (2), `almost certain` (3), `certain` (4);
"unknown" (level 0) is the status of a criterion without candidates. Each query type has one status set
(`prompts/belief_update/statuses.txt`), and each dataset's prompt holds only its own: "one answer" for
multi_clue, multi_hop and multi_member (plus the aggregate: a target for multi_clue, a computation for multi_member), "aspect" for
multi_aspect. A one-answer candidate's status combines its rivals and its own support (support is stronger when it
states more parts, or states them more directly). A candidate a passage casts doubt on is "unlikely" (one a passage
states fails a part is removed); otherwise:

| status | one answer (candidates compete) |
|---|---|
| unlikely | another candidate has stronger support |
| possible | another candidate has support as strong |
| probable | its support is the strongest, but leaves a part unstated |
| almost certain | its support is the strongest and states every part, but some only by inference, or another candidate has weaker support |
| certain | its support states every part directly, and no other candidate has support |

The definitions do not overlap, so the prompt has no tie-break rule. A candidate is what its criterion asks for, never the member a member
criterion ("<member>: <property>") is about: the member criteria only collect values, and code drops a member name
written as a candidate or proposal (`subject_candidate`). Whether a member belongs to the set is judged on the
aggregate: its passage observations state an input it needs (a member's membership, or the value of a member no
criterion lists; shown as "passages on its inputs"). Code sets an aggregate's statuses: each candidate at the lowest
status of its inputs (the closed criteria before it), "unlikely" when a passage speaks against it (it uses an input a
passage rules out, or leaves out one a passage states), at most "possible" when the reasoning computed another answer
later (`aggregate_status` when the reply wrote another status). A computed aggregate's candidates come from the
reasoning's claims only; an update never adds one (`not_computed`).

A report's aspect (multi_aspect) has one status of its own, written by the reply's item on it without a candidate:
how well its facts back what it asks, by the sides they cover and whether every fact rests on several passages:
unlikely, its facts all removed; unknown, no fact yet; possible, one side; probable, several sides (not every one) with
some fact on one passage only; almost certain, several sides with every fact on several passages, or every side with
some fact on one passage only; certain, every side with every fact on several passages. Its facts are only kept or removed. The status is the aspect's q_k; it
moves only with a passage on the aspect in the call, never with a claim alone (`aspect_no_passage`), is "unknown"
with no fact and "unlikely" when a passage removed its last facts (a proposal the agent rejected does not count,
`aspect_no_fact`), and "certain" needs every fact supported by two
documents (`aspect_one_source`: "almost certain" otherwise). A fact written with a status other than "kept" or
"removed" is kept (`bad_fact_status`). A new fact whose words contain a listed one, or the
reverse, joins it (`merged_duplicate`); a supporting span that shares no content word with its fact is flagged
(`span_off_candidate`, information only). Differences from Agent-BRACE: it ranks "unknown" lowest; here it is the
neutral level 0; its "doubtful" is a removal here. "certain" is the paper's "confirmed / certain".

The update prompt reads each criterion with its candidates (`views.tag.format_belief`):

```
c1 (one answer): the town where the Varn Folk Museum was founded
  "Osby": certain
    + "The Varn Folk Museum was founded in Osby in 1921" [Osby]
c2 (one answer): the engineer who was born in the town of c1
  "Henrik Dahl": certain
    + "Henrik Dahl (1850-1921) was a Norland civil engineer, born in Osby" [Henrik Dahl]
c3 (one answer): the lighthouse that was designed by the engineer of c2
  (no candidate)
  proposed: "Kalvik Lighthouse"
```

The certainty score s_t (`belief_model_component/core/certainty.py`) is a fixed function of the statuses:

1. Each criterion's status (the highest of its candidates) becomes a probability, `core.types.STATUS_PROBABILITY`
   (unlikely 0.10, unknown 0.25, possible 0.50, probable 0.70, almost certain 0.90, certain 0.97): q_k. The same table
   turns every status into a number, the final scorer's included. The status of a one-answer candidate already counts
   its rivals (two tied ones are at most "possible", 0.5), so the candidates share nothing.
2. On a short-answer query s_t is the lowest q_k (the weakest link: the answer is correct only when every criterion
   holds, and the minimum is the upper bound of that conjunction); it reads only the order of the statuses, so its
   AUROC does not depend on the table's values. On a report (every criterion an aspect) s_t is the mean q_k, the
   expected share of aspects covered. Every criterion counts: an aggregate's status is the lowest of its inputs', so
   under the minimum it adds only what they do not show (a target joins the clues; a computed answer a passage speaks
   against is "unlikely").

s_T is compared with the correctness of the agent's final answer. On a short-answer dataset it is the final
scorer's (`belief_model_component/update/final_score.py`, prompt in `prompts/final_score/`): one call reads the query,
the belief before the answer turn (candidates, statuses, support, proposals and the names a passage ruled out)
and the answer turn (its last 2000 characters of reasoning, shown only to tell which candidates the answer relies on,
and the final answer), and writes per criterion the answer's candidate, a reason and the status of the final answer, on the belief's scale and read as a candidate's
status is (unlikely 0.10, possible 0.50, probable 0.70, almost certain 0.90, certain 0.97). The prompt shares the belief
update's definitions and status definitions: the final answer is judged as a candidate of the query, the criteria
weighed by how much each identifies the answer.
The fixed function of the final statuses is kept next to it as a baseline (`certainty_formula`), with its form
anchored on the answer's candidates (`certainty_answer_formula`: a candidate's status, "unlikely" for a name a
passage ruled out, "unknown" for a proposal or an unlisted name). On a report, s_T is the fixed function of the
belief after the final update. `analysis/final_score_ablation.py` rescores saved runs: the scorer with and without
the belief, and both formulas.

Saved in `belief/{qid}.jsonl` (schema in `src/evaluation/belief/writer.py`): a meta line, one line per iteration and
one `final` line for the answer turn. A step line has the retrieval measures, `certainty`, `called`, the documents
(`{doc_id, new, label}`), every criterion's belief after the step (`beliefs`: `{id, status, selected,
candidates, proposals}`, each candidate with its `support` and the agent's `claim`), the candidate `updates` of the
reply (with their reasons and support labels), the criteria it `changed`, the `observations` that entered (claims and
passage observations, with their verified spans), the reply's `issues`, the agent turn and the raw reply. The `final` line has the same belief fields plus `references` (per
criterion every source assigned to it over the sample, and the candidates and proposals dropped from it); its `certainty` is s_T,
with `certainty_label`, `certainty_formula`, `certainty_answer_formula` and `final_score` (status, answer's candidates, reason, issues). The same belief is readable in the
trajectory md (`trajectory/{qid}.md`): the criteria before the first step, a belief table after each iteration and
the final belief before the result. For evaluation only, never shown to the agent:

- `new_item_precision` and `recall_so_far`, against the qrels at the dataset's `min_relevance_score` (NeuCLIR and
  RAGTIME: grade 3, the very valuable docs), the same relevant docs as the end-of-run Recall@N.
- Intermediate answers (BrowseComp-Plus and TRQA, all agents except `cpm_report`): after each iteration the agent's own
  model is asked, from its trajectory so far, for its most likely answer(s) in its answer format; it may give one,
  several or none (`IntermediateAnswerSignal`, prompt in `src/belief_model_component/prompts/`). They are saved as
  is, not evaluated.

### Inform mode

`--belief-mode inform` computes the same belief and also shows it to the agent: right after each iteration's search
results, one `<belief_state>` tag is appended to the trajectory (`src/belief_model_component/views/tag.py`):

```xml
<belief_state step="2">
  <criteria>
    <k1>the town where the Varn Folk Museum was founded
      "Osby": certain
    </k1>
    <k2>the engineer who was born in the town of k1
      "Henrik Dahl": certain
    </k2>
    <k3>the lighthouse that was designed by the engineer of k2
      unverified: "Kalvik Lighthouse"
    </k3>
  </criteria>
</belief_state>
```

The tag is descriptive only: each criterion with its candidates and statuses (no support), and the names the agent
proposed as `unverified`; a criterion id in a criterion's text is written with the tag's own id ("of k1"). Gold-based fields
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

A run is evaluated from its saved files, both at its end and with `--eval-only`, so the two write the same files.
Belief model training's phase 1 is a dra_inference run, so its runs get the same evaluation. Judge verdicts are
reused for queries whose input is unchanged, so `--eval-only` on an unchanged run makes no LLM call and needs no GPU.

The measures, by group, in the order of the files and the log:

| Group | Measures | Datasets |
|---|---|---|
| retrieval | Recall@N, Precision@N of the seen docs (N = the distinct docs shown to the agent), N | all |
| generation | accuracy (LLM judge, `--judge-model`; TRQA: numeric exact match), correct, evaluated, judge errors | QA |
| | Auto-ARGUE nugget coverage and F1 (NeuCLIR: also the weighted ones), reports evaluated, without nuggets, empty, without citations, with unmapped citations, judge failures | reports |
| uncertainty | queries scored; s_T and verbal: AUROC [CI], Brier; formula and answer formula: AUROC [CI] | QA, belief on |
| | queries scored; s_T Kendall tau-b [CI] and Brier vs nugget coverage (and weighted) | reports, belief on |
| belief | gold and control event change, criterion recall, answer reached, status Brier; status calibration table (below) | TRQA, trqa_validation, infoseek, belief on |
| trajectory | searches, agent tokens, belief model tokens (per query, means), forced answers, max iterations reached | all |

CIs are 95% percentile bootstrap intervals over queries (1000 resamples, `evaluation.metrics`). Nothing else is
computed: no @k tables, NDCG, MAP, GradedRecall, cited-doc or fusion metrics, soft match or length statistics (the
raw outputs stay in the run, for `analysis/`).

**Uncertainty** (`src/evaluation/uncertainty/`). y is the correctness of the final answer (a judge error leaves the
query out). The scores: `s_T` (the final scorer, the headline), `verbal` (the agent's own `Confidence: X%`, glm and
oss; read from `generation/{qid}.md` by a regex, `answer/confidence.py`, "~85 %" and a fraction such as "0.85"
accepted; a generation without it has none), and the ablations `formula` (the fixed function of the final statuses:
the min of the criteria's status probabilities) and `answer formula` (the same on the candidates the scorer says the
answer uses). All are measured on the same queries: those with y, s_T and, when the run has any, a verbal
confidence. Rows: AUROC [CI] for every score, Brier for s_T and verbal. On a report dataset (`uncertainty/report.py`)
s_T is the certainty of the final belief (the mean of the aspects' status probabilities; a report has no
final scorer) and y the report's Auto-ARGUE nugget coverage: Kendall's tau-b [CI] (the ranking) and Brier (s_T as the
expected coverage); an empty report is left out. Left-out queries are logged.

**Belief** (`src/evaluation/belief/evaluator.py`), on the queries whose criteria are the gold criteria (TRQA members,
InfoSeek hops; not WebShaper or FRAMES, which have no gold per criterion). Scored criteria: a member or hop (gold
answer and documents) and an aggregate (gold answer only).

- A **gold event** is a gold document of criterion k, new to the belief, whose text as the update read it (the
  `max_passage_tokens` cut) states k's gold answer (string match, else the judge). A **control event** is a new
  document observed on k that is not one of its gold documents and does not state its answer.
- `gold event: gold candidate change`: the change of the gold candidate's status probability at the criterion's
  first gold event (unknown -> certain = 0.25 -> 0.97 = +0.72; a gold answer that is not a candidate is unknown);
  `control event: ...`: the same at its control events, which should be about 0. An event whose gold candidate is
  already "certain" is left out.
- `criterion recall`: the share of the criterion's gold documents the agent saw.
- `reached answer`: at the end, the criterion's selected candidate is a gold answer; `reached answer, gold seen`: the same over members and hops with a gold document seen (the belief's own
  share).
- `status Brier`: the final status probability against reached.

Each row comes twice: per query (each query's mean over its criteria, then the mean over queries, CI over queries)
and `(c)`, the mean over every criterion. Below the table, the status calibration: per final status, how many
criteria had it and how many reached the answer. A candidate is a gold answer when it matches one, else when the
judge (`judge_model`) says so, cached in `belief_eval/judge_cache.jsonl`. For diagnosis, `belief_eval/events.csv`
has every event with a gold event's funnel stage (`call_failed`, `not_assigned`, `no_gold_candidate`,
`not_selected`, `correct_lost`, `correct_kept`), and `per_query.csv` has each query's values and `belief_correct`
(every member or hop reached).

The terminal log prints one block, one line per group:

```
EVALUATION  trqa / wiki2_test / glm / monitor (gold criteria)  (n=200)
retrieval    Recall@N 0.412 | Precision@N 0.081 (N 38.5)
generation   accuracy 0.310 (62/200)
uncertainty  s_T AUROC 0.780 [0.71,0.84] Brier 0.190 | verbal AUROC 0.610 [0.53,0.69] Brier 0.310 (n 188)
belief       gold change 0.620 | control change 0.030 | recall 0.710 | reached 0.520 (gold seen 0.730) | status Brier 0.180 (n 200)
trajectory   searches 9.2 | agent tok 41234 | belief tok 23102 | forced 4
```

Every run can be compared in one table with `analysis/collect_results.py --root run_outputs` (one row per run, the
headline measures as columns; `--all` for every measure, `--group` to keep some groups, `--ci` for the CIs):
`results.md` and `results.csv`.

`src/evaluation/` follows the groups; each group also writes its per-query files:

```
src/evaluation/
├── runner.py        save_query_outputs, build_evaluators, load_run_results, evaluate_and_save
├── report.py        metrics.md, per_query.csv and the log block
├── metrics.py       AUROC, Brier, ECE, tau-b, bootstrap CIs (shared with belief model training)
├── retrieval/       surfaced / seen / cited doc levels and their TREC files; seen-doc Recall@N, Precision@N
├── answer/          AccuracyEvaluator (LLM judge), NumericMatchEvaluator (TRQA), ArgueReportEvaluator (reports),
│                    verbal_confidence
├── uncertainty/     UncertaintyEvaluator, ReportUncertaintyEvaluator
├── belief/          save_belief (belief/{qid}.jsonl schema), BeliefUpdateEvaluator, gold match and judge
│                    (shared with belief model training)
├── trajectory/      TrajectoryEvaluator (per query), save_trajectory
├── generation/      the generation/{qid}.md writer
├── gold/            per-criterion gold (TRQA gold bank, training datasets); TRQA gold entities and their matcher
├── judge.py         LLM-judge client and DEFAULT_JUDGE_MODEL
└── common.py        file and statistics helpers
```

### Report evaluation (Auto-ARGUE)

On the report datasets (`report_eval: "argue"` in `layout.DATASET_SPECS`: NeuCLIR, RAGTIME), every run and every
`--eval-only` pass scores the reports against the nuggets with the official
[Auto-ARGUE](https://github.com/hltcoe/auto-argue) package (`src/evaluation/answer/argue.py`). There is no flag. The
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
│   └── cited/{qid}.trec            docs cited by the LLM
├── generation/{qid}.md             the final generation (with its References section)
├── belief/{qid}.jsonl              belief on only: meta line (criteria, final belief) + one line per search iteration
├── trajectory/{qid}.jsonl          one line per step (queries, seen doc ids, <belief_state> tags, per-turn
│   │                               diagnostics), then the meta line (counts, agent data)
│   └── {qid}.md                    the same trajectory, human-readable, written live
├── accuracy.jsonl                  per-query verdicts (datasets with answers)
├── report_eval/                    Auto-ARGUE inputs (nuggets/, cited docs), cached judgments/, per-query scores.tsv
├── belief_eval/                    belief update evaluation: events.csv (one row per gold or control event),
│                                   judge_cache.jsonl
├── metrics.md                      every measure: group | metric | value | ci_low | ci_high | n
└── per_query.csv                   every per-query value of every group
```

`{agent}` is the `--policy-model` name (`oss_20b`, not the internal agent `oss`), as is `agent.policy_model` in
`run_config.json`. `_{query_key}` is in the name only when it is not the dataset's default. Each fact has one file: the run settings are
in `run_config.json` only, the final generation in `generation/{qid}.md` only, the aggregates in `metrics.md` only.
`per_query.csv` repeats per-query values of the other files (correctness, s_T, verbal confidence, tokens): it is
evaluation output, rebuilt by every evaluation and never read by inference or training. An evaluation removes the
`summary.json` of earlier code.

A query is finished when its `trajectory/{qid}.jsonl` ends with the meta line, which is written after every other file
of the query (and, with the belief model on, after its belief file).
