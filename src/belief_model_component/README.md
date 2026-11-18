# Belief model component

A prompted belief state over each query's beliefs, plus per-step signals,
for deep research agents. It runs in the belief stage of
`experiments/inference/dra_inference.py` (`--stage belief`,
`src/orchestration/belief_stage.py`), over the saved trajectories of any
agent, and never changes what the agent does.

## Layout

```
belief_model_component/
  __init__.py        public API: code outside the package imports from here
  belief_model.py    BeliefModel, the per-sample entry point
  core/              the belief itself
    types.py         statuses, belief kinds, query shapes, dataclasses
    issues.py        the issue code table (effect, role) and the sets by role
    state.py         BeliefState (b_t) and its apply rules
    certainty.py     certainty s_t of a belief
  update/            the LLM calls on the belief
    updater.py       LLMBeliefUpdater: prompt, reply parsing
    signal.py        BeliefUpdateSignal, CallPlan: when to call, applying the reply
    final_score.py   LLMFinalScorer: s_T from b_{T-1} and the answer turn
    repair.py        one JSON-replying call with its parse retries
  init/              the beliefs of a query, made once per sample
    sources.py       LLM, banked, gold and fallback belief sources
    parsing.py       extractor reply parsing, belief kinds, set layout
    factory.py       the source of a dataset and belief_source; prepare
  text/              plain string helpers, no belief logic
    matching.py      letter-and-digit matching (alnum, contains_words)
    passages.py      span checks, passage titles, passage labels, Passage, Turn
    agent_turn.py    the agent's turn as the update reads it
  views/             text renderings of the belief
    belief_text.py   the belief as the update and final scorer prompts read it
    log_view.py      readable console and markdown logs
  eval_signals/      evaluation only, never shown to the agent
    retrieval_gain.py
  prompts/           one folder per call (belief_init, belief_update, belief_score): system.txt, user.txt, <shape>.txt; shared/wep_definitions.txt
```

Imports between the folders go one way: `text` -> `core` -> `prompts` ->
`views` -> `update` -> `belief_model` (`core` reads `text.matching`, so
`text` is the lowest layer); `beliefs` reads `core.types` and `prompts`.

## The belief

The belief state b_t gives each belief (an item "B<k>" of a fixed
per-query list) its candidates
(possible answers), the passages on each (+ it states a part for the
candidate, - it states the candidate fails a part), its most probable
candidate and its word of estimative probability (WEP,
`prompts/shared/wep_definitions.txt`), both judged by the update from all
the belief's passages, and the names the agent proposed that no passage
supports yet. A belief has one WEP: its most probable candidate carries
it, its other candidates have no status. On a report's aspect the
candidates are facts, the WEP rates how well the kept facts cover the
aspect, and a - removes a fact. An aggregate has no WEP of its own and
takes no belief item: a multi-clue query's target has as candidate the
name that is the most probable candidate of every clue, at the lowest of
their WEPs; a multi-member query's computed answer keeps the agent's
computed answers (saved, never shown: the update reads it as "unknown"
with the passages on its inputs), each at the lowest status of the
inputs, "unlikely" when a passage speaks against it.

The agent's reasoning only proposes: a claim ("rejected" included) is
saved on its candidate and never removes one; a rejected proposal leaves
the proposals. Only a `-` passage on an aspect's fact removes it.

A step with a passage (new or shown before) makes one call (b_t = f(query, b_{t-1}, o_t)). It
records the claims of the agent's turns since the last call (reasoning
observations: they only propose or reject names), the passage
observations (a passage, a belief, a candidate and a stance, with the
spans that state it; attached to the candidate after the claims, and kept
as the belief's references), then per belief with a new passage
observation its most probable candidate, WEP and reason. Code checks that
the most probable candidate is listed with a + passage and that "unknown"
goes with no supported candidate, and takes the WEP as written. Other
steps carry the belief forward.

Each belief's statement (built by code from its WEP and most probable
candidate), its passages (per candidate and stance the latest
`views.belief_text.MAX_SUPPORT_LINES`, the earlier ones counted) and its
proposals are shown to the next update and, before the answer turn, to
the final scorer; the claims are saved. The certainty s_t of each step is a fixed function of the
statuses. On a short-answer query, the final scorer reads b_{T-1} and the
agent's final turn and judges how likely the answer is correct (s_T, a word
of estimative probability, with a reason); on a report the answer turn is
one more update and s_T the certainty of the final belief. For evaluation
only, it also saves the retrieval gain against the qrels.

## Per sample

```
reset(query_id, query)   the belief list C is created once (fixed for the
                         sample); b_0 has no candidate ("unknown")
observe(...)             called once per search iteration of the saved
                         trajectory with the iteration's queries, seen
                         documents and the agent's turn; returns and stores
                         the step record
finish(answer_turn)      once, after the last step: the final scorer on the
                         agent's answer turn (short-answer queries; a report:
                         one more update); stores the final record
```

Order inside `observe` (the step's documents and the agent's turn move the
belief from b_{t-1} to b_t): b_t, then the retrieval gain (for evaluation
only). When the step has a document
(new or shown before, with the same label), one belief update call reads the agent's turns
since the last call and those passages, records the claims of the turns and
the passage observations, and gives the most probable candidate and WEP of
every belief with a new passage observation. Otherwise b_t = b_{t-1} and the turn
waits for the next call.

## Saved records

### Step record

One per iteration, saved as one `step` line of `belief/{model}/{query_id}.jsonl` by
`utils.run_files.save_belief`; flat scalars first, nested detail last:

```
iteration, agent_iteration,
num_subqueries, num_docs, num_new_docs,
new_item_precision,
num_new_relevant, num_repeated_relevant, num_irrelevant,
recall_so_far, num_relevant_seen, num_relevant,
certainty, called,
docs[], beliefs[], updates[], changed[], observations[], issues[], notes[],
agent_turn, reply, errors[]
```

- `iteration` counts the observed search iterations from 1, the same for
  every agent; `agent_iteration` is the agent's own counter, whose base and
  meaning differ per agent.
- `num_subqueries`: the iteration's search queries, as the trajectory saves
  them (`cpm_report` saves one per search, its keywords joined).
- A value that could not be computed is null, never 0; the reason is in
  `errors` (or a component is not configured: no beliefs, no qrels).
- `docs`: per document of the step, `{doc_id, new, label}`. `new` when its
  id was not shown in an earlier step (`num_new_docs` counts them); `label`
  the belief's label of it ("d<n>", given when the document is first shown
  to the belief update; null for a document without text).
- `called`: whether the step made a belief update call.
- `beliefs`: per belief in `beliefs` order, its belief after the step,
  `{id, status, most_probable, selected, candidates, proposals}`.
  - `candidates`: as shown to the next update, most probable first, each
    `{name, status, basis, added_step, support, claim, claim_step}`
    (`status` null for a candidate that is not the most probable).
    `support` holds the passage observations on it,
    `{label, doc_id, title, step, spans, against}`; `claim` is the agent's
    latest claim on it, never shown to the update.
  - `proposals`: the names the agent proposed that no passage supports, each
    `{name, claim, step, added_step}` (`step` its latest claim's).
  - `status` is the belief's WEP on the scale `core.types.BELIEF_STATUSES`
    (an aggregate's: its highest candidate's), `most_probable` its most
    probable candidate (null for an aspect or aggregate), and `selected` its selected candidate(s) (`BeliefState.belief_lists`); an
    aspect's facts carry its WEP as their `status`.
- `updates`: the reply's belief items,
  `{id, reason, passages, written, previous, most_probable, status}`:
  `passages` the call's passage observations attached on the belief
  (`{candidate, stance, passage}`), `written` the item's
  `{most_probable, wep}` as the reply wrote them, `previous` the belief's
  status before; `status` null when the item was not taken.
- `changed`: the ids of the beliefs whose candidates, proposals, most
  probable candidate or WEP changed.
- `observations`: the observations that entered the belief in the call
  (claims `{id, label, step, candidate, claim, spans}`, passage observations
  `{id, label, step, doc_id, candidate, stance, spans}`, verified spans
  only; a null candidate on a computed answer states an input), null when the
  call was not made or failed. Together over the steps they are every
  belief's references (the final record gathers them).
- `issues`: one `{id, code, detail, [label]}` per thing the reply got wrong
  (`core.issues.ISSUES`: each code's effect, an observation or update
  dropped, a status corrected or a flaw flagged, and its training role).
  `notes`: the same records for what is no error (codes of effect `note`:
  a name written again after it was dropped, an aspect fact's span off the
  fact).
- `agent_turn`: the agent's reasoning of the iteration
  (`text.agent_turn.agent_turn_text`; the queries are in the trajectory),
  read by the step's call or by the next one. `reply`: the raw reply of the
  call.
- `certainty`: the certainty score s_t of the belief after the step
  (`core.certainty.certainty`: over the beliefs their status's probability,
  `core.types.STATUS_PROBABILITY`; their mean, aggregates left out); null
  without beliefs.

### Final record

One `final` line; it reads the answer turn (`text.agent_turn.answer_turn`):

```
iteration, certainty, certainty_label, certainty_formula, called,
final_score, beliefs, updates, changed, observations, issues, notes,
references, agent_turn, reply, errors
```

- Short-answer query: the final scorer (`update.final_score`) reads b_{T-1}
  and the answer turn, and the belief is not updated. `certainty` is s_T, the
  probability of the WEP the scorer gives the final answer
  (`certainty_label`), null when the call failed; `final_score` holds its
  `{status, reason}`. `certainty_formula` is the certainty of b_{T-1}
  (`core.certainty.certainty`), a baseline. `updates`, `changed`
  `issues` and `notes` are empty, `observations` null.
- Report (no final scorer): the answer turn is one more update, labelled
  "t<last step + 1>" with the turns still waiting, and s_T is the certainty
  of the belief after it (`certainty` = `certainty_formula`;
  `certainty_label` and `final_score` null).
- `reply`: the raw reply of the call.
- `references`: per belief every source assigned to it over the sample
  and the candidates and proposals it dropped (`BeliefState.reference_lists`:
  `{id, references: [{label, step, [doc_id, title | candidate, claim],
  spans}], removed: [{name, status, basis, added_step, removed_step,
  reason}]}`: an aspect's fact a passage spoke against, a proposal the
  agent rejected or one dropped for room; saved, never shown).
- The final record is null without an answer turn.
