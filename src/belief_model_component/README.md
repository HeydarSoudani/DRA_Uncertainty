# Belief model component

A prompted belief state over each query's criteria, plus per-step signals,
for deep research agents. It plugs into any agent with `--belief-mode monitor`
(observe only) or `inform` (also inject a `<belief_state>` tag into the
trajectory).

## Layout

```
belief_model_component/
  __init__.py        public API: code outside the package imports from here
  belief_model.py    BeliefModel, the per-sample entry point
  core/              the belief itself
    types.py         statuses, criterion kinds, query shapes, dataclasses
    issues.py        issue codes of one update, and the sets training reads
    reason_text.py   the opening of a candidate update's reason, its labels
    state.py         BeliefState (b_t) and its apply rules
    certainty.py     certainty s_t of a belief
  update/            the LLM calls on the belief
    updater.py       LLMBeliefUpdater: prompt, reply parsing
    signal.py        BeliefUpdateSignal, CallPlan: when to call, applying the reply
    final_score.py   LLMFinalScorer: s_T from b_{T-1} and the answer turn
    repair.py        one JSON-replying call with its parse retries
  criteria/          the criteria C of a query, made once per sample
    sources.py       LLM, banked, gold and fallback criteria sources
    parsing.py       extractor reply parsing, criterion kinds, set layout
    factory.py       the source of a dataset and criteria_source; prepare
  text/              plain string helpers, no belief logic
    matching.py      letter-and-digit matching (alnum, contains_words)
    passages.py      span checks, passage titles, reply labels, Passage, Turn
    agent_turn.py    the agent's turn as the update reads it
  views/             text renderings of the belief
    tag.py           the belief as the update and final scorer prompts read it, the <belief_state> tag
    log_view.py      readable console and markdown logs
  eval_signals/      evaluation only, never shown to the agent
    retrieval_gain.py
    intermediate_answer.py
  prompts/           system prompts (.txt) and user templates
```

Imports between the folders go one way: `text` -> `core` -> `prompts` ->
`views` -> `update` -> `belief_model` (`core` reads `text.matching`, so
`text` is the lowest layer); `criteria` reads `core.types` and `prompts`,
`eval_signals` reads `prompts`.

## The belief

The belief state b_t gives each criterion of a fixed per-query criteria list
its candidates (possible answers), each with an Agent-BRACE belief status
(how likely it is an answer) and its support (the passage observations that
support it or speak against it), and the names the agent proposed that no
passage supports yet. Each query type has one status set
(`prompts/belief_update/criterion_statuses.txt`): "one answer" for a
short-answer query (every closed criterion: clues, hops, members' values),
"aspect" for a report, where the status is the aspect's own (how well its
facts with support back it; the facts are only kept or removed). An
aggregate (a multi-clue query's target, a multi-member query's
computation) has no status set: code sets its statuses, the lowest of its
inputs', "unlikely" when a passage speaks against a candidate.

A step with a new passage makes one call (b_t = f(query, b_{t-1}, o_t)). It
records the claims of the agent's turns since the last call (reasoning
observations: they only propose or reject names) and the spans of the
passages that state a part of each criterion (passage observations, kept as
the criterion's references), then gives, with the passages that support or
speak against it and its reason, the status of every candidate they change.
Code keeps every other candidate as it was and enforces the hard limits (no
support: at most "possible"). Other steps carry the belief forward.

The candidates, their support and the proposals are shown to the next
update (the agent sees the candidates and proposals); the claims and reasons
are saved. The certainty s_t of each step is a fixed function of the
statuses. On a short-answer query, the final scorer reads b_{T-1} and the
agent's answer turn and judges how likely the answer is correct (s_T, a word
of estimative probability); on a report the answer turn is one more update
and s_T the certainty of the final belief. For evaluation only, it also
saves the retrieval gain against the qrels and the agent's intermediate
answers.

## Per sample

```
reset(query_id, query)   the criteria list C is created once (fixed for the
                         sample); b_0 has no candidate ("unknown")
observe(...)             called once at the end of every search iteration
                         with the iteration's queries and seen documents;
                         returns and stores the step record
finish(answer_turn)      once, after the last step: the final scorer on the
                         agent's answer turn (short-answer queries; a report:
                         one more update); stores the final record
```

With `inform=False` (`monitor`) the model never changes the trajectory; with
`inform=True` (`inform`) each record also carries `belief_state_tag`, the
`<belief_state>` tag the agent appends to its context (`views.tag`).

Order inside `observe` (the step's documents and the agent's turn move the
belief from b_{t-1} to b_t): b_t, then the retrieval gain, then the
intermediate answer (both for evaluation only). When the step has a document
not judged in earlier steps, one belief update call reads the agent's turns
since the last call and those passages, records the claims of the turns and
the observations of the passages, and gives the support and status of every
candidate they change, with its reason. Otherwise b_t = b_{t-1} and the turn
waits for the next call.

## Saved records

### Step record

One per iteration, saved as one `step` line of `belief/{query_id}.jsonl` by
`evaluation.belief.save_belief`; flat scalars first, nested detail last:

```
iteration, agent_iteration,
num_subqueries, num_docs, num_new_docs,
new_item_precision,
num_new_relevant, num_repeated_relevant, num_irrelevant,
recall_so_far, num_relevant_seen, num_relevant,
certainty, called,
intermediate_answers, intermediate_answer_status,
docs[], beliefs[], updates[], changed[], observations[], issues[],
agent_turn, reply, intermediate_answer_reasoning, errors[],
belief_state_tag            # inform only, not saved (the trajectory step has it)
```

- `iteration` counts the observed search iterations from 1, the same for
  every agent; `agent_iteration` is the agent's own counter, whose base and
  meaning differ per agent.
- A value that could not be computed is null, never 0; the reason is in
  `errors` (or a component is not configured: no criteria, no qrels).
- `docs`: per document of the step, `{doc_id, new, label}`. `new` when its
  id was not shown in an earlier step (`num_new_docs` counts them); `label`
  the belief's label of it ("d<n>", given when the document is first shown
  to the belief update; null for a document without text).
- `called`: whether the step made a belief update call.
- `beliefs`: per criterion in `criteria` order, its belief after the step,
  `{id, status, selected, candidates, proposals}`.
  - `candidates`: as shown to the next update, highest status first, each
    `{name, status, basis, added_step, support, claim, claim_step}`.
    `support` holds the passage observations on it,
    `{label, doc_id, title, step, spans, against}`; `claim` is the agent's
    latest claim on it, never shown to the update.
  - `proposals`: the names the agent proposed that no passage supports, each
    `{name, claim, step, added_step}` (`step` its latest claim's).
  - `status` and `selected` are derived from the candidates: the
    criterion's status on the scale `core.types.BELIEF_STATUSES` and its
    selected candidate(s) (`BeliefState.belief_lists`); an aspect's
    `status` is its own, its facts' `status` "kept".
- `updates`: the reply's candidate updates,
  `{id, candidate, reason, support, against, written, previous, status}`.
  The reason is saved here only, never shown; `status` is the one code kept:
  "removed" for a removed candidate, "proposed" for a name kept as a
  proposal, null when the update was not taken; an aspect's status item has
  `candidate` null.
- `changed`: the ids of the criteria whose candidates or proposals changed.
- `observations`: the observations that entered the belief in the call
  (claims `{id, label, step, candidate, claim, spans}`, passage observations
  `{id, label, step, doc_id, spans}`, verified spans only), null when the
  call was not made or failed. Together over the steps they are every
  criterion's references (the final record gathers them).
- `issues`: one `{id, code, detail, [label]}` per thing the reply got wrong
  (`core.issues`: an observation left out, a candidate update not taken,
  corrected or flagged).
- `agent_turn`: the agent's reasoning of the iteration
  (`text.agent_turn.agent_turn_text`; the queries are in the trajectory),
  read by the step's call or by the next one. `reply`: the raw reply of the
  call.
- `certainty`: the certainty score s_t of the belief after the step
  (`core.certainty.certainty`: over the criteria their status's probability,
  `core.types.STATUS_PROBABILITY`; the lowest on a short-answer query, the
  mean on a report; every criterion counted); null without criteria.
- `intermediate_answers`: null when not configured or failed, `[]` when the
  model gave no answer. `intermediate_answer_status` says which: `ok`,
  `no_candidate` (`[]`), `unparsed` (reply without answer format), `failed`
  (call raised; see `errors`) or `disabled` (not configured, e.g.
  `--add-intermediate-answer false`).

### Final record

One `final` line; it reads the answer turn (`text.agent_turn.answer_turn`):

```
iteration, certainty, certainty_label, certainty_formula,
certainty_answer_formula, called, final_score, beliefs, updates, changed,
observations, issues, references, agent_turn, reply, errors
```

- Short-answer query: the final scorer (`update.final_score`) reads b_{T-1}
  and the answer turn, and the belief is not updated. `certainty` is s_T, the
  probability of the final answer's status (`certainty_label`), null when the
  call failed; `final_score` holds its `{status, candidates, reason, issues}`.
  `certainty_formula` is the certainty of b_{T-1} (`core.certainty.certainty`)
  and `certainty_answer_formula` its form anchored on the answer's candidates
  (`core.certainty.answer_certainty`), both baselines. `updates`, `changed`
  and `issues` are empty, `observations` null.
- Report (no final scorer): the answer turn is one more update, labelled
  "t<last step + 1>" with the turns still waiting, and s_T is the certainty
  of the belief after it (`certainty` = `certainty_formula`;
  `certainty_label`, `certainty_answer_formula` and `final_score` null).
- `reply`: the raw reply of the call.
- `references`: per criterion every source assigned to it over the sample
  and the candidates and proposals it dropped (`BeliefState.reference_lists`:
  `{id, references: [{label, step, [doc_id, title | candidate, claim],
  spans}], removed: [{name, status, basis, added_step, removed_step,
  reason, against}]}`; `against`: ruled out, by a passage or by the agent's
  claim, which the final scorer reads as removed; false for a proposal
  dropped for room).
- The final record is null without an answer turn.
