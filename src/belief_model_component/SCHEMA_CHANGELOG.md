# Belief file schema history

How the per-query belief file (`belief/{model}/{query_id}.jsonl`, written by
`utils.run_files.belief.save_belief`) changed from one `schema_version` to the next,
newest first. The current schema is described in [README.md](README.md) ("Saved records")
and in the docstring of `src/utils/run_files/belief.py`.

## 57 (was 56)

A report makes no final update (with no
passage it could only change proposals, never s_T): its file has no
`final_update` step line, and its `final` line's `beliefs` and s_T
are those of the last search step.  Short-answer files are unchanged.

## 56 (was 55)

A claim is "keep" or "drop" (was a WEP or
"rejected"); a belief computed from others (a target, a multi-member
computed answer) takes no observation, so a computed answer has no
candidates and no passages on its inputs (issues `observation_on_aggregate`
and `update_on_aggregate` replace the target and computed codes;
`answer_not_computed` and `claim_unknown` are gone).

## 55 (was 54)

The final scorer is split in two calls.  The
final belief update reads the answer turn's reasoning without the final
answer (and the turns still waiting) and is saved as one more `step`
line, `final_update` true (`agent_iteration` null, no documents, the
retrieval gain null); every step line has `final_update`.  The final
scorer reads b_T and the final answer only (no reasoning): the `final`
line has no `updates`, `changed`, `observations`, `issues` or
`notes`; its `beliefs` is b_T, `certainty_formula` the certainty of
b_T, `agent_turn` the final answer, `reply` the scorer's reply; on a
report `called` is false.  `num_iterations` counts the search
iterations only.

## 54 (was 53)

"criteria" is renamed "beliefs" everywhere,
same content: the meta line's `criteria`, `criteria_info` and
`num_criteria` are `beliefs`, `belief_info` and `num_beliefs`;
`criteria_kind` (gold|silver) in it is `belief_source`; belief ids
are "B<k>" (were saved "c<k>"); the issue code `unknown_criterion` is
`unknown_belief`.

## 53 (was 52)

The belief update and final scorer follow the
finalized prompts (`prompts/belief_*`).  The update's passage
observations carry their candidate and stance (`observations`: a
passage item has `candidate` and `stance`; null candidate on a
computed answer); its belief items are `{id, reason, most_probable,
wep}`, the WEP judged by the model (no n+/n count: `beliefs` has no
`count`; `updates` items are `{id, reason, passages, written,
previous, most_probable, status}`).  A belief has one WEP: a candidate
other than the most probable has status null.  A multi-clue target's
candidate is the name that is the most probable candidate of every clue;
a computed answer is shown as "unknown" with the passages on its inputs.
Issue codes: `count_mismatch`, `wep_mismatch`,
`ref_without_observation` and the scorer's `answer_*` codes are gone;
new `update_on_computed`, `belief_missing`, `wep_without_support`,
`unknown_with_support`.  The final scorer replies `{reason, wep}`:
`final_score` is `{status, reason}`; `certainty_answer_formula` is
gone.  The agent's "rejected" claim removes no candidate (a computed
answer stays listed with the claim; a rejected proposal leaves the
proposals), so `references[].removed` records have no `against` and
the final scorer reads no `removed:` line.  The update and scorer show
per candidate and stance the latest 5 passage lines.

## 52 (was 51)

The belief model runs over saved trajectories
(the belief stage) and no longer asks the agent for intermediate answers:
the step fields `intermediate_answers`, `intermediate_answer_status` and
`intermediate_answer_reasoning` are gone.  The belief and its updates are
unchanged.

## 51 (was 50)

The issue codes are one table
(`belief_model_component.core.issues.ISSUES`: each code's effect,
dropped / corrected / flagged / note, and its training role) and renamed.  A new `notes` list per step and final record holds the
codes that are no error (`candidate_readded`, `update_unchanged`,
`span_off_fact`; before: in `issues`).  One issue per update for its
labels and its reason's sources with no observation on the belief
(before: two); `aspect_one_source` is `confirmed_one_source`,
`aspect_no_passage` is `status_without_new_support` and `not_listed`
is `reason_previous_wrong`.  The final scorer's `final_score.issues`
are issue records (`{id, code, detail}`) in place of strings.  The belief,
its updates and the training sets of codes are unchanged.

## 50 (was 49)

Same record fields; the top status is renamed
"certain" -> "confirmed" (Agent-BRACE's label; `parse_status` reads
"certain" of older files as "confirmed").  Its meaning changes:
"confirmed" needs passages from several sources (documents), each
stating every part directly, with no rival support (Admiralty Code,
"confirmed by other independent sources"); one source is at most "almost
certain", enforced by code for one-answer candidates (issue
`one_source`) as for aspects (`aspect_one_source`).  A source is a
page: passages with the same title are one source (`Support.source`;
the chunks of one InfoSeek or TRQA page have their own doc ids).  The flag
`certain_with_rival` is `confirmed_with_rival`.  Every step with a
document makes a call, and the call shows every document of the step,
also one shown in an earlier step, with the label it got then (before: only
documents not judged yet); a passage shown again moves no status on its
own: a one-answer change needs new support (`no_new_support`), an
aspect's status needs its support to move (`aspect_no_passage`).

## 49 (was 48)

Same record fields, prompt changes only.
The status definitions have their own section ("== Belief Statuses ==")
in the belief update and final scorer prompts; each user prompt closes
with one general sentence (one per call: init, update, final score; the belief init's "FIXED" sentence and its report variant
dropped, so the belief bank's prompt key changes); the final scorer
reads the agent's final response without citations or confidence line
(the end of the reasoning only when the response is the answer alone)
and has no "judge from the belief only" rule.

## 48 (was 47)

Same record fields, new numbers.  The
certainty formula (`certainty_formula`, and s_t of every step) is the
mean of the status probabilities on every query shape (was the min on a
short-answer query), over the beliefs the updater judges: an aggregate
is left out (its status is derived from its inputs).  The answer formula
(`certainty_answer_formula`) is the same mean on the answer's
candidates, a computed aggregate kept, a target left out.  The final
scorer is unchanged.

## 47 (was 46)

Same record fields.  "unknown" is a
one-answer status (no passage supports the name and no other candidate
has support: a proposal) and a final status: the final scorer's final
status is the lowest belief status of the answer's candidates, a name
only proposed or absent counting as one no passage supports ("unknown",
or "unlikely" when another candidate has support), a removed one as
"unlikely", and an answer the query computes is judged as a candidate
of the computed aggregate (was: own ladder, weighted by the beliefs
that identify the answer).  `certainty_answer_formula` scores an
unlisted or proposed name so (was: always "unknown"), and an unlisted
answer on a computed aggregate at the lowest score of its inputs.  An update writing "unknown" on a name
without support keeps it proposed with no issue.

## 46 (was 45)

Same record fields; the belief prompts'
statuses mean something else, so older files are not comparable.  The
status definitions no longer overlap and the "give the lowest" rule is
gone (update and final scorer): a one-answer candidate is "unlikely" when
a passage casts doubt on it or a rival has stronger support, "possible"
on a tie, "probable" when the strongest support leaves a part unstated,
"almost certain" when it states every part but some by inference or a
rival has weaker support, "certain" when every part is stated directly
and no rival has support.  An aspect's status counts the sides its facts
cover (one, several, every) and whether every fact rests on several
passages.  The final status ladder is exclusive too.  Prompt files:
`belief_update/statuses.txt` (by belief type: one_answer, target,
computed, aspect) and `belief_update/<shape>.txt` (terms, rules, output
notes, example) replace `belief_statuses.txt`, `query_beliefs.txt`
and `examples.txt`.

## 45 (was 44)

Two status sets.  Kinds `hop`, `rest` and
`target` are gone: a hop is `closed` (one answer), a multi-member
query has no rest belief, a target is an `aggregate`.  An
aggregate's statuses are derived by code: the lowest status of its
inputs, "unlikely" when a passage speaks against it (issue
`aggregate_status` when the reply wrote another).  A report's aspect
has one status of the scale (`unknown` with no fact; `certain` needs
every fact supported by two documents), written by the reply's item
without a candidate; its facts are `kept` or removed.  The
`coverage` field of step and final lines is gone, and the certainty
formula counts every belief (the target was left out).  Issues
`bad_coverage`, `coverage_no_passage`, `coverage_no_fact` become
`bad_aspect_status`, `aspect_no_passage`, `aspect_no_fact`, with
`aspect_one_source`, `bad_fact_status` (a fact neither "kept" nor
"removed") and `not_computed` (an update naming a computed answer no
claim gives) new.  The reply's `reasoning_observations` have
no `turn` (code finds it from the span).

## 44 (was 43)

Same record fields, new numbers.  A status
becomes a probability by the table `core.types.STATUS_PROBABILITY`
(unlikely 0.10, unknown 0.25, possible 0.50, probable 0.70, almost
certain 0.90, certain 0.97; was (level + 2) / 6) in every number read
from a status (`certainty` of the final scorer, `certainty_formula`,
`certainty_answer_formula`, a step's `certainty`).  The certainty
formula of a short-answer query is the lowest belief probability (was
the geometric mean); of a report, their mean.

## 43 (was 42)

Step and final lines add `coverage`, the
coverage updates of a report's aspects (belief kind `aspect`; an
aspect's status is its coverage, "unknown" while no passage supports a
fact of it); a multi-member belief's subject is never its candidate
(issue `subject_candidate`), and an aggregate takes passage
observations (a member's conditions); the document cut is
`max_passage_tokens` (the first tokens, was `max_passage_chars`).

## 42, later additions

Added within schema 42 (no new version, so earlier runs stay readable):
the meta line's `tokens`, the belief update and final scorer tokens of
the sample (null when not counted; absent in earlier files).  A multi-clue
query (BrowseComp-Plus, shape renamed from single_target; set renamed
multi_member) lists its clues, then a target belief ("the <type> that
meets B1 to B<k>", kind `target`), then the asked answer when it is not
the target.  The target takes no observation or update (issues
`derived_belief`, `derived_update`); its candidates are derived
after each call (the names listed on every clue, at the lowest of their
statuses there, with their support), and it is left out of s_t and
`certainty_answer_formula`.  Only multi-clue files change; earlier
BrowseComp-Plus files have no target belief.

## 40 (was 39)

A proposal records the step it was proposed
in (`added_step`; `step` is its latest claim's), so a dropped
proposal's `added_step` is that step; past `MAX_PROPOSALS` the least
recently claimed proposal is dropped (as before), and an update without
a passage is now held to the same cap.  `certainty_answer_formula` scores an answer's candidate a
passage ruled out as "unlikely" (was "unknown").  New issues: "New:" on a
proposed name (`wrong_previous`), `unknown_candidate` on a rest
belief.  Belief ids in a reply are read case-insensitively ("C1"),
and a label string with several labels ("d1, d2") is split.

## 39 (was 38)

The final scorer writes the status of the
final answer on the belief's scale (unlikely, possible, probable, almost
certain, certain: `certainty_label`; `certainty` its
`core.types.status_probability`) in place of an ICD 203 likelihood, and per
belief only the answer's candidate (`final_score`: `{status,
candidates, reason, issues}`, no check).  Its prompt shares the belief
update's definitions and status definitions.

## 38 (was 37)

S_T of a short-answer query is the final
scorer's (`belief_model_component.update.final_score`): one call reads b_{T-1}
and the answer turn (no final update) and writes, per belief, the
answer's candidate and its check, a reason and a likelihood (a word of
ICD 203, `certainty_label`; `certainty` its probability, null when
the call failed).  The final line adds `certainty_label`,
`certainty_formula` (the fixed function of b_{T-1}),
`certainty_answer_formula` (its form on the checks' candidates) and
`final_score`; a report keeps the final update (`certainty` =
`certainty_formula`, the others null).  A removed candidate records
`against` (ruled out: a passage or the agent's claim spoke against
it, not a proposal dropped for room); the final scorer reads those names
as removed.  One document length for the agent and the belief
update (the run's `max_passage_tokens`, recorded in run_config's
`belief_model`): report documents are cut too.  A passage's title is
its metadata title, else its front matter's, else its first line.

## 37 (was 36)

A belief is judged on its own parts: a part
that refers to another belief is met when a passage links the candidate
to a candidate of that belief, whatever that candidate's status, so a
change of the referenced belief no longer lets a status move without
new support (`no_new_support` again only for one answer, another
candidate's support changed).  A call is made only for a passage new to
the belief or on the final turn: no "cited again" passages
(`passages[]` drops `again`).  Beliefs refer only to beliefs before
them: gold multi-hop texts are written in a chain order (the row's
beliefs reordered and renumbered), and an extraction with a forward
reference is asked once more (`belief_info.warnings` "order: ...").

## 36 (was 35)

A multi-hop belief names only its own hop
and refers to the hop it depends on by its id ("the engineer who was born
in the town of B1"); its hops before the answer are of kind `hop`
("several answers": candidates do not compete), the answer `closed`.
The agent's reasoning only proposes names: a reasoning observation is a
claim (`{candidate, claim}`: a status, or "rejected") on one candidate;
a claimed name no passage supports is a proposal of its belief
(`beliefs[].proposals`, no status, not a rival), a passage that
supports it makes it a candidate, a rejected claim drops it.  A
candidate's support is saved with it (`support`: the passage
observations that support it or speak against it, with their spans and
titles) and shown to the next update as quoted spans with the passage
title (no labels); its basis follows from it.  A candidate update writes
the passages of the call that support the candidate or speak against it
(`updates[].support` / `against`); "removed" needs one that speaks
against it.  A name that passages only speak against is not added (a
proposal stays one: `against_only`).  Hard limits (code): a candidate
without support is at most "possible" (`no_support`, replaces
`turn_only`); a candidate whose update adds no support keeps its status
unless, for a one-answer belief, another candidate's support changed
in the call (a proposal added or dropped does not count), or a belief
its text refers to changed (`no_new_support`, replaces
`turn_only_change`).  New flags: `certain_with_rival` and
`unequal_tie` (replace `competing_candidates`), `bad_claim`;
`repeated_observation` dropped (a passage shown again adds its new
spans).  A step makes a call only when it has a new passage or its turn
cites a judged one again (shown "cited again": an id in citation brackets
or after "doc" / "docid" / "id"; a bare number elsewhere is no citation);
otherwise `called` is false, the belief is carried forward and the turn
is read by the next call (the reply's claims carry their turn label, its
passage observations the step of the call), the final call always made.

## 35 (was 34)

The status of a candidate is defined per
belief type, in the prompt of the dataset's query shape
(`prompts/belief_update/belief_statuses.txt`): "one answer" (its
candidates compete: the status counts the other candidates, "certain" only
when no other meets a part), "several answers" (the passages on the
candidate only) and "computed" (the agent's computation).  Prompt Rule 5
is dropped.  "doubtful" is no longer a status (scale "unlikely" ..
"certain", levels -1 .. 4): a candidate a passage rules out, or a
reasoning-only one the turn rejects, is removed (the update writes
"removed"; `updates` record `status: "removed"`), kept in the final
line's `references[].removed`; a removal without the observation that
can make it is kept and flagged (`bad_removal`, `remove_unlisted`), a
removed candidate added again is flagged (`readded`).  A listed passage
candidate of a one-answer belief may change without a new passage when
another of its "passage" candidates changes in the step.  `competing_candidates`
flags more than one candidate above "possible".  The certainty score's q_k
is the belief's status probability (no share between candidates).

## 34 (was 33)

The candidates of a one-answer belief
compete (prompt Rule 5: a candidate is at most "possible" when another
meets as many parts of the belief, at most "unlikely" when another
meets more).  A reasoning-only candidate, and a new one in a step with
no new passage, is moved into "unlikely" .. "possible" (`turn_only`;
was: set to "possible"), so a reasoning rejection is kept; a listed
reasoning-only candidate may move in that range, a listed passage one
still keeps its status without a new passage (`turn_only_change`).
`competing_candidates` flags a candidate above "possible" with another
at "possible" or above (was: two above "possible").  The certainty score
divides a one-answer belief's q_k by max(1, the summed probability of
its candidates at "possible" or above).  Same record fields.

## 33 (was 32)

Replaces the statement of a belief by its
candidates: per belief a list of candidates, each with its status, its
basis ("passage" or "reasoning", set by code) and the step it was added
in; the update writes, per candidate that is new or whose status changes,
a reason ("Was <status>: ..." or "New: ...", saved in `updates`, never
shown) and the new status, and code keeps every other candidate, so the
list only grows.  `statements` becomes `beliefs` (`{id, status,
selected, candidates}`: the belief's status is the highest of its
candidates, "unknown" without one; candidates are listed highest status
first, then "passage" before "reasoning", then in the order added; the
selected candidates are those at "possible" or above, for a one-answer
belief only the first listed); new `updates`; `changed` the
beliefs with a candidate added or whose status changed.  The agent's tag
shows the candidates, no `<sources>`.  Reasoning only names candidates: a
reasoning-only candidate is "possible" (`turn_only`); in a step with no
new passage on the belief, a new candidate is "possible" too
(`turn_only`) and a listed one keeps its status, up or down, whatever
its basis (`turn_only_change`, replaces `turn_only_drop`).  A
candidate's name is matched in spans and titles as whole words.  New issue codes: `empty_candidate`, `bad_status`,
`duplicate_candidate`, `unexpected_candidate`, `no_reason_opening`,
`wrong_previous`, `not_listed`, `listed_as_new`, `near_duplicate`,
`unchanged`, `competing_candidates`; `no_opening`,
`extra_selected`, `no_selected`, `duplicate_statement` and
`unexpected_statement` dropped.

## 32 (was 31)

Passages set the status.  The update writes
a statement only for a belief whose belief changes, so a belief
with a new observation and no statement keeps its own (`missing_statement`
dropped), and `changed` lists the beliefs whose statement the reply
rewrote (was: those with a new observation).  New issue code
`turn_only_drop`: a status that rested on a passage, lowered in a step
with no new passage observation on the belief, is not taken (the old
statement is kept).  `unknown_candidate` also accepts the title of a
passage observed on the belief.

## 31 (was 30)

Keeps the belief at two levels.  Shown: the
statement of a belief is free text with a fixed opening (status,
selected candidates), written by the update and read back verbatim by the
next update (no quotes added) and by the agent; a statement record is
`{id, status, selected, text, cited, support_cited}` (no `by`,
`rivals`, `reason`).  Saved: an observation is the spans of its source
on the belief, with no value (`{id, label, step, [doc_id], spans}`),
a think observation's spans verified against the turn; the final line adds
`references`.  The agent's turn no longer lists its search queries
(`query_value` dropped).  New issue codes: `no_opening` (replaces
`no_status`), `unknown_candidate`, `extra_selected`, `no_selected`;
`unknown_value`, `no_source`, `extra_leader`, `no_by` dropped.

## 30 (was 29)

Drops the stance of an observation: the
update prompt is split into two subtasks, observations (which values each
source states for each belief, no label) and statements (which sources
support a value, "given", and which contradict it, "against"); code checks
that a cited source has an observation on the belief.  An observation
record has no `stance` field; the other saved fields are those of 29.

## 29 (was 28)

Grounds the status in the passages: the
update prompt is rewritten (a passage supports a value only when it states
a condition of the belief, another entity never contradicts one, a
search query is no value, settled beliefs still take new passages), each
cited document in the current statement comes with a quote of its span,
code lowers a status above "possible" that no passage supports
(`turn_only`) and drops a think value that is a search query
(`query_value`), and "doubtful" scores 1/12 instead of 0 in the
certainty.  The saved fields are those of 28.

## 28 (was 27)

Replaces the state of a belief (status,
covered_by, candidates with evidence) by one statement (status, the values
it is covered by and its rivals, each with its sources, reason), weighs
the agent's turns as well as the documents in the status, reads the answer
turn as an ordinary update in the `final` line (no answer ratings; meta
`final_*` dropped), scores certainty from the statuses only (aggregate
beliefs now count), and saves one `issues` list per step in place of
the per-record notes (`think_observations` / `passage_observations` become
`observations`, `beliefs_after` `statements`, `belief_updates`
`changed`, `belief_output` `reply`, `passages` the `label` of
`docs`, `seen_before` `new`).  `iteration` counts from 1 for every
agent; `agent_iteration` is the agent's own counter.  A signal that could
not be computed is null, never 0, and every float is finite and rounded to
4 decimals.  A query without any search still gets its meta line.  The file
is written atomically.
