"""``BeliefState``, b_t: per criterion its candidates with their support
and its proposals (shown), and the trace, references and candidate
updates behind them (saved).  ``apply`` takes one belief update reply."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..text.matching import alnum, contains_words, content_words, squash
from .issues import (
    ISSUE_AGAINST_ONLY, ISSUE_BAD_CLAIM, ISSUE_BAD_COVERAGE, ISSUE_BAD_REF, ISSUE_BAD_REMOVAL, ISSUE_BAD_STATUS,
    ISSUE_CERTAIN_WITH_RIVAL, ISSUE_COVERAGE_NO_FACT, ISSUE_COVERAGE_NO_PASSAGE, ISSUE_DERIVED,
    ISSUE_DERIVED_UPDATE, ISSUE_DUPLICATE_CANDIDATE, ISSUE_EMPTY_CANDIDATE,
    ISSUE_LISTED_AS_NEW, ISSUE_MERGED_DUPLICATE, ISSUE_NEAR_DUPLICATE, ISSUE_NO_NEW_SUPPORT, ISSUE_NO_REASON,
    ISSUE_NO_SPAN, ISSUE_NO_SUPPORT, ISSUE_NOT_IN_TURN, ISSUE_NOT_LISTED, ISSUE_NOT_NEW_PASSAGE, ISSUE_READDED,
    ISSUE_REMOVE_UNLISTED, ISSUE_SPAN_OFF_CANDIDATE, ISSUE_SUBJECT_CANDIDATE, ISSUE_UNCHANGED, ISSUE_UNEQUAL_TIE,
    ISSUE_UNEXPECTED_CANDIDATE, ISSUE_UNKNOWN_CANDIDATE, ISSUE_UNKNOWN_CRITERION, ISSUE_WRONG_PREVIOUS,
)
from .reason_text import parse_reason, reason_labels
from .types import (
    ASPECT, DERIVED_KINDS, DOC, NAMED_KINDS, NO_DOC_KINDS, NO_SUPPORT_MAX_STATUS, REASONING, REJECTED, REMOVED, REST,
    STATUS_LEVEL, THINK, UNKNOWN, Candidate, CandidateUpdate, CoverageUpdate, Criterion, Observation, Proposal,
    RemovedCandidate, Support, exclusive, is_removed, member_subject, parse_coverage, parse_status, subject_names,
)

# Proposals kept per criterion (beyond it the least recently claimed is dropped).
MAX_PROPOSALS = 8


def _issue(cid: str, code: str, detail: str, o: Optional[Observation] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": cid, "code": code, "detail": detail}
    if o is not None:
        out["label"] = o.label
    return out


def _level(status: str) -> int:
    return STATUS_LEVEL[status]


def _cap(status: str) -> str:
    """*status* at most ``NO_SUPPORT_MAX_STATUS``."""
    return min(status, NO_SUPPORT_MAX_STATUS, key=_level)


def _label_number(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


@dataclass
class _CallContext:
    """What the phases of one ``BeliefState.apply`` share: the call's step,
    per criterion the passages of the call observed on it (label -> its
    observation in the trace) and whether the call has a claim on it, and
    what the phases produce (issues, the observations that entered, the
    changed criteria, the update records, the deferred status changes, the
    proposals a claim of the call rejected: ``(k, alnum(name))``)."""
    step: int
    call_docs: List[Dict[str, Observation]]
    claimed: List[bool]
    issues: List[Dict[str, Any]] = field(default_factory=list)
    entered: List[Observation] = field(default_factory=list)
    changed: Set[int] = field(default_factory=set)
    records: List[Dict[str, Any]] = field(default_factory=list)
    deferred: List[Tuple[int, Candidate, str, Dict[str, Any]]] = field(default_factory=list)
    dropped: Set[Tuple[int, str]] = field(default_factory=set)


class BeliefState:
    """Belief state b_t, kept at two levels.

    Shown (``views.tag.format_belief``): per criterion its candidates,
    each with its status and its support (the passage observations that
    support it or speak against it, with their spans and passage titles),
    and its proposals (names the agent's reasoning proposed that no passage
    supports yet, without a status).  The agent's claims are kept but never
    shown to the update.

    Saved: per criterion the trace of the observations that entered it
    (``trace``) and its references (``reference_lists``): every source ever
    assigned to it, with its document, title, the step it came in and its
    verified spans; per step the candidate updates with their reasons
    (``apply``), never shown again; and per criterion the candidates and
    proposals it dropped (``removed``, ``core.types.RemovedCandidate``).

    Each call, the belief update writes the claims of the agent's turns
    (reasoning observations), the observations of the call's passages,
    then the candidates whose support or status changes, each with the
    passages that support it or speak against it, its reason and its new
    status; ``apply`` keeps what is valid:

    - an observation enters only when it is grounded (a verified span of
      its passage or turn) and names a known criterion that is not derived
      (a target); a reasoning observation also needs a candidate and a
      claim.  On a computed criterion (``NO_DOC_KINDS``) a passage
      observation states a condition of its definition for a member (its
      membership), never its answer.  A second
      observation of a passage on the same criterion adds its new spans to
      the first one;
    - a claim sets the agent's claim on a listed candidate, or adds, updates
      or (``REJECTED``) drops a proposal; it never moves a status;
    - a candidate update is taken only for a criterion with an observation
      in the call (a computed criterion: any observation in the call, since
      its status follows the members' answers), once per candidate, with a
      status of the scale other
      than "unknown", or "removed".  Its support labels must be passages of
      the call observed on the criterion (``bad_ref`` otherwise, the label
      dropped).  A name neither listed nor proposed, or a proposed one,
      becomes a candidate only with a passage that supports it (none at
      all: it stays or becomes a proposal, ``no_support``; only passages
      against it: a new name is not added, a proposal stays one,
      ``against_only``); a listed one gets the new support and status.  Candidates the reply
      does not name keep their status.  The reason is checked and flagged,
      never used to reject;
    - "removed" takes a listed candidate (or a proposal) out of the belief
      (into ``removed``) only when a passage of the call speaks against it
      (``against``); on a computed criterion, when the call has a claim on
      it.  Otherwise it is flagged and kept;
    - the hard limits (a computed criterion is exempt): a candidate no
      passage supports is at most "possible" (``no_support``); a listed
      candidate whose update adds no support keeps its status, unless in
      the same call another of its candidates gained support, was added or
      was removed (one answer: its status depends on the others; a
      proposal added or dropped does not count); ``no_new_support``
      otherwise.  A criterion that refers to another one ("the town of
      c1") is judged on its own parts, whatever the statuses of the
      candidates it links to, so a change there moves nothing here.
    - the subject of a member criterion ("<member>: <property>",
      ``core.types.member_subject``) is never its candidate nor proposal,
      and a rest criterion's name needs its value ("<member>: <value>"):
      such a claim or update is not taken (``subject_candidate``);
    - a new fact of an aspect criterion whose words contain a listed fact,
      or the reverse, joins it (``merged_duplicate``): its support is added
      to the listed fact, at the higher of the two statuses;
    - the coverage of an aspect criterion (``coverage``, the reply's
      ``coverage`` list) changes only with a passage observation on the
      aspect in the call (``coverage_no_passage`` otherwise), and is
      "unknown" while no passage supports a fact of it
      (``coverage_no_fact``);
    - flags of a one-answer criterion, statuses kept: a "certain" candidate
      while another has support (``certain_with_rival``); two candidates
      with the same support and different statuses (``unequal_tie``).
    - a target criterion of a multi-clue query (``DERIVED_KINDS``) takes no
      observation (``derived_criterion``) and no candidate update
      (``derived_update``): after each call its candidates are derived from
      the clues before it (``_derive_targets``).

    Derived per criterion (``belief_lists``): its status, the highest of
    its candidates ("unknown" without candidates), an aspect's its
    coverage; its selected
    candidate(s), among those at "possible" or above: for a one-answer
    criterion the first one shown (``ordered``), for a several-answers
    criterion every one.  A proposal is never selected.

    Documents get their label ("d<n>") when they are first shown
    (``label_doc``).
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self._index = {c.id: k for k, c in enumerate(self.criteria)}
        # Per criterion its candidates, in the order they were added, and its proposals.
        self.candidates: List[List[Candidate]] = [[] for _ in self.criteria]
        self.proposals: List[List[Proposal]] = [[] for _ in self.criteria]
        # The trace: per criterion, the observations that entered it.
        self.trace: List[List[Observation]] = [[] for _ in self.criteria]
        # Per criterion the candidates and proposals dropped, in order.
        self.removed: List[List[RemovedCandidate]] = [[] for _ in self.criteria]
        # Per aspect criterion its coverage status ("unknown" for the others).
        self.coverage: List[str] = [UNKNOWN for _ in self.criteria]
        # Per criterion the member it is about (a member criterion), or None.
        self._subjects: List[Optional[str]] = [member_subject(self.criteria, k) for k in range(len(self.criteria))]
        self._doc_labels: Dict[str, str] = {}
        self._titles: Dict[str, str] = {}

    def label_doc(self, key: str, title: str) -> str:
        """The label of the document *key* (its id, or a key of its text),
        new ("d<n>", next number) when it was not shown before."""
        label = self._doc_labels.get(key)
        if label is None:
            label = f"d{len(self._doc_labels) + 1}"
            self._doc_labels[key] = label
            self._titles[label] = title
        return label

    def label_for_key(self, key: str) -> Optional[str]:
        """The label of the document *key*, None when it was never shown."""
        return self._doc_labels.get(key)

    def title(self, label: str) -> str:
        """The title of the document labelled *label* ("" when unknown)."""
        return self._titles.get(label) or ""

    # ------------------------------------------------------------------
    # Derived per criterion
    # ------------------------------------------------------------------

    def ordered(self, k: int) -> List[Candidate]:
        """The candidates of criterion *k* as shown: highest status first,
        then "passage" before "reasoning", then in the order they were
        added (so a one-answer criterion's selected candidate is first)."""
        return sorted(self.candidates[k], key=lambda c: (-_level(c.status), c.basis == REASONING))

    def status(self, k: int) -> str:
        """The status of criterion *k*: an aspect's coverage, else the
        highest of its candidates ("unknown" without candidates)."""
        if self.criteria[k].kind == ASPECT:
            return self.coverage[k]
        return max((c.status for c in self.candidates[k]), key=_level, default=UNKNOWN)

    def selected(self, k: int) -> List[str]:
        """The selected candidate(s) of criterion *k*, among those at
        "possible" or above: for a one-answer criterion the first one shown
        (``ordered``), for a several-answers criterion every one."""
        kept = [c.name for c in self.ordered(k) if _level(c.status) >= _level(NO_SUPPORT_MAX_STATUS)]
        return kept[:1] if exclusive(self.criteria[k].kind) else kept

    def ruled_out(self, k: int) -> List[str]:
        """The names criterion *k* ruled out (``RemovedCandidate.against``:
        a passage or the agent's claim spoke against them, not a proposal
        dropped for room), each once, those listed or proposed again left
        out (the final scorer reads them as removed)."""
        names: Dict[str, str] = {}
        for x in self.removed[k]:
            if x.against and self._listed(k, x.name) is None and self._proposed(k, x.name) is None:
                names.setdefault(alnum(x.name), x.name)
        return list(names.values())

    def ruled_out_lists(self) -> Dict[str, List[str]]:
        """``ruled_out`` of every criterion, by id."""
        return {c.id: self.ruled_out(k) for k, c in enumerate(self.criteria)}

    # ------------------------------------------------------------------
    # The update
    # ------------------------------------------------------------------

    def apply(self, claims: List[Observation], docs: List[Observation],
              updates: List[CandidateUpdate], step: int,
              coverage: Optional[List[CoverageUpdate]] = None) -> Dict[str, Any]:
        """Apply one belief update of the call of step *step*: the claims of
        the turns (*claims*, reasoning observations), the observations of
        the passages (*docs*), the candidate *updates*, then the aspects'
        *coverage* updates (a report).

        Returns ``{observations, updates, changed, issues}``: the
        observations that entered the belief (``Observation.to_dict``), one
        record per candidate update of the reply (``{id, candidate, reason,
        support, against, written, previous, status}``: ``support`` and
        ``against`` the labels kept, ``written`` the status the reply wrote
        ("removed" for a removal), ``previous`` the candidate's status
        before (null for a new one, "proposed" for a proposal), ``status``
        the one kept ("removed" when it left the belief, "proposed" when it
        stays a proposal), null when the update was not taken), one record
        per coverage update (``{id, reason, written, previous, status}``,
        ``status`` the coverage kept, null when not taken), the ids of
        the criteria whose candidates or proposals changed, and one issue
        record per thing the reply got wrong (the ``ISSUE_*`` codes).

        Phases, in order: admit the observations, apply the claims, apply
        the candidate updates, resolve the status changes deferred on
        one-answer criteria, apply the coverage updates, derive the target
        criteria from their clues, flag one-answer criteria whose statuses
        do not follow their support."""
        # The proposals a claim of this call rejected.
        call = _CallContext(step=step, call_docs=[{} for _ in self.criteria], claimed=[False] * len(self.criteria))
        self._admit_observations([*claims, *docs], call)
        self._apply_claims(call)
        # The support of the one-answer criteria before the updates (only
        # they defer a status change, ``_resolve_deferred``).
        before = {k: self._support_key(k) for k, c in enumerate(self.criteria) if exclusive(c.kind)}
        done: set = set()
        for u in updates:
            self._apply_update(u, call, done)
        self._resolve_deferred(before, call)
        coverage_records = self._apply_coverage(coverage or [], call)
        self._derive_targets(call)
        for k in sorted(call.changed):
            kind = self.criteria[k].kind
            if exclusive(kind) and kind not in NO_DOC_KINDS and kind not in DERIVED_KINDS:
                self._flag_rivals(k, call.issues)
        return {"observations": [o.to_dict() for o in call.entered], "updates": call.records,
                "coverage": coverage_records,
                "changed": [self.criteria[k].id for k in sorted(call.changed)], "issues": call.issues}

    def _admit_observations(self, observations: List[Observation], call: "_CallContext") -> None:
        """Phase 1: each observation enters the trace when it passes the
        gates (a known criterion that is not derived, a passage of the
        call, a verified span, a claim with a candidate and a status);
        otherwise one issue.  A second observation of a
        passage on one criterion adds its new spans to the first."""
        issues = call.issues
        for o in observations:
            k = self._index.get(o.criterion)
            if k is None:
                issues.append(_issue(o.criterion, ISSUE_UNKNOWN_CRITERION, "unknown criterion id", o))
            elif self.criteria[k].kind in DERIVED_KINDS:
                issues.append(_issue(o.criterion, ISSUE_DERIVED,
                                     "the criterion is derived from its clues: no observation", o))
            elif o.source == DOC and o.doc_id is None:
                issues.append(_issue(o.criterion, ISSUE_NOT_NEW_PASSAGE, "no passage of the call has this label", o))
            elif not o.grounded:
                issues.append(_issue(o.criterion, ISSUE_NO_SPAN, "no verified span", o) if o.source == DOC else
                              _issue(o.criterion, ISSUE_NOT_IN_TURN, "no span of the agent's turn", o))
            elif o.source == THINK and (not alnum(o.candidate or "") or o.claim is None):
                issues.append(_issue(o.criterion, ISSUE_BAD_CLAIM, "a claim needs a candidate and a status or "
                                                                   f"'{REJECTED}'", o))
            elif o.source == DOC:
                known = next((x for x in self.trace[k] if x.source == DOC and x.label == o.label), None)
                if known is None:
                    self.trace[k].append(o)
                    known = o
                else:
                    # A second observation of the passage: its new spans join the first.
                    for span in o.verified_spans:
                        if span not in known.spans:
                            known.spans.append(span)
                            known.span_verified.append(True)
                call.call_docs[k][o.label] = known
                call.entered.append(o)
            else:
                self.trace[k].append(o)
                call.claimed[k] = True
                call.entered.append(o)

    def _apply_claims(self, call: "_CallContext") -> None:
        """Phase 2: the claims that entered, in the order of their turns
        (the latest one stands)."""
        for o in sorted((o for o in call.entered if o.source == THINK), key=lambda o: _label_number(o.label)):
            if self._claim(self._index[o.criterion], o, call):
                call.changed.add(self._index[o.criterion])

    def _apply_update(self, u: CandidateUpdate, call: "_CallContext", done: set) -> None:
        """Phase 3, one candidate update: its record, then the gates (a
        known criterion, a name, a status, once per candidate, an
        observation on the criterion in the call), its support and against
        labels, then the removal or the status update.  *done* holds the
        ``(criterion, name)`` pairs already updated in the call."""
        issues = call.issues
        name = squash(u.candidate)
        rec: Dict[str, Any] = {"id": u.id, "candidate": name, "reason": u.reason, "support": [], "against": [],
                               "written": u.status, "previous": None, "status": None}
        call.records.append(rec)
        k = self._index.get(u.id)
        if k is None:
            issues.append(_issue(u.id, ISSUE_UNKNOWN_CRITERION, f"candidate {name!r} of an unknown criterion id"))
            return
        if self.criteria[k].kind in DERIVED_KINDS:
            issues.append(_issue(u.id, ISSUE_DERIVED_UPDATE, f"{name!r}: the criterion is derived from its clues; "
                                                             "ignored"))
            return
        if not alnum(name):
            issues.append(_issue(u.id, ISSUE_EMPTY_CANDIDATE, "an update without a candidate name"))
            return
        status = REMOVED if is_removed(u.status) else parse_status(u.status)
        if status is None or status == UNKNOWN:
            issues.append(_issue(u.id, ISSUE_BAD_STATUS,
                                 f"candidate {name!r}: status {u.status!r} is not a candidate status"))
            return
        if (k, alnum(name)) in done:
            issues.append(_issue(u.id, ISSUE_DUPLICATE_CANDIDATE, f"a second update of {name!r}; ignored"))
            return
        done.add((k, alnum(name)))
        call_docs = call.call_docs[k]
        computed = self.criteria[k].kind in NO_DOC_KINDS
        if not call_docs and not call.claimed[k] and not (computed and call.entered):
            issues.append(_issue(u.id, ISSUE_UNEXPECTED_CANDIDATE, f"{name!r}: no observation on the criterion "
                                                                    "in the call; ignored"))
            return
        rec["written"] = status
        support = self._labels(k, name, u.support, call_docs, issues)
        against = [lb for lb in self._labels(k, name, u.against, call_docs, issues) if lb not in support]
        rec["support"], rec["against"] = support, against
        entries = [self._support(call_docs[lb], call.step, False) for lb in support] + \
                  [self._support(call_docs[lb], call.step, True) for lb in against]
        if status == REMOVED:
            moved = self._remove(k, name, u.reason, entries, call, rec, issues)
        else:
            moved = self._update(k, name, status, u.reason, call.step, entries, rec, issues, call.deferred)
        if moved:
            call.changed.add(k)

    def _resolve_deferred(self, before: Dict[int, frozenset], call: "_CallContext") -> None:
        """Phase 4: a one-answer candidate whose update added no support of
        its own changes status only when its criterion had a candidate
        added or removed, or whose support changed, in the call (*before*:
        each one-answer criterion's support before the updates; a proposal
        added or dropped does not count); otherwise it keeps its status
        (``no_new_support``)."""
        support_moved = {k for k in before if self._support_key(k) != before[k]} if call.deferred else set()
        for k, existing, status, rec in call.deferred:
            if exclusive(self.criteria[k].kind) and k in support_moved:
                if not existing.supports() and _cap(status) != status:
                    call.issues.append(_issue(self.criteria[k].id, ISSUE_NO_SUPPORT,
                                              f"{existing.name!r}: {status!r} with no passage supporting it; "
                                              f"set to {_cap(status)!r}"))
                    status = _cap(status)
                existing.status = status
                rec["status"] = existing.status
                call.changed.add(k)
            else:
                call.issues.append(_issue(self.criteria[k].id, ISSUE_NO_NEW_SUPPORT,
                                          f"{existing.name!r}: {status!r} from {existing.status!r} with no new "
                                          f"passage on it, and no other candidate's support changed; kept"))
                rec["status"] = existing.status

    def _derive_targets(self, call: "_CallContext") -> None:
        """Phase 6: the candidates of every target criterion, derived from
        the clues before it (its closed criteria): a name listed on every
        clue, at the lowest of its statuses there (a rival or a passage
        against it on one clue lowers it), with the support of all of them.
        A name missing from one clue is no candidate (its status there is
        "unknown"); a target candidate kept from before keeps the step it
        was added in."""
        for k, c in enumerate(self.criteria):
            if c.kind not in DERIVED_KINDS:
                continue
            clues = [j for j in range(k) if self.criteria[j].kind not in DERIVED_KINDS]
            old = {alnum(x.name): x for x in self.candidates[k]}
            derived: List[Candidate] = []
            if clues:
                for x in self.candidates[clues[0]]:
                    on_clues = [self._listed(j, x.name) for j in clues]
                    if any(y is None for y in on_clues):
                        continue
                    prev = old.get(alnum(x.name))
                    derived.append(Candidate(
                        name=x.name, status=min((y.status for y in on_clues), key=_level),
                        added_step=prev.added_step if prev is not None else call.step,
                        support=[s for y in on_clues for s in y.support]))
            before = [(alnum(x.name), x.status, len(x.support)) for x in self.candidates[k]]
            self.candidates[k] = derived
            if [(alnum(x.name), x.status, len(x.support)) for x in derived] != before:
                call.changed.add(k)

    def _support_key(self, k: int) -> frozenset:
        """The candidates of criterion *k* with their support (passage
        labels and sides), statuses left out: what ``apply`` compares to
        tell whether the support of the criterion moved in a call."""
        return frozenset((alnum(x.name), frozenset((s.label, s.against) for s in x.support))
                         for x in self.candidates[k])

    def _listed(self, k: int, name: str) -> Optional[Candidate]:
        """The listed candidate of criterion *k* named *name* (letters and
        digits compared), or None."""
        key = alnum(name)
        return next((x for x in self.candidates[k] if alnum(x.name) == key), None)

    def _proposed(self, k: int, name: str) -> Optional[Proposal]:
        key = alnum(name)
        return next((x for x in self.proposals[k] if alnum(x.name) == key), None)

    def _dropped(self, k: int, name: str) -> bool:
        key = alnum(name)
        return any(alnum(x.name) == key for x in self.removed[k])

    def _claim(self, k: int, o: Observation, call: _CallContext) -> bool:
        """Apply the claim *o* on criterion *k*; returns whether the
        proposals changed."""
        step, issues = o.step, call.issues
        existing = self._listed(k, o.candidate or "")
        if existing is not None:
            existing.claim, existing.claim_step = o.claim, step
            return False
        proposal = self._proposed(k, o.candidate or "")
        if proposal is not None:
            if o.claim == REJECTED:
                self.proposals[k].remove(proposal)
                call.dropped.add((k, alnum(proposal.name)))
                self.removed[k].append(RemovedCandidate(
                    name=proposal.name, status=None, basis=REASONING, added_step=proposal.added_step, removed_step=step,
                    reason=f"rejected in {o.label}: {' ... '.join(o.verified_spans)}", against=True))
                return True
            proposal.claim, proposal.step = o.claim, step
            return False
        if o.claim == REJECTED:
            return False
        problem = self._subject_problem(k, o.candidate or "")
        if problem:
            issues.append(_issue(self.criteria[k].id, ISSUE_SUBJECT_CANDIDATE, f"claim: {problem}; not proposed", o))
            return False
        self._add_proposal(k, squash(o.candidate), o.claim, step, issues, o)
        return True

    def _subject_problem(self, k: int, name: str) -> Optional[str]:
        """Why *name* cannot be a candidate of criterion *k* (None when it
        can): it is the member the member criterion is about, or a rest
        criterion's name without its value."""
        subject = self._subjects[k]
        if subject is not None and alnum(name) in {alnum(n) for n in subject_names(subject)}:
            return f"{name!r} is the member the criterion is about, not its {self.criteria[k].text.strip()[len(subject):].lstrip(':').strip()}"
        if self.criteria[k].kind == REST:
            member, _, value = name.partition(":")
            if not alnum(member) or not alnum(value):
                return f"{name!r} is not written \"<member>: <value>\""
        return None

    def _add_proposal(self, k: int, name: str, claim: Optional[str], step: int, issues: List[Dict[str, Any]],
                      o: Optional[Observation] = None) -> None:
        """Propose *name* for criterion *k* (from a claim *o*, or an update
        without a passage): flagged when it was dropped before; past
        ``MAX_PROPOSALS`` the least recently claimed proposal is dropped
        (the agent's current names stay)."""
        if self._dropped(k, name):
            issues.append(_issue(self.criteria[k].id, ISSUE_READDED, f"{name!r} was dropped before; proposed again", o))
        self.proposals[k].append(Proposal(name=name, claim=claim, step=step, added_step=step))
        if len(self.proposals[k]) > MAX_PROPOSALS:
            old = min(self.proposals[k], key=lambda x: x.step)
            self.proposals[k].remove(old)
            self.removed[k].append(RemovedCandidate(name=old.name, status=None, basis=REASONING,
                                                    added_step=old.added_step, removed_step=step,
                                                    reason=f"more than {MAX_PROPOSALS} proposals: the least recently claimed dropped"))

    def _labels(self, k: int, name: str, labels: List[str], call_docs: Dict[str, Observation],
                issues: List[Dict[str, Any]]) -> List[str]:
        """The passage *labels* of an update of candidate *name* that are
        passages of the call observed on criterion *k*; the others are
        flagged and dropped."""
        kept = [lb for lb in dict.fromkeys(labels) if lb in call_docs]
        bad = [lb for lb in dict.fromkeys(labels) if lb not in call_docs]
        if bad:
            issues.append(_issue(self.criteria[k].id, ISSUE_BAD_REF,
                                 f"{name!r}: passages {bad} have no observation on the criterion in the call"))
        return kept

    def _support(self, o: Observation, step: int, against: bool) -> Support:
        return Support(label=o.label, doc_id=o.doc_id, title=self.title(o.label), step=step,
                       spans=o.verified_spans, against=against)

    def _check_reason(self, k: int, name: str, reason: str, existing: Optional[Candidate],
                      proposal: Optional[Proposal], issues: List[Dict[str, Any]]) -> None:
        """Flag what the *reason* of an update of *name* of criterion *k*
        gets wrong: its opening against the candidate's current status (or
        its proposal), its labels against the trace."""
        c = self.criteria[k]
        opening = parse_reason(reason)
        if opening is None:
            issues.append(_issue(c.id, ISSUE_NO_REASON, f"{name!r}: the reason opens with none of 'Was <status>:', "
                                                        "'Was proposed:', 'New:'"))
        elif opening.new and existing is not None:
            issues.append(_issue(c.id, ISSUE_LISTED_AS_NEW, f"{name!r} is listed ({existing.status}); its status is updated"))
        elif opening.new and proposal is not None:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS, f"{name!r}: the reason says it is new, it was proposed"))
        elif opening.proposed and proposal is None:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS, f"{name!r}: the reason says it was proposed, it is "
                                                             f"{existing.status if existing else 'not listed'}"))
        elif not opening.new and not opening.proposed and existing is None:
            issues.append(_issue(c.id, ISSUE_NOT_LISTED, f"'Was' on {name!r}, which is not listed"))
        elif not opening.new and not opening.proposed and opening.previous != existing.status:
            issues.append(_issue(c.id, ISSUE_WRONG_PREVIOUS,
                                 f"{name!r}: the reason says it was {opening.previous!r}, it is {existing.status!r}"))
        observed = {o.label for o in self.trace[k]}
        bad = [lb for lb in reason_labels(reason) if lb not in observed]
        if bad:
            issues.append(_issue(c.id, ISSUE_BAD_REF, f"{name!r}: the reason cites sources {bad} with no "
                                                      "observation on the criterion"))

    def _remove(self, k: int, name: str, reason: str, entries: List[Support], call: _CallContext,
                rec: Dict[str, Any], issues: List[Dict[str, Any]]) -> bool:
        """Remove candidate or proposal *name* of criterion *k* when a
        passage of the call speaks against it (``apply``); fills *rec* and
        returns whether it was removed."""
        step, claimed = call.step, call.claimed[k]
        c = self.criteria[k]
        existing, proposal = self._listed(k, name), self._proposed(k, name)
        if existing is None and proposal is None and (k, alnum(name)) in call.dropped:
            # A proposal a claim of the call rejected: already dropped.
            rec["previous"], rec["status"] = "proposed", REMOVED
            return False
        if existing is None and proposal is None:
            issues.append(_issue(c.id, ISSUE_REMOVE_UNLISTED, f"'removed' on {name!r}, which is neither listed nor "
                                                              "proposed; ignored"))
            return False
        rec["previous"] = existing.status if existing is not None else "proposed"
        rec["candidate"] = existing.name if existing is not None else proposal.name
        self._check_reason(k, rec["candidate"], reason, existing, proposal, issues)
        against = [x for x in entries if x.against]
        if not (against or (c.kind in NO_DOC_KINDS and claimed)):
            issues.append(_issue(c.id, ISSUE_BAD_REMOVAL, f"{rec['candidate']!r}: removed, but no passage of the "
                                                          "call speaks against it; kept"))
            rec["status"] = rec["previous"]
            return False
        if existing is not None:
            self.candidates[k].remove(existing)
            self.removed[k].append(RemovedCandidate(name=existing.name, status=existing.status, basis=existing.basis,
                                                    added_step=existing.added_step, removed_step=step, reason=reason,
                                                    against=True))
        else:
            self.proposals[k].remove(proposal)
            self.removed[k].append(RemovedCandidate(name=proposal.name, status=None, basis=REASONING,
                                                    added_step=proposal.added_step, removed_step=step, reason=reason,
                                                    against=True))
        rec["status"] = REMOVED
        return True

    def _update(self, k: int, name: str, status: str, reason: str, step: int, entries: List[Support],
                rec: Dict[str, Any], issues: List[Dict[str, Any]],
                deferred: List[Tuple[int, Candidate, str, Dict[str, Any]]]) -> bool:
        """Add candidate *name* of criterion *k* with *status* and its
        support *entries*, promote its proposal, or change a listed one,
        within the hard limits (``apply``); fills *rec* and returns whether
        the belief changed.  A status change ``apply`` decides after the
        reply's other updates goes to *deferred*."""
        c = self.criteria[k]
        existing, proposal = self._listed(k, name), self._proposed(k, name)
        if existing is not None:
            rec["previous"], rec["candidate"] = existing.status, existing.name
        elif proposal is not None:
            rec["previous"], rec["candidate"] = "proposed", proposal.name
        problem = self._subject_problem(k, name) if existing is None else None
        if problem:
            # Never a proposal either (``_claim`` and this check refuse it).
            issues.append(_issue(c.id, ISSUE_SUBJECT_CANDIDATE, f"{problem}; not added"))
            rec["status"] = None
            return False
        self._check_reason(k, rec["candidate"], reason, existing, proposal, issues)
        computed = c.kind in NO_DOC_KINDS

        if existing is None:
            if not entries and not computed:
                issues.append(_issue(c.id, ISSUE_NO_SUPPORT, f"{name!r}: no passage of the call supports it or speaks "
                                                             "against it; kept as proposed"))
                rec["status"] = "proposed"
                if proposal is None:
                    self._add_proposal(k, name, None, step, issues)
                    return True
                return False
            if not computed and not any(not x.against for x in entries):
                # Only passages against it: no possible answer (a passage
                # that states it fails a part removes a proposal instead).
                issues.append(_issue(c.id, ISSUE_AGAINST_ONLY, f"{name!r}: only passages against it; " +
                                     ("kept as proposed" if proposal is not None else "not added")))
                rec["status"] = "proposed" if proposal is not None else None
                return False
            if not computed and c.kind in NAMED_KINDS and self._unknown_candidate(k, name):
                issues.append(_issue(c.id, ISSUE_UNKNOWN_CANDIDATE, f"{name!r} in no span or title of the "
                                                                    "criterion's passages"))
            near = [x for x in self.candidates[k] if contains_words(x.name, name) or contains_words(name, x.name)]
            if near and c.kind == ASPECT:
                return self._merge(k, near[0], name, status, entries, proposal, rec, issues)
            if near:
                issues.append(_issue(c.id, ISSUE_NEAR_DUPLICATE, f"{name!r} is close to listed {[x.name for x in near]}"))
            if proposal is None and self._dropped(k, name):
                issues.append(_issue(c.id, ISSUE_READDED, f"{name!r} was dropped before; it is added again"))
            cand = Candidate(name=proposal.name if proposal is not None else name, status=status,
                             added_step=step, support=list(entries))
            if proposal is not None:
                cand.claim, cand.claim_step = proposal.claim, proposal.step
                self.proposals[k].remove(proposal)
            if not computed and not cand.supports() and _cap(status) != status:
                issues.append(_issue(c.id, ISSUE_NO_SUPPORT, f"{name!r}: {status!r} with no passage supporting it; "
                                                             f"set to {_cap(status)!r}"))
                cand.status = _cap(status)
            self._check_spans(k, cand.name, entries, issues)
            self.candidates[k].append(cand)
            rec["status"] = cand.status
            return True

        known = {(x.label, x.against) for x in existing.support}
        added = [x for x in entries if (x.label, x.against) not in known]
        if not added and not computed:
            if status == existing.status:
                issues.append(_issue(c.id, ISSUE_UNCHANGED, f"{existing.name!r}: no new support and the status "
                                                            "written is its current one"))
                rec["status"] = existing.status
                return False
            deferred.append((k, existing, status, rec))
            return False
        existing.support.extend(added)
        self._check_spans(k, existing.name, added, issues)
        if not computed and not existing.supports() and _cap(status) != status:
            issues.append(_issue(c.id, ISSUE_NO_SUPPORT, f"{existing.name!r}: {status!r} with no passage supporting "
                                                         f"it; set to {_cap(status)!r}"))
            status = _cap(status)
        moved = bool(added) or status != existing.status
        existing.status = status
        rec["status"] = status
        return moved

    def _merge(self, k: int, listed: Candidate, name: str, status: str, entries: List[Support],
               proposal: Optional[Proposal], rec: Dict[str, Any], issues: List[Dict[str, Any]]) -> bool:
        """A new fact *name* of aspect criterion *k* that repeats the listed
        fact *listed* (the words of one contain the other's): its support
        joins the listed fact, at the higher of the two statuses (a fact
        no passage supports at most "possible"); a proposal of that name is
        dropped.  Fills *rec* (under the listed name) and returns whether
        the belief changed."""
        issues.append(_issue(self.criteria[k].id, ISSUE_MERGED_DUPLICATE, f"{name!r} repeats listed {listed.name!r}; "
                                                                         "merged into it"))
        if proposal is not None:
            self.proposals[k].remove(proposal)
        known = {(x.label, x.against) for x in listed.support}
        added = [x for x in entries if (x.label, x.against) not in known]
        listed.support.extend(added)
        self._check_spans(k, listed.name, added, issues)
        new_status = max(listed.status, status, key=_level)
        if not listed.supports():
            new_status = _cap(new_status)
        moved = bool(added) or new_status != listed.status or proposal is not None
        rec["candidate"], rec["previous"] = listed.name, listed.status
        listed.status = new_status
        rec["status"] = listed.status
        return moved

    def _check_spans(self, k: int, name: str, entries: List[Support], issues: List[Dict[str, Any]]) -> None:
        """Flag (information only) a supporting span of an aspect's fact
        *name* that shares no content word with it: support attached to
        another fact."""
        if self.criteria[k].kind != ASPECT:
            return
        words = content_words(name)
        for x in entries:
            if not x.against and words and not words & content_words(" ".join(x.spans)):
                issues.append(_issue(self.criteria[k].id, ISSUE_SPAN_OFF_CANDIDATE,
                                     f"{name!r}: the spans of {x.label} share no content word with it"))

    def _apply_coverage(self, items: List[CoverageUpdate], call: "_CallContext") -> List[Dict[str, Any]]:
        """Phase 5: the coverage updates of the aspects, once per aspect,
        each taken only with a passage observation on the aspect in the
        call, "unknown" while no passage supports a fact of it (also for an
        aspect the reply left out); returns one record per item."""
        records: List[Dict[str, Any]] = []
        done: Set[int] = set()
        for u in items:
            rec: Dict[str, Any] = {"id": u.id, "reason": u.reason, "written": u.status, "previous": None,
                                   "status": None}
            records.append(rec)
            k = self._index.get(u.id)
            status = parse_coverage(u.status)
            if k is None or self.criteria[k].kind != ASPECT or status is None or k in done:
                why = ("unknown criterion" if k is None else "not an aspect" if self.criteria[k].kind != ASPECT
                       else "a second coverage of the aspect" if k in done else f"coverage {u.status!r} is not a status")
                call.issues.append(_issue(u.id, ISSUE_BAD_COVERAGE, f"{why}; ignored"))
                continue
            done.add(k)
            rec["written"], rec["previous"] = status, self.coverage[k]
            opening = parse_reason(u.reason)
            if opening is None or opening.new or opening.proposed:
                call.issues.append(_issue(u.id, ISSUE_NO_REASON, "coverage: the reason does not open with "
                                                                 "'Was <coverage>:'"))
            elif opening.previous != self.coverage[k]:
                call.issues.append(_issue(u.id, ISSUE_WRONG_PREVIOUS, f"coverage: the reason says it was "
                                                                      f"{opening.previous!r}, it is {self.coverage[k]!r}"))
            if not call.call_docs[k]:
                if status != self.coverage[k]:
                    call.issues.append(_issue(u.id, ISSUE_COVERAGE_NO_PASSAGE, f"coverage {status!r} from "
                                              f"{self.coverage[k]!r} with no passage on the aspect in the call; kept"))
                rec["status"] = self.coverage[k]
                continue
            if status != UNKNOWN and not any(x.supports() for x in self.candidates[k]):
                call.issues.append(_issue(u.id, ISSUE_COVERAGE_NO_FACT, f"coverage {status!r} with no fact a passage "
                                                                        "supports; set to 'unknown'"))
                status = UNKNOWN
            if status != self.coverage[k]:
                call.changed.add(k)
            self.coverage[k] = rec["status"] = status
        # An aspect whose last supported fact left in this call, its coverage
        # not written: back to "unknown" as well.
        for k, c in enumerate(self.criteria):
            if (c.kind == ASPECT and k not in done and self.coverage[k] != UNKNOWN
                    and not any(x.supports() for x in self.candidates[k])):
                call.issues.append(_issue(c.id, ISSUE_COVERAGE_NO_FACT, f"coverage {self.coverage[k]!r} with no fact "
                                                                        "a passage supports; set to 'unknown'"))
                self.coverage[k] = UNKNOWN
                call.changed.add(k)
        return records

    def _flag_rivals(self, k: int, issues: List[Dict[str, Any]]) -> None:
        """The flags of a one-answer criterion *k* (``apply``)."""
        c = self.criteria[k]
        cands = self.candidates[k]
        for x in cands:
            if x.status == "certain":
                rivals = [y.name for y in cands if y is not x and y.supports()]
                if rivals:
                    issues.append(_issue(c.id, ISSUE_CERTAIN_WITH_RIVAL,
                                         f"{x.name!r} is certain while {rivals} have support"))
        groups: Dict[Tuple, List[Candidate]] = {}
        for x in cands:
            if x.supports():
                key = (frozenset(s.label for s in x.supports()), frozenset(s.label for s in x.supports(True)))
                groups.setdefault(key, []).append(x)
        for same in groups.values():
            if len({x.status for x in same}) > 1:
                issues.append(_issue(c.id, ISSUE_UNEQUAL_TIE, "the same support, different statuses: " +
                                     ", ".join(f"{x.name!r} {x.status}" for x in same)))

    def _passage_texts(self, k: int) -> List[str]:
        """The verified spans of the passage observations on criterion *k*
        and the titles of their passages (a passage may name its subject in
        its title only)."""
        out: List[str] = []
        for o in self.trace[k]:
            if o.source == DOC:
                out += [*o.verified_spans, self.title(o.label)]
        return out

    def _parts(self, k: int, name: str) -> List[str]:
        """The parts of a candidate a source writes out: a rest criterion's
        "<member>: <value>" the member and the value, else the name whole."""
        return name.split(":", 1) if self.criteria[k].kind == REST and ":" in name else [name]

    def _unknown_candidate(self, k: int, name: str) -> bool:
        """Whether candidate *name* occurs in none of the spans of the
        passage observations on criterion *k* (its support among them) nor
        in their passages' titles (a candidate a passage names in its title
        only, or a hop's candidate named by the passage of another span).
        Checked for the kinds whose answer is a name or figure a source
        writes out (``NAMED_KINDS``; a rest criterion's "<member>:
        <value>", the member and the value each)."""
        texts = self._passage_texts(k)
        return not all(any(contains_words(t, p) for t in texts) for p in self._parts(k, name) if alnum(p))

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    def belief_lists(self) -> List[Dict[str, Any]]:
        """Per criterion ``{id, status, selected, candidates, proposals}``:
        its derived status and selection, its candidates as shown
        (``Candidate.to_dict``, highest status first) and its proposals
        (``Proposal.to_dict``)."""
        return [{"id": c.id, "status": self.status(k), "selected": self.selected(k),
                 "candidates": [x.to_dict() for x in self.ordered(k)],
                 "proposals": [x.to_dict() for x in self.proposals[k]]}
                for k, c in enumerate(self.criteria)]

    def reference_lists(self) -> List[Dict[str, Any]]:
        """Per criterion ``{id, references, removed}``: every source assigned
        to it so far, in the order it came in, as ``{label, step, spans}``
        plus ``doc_id`` and ``title`` for a document, ``candidate`` and
        ``claim`` for a claim; and the candidates and proposals dropped
        (``RemovedCandidate.to_dict``)."""
        out = []
        for k, (c, obs) in enumerate(zip(self.criteria, self.trace)):
            refs = []
            for o in obs:
                r: Dict[str, Any] = {"label": o.label, "step": o.step}
                if o.source == DOC:
                    r.update(doc_id=o.doc_id, title=self.title(o.label))
                else:
                    r.update(candidate=o.candidate, claim=o.claim)
                r["spans"] = o.verified_spans
                refs.append(r)
            out.append({"id": c.id, "references": refs, "removed": [x.to_dict() for x in self.removed[k]]})
        return out
