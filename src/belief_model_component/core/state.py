"""``BeliefState``, b_t: per belief its candidates with the passages that
support them or speak against them, its most probable candidate and WEP,
and its proposals (shown), and the trace and references behind them
(saved).  ``apply`` takes one belief update reply."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..text.matching import alnum, contains_words, content_words, squash
from .issues import (
    ISSUE_BELIEF_MISSING, ISSUE_BELIEF_REPEATED, ISSUE_CANDIDATE_MISSING,
    ISSUE_CANDIDATE_NEAR_DUPLICATE, ISSUE_CANDIDATE_NOT_IN_PASSAGES, ISSUE_CANDIDATE_READDED,
    ISSUE_CLAIM_INCOMPLETE, ISSUE_FACT_MERGED, ISSUE_MOST_PROBABLE_WRONG,
    ISSUE_OBSERVATION_ON_AGGREGATE, ISSUE_SPAN_OFF_BELIEF, ISSUE_SPAN_OFF_FACT, ISSUE_SPAN_UNVERIFIED,
    ISSUE_STANCE_INVALID, ISSUE_SUBJECT_CANDIDATE, ISSUE_TURN_SPAN_UNVERIFIED, ISSUE_UNKNOWN_BELIEF,
    ISSUE_UNKNOWN_PASSAGE, ISSUE_UNKNOWN_WITH_SUPPORT, ISSUE_UPDATE_ON_AGGREGATE,
    ISSUE_UPDATE_WITHOUT_OBSERVATION, ISSUE_WEP_INVALID, ISSUE_WEP_WITHOUT_SUPPORT, NOTE_CODES,
)
from .types import (
    AGGREGATE, ASPECT, CLOSED, DOC, DROP, KEEP, NAMED_KINDS, NO_SUPPORT_MAX_STATUS, REASONING, STATUS_LEVEL, THINK,
    UNKNOWN, BeliefUpdate, Candidate, Belief, Observation, Proposal, RemovedCandidate, Support,
    is_target, member_subject, parse_status, subject_names,
)

# Proposals kept per belief (beyond it the least recently claimed is dropped).
MAX_PROPOSALS = 8

# A figure candidate (``BeliefState._unknown_candidate``): its numbers, and
# the most words (a unit) it may have besides them.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[^\W\d_]+")
_FIGURE_WORDS = 2


def _issue(cid: str, code: str, detail: str, o: Optional[Observation] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": cid, "code": code, "detail": detail}
    if o is not None:
        out["label"] = o.label
    return out


def _level(status: Optional[str]) -> int:
    return STATUS_LEVEL[status or UNKNOWN]


def _label_number(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


@dataclass
class _CallContext:
    """What the phases of one ``BeliefState.apply`` share: the call's step,
    per belief the passages of the call observed on it (label -> its
    observation in the trace), the passage observations that entered (in
    reply order, attached to their candidates after the claims) and, per
    belief, the ones attached (``{candidate, stance, passage}``), and
    what the phases produce (issues, the observations that entered, the
    changed beliefs, the records of the belief items)."""
    step: int
    call_docs: List[Dict[str, Observation]]
    attached: List[List[Dict[str, Any]]]
    passages: List[Tuple[int, Observation]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    entered: List[Observation] = field(default_factory=list)
    changed: Set[int] = field(default_factory=set)
    records: List[Dict[str, Any]] = field(default_factory=list)


class BeliefState:
    """Belief state b_t, kept at two levels.

    Shown (``views.belief_text.format_belief``): per belief its statement
    (its WEP and most probable candidate), the passages on each candidate
    (+ or -, with their spans and titles), and its proposals (names the
    agent's reasoning proposed that no passage is on).  The agent's claims
    are kept but never shown to the update.

    Saved: per belief the trace of the observations that entered it
    (``trace``), its references (``reference_lists``), the records of the
    belief items (``apply``) and the candidates and proposals it dropped
    (``removed``).

    Each call, the belief update writes the claims of the agent's turns,
    the passage observations (a passage, a belief, a candidate and a
    stance, + or -, with their spans), then per belief with a new passage
    observation its most probable candidate and WEP, with a reason;
    ``apply`` keeps what is valid:

    - an observation enters only when it is grounded (a verified span of
      its passage or turn) and names a known belief that is not an
      aggregate (a belief computed from others takes no observation); a
      reasoning observation also needs a candidate and a claim (``KEEP``
      or ``DROP``), a passage observation a candidate and a stance "+" or
      "-";
    - a claim sets the agent's claim on a listed candidate, or adds or
      updates (``KEEP``) or drops (``DROP``) a proposal; it never moves a
      status and never removes a candidate;
    - a passage observation, after the claims, attaches to its candidate
      (a new name becomes a candidate, a proposal is promoted).  A - on an aspect's fact removes it;
      a one-answer candidate is never removed by a passage;
    - a belief item is taken once per belief, only with a passage
      observation on it in the call, never on an aggregate
      (``_set_wep``): the most probable candidate written must be listed
      with a + passage, the WEP is taken as written ("unknown" when no
      passage supports a candidate); a belief with a new passage
      observation and no item keeps its WEP (``belief_missing``);
    - a belief has one WEP: its most probable candidate (an aspect's
      facts) carries it, its other candidates have no status;
    - an aggregate's statuses are code's (``_derive_aggregates``): a
      target's derived candidate; a computed answer has none.

    Documents get their label ("d<n>") when they are first shown
    (``label_doc``).
    """

    def __init__(self, beliefs: List[Belief]) -> None:
        self.beliefs = list(beliefs)
        self._index = {c.id: k for k, c in enumerate(self.beliefs)}
        # Per belief its candidates, in the order they were added, and its proposals.
        self.candidates: List[List[Candidate]] = [[] for _ in self.beliefs]
        self.proposals: List[List[Proposal]] = [[] for _ in self.beliefs]
        # The trace: per belief, the observations that entered it.
        self.trace: List[List[Observation]] = [[] for _ in self.beliefs]
        # Per belief the candidates and proposals dropped, in order.
        self.removed: List[List[RemovedCandidate]] = [[] for _ in self.beliefs]
        # Per belief its WEP (one answer, aspect) and its most probable
        # candidate (one answer).
        self.wep: List[str] = [UNKNOWN for _ in self.beliefs]
        self._most_probable: List[Optional[str]] = [None for _ in self.beliefs]
        # Per belief the member it is about (a member belief), or None.
        self._subjects: List[Optional[str]] = [member_subject(self.beliefs, k) for k in range(len(self.beliefs))]
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
    # Derived per belief
    # ------------------------------------------------------------------

    def most_probable(self, k: int) -> Optional[Candidate]:
        """The most probable candidate of one-answer belief *k* (the
        update's pick), None without one (an aspect, an aggregate)."""
        name = self._most_probable[k]
        return self._listed(k, name) if name else None

    def ordered(self, k: int) -> List[Candidate]:
        """The candidates of belief *k* as shown: an aspect's facts in the
        order they were added; a one-answer belief's most probable
        candidate first, then by their + passages, then in the order they
        were added; an aggregate's as listed (a target has at most one
        derived candidate, a computed answer none)."""
        cands = self.candidates[k]
        kind = self.beliefs[k].kind
        if kind in (ASPECT, AGGREGATE):
            return list(cands)
        top = self.most_probable(k)
        return sorted(cands, key=lambda c: (c is not top, -len(c.supports())))

    def status(self, k: int) -> str:
        """The status of belief *k*: its WEP (one answer, aspect); an
        aggregate's the highest of its candidates ("unknown" without)."""
        if self.beliefs[k].kind != AGGREGATE:
            return self.wep[k]
        return max((c.status or UNKNOWN for c in self.candidates[k]), key=_level, default=UNKNOWN)

    def selected(self, k: int) -> List[str]:
        """The selected candidate(s) of belief *k*: for a one-answer
        belief its most probable candidate at "possible" or above (an
        aggregate's first one shown), for an aspect every fact."""
        if self.beliefs[k].kind == ASPECT:
            return [c.name for c in self.candidates[k]]
        if self.beliefs[k].kind == CLOSED:
            top = self.most_probable(k)
            return [top.name] if top is not None and _level(self.wep[k]) >= _level(NO_SUPPORT_MAX_STATUS) else []
        kept = [c.name for c in self.ordered(k) if _level(c.status) >= _level(NO_SUPPORT_MAX_STATUS)]
        return kept[:1]

    def _target(self, k: int) -> bool:
        return is_target(self.beliefs[k].kind, self.beliefs[k].text)

    # ------------------------------------------------------------------
    # The update
    # ------------------------------------------------------------------

    def apply(self, claims: List[Observation], docs: List[Observation],
              updates: List[BeliefUpdate], step: int) -> Dict[str, Any]:
        """Apply one belief update of the call of step *step*: the claims of
        the turns (*claims*, reasoning observations), the passage
        observations (*docs*) and the belief items (*updates*).

        Returns ``{observations, updates, changed, issues, notes}``: the
        observations that entered the belief (``Observation.to_dict``), one
        record per belief item of the reply (``{id, reason, passages,
        written, previous, most_probable, status}``: ``passages`` the call's
        passage observations attached on the belief (``{candidate, stance,
        passage}``), ``written`` the item's most probable candidate and WEP
        as written, ``previous`` the belief's status before, then the most
        probable candidate and the status kept; null when the item was not
        taken), the ids of the beliefs whose belief changed, one issue
        record per thing the reply got wrong, and one note record per thing
        worth knowing that is no error (``core.issues``).

        Phases, in order: admit the observations, apply the claims, attach
        the passage observations to their candidates, apply the belief
        items, derive the aggregates from their inputs."""
        n = len(self.beliefs)
        call = _CallContext(step=step, call_docs=[{} for _ in range(n)], attached=[[] for _ in range(n)])
        self._admit_observations([*claims, *docs], call)
        self._apply_claims(call)
        for k, o in call.passages:
            if self._attach(k, o, call):
                call.changed.add(k)
        done: Set[int] = set()
        for u in updates:
            self._apply_belief(u, call, done)
        for k, c in enumerate(self.beliefs):
            if call.attached[k] and k not in done and c.kind != AGGREGATE:
                call.issues.append(_issue(c.id, ISSUE_BELIEF_MISSING,
                                          f"new passage observations, no belief item; WEP {self.wep[k]!r} kept"))
            if c.kind != AGGREGATE:
                self._refresh_statuses(k)
        self._derive_aggregates(call)
        return {"observations": [o.to_dict() for o in call.entered], "updates": call.records,
                "changed": [self.beliefs[k].id for k in sorted(call.changed)],
                "issues": [i for i in call.issues if i["code"] not in NOTE_CODES],
                "notes": [i for i in call.issues if i["code"] in NOTE_CODES]}

    def _admit_observations(self, observations: List[Observation], call: "_CallContext") -> None:
        """Phase 1: each observation enters the trace when it passes the
        gates (a known belief that is not an aggregate, a passage of the
        call, a verified span; a claim with a candidate and keep or drop; a
        passage observation with a stance, a candidate, and not the member
        of a member belief); otherwise one
        issue.  A second observation of a passage on one belief adds its
        new spans to the first in the trace."""
        issues = call.issues
        for o in observations:
            k = self._index.get(o.belief)
            if k is None:
                issues.append(_issue(o.belief, ISSUE_UNKNOWN_BELIEF, "unknown belief id", o))
            elif self.beliefs[k].kind == AGGREGATE:
                issues.append(_issue(o.belief, ISSUE_OBSERVATION_ON_AGGREGATE,
                                     "the belief is computed from other beliefs: no observation", o))
            elif o.source == DOC and o.doc_id is None:
                issues.append(_issue(o.belief, ISSUE_UNKNOWN_PASSAGE, "no passage of the call has this label", o))
            elif not o.grounded:
                issues.append(_issue(o.belief, ISSUE_SPAN_UNVERIFIED, "no verified span", o) if o.source == DOC else
                              _issue(o.belief, ISSUE_TURN_SPAN_UNVERIFIED, "no span of the agent's turn", o))
            elif o.source == THINK and (not alnum(o.candidate or "") or o.claim is None):
                why = (f"agent_claim {o.claim_written!r} is not a claim ('{KEEP}' or '{DROP}')"
                       if alnum(o.candidate or "") else "no candidate")
                issues.append(_issue(o.belief, ISSUE_CLAIM_INCOMPLETE,
                                     f"{o.candidate!r}: {why}" if o.candidate else why, o))
            elif o.source == DOC and o.bad_stance is not None:
                issues.append(_issue(o.belief, ISSUE_STANCE_INVALID,
                                     f"stance {o.bad_stance!r} is neither '+' nor '-'", o))
            elif o.source == DOC and not alnum(o.candidate or ""):
                issues.append(_issue(o.belief, ISSUE_CANDIDATE_MISSING, "a passage observation without a candidate",
                                     o))
            elif o.source == DOC and (problem := self._subject_problem(k, o.candidate or "")):
                issues.append(_issue(o.belief, ISSUE_SUBJECT_CANDIDATE, f"{problem}; not added", o))
            elif o.source == DOC:
                known = next((x for x in self.trace[k] if x.source == DOC and x.label == o.label), None)
                if known is None:
                    known = Observation(belief=o.belief, source=DOC, label=o.label, step=o.step,
                                        doc_id=o.doc_id, grounded=True, candidate=o.candidate, against=o.against)
                    self.trace[k].append(known)
                # The trace keeps one observation per passage: its spans on
                # the belief, for any candidate.
                for span in o.verified_spans:
                    if span not in known.spans:
                        known.spans.append(span)
                        known.span_verified.append(True)
                call.call_docs[k][o.label] = known
                call.passages.append((k, o))
                call.entered.append(o)
            else:
                self.trace[k].append(o)
                call.entered.append(o)

    def _apply_claims(self, call: "_CallContext") -> None:
        """Phase 2: the claims that entered, in the order of their turns
        (the latest one stands)."""
        for o in sorted((o for o in call.entered if o.source == THINK), key=lambda o: _label_number(o.label)):
            if self._claim(self._index[o.belief], o, call):
                call.changed.add(self._index[o.belief])

    def _attach(self, k: int, o: Observation, call: "_CallContext") -> bool:
        """Phase 3, one passage observation *o* on belief *k*: it joins the
        support of its candidate.  A new name becomes a candidate (a
        proposal is promoted; an aspect's fact whose words contain a listed
        one, or the reverse, joins it).  A passage already on the candidate with the
        same stance is not added twice; a - on an aspect's fact removes it
        (only the fact or proposal it names, never a near one).  Returns
        whether the belief changed."""
        c, issues, step = self.beliefs[k], call.issues, call.step
        name = squash(o.candidate)
        existing = self._listed(k, name) if alnum(name) else None
        if existing is None:
            proposal = self._proposed(k, name)
            near = [x for x in self.candidates[k] if contains_words(x.name, name) or contains_words(name, x.name)]
            if near and c.kind == ASPECT and o.against:
                # A - removes only the fact it names: a near one is another fact.
                near = []
            if near and c.kind == ASPECT:
                existing = near[0]
                issues.append(_issue(c.id, ISSUE_FACT_MERGED, f"{name!r} repeats listed {existing.name!r}; merged "
                                                              "into it", o))
                if proposal is not None:
                    self.proposals[k].remove(proposal)
            else:
                # A passage speaks against a name without writing it.
                if c.kind in NAMED_KINDS and not o.against and self._unknown_candidate(k, name):
                    issues.append(_issue(c.id, ISSUE_CANDIDATE_NOT_IN_PASSAGES, f"{name!r} in no span or title of "
                                                                                "the belief's passages", o))
                if near:
                    issues.append(_issue(c.id, ISSUE_CANDIDATE_NEAR_DUPLICATE,
                                         f"{name!r} is close to listed {[x.name for x in near]}", o))
                if proposal is None and self._dropped(k, name):
                    issues.append(_issue(c.id, ISSUE_CANDIDATE_READDED, f"{name!r}: dropped before, written again",
                                         o))
                existing = Candidate(name=proposal.name if proposal is not None else name, status=None,
                                     added_step=step)
                if proposal is not None:
                    existing.claim, existing.claim_step = proposal.claim, proposal.step
                    self.proposals[k].remove(proposal)
                self.candidates[k].append(existing)
        if any(x.label == o.label and x.against == o.against for x in existing.support):
            return False
        entry = Support(label=o.label, doc_id=o.doc_id, title=self.title(o.label), step=step,
                        spans=o.verified_spans, against=o.against)
        existing.support.append(entry)
        self._check_spans(k, existing.name, entry, issues)
        call.attached[k].append({"candidate": existing.name, "stance": o.stance, "passage": o.label})
        if c.kind == ASPECT and o.against:
            self.candidates[k].remove(existing)
            self.removed[k].append(RemovedCandidate(
                name=existing.name, status=existing.status, basis=existing.basis, added_step=existing.added_step,
                removed_step=step, reason=f"{o.label} states it is wrong"))
        return True

    def _apply_belief(self, u: BeliefUpdate, call: "_CallContext", done: Set[int]) -> None:
        """Phase 4, one belief item: its record, the gates (a known
        belief that is not an aggregate, once per belief, a passage
        observation on it in the call), then its most probable candidate and
        WEP (``_set_wep``)."""
        issues = call.issues
        rec: Dict[str, Any] = {"id": u.id, "reason": u.reason, "passages": [],
                               "written": {"most_probable": u.most_probable, "wep": u.wep},
                               "previous": None, "most_probable": None, "status": None}
        call.records.append(rec)
        k = self._index.get(u.id)
        if k is None:
            issues.append(_issue(u.id, ISSUE_UNKNOWN_BELIEF, "a belief item of an unknown belief id"))
            return
        rec["passages"] = list(call.attached[k])
        if self.beliefs[k].kind == AGGREGATE:
            issues.append(_issue(u.id, ISSUE_UPDATE_ON_AGGREGATE,
                                 "the belief is computed from other beliefs: no belief item; ignored"))
            return
        if k in done:
            issues.append(_issue(u.id, ISSUE_BELIEF_REPEATED, "a second item of the belief; ignored"))
            return
        if not call.call_docs[k]:
            issues.append(_issue(u.id, ISSUE_UPDATE_WITHOUT_OBSERVATION, "no passage observation on the belief in "
                                                                         "the call; ignored"))
            return
        done.add(k)
        rec["previous"] = self.status(k)
        if self._set_wep(k, u, call, rec):
            call.changed.add(k)

    def _set_wep(self, k: int, u: BeliefUpdate, call: "_CallContext", rec: Dict[str, Any]) -> bool:
        """The most probable candidate and WEP of belief *k* from its item
        *u*: the written most probable candidate when it is listed with a +
        passage, else code's (the first added of those with the most +
        passages; ``most_probable_wrong``); an aspect has none.  The WEP as
        written: no word of the scale keeps the previous one
        (``wep_invalid``); a word above "unknown" with no supported
        candidate (an aspect: no kept fact) is set to "unknown"
        (``wep_without_support``); "unknown" with one is kept
        (``unknown_with_support``).  Returns whether the belief's most
        probable candidate or status changed."""
        c, issues = self.beliefs[k], call.issues
        before = (self._most_probable[k], self.wep[k])
        supported = [x for x in self.candidates[k] if x.supports()]
        if c.kind != ASPECT:
            written = self._listed(k, u.most_probable) if alnum(u.most_probable or "") else None
            if written is not None and written.supports():
                top = written
            else:
                best = max((len(x.supports()) for x in supported), default=0)
                top = next((x for x in supported if len(x.supports()) == best), None)
                if top is not None or alnum(u.most_probable or ""):
                    said = repr(u.most_probable) if alnum(u.most_probable or "") else "null"
                    issues.append(_issue(c.id, ISSUE_MOST_PROBABLE_WRONG,
                                         f"wrote {said}, not a listed candidate with a + passage; set to "
                                         f"{top.name if top else None!r}"))
            self._most_probable[k] = top.name if top is not None else None
        wep = parse_status(u.wep)
        if wep is None:
            wep = self.wep[k]
            issues.append(_issue(c.id, ISSUE_WEP_INVALID, f"{u.wep!r} is no word of the scale; {wep!r} kept"))
        elif not supported and wep != UNKNOWN:
            issues.append(_issue(c.id, ISSUE_WEP_WITHOUT_SUPPORT, f"{wep!r} written, no passage supports a "
                                                                  "candidate; set to 'unknown'"))
            wep = UNKNOWN
        elif supported and wep == UNKNOWN:
            issues.append(_issue(c.id, ISSUE_UNKNOWN_WITH_SUPPORT, "'unknown' written, a passage supports "
                                                                   f"{[x.name for x in supported]}; kept"))
        self.wep[k] = wep
        rec["most_probable"], rec["status"] = self._most_probable[k], wep
        return (self._most_probable[k], self.wep[k]) != before

    def _refresh_statuses(self, k: int) -> None:
        """Each candidate's status on belief *k*: an aspect's facts and the
        most probable candidate the belief's WEP; any other candidate none
        (a belief has one WEP)."""
        top = self.most_probable(k)
        for x in self.candidates[k]:
            x.status = self.wep[k] if self.beliefs[k].kind == ASPECT or x is top else None

    def _derive_aggregates(self, call: "_CallContext") -> None:
        """Phase 5: the statuses of every aggregate, from its inputs (the
        closed beliefs before it).

        A target's candidate is the name that is the most probable
        candidate of every clue, at the lowest of the clues' WEPs, with the
        support of all of them; none when a clue has no most probable
        candidate, an "unknown" WEP or another name.  A candidate kept from
        before keeps the step it was added in.

        A computed answer (a multi-member query's) has no candidates: it
        stays "unknown"."""
        for k in range(len(self.beliefs)):
            if self._target(k):
                self._derive_target(k, [j for j in range(k) if self.beliefs[j].kind == CLOSED], call)

    def _derive_target(self, k: int, clues: List[int], call: "_CallContext") -> None:
        """The candidate of target *k* from its *clues* (``_derive_aggregates``)."""
        old = {alnum(x.name): x for x in self.candidates[k]}
        derived: List[Candidate] = []
        tops = [self.most_probable(j) for j in clues]
        if (clues and all(t is not None for t in tops) and len({alnum(t.name) for t in tops}) == 1
                and all(self.wep[j] != UNKNOWN for j in clues)):
            prev = old.get(alnum(tops[0].name))
            derived.append(Candidate(
                name=tops[0].name, status=min((self.wep[j] for j in clues), key=_level),
                added_step=prev.added_step if prev is not None else call.step,
                support=[s for t in tops for s in t.support]))
        before = [(alnum(x.name), x.status, len(x.support)) for x in self.candidates[k]]
        self.candidates[k] = derived
        if [(alnum(x.name), x.status, len(x.support)) for x in derived] != before:
            call.changed.add(k)

    def _listed(self, k: int, name: str) -> Optional[Candidate]:
        """The listed candidate of belief *k* named *name* (letters and
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
        """Apply the claim *o* on belief *k*: a listed candidate keeps
        it as its claim (a drop removes no candidate); a proposal the
        claim drops leaves the proposals (kept in the removal log), a keep
        updates or adds one.  Returns whether the proposals changed."""
        step, issues = o.step, call.issues
        existing = self._listed(k, o.candidate or "")
        if existing is not None:
            existing.claim, existing.claim_step = o.claim, step
            return False
        proposal = self._proposed(k, o.candidate or "")
        if proposal is not None:
            if o.claim == DROP:
                self.proposals[k].remove(proposal)
                self.removed[k].append(RemovedCandidate(
                    name=proposal.name, status=None, basis=REASONING, added_step=proposal.added_step, removed_step=step,
                    reason=f"dropped in {o.label}: {' ... '.join(o.verified_spans)}"))
                return True
            proposal.claim, proposal.step = o.claim, step
            return False
        if o.claim == DROP:
            return False
        problem = self._subject_problem(k, o.candidate or "")
        if problem:
            issues.append(_issue(self.beliefs[k].id, ISSUE_SUBJECT_CANDIDATE, f"claim: {problem}; not proposed", o))
            return False
        self._add_proposal(k, squash(o.candidate), o.claim, step, issues, o)
        return True

    def _subject_problem(self, k: int, name: str) -> Optional[str]:
        """Why *name* cannot be a candidate of belief *k* (None when it
        can): it is the member the member belief is about."""
        subject = self._subjects[k]
        if subject is not None and alnum(name) in {alnum(n) for n in subject_names(subject)}:
            return f"{name!r} is the member the belief is about, not its {self.beliefs[k].text.strip()[len(subject):].lstrip(':').strip()}"
        return None

    def _add_proposal(self, k: int, name: str, claim: Optional[str], step: int, issues: List[Dict[str, Any]],
                      o: Optional[Observation] = None) -> None:
        """Propose *name* for belief *k* (from a claim *o*): noted when
        it was dropped before; past
        ``MAX_PROPOSALS`` the least recently claimed proposal is dropped
        (the agent's current names stay)."""
        if self._dropped(k, name):
            issues.append(_issue(self.beliefs[k].id, ISSUE_CANDIDATE_READDED,
                    f"{name!r}: dropped before, written again", o))
        self.proposals[k].append(Proposal(name=name, claim=claim, step=step, added_step=step))
        if len(self.proposals[k]) > MAX_PROPOSALS:
            old = min(self.proposals[k], key=lambda x: x.step)
            self.proposals[k].remove(old)
            self.removed[k].append(RemovedCandidate(name=old.name, status=None, basis=REASONING,
                                                    added_step=old.added_step, removed_step=step,
                                                    reason=f"more than {MAX_PROPOSALS} proposals: the least recently claimed dropped"))

    def _check_spans(self, k: int, name: str, x: Support, issues: List[Dict[str, Any]]) -> None:
        """Note a supporting span of an aspect's fact *name* that shares no
        content word with it (support attached to another fact); flag a -
        passage on a one-answer candidate whose spans share no content word
        with the belief (a passage that leaves the part out states
        nothing on it)."""
        if self.beliefs[k].kind != ASPECT:
            if x.against and not content_words(self.beliefs[k].text) & content_words(" ".join(x.spans)):
                issues.append(_issue(self.beliefs[k].id, ISSUE_SPAN_OFF_BELIEF,
                                     f"{name!r}: the - spans of {x.label} share no content word with the belief"))
            return
        words = content_words(name)
        if not x.against and words and not words & content_words(" ".join(x.spans)):
            issues.append(_issue(self.beliefs[k].id, ISSUE_SPAN_OFF_FACT,
                                 f"{name!r}: the spans of {x.label} share no content word with it"))

    def _passage_texts(self, k: int) -> List[str]:
        """The verified spans of the passage observations on belief *k*
        and the titles of their passages (a passage may name its subject in
        its title only)."""
        out: List[str] = []
        for o in self.trace[k]:
            if o.source == DOC:
                out += [*o.verified_spans, self.title(o.label)]
        return out

    def _unknown_candidate(self, k: int, name: str) -> bool:
        """Whether candidate *name* occurs in none of the spans of the
        passage observations on belief *k* (its support among them) nor
        in their passages' titles (a candidate a passage names in its title
        only, or a hop's candidate named by the passage of another span).
        Checked for the kinds whose answer is a name or figure a source
        writes out (``NAMED_KINDS``).  A figure (numbers and at most
        ``_FIGURE_WORDS`` words, a unit: "120 m") is found when one text
        has all its numbers, however it writes the unit ("120 metres")."""
        texts = self._passage_texts(k)
        if any(contains_words(t, name) for t in texts):
            return False
        numbers = _NUMBER_RE.findall(name)
        if not numbers or len(_WORD_RE.findall(name)) > _FIGURE_WORDS:
            return True
        return not any(all(contains_words(t, n) for n in numbers) for t in texts)

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    def belief_lists(self) -> List[Dict[str, Any]]:
        """Per belief ``{id, status, most_probable, selected, candidates,
        proposals}``: its status (``status``), its most probable candidate
        (null for an aspect or aggregate), its selection, its candidates as
        shown (``Candidate.to_dict``, ``ordered``) and its proposals
        (``Proposal.to_dict``)."""
        out = []
        for k, c in enumerate(self.beliefs):
            out.append({"id": c.id, "status": self.status(k), "most_probable": self._most_probable[k],
                        "selected": self.selected(k),
                        "candidates": [x.to_dict() for x in self.ordered(k)],
                        "proposals": [x.to_dict() for x in self.proposals[k]]})
        return out

    def reference_lists(self) -> List[Dict[str, Any]]:
        """Per belief ``{id, references, removed}``: every source assigned
        to it so far, in the order it came in, as ``{label, step, spans}``
        plus ``doc_id`` and ``title`` for a document, ``candidate`` and
        ``claim`` for a claim; and the candidates and proposals dropped
        (``RemovedCandidate.to_dict``)."""
        out = []
        for k, (c, obs) in enumerate(zip(self.beliefs, self.trace)):
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
