"""What an agent calls the parts of its own run.

:mod:`evaluation.table` scores any agent that keeps a grid, and its contract is
the *shape* of the result (``table_snapshots`` / ``table``), not the identity of
the producer.  But two things in a result are agent-specific vocabulary rather
than shape:

* the ``phase`` tag on a trajectory step -- which part of the agent's schedule
  issued that retrieval, and
* the ``stage`` name on a snapshot -- which moment of the run it froze.

Those were GridFill's names, hardcoded.  A second agent with a different
schedule silently scored as if it had no phases at all.  This module makes the
vocabulary a parameter, defaulting to GridFill's so its numbers are unchanged.

A vocabulary is deliberately thin.  It says what things are *called* and in what
order they ran; it never says how to score them.  Anything that reads a snapshot
and computes a number (``score_barrier``, ``diff_stages``, the entity matchers)
is already vocabulary-free and stays that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TableVocabulary:
    """The names one agent gives the parts of its run.

    Attributes:
        name: The agent this describes, for error messages and provenance.
        retrieval_phases: Schedule-ordered phase tags that issue searches.
            Phases that never retrieve (GridFill's ``report``, TaS's final
            synthesis) are left out -- they would contribute empty buckets.
        cumulative_names: Display names for the running prefixes of
            ``retrieval_phases``, same length and order.
        scored_stages: Snapshot stages scored in their own right in the headline.
        stage_labels: Optional prettier names for stages in printed output.
        stage_sequence: The fixed stages, in the order the schedule ran them.
        numbered_after: The fixed stage that ``numbered_stages`` follow, or
            ``None`` when the agent emits no numbered stages. GridFill's
            ``fill_round_N`` / ``modify_round_N`` sit between ``init`` and
            ``final``, so this is ``"init"`` there.
        numbered_stages: Numbered stage families as ``(prefix, tiebreak,
            offset)``. Within the band, stages sort by ``int(suffix) + offset``,
            then by ``tiebreak`` -- which is how a barrier sorts immediately
            after the sweep it followed rather than the sweep sharing its number.
        revision_prefix: Stage-name prefix marking a *revision barrier* -- a
            moment the agent deliberately revisited an already-populated grid.
            ``None`` for an agent that has no such moment, which correctly
            leaves the ``barriers`` block absent rather than empty.
        records_evidence: Whether the agent attaches document ids to filled
            cells. ``False`` suppresses the cell-grounding block entirely: an
            agent that cites nothing has no grounding rate, and reporting 0.0
            would state that every cell is *un*grounded, which is a different
            and false claim.
    """

    name: str
    retrieval_phases: Tuple[str, ...]
    cumulative_names: Tuple[str, ...]
    scored_stages: Tuple[str, ...] = ("init", "final")
    stage_labels: Mapping[str, str] = field(default_factory=dict)
    stage_sequence: Tuple[str, ...] = ()
    numbered_after: Optional[str] = None
    numbered_stages: Tuple[Tuple[str, int, int], ...] = ()
    revision_prefix: Optional[str] = None
    records_evidence: bool = True

    def __post_init__(self) -> None:
        if len(self.retrieval_phases) != len(self.cumulative_names):
            raise ValueError(
                f"{self.name}: retrieval_phases and cumulative_names must be the "
                f"same length, got {len(self.retrieval_phases)} and "
                f"{len(self.cumulative_names)}"
            )
        if self.numbered_stages and self.numbered_after not in self.stage_sequence:
            raise ValueError(
                f"{self.name}: numbered_after={self.numbered_after!r} is not one of "
                f"stage_sequence={self.stage_sequence!r}"
            )

    def is_revision_stage(self, stage: str) -> bool:
        """Whether *stage* is a revision barrier for this agent."""
        return bool(self.revision_prefix) and stage.startswith(self.revision_prefix)

    def _bands(self) -> Tuple[Optional[str], ...]:
        """The schedule as bands: each fixed stage, plus the numbered slot."""
        bands: List[Optional[str]] = []
        for stage in self.stage_sequence:
            bands.append(stage)
            if self.numbered_stages and stage == self.numbered_after:
                bands.append(None)  # the numbered family's slot
        return tuple(bands)

    def order_stage(self, stage: str) -> tuple:
        """Sort key putting *stage* where the schedule ran it.

        Anything unrecognised sorts last, by name, rather than crashing the
        report -- or, worse, silently displacing a real stage.
        """
        bands = self._bands()
        if stage in bands:
            return (bands.index(stage), 0, 0, stage)
        for prefix, tiebreak, offset in self.numbered_stages:
            if stage.startswith(prefix):
                suffix = stage[len(prefix):]
                try:
                    number = int(suffix)
                except ValueError:
                    break
                return (bands.index(None), number + offset, tiebreak, stage)
        return (len(bands), 0, 0, stage)


#: GridFill: a planner lays out the grid, cells are filled, then revision
#: barriers revisit it. ``report`` issues no searches, so it is not a retrieval
#: phase. These are the values that used to be module constants.
GRIDFILL_VOCAB = TableVocabulary(
    name="gridfill",
    retrieval_phases=("init", "fill", "modify"),
    cumulative_names=("after_init", "after_fill", "after_modify"),
    scored_stages=("init", "final"),
    stage_labels={"init": "init [planned]", "final": "final [revised]"},
    stage_sequence=("init", "final"),
    numbered_after="init",
    numbered_stages=(("fill_round_", 0, 0), ("modify_round_", 1, -1)),
    revision_prefix="modify_round_",
    records_evidence=True,
)

#: Table-as-Search, named for its own workflow: the main agent fixes the schema
#: (Step 2), tabular sub-agents discover rows (Step 3), deep sub-agents fill
#: their cells (Step 4), and the main agent synthesises an answer (Step 5).
#: There is no revision barrier -- TaS never revisits a populated grid as a
#: distinct act -- and it records no citations, upstream or here.
TAS_VOCAB = TableVocabulary(
    name="tas",
    retrieval_phases=("main", "tabular", "deep"),
    cumulative_names=("after_main", "after_tabular", "after_deep"),
    scored_stages=("schema", "rows", "final"),
    stage_labels={
        "schema": "schema [columns fixed]",
        "rows": "rows [candidates discovered]",
        "final": "final [cells filled]",
    },
    stage_sequence=("schema", "rows", "final"),
    revision_prefix=None,
    records_evidence=False,
)

#: Agents whose grid this module knows how to name. Anything else gets the
#: GridFill vocabulary, which is what it got before this module existed.
_BY_AGENT: Dict[str, TableVocabulary] = {
    "gridfill": GRIDFILL_VOCAB,
    "tas": TAS_VOCAB,
}

DEFAULT_VOCAB = GRIDFILL_VOCAB


def vocabulary_for(agentic_model: Optional[str]) -> TableVocabulary:
    """The vocabulary for *agentic_model*, or the default if it is not known."""
    if not agentic_model:
        return DEFAULT_VOCAB
    return _BY_AGENT.get(agentic_model, DEFAULT_VOCAB)
