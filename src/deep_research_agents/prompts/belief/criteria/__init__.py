"""Criteria-updater prompts for the belief agent.

The four system prompts and the user templates are copied from the
controller's criteria-coverage prompts, so the belief agent does not depend on
``controller_component``.  Fixed here only: the dynamic update prompt carries
the ``{frozen_instruction}`` placeholder the code fills (the original drops
it), its example JSON has no trailing comma, and a few wording slips.  They
use str.format (literal braces are doubled).

Two modes:
- **static**: criteria are copied from the query and the list is fixed
  (BrowseComp-Plus, whose queries enumerate their conditions).
- **dynamic**: the query is decomposed into criteria and the list may change
  on each update until it stabilises (TRQA).
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

CRITERIA_INIT_STATIC_SYSTEM = (_DIR / "init_static_system.txt").read_text(encoding="utf-8").strip()
CRITERIA_UPDATE_STATIC_SYSTEM = (_DIR / "update_static_system.txt").read_text(encoding="utf-8").strip()
CRITERIA_INIT_DYNAMIC_SYSTEM = (_DIR / "init_dynamic_system.txt").read_text(encoding="utf-8").strip()
CRITERIA_UPDATE_DYNAMIC_SYSTEM = (_DIR / "update_dynamic_system.txt").read_text(encoding="utf-8").strip()

FROZEN_INSTRUCTION = "The criterion list is FROZEN: you may only change statuses (tick). Do NOT add or remove criteria."
UNFROZEN_INSTRUCTION = "The criterion list is open for modification: you may add, remove, or tick criteria."


CRITERIA_INIT_DYNAMIC_USER_TEMPLATE = """\
Query: {query}

Decompose this query into its key information-need criteria. \
Identify {min_criteria} to {max_criteria} distinct criteria that a comprehensive answer must address.

All criteria start with status "not_covered" and empty evidence.\
"""

CRITERIA_INIT_STATIC_USER_TEMPLATE = """\
Query: {query}

Identify each criterion, clue, or condition stated in the query and list them as criteria. \
This list will be FIXED for the entire search.

All criteria start with status "not_covered" and empty evidence.\
"""

CRITERIA_UPDATE_DYNAMIC_USER_TEMPLATE = """\
Query: {query}

Current criteria:
{current_criteria_formatted}

New search query: {subqueries}

New retrieved documents:
{doc_snippets}

Review the criterion list given the new evidence. \
Return only the actions for criteria that changed.\
"""

CRITERIA_UPDATE_STATIC_USER_TEMPLATE = """\
Query: {query}

Current criteria (FIXED — do not add or remove):
{current_criteria_formatted}

New search query: {subqueries}

New retrieved documents:
{doc_snippets}

Review the evidence and update statuses for criteria that changed. \
Return only tick actions for criteria whose status or evidence changed.\
"""
