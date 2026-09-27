"""Prompt templates for the belief agent (plain-text files next to this module).

Placeholders are filled with str.replace, not str.format, so a literal brace
added to a template later cannot break rendering.  The criteria-updater
prompts live in the ``criteria`` subpackage.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

SYSTEM_TEMPLATE: str = (_DIR / "system.txt").read_text()
BELIEF_SECTION: str = (_DIR / "belief_section.txt").read_text()
USER_TEMPLATE: str = (_DIR / "user.txt").read_text()
FORMAT_ERROR_TEMPLATE: str = (_DIR / "format_error.txt").read_text()


def render_system(max_turns: int, show_belief: bool = True) -> str:
    """The policy's system prompt; *show_belief* False drops the belief section."""
    return (SYSTEM_TEMPLATE
            .replace("{max_turns}", str(max_turns))
            .replace("{belief_section}", BELIEF_SECTION if show_belief else ""))


def render_user(question: str) -> str:
    return USER_TEMPLATE.replace("{question}", question).rstrip("\n")


def render_format_error(errors) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{errors}", "\n".join(f"- {e}" for e in errors))
