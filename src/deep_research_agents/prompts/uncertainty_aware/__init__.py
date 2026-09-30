"""Prompt templates for the uncertainty-aware agent (plain-text files next to this module).

Placeholders are filled with str.replace, not str.format, so a literal brace
added to a template later cannot break rendering.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

SYSTEM_TEMPLATE: str = (_DIR / "system.txt").read_text()
CERTAINTY_SECTION: str = (_DIR / "certainty_section.txt").read_text()
USER_TEMPLATE: str = (_DIR / "user.txt").read_text()
FORMAT_ERROR_TEMPLATE: str = (_DIR / "format_error.txt").read_text()


def render_system(inform: bool) -> str:
    """The policy's system prompt.

    With *inform* (``--uncertainty-estimator-mode inform``) it includes the
    section that explains <certainty>; otherwise the prompt never mentions the
    tag, since the trajectory carries none.  The turn cap is enforced in code
    and never stated to the policy.
    """
    return (SYSTEM_TEMPLATE
            .replace("{certainty_section}", CERTAINTY_SECTION if inform else "")
            .replace("{injected_tags}", "<information> or <certainty>" if inform else "<information>"))


def render_user(question: str) -> str:
    return USER_TEMPLATE.replace("{question}", question).rstrip("\n")


def render_format_error(errors) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{errors}", "\n".join(f"- {e}" for e in errors))
