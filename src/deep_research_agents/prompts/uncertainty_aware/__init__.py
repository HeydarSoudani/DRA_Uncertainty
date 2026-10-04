"""Prompt templates for the uncertainty-aware agent (plain-text files next to this module).

Two task variants, picked by the dataset's ``task`` (layout.DATASET_SPECS):
``qa`` answers a question (``system.txt``, ``certainty_section.txt``,
``user.txt``); ``report`` writes a report for a request (the ``*_report.txt``
files).  The turn protocol and the format-error message are shared.

Placeholders are filled with str.replace, not str.format, so a literal brace
added to a template later cannot break rendering.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

TASKS = ("qa", "report")


def _read(name: str) -> str:
    return (_DIR / name).read_text()


def _suffix(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return "" if task == "qa" else f"_{task}"


SYSTEM_TEMPLATES = {task: _read(f"system{_suffix(task)}.txt") for task in TASKS}
CERTAINTY_SECTIONS = {task: _read(f"certainty_section{_suffix(task)}.txt") for task in TASKS}
USER_TEMPLATES = {task: _read(f"user{_suffix(task)}.txt") for task in TASKS}
FORMAT_ERROR_TEMPLATE: str = _read("format_error.txt")


def render_system(inform: bool, task: str = "qa", report_chars: int | None = None) -> str:
    """The policy's system prompt.

    With *inform* (``--uncertainty-estimator-mode inform``) it includes the
    section that explains <certainty>; otherwise the prompt never mentions the
    tag, since the trajectory carries none.  The turn cap is enforced in code
    and never stated to the policy.  *report_chars* is the target report
    length of the ``report`` task.
    """
    _suffix(task)
    prompt = (SYSTEM_TEMPLATES[task]
              .replace("{certainty_section}", CERTAINTY_SECTIONS[task] if inform else "")
              .replace("{injected_tags}", "<information> or <certainty>" if inform else "<information>"))
    if task == "report":
        if report_chars is None:
            raise ValueError("the report task needs report_chars")
        prompt = prompt.replace("{report_chars}", str(report_chars))
    return prompt


def render_user(question: str, task: str = "qa") -> str:
    _suffix(task)
    return USER_TEMPLATES[task].replace("{question}", question).rstrip("\n")


def render_format_error(errors) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{errors}", "\n".join(f"- {e}" for e in errors))
