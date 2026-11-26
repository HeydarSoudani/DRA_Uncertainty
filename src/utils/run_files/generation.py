"""The per-query generation file ``generation/{query_id}.md``.

The generation (the report or short answer, with its References block) is
saved as is; the evaluation scores it (``evaluation.generation``).
"""

from pathlib import Path
from typing import Any, Dict, Union

GENERATION_DIR = "generation"


def save_generation(run_dir: Union[str, Path], query_id: str, result: Dict[str, Any]) -> None:
    """Write the generation text to ``{run_dir}/generation/{query_id}.md``."""
    out_dir = Path(run_dir) / GENERATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{query_id}.md", "w") as f:
        f.write(result.get("generation", ""))


def read_generation(run_dir: Union[str, Path], query_id: str) -> str:
    """The saved generation of a query, "" when it has none."""
    try:
        return (Path(run_dir) / GENERATION_DIR / f"{query_id}.md").read_text(encoding="utf-8")
    except OSError:
        return ""
