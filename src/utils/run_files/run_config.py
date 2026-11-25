"""The run's ``run_config.json``, as the stages read it.

``run_config.json`` holds every setting that changes what the agent does
(written by the generate stage, ``pipeline.run_config.build_run_config``),
the belief stage's settings per belief model (``belief_models``) and the
evaluation's judge model; the per-query files carry no run settings.
"""

import json as _json
from pathlib import Path
from typing import Any, Dict, Optional, Union

RUN_CONFIG_FILE = "run_config.json"
#: The belief stage's section: ``{model: settings}``, one entry per belief
#: model (its short ``--belief-model`` name, the folder ``belief/{model}/``).
BELIEF_CONFIG_SECTION = "belief_models"


def read_run_config(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """The run's ``run_config.json`` (``{}`` when missing)."""
    path = Path(run_dir) / RUN_CONFIG_FILE
    return _json.loads(path.read_text()) if path.exists() else {}


def belief_config_of(config: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """The saved settings of the belief model *model* in a
    ``run_config.json`` *config* (None when it has none)."""
    return (config.get(BELIEF_CONFIG_SECTION) or {}).get(model)


def belief_model_of(section: Optional[Dict[str, Any]]) -> Optional[str]:
    """The belief model LLM of one saved belief model entry."""
    return (section or {}).get("belief_model")
