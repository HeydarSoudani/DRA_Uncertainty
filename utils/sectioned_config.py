"""Loader of the sectioned YAML configs (one dataclass per section) of the
training scripts (belief_model_training, build_training_pool)."""

import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TypeVar

import yaml

C = TypeVar("C")


def coerce(value: Any, current: Any, key: str) -> Any:
    """A YAML string (PyYAML reads e.g. ``5e-6`` as one) as the type of the key's current value."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("null", "none"):
        return None
    if isinstance(current, bool):
        if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"{key}: expected a boolean, got {value!r}")
        return value.lower() in ("true", "1", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_sectioned(cfg: C, sections: Iterable[str], path: Optional[str],
                   overrides: Optional[Dict[str, Any]] = None) -> C:
    """*cfg* (the defaults) with the YAML of *path* (None = none) and then
    *overrides* (``{"section.key": value}``) applied; an unknown section or
    key is an error."""
    sections = list(sections)
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    items = [(f"{section}.{key}", value)
             for section, values in (data or {}).items() for key, value in (values or {}).items()]
    items += list((overrides or {}).items())
    for dotted, value in items:
        section, _, key = dotted.partition(".")
        if section not in sections:
            raise ValueError(f"unknown config section {section!r}; expected one of {sections}")
        obj = getattr(cfg, section)
        if key not in {f.name for f in dataclasses.fields(obj)}:
            raise ValueError(f"unknown config key {dotted!r}")
        setattr(obj, key, coerce(value, getattr(obj, key), dotted))
    return cfg
