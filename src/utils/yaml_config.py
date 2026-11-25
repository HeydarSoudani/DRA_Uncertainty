"""YAML configs and their command-line overrides.

Two shapes of config file:

* sectioned (one dataclass per section; the training scripts:
  train_belief_model, the dataset build): :func:`load_sectioned`;
* flat, merged onto the argparse namespace with ad-hoc ``--key value``
  overrides (the inference CLI, ``pipeline.cli``; the index builder,
  ``indexing_corpus_dataset.indexing.index_config``):
  :func:`parse_cli_overrides` + :func:`apply_config_to_args`.
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar

import yaml

C = TypeVar("C")


# ===========================================================================
# Sectioned configs
# ===========================================================================

_TRUE = ("true", "1", "yes", "t")
_FALSE = ("false", "0", "no", "f")


def _to_bool(value: str) -> bool:
    """A boolean written as text (ValueError when it is none)."""
    if value.lower() in _TRUE:
        return True
    if value.lower() in _FALSE:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _to_type_of(value: str, reference: Any) -> Any:
    """*value* (text) as the type of *reference* (bool, int, float; else text)."""
    if isinstance(reference, bool):
        return _to_bool(value)
    if isinstance(reference, int):  # bool already handled above
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _coerce(value: Any, current: Any, key: str) -> Any:
    """A YAML string (PyYAML reads e.g. ``5e-6`` as one) as the type of the key's current value."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("null", "none"):
        return None
    if isinstance(current, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    try:
        return _to_type_of(value, current)
    except ValueError as e:
        raise ValueError(f"{key}: {e}") from None


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
        setattr(obj, key, _coerce(value, getattr(obj, key), dotted))
    return cfg


# ===========================================================================
# Flat configs with CLI overrides
# ===========================================================================

def parse_bool(v):
    """Boolean coercion for argparse ``type=`` (``--quiet false``): real
    booleans and their text forms (true/false, 1/0, yes/no, t/f)."""
    if isinstance(v, bool):
        return v
    try:
        return _to_bool(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}") from None


def parse_cli_overrides(extras) -> dict:
    """Turn argparse ``parse_known_args`` leftovers into an overrides dict.

    Handles ``--key value``, ``--key=value``, space-separated values (kept
    for a list-valued key only; :func:`apply_config_to_args` rejects several
    values for any other key, so ``--gpu-ids 4,5`` and not ``--gpu-ids 4 5``),
    and bare boolean flags (``--quiet``).
    Dashes in flag names are normalised to underscores to match attr names.
    """
    overrides: dict = {}
    i = 0
    while i < len(extras):
        tok = extras[i]
        if not tok.startswith("--"):
            # A stray token (e.g. the single-dash typo "-quiet") would
            # otherwise be dropped and the run would go on with the default.
            raise SystemExit(f"error: unrecognized argument: {tok} (flags take two dashes, e.g. --quiet)")
        if "=" in tok:
            key, val = tok[2:].split("=", 1)
            overrides[key.replace("-", "_")] = [val]
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        vals = []
        i += 1
        while i < len(extras) and not extras[i].startswith("--"):
            vals.append(extras[i])
            i += 1
        overrides[key] = vals  # empty list = bare flag
    return overrides


def validate_choices(values: dict, choices: Dict[str, list]) -> None:
    """Validate the enum knobs of *values* against their allowed values (*choices*)."""
    for key, allowed in choices.items():
        if key in values and values[key] not in allowed:
            raise ValueError(
                f"Invalid value for '{key}': {values[key]!r}. Choose from: {allowed}"
            )


def apply_config_to_args(args, config: dict, overrides: dict, *, types: Optional[Dict[str, Callable]] = None,
                         choices: Optional[Dict[str, list]] = None) -> None:
    """Merge file-backed config and CLI overrides onto the ``args`` namespace.

    Sets every key of ``config`` on ``args``, then applies CLI ``overrides``
    (coerced to each key's config type; *types* names the type of a key whose
    default is ``None``) so an explicit flag wins over the file for a single
    run, then checks the enum knobs against *choices* (:func:`validate_choices`).
    An unknown flag is an error: ignoring it would silently run with the
    default instead.
    """
    types = types or {}
    unknown = sorted(f"--{key.replace('_', '-')}" for key in overrides if key not in config)
    if unknown:
        raise SystemExit(f"error: unrecognized arguments: {' '.join(unknown)}")

    for key, value in config.items():
        setattr(args, key, value)

    for key, raw_vals in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        ref = config[key]
        try:
            if isinstance(ref, list):
                flat = [tok for v in raw_vals for tok in v.split()]
                elem_ref = ref[0] if ref else ""
                coerced = [_to_type_of(tok, elem_ref) for tok in flat]
            elif not raw_vals:
                # A bare flag (``--quiet``) only means "on" for a boolean key;
                # elsewhere it is a missing value, not True.
                if not isinstance(ref, bool):
                    raise SystemExit(f"error: {flag} needs a value")
                coerced = True
            elif len(raw_vals) > 1:
                # One value only: the extra ones would be dropped silently.
                raise SystemExit(f"error: {flag} takes one value, got {' '.join(raw_vals)}")
            elif ref is None and key in types:
                coerced = types[key](raw_vals[0])
            else:
                coerced = _to_type_of(raw_vals[0], ref)
        except ValueError as e:
            raise SystemExit(f"error: {flag}: {e}") from None
        setattr(args, key, coerced)

    validate_choices(vars(args), choices or {})
