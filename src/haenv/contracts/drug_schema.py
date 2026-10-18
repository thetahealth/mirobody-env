"""The field checks one drug-effect entry must pass, in-repo or from a world plugin.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from typing import Any


class DrugEffectsError(ValueError):
    """Registry validation failure, or a drug effect applied to a forbidden signal."""


def check_entry(name: str, spec: Any) -> None:
    """Field checks for one drug entry (in-repo or plugin); raises `DrugEffectsError`."""
    if not isinstance(spec, dict):
        raise DrugEffectsError(f"{name}: entry must be a mapping")
    for f in ("total_hba1c_pp", "trial_weight_kg", "readout_weeks",
              "background", "source", "review"):
        if f not in spec:
            raise DrugEffectsError(f"{name} is missing `{f}:`")
    if spec["background"] not in ("monotherapy_vs_placebo", "add_on_to_metformin"):
        raise DrugEffectsError(
            f"{name}.background={spec['background']!r} is not an allowed value; "
            "state whether the trial is monotherapy vs placebo or add-on to metformin")
    for f in ("total_fpg_mmol", "sd_change_hba1c_pp"):
        if f not in spec:
            raise DrugEffectsError(f"{name} is missing `{f}:` (null is allowed, absence is not)")
