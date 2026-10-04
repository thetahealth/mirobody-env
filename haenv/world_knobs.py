"""Switches of the generated world that a job turns on (`job.findings`, `job.physio`).

One-element lists, so every module that imported them sees the value a job set.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations


# ---------------------------------------------------------------- Findings-layer hook
FINDINGS_ENABLED: list = [False]      # set by run/build from job.findings; off by default


#: Physiology-layer switch (`job.yaml: physio`), off by default. When on,
#: rendered streams go through `haenv.physio.apply_physio`.
PHYSIO_ENABLED: list = [False]


STEP, END = 7, 365
