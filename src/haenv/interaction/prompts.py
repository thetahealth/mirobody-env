"""prompts.py -- the solver prompt template `PROMPT`.

Kept in a data-only module so `build.py` can read it without importing `evaluate.py`.
The differential-diagnosis templates live in `framings.py`.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import L as _L

PROMPT = _L('runtime.interaction.prompts.module.1')
