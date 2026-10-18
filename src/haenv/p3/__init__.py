"""Pack 3 -- chronic medication adjustment.

One single-shot item per case: a treated chronic condition at a follow-up visit; what to do
with the drug next (uptitrate / downtitrate / maintain / switch / check adherence or adverse
effect). Gold is G3 on the world records (dose line, refill coverage, control-axis truth,
symptom causes), scored by code (no LLM judge). World pieces: `haenv/world/labworld/meds.py`.

Modules: plan, realize, render, gold, gate, features, score, stubs, hooks, plugin.

SYNTHETIC data, evaluation only, not medical advice.
"""
from haenv.world.history.history_r2 import enable as _enable_history_generation

_enable_history_generation()

from haenv.world.protocols.delivery import install as _install_protocol_delivery

_install_protocol_delivery()

from haenv.scoring.management.plan import install as _install_management_plan

_install_management_plan()
