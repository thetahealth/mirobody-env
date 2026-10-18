"""Pack 2 rebuilt -- acute triage on wearable trends (world 1.4).

One single-shot item per case: a patient sends 10 days of wearable and home readings and a message
and asks whether to go to hospital. The gold is a set of acceptable dispositions given by the rule
reader (`rules.assess`) on the visible block and the printed rules; scored by ordinal expected
cost (under-triage 3 per step, over-triage 1 per step), code only.

Modules: rules (rule reader, P2_WORLD) · render (visible block, P2_WORLD) · gate (emission, P2_WORLD) ·
hooks (production wrappers, both segments) · plugin (judge group, P2_JUDGING) · score (P2_JUDGING) ·
audit (generation-time audit, tool side).

The previous pack 2 (`haenv/pack2/`, job `pack2-triage`, board 1.3.0) is untouched.
SYNTHETIC data, evaluation only, not medical advice.
"""
from haenv.world.history.history_r2 import enable as _enable_history_generation

_enable_history_generation()

from haenv.world.protocols.delivery import install as _install_protocol_delivery

_install_protocol_delivery()

from haenv.scoring.management.plan import install as _install_management_plan

_install_management_plan()
