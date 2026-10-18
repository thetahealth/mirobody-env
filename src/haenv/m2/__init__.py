"""M2 single track.

Four frames over the v1.0.1 tool-track cases (F0 weight-regain bridge, F1 acute triage, F2 chronic
medication adjustment, F3 follow-up interpretation) and two answer rounds: round 1 is the gated
workup with on-demand purchases, round 2 pushes a fixed block of five readings and asks for a
revision.

Modules:
  core     -- threads and key signals, push-block plan and rendering, frame gold, revision gold
  plugin   -- the `haenv.m2` judge group: registers the gold blocks and the frame judge
  round2   -- the second answer round, called from `evaluate._row_gated`
  score    -- per-row round-2 atoms and the M2 composite (chance correction on raw BA)
  stubs    -- capability-known offline stubs (oracle-degradation ladder, split, random, constant)

Every text the model sees that a generation LLM would normally word is written from templates
here; the solver-facing blocks do not label their source.

SYNTHETIC data, evaluation only, not medical advice.
"""

from haenv.world.history.history_r2 import enable as _enable_history_generation

_enable_history_generation()

from dataclasses import replace as _replace_history_policy
from haenv.m2 import history as _history
from haenv.m2.history_r2 import finalize as _finalize_history

_history.POLICY = _replace_history_policy(_history.POLICY, finalize=_finalize_history)

from haenv.m2.workup_generation import install as _install_conditional_workup

_install_conditional_workup()

from haenv.scoring.management.plan import install as _install_management_plan

_install_management_plan()

from haenv.m2.confirmation import install_results as _install_confirmation_results

_install_confirmation_results()
