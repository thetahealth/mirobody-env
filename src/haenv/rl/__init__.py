"""Pack rl -- red-line vigilance pairs.

The user asks an everyday question; one registered red-line fact sits in the longitudinal record. Each item is a
pair on the same world: the redline arm (the trigger holds) and the control arm (a registered resolution).

Modules:
  rules   -- the rule registry (`registry/redline_rules.yaml`), `fires` / `sweep`
  plan    -- case-keyed allocation (rule, question, imminence, dates)
  realize -- the record block, facts, patch span and gold of one arm on the built world
  render  -- the readable record and its parser (G3 reads only the text)
  gate    -- emission gate (G1 single red line, G2 question independence, G3 answerable)
  audit   -- batch audit (SA gates over both arms, G4 one-patch pair, G5 paired imminence)
  hooks   -- the OBSERVE and PROMPT hooks
  plugin  -- the `haenv.rl` judge group
  judge, score -- the LLM judge and pair scoring (segment RL_JUDGING)

SYNTHETIC data, evaluation only, not medical advice.
"""
from haenv.world.history.history_r2 import enable as _enable_history_generation

_enable_history_generation()
