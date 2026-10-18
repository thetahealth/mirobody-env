"""Pack 2 · acute triage.

A single request per case: today's acute visit (complaint, vitals now, point-of-care labs, 72 h
logs) on top of a ddx-workup world, asking where the patient should go now. The disposition gold
is G2 on the shown values, scored by code; the hidden disease H is crossed with the disposition
2 x 4 by construction.

Modules:
  world    -- snapshot, event pool, G2, gold/probe blocks, per-case emission gate (PACK2_WORLD)
  framing  -- question template and probe (PACK2_WORLD)
  hooks    -- build/wq/evaluate wrappers installed when the plugin loads (PACK2_JUDGING)
  plugin   -- the `haenv.pack2` judge group (PACK2_JUDGING)
  score    -- atoms, chance correction, composite (PACK2_JUDGING)
  audit    -- batch-level generation-time audits and their mutation variants (tool side)

Load: `PYTHONPATH=<repo>:<repo>/plugins`, job `inputs/pack2-triage.job.yaml` (`plugins.judges:
[haenv.pack2]`). SYNTHETIC data, evaluation only, not medical advice.
"""
