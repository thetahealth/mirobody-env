"""Pack 4 -- follow-up interpretation.

One single-shot item per case: a follow-up lab result arrives today; is the change since the
previous draw a true change, analytic/biological noise, a pre-analytic artefact or a method
difference, and how large is the true change? Gold is the world truth (`haenv/labworld`) and the
per-point perturbation records, scored by code (no LLM judge).

Modules:
  plan      -- case-keyed allocation (class within H, analyte, sign, band position, decoys)
  realize   -- item realization on the built world (truth, noise pair, perturbations, records)
  render    -- production renderer of the visible follow-up block
  gold      -- G4 zones, re-derivation from records
  gate      -- emission gate and leak scan
  features  -- surface and factor features read off the solver payload (audits)
  score     -- answer parsing, row atoms, pack composite, chance correction
  stubs     -- capability-known offline stubs
  hooks     -- the two wrapped production attributes and the two fingerprints
  plugin    -- the `haenv.p4` judge group

SYNTHETIC data, evaluation only, not medical advice.
"""
