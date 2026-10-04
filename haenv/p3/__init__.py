"""Pack 3 -- chronic medication adjustment.

One single-shot item per case: a treated chronic condition at a follow-up visit; what to do
with the drug next (uptitrate / downtitrate / maintain / switch / check adherence or adverse
effect). Gold is G3 on the world records (dose line, refill coverage, control-axis truth,
symptom causes), scored by code (no LLM judge). World pieces: `haenv/labworld/meds.py`.

Modules: plan, realize, render, gold, gate, features, score, stubs, hooks, plugin.

SYNTHETIC data, evaluation only, not medical advice.
"""
