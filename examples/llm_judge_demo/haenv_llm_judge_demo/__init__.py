"""LLM judge plugin example: a thin shell that re-exports `haenv.judges.llm`.

It shows how a judge from outside the repository is wired in through an entry point
(`[project.entry-points."haenv.judges.llm_demo"]` -> `load_judge_plugins()` ->
`register_judge`). To mount the built-in LLM judge directly, use
`haenv.judges.register_llm_judges()` instead.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

# Re-exported, not copied: there is one implementation, in `haenv.judges.llm`.
from haenv.judges.llm import (  # noqa: F401
    DEFAULT_JUDGE_MODEL, ENV_SWITCH, KEYS, NAME, PROMPT, STATUS, VERDICTS,
    build_prompt, claims_of, gold_name, judge_rival_discriminator, judges,
    parse_verdict, rivals_of, set_dispatch)
