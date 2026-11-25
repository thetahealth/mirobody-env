"""The `haenv.rl` judge group (entry point declared by `plugins/haenv_rl-0.1.0.dist-info`).

* gold block `redline_rl` (latent key `rl`, class `knob`): at generation it carries the plan; the OBSERVE hook
  (`hooks.observe`) replaces it by the realized truth and writes `prediction_context.redline_rl`.
* emission gate `rl` (`gate.py`: shared SA-1 + SA-8 with G1-G3).
* framing `RL_VIGILANCE_PROMPT` and probe `rl.vigilance`.
* judge `redline_rl` (`haenv/rl/judge.py`, LLM, segment `RL_JUDGING`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import dataclasses

from haenv.i18n.text import templates as _templates, text_of as _rt

NAME = "redline_rl"
LATENT = "rl"
PROBE_ID = "rl.vigilance"
FRAMING = "RL_VIGILANCE_PROMPT"

#: the probe's system text verbatim, then date, record, question and the reply envelope (`registry/i18n/en/templates/rl.yaml`)
RL_VIGILANCE_PROMPT = _templates("en")["runtime.rl.plugin.framing"]


def _gold(meta):
    from haenv.contracts.external_gold import SLOT
    p = ((meta or {}).get(SLOT) or {}).get(LATENT)
    return {"plan": dict(p), "rule_id": p.get("rule_id"), "family": p.get("family"),
            "variant": p.get("variant")} if isinstance(p, dict) else None


def probe_dict() -> dict:
    from haenv.world.generation.framings import framing_sha256
    return {"probe_id": PROBE_ID, "mode": "ddx", "difficulty_probe": "P0", "framing_ref": FRAMING,
            "framing_sha256": framing_sha256(FRAMING), "answer_space": "rl_reply", "hint_level": 0,
            "budget": {"max_rounds": 1}}


def _judge(out, vp, ctx=None):
    from haenv.rl.judge import judge
    return judge(out, vp, ctx)


def _when(vp) -> bool:
    from haenv.rl.judge import when
    return when(vp)


def _owns(raw) -> bool:
    from haenv.rl.gate import plan_of
    return plan_of(raw) is not None


def pack():
    from haenv.app.packs import Pack, PayloadView
    from haenv.rl.gate import emission
    from haenv.rl.judge import KINDS
    return Pack(
        name="rl", owns=_owns, source="haenv.rl (haenv/rl/plugin.py)",
        blocks=((NAME, _gold, lambda meta: None, (LATENT,), {LATENT: "knob"}, _rt('runtime.rl.plugin.block')),),
        emission=emission(),
        framing=(FRAMING, RL_VIGILANCE_PROMPT, _rt('runtime.rl.plugin.framing_why')),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": _rt('runtime.rl.plugin.why_not'), "slices": _rt('runtime.rl.plugin.why_not'),
                 "multi": _rt('runtime.rl.plugin.why_not')},
        prompt_gold_meta=("outcome", "driver"), judging_segment="RL_JUDGING", base_labels=(),
        answers_action=False,
        view=PayloadView(streams=()))


def register() -> list:
    from haenv.app.packs import register_pack
    from haenv.rl.hooks import install
    install()
    return [dataclasses.replace(j, category="llm") for j in register_pack(pack())]


def judges():
    """Entry point of the `haenv.rl` group."""
    return register()
