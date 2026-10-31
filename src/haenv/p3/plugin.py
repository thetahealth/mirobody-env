"""The `haenv.p3` judge group (entry point declared by `plugins/haenv_p3-0.1.0.dist-info`).

`judges()` registers, through the shared skeleton (`haenv/app/packs.py`):

* gold block `meds_p3` (latent key `p3`, class `knob`): at generation it carries the plan; after
  the measurement layer (`hooks.observe_meds`) it is replaced by the realized gold with its
  records. The question side is written by the same hook into `prediction_context.meds_p3`, so
  `probe_fn` returns `None`;
* emission gate `p3`: the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8) on the declaration of
  `gate.py`, with the pack callback P-9;
* question template `P3_MEDS_PROMPT` and probe `p3.meds`; the probe carries the pack-3 world
  fingerprint, so it moves `world_sha` of a pack-3 batch;
* judge `meds_p3` (deterministic, single-shot geometry): the row atoms of `score.row_atoms`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import L as _L, text_of as _rt

NAME = "meds_p3"
LATENT = "p3"
PROBE_ID = "p3.meds"
FRAMING = "P3_MEDS_PROMPT"

P3_MEDS_PROMPT = _L('runtime.p3.plugin.module.1')


def _gold(meta):
    from haenv.contracts.external_gold import SLOT
    p = ((meta or {}).get(SLOT) or {}).get(LATENT)
    return {"plan": dict(p)} if isinstance(p, dict) else None


def probe_dict() -> dict:
    from haenv.world.generation.framings import framing_sha256
    from haenv.p3.hooks import p3_world_fingerprint
    return {"probe_id": PROBE_ID, "mode": "ddx", "difficulty_probe": "P0", "framing_ref": FRAMING,
            "framing_sha256": framing_sha256(FRAMING), "answer_space": "p3_meds", "hint_level": 0,
            "budget": {"max_rounds": 1}, "p3_world_sha16": p3_world_fingerprint()}


def _judge(out, vp, ctx):
    from haenv.p3.hooks import p3_judging_fingerprint
    from haenv.p3.score import row_atoms
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return row_atoms(getattr(out, "_raw", None) or {}, g, p3_judging_fingerprint())


def _when(vp) -> bool:
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return isinstance(g, dict) and "class" in g


#: Gold kinds the judge mounts on: all of them (`_when` decides by the pack-3 gold block; a host
#: case of every kind, `ddx:insufficient` included, can carry a pack-3 item).
KINDS = ("*",)


def pack():
    from haenv.app.packs import Pack
    from haenv.app.packs import PayloadView
    from haenv.p3.gate import emission
    from haenv.p3.history import POLICY
    return Pack(
        name="p3", history_dependency=POLICY, owns=_when, source="haenv.p3 (haenv/p3/plugin.py)",
        blocks=((NAME, _gold, lambda meta: None, (LATENT,), {LATENT: "knob"},
                 _rt('runtime.p3.plugin.pack.1')),),
        emission=emission(),
        framing=(FRAMING, P3_MEDS_PROMPT, _rt('runtime.p3.plugin.pack.2')),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": _rt('runtime.p3.plugin.pack.3'),
                 "slices": _rt('runtime.p3.plugin.pack.4'), "multi": _rt('runtime.p3.plugin.pack.5')},
        prompt_gold_meta=("outcome", "driver"), judging_segment="P3_JUDGING",
        base_labels=(),                         # the score reads its own block, no base-world label
        # the decision reads the visit block (`prediction_context.meds_p3`); of the streams, the clinical
        # control axes (a drug's own bearing streams are already kept out by `plan`)
        view=PayloadView(streams=("weight", "systolic_bp", "diastolic_bp", "resting_hr", "HbA1c", "fasting_glucose",
                                  "CGM_TIR", "LDL", "triglycerides", "ALT", "AST", "FIB4", "TSH")))


def register() -> list:
    """Idempotent registration through the shared skeleton; returns the judge the first time."""
    from haenv.app.packs import register_pack
    from haenv.p3.hooks import install
    install()
    judges = register_pack(pack())
    from haenv.p3.emr import install as install_emr
    install_emr()
    return judges


def judges():
    """Entry point of the `haenv.p3` group."""
    return register()
