"""The `haenv.p4` judge group (entry point declared by `plugins/haenv_p4-0.1.0.dist-info`).

`judges()` registers, through the shared skeleton (`haenv/app/packs.py`):

* gold block `followup_p4` (latent key `p4`, class `knob`): at generation it carries the plan;
  after the measurement layer (`hooks.observe_followup`) it is replaced by the realized gold
  with its per-point records. Its question side is written by the same hook into
  `prediction_context.followup_p4` once the readings exist, so `probe_fn` returns `None`.
* emission gate `p4`: the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8) on the declaration
  of `gate.py`.
* question template `P4_FOLLOWUP_PROMPT` and probe `p4.followup`; the probe carries the pack-4
  world fingerprint and the lab-world table fingerprint, so both move `world_sha` of a pack-4
  batch (`external_gold.manifest_sha` hashes the probe).
* judge `followup_p4` (deterministic, single-shot geometry): the row atoms of `score.row_atoms`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import L as _L, text_of as _rt

NAME = "followup_p4"
LATENT = "p4"
PROBE_ID = "p4.followup"
FRAMING = "P4_FOLLOWUP_PROMPT"

P4_FOLLOWUP_PROMPT = _L('runtime.p4.plugin.module.1')


def _gold(meta):
    from haenv.contracts.external_gold import SLOT
    p = ((meta or {}).get(SLOT) or {}).get(LATENT)
    return {"plan": dict(p)} if isinstance(p, dict) else None


def probe_dict() -> dict:
    from haenv.world.generation.framings import framing_sha256
    from haenv.world.labworld import tables_sha16
    from haenv.p4.hooks import p4_world_fingerprint
    return {"probe_id": PROBE_ID, "mode": "ddx", "difficulty_probe": "P0", "framing_ref": FRAMING,
            "framing_sha256": framing_sha256(FRAMING), "answer_space": "p4_followup", "hint_level": 0,
            "budget": {"max_rounds": 1}, "p4_world_sha16": p4_world_fingerprint(),
            "labworld_tables_sha16": tables_sha16()}


def _judge(out, vp, ctx):
    from haenv.p4.hooks import p4_judging_fingerprint
    from haenv.p4.score import row_atoms
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return row_atoms(getattr(out, "_raw", None) or {}, g, p4_judging_fingerprint())


def _when(vp) -> bool:
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return isinstance(g, dict) and "class" in g


#: Gold kinds the judge mounts on: all of them (`_when` decides by the ④ gold block).
KINDS = ("*",)


def pack():
    from haenv.app.packs import Pack
    from haenv.app.packs import PayloadView
    from haenv.p4.gate import emission
    from haenv.p4.history import POLICY
    return Pack(
        name="p4", history_dependency=POLICY, owns=_when, source="haenv.p4 (haenv/p4/plugin.py)",
        blocks=((NAME, _gold, lambda meta: None, (LATENT,), {LATENT: "knob"},
                 _rt('runtime.p4.plugin.pack.1')),),
        emission=emission(),
        framing=(FRAMING, P4_FOLLOWUP_PROMPT, _rt('runtime.p4.plugin.pack.2')),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": _rt('runtime.p4.plugin.pack.3'),
                 "slices": _rt('runtime.p4.plugin.pack.4'), "multi": _rt('runtime.p4.plugin.pack.5')},
        prompt_gold_meta=("outcome", "driver"), judging_segment="P4_JUDGING",
        base_labels=(),                         # the score reads its own block, no base-world label
        answers_action=False,                   # the answer is source class / factor / % / next step only
        # the reading is interpreted off the follow-up block (`prediction_context.followup_p4`); of the streams,
        # weight (muscle mass, `features.factor_flags`), blood pressure, glycaemia and exercise load
        view=PayloadView(streams=("weight", "systolic_bp", "diastolic_bp", "HbA1c", "fasting_glucose",
                                  "steps", "exercise_mets", "activity_index")))


def register() -> list:
    """Idempotent registration through the shared skeleton (also used by tools that skip the job
    loader); returns the judge the first time."""
    from haenv.app.packs import register_pack
    from haenv.p4.hooks import install
    install()
    return register_pack(pack())


def _ensure_judge():
    """The judge as `register_pack` builds it, for a caller that changed `KINDS`."""
    from haenv.judges import Judge
    return Judge(name=NAME, fn=_judge, kinds=KINDS, category="deterministic", when=_when)


def judges():
    """Entry point of the `haenv.p4` group."""
    return register()
