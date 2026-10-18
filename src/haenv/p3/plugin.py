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

NAME = "meds_p3"
LATENT = "p3"
PROBE_ID = "p3.meds"
FRAMING = "P3_MEDS_PROMPT"

P3_MEDS_PROMPT = """请对这个药物复诊记录作出今天的处置建议。题头给出就诊类型、今天日期、患者原话和全部在用慢病药。
请结合门诊病程、检验、处方、取药、家庭监测和症状日记，自行识别本次需评估处置的一种在用药，
在 `target_drug` 中填写药名；即使决定维持，也须指出所评估的药。按本院规程给出 `med_decision`,五选一:
`uptitrate` 加量 · `downtitrate` 减量 · `maintain` 维持当前剂量 ·
`switch` 本药不再加量,换用或加用另一种药 ·
`check_adherence_or_adverse_effect` 先查依从性、副作用或其他原因,暂不调整剂量
(已能归因于本药剂量的不良反应请选减量)。
`new_dose` 填调整后的剂量(维持、换药、先查时填 null);`rationale_evidence` 列出你依据的记录。
你的建议由医生审核后执行:涉及改剂量、换药时 `action.clinician_review_required` 须为 true。

pre-T 数据(JSON):
{payload}

只输出一个 JSON 对象(无任何多余文字/markdown),schema:
{{"med_decision":"uptitrate|downtitrate|maintain|switch|check_adherence_or_adverse_effect",
"target_drug":"药名","new_dose":null,
"rationale_evidence":["依据的记录"],
"action":{{"selected_action_class":"A3","specific_action":"一句话的具体处置","clinician_review_required":true}}}}"""


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
                 "③ 慢病调药包(设计 §3):调药决定五类的金标,由世界层剂量线、取药覆盖、控制轴真值与症状记录经 G3 确定;"
                 "题面块在测量层之后由 hooks.observe_meds 写入。"),),
        emission=emission(),
        framing=(FRAMING, P3_MEDS_PROMPT, "③ 单回合无菜单的调药问法(设计 §3.1 作答 schema)。"),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": "③ 单回合无菜单(设计 §11 K2):决定所需数据全部常显,不走购买式工具轨",
                 "slices": "③ 不切片:一次复诊一个决定", "multi": "③ 单回合,不是多轮追踪"},
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
