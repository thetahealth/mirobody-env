"""The `haenv.p4` judge group (entry point declared by `plugins/haenv_p4-0.1.0.dist-info`).

`judges()` registers, through the shared skeleton (`haenv/pack_skeleton.py`):

* gold block `followup_p4` (latent key `p4`, class `knob`): at generation it carries the plan;
  after the measurement layer (`hooks.observe_followup`) it is replaced by the realized gold
  with its per-point records. Its question side is written by the same hook into
  `prediction_context.followup_p4` once the readings exist, so `probe_fn` returns `None`.
* emission gate `p4`: the shared gate (`haenv/shared_audit.py`, SA-1 + SA-8) on the declaration
  of `gate.py`.
* question template `P4_FOLLOWUP_PROMPT` and probe `p4.followup`; the probe carries the pack-4
  world fingerprint and the lab-world table fingerprint, so both move `world_sha` of a pack-4
  batch (`external_gold.manifest_sha` hashes the probe).
* judge `followup_p4` (deterministic, single-shot geometry): the row atoms of `score.row_atoms`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

NAME = "followup_p4"
LATENT = "p4"
PROBE_ID = "p4.followup"
FRAMING = "P4_FOLLOWUP_PROMPT"

P4_FOLLOWUP_PROMPT = """你是"随访复查解读"求解器。下面是一位患者在 index-time T 之前(含 T 日)的全部可见数据。
`prediction_context.followup_p4` 是今天(T 日)到来的一次随访复查:被随访指标的全部历史读数与本次读数,
每次带采血时刻、空腹登记、标本处理(离心前放置时长、溶血指数)、检测机构与方法;
`diet_record` 是上次(`previous_draw_day`)与本次两次采血前的饮食记录;`medication_record` 是用药记录。
体重等其余数据见 `longitudinal_data`。

请判断本次读数相对上次读数的变化:
① 变化的来源 `change_source`,四选一:
   `true_change` = 患者体内该指标的真实水平变了;
   `analytic_biological_noise` = 测量误差与个体内生物学波动,真实水平没变;
   `preanalytical` = 采血或标本环节的因素(进食、标本放置、采血前饮食等)使这次或上次的读数偏离真实水平;
   `method_difference` = 两次检测的方法或器械不同,使读数偏离。
② 造成变化的因素 `factor`(从 schema 列出的值中选一个;来源为测量误差与生物学波动时填 null)。
③ 真实水平的百分比变化 `true_change_pct`:本次真实水平相对上次真实水平变化了百分之几(下降为负)。
④ 下一步 `next_step`:`act_on_change` 按这次变化处理 · `repeat_standardized` 在标准条件下复查确认后再判断 ·
   `no_action` 不需要因这次变化做任何处理。

医疗边界(硬约束):不自主开药/停药/改剂量。

pre-T 数据(JSON):
{payload}

只输出一个 JSON 对象(无任何多余文字/markdown),schema:
{{"change_source":"true_change|analytic_biological_noise|preanalytical|method_difference",
"factor":"glp1_dose_change|weight_change|acei_start_or_stop|postprandial_labelled_fasting|delayed_processing|meat_meal|poc_meter|jaffe_enzymatic_switch|null",
"true_change_pct":-12.0,
"next_step":"act_on_change|repeat_standardized|no_action",
"reason":"一句话依据"}}"""


def _gold(meta):
    from ..external_gold import SLOT
    p = ((meta or {}).get(SLOT) or {}).get(LATENT)
    return {"plan": dict(p)} if isinstance(p, dict) else None


def probe_dict() -> dict:
    from ..framings import framing_sha256
    from ..labworld import tables_sha16
    from .hooks import p4_world_fingerprint
    return {"probe_id": PROBE_ID, "mode": "ddx", "difficulty_probe": "P0", "framing_ref": FRAMING,
            "framing_sha256": framing_sha256(FRAMING), "answer_space": "p4_followup", "hint_level": 0,
            "budget": {"max_rounds": 1}, "p4_world_sha16": p4_world_fingerprint(),
            "labworld_tables_sha16": tables_sha16()}


def _judge(out, vp, ctx):
    from .hooks import p4_judging_fingerprint
    from .score import row_atoms
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return row_atoms(getattr(out, "_raw", None) or {}, g, p4_judging_fingerprint())


def _when(vp) -> bool:
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return isinstance(g, dict) and "class" in g


#: Gold kinds the judge mounts on: all of them (`_when` decides by the ④ gold block).
KINDS = ("*",)


def pack():
    from ..pack_skeleton import Pack
    from .gate import emission
    return Pack(
        name="p4", owns=_when, source="haenv.p4 (haenv/p4/plugin.py)",
        blocks=((NAME, _gold, lambda meta: None, (LATENT,), {LATENT: "knob"},
                 "④ 随访解读包(设计 §4):来源四类与真变化幅度的金标,由世界层真值与逐点扰动记录经 G4 确定;"
                 "题面块在测量层之后由 hooks.observe_followup 写入。"),),
        emission=emission(),
        framing=(FRAMING, P4_FOLLOWUP_PROMPT, "④ 单回合无菜单的随访解读问法(设计 §4.1 作答 schema)。"),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": "④ 单回合无菜单(设计 §11 K2):决定所需数据全部常显,不走购买式工具轨",
                 "slices": "④ 不切片:一次复查一个问题", "multi": "④ 单回合,不是多轮追踪"},
        prompt_gold_meta=("outcome", "driver"), judging_segment="P4_JUDGING")


def register() -> list:
    """Idempotent registration through the shared skeleton (also used by tools that skip the job
    loader); returns the judge the first time."""
    from ..pack_skeleton import register_pack
    from .hooks import install
    install()
    return register_pack(pack())


def _ensure_judge():
    """The judge as `register_pack` builds it, for a caller that changed `KINDS`."""
    from ..judges import Judge
    return Judge(name=NAME, fn=_judge, kinds=KINDS, category="deterministic", when=_when)


def judges():
    """Entry point of the `haenv.p4` group."""
    return register()
