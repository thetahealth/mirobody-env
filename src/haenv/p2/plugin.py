"""The `haenv.p2` judge group: pack 2 rebuilt (entry point `plugins/haenv_p2-0.1.0.dist-info`).

Registers through the shared skeleton (`haenv/app/packs.py`):
* gold block `triage_p2` (latent key `p2`, class `knob`): the plan at generation; after the
  measurement layer (`hooks.observe_triage`) the rule reader's verdict on the shown block;
* emission gate `p2` (`gate.py`);
* question template `P2_TRIAGE_PROMPT` (prints the rules of `registry/pack2_triage_scenarios.yaml`
  once for every item) and probe `p2.triage`, carrying the P2_WORLD fingerprint and the table hash;
* judge `triage_p2` (deterministic, single shot): the atoms of `score.atoms`.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

NAME = "triage_p2"
LATENT = "p2"
PROBE_ID = "p2.triage"
FRAMING = "P2_TRIAGE_PROMPT"

_HEAD = """你是"健康助手分诊"求解器。一位病人通过线上健康助手发来最近的可穿戴与家庭测量数据和一段话,问要不要去医院。
`prediction_context.triage_p2` 是今天(`today_day`)这位病人的全部分诊信息:
`message` 病人的原话;`intake_checklist` 已完成的问诊筛查,`urgent_symptoms` 只列阳性症状,
`urgent_screen_complete=true` 表示其余急性症状均已询问并否认,空列表表示均无;其余项目为是/否,
`mild_new_symptom` 指今天或前 3 天新出现的轻症;
`background` 已知慢病与用药;
`home_check_today` 今天的家庭测量(体温计、指夹血氧仪、血压计;呼吸频率由家属数一分钟;未吸氧);
`wearable_daily` 最近 10 天手表日数据(静息心率、HRV、夜间最低血氧、皮温偏差、步数、睡眠、心律不齐提示次数、运动最高心率、佩戴备注);
`home_bp_log` 最近 7 天家庭血压(每天两次,今天最后一次即今天的测量);`morning_weight` 最近 10 天晨重;
`cgm_nightly` 糖尿病人最近 7 晚 CGM 夜间最低值(`night_of_day` = 该日之后的那一晚)。
其余字段是病人的既往记录,作背景。

本服务的分诊规则(按顺序,第一条符合的规则决定;"都可"表示两个去向都算对):
"""
_TAIL = """
去向四选一:`ed_now` 立即急诊 · `within_24h` 24 小时内就医 · `routine_followup` 常规复诊 · `watchful_waiting` 居家观察。

医疗边界(硬约束):不自主开药/停药/改剂量。

病人数据(JSON):
{payload}

只输出一个 JSON 对象(无任何多余文字/markdown),schema:
{{"disposition":"ed_now|within_24h|routine_followup|watchful_waiting",
"rule":"R0..R14 中你用的那条规则编号",
"evidence":["你依据的读数,如 home_check_today.spo2_pct=92"],
"reason":"一句话依据"}}"""


def prompt() -> str:
    from haenv.p2.rules import policy_text
    return _HEAD + policy_text().replace("{", "{{").replace("}", "}}") + "\n" + _TAIL


def _gold(meta):
    from haenv.contracts.external_gold import SLOT
    p = ((meta or {}).get(SLOT) or {}).get(LATENT)
    return {"plan": dict(p)} if isinstance(p, dict) else None


def probe_dict() -> dict:
    from haenv.world.generation.framings import framing_sha256
    from haenv.p2.hooks import p2_world_fingerprint
    from haenv.p2.rules import table_sha16
    return {"probe_id": PROBE_ID, "mode": "ddx", "difficulty_probe": "P0", "framing_ref": FRAMING,
            "framing_sha256": framing_sha256(FRAMING), "answer_space": "p2_triage", "hint_level": 0,
            "budget": {"max_rounds": 1}, "p2_world_sha16": p2_world_fingerprint(), "p2_table_sha16": table_sha16()}


def _judge(out, vp, ctx):
    from haenv.p2.hooks import p2_judging_fingerprint
    from haenv.p2.score import atoms
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return {**atoms(getattr(out, "_raw", None) or {}, g), "p2_judging_sha16": p2_judging_fingerprint()}


def _when(vp) -> bool:
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return isinstance(g, dict) and bool(g.get("set"))


KINDS = ("*",)


def pack():
    from haenv.app.packs import Pack
    from haenv.app.packs import PayloadView
    from haenv.p2.gate import emission
    from haenv.p2.history import POLICY
    return Pack(
        name="p2", history_dependency=POLICY, owns=_when, source="haenv.p2 (haenv/p2/plugin.py)",
        blocks=((NAME, _gold, lambda meta: None, (LATENT,), {LATENT: "knob"},
                 "② 分诊重做(设计 2026-10-08):可接受去向集合由规则读取器在题面块上按印出的规则给出;"
                 "题面块在测量层之后由 hooks.observe_triage 写入。"),),
        emission=emission(),
        framing=(FRAMING, prompt(), "② 重做:单回合、规则印在题面、四个去向、作答带规则编号与依据读数。"),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": "② 单回合无菜单:决定所需数据全部常显", "slices": "② 是今天的一次分诊,不切片",
                 "multi": "② 单回合,不是多轮追踪"},
        prompt_gold_meta=("outcome", "driver"), judging_segment="P2_JUDGING",
        base_labels=(), answers_action=False,
        # the item is read off the triage block; world streams are background only and the
        # quantities the block shows are removed in its window (`hooks.OVERLAP_STREAMS`)
        view=PayloadView(streams=("HbA1c",)))


def register() -> list:
    from haenv.app.packs import register_pack
    from haenv.p2.hooks import install
    install()
    return register_pack(pack())


def judges():
    """Entry point of the `haenv.p2` group."""
    return register()
