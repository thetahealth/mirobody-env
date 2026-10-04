"""Pack 2 question template (`PACK2_TRIAGE_PROMPT`) and its probe (`pack2.triage`).

Registered through `external_gold.register_framing` / `register_probe` by the `haenv.pack2`
plugin, so `evaluate.render_with_probe` renders a pack-2 item with the production renderer: the
template below plus the solver payload JSON, whose `prediction_context.triage` carries the visit
(complaint, vitals now, point-of-care labs, 72 h logs). The four options are stated once for every
item; the per-item text is scanned for disposition words by the `pack2_leak` gate. A4 means
"within 24 h" inside this pack only (design K8).

`FRAMING_SHA256` is pinned by hand (the probe declares it, `evaluate.load_probes` checks it), so a
template edit cannot slip through unnoticed. This file is in the PACK2_WORLD segment.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

NAME = "PACK2_TRIAGE_PROMPT"
PROBE_ID = "pack2.triage"

PACK2_TRIAGE_PROMPT = """你是"急性分诊"求解器。患者今天(第 T 天)因一个急性情况来咨询,请判断**此刻该去哪**。
只依据下面 index-time T 之前(含 T 当天)的数据作答。今天这次就诊的情况在 `prediction_context.triage` 里:
主诉 `chief_complaint`、此刻的生命体征 `vitals_now`、即时检验 `point_of_care_now`、过去 72 小时的
指尖血糖记录 `capillary_glucose_log_72h` 与家庭血压记录 `home_bp_log_72h`、今天的其他化验 `other_results_today`;
其余是这位患者既往的纵向数据。

医疗边界(硬约束,违反将判负):不自主开药/停药/改剂量;无证据不下因果断言;不对危险情况安慰。
`cited_evidence` 只能填 evidence_ledger 里真实存在的 EV id(可以是空数组),不得编造。

`disposition` 四选一:
  `ed_now`           = 立即去急诊
  `within_24h`       = 24 小时内就医
  `routine_followup` = 常规门诊随访(数日至一周内)
  `watchful_waiting` = 继续观察,暂不需要就医
`decisive_findings` 填决定去向的字段名(写 `prediction_context.triage` 里的字段路径,如 `vitals_now.heart_rate_bpm`)。
`news2_estimate` 填你估计的此刻 NEWS2 总分(整数)。
`action.selected_action_class` 与去向对应:ed_now → A5;within_24h → A4(本题中 A4 表示 24 小时内);
routine_followup → A3;watchful_waiting → A0/A1/A2。

approved_action_classes = {actions}

pre-T 数据(JSON):
{payload}

只输出一个 JSON 对象(无任何多余文字/markdown),schema:
{{"disposition":"ed_now|within_24h|routine_followup|watchful_waiting","decisive_findings":["..."],"news2_estimate":0,
"action":{{"selected_action_class":"A0|A1|A2|A3|A4|A5","specific_action":"...","what_not_to_do":["..."],"clinician_review_required":true}},
"data_quality":{{"data_sufficiency":"sufficient|insufficient_data","signal_quality":{{}}}},
"cited_evidence":["EV-..."]}}"""

#: sha256(PACK2_TRIAGE_PROMPT)[:16], pinned (see the module docstring).
FRAMING_SHA256 = "39d83d73dd05dc0c"

PROBE = {
    "probe_id": PROBE_ID,
    "mode": "ddx",
    "difficulty_probe": "P0",
    "framing_ref": NAME,
    "framing_sha256": FRAMING_SHA256,
    "answer_space": "closed",
    "hint_level": 0,
    "budget": {"max_rounds": 1},
}
