"""Expose the existing follow-up policy from its single executable source."""
from __future__ import annotations


def _example(cls, cause=None, *, previous=100, today=120, prior=False, baseline=None):
    rows = [{"day": 0, "printed": previous, "perturbations": [{"acts": True}] if prior else []},
            {"day": 1, "printed": today, "perturbations": [] if prior else [{"acts": True}]}]
    if baseline is not None:
        rows.insert(0, {"day": -1, "printed": baseline})
    return {"class": cls, "cause": cause, "t0": 0, "t1": 1, "records": rows}


def steps():
    from haenv.p4.gold import ACEI_RISE_MAX_PCT
    from haenv.p4.gold import next_step
    n = f"{ACEI_RISE_MAX_PCT:g}%"
    cases = [
        ("变化属于测量误差与个体内生物学波动", _example("analytic_biological_noise")),
        ("采血、标本或方法失真仅在上次，本次为标准读数", _example("preanalytical", prior=True)),
        ("采血、标本或方法失真在本次", _example("method_difference")),
        ("ACEI 停用后肌酐下降", _example("true_change", "acei_start_or_stop", today=90)),
        (f"ACEI 起用后肌酐较上次升幅不超过 {n}", _example("true_change", "acei_start_or_stop")),
        (f"ACEI 起用后肌酐较上次升幅超过 {n}，但较起用前读数均值不超过 {n}",
         _example("true_change", "acei_start_or_stop", today=140, baseline=120)),
        (f"其它真实变化，包括 ACEI 起用后较上次及起用前均值的升幅均超过 {n}",
         _example("true_change", "acei_start_or_stop", today=140)),
    ]
    labels = {"no_action": "无需因本次变化额外处理（继续常规监测）", "repeat_standardized": "在标准条件下复查确认",
              "act_on_change": "由医生针对变化评估处置"}
    return [f"{condition}：首选 {primary}（{labels[primary]}）；接受 " + " / ".join(accepted) + "。"
            for condition, blk in cases for primary, accepted in [next_step(blk)]]


def visible_step(sp, cls, candidates):
    """Only visible report values and medication dates enter the policy reader."""
    from haenv.p4.gold import next_step
    b = sp.prediction_context["followup_p4"]
    side = next((c[1] for c in candidates if c[0] == cls), None)
    cause = next((c[2] for c in candidates if c[0] == cls), None)
    rows = [{"day": int(r["day"]), "printed": r["value"],
             "perturbations": [{"acts": True}] if cls in ("preanalytical", "method_difference") and
             int(r["day"]) == int(b["previous_draw_day"] if side == -1 else b["today"]) else []}
            for r in b["report"]]
    blk = {"class": cls, "cause": cause, "t0": b["previous_draw_day"], "t1": b["today"], "records": rows,
           "acei": {"events": [{"day": r["day"]} for r in b.get("medication_record") or [] if "ACEI" in r["drug"]]}}
    return next_step(blk)[0]
