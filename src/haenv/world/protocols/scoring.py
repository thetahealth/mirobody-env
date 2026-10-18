"""A fallible electronic score receipt; clinical rule readers never consume it."""
from __future__ import annotations

from copy import deepcopy

from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached
from haenv.world.history.history_dependency import uniform

FIELDS = {"rr": "respiratory_rate_per_min", "spo2": "spo2_pct", "sbp": "systolic_bp_mmHg",
          "hr": "heart_rate_bpm", "temp": "temperature_c"}


@registry_cached("system_scoring.yaml")
def table():
    return load_registry("system_scoring.yaml")


def scale_of(block: dict) -> int:
    # A COPD label alone is insufficient; require the clinician's documented
    # diagnosis and scoring instruction, not the hidden plan or the gold.
    for row in block.get("clinical_history_record") or ():
        if row.get("condition") == "hypercapnic_respiratory_failure":
            if row.get("blood_gas_confirmed") is True and row.get("target_spo2") == [88, 92] and row.get("clinical_scale") == 2:
                return 2
    return 1


def parts(block: dict, *, scale: int | None = None, missing: str | None = None) -> dict:
    from haenv.p2 import rules as R
    from haenv.world.history.history_dependency import table as history_table
    n = R.table()["news2"]
    cfg = history_table()["packs"]["p2"].get("scale2") or table()["scale2"]
    check, ck = block["home_check_today"], R.checklist_of(block)
    scale = scale_of(block) if scale is None else scale
    oxy = bool(check.get("supplemental_oxygen", check.get("on_oxygen", False)))
    out = {}
    for k, field in FIELDS.items():
        bands = cfg.get("on_oxygen", cfg.get("oxygen")) if oxy else cfg["room_air"]
        bands = bands if k == "spo2" and scale == 2 else n[k]
        out[k] = 0 if field == missing else R._points(bands, check[field])
    out["acvpu"] = int(n["new_confusion_points"]) if ck.get("new_confusion") else 0
    out["oxygen"] = int(table()["scale2"]["oxygen_points"]) if oxy else 0
    return out


def receipt(block: dict, scale: int, missing: str | None = None) -> dict:
    inputs = {field: (None if field == missing else block["home_check_today"][field]) for field in FIELDS.values()}
    scores = parts(block, scale=scale, missing=missing)
    return {"instrument": "NEWS2", "spo2_scale": scale, "total": sum(scores.values()),
            "parameter_scores": scores, "input_observations": inputs,
            "input_stage": "本次家庭测量首次上传（补测前）", "source": "家庭监测系统自动计算",
            "missing_handling": "未收到字段计 0 分" if missing else "所有字段均已收到"}


def _normal(field: str, block: dict):
    from haenv.p2 import rules as R
    k = next(k for k, f in FIELDS.items() if f == field)
    bands = R.table()["news2"][k]
    if k == "spo2" and scale_of(block) == 2:
        from haenv.world.history.history_dependency import table as history_table
        cfg = history_table()["packs"]["p2"].get("scale2") or table()["scale2"]
        check = block["home_check_today"]
        oxy = bool(check.get("supplemental_oxygen", check.get("on_oxygen", False)))
        bands = cfg.get("on_oxygen", cfg.get("oxygen")) if oxy else cfg["room_air"]
    lo, hi, _ = next(b for b in bands if b[2] == 0)
    v = (max(0, lo) + hi) / 2
    return round(v, 1) if k == "temp" else int(v)


def trusted_assessment(block: dict, rec: dict) -> dict:
    """The blind-system stub: same rule reader on the system's effective inputs.

    A missing field counted as zero is represented by its zero-point band. The
    stub leaves all non-score evidence alone and ignores newer completed readings.
    """
    from haenv.p2 import rules as R
    b = deepcopy(block)
    if rec["spo2_scale"] == 1:
        b["clinical_history_record"] = [r for r in b.get("clinical_history_record") or ()
                                        if r.get("condition") != "hypercapnic_respiratory_failure"]
    check = b["home_check_today"]
    for field, value in rec["input_observations"].items():
        check[field] = _normal(field, b) if value is None else value
    return R.assess(b)


def candidates(block: dict) -> tuple[bool, list[str]]:
    from haenv.p2 import rules as R
    gold = R.assess(block)["set"]
    wrong_scale = scale_of(block) == 2
    missing = [f for f in FIELDS.values() if trusted_assessment(block, receipt(block, scale_of(block), f))["set"][0] not in gold]
    return wrong_scale, missing


def expected(block: dict, case_id: str) -> tuple[dict, str, dict]:
    rates = table()["rates"]
    ws, ms = candidates(block)
    if ws and uniform("system-score-scale", case_id) < rates["wrong_scale"]["value"]:
        return receipt(block, 1), "wrong_scale", {"wrong_scale": ws, "missing_zero": False}
    if ms and uniform("system-score-missing", case_id) < rates["missing_zero"]["value"]:
        field = ms[int(uniform("system-score-field", case_id) * len(ms))]
        return receipt(block, scale_of(block), field), "missing_zero", {"wrong_scale": ws, "missing_zero": True}
    return receipt(block, scale_of(block)), "correct", {"wrong_scale": ws, "missing_zero": bool(ms)}


def attach(block: dict, case_id: str):
    rec, _kind, _eligible = expected(block, case_id)
    block["home_check_today"]["system_news2"] = rec
    if any(v is None for v in rec["input_observations"].values()):
        block["home_check_today"]["measurement_stage"] = "首次上传后完成补测；下列生命体征为今天补齐后的读数"
