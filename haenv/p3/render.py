"""Pack 3 production renderer: the chronic-medication clinic record the solver reads.

Writes `prediction_context.meds_p3` from the realized item (`realize.realize`). It carries every
input the gold depends on -- the drug and its ladder, the target, the visit table (reading,
prescription, start / increase / continue), the refill record and today's pill count, the
symptom diary with onset days -- and nothing that names the gold (no cause of a symptom, no
truth, no adherence figure). The leak gate (`gate.leak_hits`) scans the rendered block and the
solver payload.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

TEXT_SOURCE = "template"
NOTE = ("你的建议由医生审核后执行。日数相对于本数据的第 0 天，负数为第 0 天之前的院内慢病档案记录。"
        "longitudinal_data 里的 dose_timeline 与 medication_adherence 属于患者的另一种在用药，与本次要调整的药无关。")


def _fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}" if nd else f"{int(round(v))}"


def _dose(v: float, unit: str) -> str:
    return f"{int(v) if float(v).is_integer() else v} {unit.split('/')[0]}"


def _labs(renal: dict, day: int, uacr: dict | None = None) -> dict:
    """Creatinine and potassium of the visit (A12: an ACEI is monitored on them); with CKD (A24)
    the eGFR beside the creatinine and the urine ACR where it was measured."""
    if day not in renal:
        return {}
    v = renal[day]
    egfr = f"（eGFR {v[2]} mL/min/1.73m²）" if len(v) > 2 else ""
    acr = f"，尿白蛋白/肌酐比 {uacr[day]} mg/g" if uacr and day in uacr else ""
    return {"labs": f"肌酐 {v[0]} μmol/L{egfr}，血钾 {v[1]:.1f} mmol/L{acr}"}


def visit_rows(item: dict) -> list[dict]:
    from ..labworld.meds import axis, drug
    d = drug(item["drug"])
    ax = axis(item["axis"])
    name = d["display"].split("（")[0]
    line = sorted(item["dose_line"], key=lambda x: int(x[0]))
    on_day = {int(dd): v for dd, v in line}
    renal = {int(r[0]): tuple(r[1:]) for r in item.get("renal") or ()}
    cm = item.get("comorbidity")
    uacr = None
    if cm:
        days = sorted(int(r["day"]) for r in item["visits"])
        uacr = {days[0]: cm["uacr"][0], int(item["T"]): cm["uacr"][1]}
    rows, cur = [], None
    for r in sorted(item["visits"], key=lambda x: int(x["day"])):
        day = int(r["day"])
        reading = f"{_fmt(float(r['printed']), int(ax['ndigits']))} {ax['unit']}"

        if day == int(item["T"]):
            rows.append({"day": day, "reading": reading, "prescription": "本次待定", "change": "—", **_labs(renal, day, uacr)})
            continue
        if cur is None and day not in on_day:
            # A19: a monitoring visit before the start
            rows.append({"day": day, "reading": reading, "prescription": "未用药", "change": "观察", **_labs(renal, day, uacr)})
            continue
        if day in on_day:
            new = on_day[day]
            change = "起始" if cur is None else f"加量（{_dose(cur, d['unit'])} → {_dose(new, d['unit'])}）"   # A10: the record's own wording
            cur = new
        else:
            change = "续方"
        rows.append({"day": day, "reading": reading, "prescription": f"{name} {_dose(cur, d['unit'])} 每日",
                     "change": change, **_labs(renal, day, uacr)})
    return rows


def render_block(item: dict) -> dict:
    from ..labworld.meds import axis, drug
    sup = int(item.get("supply", 30))
    d = drug(item["drug"])
    ax = axis(item["axis"])
    up = item.get("upper", ax["upper"])          # this patient's target (A12)
    tgt = f"{ax['display']} < {up} {ax['unit']}"
    if ax.get("lower") is not None:
        tgt = f"{ax['display']} {ax['lower']}–{up} {ax['unit']}"
    diary = sorted(item["diary"], key=lambda e: int(e["onset"]))
    cm = item.get("comorbidity")
    known = {}
    if cm:      # A24: the comorbidity the clinic knows, with its stage and the nephrology medication
        known = {"known_comorbidity": f"{cm['display']} {cm['stage']} 期，白蛋白尿 A3（肾内科随诊）"
                 + (f"；在用{cm['co_medication']}" if cm.get("co_medication") else "")}
    return {
        "condition": d["condition"], "drug": d["display"], **known,
        "dose_ladder": " / ".join(_dose(x, d["unit"]) for x in d["ladder"]) + "（每日）",
        "control_axis": ax["display"], "clinic_target": "本院目标：" + tgt,
        "visits": visit_rows(item),
        "refill_record": [{"day": int(p), "dispensed": f"{sup} 天量"} for p in item["refills"]],
        "pill_count_today": f"末次取药（第 {int(item['refills'][-1])} 天，{sup} 天量）后，今天药盒剩余 {int(item['pill_count'])} 天量",
        "symptom_diary": [{"symptom": e["symptom"], "onset_day": int(e["onset"]),
                           "context": e.get("context") or ("无明显诱因" if e.get("kind") == "drug_symptom" else "—"),
                           "severity": e["severity"], "status": e["status"]}
                          for e in diary],
        "today": int(item["T"]), "note": NOTE, "text_source": TEXT_SOURCE,
    }
