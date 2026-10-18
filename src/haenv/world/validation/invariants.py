"""Whole-pool deterministic contracts and diagnostic screens, checked on the actual
emission (`cases.jsonl`, `payloads.jsonl`, and for M2 what the production gated loop returns on a full
purchase, `haenv/world/validation/invariants_purchase.py`). `tools/validation/invariant_gates.py` reports every gate; `tools/generation/make_pack.py` refuses a pack on the gates in
`FAIL_CLOSED`. Statistical screens remain report-only and do not establish clinical impossibility.

Each gate: an id, a class (① same fact in several places · ② plausible values · ③ rule coverage ·
④ shortcuts), one sentence stating the property, the packs it applies to, and a check. Item gates
return a list of violation strings per item; pack gates read the whole published subset. Thresholds
and medical relations carry their source; `[memory]` marks one not checked against the original.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import functools
import math
import pathlib
import re
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached
from dataclasses import dataclass, field
from typing import Callable

from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data

ROOT = _dr()
PACK_BLOCK = {"p2": "triage_p2", "p3": "meds_p3", "p4": "followup_p4", "m2": "m2_frame"}

# ----------------------------------------------------------------------------------------- loading


def load(batch, pack: str, purchases=None) -> list[dict]:
    """One dict per emitted case: case (truth included), sp (the solver payload), pack, buy (M2)."""
    if pack not in PACK_BLOCK:
        raise ValueError(f"unsupported invariant pack: {pack}")
    batch = pathlib.Path(batch)
    pays = {}
    for line in open(batch / "payloads.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["case_id"] in pays:
            raise ValueError(f"duplicate payload: {r['case_id']}")
        pays[r["case_id"]] = r["sp"]
    buys = {}
    if purchases:
        for line in open(purchases, encoding="utf-8"):
            r = json.loads(line)
            if r["case_id"] in buys:
                raise ValueError(f"duplicate purchase: {r['case_id']}")
            buys[r["case_id"]] = r
    out = []
    ids = set()
    for line in open(batch / "cases.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["case_id"] in ids:
            raise ValueError(f"duplicate case: {r['case_id']}")
        ids.add(r["case_id"])
        out.append({"case_id": r["case_id"], "pack": pack, "case": r["case"], "sp": pays.get(r["case_id"]),
                    "buy": buys.get(r["case_id"])})
    if not ids or set(pays) != ids or any(not it["sp"] for it in out):
        raise ValueError("invariant pool needs nonempty cases and complete, matching payloads")
    if pack == "m2" and (set(buys) != ids or any(
            not isinstance(b.get("results"), dict) or not isinstance(b.get("world_ld"), dict)
            or "menu" not in b for b in buys.values())):
        raise ValueError("M2 invariant pool needs complete, matching purchase records")
    return out


# ----------------------------------------------------------------------------------------- views

def T_of(it) -> int:
    return int((it["sp"].get("prediction_context") or {}).get("prediction_time_T") or 0)


def gold(it) -> dict:
    return ((it["case"].get("adjudication") or {}).get(PACK_BLOCK[it["pack"]]) or {})


def block(it) -> dict:
    return ((it["sp"].get("prediction_context") or {}).get(PACK_BLOCK[it["pack"]]) or {})


def age_band(it) -> tuple[int, int] | None:
    m = re.match(r"(\d+)-(\d+)", str((it["sp"].get("user_profile") or {}).get("age_range") or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def sex(it) -> str:
    return str((it["sp"].get("user_profile") or {}).get("sex") or "")


def world_conditions(it) -> set[str]:
    pb = (it["case"].get("latent_premise") or {}).get("patient_basics") or {}
    return {str(pb.get("disease") or "")} | {str(x) for x in pb.get("comorbidities") or ()}


def record_conditions(it) -> set[str]:
    out = set((it["sp"].get("user_profile") or {}).get("known_conditions") or ())
    if it["pack"] == "p2":
        out |= set((block(it).get("background") or {}).get("known_conditions") or ())
    return {str(x) for x in out}


#: world drug id -> the names the visible text uses (registry/labworld.yaml:drugs, haenv/p2/render.py:MEDS,
#: haenv/world/observations/events.py:DRUG_ZH, p3 registry names); the M2 scene prints the id itself
DRUG_NAMES = {
    "semaglutide": ("司美格鲁肽", "semaglutide"), "liraglutide": ("利拉鲁肽", "liraglutide"),
    "dulaglutide": ("度拉糖肽", "dulaglutide"), "tirzepatide": ("替尔泊肽", "tirzepatide"),
    "metformin": ("二甲双胍", "metformin"), "amlodipine": ("氨氯地平", "amlodipine"),
    "atorvastatin": ("阿托伐他汀", "atorvastatin"), "losartan": ("氯沙坦", "losartan"),
    "enalapril": ("依那普利", "enalapril"), "levothyroxine": ("左甲状腺素", "levothyroxine", "优甲乐"),
    "gliclazide": ("格列齐特", "gliclazide"), "glimepiride": ("格列美脲", "glimepiride"),
    "insulin_glargine": ("甘精胰岛素", "胰岛素", "insulin"), "dapagliflozin": ("达格列净", "dapagliflozin"),
    "furosemide": ("呋塞米", "furosemide"), "bisoprolol": ("比索洛尔", "bisoprolol"),
    "metoprolol": ("美托洛尔", "metoprolol"),
}
ANTIHYPERTENSIVES = {"amlodipine", "losartan", "enalapril", "bisoprolol", "metoprolol", "furosemide"}
GLUCOSE_LOWERING = {"semaglutide", "liraglutide", "dulaglutide", "tirzepatide", "metformin", "gliclazide",
                    "glimepiride", "insulin_glargine", "dapagliflozin"}
#: cause hypoglycaemia by themselves (ADA Standards of Care 2025 §6 / §9 [memory]): insulin and secretagogues
HYPO_DRUGS = {"gliclazide", "glimepiride", "insulin_glargine"}


def drugs_in(text: str) -> set[str]:
    t = str(text or "")
    return {d for d, names in DRUG_NAMES.items() if any(n in t for n in names)}


def visible_med_text(it) -> str:
    """The text where the solver reads the patient's current medicines."""
    b = block(it)
    if it["pack"] == "p2":
        return " ".join((b.get("background") or {}).get("medications") or ())
    if it["pack"] == "p3":
        return " ".join([str(b.get("drug") or ""), str(b.get("known_comorbidity") or "")]
                        + [str(v.get("prescription") or "") for v in b.get("visits") or ()]
                        + [str(x) for x in b.get("other_medications") or ()])
    if it["pack"] == "p4":
        return " ".join(str(m.get("drug") or "") for m in b.get("medication_record") or ())
    if it["pack"] == "m2":
        current = (it["sp"].get("prediction_context") or {}).get("current_medications") or ()
        s = str(b.get("scene") or "")
        return " ".join([*current, *re.findall(r"在用 ([^。]*)", s)])
    return ""


def visible_drugs(it) -> set[str]:
    return drugs_in(visible_med_text(it))


def world_drugs(it, T: int | None = None) -> dict[str, float]:
    """World drug -> dose on day T (0 = stopped)."""
    wm = (it["case"].get("latent_premise") or {}).get("world_medication") or {}
    T = T_of(it) if T is None else T
    out = {}
    for x in [wm.get("primary")] + list(wm.get("concurrent") or ()):
        if not x or not x.get("drug"):
            continue
        dose = 0.0
        for q in x.get("dose_timeline") or ():
            if int(q["ts"]) <= T:
                dose = float(q["value"] or 0)
        out[str(x["drug"])] = dose
    return out


def dose_timeline(it, drug: str) -> list[tuple[int, float]]:
    wm = (it["case"].get("latent_premise") or {}).get("world_medication") or {}
    for x in [wm.get("primary")] + list(wm.get("concurrent") or ()):
        if x and x.get("drug") == drug:
            return [(int(q["ts"]), float(q["value"] or 0)) for q in x.get("dose_timeline") or ()]
    return []


#: purchased finding id -> the stream it measures
FINDING_STREAM = {"FBG": "fasting_glucose", "HbA1c": "HbA1c", "LDL": "LDL", "TG": "triglycerides",
                  "ALT": "ALT", "AST": "AST", "FIB4": "FIB4", "Cr": "creatinine"}
LAB_STREAMS = ("HbA1c", "fasting_glucose", "LDL", "triglycerides", "ALT", "AST", "FIB4", "CGM_TIR", "creatinine")
BP_STREAMS = ("systolic_bp", "diastolic_bp")


def record_streams(it) -> dict:
    """The longitudinal record the solver reads: M2 reads the world its plugin rewrites before the loop
    (F3 follow-up, FIB-4 from its inputs), captured with the purchases."""
    buy = it.get("buy") or {}
    if it["pack"] == "m2" and buy.get("world_ld") is not None:
        return buy["world_ld"]
    return it["sp"].get("longitudinal_data") or {}


def _num(v):
    try:
        x = float(str(v).split()[0])
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, IndexError):
        return None


def labs(it) -> list[dict]:
    """Every lab value the solver can read or buy up to T: {an, day, v, ch, ref_low, ref_high, flag, raw}."""
    T = T_of(it)
    out = []
    ld = record_streams(it)
    for s in LAB_STREAMS + BP_STREAMS:
        for q in ld.get(s) or ():
            if int(q["ts"]) <= T and _num(q.get("value")) is not None:
                out.append({"an": s, "day": int(q["ts"]), "v": _num(q["value"]), "ch": "record", "time": q.get("time")})
    b = block(it)
    if it["pack"] == "p4":
        an = str(b.get("analyte") or "")
        for r in b.get("report") or ():
            if _num(r.get("value")) is not None:
                out.append({"an": an, "day": int(r["day"]), "v": _num(r["value"]), "ch": "report", "time": r.get("time")})
    if it["pack"] == "p3":
        an = str(gold(it).get("axis") or "")
        for v in b.get("visits") or ():
            if _num(v.get("reading")) is not None:
                out.append({"an": an, "day": int(v["day"]), "v": _num(v["reading"]), "ch": "visit"})
    if it["pack"] == "p2":
        for r in b.get("cgm_nightly") or ():
            x = _num(r.get("min_mmol_L"))
            if x is not None:
                out.append({"an": "cgm_night_min", "day": int(r["night_of_day"]), "v": x, "ch": "block"})
    buy = it.get("buy") or {}
    for tgt, res in (buy.get("results") or {}).items():
        for o in res.get("observation") or ():
            if not isinstance(o, dict) or not o.get("finding_id"):
                continue
            fid = str(o["finding_id"])
            out.append({"an": FINDING_STREAM.get(fid, fid), "day": int(o.get("ts", T)), "v": _num(o.get("value")),
                        "raw": o.get("value"), "ch": "buy", "target": tgt, "ref_low": o.get("ref_low"),
                        "ref_high": o.get("ref_high"), "flag": o.get("flag"), "fid": fid})
    fu = _followup(it)
    if fu:
        out.append({"an": fu["an"], "day": T, "v": fu["v"], "ch": "followup"})
    return out


def _followup(it):
    p = str((it.get("buy") or {}).get("prompt") or "")
    m = re.search(r"followup_result: (\{.*?\})\s*$", p, re.M)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except ValueError:
        return None
    an = str(d.get("indicator") or d.get("stream") or (block(it).get("followed_indicator") or ""))
    v = _num((d.get("today") or {}).get("value") if isinstance(d.get("today"), dict) else d.get("today"))
    return {"an": an, "v": v} if v is not None else None


def texts(it) -> list[tuple[int, str]]:
    """Free text the solver reads: ledger entries, the message / scene, the symptom diary."""
    out = []
    for e in it["sp"].get("evidence_ledger") or ():
        t = " ".join(str(e.get(k) or "") for k in ("symptom", "context", "note", "text", "claim"))
        out.append((int(e.get("source_timestamp") or 0), t))
    b = block(it)
    for k in ("message", "scene"):
        if b.get(k):
            out.append((T_of(it), str(b[k])))
    for d in b.get("symptom_diary") or ():
        out.append((int(d.get("onset_day") or 0), f"{d.get('symptom') or ''} {d.get('context') or ''}"))
    return out


def weights(it) -> list[float]:
    T = T_of(it)
    kg = [float(q["value"]) for q in (it["sp"].get("longitudinal_data") or {}).get("weight") or ()
          if int(q["ts"]) <= T and _num(q.get("value")) is not None]
    kg += [float(r["kg"]) for r in block(it).get("morning_weight") or () if _num(r.get("kg")) is not None]
    return kg


# ----------------------------------------------------------------------------------------- gates

@dataclass
class Gate:
    id: str
    cls: str
    statement: str
    packs: tuple[str, ...]
    check: Callable                    # item -> [violation] (scope item) | items -> [violation] (scope pack)
    scope: str = "item"
    gold_risk: str = ""                # how a violation can make a clinically right answer score wrong
    source: str = ""
    control: str = ""                  # description of the paired negative control
    evidence_kind: str = "contract"    # contract | screening | mixed; a screen is not a clinical impossibility


GATES: list[Gate] = []


def gate(**kw):
    def deco(fn):
        GATES.append(Gate(check=fn, **kw))
        return fn
    return deco


ALL = ("m2", "p2", "p3", "p4")


# ① same fact in several places --------------------------------------------------------------------

@gate(id="C1", cls="①", packs=ALL, control="add a condition to the shown record that the world lacks",
      statement="Every chronic condition on the shown record is a condition of the world the labs come from.",
      gold_risk="low: the record names a disease whose labs never move")
def c1(it):
    w = world_conditions(it)
    return [f"record:{c}" for c in sorted(record_conditions(it)) if c not in w]


#: drug -> the visible streams it acts on (registry/drug_effects.yaml direction table, by class)
DRUG_ACTS = {d: ("HbA1c", "fasting_glucose", "CGM_TIR", "weight") for d in
             ("semaglutide", "liraglutide", "dulaglutide", "tirzepatide")}
DRUG_ACTS.update({"metformin": ("HbA1c", "fasting_glucose", "CGM_TIR"), "atorvastatin": ("LDL",),
                  "amlodipine": ("systolic_bp", "diastolic_bp"), "losartan": ("systolic_bp", "diastolic_bp"),
                  "enalapril": ("systolic_bp", "diastolic_bp", "creatinine"), "levothyroxine": ("TSH",)})


@gate(id="C2", cls="①", packs=ALL, control="drop the active world drug from the visible medication text",
      statement="A world drug active on day T that acts on a readable stream is visible; every shown drug "
                "exists in the world. M2 current lists and scene must agree with active doses on T.",
      gold_risk="high: a reading explained by an unseen drug (HbA1c 4.5 on 'metformin only') reads as disease or "
                "overtreatment; F2/P3 directions hinge on what the patient takes")
def c2(it):
    vis, wd = visible_drugs(it), world_drugs(it)
    seen = {x["an"] for x in labs(it)} | ({"weight"} if weights(it) else set())
    ld = it["sp"].get("longitudinal_data") or {}
    seen |= {k for k in ("systolic_bp", "diastolic_bp") if ld.get(k)} | (
        {"systolic_bp"} if it["pack"] == "p2" and block(it).get("home_bp_log") else set())
    out = [f"world_not_shown:{d}" for d, dose in sorted(wd.items())
           if dose > 0 and d not in vis and set(DRUG_ACTS.get(d, ())) & seen]
    out += [f"shown_not_world:{d}" for d in sorted(vis) if d not in wd]
    if it["pack"] == "m2":
        from haenv.world.clinical.med_course import active_medicines
        from haenv.world.clinical.med_course import display
        wm = (it["case"].get("latent_premise") or {}).get("world_medication") or {}
        active = {drug: dose for drug, dose, _ in active_medicines(wm, T_of(it))}
        out += [f"shown_not_active:{d}" for d in sorted(vis & wd.keys()) if d not in active]
        pc = it["sp"].get("prediction_context") or {}
        if "current_medications" in pc:
            expected = [display(d, dose) for d, dose in active.items()]
            if sorted(pc["current_medications"]) != sorted(expected):
                out.append("current_medications_active_dose_mismatch")
            scene = str(block(it).get("scene") or "")
            if block(it):
                shown = re.findall(r"在用 ([^。]*)", scene)
                if (expected and shown != ["、".join(expected)]) or (
                        not expected and (shown or "目前未使用药物" not in scene)):
                    out.append("scene_active_dose_mismatch")
        else:
            # Legacy framed payloads: do not exempt stopped/future drugs simply
            # because no canonical table has been written yet.
            text = visible_med_text(it)
            for drug, dose in active.items():
                for name in DRUG_NAMES.get(drug, ()):
                    for m in re.finditer(re.escape(name) + r"\s+(\d+(?:\.\d+)?)\s*(?:mg|U)\b", text):
                        if float(m[1]) != dose:
                            out.append(f"shown_dose_mismatch:{drug}")
    return out


#: text conclusion patterns: (regex, analyte, kind) — kind "num" reads the number, "low"/"high"/"pos" a direction
# A bare element must be a token or follow an assay context, not the suffix of
# a chemical/drug name. Mass/tablet units describe a dose, not a concentration.
# Keep the whole numeric token: a rejected "1000 mg" must not backtrack to "100".
_ELECTROLYTE_CONTEXT = r"(?:血(?:清|浆|液)?|(?:化验|复查|测得|检测|检查)(?:结果|显示|提示)?|(?<!\w))"
_ELECTROLYTE_NUMBER = (r"[^0-9，。,;；]{0,6}?(\d+(?:\.\d+)?)(?![\d.])"
                       r"(?!\s*(?:μg|µg|mcg|ug|mg|g|微克|毫克|克|片|粒)(?![A-Za-z]|/(?:dL|L)))")
TEXT_LAB = [
    (_ELECTROLYTE_CONTEXT + "钾" + _ELECTROLYTE_NUMBER, "K", "num"),
    (_ELECTROLYTE_CONTEXT + "钠" + _ELECTROLYTE_NUMBER, "Na", "num"),
    (r"(?:糖化|HbA1c|A1c)[^0-9，。,;；]{0,6}?(\d+(?:\.\d+)?)", "HbA1c", "num"),
    (r"空腹血糖[^0-9，。,;；]{0,6}?(\d+(?:\.\d+)?)", "fasting_glucose", "num"),
    (r"低钾", "K", "low"), (r"高钾", "K", "high"), (r"低钠", "Na", "low"),
    (r"(?:尿|查出|发现有?)酮体|酮症", "UrineKetone", "pos"),
]
#: normal band per analyte for the direction check (registry/findings.yaml ref)
DIR_REF = {"K": (3.5, 5.5), "Na": (135, 145)}


@gate(id="C3", cls="①", packs=ALL, control="write a potassium number into a ledger entry far from every K the case returns",
      statement="A lab conclusion in the visible text (a number or a direction) agrees with the same analyte's "
                "readable or purchasable values within 30 days.",
      gold_risk="high: the text and the purchase point to different urgency (ketones in the complaint, negative on purchase)")
def c3(it):
    L = labs(it)
    out = []
    for day, t in texts(it):
        for rx, an, kind in TEXT_LAB:
            for m in re.finditer(rx, t):
                near = [x for x in L if x["an"] == an and abs(x["day"] - day) <= 30]
                if not near:
                    continue
                if kind == "num":
                    v = float(m.group(1))
                    if not any(x["v"] is not None and abs(x["v"] - v) <= 0.15 * max(abs(v), 1e-9) for x in near):
                        out.append(f"{an} text {v} vs {[x['v'] for x in near][:4]} (day {day})")
                elif kind in ("low", "high"):
                    lo, hi = DIR_REF[an]
                    vs = [x["v"] for x in near if x["v"] is not None]
                    ok = any(v < lo for v in vs) if kind == "low" else any(v > hi for v in vs)
                    if vs and not ok:
                        out.append(f"{an} text '{m.group(0)}' vs {vs[:4]} (day {day})")
                elif kind == "pos":
                    if all(str(x.get("flag")) == "NEGATIVE" for x in near):
                        out.append(f"{an} text '{m.group(0)}' vs purchase NEGATIVE (day {day})")
    return sorted(set(out))


@gate(id="C4", cls="①", packs=ALL, control="change a purchased HbA1c to differ from the record's HbA1c of the same day",
      statement="An analyte read on the same day in two channels (record stream, report, visit, purchase, follow-up) "
                "has one value (tolerance 2% or one rounding unit).",
      gold_risk="high: F3 exceeds/within and F2 directions flip with the channel the solver reads")
def c4(it):
    by = {}
    for x in labs(it):
        if x["v"] is None:
            continue
        by.setdefault((x["an"], x["day"]), []).append(x)
    out = []
    for (an, d), xs in sorted(by.items()):
        chans = {}
        for x in xs:
            chans.setdefault(x["ch"], set()).add(x["v"])
        if len(chans) < 2:
            continue
        vals = sorted(v for s in chans.values() for v in s)
        if vals[-1] - vals[0] > max(0.02 * abs(vals[-1]), 0.051 if vals[-1] < 20 else 0.51):
            out.append(f"{an}@{d}: " + ", ".join(f"{c}={sorted(s)}" for c, s in sorted(chans.items())))
    return out


@gate(id="C5", cls="①", packs=("m2", "p3", "p4"), control="move the gold's F2 reading off every readable value",
      statement="Every value the gold reads is one the solver can read or buy (M2 F2 reading; P3 visit readings, "
                "and the adherence a(T) equal to the printed rule's coverage of the printed refills, outside the gray "
                "band on target-missed items; P4 printed records), and every M2 gold test returns a result.",
      gold_risk="high: the gold is computed on a value the solver never sees")
def c5(it):
    out = []
    g = gold(it)
    if it["pack"] == "m2":
        fr = g.get("f2_reading") or {}
        if fr.get("marker") and fr.get("last") is not None:
            an = FINDING_STREAM.get(fr["marker"], fr["marker"])
            d = int(fr.get("last_day") if fr.get("last_day") is not None else T_of(it))
            seen = [x["v"] for x in labs(it) if x["an"] == an and x["day"] == d and x["v"] is not None]
            if an == "systolic_bp":         # the gold reads the mean home SBP of the 28 days up to T (m2/core.py)
                sb = [x["v"] for x in labs(it) if x["an"] == an and x["ch"] == "record" and T_of(it) - 28 < x["day"] <= T_of(it)]
                seen = [sum(sb) / len(sb)] if sb else []
            if not any(abs(v - float(fr["last"])) <= 0.02 * abs(float(fr["last"])) + 1e-9 for v in seen):
                out.append(f"f2_reading {an}={fr['last']}@{d} not readable (seen {seen[:3]})")
        buy = (it.get("buy") or {}).get("results") or {}
        if buy and g:
            ddx = (it["case"].get("adjudication") or {}).get("ddx") or {}
            for t in ddx.get("tests") or ():
                r = buy.get(t)
                if r is not None and r.get("outcome") != "returned":
                    out.append(f"gold test unavailable: {t}")
    if it["pack"] == "p3":
        vis = {int(v["day"]): _num(v.get("reading")) for v in block(it).get("visits") or ()}
        for v in g.get("visits") or ():
            if int(v["day"]) in vis and vis[int(v["day"])] is not None and abs(vis[int(v["day"])] - float(v["printed"])) > 1e-6:
                out.append(f"visit {v['day']} printed {v['printed']} shown {vis[int(v['day'])]}")
        cov = p3_visible_coverage(it)
        g3 = g.get("g3") or {}
        a = g3.get("a_T")
        if cov is not None and a is not None:
            from haenv.world.labworld.meds import rule     # the printed adherence rule (`haenv/p3/render.py:adherence_rule`)
            line, (lo, hi) = float(rule()["adherence_line"]), rule()["adherence_gray"]
            if abs(cov - float(a)) > 1e-3:
                out.append(f"gold a_T {float(a):.4f} is not the printed record's coverage {cov:.4f}")
            if g3.get("control") == "off" and ((cov >= line) != (float(a) >= line) or lo <= cov < hi):
                out.append(f"coverage from visible refills {cov:.3f} vs gold a_T {float(a):.3f}: side of {line} "
                           "not settled by the printed record")
    if it["pack"] == "p4":
        rep = {int(r["day"]): _num(r.get("value")) for r in block(it).get("report") or ()}
        for r in g.get("records") or ():
            if int(r["day"]) in rep and rep[int(r["day"])] is not None and abs(rep[int(r["day"])] - float(r["printed"])) > 1e-6:
                out.append(f"record {r['day']} printed {r['printed']} shown {rep[int(r['day'])]}")
    return out


def p3_visible_coverage(it) -> float | None:
    """The coverage the printed adherence rule gives on the visible block alone: the last window days, from
    the visit marked 起始 when the drug started inside them; a pickup's supply covers the days after it,
    an early refill's surplus carried forward, a new-strength pickup (新规格) setting the old stock aside;
    after the last pickup the covered days are the doses today's pill count says were taken. A day on the
    drug before the first printed pickup is uncovered (proportion of days covered; PQA / Nau 2012 [memory])."""
    b, T = block(it), T_of(it)
    rec = []
    for r in b.get("refill_record") or ():
        m = re.match(r"(\d+)", str(r.get("dispensed") or ""))
        if m:
            rec.append((int(r["day"]), int(m.group(1)), "新规格" in str(r.get("dispensed"))))
    pc = re.search(r"剩余\s*(\d+)", str(b.get("pill_count_today") or ""))
    if not rec or not pc:
        return None
    rec.sort()
    from haenv.world.labworld.meds import rule
    starts = [int(v["day"]) for v in b.get("visits") or () if v.get("change") == "起始"]
    lo = max(T - int(rule()["adherence_window_days"]) + 1, min(starts) if starts else rec[0][0])
    covered, stock = 0, 0
    for (d, n, new), nxt in zip(rec, [x[0] for x in rec[1:]] + [None]):
        stock = n + (0 if new else stock)
        k = max(0, min(T - d, stock - int(pc.group(1)))) if nxt is None else min(stock, nxt - d)
        stock -= k
        covered += sum(1 for x in range(d, d + k) if lo <= x <= T)
    return covered / float(T + 1 - lo)


#: adult obesity BMI >= 28.0 (WS/T 428-2013); a short adult's height (GB/T 10000-1988 5th percentile,
#: haenv/p2/gate.py:SHORT_HEIGHT_M); a tall adult's height (95th percentile, men 1.775 / women 1.659,
#: GB/T 10000-1988 [memory]) for the "always thin" direction
OBESE_BMI, SHORT_H, TALL_H = 28.0, {"M": 1.583, "F": 1.484}, {"M": 1.775, "F": 1.659}
THIN_WORDS = ("一直瘦", "瘦瘦的", "我也不胖", "不胖", "偏瘦", "一直很瘦")


@gate(id="C6", cls="①", packs=ALL, control="set every visible weight of an obesity-record case to 45 kg",
      statement="Body build on the record or in the patient's words agrees with the visible weights: obesity needs a "
                "weight reaching BMI 28 at a short adult's height; 'always thin' allows no weight reaching BMI 28 "
                "at a tall adult's height and no obesity on the record.",
      gold_risk="low")
def c6(it):
    out, kg, s = [], weights(it), sex(it)
    if s not in SHORT_H or not kg:
        return out
    rec = record_conditions(it)
    if "obesity" in rec and max(kg) < OBESE_BMI * SHORT_H[s] ** 2:
        out.append(f"obesity on record, max weight {max(kg):.1f} kg")
    thin = [t for _, t in texts(it) if any(w in t for w in THIN_WORDS)]
    if thin and ("obesity" in rec or min(kg) >= OBESE_BMI * TALL_H[s] ** 2):
        out.append(f"'thin' in text ({thin[0][:30]}) vs obesity/min weight {min(kg):.1f} kg")
    return out


#: critical values (Chinese expert consensus on laboratory critical values 2017 and CAP lists [memory]):
#: K < 2.5 or > 6.5 mmol/L; glucose < 2.8 (2.2 in some lists) or > 25 mmol/L; Na < 120 or > 160 mmol/L
CRITICAL = {"K": (2.5, 6.5), "Na": (120, 160), "fasting_glucose": (2.8, 25.0)}


@gate(id="C7", cls="①", packs=("m2", "p2"), control="set a purchased K to 2.2 in a 🟡 case",
      statement="A readable or purchasable critical lab value on day T (K <2.5/>6.5, Na <120/>160, glucose "
                "<2.8/>25 mmol/L) comes with urgency 🔴 (M2) or a disposition set reaching ed_now/within_24h (P2).",
      gold_risk="high: the clinically right escalation is scored as over-triage")
def c7(it):
    T, out = T_of(it), []
    crit = [x for x in labs(it) if x["an"] in CRITICAL and x["v"] is not None and T - 3 <= x["day"] <= T
            and not (CRITICAL[x["an"]][0] <= x["v"] <= CRITICAL[x["an"]][1])]
    if not crit:
        return out
    ddx = (it["case"].get("adjudication") or {}).get("ddx") or {}
    if it["pack"] == "m2":
        g = gold(it)
        urg = g.get("urgency") or ddx.get("urgency")
        if g.get("frame") == "F1" and urg != "🔴":
            out.append(f"critical {[(x['an'], x['v']) for x in crit][:3]} with urgency {urg}")
    if it["pack"] == "p2":
        st = set(gold(it).get("set") or ())
        if not st & {"ed_now", "within_24h"}:
            out.append(f"critical {[(x['an'], x['v']) for x in crit][:3]} with set {sorted(st)}")
    return out


@gate(id="C8", cls="①", packs=("m2", "p3", "p4"), control="double a purchased TC",
      statement="A composite agrees with its parts on the same day: FIB-4 = age x AST / (PLT x sqrt(ALT)) within 15% "
                "(Sterling 2006, age at the band midpoint ±2.5 y); purchased TC follows the generator's declared "
                "direct-LDL residual model when TG < 4.5, not exact Friedewald equality; a FIB-4 stream with "
                "AST/ALT of the day implies a platelet count 50–600.",
      gold_risk="medium: FIB-4 crosses 1.3/2.67 between channels", evidence_kind="mixed")
def c8(it):
    by = {}
    for x in labs(it):
        if x["v"] is not None:
            by.setdefault(x["day"], {}).setdefault(x["an"], []).append(x["v"])
    ab = age_band(it)
    out = []
    for d, m in sorted(by.items()):
        if ab and all(k in m for k in ("AST", "ALT", "PLT", "FIB4")):
            ages = (ab[0], ab[1] + 1)
            for f in m["FIB4"]:
                lo = min(a * m["AST"][0] / (m["PLT"][0] * math.sqrt(m["ALT"][0])) for a in ages)
                hi = max(a * m["AST"][0] / (m["PLT"][0] * math.sqrt(m["ALT"][0])) for a in ages)
                if not (0.85 * lo <= f <= 1.15 * hi):
                    out.append(f"FIB4@{d} {f} vs parts [{lo:.2f},{hi:.2f}]")
        elif ab and all(k in m for k in ("AST", "ALT", "FIB4")) and "PLT" not in m:
            age = sum(ab) / 2
            plt = age * m["AST"][0] / (m["FIB4"][0] * math.sqrt(m["ALT"][0])) if m["FIB4"][0] > 0 else None
            if plt is not None and not (50 <= plt <= 600):
                out.append(f"screen: FIB4@{d} implies PLT {plt:.0f}; no measured PLT to verify the formula")
        if all(k in m for k in ("TC", "LDL", "HDL", "triglycerides")) and m["triglycerides"][0] < 4.5:
            from haenv.world.observations.findings_render import tc_friedewald_residual
            ldl, hdl, tg = m["LDL"][0], m["HDL"][0], m["triglycerides"][0]
            # LDL is measured directly. The production model deliberately draws
            # a signed residual fitted to the calibration split. Check that
            # declared composition, rather than treating its residual as error.
            rho = tc_friedewald_residual(str(it["case_id"]), ldl, tg)
            calc = ldl + hdl + tg / 2.2 - rho * ldl
            if abs(m["TC"][0] - calc) > 0.02:
                out.append(f"TC@{d} {m['TC'][0]} vs direct-LDL residual composition {calc:.2f}")
    return out


@gate(id="C9", cls="①", packs=ALL, control="duplicate one HbA1c record on its own day",
      statement="A lab stream has at most one record per day.",
      gold_risk="low")
def c9(it):
    ld, T, out = record_streams(it), T_of(it), []
    for s in LAB_STREAMS:
        days = [int(q["ts"]) for q in ld.get(s) or () if int(q["ts"]) <= T]
        dup = sorted({d for d in days if days.count(d) > 1})
        if dup:
            out.append(f"{s} days {dup[:5]}")
    return out


AGE_CN = {"二十": 20, "三十": 30, "四十": 40, "五十": 50, "六十": 60, "七十": 70, "八十": 80}
DIGIT_CN = {"": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
KIN = ("儿子", "女儿", "孩子", "孙", "爸", "妈", "父", "母", "老伴", "丈夫", "妻子", "爱人", "同事", "朋友", "邻居", "哥", "姐", "弟", "妹")
MENSES = ("月经", "例假", "经期", "来事")


@gate(id="C10", cls="①", packs=ALL, control="write '我今年三十八岁' into a 60-64 case's ledger",
      statement="The patient's own age and sex in the visible text agree with the profile: a stated own age falls in "
                "the age band (±1 y); no menstruation text for a man or a woman whose band starts at 55 unless it "
                "mentions menopause.",
      gold_risk="low")
def c10(it):
    ab, s, out = age_band(it), sex(it), []
    for day, t in texts(it):
        for m in re.finditer(r"(二十|三十|四十|五十|六十|七十|八十)([一二三四五六七八九]?)岁|(\d{2})岁", t):
            pre = t[max(0, m.start() - 6):m.start()]
            if any(k in pre for k in KIN):
                continue
            a = int(m.group(3)) if m.group(3) else AGE_CN[m.group(1)] + DIGIT_CN[m.group(2)]
            if ab and not (ab[0] - 1 <= a <= ab[1] + 1):
                out.append(f"age {a} in text vs band {ab[0]}-{ab[1]} (day {day})")
        if any(w in t for w in MENSES) and "绝经" not in t and (s == "M" or (s == "F" and ab and ab[0] >= 55)):
            out.append(f"menses text for {s} {ab} (day {day}): {t[:30]}")
    return sorted(set(out))


COUNT_CN = {"两": 2, "二": 2, "三": 3, "四": 4}


@gate(id="C11", cls="①", packs=ALL, control="write '降压药三种都用上了' into a case with no antihypertensive",
      statement="Drugs the visible text says the patient takes are on the visible medication list or in the world, "
                "and a stated number of antihypertensives equals the visible count.",
      gold_risk="medium: a 'three antihypertensives' resistant-hypertension cue the medication list contradicts")
def c11(it):
    vis, wd, out = visible_drugs(it), set(world_drugs(it)), []
    n_htn = len(vis & ANTIHYPERTENSIVES)
    for day, t in texts(it):
        if it["pack"] == "m2" and t == str(block(it).get("scene") or ""):
            continue
        for d in sorted(drugs_in(t) - vis - wd):
            out.append(f"text names {d} (day {day})")
        for m in re.finditer(r"([两二三四])种(?:降压)?药", t):
            if "降压" in t or "血压" in t:
                if COUNT_CN[m.group(1)] != n_htn:
                    out.append(f"text '{m.group(0)}' vs {n_htn} visible antihypertensives (day {day})")
    return sorted(set(out))


# ② plausible values --------------------------------------------------------------------------------

def _domain(it) -> dict:
    return ((it["case"].get("latent_premise") or {}).get("device_signals") or {}).get("signals") or {}


@gate(id="V1", cls="②", packs=ALL, control="set two HbA1c readings to the case's registered lower bound",
      statement="Screen for readings on the registered value-domain boundary. Display rounding can legitimately "
                "hit a boundary; diagnose clipping from the generating operation, not equality alone.",
      gold_risk="medium: boundary accumulation needs source inspection", evidence_kind="screening")
def v1(it):
    dom, T, out = _domain(it), T_of(it), []
    ld = record_streams(it)
    for s in LAB_STREAMS:
        rng = (dom.get(s) or {}).get("plausible_range")
        if not rng:
            continue
        vs = [(int(q["ts"]), _num(q["value"])) for q in ld.get(s) or () if int(q["ts"]) <= T]
        hit = [(d, v) for d, v in vs if v is not None and (abs(v - rng[0]) < 1e-9 or abs(v - rng[1]) < 1e-9)]
        if hit:
            out.append(f"{s} at bound {rng}: {hit[:4]} ({len(hit)}/{len(vs)})")
    return out


#: upper plausibility for a reported value [memory]: beyond these a reading is an assay or rendering error
PLAUSIBLE_HIGH = {"K": 8.5, "Na": 180, "HbA1c": 18.0, "fasting_glucose": 40.0, "LDL": 15.0, "triglycerides": 50.0,
                  "ALT": 5000, "AST": 5000, "TSH": 500, "Hb": 220, "creatinine": 2000, "Ca": 4.5}


@registry_cached("findings.yaml")
def _plaus_low() -> dict:
    d = load_registry("findings.yaml")["findings"]
    return {k: float(v["plausible_low"]) for k, v in d.items() if v.get("plausible_low") is not None}


@gate(id="V2", cls="②", packs=ALL, control="set a purchased B12 to 1 pg/mL",
      statement="Every readable or purchasable value lies within the population-plausible range "
                "(registry/findings.yaml plausible_low; PLAUSIBLE_HIGH [memory]).",
      gold_risk="low")
def v2(it):
    plausible_low = _plaus_low()
    out = []
    for x in labs(it):
        k = x.get("fid") or x["an"]
        if x["v"] is None:
            continue
        if k in plausible_low and x["v"] < plausible_low[k]:
            out.append(f"{k}={x['v']} < {plausible_low[k]} ({x['ch']}@{x['day']})")
        if k in PLAUSIBLE_HIGH and x["v"] > PLAUSIBLE_HIGH[k]:
            out.append(f"{k}={x['v']} > {PLAUSIBLE_HIGH[k]} ({x['ch']}@{x['day']})")
    return sorted(set(out))


@registry_cached("findings.yaml")
def _qual() -> set:
    d = load_registry("findings.yaml")["findings"]
    return {k for k, v in d.items() if v.get("qualitative")}


@gate(id="V3", cls="②", packs=("m2",), control="give a qualitative finding a numeric value",
      statement="A qualitative finding returns a category, never a number; a quantitative one returns a number.",
      gold_risk="low")
def v3(it):
    qualitative = _qual()
    out = []
    for x in labs(it):
        if x["ch"] != "buy" or not x.get("fid"):
            continue
        if x["fid"] in qualitative and x["v"] is not None:
            out.append(f"{x['fid']} qualitative returned {x['raw']}")
    return sorted(set(out))


def eag(a1c: float) -> float:
    """ADAG: estimated average glucose (mmol/L) = 1.59 x HbA1c - 2.59 (Nathan 2008, Diabetes Care 31:1473)."""
    return 1.59 * a1c - 2.59


def _distinct_readings(readings, analyte):
    """Same-day, same-value copies across display channels count once. Without a
    specimen identifier, a repeated display does not prove an independent draw.
    """
    return sorted({(x["day"], x["v"]) for x in readings if x["an"] == analyte and x["v"] is not None})


@gate(id="V4", cls="②", packs=ALL, control="set FBG to 10 mmol/L in the 90 days before an HbA1c of 5.0",
      statement="Screen for discordant HbA1c, sparse FPG and CGM summaries using the original thresholds. "
                "ADAG relates HbA1c to average glucose, not sparse fasting measurements; treatment transitions, "
                "sampling and assay interference require review. These flags do not establish impossibility.",
      gold_risk="medium: discordance needs review of timing, sampling and gold", evidence_kind="screening",
      source="ADA 2026 section 2; Nathan 2008; Battelino 2019 table 5")
def v4(it):
    from statistics import median
    L = [x for x in labs(it) if x["v"] is not None]
    a1c = [{"day": day, "v": value} for day, value in _distinct_readings(L, "HbA1c")]
    out = []
    for a in a1c:
        fpg = [value for day, value in _distinct_readings(L, "fasting_glucose") if a["day"] - 90 <= day <= a["day"]]
        if fpg:
            med = median(fpg)
            if (med >= 7.0 and a["v"] < 5.7) or (med < 5.6 and a["v"] >= 8.0):
                out.append(f"HbA1c {a['v']}@{a['day']} vs median FPG {med} ({len(fpg)})")
        night = [x["v"] for x in L if x["an"] == "cgm_night_min" and a["day"] - 14 <= x["day"] <= a["day"]]
        if len(night) >= 3 and min(night) > eag(a["v"]) + 1.7:
            out.append(f"HbA1c {a['v']}@{a['day']} (eAG {eag(a['v']):.1f}) vs CGM night min ≥ {min(night)}")
        tir = [x["v"] for x in L if x["an"] == "CGM_TIR" and abs(x["day"] - a["day"]) <= 30]
        if tir:
            t = sum(tir) / len(tir)
            pred = 7.0 + 0.08 * (70 - t)
            if abs(pred - a["v"]) > 2.0:
                out.append(f"HbA1c {a['v']}@{a['day']} vs TIR {t:.0f}% (≈{pred:.1f})")
    return sorted(set(out))


#: within-subject CV (registry/physio_streams.yaml observation cv): RCV = 1.96 x sqrt(2) x cv
CV = {"LDL": 0.083, "HbA1c": 0.018, "fasting_glucose": 0.056}


@gate(id="V5", cls="②", packs=ALL, control="raise LDL by 40% after an atorvastatin dose increase",
      statement="Screen for LDL increases after dose escalation and low glucose with low-hypoglycaemia-risk "
                "drugs. Total observed changes do not isolate drug effects; exposure timing, adherence, "
                "background disease and noise require review. Thresholds are screening thresholds.",
      gold_risk="medium: unexpected response needs causal and clinical review", evidence_kind="screening")
def v5(it):
    L = sorted((x for x in labs(it) if x["v"] is not None),
               key=lambda x: (x["day"], x.get("time") or ""))
    out = []
    tl = dose_timeline(it, "atorvastatin")
    for (d0, a), (d1, b) in zip(tl, tl[1:]):
        if b > a:
            before = [x for x in L if x["an"] == "LDL" and x["day"] <= d1]
            after = [x for x in L if x["an"] == "LDL" and d1 + 28 <= x["day"] <= d1 + 120]
            if before and after:
                v0 = before[-1]["v"]
                rcv = 1.96 * math.sqrt(2) * CV["LDL"]
                for x in after:
                    if math.log(x["v"] / v0) > rcv:
                        out.append(f"atorvastatin {a}->{b}@{d1}: LDL {v0}->{x['v']}@{x['day']}")
    low_by_drugs = {}
    for x in L:
        if x["an"] != "fasting_glucose" or x["v"] >= 3.9:
            continue
        # Exposure is the dated world record at the draw, not today's medication text.
        wd = {d for d, dose in world_drugs(it, x["day"]).items() if dose > 0}
        if wd & GLUCOSE_LOWERING and not (wd & HYPO_DRUGS):
            low_by_drugs.setdefault(tuple(sorted(wd & GLUCOSE_LOWERING)), []).append((x["v"], x["day"]))
    for drugs, low in low_by_drugs.items():
        out.append(f"FPG {low[:3]} on {list(drugs)}")
    return sorted(set(out))


#: hidden diagnoses that raise glucose by themselves (secondary diabetes; ADA 2025 §2 "other specific types",
#: Endocrine Society Cushing / acromegaly / pheochromocytoma guidance [memory]) -- matched in the gold's name
GLUCOSE_RAISING_DX = ("糖尿病", "血糖", "库欣", "皮质醇增多", "血色病", "肢端肥大", "嗜铬", "胰腺", "胰岛素", "LADA",
                      "胰高", "生长激素", "甲亢", "Graves", "多囊")


@gate(id="V6", cls="②", packs=ALL, control="set every HbA1c of a non-diabetic case to 7.0",
      statement="Screen diagnosis-marker discordance using the original thresholds. Missing T2D labels may "
                "represent undiagnosed disease; normal values with a T2D history may represent remission. "
                "Repeated displays of one reading are not separate confirmation samples.",
      gold_risk="medium: unexplained diagnostic-range readings or normal follow-up require review",
      evidence_kind="screening", source="ADA 2026 section 2; Riddle et al. 2021 remission consensus")
def v6(it):
    L = [x for x in labs(it) if x["v"] is not None]
    a1c = [x["v"] for x in L if x["an"] == "HbA1c"]
    fpg = [value for _, value in _distinct_readings(L, "fasting_glucose")]
    if not a1c and not fpg:
        return []
    t2d = "T2D" in world_conditions(it) | record_conditions(it)
    hidden = str(((it["case"].get("adjudication") or {}).get("ddx") or {}).get("diagnosis") or "")
    explained = any(w in hidden for w in GLUCOSE_RAISING_DX)
    out = []
    if t2d and all(v < 5.7 for v in a1c) and all(v < 5.6 for v in fpg) and not (visible_drugs(it) & GLUCOSE_LOWERING):
        out.append(f"T2D with HbA1c max {max(a1c, default=None)} FPG max {max(fpg, default=None)} and no visible drug")
    if not t2d and not explained and (any(v >= 6.5 for v in a1c) or sum(v >= 7.0 for v in fpg) >= 2):
        out.append(f"no T2D, HbA1c max {max(a1c, default=None)} FPG≥7 x{sum(v >= 7.0 for v in fpg)}")
    return out


@gate(id="V7", cls="②", packs=("m2",), control="give a woman's purchased Hb the male reference 130-175",
      statement="Sex-specific analytes (Hb) carry the patient's sex reference interval (WS/T 405-2012: men 130-175, "
                "women 115-150 g/L [memory]).",
      gold_risk="medium: a normal woman flagged anaemic")
def v7(it):
    out = []
    for x in labs(it):
        if x.get("fid") == "Hb" and x.get("ref_low") is not None:
            if sex(it) == "F" and float(x["ref_low"]) >= 130:
                out.append(f"F Hb ref {x['ref_low']}-{x['ref_high']}")
            if sex(it) == "M" and float(x["ref_low"]) < 125:
                out.append(f"M Hb ref {x['ref_low']}-{x['ref_high']}")
    return out


#: biological maximum change per week (haenv_kernel/latent.py:_DISEASE_SIGNAL_DOMAIN_BASE max_weekly_delta)
#: fasting glucose and triglycerides are left out: their day-to-day variation in patients is far above the
#: healthy-subject CV this noise allowance uses
MAX_WEEKLY = {"HbA1c": 0.2, "LDL": 0.3, "ALT": 12.0, "AST": 12.0, "FIB4": 0.2}
NOISE_CV = {"HbA1c": 0.018, "fasting_glucose": 0.056, "LDL": 0.083, "triglycerides": 0.222, "ALT": 0.141,
            "AST": 0.096, "FIB4": 0.14}


@gate(id="V8", cls="②", packs=ALL, control="raise an HbA1c by 1.5 points 8 days after the previous draw",
      statement="Screen changes exceeding the registered weekly model rate plus 2.77 x CV x value. This "
                "statistical scale is not a universal physiological limit; acute changes and noise tails need review.",
      gold_risk="high: suspicious dynamics may affect F3 interpretation", evidence_kind="screening")
def v8(it):
    out = []
    by = {}
    for x in labs(it):
        if x["an"] in MAX_WEEKLY and x["v"] is not None and x["ch"] in ("record", "followup", "buy"):
            by.setdefault(x["an"], {})[x["day"]] = x["v"]
    for an, pts in by.items():
        ds = sorted(pts)
        for a, b in zip(ds, ds[1:]):
            allow = MAX_WEEKLY[an] * max(b - a, 7) / 7 + 2.77 * NOISE_CV[an] * max(pts[a], pts[b])
            if abs(pts[b] - pts[a]) > allow:
                out.append(f"{an} {pts[a]}@{a} -> {pts[b]}@{b}")
    return out


# ③ coverage (pack scope) ---------------------------------------------------------------------------

def _p2_rule_floors(n_items: int = 100) -> dict:
    """Original 100-item minima; smaller packs use the production integer rule allocation.

    Registry loading remains cached/override-aware. Do not cache this size-dependent reader
    with the zero-argument registry decorator or infer a denominator from observed rule counts.
    """
    d = load_registry("pack2_triage_scenarios.yaml")
    out = {}
    for r in d.get("rules") or ():
        order = ["ed_now", "within_24h", "routine_followup", "watchful_waiting"]
        acc = [x for x in r.get("set") or () if x in order]
        floor = max(acc, key=order.index) if acc else None
        out[r["id"]] = 3 if floor == "within_24h" else 5
    if n_items < 100:
        from haenv.p2.audit import planned_cells
        counts = dict.fromkeys(out, 0)
        for tier, count in planned_cells(n_items).items():
            counts[d["tiers"][tier]["rule"]] += count
        # A printed rule absent even from the allocation still fails coverage; never waive it.
        return {rid: max(1, count) for rid, count in counts.items()}
    return out


def _counts(items, key) -> dict:
    c = {}
    for it in items:
        k = key(it)
        if k is not None:
            c[k] = c.get(k, 0) + 1
    return c


#: P3 / P4 / M2 gold branches (the values each gold field can take)
P3_CLASSES = ("uptitrate", "maintain", "downtitrate", "switch", "check_adherence_or_adverse_effect")
P4_CLASSES = ("true_change", "preanalytical", "method_difference", "analytic_biological_noise")
P4_NEXT = ("no_action", "act_on_change", "repeat_standardized")
URG = ("🔴", "🟠", "🟡", "🟢")
MED = ("investigate_first", "uptitrate", "keep", "downtitrate")


@gate(id="R1", cls="③", packs=("p2", "p3", "p4"), scope="pack", control="drop every item of one rule from the pack",
      statement="Every printed rule / decision class is exercised by enough published items (P2: 24-hour rules ≥ 3, "
                "other rules ≥ 5 per 100; smaller P2 packs use production integer quotas, minimum one per rule; "
                "P3 decisions and P4 classes ≥ 5).",
      gold_risk="none (coverage)")
def r1(items):
    if not items:
        return []
    pack = items[0]["pack"]
    if pack == "p2":
        floors = _p2_rule_floors(len(items))
        c = _counts(items, lambda it: gold(it).get("rule"))
        return [f"{r}: {c.get(r, 0)} < {f}" for r, f in floors.items() if c.get(r, 0) < f]
    if pack == "p3":
        c = _counts(items, lambda it: gold(it).get("class"))
        return [f"{k}: {c.get(k, 0)} < 5" for k in P3_CLASSES if c.get(k, 0) < 5]
    if pack == "p4":
        c = _counts(items, lambda it: gold(it).get("class"))
        return [f"{k}: {c.get(k, 0)} < 5" for k in P4_CLASSES if c.get(k, 0) < 5]
    return []


@gate(id="R2", cls="③", packs=("m2", "p3", "p4"), scope="pack", control="drop every 🔴 F1 item",
      statement="Every gold branch the score can award is exercised by at least one published item (M2 F1 urgency x4, "
                "F2 accepted directions x4, F3 within/exceeds; P3 decision x subtype; P4 class x next step).",
      gold_risk="none (coverage)")
def r2(items):
    if not items:
        return []
    pack, out = items[0]["pack"], []
    if pack == "m2":
        f1 = _counts([i for i in items if gold(i).get("frame") == "F1"], lambda it: gold(it).get("urgency"))
        out += [f"F1 urgency {u}: 0" for u in URG if not f1.get(u)]
        f2 = {}
        for i in items:
            if gold(i).get("frame") == "F2":
                for m in gold(i).get("med_accept") or [gold(i).get("med_direction")]:
                    f2[m] = f2.get(m, 0) + 1
        out += [f"F2 accept {m}: 0" for m in MED if not f2.get(m)]
        f3 = _counts([i for i in items if gold(i).get("frame") == "F3"],
                     lambda it: (gold(it).get("f3_plan") or {}).get("arm"))
        out += [f"F3 arm {a}: 0" for a in ("within", "exceeds") if not f3.get(a)]
    if pack in ("p3", "p4"):
        key = (lambda it: (gold(it).get("class"), gold(it).get("subtype"))) if pack == "p3" else \
              (lambda it: (gold(it).get("class"), gold(it).get("next_step")))
        c = _counts(items, key)
        out += [f"{k[0]} x {k[1]}: 0" for k in BRANCHES[pack] if not c.get(k)]
    return out


#: the gold branches the generator plans (P3 g3 class x control subtype, P4 class x next step), as emitted in
#: the 2026-10-10 N100 pools
BRANCHES = {
    "p3": [("maintain", "on"), ("maintain", "early"), ("uptitrate", "off"), ("downtitrate", "ae_dose"),
           ("downtitrate", "over"), ("switch", "top_off"), ("check_adherence_or_adverse_effect", "adh_low"),
           ("check_adherence_or_adverse_effect", "unattributed")],
    "p4": [("analytic_biological_noise", "no_action"), ("preanalytical", "no_action"),
           ("preanalytical", "repeat_standardized"), ("method_difference", "no_action"),
           ("method_difference", "repeat_standardized"), ("true_change", "no_action"),
           ("true_change", "act_on_change")],
}


def r2_table(items) -> dict:
    """The branch counts themselves (reported whether or not a branch is 0)."""
    if not items:
        return {}
    pack = items[0]["pack"]
    if pack == "m2":
        return {"F1_urgency": _counts([i for i in items if gold(i).get("frame") == "F1"], lambda it: gold(it).get("urgency")),
                "F2_direction": _counts([i for i in items if gold(i).get("frame") == "F2"], lambda it: gold(it).get("med_direction")),
                "F3_arm": _counts([i for i in items if gold(i).get("frame") == "F3"], lambda it: (gold(it).get("f3_plan") or {}).get("arm"))}
    if pack == "p2":
        return {"rule": _counts(items, lambda it: gold(it).get("rule"))}
    if pack == "p3":
        return {"class_subtype": {f"{k[0]}|{k[1]}": v for k, v in _counts(items, lambda it: (gold(it).get("class"), gold(it).get("subtype"))).items()}}
    if pack == "p4":
        return {"class_next": {f"{k[0]}|{k[1]}": v for k, v in _counts(items, lambda it: (gold(it).get("class"), gold(it).get("next_step"))).items()}}
    return {}


# ----------------------------------------------------------------------------------------- clone / evidence contracts

def benign_set(it) -> frozenset[str]:
    """Canonical benign phrases only; diagnostic signal is exempt from clone checks.

    Canonicalizing lay variants prevents punctuation/wording from hiding a copy.
    Unknown phrases are not invented into the benign vocabulary.
    """
    vocab = _benign_vocabulary()
    return frozenset(vocab[s] for e in it["sp"].get("evidence_ledger") or ()
                     if (s := str(e.get("symptom") or "")) in vocab)


@functools.lru_cache(maxsize=1)
def _benign_vocabulary():
    from haenv.world.observations.events import event_pools
    from haenv.world.observations.events import symptom_lay_every
    from haenv.world.overlay import condition_registry
    phrases = {e["text"]: e.get("topic", e["text"]) for e in event_pools()[0]}
    for spec in condition_registry(include_draft=True).values():
        if spec["join_gold"] == "independent":
            phrases.update({str(s[1]): str(s[1]) for s in spec["symptoms"]})
    vocab = dict(phrases)
    lay = symptom_lay_every()
    for phrase, canonical in phrases.items():
        vocab.update({s: canonical for s in lay.get(phrase) or ()})
    return vocab


def skeleton_key(it) -> str:
    """Facts fingerprint; ids, symptom dates and gold diagnosis are not patient facts."""
    lp = it["case"].get("latent_premise") or {}
    pb, meta = lp.get("patient_basics") or {}, lp.get("meta") or {}
    return json.dumps({"basics": pb, "start": meta.get("start"), "nadir": meta.get("nadir"),
                       "benign": sorted(benign_set(it))}, sort_keys=True, ensure_ascii=False)


@gate(id="diversity", cls="④", packs=ALL, scope="pack",
      statement="One independently sampled patient per item; no copied complete benign complaint bundle (Jaccard >0.9, >=4 distinct complaints).",
      source="design choice: four common phrases together are a bundle; individual common phrases may recur",
      control="clone a patient's facts, or copy four or more benign complaints into another patient")
def diversity(items):
    from collections import Counter
    count = Counter(skeleton_key(it) for it in items)
    bad = [f"skeleton occurs {v} times (maximum 1)" for v in count.values() if v > 1]
    # Legacy patient IDs identify copied seed patients in P2-P4. M2's
    # diagnosis scaffold is explicitly a spec, governed by its spec cap;
    # its patient facts still obey the unique-fingerprint rule above.
    if items and items[0]["pack"] in ("p2", "p3", "p4"):
        bases = Counter(re.sub(r"v\d+$", "", it["case_id"]) for it in items)
        bad += [f"seed patient {cid} occurs {v} times (maximum 1)" for cid, v in bases.items() if v > 1]
    sets = [(it["case_id"], benign_set(it)) for it in items]
    for i, (cid, a) in enumerate(sets):
        for other, b in sets[i + 1:]:
            if min(len(a), len(b)) >= 4:
                j = len(a & b) / len(a | b)
                if j > 0.9:
                    bad.append(f"{cid}/{other}: benign Jaccard {j:.3f} >0.9")
    return bad


@gate(id="evidence_location", cls="①", packs=ALL,
      statement="Non-acting factors have dated retrievable source records; value fields contain no inline exemption.",
      control="delete a laboratory/encounter/device record, or restore the report's self-explaining parenthesis")
def evidence_location(it):
    b, pack = block(it), it["pack"]
    if pack == "m2":
        b = it["sp"].get("prediction_context") or {}
    from haenv.interaction.query.adapter import ADAPTERS
    from haenv.interaction.query.adapter import load_pack_adapters
    load_pack_adapters()
    required = {"p2": ("device_record",), "p3": ("encounter_record",),
                "p4": ("laboratory_method_record", "specimen_procedure"),
                "m2": ("laboratory_method_record",)}[pack]
    bad = [f"missing retrievable {s}" for s in required if s not in b or s not in ADAPTERS[pack].sections]
    pc = it["sp"].get("prediction_context") or {}
    if "event_history_record" not in pc or "event_history_record" not in ADAPTERS[pack].sections:
        bad.append("missing retrievable event history")
    if any(e.get("symptom") and e.get("context") for e in it["sp"].get("evidence_ledger") or ()):
        bad.append("inline symptom history")
    if pack == "p4":
        fields = [{k: r.get(k) for k in ("lab", "method", "specimen", "note")} for r in b.get("report") or ()]
        forbidden = ("同型号", "同厂家", "比对已通过", "不受影响", "不影响结果")
        text = json.dumps(fields, ensure_ascii=False)
        bad += [f"inline explanation:{w}" for w in forbidden if w in text]
        if "specimen_policy" in b:
            bad.append("inline specimen policy")
        methods = b.get("laboratory_method_record") or []
        for r in b.get("report") or []:
            applicable = [e for e in methods if int(e.get("valid_from", 10**9)) <= int(r["day"])]
            if not applicable or (applicable[-1].get("method"), applicable[-1].get("lab")) != (r["method"], r["lab"]):
                bad.append(f"report@{r['day']}: no applicable method record")
    elif pack == "p3":
        histories = {(e.get("symptom"), e.get("onset_day")) for e in b.get("encounter_record") or ()
                     if "history" in e and "day" in e}
        for e in b.get("symptom_diary") or ():
            if (e.get("symptom"), e.get("onset_day")) not in histories:
                bad.append("diary symptom has no encounter history")
            if "context" in e:
                bad.append("inline diary context")
    elif pack == "p2":
        if any(r.get("fit_note") for r in b.get("wearable_daily") or ()):
            bad.append("inline wearable fit note")
        plan = gold(it).get("plan") or {}
        if plan.get("artefact") == "strap" or gold(it).get("tier") == "ben_spo2":
            from haenv.p2.background import strap_notes
            recorded = {e.get("day") for record in b.get("device_record") or () for e in record.get("entries") or ()
                        if e.get("text") in strap_notes(plan)}
            if not recorded:
                bad.append("strap artefact has no retrievable wear record")
    else:
        for result in (it.get("buy") or {}).get("results", {}).values():
            for r in result.get("observation") or ():
                if isinstance(r, dict) and "不适用" in str(r.get("method") or ""):
                    bad.append("inline laboratory method exemption")
    return bad


# ----------------------------------------------------------------------------------------- run

#: Gates that refuse a pack: `tools/generation/make_pack.py` fails and writes no publishable manifest when one reports a
#: violation on the pool or the published items. This is the approved fixed list, not automatic promotion
#: of zero-count gates. Screening checks remain report-only even when their counts reach zero.
@gate(id="history_timing", cls="①", packs=ALL,
      statement="Every history-dependent decisive record precedes the recent boundary by at least the registered gap and is retrievable.",
      source="registry/history_dependency.yaml", control="move a decisive record into the recent window, or delete its visible source")
def history_timing(it):
    from haenv.world.history.history_dependency import metadata
    from haenv.world.history.history_dependency import table
    m = metadata(it["case"], it["pack"])
    if not m.get("active"):
        return []
    bad = []
    gap = int(table()["timing"]["min_separation_days"]["value"])
    for ref in m.get("records") or ():
        if int(ref["day"]) > int(m["recent_lower"]) - gap:
            bad.append("decisive old record is inside the forbidden date margin")
        b = block(it) if it["pack"] != "m2" else it["sp"].get("prediction_context") or {}
        if not any(int(x.get("day", x.get("draw_day", 10**9))) == int(ref["day"]) for x in b.get(ref["section"]) or ()):
            bad.append("decisive dated source missing from solver records")
    return bad or ([] if m.get("records") else ["no decisive history reference"])


@gate(id="history_twin", cls="①", packs=ALL,
      statement="Deleting each decisive old source changes gold; a registered treatment baseline can become indeterminate; ordinary cases keep gold after deleting every earlier date individually and together.",
      source="registry/history_dependency.yaml", control="mark an inert record active, or hide an acting record in an ordinary case")
def history_twin(it):
    from haenv.world.history.history_dependency import metadata
    from haenv.world.history.history_dependency import policy
    from haenv.world.history.history_dependency import remove_early
    from haenv.world.history.history_dependency import early_days
    m = metadata(it["case"], it["pack"])
    if not m:
        return []
    P, raw, lo = policy(it["pack"]), it["case"], int(m["recent_lower"])
    full = P.answer(raw)
    if full is None:
        return ["full-history gold is not derivable"]
    bad = []
    if m.get("active"):
        for ref in m["records"]:
            alternate = P.answer(P.twin(raw, int(ref["day"])))
            baseline = (it["pack"] == "p4" and gold(it).get("class") == "true_change"
                        and m.get("deletion") == "indeterminate"
                        and ref["section"] == "report" and int(ref["day"]) == int(gold(it)["t0"]))
            if alternate is None and not baseline:
                bad.append("deletion twin has no derivable alternative gold")
            if alternate == full:
                bad.append("deleting decisive source leaves gold unchanged")
    else:
        for day in [None, *early_days(raw, lo)]:
            try:
                alternate = P.answer(remove_early(raw, lo, day))
            except (ValueError, KeyError, IndexError):
                alternate = None
            if alternate != full:
                bad.append("ordinary gold changed after deleting " + str(day))
                break
    return bad


@gate(id="history_source_distribution", cls="④", packs=ALL, scope="pack",
      statement="Acting and inactive archived sources share record type and date-age distribution.",
      source="registry/history_dependency.yaml", control="use an exclusive source type or segregate acting dates from inactive dates")
def history_source_distribution(items):
    from collections import Counter
    from haenv.world.history.history_dependency import metadata
    from haenv.world.history.history_dependency import table
    groups = {True: [], False: []}
    for it in items:
        m = metadata(it["case"], it["pack"])
        if m:
            groups[bool(m["active"])].extend((r["section"], int(m["recent_lower"]) - int(r["day"])) for r in m["records"])
    a, b = groups[True], groups[False]
    if not a or not b:
        return []
    ca, cb = Counter(x[0] for x in a), Counter(x[0] for x in b)
    tv = sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb)) / 2
    aa, bb = [x[1] for x in a], [x[1] for x in b]
    ks = max(abs(sum(x <= d for x in aa) / len(aa) - sum(x <= d for x in bb) / len(bb)) for d in set(aa + bb))
    cfg = table()["distribution_gate"]
    # DKW sampling allowance, fixed before measuring the pools. Tiny clinical
    # subgroups cannot establish a population timing shift from one observation.
    allowance = math.sqrt(math.log(200) / (2 * len(a))) + math.sqrt(math.log(200) / (2 * len(b)))
    bad = [f"source type TV {tv:.3f}"] if tv > cfg["type_tv_max"]["value"] else []
    if ks > cfg["ks_max"]["value"] + allowance:
        bad.append(f"date age KS {ks:.3f}")
    return bad


@gate(id="look_alike_source_distribution", cls="④", packs=ALL, scope="pack",
      statement="Acting and inactive look-alike records have identical registered source distributions.",
      source="registry/difficulty_r2.yaml", control="put acting records in an exclusive source section")
def look_alike_source_distribution(items):
    from collections import Counter
    groups = {True: Counter(), False: Counter()}
    for it in items:
        g = gold(it) or (it["case"].get("adjudication") or {})
        look = g.get("look_alike") or (g.get("history_dependency") or {}).get("look_alike") or {}
        for ref in look.get("records") or ():
            groups[bool(ref["acts"])][ref["section"]] += 1
    a, b = groups[True], groups[False]
    if not a or not b:
        return []
    tv = sum(abs(a[k] / sum(a.values()) - b[k] / sum(b.values())) for k in set(a) | set(b)) / 2
    return [f"look-alike source TV {tv:.6f}"] if tv else []




@gate(id="p4_magnitude_overlap", cls="④", packs=("p4",), scope="pack",
      statement="Header-amplitude-only effect among true change, method difference and preanalytical causes is below the registered shortcut threshold.",
      source="docs/design/p4-visible-change.md", control="assign separated amplitude ranges to non-noise causes")
def p4_magnitude_overlap(items):
    if not any(gold(it).get("look_alike") for it in items):
        return []
    from haenv.p4.magnitude import readout
    # RCV separates noise from significant change by design; audit whether
    # magnitude alone identifies the cause after that clinical first step.
    r = readout([it for it in items if gold(it).get("class") in
                 ("true_change", "method_difference", "preanalytical")])
    bad = []
    if r["chance_corrected"] >= 0.20:
        bad.append(f"header magnitude-only chance-corrected effect {r['chance_corrected']:.6f}")
    return bad




@gate(id="p3_symptom_status", cls="④", packs=("p3",), scope="pack",
      statement="A stub reading only recorded symptom status has chance-corrected effect below 0.20.",
      source="registry/difficulty_r2.yaml", control="assign symptom status patterns by the answer class")
def p3_symptom_status(items):
    from haenv.world.validation.difficulty_r2 import enabled
    if not items or not enabled(gold(items[0]).get("plan")):
        return []
    from haenv.p3.audit import symptom_status_readout
    reading = symptom_status_readout(items)
    return [reading] if reading["chance_corrected"] >= 0.20 else []



@gate(id="M2_f2", cls="①", packs=("m2",), control="remove keep from an unrelated urgent line or erase adverse-reduction history and restore uptitration",
      statement="Named-drug directions respect dated adverse-effect reductions, dose feasibility and hidden-line relevance.",
      source="registry/m2_pre_review.yaml:drug_targets,adverse_reduction", gold_risk="high: incorrect named-drug direction")
def m2_f2(it):
    g = gold(it)
    f2 = g.get("f2_reading") or {}
    if g.get("frame") != "F2":
        return []
    from haenv.m2.core import f2_accept
    from haenv.m2.core import f2_dose_state
    from types import SimpleNamespace
    raw = SimpleNamespace(**it["case"])
    pb = (getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {}
    primary = ((getattr(raw, "latent_premise", None) or {}).get("world_medication") or {}).get("primary") or {}
    drug = primary.get("drug")
    out = []
    reading = dict(f2)
    if drug and (f2.get("dose_state") or drug == "atorvastatin"):
        disease = pb.get("disease") or primary.get("indication")
        state = f2_dose_state(raw, disease, drug, T_of(it))
        if f2.get("dose_state") != state:
            out.append("F2 dose/adverse history disagrees with the dated medication record")
        reading["dose_state"] = state
    if f2.get("hidden_linked") is False and not f2.get("hidden_intake_loss"):
        expected = f2_accept(False, None, reading)
    elif (reading.get("dose_state") or {}).get("adverse_reduction"):
        ddx = (it["case"].get("adjudication") or {}).get("ddx") or {}
        expected = f2_accept(g.get("hidden_named_condition", True), ddx.get("urgency"), reading)
    else:
        return out
    if g.get("med_accept") != expected or g.get("med_direction") != expected[0]:
        out.append(f"named-drug direction: expected {expected}, got {g.get('med_accept')}")
    return out


@gate(id="M2_workup", cls="①", packs=("m2",), control="lower the purchase budget below the minimum required cost",
      statement="The sourced initial required set fits the actual menu budget with registered margin; post-diagnosis tests and duplicate alternatives are not mandatory.",
      source="registry/m2_pre_review.yaml", gold_risk="high: required workup cannot be purchased")
def m2_workup(it):
    from haenv.m2 import core as C
    adj = it["case"].get("adjudication") or {}
    ddx = adj.get("ddx") or {}
    tests = ddx.get("tests") or []
    plan = adj.get("m2_workup") or {}
    want = [] if ddx.get("insufficient") else C.workup_tests(ddx["spec_id"], plan.get("optional") or ())
    out = [] if tests == want and plan.get("required") == want else ["mandatory tests disagree with sourced initial stage"]
    alternatives = {t: C.workup_alternatives(t) for t in want if C.workup_alternatives(t)}
    if plan.get("alternatives", {}) != alternatives:
        out.append("registered alternative groups disagree with sourced initial stage")
    menu = it["buy"]["menu"]
    try:
        prices = {i["target"]: float(i["cost"]) for i in menu if i.get("is_test")}
        for alt in C.pre_review_rules()["alternatives"]:
            chosen = set(want) & set(alt["tests"])
            if chosen and min(prices[t] for t in chosen) != min(prices[t] for t in alt["tests"]):
                out.append("initial alternative is not a minimum-cost menu choice")
        expected = C.workup_budget(want, menu)
        cost = C.workup_cost(want, menu)
        if plan.get("minimum_cost") != cost:
            out.append(f"registered minimum cost {plan.get('minimum_cost')} != {cost}")
        if float(plan.get("budget", 0)) != expected:
            out.append(f"budget {plan.get('budget')} != required-set budget {expected}")
        visible = (it["buy"].get("visible_menu") or {}).get("budget")
        if visible != plan.get("budget") or visible is None or visible < cost:
            out.append(f"visible budget {visible} != registered {plan.get('budget')} or below minimum cost {cost}")
        if it["buy"].get("rebuilt_budget") != visible:
            out.append("rebuilt-menu budget disagrees with actual default get_menu")
    except (KeyError, ValueError, TypeError) as e:
        out.append(str(e))
    return out


@gate(id="M2_menu", cls="①", packs=("m2",), control="append a nonexistent compound test name",
      statement="Each M2 gold test is an exact, returned menu item.", gold_risk="high: nonexistent or unavailable gold test")
def m2_menu(it):
    names = {i["target"] for i in it["buy"]["menu"] if i.get("is_test")}
    ddx = (it["case"].get("adjudication") or {}).get("ddx") or {}
    out = [f"gold test absent from menu: {t}" for t in ddx.get("tests") or [] if t not in names]
    for t in ddx.get("tests") or []:
        result = (it["buy"].get("results") or {}).get(t)
        if result is not None and result.get("outcome") != "returned":
            out.append(f"gold test unavailable: {t}")
    return out


@gate(id="M2_imaging", cls="①", packs=("m2",), control="return off-scope CTPA/abdominal evidence, short ECG from Holter, or serology from elastography",
      statement="CTPA, abdominal imaging, ambulatory ECG and elastography return only their named acquisition scope.",
      source="registry/m2_pre_review.yaml:imaging", gold_risk="high: purchase gives evidence outside its stated scope")
def m2_imaging(it):
    out = []
    for target, result in (it["buy"].get("results") or {}).items():
        for p in result.get("observation") or []:
            if target == "CT肺动脉造影(CTPA)" and p.get("finding_id") != "CTPA":
                out.append(f"CTPA returned {p.get('finding_id')}")
            if target in ("腹部CT/MRI", "胰腺增强CT或MRI/MRCP", "必要时胰腺CT/MRI") and (
                    "分侧取血" in str(p) or re.search(r"\bAVS\b|adrenal[\s_-]*vein[\s_-]*sampling", str(p), re.I)):
                out.append(f"{target} returned adrenal vein sampling")
            if target == "动态心电图" and ((p.get("details") or {}).get("acquisition") == "12_lead_ecg"
                                           or "十二导联" in str(p) or "12_lead_ecg" in str(p)):
                out.append("ambulatory ECG returned a short 12-lead acquisition")
            if target == "肝脏弹性成像(VCTE/必要时MRE)" and (
                    any(k in str(p) for k in ("etiology_review", "alcohol_g_per_day", "HBsAg", "抗HCV",
                                              "hepatitis_b_surface_antigen", "hepatitis_c_antibody", "不饮酒",
                                              "mri_pdff_percent"))):
                out.append("elastography returned serology/interview or other MRI acquisition")
    return out


@gate(id="M2_urgency", cls="①", packs=("m2",), scope="pack", control="raise the same LADA line and beta-hydroxybutyrate reading to red without a critical value",
      statement="A component with identical key readings has the same urgency when it determines case urgency; companion-line and critical-value escalations are explicit.",
      source="registry/composition_comorbid.yaml; registry/condition_findings.yaml", gold_risk="high: urgency depends on case composition rather than current severity")
def m2_urgency(items):
    from haenv.m2 import core as C
    from haenv.world.overlay import condition_registry
    from haenv.world.overlay import HAENV_COMORBID_PAIRS
    from haenv.world.overlay import URGENCY_ORDER
    from haenv.world.clinical.ddx import URGENCY_NEEDS
    reg = condition_registry()
    groups, out = {}, []
    for it in items:
        ddx = (it["case"].get("adjudication") or {}).get("ddx") or {}
        if ddx.get("insufficient"):
            continue
        sid = ddx["spec_id"]
        comps = C.components(sid)
        cfg = HAENV_COMORBID_PAIRS.get(sid) or {}
        readings = {x["an"]: x["v"] for x in labs(it) if x["ch"] == "buy" and x["day"] == T_of(it)}
        if readings.get("K") is not None and readings["K"] < 2.5:
            continue  # C7 checks the independent critical-value escalation
        ceiling = max([reg[c]["urgency"] for c in comps] + ([cfg["urgency"]] if cfg.get("urgency") else []), key=URGENCY_ORDER.index)
        if URGENCY_ORDER.index(ddx["urgency"]) > URGENCY_ORDER.index(ceiling):
            out.append(f"{it['case_id']}: urgency exceeds component/interaction ceiling {ceiling}")
        for comp in comps:
            if comp in URGENCY_NEEDS or any(c in URGENCY_NEEDS for c in comps) or cfg.get("urgency"):
                continue  # those tiers also depend on expressed symptoms / registered interactions
            if reg[comp]["urgency"] != ceiling:
                continue  # a more urgent companion owns total urgency
            keys = C.confirmatory_all(comp)
            signature = tuple((k, readings.get(FINDING_STREAM.get(k, k))) for k in keys)
            key = (comp, signature)
            previous = groups.setdefault(key, (ddx["urgency"], it["case_id"]))
            if previous[0] != ddx["urgency"]:
                out.append(f"{comp} same key readings: {previous[1]}={previous[0]}, {it['case_id']}={ddx['urgency']}")
    return out


@gate(id="protocol_distribution", cls="④", packs=("p2", "p3"), scope="pack",
      statement="Bearing and unrelated procedures have comparable length and step distributions and identical format.",
      source="registry/protocols.yaml", control="shorten only the bearing procedures, change their schema, or make them one-step entries")
def protocol_distribution(items):
    from haenv.world.protocols.gates import distribution_hits
    return distribution_hits() if any(block(it).get("rules_delivery") == "library" for it in items) else []


@gate(id="system_score_receipt", cls="①", packs=("p2",),
      statement="The automatic score is the registered fallible receipt; gold is recomputable from the completed observations.",
      source="registry/system_scoring.yaml", control="change an automatic total or scale, delete the receipt, or change a completed observation")
def system_score_receipt(it):
    if block(it).get("rules_delivery") != "library":
        return []
    from haenv.world.protocols.gates import receipt_hits
    from haenv.p2.rules import assess
    bad = receipt_hits(block(it), it["case_id"])
    if set(assess(block(it))["set"]) != set(gold(it)["set"]):
        bad.append("gold changed after attaching protocol library or automatic receipt")
    return bad


@gate(id="system_score_rates", cls="④", packs=("p2",), scope="pack",
      statement="Automatic scale and missing-input error shares agree with the registered populations and whole-pack cap.",
      source="registry/system_scoring.yaml", control="make every automatic receipt erroneous")
def system_score_rates(items):
    from haenv.world.protocols.gates import rates_hits
    return rates_hits(items)





FAIL_CLOSED = ('C1', 'C3', 'C4', 'C7', 'C11', 'V2', 'V3', 'V7', 'R1', 'R2', 'diversity', 'evidence_location', 'history_timing', 'history_twin', 'history_source_distribution', 'look_alike_source_distribution', 'p4_magnitude_overlap', 'p3_symptom_status', 'M2_f2', 'M2_workup', 'M2_menu', 'M2_imaging', 'M2_urgency', 'protocol_distribution', 'system_score_receipt', 'system_score_rates')


def refusals(rep: dict) -> list[str]:
    """The fail-closed gates of a `run` report that found violations."""
    return [f"{gid}: pool {r['pool']}, pack {r['pack']}" for gid, r in rep.items()
            if gid in FAIL_CLOSED and (r["pool"] or r["pack"])]


def run(items: list[dict], pack: str, gates: list[Gate] | None = None, published: set[str] | None = None,
        jobs: int = 1) -> dict:
    """{gate id: {pool, pack, ids_pool, ids_pack, examples}} on one pack's pool (and its published subset)."""
    rep = {}
    pub = [it for it in items if published is None or it["case_id"] in published]
    selected = [g for g in gates or GATES if pack in g.packs]
    item_gates = [g for g in selected if g.scope != "pack"]
    def check_item(it):
        out = {}
        for g in item_gates:
            try:
                v = g.check(it)
            except Exception as e:
                v = [f"gate error: {type(e).__name__}: {e}"]
            if v:
                out[g.id] = v
        return it["case_id"], out
    if jobs > 1:
        from haenv.world.generation.build_pool import map_ordered
        checked = map_ordered(check_item, items, workers=jobs)
    else:
        checked = list(map(check_item, items))
    for g in selected:
        if pack not in g.packs:
            continue
        if g.scope == "pack":
            vp, vk = g.check(items), g.check(pub)
            rep[g.id] = {"cls": g.cls, "statement": g.statement, "scope": "pack", "pool": len(vp), "pack": len(vk),
                         "pool_detail": vp, "pack_detail": vk, "gold_risk": g.gold_risk,
                         "evidence_kind": g.evidence_kind}
            continue
        hits = {cid: result[g.id] for cid, result in checked if g.id in result}
        ids_pack = sorted(c for c in hits if published is None or c in published)
        rep[g.id] = {"cls": g.cls, "statement": g.statement, "scope": "item", "pool": len(hits), "pack": len(ids_pack),
                     "n_pool": len(items), "n_pack": len(pub), "gold_risk": g.gold_risk,
                     "evidence_kind": g.evidence_kind,
                     "examples": [{"case_id": c, "violations": hits[c][:3]} for c in (ids_pack or sorted(hits))[:5]],
                     "ids_pool": sorted(hits), "ids_pack": ids_pack}
    return rep
