"""The rl record as the solver reads it: `render_text(block)` turns `prediction_context.redline_rl` into readable
sections; `parse_text(text)` reads the same sections back (the re-derivation gate G3 reads only this text).

Sections: Patient, Problem list, Past history (families A/B/G), Allergies, Medications (with record),
Dispensing, Visits, Results, Home monitoring, Home glucose log (when kept), Messages, Earlier conversations
with this assistant (family F), Planned. The user's question is not part of the record; the prompt places it after.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from datetime import date

H = {"patient": "Patient", "problems": "Problem list", "past": "Past history", "allergies": "Allergies",
     "meds": "Medications", "record": "Medication record", "dispensing": "Dispensing", "visits": "Visits",
     "results": "Results", "home": "Home monitoring", "glucose": "Home glucose log", "messages": "Messages",
     "conversation": "Earlier conversations with this assistant", "planned": "Planned"}
#: the same sections of a zh record (`block["lang"] == "zh"`)
H_ZH = {"patient": "患者", "problems": "问题列表", "past": "既往史", "allergies": "过敏史", "meds": "当前用药",
        "record": "用药记录", "dispensing": "配药记录", "visits": "就诊记录", "results": "检验结果",
        "home": "家庭监测", "glucose": "家庭血糖记录", "messages": "患者留言", "conversation": "与本助手的既往对话",
        "planned": "计划事项"}
#: words of a zh record's data lines (the parser reads both languages)
W = {"en": {"ref": "ref", "collected": ", collected ", "hr": "resting HR {} bpm", "bp": "BP {}/{} mmHg",
            "weight": "weight {} kg", "days": "{} days", "sep": "; ", "breakfast": "before breakfast",
            "user": "User: ", "assistant": "Assistant: "},
     "zh": {"ref": "参考", "collected": "，采样 ", "hr": "静息心率 {} 次/分", "bp": "血压 {}/{} mmHg",
            "weight": "体重 {} kg", "days": "{} 天", "sep": "；", "breakfast": "早餐前",
            "user": "用户：", "assistant": "助手："}}

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def weekday_name(day: date, lang: str) -> str:
    return "周" + "一二三四五六日"[day.weekday()] if lang == "zh" else _WEEKDAYS[day.weekday()]


def _series_line(name, unit, ref, rows, w=None) -> str:
    w = w or W["en"]
    head = f"{name} ({unit}; {w['ref']} {ref})" if ref else f"{name} ({unit})"
    return head + ": " + " | ".join(f"{r['date']}: {r['value']}" + (f" {r['flag']}" if r.get("flag") else "")
                                   for r in rows)


def render_text(block: dict) -> str:
    zh = block.get("lang") == "zh"
    H_, w = (H_ZH, W["zh"]) if zh else (H, W["en"])
    return _render(block, H_, w)


def _render(block: dict, H: dict, w: dict) -> str:
    out = [f"# {block.get('record_type', 'Patient record')}", ""]
    out += [f"## {H['patient']}", block.get("patient", ""), ""]
    out += [f"## {H['problems']}"] + [f"- {p}" for p in block.get("problem_list") or ()] + [""]
    if block.get("past_history"):
        out += [f"## {H['past']}"] + [f"- {p}" for p in block["past_history"]] + [""]
    out += [f"## {H['allergies']}"] + [f"- {a}" for a in block.get("allergies") or ()] + [""]
    out += [f"## {H['meds']}"] + [f"- {m}" for m in block.get("medications") or ()] + [""]
    if block.get("medication_record"):
        out += [f"## {H['record']}"]
        out += [f"- {r['date']} {r['drug']}: {r['event']}" + (f" ({r['reason']})" if r.get("reason") else "")
                for r in block["medication_record"]] + [""]
    out += [f"## {H['dispensing']}"] + [f"- {d}" for d in block.get("dispensing") or ()] + [""]
    out += [f"## {H['visits']}"]
    for v in block.get("visits") or ():
        out.append(f"### {v['date']} | {v['type']} | {v['clinician']}")
        out.append(v["summary"])
    out.append("")
    out += [f"## {H['results']}"]
    if block.get("results_note"):
        out.append(block["results_note"])
    series = OrderedDict()
    singles = []
    for r in block.get("results") or ():
        if "value" in r:
            series.setdefault((r["analyte"], r.get("unit", ""), r.get("ref", "")), []).append(r)
        else:
            singles.append(r)
    for (name, unit, ref), rows in series.items():
        out.append(_series_line(name, unit, ref, sorted(rows, key=lambda x: x["date"]), w))
    for r in sorted(singles, key=lambda x: (x["date"], x["analyte"])):
        when = r["date"] + (f"{w['collected']}{r['time']}" if r.get("time") else "")
        out.append(f"{r['analyte']} ({when}): {r['result']}")
    out.append("")
    if block.get("home_monitoring"):
        out += [f"## {H['home']}"]
        for h in block["home_monitoring"]:
            bits = []
            if h.get("resting_hr") is not None:
                bits.append(w["hr"].format(h["resting_hr"]))
            if h.get("sbp") is not None and h.get("dbp") is not None:
                bits.append(w["bp"].format(h["sbp"], h["dbp"]))
            if h.get("weight_kg") is not None:
                bits.append(w["weight"].format(h["weight_kg"]))
            bits.append(w["days"].format(h["days"]))
            out.append(f"- {h['month']}: " + w["sep"].join(bits))
        out.append("")
    if block.get("glucose_log"):
        out += [f"## {H['glucose']}"]
        if block.get("glucose_note"):
            out.append(block["glucose_note"])
        out += [f"- {g['date']} {w['breakfast']}: {g['value']} {g['unit']}" for g in block["glucose_log"]] + [""]
    out += [f"## {H['messages']}"] + [f"- {m['date']} ({m['from']}): {m['text']}" for m in block.get("messages") or ()]
    if block.get("conversation"):
        out += ["", f"## {H['conversation']}"]
        for c in block["conversation"]:
            out += [f"### {c['date']}", f"{w['user']}{c['user']}", f"{w['assistant']}{c['assistant']}"]
    out += ["", f"## {H['planned']}"] + [f"- {p}" for p in block.get("planned") or ()]
    return "\n".join(out).rstrip() + "\n"


_SERIES = re.compile(r"^(?P<name>.+?) \((?P<unit>[^;()]+?)(?:; (?:ref|参考) (?P<ref>[^)]*))?\): (?P<rest>\d{4}-\d{2}-\d{2}: .*)$")
_SINGLE = re.compile(r"^(?P<name>.+?) \((?P<date>\d{4}-\d{2}-\d{2})(?:(?:, collected |，采样 )(?P<time>\d{2}:\d{2}))?\): "
                     r"(?P<result>.*)$")
_HM = re.compile(r"^- (?P<month>\d{4}-\d{2})[^:]*: (?P<rest>.*)$")
_GL = re.compile(r"^- (?P<date>\d{4}-\d{2}-\d{2}) (?:before breakfast|早餐前): (?P<value>[0-9.]+) (?P<unit>\S+)$")
_CONV = re.compile(r"^### (?P<date>\d{4}-\d{2}-\d{2})$")
_VISIT = re.compile(r"^### (?P<date>\d{4}-\d{2}-\d{2}) \| (?P<type>.*?) \| (?P<clinician>.*)$")


def parse_text(text: str) -> dict:
    """The sections of a rendered record, as the facts reader (`realize.facts_from_rows`) takes them."""
    sec, out = None, {"patient": "", "problem_list": [], "medications": [], "planned": [], "results": [], "home_monitoring": [],
                      "today": None, "allergies": [], "past_history": [], "dispensing": [], "visits": [],
                      "glucose_log": [], "conversation": []}
    names = {v: k for k, v in H.items()} | {v: k for k, v in H_ZH.items()}
    for line in text.splitlines():
        if line.startswith("## "):
            sec = names.get(line[3:].strip())
            continue
        if not line.strip() or line.startswith("# "):
            continue
        if sec == "patient":
            out["patient"] += line
        elif sec in ("problems", "meds", "planned", "allergies", "past", "dispensing") and line.startswith("- "):
            key = {"problems": "problem_list", "meds": "medications", "past": "past_history"}.get(sec, sec)
            out[key].append(line[2:])
        elif sec == "visits":
            m = _VISIT.match(line)
            if m:
                out["visits"].append({**m.groupdict(), "summary": ""})
            elif out["visits"]:
                out["visits"][-1]["summary"] += line
        elif sec == "conversation":
            m = _CONV.match(line)
            if m:
                out["conversation"].append({"date": m.group("date"), "user": "", "assistant": ""})
            elif out["conversation"] and line.startswith(("User: ", "用户：")):
                out["conversation"][-1]["user"] = line.split(": ", 1)[1] if line.startswith("User: ") else line[3:]
            elif out["conversation"] and line.startswith(("Assistant: ", "助手：")):
                out["conversation"][-1]["assistant"] = (line.split(": ", 1)[1] if line.startswith("Assistant: ")
                                                        else line[3:])
        elif sec == "glucose":
            m = _GL.match(line)
            if m:
                out["glucose_log"].append(m.groupdict())
        elif sec == "results":
            m = _SERIES.match(line)
            if m:
                for part in m.group("rest").split(" | "):
                    d, v = part.split(": ", 1)
                    vv = v.split()
                    out["results"].append({"date": d, "analyte": m.group("name"), "value": vv[0],
                                           "unit": m.group("unit"), "ref": m.group("ref") or "",
                                           "flag": vv[1] if len(vv) > 1 else ""})
                continue
            m = _SINGLE.match(line)
            if m:
                out["results"].append({"date": m.group("date"), "analyte": m.group("name"),
                                       "result": m.group("result"),
                                       **({"time": m.group("time")} if m.group("time") else {})})
        elif sec == "home":
            m = _HM.match(line)
            if m:
                hr = re.search(r"resting HR (\d+) bpm|静息心率 (\d+) 次/分", m.group("rest"))
                out["home_monitoring"].append({"month": m.group("month"),
                                               "resting_hr": int(hr.group(1) or hr.group(2)) if hr else None})
    pm = re.match(r"(female|male|女|男)(?:, |，)(\d+) (?:years|岁)", out["patient"])
    out["sex"] = {"female": "F", "male": "M", "女": "F", "男": "M"}.get(pm.group(1)) if pm else None
    out["age"] = int(pm.group(2)) if pm else None
    wm = re.search(r"(?:weight|体重) ([0-9.]+) kg", out["patient"])
    out["weight_kg"] = float(wm.group(1)) if wm else None
    am = re.search(r"ancestry: ([^;]+)", out["patient"])
    if am:
        out["ancestry"] = am.group(1).strip()
    else:
        from haenv.rl import zh as Z
        az = re.search(r"祖源：([^；]+)", out["patient"])
        out["ancestry"] = Z.ancestry_en(az.group(1).strip()) if az else None
    return out


def prompt_record(block: dict) -> str:
    return render_text(block)
