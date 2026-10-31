"""P3 source records, projected from existing facts without adding clinical events.

The reader below reconstructs decision inputs from the visible notes, results and
prescriptions. Neither that reader nor the renderer uses a gold class or event role.
"""
from __future__ import annotations

from copy import deepcopy

from haenv.i18n.text import J, L, Msg, T, is_text
from haenv.p3.textread import canonical, field, is_one, number, prescription_dose, slots

#: lab field -> (test name phrase, unit); a unit phrase is rendered, a plain unit is printed as is
LABS = {
    "ck_uln": (L("p3.lab.ck"), L("p3.unit.x_uln")),
    "potassium_mmol_L": (L("p3.lab.potassium"), "mmol/L"),
    "creatinine_umol_L": (L("p3.lab.creatinine"), "μmol/L"),
    "glucose_mmol_L": (L("p3.lab.glucose"), "mmol/L"),
    "tsh_mIU_L": ("TSH", "mIU/L"),
    "alt_u_L": ("ALT", "U/L"), "ast_u_L": ("AST", "U/L"),
}


def _line(key: str, text):
    return T(f"p3.emr.line.{key}", text=text)


def _field(text, label: str):
    """The value of the `label` line of a note (`p3.emr.line.<label>`), in any language."""
    return field(text, f"p3.emr.line.{label}")


def _short(name):
    """The short printed name of a drug: a world drug phrase as it is, a managed drug's display
    (`meds.drug(k)["display"]`) as `drug.<k>`."""
    from haenv.world.clinical.med_course import drug_key
    if drug_key(name) is not None:
        return name
    from haenv.i18n.text import rewrap
    from haenv.p3.policy import identify_drug as parse_drug
    key = parse_drug(name)
    if key is None:
        return rewrap(name, str.__str__(name).split("\uff08", 1)[0])
    return L(f"drug.{key}")


def _drug_sort(name) -> str:
    """Records are ordered by the drug's canonical name, whatever language they are printed in."""
    from haenv.world.clinical.med_course import drug_key
    k = drug_key(name)
    return str.__str__(L(f"drug.{k}")) if k else canonical(name)


_DOSE_EVENTS = ("med.event.start", "med.event.restart", "med.event.before_record")


def medication_rows(rows):
    """Use the same prescribing fields for every medicine in the dated record."""
    from haenv.world.clinical.med_course import display
    from haenv.world.clinical.med_course import drug_key
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("prescribing history requires dated records")
        row = dict(r)
        row["drug"] = _short(row["drug"])
        if "prescription" not in row:
            key, event = drug_key(row["drug"]), row["event"]
            if slots(event, "med.event.hold") is not None or slots(event, "med.event.stop") is not None:
                row["prescription"] = L("p3.rx.stopped")
            else:
                got = slots(event, "med.event.dose_change")
                dose = got["new"] if got else next((slots(event, k)["dose"] for k in _DOSE_EVENTS
                                                    if slots(event, k) is not None), None)
                if dose is None:
                    raise ValueError("recorded medication dose is missing")
                row["prescription"] = display(key, float(dose))
        row.setdefault("event", L("p3.event.prescribed"))
        out.append(row)
    return sorted(out, key=lambda r: (r["day"], _drug_sort(r.get("drug", ""))))


def source_block(legacy, current_medications, patient_quote):
    """One outpatient-note stream; old chemistry and prescribing leave its prose."""
    from haenv.world.labworld.meds import drug
    from haenv.p3.policy import identify_drug as parse_drug
    b = deepcopy(legacy)
    key = parse_drug(b["drug"])
    medication_name = L(f"drug.{key}")
    unit = drug(key)["unit"].split("/")[0]
    notes, labs, prescriptions, home = [], [], [], []
    for r in b["visits"]:
        day = int(r["day"])
        has_rx = not is_one(r["prescription"], "p3.rx.pending", "p3.rx.none")
        text = [_line("complaint", L("p3.emr.complaint.visit")), _line("diagnosis", b["condition"]),
                _line("rx", L("p3.emr.rx.see_record") if has_rx else r["prescription"])]
        if not has_rx:
            text.append(_line("record", r["change"]))
        if not is_one(r["reading"], "p3.reading.not_measured", "p3.reading.not_due"):
            dest = home if drug(key).get("self_measured") else labs
            dest.append({"day": day, "test": b["control_axis"], "result": r["reading"]})
        else:
            text.append(_line("exam", r["reading"]))
        if r.get("labs"):
            labs.append({"day": day, "test": L("p3.lab.renal_panel"), "result": r["labs"]})
        if day == min(int(v["day"]) for v in b["visits"]):
            text.append(_line("order", b["clinic_target"]))
            if b.get("known_comorbidity"):
                text.append(_line("pmh", b["known_comorbidity"]))
        notes.append({"day": day, "source": L("p3.source.progress_note"), "text": J("lines", text)})
        if has_rx:
            prescriptions.append({"day": day, "drug": b["drug"], "prescription": r["prescription"],
                                  "event": r["change"]})
    for r in b.get("encounter_record") or ():
        # Structured symptom/onset indexing is ordinary encounter metadata;
        # the clinical history remains in the note body, never in a directory label.
        notes.append({**r, "text": J("lines", [_line("complaint", r["symptom"]), _line("hpi", r["history"])])})
    for r in b.get("adverse_event_record") or ():
        day = int(r["day"])
        if r.get("exposure_class"):
            text = [_line("complaint", r["symptom"]), _line("drug_history", r["exposure_class"]),
                    _line("clinical_note", r["reaction"]), _line("verification", r["verification"]),
                    _line("record_state", r["record_state"]), _line("record_received", r["record_received"])]
        else:
            text = [_line("complaint", r.get("symptom", L("p3.emr.complaint.revisit"))), _line("medication", _short(r["drug"]))]
            for fld, label in (("other_causes", "inquiry"), ("event_record", "course"),
                               ("secondary_cause", "workup"), ("dechallenge", "management"),
                               ("rechallenge", "followup"), ("course", "hpi")):
                if r.get(fld):
                    text.append(_line(label, r[fld]))
            for p in r.get("prescriptions") or ():
                prescriptions.append({"day": int(p["day"]), "drug": r["drug"],
                                      "prescription": T("p3.rx.daily_unit", name=_short(r["drug"]), dose=f"{p['dose_mg']:g}", unit=unit)})
            old_labs = r.get("labs") or [{"day": day, **{k: r[k] for k in LABS if k in r}}]
            for p in old_labs:
                for fld, (name, lab_unit) in LABS.items():
                    if fld in p:
                        labs.append({"day": int(p["day"]), "test": name, "result": T("p3.lab.result", value=p[fld], unit=lab_unit)})
            if "dose_mg" in r:
                prescriptions.append({"day": day, "drug": r["drug"],
                                      "prescription": T("p3.rx.daily_unit", name=_short(r["drug"]), dose=f"{r['dose_mg']:g}", unit="mg")})
        notes.append({"day": day, "source": r["source"], "text": J("lines", text)})
    prescriptions.extend(deepcopy(b.get("other_medications") or []))
    out = {"today_date": b.get("today_date", T("p3.today_day", day=b['today'])),
           "encounter_type": L("encounter.p3"), "patient_quote": patient_quote,
           "current_medications": deepcopy(current_medications),
           "encounter_record": sorted(notes, key=lambda r: r["day"]),
           "lab_results": sorted(labs, key=lambda r: (r["day"], canonical(r["test"]))),
           "prescription_record": medication_rows(prescriptions),
           "dispensing_record": [{**r, "drug": medication_name} for r in b["refill_record"]],
           "medication_review": [{"day": int(b["today"]), "drug": medication_name, "text": b["pill_count_today"]}],
           "symptom_diary": b["symptom_diary"], "text_source": b.get("text_source", "template")}
    if b.get("administration_record"):
        out["administration_record"] = deepcopy(b["administration_record"])
    if b.get("medication_instructions"):
        out["medication_instructions"] = deepcopy(b["medication_instructions"])
    if home:
        out["home_measurements"] = home
    if "protocol_library" in b:
        out.update(protocol_library=b["protocol_library"], rules_delivery=b["rules_delivery"])
    else:
        # Printed delivery is retained as a source document for offline controls.
        out["clinic_documents"] = b["clinic_protocol"]
    return out


def _contains_phrase(text, phrase) -> bool:
    """Inspect phrase IDs in live message slots; support persisted source documents too."""
    if isinstance(text, Msg):
        if text.kind == "lex":
            return text.key == getattr(phrase, "key", phrase)
        return any(_contains_phrase(v, phrase) for v in text.slots.values()
                   if isinstance(v, (str, list, tuple)))
    if isinstance(text, (list, tuple)):
        return any(_contains_phrase(v, phrase) for v in text)
    import haenv.i18n as _i18n
    from haenv.i18n.text import MissingText, render
    for lang in _i18n.LANGUAGES:
        try:
            if str.__str__(render(phrase, lang)) in text:
                return True
        except MissingText:
            continue
    return False


def clinical_view(b):
    """Parse only source contents into the existing G3 reader's input schema."""
    if "lab_results" not in b:
        return b
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    from haenv.p3.policy import identify_drug as parse_drug
    notes = b.get("encounter_record") or []
    visits = [r for r in notes if is_text(_field(r.get("text", ""), "complaint"), "p3.emr.complaint.visit")]
    orders = [(r, _field(r["text"], "order")) for r in visits if _field(r["text"], "order")]
    if len(orders) != 1:
        raise ValueError("source records do not identify a single review course")
    order, target = orders[0]
    from haenv.p3.policy import drug_keys
    named = slots(target, "p3.target.clinic_drug")
    candidates = ([parse_drug(named["name"])] if named else
                  [k for k in drug_keys() if _contains_phrase(target, axis(drug(k)["axis"])["display"])])
    if len(candidates) != 1 or candidates[0] is None:
        raise ValueError("review target is missing or ambiguous")
    key = candidates[0]
    name, ax = drug(key)["display"], axis(drug(key)["axis"])["display"]
    T_ = max(int(r["day"]) for r in visits)
    rows = []
    for r in visits:
        day = int(r["day"])
        results = [p["result"] for p in b["lab_results"] + (b.get("home_measurements") or [])
                   if int(p["day"]) == day and is_text(p["test"], getattr(ax, "key", ax))]
        prescription = _field(r["text"], "rx")
        change = _field(r["text"], "record")
        if is_text(prescription, "p3.emr.rx.see_record"):
            matches = [p for p in b.get("prescription_record") or []
                       if int(p["day"]) == day and parse_drug(p.get("drug")) == key]
            if len(matches) != 1:
                raise ValueError("dated prescription missing or ambiguous")
            prescription, change = matches[0]["prescription"], matches[0]["event"]
        row = {"day": day, "reading": results[0] if len(results) == 1 else _field(r["text"], "exam"),
               "prescription": prescription, "change": change}
        renal = [p["result"] for p in b["lab_results"] if int(p["day"]) == day and is_text(p["test"], "p3.lab.renal_panel")]
        if renal:
            row["labs"] = renal[0]
        rows.append(row)
    reviews = [r for r in b.get("medication_review") or [] if parse_drug(r.get("drug")) == key]
    if len(reviews) != 1:
        raise ValueError("medication reconciliation missing")
    out = {"drug": name, "control_axis": ax, "clinic_target": target, "today": T_,
           "condition": _field(order["text"], "diagnosis"),
           "visits": sorted(rows, key=lambda r: r["day"]), "symptom_diary": b.get("symptom_diary") or [],
           "refill_record": [{k: v for k, v in p.items() if k != "drug"} for p in b.get("dispensing_record") or []
                             if parse_drug(p.get("drug")) == key],
           "pill_count_today": reviews[0]["text"],
           "encounter_record": [r for r in notes if "symptom" in r],
           "other_medications": [r for r in b.get("prescription_record") or [] if parse_drug(r.get("drug")) != key],
           "adverse_event_record": [],
           **({"administration_record": b["administration_record"]} if b.get("administration_record") else {})}
    documented = slots(reviews[0]["text"], "p3.review.documented_only")
    if documented is not None:
        out["documented_course_only"] = True
    if _field(order["text"], "pmh"):
        out["known_comorbidity"] = _field(order["text"], "pmh")
    if b.get("protocol_library"):
        out["protocol_library"] = b["protocol_library"]
    if b.get("medication_instructions"):
        out["medication_instructions"] = b["medication_instructions"]
    elif "clinic_documents" in b:
        out["clinic_protocol"] = b["clinic_documents"]
    lower = min(int(r["day"]) for r in visits)
    for note in notes:
        text, day = note.get("text", ""), int(note["day"])
        if day >= lower:
            continue
        exposure = _field(text, "drug_history")
        if exposure:
            out["adverse_event_record"].append({"day": day, "exposure_class": exposure,
                "reaction": _field(text, "clinical_note"), "verification": _field(text, "verification"),
                "record_state": _field(text, "record_state")})
            continue
        old_drug = _field(text, "medication")
        old_key = parse_drug(old_drug)
        if old_key is None:
            continue
        old = {"day": day, "drug": old_drug, "drug_id": old_key,
               "other_causes": _field(text, "inquiry")}
        rx = sorted([r for r in b.get("prescription_record") or [] if parse_drug(r.get("drug")) == old_key
                     and "prescription" in r and int(r["day"]) < lower], key=lambda r: int(r["day"]))
        prior, later = [r for r in rx if r["day"] <= day], [r for r in rx if r["day"] > day]
        selected = prior[-2:] + later[:1]
        old["prescriptions"] = [{"day": int(r["day"]),
            "dose_mg": prescription_dose(r["prescription"])} for r in selected]
        panels = {}
        for result in b.get("lab_results") or []:
            if not day - 28 <= int(result["day"]) <= day + 28:
                continue
            for fld, (lab_name, _) in LABS.items():
                if is_text(result["test"], lab_name.key if isinstance(lab_name, Msg) else lab_name) or result["test"] == lab_name:
                    panels.setdefault(int(result["day"]), {"day": int(result["day"])})[fld] = number(result["result"])
        old["labs"] = [panels[d] for d in sorted(panels)]
        if old_key == "gliclazide":
            event = _field(text, "course")
            for panel in old["labs"]:
                panel["external_assistance"] = panel["day"] == day and is_text(event, "p3.hist.event.assisted")
                panel["unable_to_self_treat"] = panel["external_assistance"]
        if _field(text, "management"):
            old.update(secondary_cause=_field(text, "workup"), dechallenge=_field(text, "management"),
                       rechallenge=_field(text, "followup"))
            old["dose_mg"] = old["prescriptions"][-1]["dose_mg"]
            old["ck_uln"] = next(p["ck_uln"] for p in old["labs"] if p["day"] == day)
        out["adverse_event_record"].append(old)
    return out


def finalize(raw):
    """Project only P3 after its existing history finalizer, before snapshot collection."""
    import haenv.world.clinical.med_course as MC
    from haenv.p3.gate import BLOCK
    from haenv.p3.render import render_block
    g = (raw.adjudication or {}).get(BLOCK) or {}
    if "class" not in g:
        return
    wm = (raw.latent_premise or {}).get("world_medication") or {}
    medications = sorted(MC.display(k, dose) for k, dose, _ in MC.active_medicines(wm, int(g["T"])))
    if g.get("administration_record"):
        medications.append(L("p3.med.calcium"))
    utterances = sorted((e for e in raw.evidence_ledger or [] if e.get("source_type") == "patient_reported_symptom"),
                        key=lambda e: int(e.get("source_timestamp", 0)))
    if not utterances:
        raise ValueError("P3 needs a recorded patient utterance")
    quote = utterances[-1]["symptom"]
    g.update(record_format="emr", current_medications=medications, patient_quote=quote)
    raw.prediction_context[BLOCK] = source_block(render_block(g), medications, quote)
    for manifest in (g.get("history_dependency") or {}, g.get("look_alike") or {}):
        for ref in manifest.get("records") or []:
            if ref["section"] == "adverse_event_record":
                ref["section"] = "encounter_record"


def observe(next_call, raw, case_id):
    result = next_call(raw, case_id)
    finalize(raw)
    return result


def install():
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.OBSERVE, "p3:emr", observe)
