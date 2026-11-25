"""Documented medication limits, with content-matched inactive records."""
from __future__ import annotations
from copy import deepcopy

from haenv.i18n.text import L, T as template, is_text
from haenv.world.history.history_dependency import HistoryDependency
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
from haenv.p3.gate import BLOCK
from haenv.p3.gate import plan_of


def authored_coverage(subject=None):
    """Keep a source-bound recipe when a coupled plan is replayed in a control.

    Independently generated off-world plans carry no recipe and retain their
    original course. A supplied plan's earlier documented episode is not
    replaced merely because the surrounding background is rendered differently.
    """
    from haenv.world.truth import coupling_enabled
    plan = (subject or {}).get("plan") or subject or {}
    if not coupling_enabled() and plan.get("mechanism_mix_version") != 1:
        return None
    from haenv.resources.regpath import load_registry
    return load_registry("p3_composition.yaml").get("mechanism_coverage")


def limiting_inputs(rows):
    """Only the original clinical premises read by limiting_record.

    Directory labels and a redundant symptom summary are not decision inputs.
    This projection lets C1 compare a source reconstruction with the hidden
    record without requiring the reconstruction to recreate unused metadata.
    """
    from haenv.p3.textread import canonical
    keys = {"day", "drug_id", "prescriptions", "labs", "other_causes", "exposure_class",
            "reaction", "verification", "record_state", "dose_mg", "ck_uln", "secondary_cause",
            "dechallenge", "rechallenge"}
    lab_keys = {"day", "ck_uln", "creatinine_umol_L", "potassium_mmol_L",
                "external_assistance", "unable_to_self_treat"}
    def canon(obj):
        if isinstance(obj, str):
            return canonical(obj)
        if isinstance(obj, list):
            return [canon(v) for v in obj]
        if isinstance(obj, dict):
            return {k: canon(v) for k, v in obj.items()}
        return obj
    out = []
    for row in rows or ():
        observations = any(row.get(k) for k in ("other_causes", "dechallenge", "rechallenge", "labs"))
        if row.get("drug_id") in ("amlodipine", "losartan") and (
                len(row.get("prescriptions") or ()) < 3 or not observations):
            # A canonical archive can document an unrelated medicine without
            # a dose-limit episode. The source reader enriches it with empty
            # arrays or unrelated dated prescriptions. Without episode
            # observations neither form is an input to limiting_record.
            # A damaged required sequence still differs from its complete
            # hidden counterpart, so the visibility gate remains fail-closed.
            continue
        if "secondary_cause" in row and "dose_mg" in row and "ck_uln" in row:
            # Legacy statin archives are one dated prescription/lab event.
            # The EMR reader adds convenient arrays, but limiting_record's
            # legacy branch reads these original flat clinical fields. Compare
            # that same representation on both sides, including drug identity.
            from haenv.p3.policy import identify_drug
            flat = {k: row[k] for k in ("day", "dose_mg", "ck_uln", "secondary_cause",
                                       "dechallenge", "rechallenge") if k in row}
            flat["drug_id"] = row.get("drug_id") or identify_drug(row.get("drug"))
            out.append(canon(flat))
            continue
        if row.get("drug_id") not in ("atorvastatin", "enalapril", "gliclazide", "amlodipine", "losartan") and "exposure_class" not in row and not row.get("ck_uln"):
            continue  # this record cannot establish a limit in limiting_record
        projected = {k: v for k, v in row.items() if k in keys and v is not None}
        if "labs" in projected:
            projected["labs"] = [{k: v for k, v in lab.items() if k in lab_keys} for lab in projected["labs"]]
            if row.get("drug_id") == "gliclazide":
                for lab in projected["labs"]:
                    for key in ("external_assistance", "unable_to_self_treat"):
                        lab[key] = bool(lab.get(key))
        if not row.get("prescriptions") and "drug" in row:
            projected["drug"] = row["drug"]
        out.append(canon(projected))
    return out


def acei_contraindicated(rows, drug):
    """Verified active ACEI angioedema excludes enalapril at any BP response."""
    return drug == "enalapril" and any(
        is_text(r.get("reaction"), "p3.hist.reaction.angioedema")
        and r.get("exposure_class") == "ACEI"
        and is_text(r.get("verification"), "p3.hist.verified")
        and is_text(r.get("record_state"), "p3.hist.state.active") for r in rows or ())


def limiting_record(rows, drug, dose):
    from haenv.world.validation.difficulty_r2 import config
    limits = config("p3")["dose_limit"]
    if acei_contraindicated(rows, drug):
        return True
    for r in rows or ():
        seq = r.get("prescriptions") or []
        labs = r.get("labs") or []
        if r.get("drug_id") != drug or len(seq) < 3:
            continue
        before, higher, later = seq[:3]
        raised = float(higher["dose_mg"]) > float(before["dose_mg"]) == float(dose)
        returned = float(later["dose_mg"]) == float(before["dose_mg"])
        if drug == "amlodipine":
            ordered = before["day"] < higher["day"] <= r["day"] < later["day"]
            if (raised and returned and ordered
                    and is_text(r.get("other_causes"), "p3.hist.cause.none_edema")
                    and is_text(r.get("dechallenge"), "p3.hist.edema.improved")
                    and is_text(r.get("rechallenge"), "p3.hist.rechallenge.lower_tolerated")):
                return True
            continue
        if len(labs) < 3:
            continue
        a, b, c = labs[:3]
        ordered = before["day"] < higher["day"] <= b["day"] < later["day"] <= c["day"]
        if not (raised and returned and ordered):
            continue
        if drug in ("enalapril", "losartan"):
            renal = float(b["creatinine_umol_L"]) / float(a["creatinine_umol_L"]) >= limits["enalapril"]["creatinine_ratio"]
            potassium = float(b["potassium_mmol_L"]) >= limits["enalapril"]["potassium_mmol_L"]
            restored = float(c["potassium_mmol_L"]) < limits["recovery"]["potassium_mmol_L"] and float(c["creatinine_umol_L"]) < limits["recovery"]["creatinine_ratio"] * float(a["creatinine_umol_L"])
            if (renal or potassium) and restored and is_text(r.get("other_causes"), "p3.hist.cause.none_renal"):
                return True
        elif drug == "gliclazide" and b.get("external_assistance") is True and b.get("unable_to_self_treat") is True:
            return True
        elif drug == "atorvastatin" and float(b["ck_uln"]) > limits["atorvastatin"]["ck_uln"] and float(c["ck_uln"]) < limits["recovery"]["ck_uln"] and is_text(r.get("other_causes"), "p3.hist.cause.none_statin"):
            return True
    if drug != "atorvastatin":
        return False
    return any(is_text(r.get("drug"), "drug.atorvastatin") and float(r.get("dose_mg", 0)) > float(dose)
               and float(r.get("ck_uln", 0)) > 10 and is_text(r.get("secondary_cause"), "p3.hist.secondary.none")
               and is_text(r.get("dechallenge"), "p3.hist.dechallenge.ck_normal")
               and is_text(r.get("rechallenge"), "p3.hist.rechallenge.lower_tolerated")
               for r in rows or ())


def prepare(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan):
        return prepare_r2(raw, cid, plan)
    if "history_record" in plan:
        return
    T = int(raw.prediction_context["prediction_time_T"])
    active = plan["drug"] == "atorvastatin" and plan["class"] == "switch" and rate("p3", plan) > 0
    phr = table()["phrasing"]
    source = phr["encounter_sources"][int(uniform("history-source", cid) * len(phr["encounter_sources"]))]
    symptom = phr["muscle_symptoms"][int(uniform("history-word", cid) * len(phr["muscle_symptoms"]))]
    cklo, ckhi = table()["packs"]["p3"]["ck_active" if active else "ck_control"]["value"]
    ck = round(cklo + uniform("history-ck", cid) * (ckhi - cklo), 1)
    control = phr["statin_control"][int(uniform("history-control", cid) * len(phr["statin_control"]))]
    row = {"day": old_day("p3", cid, T), "source": source, "drug": L("drug.atorvastatin"), "dose_mg": 20,
           "symptom": symptom, "ck_uln": ck, "secondary_cause": L("p3.hist.secondary.none" if active else "p3.hist.see_course"),
           "dechallenge": L("p3.hist.dechallenge.ck_normal" if active else "p3.hist.see_course"),
           "rechallenge": (L("p3.hist.rechallenge.lower_tolerated") if active else
                           template("p3.hist.rechallenge.same_tolerated", dose=20)),
           "course": L("p3.hist.course.statin_recovered") if active else template(control.key, ck=ck)}
    plan["history_record"] = row
    plan["history_active"] = active
    if active:
        plan["position"] = "min"
        plan["n_inc"] = 0


def finalize(raw, cid, plan):
    g = raw.adjudication[BLOCK]
    if "class" not in g:
        return
    active = bool(plan["history_active"])
    from haenv.p3.realize import Unrealizable
    if active and 'history_world_source' in plan:
        raise Unrealizable('active history cannot use this inactive medication archive source')
    if plan.get("allergy_control"):
        g.setdefault("adverse_event_record", []).append(deepcopy(plan["allergy_control"]))
    rows = g.get("adverse_event_record") or []
    if active and not rows:
        raise Unrealizable('active history is missing its required source')
    lower = min(int(x["day"]) for x in g["visits"])
    raw.prediction_context[BLOCK]["adverse_event_record"] = deepcopy(rows)
    g["review_from_day"] = lower
    raw.prediction_context[BLOCK]["review_from_day"] = lower
    refs = [{"section": "adverse_event_record", "day": rows[0]["day"]}] if rows else []
    g["history_dependency"] = {"active": active, "recent_lower": lower,
                               "records": refs, "source": table()["packs"]["p3"]["source"]}
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import register
    if enabled(plan):
        if rows:
            register(g, "adverse_event_record", rows[0]["day"], active, recent_lower=lower)
        align_home_pressure_record(raw, g)
    if 'history_world_source' in plan:
        source = plan['history_world_source']
        g['history_dependency']['medication_archive_availability'] = {
            'status': source.get('availability', 'available'), 'requested_day': source['day'],
            **({'reason': source['reason']} if 'reason' in source else {})}


def align_home_pressure_record(raw, g):
    """The measured home-pressure axis requires a cuff and its dated aggregate series."""
    if g.get("axis") != "home_sbp":
        return
    devices = list((raw.user_profile or {}).get("device_inventory") or ())
    if "bp_cuff" not in devices:
        devices.append("bp_cuff")
    raw.user_profile["device_inventory"] = devices
    from haenv.p3.render import home_pressure_rows
    raw.prediction_context.setdefault(BLOCK, {})["home_pressure_record"] = home_pressure_rows(g)


def answer(raw):
    from haenv.p3.gold import g3
    g = raw["adjudication"][BLOCK]
    return g3(g)["class"]


def twin(raw, day=None):
    out = {**raw, "adjudication": {**raw["adjudication"], BLOCK: deepcopy(raw["adjudication"][BLOCK])}}
    g = out["adjudication"][BLOCK]
    g["adverse_event_record"] = [x for x in g.get("adverse_event_record", ()) if day is not None and int(x["day"]) != day]
    return out


POLICY = HistoryDependency(BLOCK, plan_of, prepare, finalize, answer, twin)


def _coupled_glucose(raw, days):
    """Coupled world (B4): the archived record's fasting glucose on `days` is the person's truth -- the
    indicator-cohort baseline the routine panels and course streams read for this person (no drug term is
    added: the cohort values are those of treated patients) -- with its EFLM per-draw variation, printed to
    the record's one decimal; None when coupling is off."""
    from haenv.world.truth import coupling_enabled
    if not coupling_enabled():
        return None
    import math
    from haenv.world.generation.build_clinical import _z
    from haenv.world.truth import panels
    per = panels.person(raw)
    truth = panels._Truth(per, None).base["fasting_glucose"]
    cv = panels._cv("fasting_glucose")
    return tuple(round(truth * math.exp(cv * _z(f"{per['cid']}|b4-history|FBG|{int(d)}")), 1) for d in days)


def _coupled_tsh(raw, cid, day):
    """Coupled world (B4): TSH of the archived levothyroxine record read from the thyroid axis, None when
    coupling is off. On the lower dose before the increase the person is under-replaced (a TSH in the sourced
    under-replacement band, `truth.thyroid`); 14 days on the higher dose TSH has moved part of the way to the
    person's set point, and 27 days back on the lower dose part of the way back (first-order approach with the
    registered half-time, `truth.thyroid.approach_log`). Each value carries the EFLM per-draw variation."""
    from haenv.world.truth import coupling_enabled
    if not coupling_enabled():
        return None
    import math
    from haenv.world.generation import rng
    from haenv.world.generation.build_clinical import _z
    from haenv.world.truth import panels
    from haenv.world.truth import thyroid
    per = panels.person(raw)
    # the person's TSH set point: the same general-population draw the routine panels use
    set_point = panels.table_value("TSH", per["sex"], rng.unit(per["cid"], "panels", "base", "TSH"))
    lo, hi = (float(x) for x in thyroid.lt4_row()["on_treatment_tsh"]["under"]["band_miu_l"])
    u = rng.unit(str(cid), "b4-thyroid", "history-under")
    pre = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    mid = thyroid.approach_log(pre, set_point, 14)
    post = thyroid.approach_log(mid, pre, 27)
    cv = panels._cv("TSH")
    return tuple(round(v * math.exp(cv * _z(f"{cid}|b4-history|TSH|{d}")), 1)
                 for v, d in ((pre, day - 28), (mid, day), (post, day + 28)))


def _world_archive_record(source, cid):
    """Project one existing historical medication reconciliation, without new facts."""
    from haenv.world.clinical.med_course import record_rows
    from haenv.i18n.text import J
    from haenv.p3.realize import Unrealizable
    if source.get('availability') == 'unavailable':
        return None
    day = int(source['day'])
    wm = {'primary': source['medication'], 'concurrent': []}
    facts = record_rows(wm, day, day, primary=True)
    if not facts:
        raise Unrealizable('canonical LT4 archive source has no dated medication fact')
    phrases = table()['phrasing']['encounter_sources']
    text = [part for fact in facts for part in (fact['event'], fact.get('reason')) if part]
    return {'day': day, 'source': phrases[int(uniform('history-source', cid) * len(phrases))],
            'drug_id': source['medication']['drug'], 'drug': facts[0]['drug'],
            'course': J('lines', text)}


def _world_archive_inputs(wm):
    """The relevant source inputs; task overlays cannot bear an earlier archive."""
    rows = [('primary', wm.get('primary'))]
    rows += [('concurrent', med) for med in wm.get('concurrent') or ()]
    meds = [{'origin': origin, **{key: deepcopy(med[key]) for key in
             ('drug', 'indication', 'dose_timeline') if key in med}}
            for origin, med in rows if med and not med.get('overlay')
            and med.get('drug') != 'levothyroxine']
    return {'window_declared': 'history_lo' in wm, 'history_lo': wm.get('history_lo'),
            'medications': meds}


def _world_archive_source(raw, cid, source=None):
    """Select actual earlier exposure; preserve unavailable evidence as unknown."""
    from haenv.world.clinical.med_course import DISPLAY, dose_at
    from haenv.p3.realize import Unrealizable
    def refuse(reason):
        raise Unrealizable('canonical LT4 archive source ' + reason)
    wm = (getattr(raw, 'latent_premise', None) or {}).get('world_medication') or {}
    T = int(raw.prediction_context['prediction_time_T'])
    medicines = [wm.get('primary'), *(wm.get('concurrent') or [])]
    if any(med and not med.get('overlay') and med.get('drug') == 'levothyroxine'
           and any(int(point['ts']) <= T and float(point['value']) > 0
                   for point in med.get('dose_timeline') or []) for med in medicines):
        refuse('has an existing managed medication course; provide an explicit canonical course source')
    expected_day = old_day('p3', cid, T)
    day = int(source['day']) if source is not None else expected_day
    if day != expected_day:
        refuse('date differs from the recorded archive-date policy')
    if source is not None:
        if source.get('version') != 1 or source.get('case_id') != str(cid):
            refuse('has an unsupported version or patient identity')
        status = source.get('availability', 'available')
        if status not in ('available', 'unavailable'):
            refuse('has an unsupported availability state')
        if status == 'unavailable':
            # Re-evaluate the saved query, never overwrite its evidence with a
            # different source or convert unknown into a negative clinical fact.
            if _world_archive_source(raw, cid) != source:
                refuse('unavailable snapshot no longer matches the supplied records')
            return deepcopy(source)
    def unavailable(reason):
        if source is not None:
            refuse('saved available evidence is no longer supported: ' + reason)
        return {'version': 1, 'case_id': str(cid), 'day': day,
                'availability': 'unavailable', 'reason': reason,
                'inputs': _world_archive_inputs(wm)}
    if 'history_lo' not in wm:
        return unavailable('history_window_unavailable')
    if wm['history_lo'] is None:
        refuse('has an invalid declared history window')
    lower = int(wm['history_lo'])
    if day > int(raw.prediction_context['prediction_time_T']):
        refuse('is after the question date')
    if day < lower:
        return unavailable('outside_supplied_history_window')
    entries = [('primary', wm.get('primary'))]
    entries += [('concurrent', row) for row in wm.get('concurrent') or ()]
    choices = []
    for origin, med in entries:
        if (not med or med.get('overlay') or med.get('drug') == 'levothyroxine'
                or med.get('drug') not in DISPLAY):
            continue
        snapshot = {key: deepcopy(med[key]) for key in ('drug', 'indication', 'dose_timeline') if key in med}
        if dose_at(snapshot.get('dose_timeline') or [], day) > 0:
            choices.append((origin, snapshot))
    names = [med['drug'] for _, med in choices]
    if len(names) != len(set(names)):
        refuse('has ambiguous duplicate medication courses')
    if source is not None:
        matches = [(origin, med) for origin, med in choices
                   if origin == source.get('origin') and med['drug'] == source['medication']['drug']]
        if len(matches) != 1 or lower != source['history_lo'] or matches[0][1] != source['medication']:
            refuse('does not match the saved patient medication course')
        return deepcopy(source)
    if not choices:
        return unavailable('no_documented_nonmanaged_exposure')
    origin, medication = min(choices, key=lambda pair: uniform('p3-world-history-source', cid, pair[1]['drug']))
    return {'version': 1, 'case_id': str(cid), 'day': day, 'history_lo': lower,
            'availability': 'available', 'origin': origin, 'medication': medication}


def _restore_archive_text(actual, expected):
    """Restore only exact, source-bound clinical text; never redraw clinical facts."""
    from haenv.i18n.identity import source, restore, display
    from haenv.p3.realize import Unrealizable
    bound = restore(actual, expected)
    if source(bound) == source(expected):
        return bound
    for language in ('zh', 'en'):
        bound = restore(actual, display(expected, language))
        if source(bound) == source(expected):
            return bound
    raise Unrealizable('canonical LT4 archive source text differs from its saved facts')


def _save_allergy_messages(plan):
    if plan.get('allergy_control'):
        import json
        from haenv.i18n.identity import encode
        # ASCII metadata survives whole-case English rendering. It is never
        # emitted as clinical content or exposed to the solver.
        plan['history_allergy_messages'] = json.dumps(encode(plan['allergy_control']), ensure_ascii=True)


def _restore_allergy_messages(plan):
    from haenv.p3.realize import Unrealizable
    actual = plan.get('allergy_control')
    if not actual:
        if 'history_allergy_messages' in plan:
            raise Unrealizable('canonical LT4 allergy source is missing its saved record')
        return None
    if 'history_allergy_messages' not in plan:
        # Earlier in-memory snapshots can retain their original bound messages;
        # unbound persisted clinical strings require an explicit source sidecar.
        from haenv.i18n.text import Msg
        fields = ('source', 'reaction', 'verification', 'record_state', 'symptom', 'record_received')
        if not all(isinstance(actual.get(field), Msg) for field in fields):
            raise Unrealizable('canonical LT4 allergy source message identity is unavailable')
        return deepcopy(actual)
    import json
    from haenv.i18n.identity import decode
    try:
        expected = decode(json.loads(plan['history_allergy_messages']))
        return _restore_archive_text(actual, expected)
    except (KeyError, ValueError, TypeError) as exc:
        raise Unrealizable('canonical LT4 allergy source message descriptor is invalid') from exc


def _recovery_ratio() -> float:
    from haenv.world.validation.difficulty_r2 import config
    return float(config("p3")["dose_limit"]["recovery"]["creatinine_ratio"])


def _recovery_k() -> float:
    from haenv.world.validation.difficulty_r2 import config
    return float(config("p3")["dose_limit"]["recovery"]["potassium_mmol_L"])


def prepare_r2(raw, cid, plan):
    if 'history_world_source' in plan and 'history_record' not in plan:
        from haenv.p3.realize import Unrealizable
        raise Unrealizable('canonical LT4 archive source is missing its saved record')
    if "history_record" in plan:
        if 'history_world_source' in plan:
            # Saved source reuse is checked independently of today's switch.
            from haenv.p3.realize import Unrealizable
            source = _world_archive_source(raw, cid, plan['history_world_source'])
            if plan.get('history_active') is not False:
                raise Unrealizable('canonical LT4 archive source is not an active dose-limit record')
            record = _restore_archive_text(plan['history_record'], _world_archive_record(source, cid))
            allergy = _restore_allergy_messages(plan)
            # Install translation identities only after all facts agree.
            plan['history_record'] = record
            if allergy is not None:
                plan['allergy_control'] = allergy
        return
    from haenv.world.truth import coupling_enabled
    if plan['drug'] == 'levothyroxine' and coupling_enabled() and 'supported_course' in plan:
        # The explicit source already carries real archived facts. Never draw
        # a second prescription episode or replace caller-provided evidence.
        rows = plan['supported_course'].get('adverse_event_record') or []
        if rows:
            plan['history_record'] = deepcopy(rows[0])
            plan['history_active'] = False
        return
    if plan['drug'] == 'levothyroxine' and coupling_enabled():
        # New plans cannot assert an earlier same-drug episode that their
        # patient course does not contain. Keep real unrelated history instead.
        source = _world_archive_source(raw, cid)
        plan['history_record'] = _world_archive_record(source, cid)
        plan['history_world_source'] = source
        plan['history_active'] = False
        prepare_allergy(raw, cid, plan)
        _save_allergy_messages(plan)
        return
    from haenv.world.labworld.meds import drug
    name = plan["drug"]
    coverage = authored_coverage(plan)
    eligible = name in (coverage["historical_limit_drugs"] if coverage else
                        ("atorvastatin", "enalapril", "gliclazide"))
    active = (eligible and plan["class"] == "switch" and plan.get("subtype", "off") == "off"
              and uniform("history-role", cid) < rate("p3", plan) *
              (coverage["historical_limit_share_within_eligible_off_switches"] if coverage else 1))
    # R1: an archived limit that holds at the current dose while another printed signal decides the
    # item (`plan.signal_conflict`); the record meets every criterion but the decision does not
    # depend on it, so it is registered as not acting (`finalize`).
    conflict = bool(eligible and coverage and plan.get("signal_conflict") == "history_limit")
    limit = active or conflict
    family = name
    spec = drug(family)
    rung = int(uniform("history-tolerated-rung", cid) * (len(spec["ladder"]) - 1))
    if coverage and name in coverage["tolerated_rung_by_drug"]:
        rung = int(coverage["tolerated_rung_by_drug"][name])
        if uniform("history-tolerated-rung", cid) < coverage["lowest_tolerated_rung_share"]:
            rung = 0
    if conflict and plan.get("subtype") in ("early", "off_ae"):
        rung = max(1, int(coverage["tolerated_rung_by_drug"].get(name, 1)))   # a step to it / a rung below it
    variant, fail = near_miss(plan, cid, name, spec, coverage, rung) if coverage and not limit else (None, None)
    positive = limit or variant is not None
    low, high = spec["ladder"][rung:rung + 2]
    T = int(raw.prediction_context["prediction_time_T"])
    day = old_day("p3", cid, T)
    sources = table()["phrasing"]["encounter_sources"]
    row = {"day": day, "source": sources[int(uniform("history-source", cid) * len(sources))],
           "drug_id": family, "drug": spec["display"],
           "prescriptions": [{"day": day - 28, "dose_mg": low}, {"day": day - 14, "dose_mg": high},
                             {"day": day + 1, "dose_mg": low}],
           "labs": []}
    if family == "atorvastatin":
        # A striking CK result is also possible after strenuous exercise. Its
        # cause, rather than an independently lower control value, determines
        # whether the old episode limits the statin dose.
        ck = round(12 + 6 * uniform("history-ck", cid), 1)
        row["labs"] = [{"day": day - 28, "ck_uln": 0.7}, {"day": day, "ck_uln": ck}, {"day": day + 28, "ck_uln": 0.8}]
        from haenv.world.observations.findings_render import _value
        from haenv.resources.registry import load_findings
        findings = load_findings()
        from haenv.world.validation.difficulty_r2 import config
        baseline = row["labs"][0]
        for fid in config("p3")["archived_safety_panel"]["findings"]:
            baseline[fid.lower() + "_u_L"] = _value(findings[fid], "normal", None,
                cid, fid, baseline["day"], declared=False)
        row["symptom"] = table()["phrasing"]["muscle_symptoms"][int(uniform("history-word", cid) * 3)]
        row["other_causes"] = L("p3.hist.cause.none_statin" if limit else "p3.hist.cause.hiking")
    elif family in ("enalapril", "losartan"):
        from haenv.p3.realize import _renal, _sex
        renal = _renal(cid, [day - 28, day, day + 28], 0, _sex(raw))
        c0, k0 = renal[0][1:]
        hyper = uniform("history-event", cid) < 0.5
        row["labs"] = [{"day": day - 28, "potassium_mmol_L": k0, "creatinine_umol_L": c0},
            {"day": day, "potassium_mmol_L": round(6.2 + 0.6 * uniform("history-k", cid), 1) if positive and hyper else renal[1][2],
             "creatinine_umol_L": round(c0 * (1.4 + 0.2 * uniform("history-rise", cid))) if positive and not hyper else renal[1][1]},
            # A limiting archive must meet the printed recovery criterion (potassium below the
            # recovery line, creatinine below the recovery ratio of the pre-rise value); the
            # drawn post-episode value is clamped there, never above it.
            {"day": day + 28, "potassium_mmol_L": min(renal[2][2], _recovery_k()) if positive else renal[2][2],
             "creatinine_umol_L": min(renal[2][1], int(c0 * _recovery_ratio()) - 1) if positive else renal[2][1]}]
        row["other_causes"] = L("p3.hist.cause.nsaid_renal" if fail == "cause" else "p3.hist.cause.none_renal")
    elif family == "gliclazide":
        g0, g2 = _coupled_glucose(raw, (day - 28, day + 28)) or (6.1, 6.0)
        assisted = positive and fail != "assistance"
        row["labs"] = [{"day": day - 28, "glucose_mmol_L": g0, "external_assistance": False},
            {"day": day, "glucose_mmol_L": 2.3 if positive else 4.3, "external_assistance": assisted, "unable_to_self_treat": assisted},
            {"day": day + 28, "glucose_mmol_L": g2, "external_assistance": False}]
        row["symptom"] = L("p3.hist.symptom.sweat_tremor" if uniform("history-word", cid) < 0.5 else "p3.hist.symptom.dizzy_sweat")
        row["event_record"] = L("p3.hist.event.assisted" if assisted else "p3.hist.event.self_treated")
        if variant == "different_dose":
            row["course"] = template("p3.hist.hypoglycemia.tolerated_dose", dose=low)
    elif family == "amlodipine":
        if coverage:
            row.update(symptom=L("p3.hist.edema.symptom"),
                other_causes=L("p3.hist.cause.none_edema" if positive and fail != "cause" else "p3.hist.cause.dependent_edema"),
                dechallenge=L("p3.hist.edema.improved" if positive else "p3.hist.edema.unchanged"),
                rechallenge=(L("p3.hist.rechallenge.lower_tolerated") if positive and fail != "rechallenge"
                             else template("p3.hist.rechallenge.same_tolerated", dose=high)))
        else:
            row["course"] = L("p3.hist.course.amlodipine_tolerated")
    else:
        t0, t1, t2 = _coupled_tsh(raw, cid, day) or (12.1, 3.6, 2.5)
        row["labs"] = [{"day": day - 28, "tsh_mIU_L": t0},
                       {"day": day, "tsh_mIU_L": t1}, {"day": day + 28, "tsh_mIU_L": t2}]
        row["symptom"] = L("p3.hist.symptom.cold_fatigue_resolved")
    plan["history_record"], plan["history_active"] = row, active
    if variant:
        plan["history_near_miss"] = variant if variant == "different_dose" else f"{variant}:{fail}"
    if limit:
        plan.update(position="min" if rung == 0 else "mid", tolerated_rung=rung,
                    n_inc=min(int(plan.get("n_inc", 0)), rung), **({} if conflict else {"subtype": "off"}))
    prepare_allergy(raw, cid, plan)


def near_miss(plan, cid, name, spec, coverage, rung):
    """R2 (ours, `signal_conflicts.near_miss_archives`): the inactive archive's variant and the one
    criterion it fails, or (None, None) for the all-negative archive. The archive keeps the rung draw
    acting records use; the variant applies only where the planned dose position fits it, so no dose
    position or composition changes. `one_fact`: the archive's earlier dose is today's dose (pinned,
    `tolerated_rung`) and exactly one printed criterion fails; `different_dose`: every criterion is
    met one rung above today's lowest dose ("A different current dose ... does not establish that
    limit"). The statin keeps its existing alternative-cause archive."""
    from haenv.p3.plan import conflicts
    cfg = conflicts().get("near_miss_archives")
    pos = plan.get("position")
    if (not cfg or name not in coverage["historical_limit_drugs"] or name == "atorvastatin"
            or plan.get("comorbidity") or uniform("history-near-miss", cid) >= float(cfg["share"])):
        return None, None
    lowest = min(int(x) for x in spec["start_rungs"])
    if (pos == "min" and int(rung) == 1 and lowest == 0
            and uniform("history-near-miss-kind", cid) < float(cfg["different_dose_share"])):
        plan["tolerated_rung"] = 0
        return "different_dose", None
    # The archive's pinned tolerated rung must be the item's own rung (so its dose position does not
    # move) and one rung must sit above it. Any mid-ladder rung qualifies, not only the first: the
    # archive is an inactive episode on a non-switch gold, where a shortcut that reads any archived
    # episode as a limit answers switch.
    if pos == "top" or int(rung) < lowest or int(rung) + 1 >= len(spec["ladder"]):
        return None, None
    plan["tolerated_rung"] = int(rung)
    if name in ("enalapril", "losartan"):
        return "one_fact", "cause"
    if name == "gliclazide":
        return "one_fact", "assistance"
    return "one_fact", ("cause" if uniform("history-near-miss-fact", cid) < 0.5 else "rechallenge")


def prepare_allergy(raw, cid, plan):
    """Rare patient-specific records; applicability is in their clinical facts."""
    from haenv.world.validation.difficulty_r2 import config
    cfg = config("p3")["allergy_record"]
    share = cfg["acei_exposed_record_share"] if plan["drug"] == "enalapril" else cfg["patient_record_share"]
    if uniform("p3-patient-allergy-record", cid) >= share["value"]:
        return
    eligible = plan["drug"] == "enalapril" and plan["class"] == "switch" and plan.get("subtype") == "off"
    # Absolute contraindications are single-factor safety anchors. A response
    # intersection instead uses the dose-specific intolerance history; give
    # it the same content-matched inactive allergy controls as other items.
    active = eligible and not plan.get("interaction_family") and uniform("p3-allergy-confirmation", cid) < cfg["confirmed_applicable_share"]["value"]
    T = int(raw.prediction_context["prediction_time_T"])
    original = plan.get("history_record")
    if original is None:
        phrases = table()["phrasing"]["encounter_sources"]
        original = {"source": phrases[int(uniform("history-source", cid) * len(phrases))]}
    verified, pending = L("p3.hist.verified"), L("p3.hist.unverified")
    revoked, valid = L("p3.hist.state.revoked"), L("p3.hist.state.active")
    variants = [("ACEI", verified, revoked), (L("p3.hist.exposure.penicillin"), verified, valid), ("ACEI", pending, valid)]
    exposure, verification, state = ("ACEI", verified, valid) if active else variants[int(uniform("p3-allergy-content", cid) * len(variants))]
    record = {"day": old_day("p3", cid, T), "source": original["source"],
              "exposure_class": exposure, "reaction": L("p3.hist.reaction.angioedema"), "verification": verification,
              "record_state": state, "symptom": L("p3.hist.symptom.angioedema"),
              "record_received": L("p3.hist.received.outside_record")}
    if active:
        # This replaces the limiting episode before it is generated, so there
        # is only one decisive record, never two independent overrides.
        plan["history_record"] = record
        plan["history_active"] = True
        plan["history_mechanism"] = "contraindication"
    else:
        plan["allergy_control"] = record


def save_course(item, *, r=None, y_u=None):
    """Snapshot the existing P3 kinetic model, not a second thyroid model.

    This is a synthetic model choice: registry/p3_meds.yaml explicitly marks
    tau and dose-response synthesis provisional. No B4 coefficients are used.
    Full precision response inputs are retained before legacy params rounding.
    """
    from haenv.world.labworld.meds import drug, axis, coverage_segments, input_steps
    from haenv.p3.plan import specification
    d = drug(item['drug'])
    a = axis(item['axis'])
    line = deepcopy(item['dose_line'])
    segments = coverage_segments(item['refills'], item['a_before'], item['a_last'],
                                 item.get('supply', 30), [day for day, _ in line[1:]])
    admin = item.get('administration_record')
    from haenv.p3.policy import absorption_interaction
    if absorption_interaction(admin):
        fraction = specification()['levothyroxine_absorption']['absorption_fraction']
        segments = [(day, value * fraction) for day, value in segments]
    return dict(version=1, drug=item['drug'], axis=item['axis'],
                dose_line=line, exposure_segments=segments,
                input_steps=input_steps(item['drug'], line, segments),
                response=dict(r=item['params']['r'] if r is None else r,
                              y_u=item['params']['y_u'] if y_u is None else y_u,
                              tau_days=d['tau_truth_days'], model=d['model']),
                measurement=dict(ndigits=a['ndigits'], range=list(a['range'])),
                observations=deepcopy(item['visits']),
                source='registry/p3_meds.yaml; registry/control_targets.yaml',
                modeling='Existing P3 first-order dose/coverage response; provisional synthetic kinetics')


def replay_course(course):
    """Replay saved dated measurements independently of the coupling switch."""
    import math
    from haenv.world.labworld.meds import response, input_steps
    from haenv.world.labworld.observe import make_record
    p, m = course['response'], course['measurement']
    if course['version'] != 1:
        raise ValueError('unsupported medication course version')
    expected = input_steps(course['drug'], course['dose_line'], course['exposure_segments'])
    if [list(x) for x in expected] != [list(x) for x in course['input_steps']]:
        raise ValueError('canonical LT4 dose/exposure initial-state mismatch')
    out = []
    for rec in course['observations']:
        X = response(course['input_steps'], rec['day'], p['tau_days'])
        value = (p['y_u'] * math.exp(-p['r'] * X) if p['model'] == 'log'
                 else p['y_u'] - p['r'] * X)
        out.append(make_record(rec['day'], rec['time'], value, rec['e'],
                               rec['perturbations'], m['ndigits'], tuple(m['range'])))
    return out


def _lt4_archive_rows(item):
    """Identify archive facts by their registered drug name, checking optional IDs."""
    from haenv.p3.policy import identify_drug
    from haenv.p3.realize import Unrealizable
    rows = []
    for row in item.get('adverse_event_record') or ():
        named, declared = identify_drug(row.get('drug')), row.get('drug_id')
        if named == 'levothyroxine' or declared == 'levothyroxine':
            if named != 'levothyroxine' or declared not in (None, 'levothyroxine'):
                raise Unrealizable('canonical LT4 archive source identity mismatch')
            rows.append(row)
    return rows


def connect_course(item, *, r=None, y_u=None):
    """One source for the live and archived LT4 facts; conflicts refuse intact."""
    from haenv.p3.realize import Unrealizable
    saved = item.get('medication_course')
    if saved is not None:
        # These are the existing legacy parameter serialization precisions.
        for field, digits in (('r', 6), ('y_u', 4)):
            if round(item['params'][field], digits) != round(saved['response'][field], digits):
                raise Unrealizable('canonical LT4 saved response/parameter mismatch')
        r = saved['response']['r'] if r is None else r
        y_u = saved['response']['y_u'] if y_u is None else y_u
    source = save_course(item, r=r, y_u=y_u)
    if saved is not None:
        # JSON round-trips convert tuple-valued exposure segments into lists.
        # Compare the serialized data model, preserving every field and value.
        import json
        if json.loads(json.dumps(saved)) != json.loads(json.dumps(source)):
            raise Unrealizable('canonical LT4 saved source/projection mismatch')
    if replay_course(source) != item['visits']:
        raise Unrealizable('canonical LT4 live TSH/exposure mismatch')
    visits = {int(v['day']): v for v in source['observations']}
    # A prescription is an actual dated visit, including same-dose renewals.
    from haenv.p3.gold import dose_at
    start = min(int(d) for d, _ in source['dose_line'])
    lt4_rows = _lt4_archive_rows(item)
    for row in item.get('adverse_event_record') or ():
        if row in lt4_rows:
            # The legacy flat prescription renderer uses mg, not LT4's
            # micrograms. Reject that unsupported shape without rewriting it.
            if 'dose_mg' in row:
                raise Unrealizable('canonical LT4 unsupported flat archive prescription')
            for rx in row.get('prescriptions') or ():
                day = int(rx['day'])
                if day < start or day not in visits or float(rx['dose_mg']) != dose_at(source['dose_line'], day):
                    raise Unrealizable('canonical LT4 archive/live dose mismatch (no supported earlier prescription)')
        # A thyroid result is a fact about this patient even if filed in
        # another drug's encounter. Validate both legacy and nested forms.
        labs = list(row.get('labs') or ())
        if 'tsh_mIU_L' in row:
            labs.append({'day': row['day'], 'tsh_mIU_L': row['tsh_mIU_L']})
        for lab in labs:
            day = int(lab['day'])
            if 'tsh_mIU_L' in lab and (day not in visits or lab['tsh_mIU_L'] != visits[day]['printed']):
                raise Unrealizable('canonical LT4 archive/live TSH mismatch')
    item['medication_course'] = source
    # Reconstruct both views from that source after equality has been checked.
    item['dose_line'] = deepcopy(source['dose_line'])
    item['visits'] = replay_course(source)
    archives = deepcopy(item.get('adverse_event_record') or [])
    for row in archives:
        if row in lt4_rows:
            for rx in row.get('prescriptions') or ():
                rx['dose_mg'] = dose_at(source['dose_line'], rx['day'])
            for lab in row.get('labs') or ():
                if 'tsh_mIU_L' in lab:
                    lab['tsh_mIU_L'] = visits[int(lab['day'])]['printed']
    if archives:
        item['adverse_event_record'] = archives
    return item
