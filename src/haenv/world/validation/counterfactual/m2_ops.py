"""Pure m2 intervention candidates ported from the emission-check investigation."""
import copy
from dataclasses import replace

from haenv.i18n.text import L as Lmsg
from haenv.m2 import core, premises
from haenv.world.validation import gates_outcome as GO
from haenv.world.clinical.med_course import DISPLAY
from haenv.resources.registry import condition_findings_for_case


def names_of(fid):
    from haenv.m2 import core
    f = core._findings().get(fid) or {}
    return [str(n).strip() for n in [f.get("name_cn"), f.get("name_en"), *(f.get("aliases") or [])] if n and str(n).strip()]

def stream_of(fid):
    from haenv.world.observations.findings_render import _TS_TO_PANEL
    return {v: k for k, v in _TS_TO_PANEL.items()}.get(fid)

def abnormal(pts):
    for p in pts or ():
        fl = str(p.get("flag") or "")
        if fl in ("POSITIVE",):
            return True
        v, lo, hi = p.get("value"), p.get("ref_low"), p.get("ref_high")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and (
                (hi is not None and v > hi) or (lo is not None and v < lo)):
            return True
        if isinstance(p.get("details"), dict) and p.get("value") and fl not in ("NORMAL", "NEGATIVE", ""):
            return True
    return False

def norm_point(p):
    v, lo, hi = p.get("value"), p.get("ref_low"), p.get("ref_high")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and (lo is not None or hi is not None):
        lo_ = float(lo) if lo is not None else 0.0
        hi_ = float(hi) if hi is not None else lo_ * 2
        p["value"] = round((lo_ + hi_) / 2, 2); p["flag"] = "NORMAL"
    elif str(p.get("flag")) not in ("NORMAL", "NEGATIVE", "N"):
        p["flag"] = "NEGATIVE"; p["value"] = Lmsg("m2.obs.negative_normal")
        if isinstance(p.get("details"), dict):
            p["details"] = {}
    return p

def set_value(v):
    def f(p):
        p["value"] = v; p["flag"] = "NORMAL"
        return p
    return f

def renal_rows(p):
    d = dict(p.get("details") or {}); d["uacr_history"] = list(d.get("uacr_history") or [])[:3]; p["details"] = d
    return p

def renal_below(p):
    d = dict(p.get("details") or {})
    d["uacr_history"] = [{**r, "value": 20.0} for r in d.get("uacr_history") or []]; p["details"] = d
    return p

def W_(raw, **fields):
    return replace(raw, **{k: v for k, v in fields.items()})

def twins(c):
    raw, T, ddx, th = c["raw"], c["T"], c["ddx"], c["th"]
    pc = raw.prediction_context; led = raw.evidence_ledger or []
    man = c["manifest"]
    real = {str(x) for x in man.get("real_symptom_evidence_ids") or []}
    benign = sorted(str(x) for x in man.get("benign_evidence_ids") or [])
    look = sorted(str(x) for x in man.get("lookalike_evidence_ids") or [])
    vis = lambda ids: [e for e in led if str(e.get("evidence_id")) in ids and int(e.get("source_timestamp", 1e9)) <= T]
    drop = lambda ids: W_(raw, evidence_ledger=[e for e in led if str(e.get("evidence_id")) not in ids])
    out = []
    keyfids = set()
    # --- (a)/(b) key-signal operators
    for t in th:
        for k in t["key_signals"]:
            fid = k[8:] if k.startswith("archive:") else k
            keyfids.add(fid)
            base = core._on_demand(raw, fid, T)
            meta = {"line": t["name"], "component": t["component"], "k": k, "fid": fid,
                    "baseline_od_abnormal": abnormal(base), "archive_key": k.startswith("archive:")}
            op_rows = [r for r in pc.get("outpatient_laboratory_record") or [] if r.get("finding") == fid]
            pc2 = copy.deepcopy(pc)
            if op_rows:
                for r in pc2.get("outpatient_laboratory_record") or []:
                    if r.get("finding") == fid and r.get("ref_high") is not None:
                        r["value"] = round((float(r.get("ref_low") or 0) + float(r["ref_high"])) / 2, 2)
                out.append(("a_outpatient", meta, W_(raw, prediction_context=copy.deepcopy(pc2)), None))
            arch = [r for r in pc.get("archived_laboratory_record") or [] if r.get("finding") == fid]
            if arch:
                pa = copy.deepcopy(pc2)
                for r in pa.get("archived_laboratory_record") or []:
                    if r.get("finding") == fid:
                        r["observation"] = [norm_point(dict(o)) for o in r.get("observation") or []]
                out.append(("a_archive_norm", meta, W_(raw, prediction_context=pa), None))
                pd_ = copy.deepcopy(pc)
                pd_["archived_laboratory_record"] = [r for r in pd_.get("archived_laboratory_record") or [] if r.get("finding") != fid]
                out.append(("a_archive_drop", meta, W_(raw, prediction_context=pd_), None))
                pc2 = pa
            if meta["baseline_od_abnormal"]:
                out.append(("b_ondemand", meta, raw, {fid: norm_point}))
    # --- (c) real symptoms
    rv = vis(real)
    comps = [t["component"] for t in th]
    owner = {}
    for e in rv:
        eid = str(e["evidence_id"])
        if len(th) == 1:
            owner[eid] = th[0]["name"]
        elif th:
            o = core.symptom_owner(c["sources"].get(eid, str(e.get("symptom"))), comps)
            owner[eid] = next((t["name"] for t in th if t["component"] == o), None) if o else None
    for e in rv:
        eid = str(e["evidence_id"])
        out.append(("c_sym_one", {"line": owner.get(eid), "eid": eid, "remaining": len(rv) - 1}, drop({eid}), None))
    for t in th:
        ids = {k for k, v in owner.items() if v == t["name"]}
        if ids:
            out.append(("c_sym_line", {"line": t["name"], "n": len(ids), "remaining": len(rv) - len(ids)}, drop(ids), None))
    if rv:
        out.append(("c_sym_all", {"n": len(rv), "owned": sum(v is not None for v in owner.values())}, drop({str(e["evidence_id"]) for e in rv}), None))
    # --- (d) premise operators
    for comp, chk in premises.requirements(ddx):
        meta = {"component": comp, "kind": chk["kind"]}
        if chk["kind"] == "medication_exposure":
            wm = raw.latent_premise.get("world_medication") or {}
            names = {DISPLAY[str(m.get("drug"))][0] for m in [wm.get("primary") or {}, *(wm.get("concurrent") or [])]
                     if chk["drug_class"] in str(m.get("drug") or "") and str(m.get("drug")) in DISPLAY}
            p1 = copy.deepcopy(pc); p1["medication_record"] = [r for r in p1.get("medication_record") or [] if r.get("drug") not in names]
            out.append(("d_med_drop_records", {**meta, "drugs": sorted(names)}, W_(raw, prediction_context=p1), None))
            p2 = copy.deepcopy(pc)
            syms = [int(s["day"]) for s in c["cs"].raw.get("symptoms") or [] if int(s["day"]) <= T]
            onset = min(syms) if syms else T
            for r in p2.get("medication_record") or []:
                if r.get("drug") in names:
                    r["day"] = onset + 1
            out.append(("d_med_after_onset", {**meta, "onset": onset}, W_(raw, prediction_context=p2), None))
        elif chk["kind"] == "glycaemic_diagnosis":
            row = {"day": T, "source": "外院检验报告", "finding": "FBG", "name": "空腹血糖", "value": 6.0, "unit": "mmol/L", "ref_low": 3.9, "ref_high": 6.1}
            p1 = copy.deepcopy(pc); p1.setdefault("outpatient_laboratory_record", []).append(dict(row))
            out.append(("d_glyc_fpg_today_row", meta, W_(raw, prediction_context=p1), None))
            out.append(("d_glyc_fpg_hook", meta, raw, {"FBG": set_value(6.0)}))
            out.append(("d_glyc_hba1c_hook", meta, raw, {"HbA1c": set_value(6.0)}))
            from haenv.m2.glycaemia import _onset
            on = _onset(raw, {comp}, T)
            if on is not None and on < T - 1:
                p2 = copy.deepcopy(pc); p2.setdefault("outpatient_laboratory_record", []).append({**row, "day": (on + T) // 2})
                out.append(("d_glyc_post_onset_row", {**meta, "onset": on}, W_(raw, prediction_context=p2), None))
        elif chk["kind"] == "renal_course":
            out.append(("d_renal_rows_hook", meta, raw, {chk["finding"]: renal_rows}))
            out.append(("d_renal_below_hook", meta, raw, {chk["finding"]: renal_below}))
    # --- negative controls
    for eid in [str(e["evidence_id"]) for e in vis(set(benign))][:3]:
        out.append(("neg_benign_drop", {"eid": eid}, drop({eid}), None))
    for eid in [str(e["evidence_id"]) for e in vis(set(look))]:
        out.append(("neg_lookalike_drop", {"eid": eid}, drop({eid}), None))
    for f in sorted({r.get("finding") for r in pc.get("outpatient_laboratory_record") or []} - keyfids - {None}):
        p1 = copy.deepcopy(pc)
        for r in p1["outpatient_laboratory_record"]:
            if r.get("finding") == f and r.get("ref_high") is not None:
                r["value"] = round((float(r.get("ref_low") or 0) + float(r["ref_high"])) / 2, 2)
        out.append(("neg_outpatient_nonkey", {"fid": f}, W_(raw, prediction_context=p1), None))
    sid = str(ddx.get("spec_id") or "")
    decl = [str(x.get("id")) for x in ((condition_findings_for_case(core._findings()) or {}).get(sid) or {}).get("findings") or []]
    n = 0
    for f in decl:
        if f in keyfids or n >= 3:
            continue
        try:
            if abnormal(core._on_demand(raw, f, T)):
                out.append(("neg_ondemand_nonkey", {"fid": f}, raw, {f: norm_point})); n += 1
        except Exception:
            pass
    return out
