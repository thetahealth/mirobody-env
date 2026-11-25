"""Encounter-driven routine laboratory panels (world 1.5, B3 / fidelity F1). SYNTHETIC, evaluation only.

Every laboratory encounter of a p2/p3/p4 case's record (the world's visit calendar, the history prefix and
the pack's own draws, day <= T) is checked against the ordering rules of `registry/routine_panels.yaml`
(annual review of hypertension and diabetes, screening by age, monitoring of treated hypothyroidism). A rule
that is due orders its panels; a panel draws every member the case does not already carry on another
surface. Values come from the person's latent baseline (the indicator cohort draw the course streams use,
else the general-population table), the world's drug terms where the world has them, the background
constraints of `effects.yaml` and the registered per-draw variation. The laboratory prints eGFR with every
creatinine (CKD-EPI 2021) and free T4 with every TSH (`truth.thyroid.ft4`), including the pack's own
creatinine and TSH draws.

Ordering never reads the case's gold or its tier, only the conditions and age a guideline names; the panel
analytes are background: they are never the pack's judged analyte (those are owned and left alone), so the
gold and the emission gates do not move (the layer runs after the gates). A panel is written to
`longitudinal_data` only when the pack's payload view keeps every one of its streams; otherwise it is kept
in `latent_premise.routine_panels.withheld` (verifier side), so a view that drops part of a panel never
shows a partial panel.

Opt-in: `synth.course_coupling` and `world.routine_panels` (unset = on under coupling). Off: unreachable.
"""
from __future__ import annotations

import functools
import math
import os
import re

from haenv.resources.regpath import load_registry, registry_cached

PANELS_KEY = "routine_panels"
_TOP = {"version", "sources", "packs", "panels", "conditions", "rules", "analytes", "thyroid"}
_RULE_KEYS = {"panels", "when", "interval_days", "phase", "source", "note"}
_WHEN_KEYS = {"any_condition", "no_condition", "min_age", "or_min_age"}
_PHASES = {"record_start", "uniform"}
#: lab-calendar streams of the world (`build_clinical.CLINICAL_SPEC`) with registered indicator cohorts
COHORT_ANALYTES = ("HbA1c", "fasting_glucose", "LDL", "triglycerides", "ALT", "AST")
#: p4 follow-up analyte -> panel stream
P4_ANALYTES = {"creatinine": "creatinine", "potassium": "potassium", "glucose": "fasting_glucose",
               "tsh": "TSH", "alt": "ALT", "triglycerides": "triglycerides"}


# ------------------------------------------------------------------ registry
def validate(doc: dict) -> dict:
    """Refuse unknown keys, panels, analytes, rules or conditions, unverified sources and unsourced numbers."""
    unknown = set(doc) - _TOP
    if unknown:
        raise ValueError(f"unknown routine_panels keys: {sorted(unknown)}")
    sources = doc.get("sources") or {}
    for key, row in sources.items():
        if row.get("status") != "verified":
            raise ValueError(f"routine_panels source not verified: {key}")

    def walk(node, path, inherited=()):
        if isinstance(node, dict):
            source = node.get("source", inherited)
            source = [source] if isinstance(source, str) else source
            if not isinstance(source, (list, tuple)) or any(s not in sources for s in source):
                raise ValueError(f"unresolved source at {path}: {source}")
            for key, value in node.items():
                if key not in ("source", "note"):
                    walk(value, f"{path}.{key}", source)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]", inherited)
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and not inherited:
            raise ValueError(f"number without source at {path}")

    for key in ("panels", "rules", "analytes", "thyroid"):
        walk(doc.get(key) or {}, key)
    analytes = doc.get("analytes") or {}
    for name, row in (doc.get("panels") or {}).items():
        for a in list(row.get("members") or ()) + list(row.get("derived") or ()):
            if a not in analytes:
                raise ValueError(f"panel {name}: unknown analyte {a}")
    conds = set(doc.get("conditions") or {}) | {"diabetes"}
    for name, rule in (doc.get("rules") or {}).items():
        if set(rule) - _RULE_KEYS or set(rule.get("when") or {}) - _WHEN_KEYS:
            raise ValueError(f"rule {name}: unknown fields")
        if rule.get("phase") not in _PHASES:
            raise ValueError(f"rule {name}: unknown phase {rule.get('phase')}")
        if not int(rule["interval_days"]) > 0:
            raise ValueError(f"rule {name}: interval must be positive")
        for p in rule["panels"]:
            if p not in (doc.get("panels") or {}):
                raise ValueError(f"rule {name}: unknown panel {p}")
        w = rule.get("when") or {}
        for c in list(w.get("any_condition") or ()) + list(w.get("no_condition") or ()):
            if c not in conds:
                raise ValueError(f"rule {name}: unknown condition {c}")
    for name, row in analytes.items():
        cv = row.get("cv")
        if not row.get("equation") and cv != "biological_variation":
            raise ValueError(f"analyte {name}: cv must be biological_variation")
    return doc


@registry_cached("routine_panels.yaml")
def registry() -> dict:
    return validate(load_registry("routine_panels.yaml"))


@functools.lru_cache(maxsize=8)
def _panels_on(key: str) -> bool:
    from haenv.app.config import load_cfg
    value = (load_cfg().get("world") or {}).get("routine_panels")
    return True if value is None else bool(value)


def enabled() -> bool:
    """Panels are drawn only in the course-coupled world and when `world.routine_panels` is not false."""
    from haenv.world.truth import coupling_enabled
    if not coupling_enabled():
        return False
    from haenv.app.config import CONFIG_OVERLAY_ENV
    from haenv.app.settings import SET_ENV
    return _panels_on(os.environ.get(SET_ENV, "") + "|" + os.environ.get(CONFIG_OVERLAY_ENV, ""))


# ------------------------------------------------------------------ the person
def _conditions(raw) -> tuple[str, list[str], str]:
    pb = (getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {}
    disease, comorb = str(pb.get("disease") or ""), [str(c) for c in pb.get("comorbidities") or ()]
    return disease, comorb, " ".join([disease, *comorb]).lower()


def has_condition(hay: str, name: str) -> bool:
    if name == "diabetes":
        markers = load_registry("clinical_baselines.yaml")["diabetes_markers"]
    else:
        markers = registry()["conditions"][name]
    return any(str(m).lower() in hay for m in markers)


def person(raw) -> dict:
    from haenv.world.generation import rng
    cid = str(raw.case_id)
    user = getattr(raw, "user_profile", None) or {}
    a, b = (float(x) for x in str(user["age_range"]).split("-"))
    disease, comorb, hay = _conditions(raw)
    return {"cid": cid, "sex": str(user.get("sex") or "").upper()[:1] or None,
            "age_T": a + rng.unit(cid, "panels", "age") * (b + 1 - a),
            "disease": disease, "comorbidities": comorb, "hay": hay,
            "T": int((getattr(raw, "prediction_context", None) or {}).get("prediction_time_T") or 0)}


def age_at(per: dict, day: int) -> float:
    return per["age_T"] - (per["T"] - int(day)) / 365.25


# ------------------------------------------------------------------ what the case already shows
def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x or ""))
    return float(m.group()) if m else None


def _p3_rows(block: dict, T: int) -> list[tuple[str, int, float | None, bool]]:
    """(analyte, day, value, eGFR already printed) of the pack-3 laboratory rows."""
    from haenv.i18n.text import is_text
    from haenv.world.labworld.meds import axis
    names = {}
    for ax, analyte in (("TSH", "TSH"), ("HbA1c", "HbA1c"), ("LDL", "LDL")):
        disp = axis(ax)["display"]
        names[ax] = (disp, getattr(disp, "key", str(disp)), analyte)
    out = []
    for r in block.get("lab_results") or ():
        day, test, res = int(r["day"]), r.get("test"), r.get("result")
        if day > T:
            continue
        slots = getattr(res, "slots", None) or {}
        if is_text(test, "p3.lab.renal_panel"):
            text = str(res)
            m = re.search(r"(\d+(?:\.\d+)?)\s*μmol/L", text)
            cr = _num(slots["cr"]) if "cr" in slots else (_num(m.group(1)) if m else None)
            egfr = bool(str(slots.get("egfr") or "")) if slots else "eGFR" in text
            out.append(("creatinine", day, cr, egfr))
            k = _num(slots["k"]) if "k" in slots else None
            out.append(("potassium", day, k, False))
            continue
        value = _num(slots["value"]) if "value" in slots else _num(res)
        analyte = None
        for disp, key, name in names.values():
            if test == disp or is_text(test, key):
                analyte = name
        if analyte is None:
            for msg, name in (("p3.lab.creatinine", "creatinine"), ("p3.lab.potassium", "potassium"),
                              ("p3.lab.glucose", "fasting_glucose"), ("p3.lab.ck", "CK")):
                if is_text(test, msg):
                    analyte = name
        if analyte is None and str(test) in ("TSH", "ALT", "AST"):
            analyte = str(test)
        if analyte:
            out.append((analyte, day, value, False))
    return out


#: text-lab analyte of the invariants' reader (`validation.invariants.TEXT_LAB`) -> panel stream
TEXT_ANALYTES = {"K": "potassium", "HbA1c": "HbA1c", "fasting_glucose": "fasting_glucose"}


def stated_in_text(raw) -> set:
    """Panel analytes whose value (a number or a direction) the evidence ledger's free text already states,
    read in the language-free source form with the patterns the C3 invariant uses; such an analyte is shown on
    that surface, so the panels never draw it (a second, different value would contradict the text)."""
    from haenv.world.validation.invariants import TEXT_LAB
    out = set()
    for e in getattr(raw, "evidence_ledger", None) or ():
        if not isinstance(e, dict):
            continue
        t = " ".join(str(getattr(e.get(k), "key", None) or e.get(k) or "")
                     for k in ("symptom", "context", "note", "text", "claim"))
        for rx, an, _kind in TEXT_LAB:
            if an in TEXT_ANALYTES and re.search(rx, t):
                out.add(TEXT_ANALYTES[an])
    return out


def owned(raw, T: int) -> tuple[set, dict, set]:
    """Analytes the case already shows (streams, pack blocks, values stated in the ledger text), the pack's own creatinine and TSH draws
    `{analyte: {day: value}}`, and the days whose creatinine row already prints an eGFR."""
    ld = getattr(raw, "longitudinal_data", None) or {}
    pc = getattr(raw, "prediction_context", None) or {}
    analytes = set(registry()["analytes"])
    have = {k for k in ld if k in analytes}
    draws: dict[str, dict[int, float]] = {}
    printed: set[int] = set()
    fp = pc.get("followup_p4") or {}
    a = P4_ANALYTES.get(str(fp.get("analyte") or ""))
    if fp.get("analyte"):
        have.add(a or str(fp["analyte"]))
    for row in fp.get("report") or ():
        v = _num(row.get("value"))
        if a and v is not None and int(row["day"]) <= T:
            draws.setdefault(a, {})[int(row["day"])] = v
    have |= stated_in_text(raw)
    block = pc.get("meds_p3") or {}
    if block.get("lab_results"):
        for analyte, day, v, egfr in _p3_rows(block, T):
            have.add(analyte)
            if v is not None:
                draws.setdefault(analyte, {})[day] = v
            if egfr:
                printed.add(day)
    return have, draws, printed


# ------------------------------------------------------------------ calendar and ordering
def encounters(raw, ctx: dict | None, T: int, pack_days) -> list[int]:
    from haenv.world.clinical.med_course import history_visit_days
    days = {int(d) for d in (ctx or {}).get("visits") or () if int(d) <= T}
    hist = (ctx or {}).get("history") or {}
    if hist.get("long"):
        days |= set(history_visit_days(str(raw.case_id), int(hist["lo"])))
    days |= {int(d) for d in pack_days if int(d) <= T}
    return sorted(days)


def _eligible(per: dict, when: dict, day: int) -> bool:
    hay, age = per["hay"], age_at(per, day)
    if any(has_condition(hay, c) for c in when.get("no_condition") or ()):
        return False
    if "min_age" in when and age < float(when["min_age"]):
        return False
    if when.get("any_condition"):
        return (any(has_condition(hay, c) for c in when["any_condition"])
                or ("or_min_age" in when and age >= float(when["or_min_age"])))
    return True


def schedule(per: dict, days: list[int]) -> dict[int, list[str]]:
    """`{day: [rule, ...]}`: each rule fires at the first eligible encounter on or after its due day."""
    from haenv.world.generation import rng
    out: dict[int, list[str]] = {}
    if not days:
        return out
    for name, rule in registry()["rules"].items():
        step = int(rule["interval_days"])
        due = days[0]
        if rule["phase"] == "uniform":
            due = days[0] + int(rng.unit(per["cid"], "panels", "phase", name) * step)
        for d in days:
            if d < due or not _eligible(per, rule.get("when") or {}, d):
                continue
            out.setdefault(d, []).append(name)
            due = d + step
    return out


# ------------------------------------------------------------------ values
@functools.lru_cache(maxsize=1)
def _tables() -> dict:
    doc = load_registry("panel_background.yaml") or {}
    return {fid: {sx: tuple(float(x) for x in e[sx]["q"]) for sx in ("F", "M", "all") if sx in e}
            for fid, e in (doc.get("background") or {}).items()}


def table_value(item: str, sex: str | None, u: float) -> float:
    t = _tables()[item]
    q = t.get(sex or "all") or t["all"]
    x = min(max(u, 0.0), 1.0) * (len(q) - 1)
    i = min(int(x), len(q) - 2)
    return q[i] + (x - i) * (q[i + 1] - q[i])


#: panel stream -> its row in `registry/biological_variation.yaml`
BV_ANALYTE = {"creatinine": "creatinine", "potassium": "potassium", "fasting_glucose": "glucose", "HbA1c": "HbA1c",
              "LDL": "LDL", "triglycerides": "triglycerides", "ALT": "ALT", "AST": "AST", "TSH": "TSH", "FT4": "FT4"}
#: analytes whose person baseline is `truth.variation.person_baseline` (NHANES centre x EFLM CVG)
VARIATION_BASELINE = ("creatinine", "potassium")


def _cv(a: str) -> float:
    """Per-draw variation on the log scale: `truth.variation.draw_cv` (EFLM CVI and CVA)."""
    if registry()["analytes"][a].get("cv") != "biological_variation":
        raise ValueError(f"{a}: no per-draw variation registered")
    from haenv.world.truth import variation as V
    return V.draw_cv(BV_ANALYTE[a])


def _ndigits(a: str, cid: str) -> int:
    row = registry()["analytes"][a]
    if isinstance(row.get("ndigits"), dict):
        return int(row["ndigits"]["value"])
    if a in COHORT_ANALYTES:
        from haenv.world.generation.build_clinical import lab_print_ndigits
        return lab_print_ndigits(cid, a)
    import haenv.world.clinical.indicators as ind
    return int(ind.of(row["item"])["ndigits"])


def _limits(per: dict) -> dict[str, float]:
    """Upper limits a person without the condition stays below (`effects.yaml` background_constraints)."""
    from haenv.world.truth.derived import cr_at_egfr
    from haenv.world.truth.registry import effects
    out = {}
    for row in effects()["background_constraints"]:
        cond = "diabetes" if row["unless_world_condition"].lower() == "diabetes" else "ckd"
        if has_condition(per["hay"], cond):
            continue
        stream = {"HbA1c": "HbA1c", "FBG": "fasting_glucose", "Cr": "creatinine"}[row["observable"]]
        out[stream] = ("egfr", float(row["egfr_limit"])) if "egfr_limit" in row else ("value", float(row["limit"]))
    return out


def _constrain(per: dict, a: str, v: float, day: int, nd: int) -> float:
    from haenv.world.truth.derived import cr_at_egfr, reflect_below
    lim = _limits(per).get(a)
    if not lim:
        return v
    limit = cr_at_egfr(lim[1], age_at(per, day), per["sex"]) if lim[0] == "egfr" else lim[1]
    step = 10.0 ** -nd
    item = registry()["analytes"][a]["item"]
    floor = _tables()[item]["all"][0]
    v = reflect_below(v, limit, step, floor)
    r = round(v, nd)
    if r >= limit:
        r = round(reflect_below(r, limit, step, floor), nd)
    return r


class _Truth:
    """Noise-free truth of the panel analytes of one case."""

    def __init__(self, per: dict, ctx: dict | None):
        from haenv.world.generation import rng
        self.per, self.ctx, self.cid = per, ctx or {}, per["cid"]
        import haenv.world.clinical.indicators as ind
        cohort = [a for a in COHORT_ANALYTES if ind.cohort_of(a, per["disease"], per["comorbidities"])]
        try:
            self.base = dict(ind.sample_case(per["disease"], cohort, per["comorbidities"], case_id=self.cid))
        except ind.UmbrellaUnsatisfiable:
            self.base = {a: ind.sample_baseline(a, per["disease"], per["comorbidities"], case_id=self.cid)
                         for a in cohort}
        from haenv.world.truth import variation as V
        for a, row in registry()["analytes"].items():
            if a in self.base or "item" not in row or row.get("derived_from"):
                continue
            if a in VARIATION_BASELINE:
                self.base[a] = V.person_baseline(self.cid, BV_ANALYTE[a], per["sex"] or "")
            else:
                self.base[a] = table_value(row["item"], per["sex"], rng.unit(self.cid, "panels", "base", a))
        # B4: a person with hypothyroidism is an established levothyroxine user at a stable dose; the
        # general-population draw is the person's TSH set point and the steady state is drawn from the
        # sourced replacement states (`truth.thyroid.lt4_steady_tsh`)
        self.thyroid = None
        if "TSH" in self.base and has_condition(per["hay"], "hypothyroidism"):
            from haenv.world.truth.thyroid import lt4_steady_tsh
            state, tsh = lt4_steady_tsh(self.cid, self.base["TSH"])
            self.thyroid = {"etiology": "primary_autoimmune", "treatment": "levothyroxine_established",
                            "state": state, "set_point_tsh": round(self.base["TSH"], 4), "steady_tsh": round(tsh, 4)}
            self.base["TSH"] = tsh
        self.terms = {}

    def at(self, a: str, day: int) -> float:
        if a not in self.terms:
            fn = None
            drug_term = self.ctx.get("drug_term")
            if a in COHORT_ANALYTES and drug_term is not None:
                import haenv.world.clinical.indicators as ind
                from haenv.world.generation.build_clinical import CLINICAL_SPEC
                spec = {"base": self.base[a], "per_kg": CLINICAL_SPEC[a][0],
                        "cohort": ind.cohort_of(a, self.per["disease"], self.per["comorbidities"])}
                fn = drug_term(a, spec)
                self.base[a] = float(spec["base"])
            self.terms[a] = fn
        fn = self.terms[a]
        v = self.base[a] + (fn(int(day)) if fn is not None else 0.0)
        return max(v, 1e-6)


def _observe(per: dict, truth: _Truth, a: str, day: int) -> float:
    from haenv.world.generation.build_clinical import _z
    cid = per["cid"]
    nd = _ndigits(a, cid)
    v = truth.at(a, day) * math.exp(_cv(a) * _z(f"{cid}|panelz|{a}|{day}"))
    v = round(v, nd) if nd else float(round(v))
    return _constrain(per, a, v, day, nd)


def ft4_reading(cid: str, tsh: float, day: int) -> float:
    from haenv.world.generation.build_clinical import _z
    from haenv.world.truth.thyroid import ft4
    v = ft4(tsh, cid) * math.exp(_cv("FT4") * _z(f"{cid}|panelz|FT4|{day}"))
    return round(v, _ndigits("FT4", cid))


def _p4_tsh_truth(raw, T: int) -> dict[int, float] | None:
    """Bind P4's full biological records to its actual reported collections.

    The background source omits the active biological cause. P4's canonical
    four-decimal record truth includes it; printed TSH additionally includes
    TSH-only assay effects. Neither a nearby date nor a verification draw is
    a substitute for the original collection.
    """
    g = ((getattr(raw, "adjudication", None) or {}).get("followup_p4") or {})
    block = ((getattr(raw, "prediction_context", None) or {}).get("followup_p4") or {})
    if "background_source" not in g or "tsh" not in (g.get("analyte"), block.get("analyte")):
        return None
    from haenv.world.truth.dated import Source
    from haenv.world.labworld import analyte
    from haenv.world.labworld.observe import record_consistent
    def refuse(reason):
        raise ValueError("P4 paired TSH source: " + reason)
    if g.get("analyte") != "tsh":
        refuse("wrong record analyte")
    if "diversity_version" in g and (type(g["diversity_version"]) is not int or g["diversity_version"] != 1):
        refuse("unsupported item version")
    source = Source(g["background_source"])
    s = source.source
    if s.get("stream") != "TSH":
        refuse("wrong source stream")
    identities = (s.get("origin") or {}, s.get("medication") or {})
    if any(not isinstance(identity, dict) for identity in identities):
        refuse("invalid source case identity")
    owners = [identity["case_id"] for identity in identities if "case_id" in identity]
    if not owners or any(owner != str(raw.case_id) for owner in owners):
        refuse("missing or wrong source case")
    def date(value):
        if type(value) is not int:
            refuse("collection date must be an integer day")
        return value
    known = {}
    for q in s.get("known") or []:
        d, v = date(q["ts"]), float(q["value"])
        if d in known and known[d] != v:
            refuse(f"conflicting source points on day {d}")
        known[d] = v
    if block.get("analyte") != "tsh":
        refuse("wrong report analyte")
    def dated(rows):
        out = {}
        for r in rows:
            d = date(r["day"])
            if d <= T:
                if d in out:
                    refuse(f"ambiguous collection on day {d}")
                out[d] = r
        return out
    records, reports = dated(g.get("records") or []), dated(block.get("report") or [])
    if not records or records.keys() != reports.keys():
        refuse("missing original record/report day")
    cfg = analyte("tsh")
    out = {}
    for d, r in records.items():
        q = reports[d]
        clock = str(q.get("time") or "")
        if len(clock) == 5:
            clock += ":00"  # Legacy reports print hours/minutes of the same collection.
        if (q.get("analyte", block["analyte"]) != "tsh" or
                _num(q.get("value")) != r["printed"] or clock != r["time"]):
            refuse(f"measurement/report binding disagrees on day {d}")
        if record_consistent(r, int(cfg["ndigits"]), tuple(cfg["range"])):
            refuse(f"inconsistent measurement on day {d}")
        source(d)  # Require the saved physical source to cover this exact date.
        t = float(r["truth"])
        if not math.isfinite(t) or t <= 0:
            refuse(f"invalid biological parent on day {d}")
        out[d] = t
    return out


def egfr_reading(per: dict, cr: float, day: int) -> float:
    from haenv.world.truth.derived import egfr_ckd_epi_2021
    return float(round(egfr_ckd_epi_2021(cr, age_at(per, day), per["sex"]), _ndigits("eGFR", per["cid"])))


# ------------------------------------------------------------------ the layer
def _view_keeps(raw) -> frozenset | None:
    """Streams the pack's payload view can show: its `streams` and its `panel_streams` (None: all)."""
    from haenv.app.packs import owner
    P = owner(raw)
    V = P.view if P else None
    return None if V is None or V.streams is None else frozenset(V.streams) | frozenset(V.panel_streams)


def item_bearing(raw) -> frozenset:
    """A p3 item drug's own bearing streams: the decisive values stay in the visit block (`p3.gate` P-9),
    so a panel holding one of them stays withheld for that item."""
    ext = (((getattr(raw, "latent_premise", None) or {}).get("meta") or {}).get("_external") or {}).get("p3") or {}
    if not ext.get("drug"):
        return frozenset()
    from haenv.world.labworld.meds import drug
    return frozenset(drug(str(ext["drug"])).get("bearing_streams") or ())


def observe(raw, ctx: dict | None, T: int) -> dict | None:
    """Draw the case's routine panels; returns the audit record (None: not a panel pack)."""
    from haenv.app.packs import owner
    from haenv.world.generation.build_clinical import _draw_clock
    reg = registry()
    P = owner(raw)
    if P is None or P.name not in reg["packs"]:
        return None
    per = person(raw)
    cid = per["cid"]
    have, pack_draws, printed = owned(raw, T)
    p4_tsh = _p4_tsh_truth(raw, T) if P.name == "p4" else None
    pack_days = {d for series in pack_draws.values() for d in series}
    days = encounters(raw, ctx, T, pack_days)
    fired = schedule(per, days)
    thyroid_ok = not has_condition(per["hay"], "thyroid_unmodelled")
    truth = _Truth(per, ctx)
    streams: dict[str, dict[int, float]] = {}
    panel_of: dict[str, str] = {}
    for day in sorted(fired):
        panels = []
        for rule in fired[day]:
            panels += [p for p in reg["rules"][rule]["panels"] if p not in panels]
        for p in panels:
            row = reg["panels"][p]
            if p == "thyroid" and (not thyroid_ok or "TSH" in have):
                continue
            for a in row["members"]:
                if a in have or reg["analytes"][a].get("derived_from"):
                    continue
                streams.setdefault(a, {})[day] = _observe(per, truth, a, day)
                panel_of[a] = p
            if p == "thyroid" and day in streams.get("TSH", {}):
                streams.setdefault("FT4", {})[day] = ft4_reading(cid, truth.at("TSH", day), day)
                panel_of["FT4"] = p
            if p == "renal" and day in streams.get("creatinine", {}):
                streams.setdefault("eGFR", {})[day] = egfr_reading(per, streams["creatinine"][day], day)
                panel_of["eGFR"] = p
    # the laboratory prints eGFR with the pack's own creatinine and free T4 with its own TSH; pack 3's renal
    # rows follow its own print rule (eGFR beside creatinine only with CKD, A24), so there the eGFR stays on
    # the verifier side
    for day, cr in sorted((pack_draws.get("creatinine") or {}).items()):
        if day not in printed and cr:
            streams.setdefault("eGFR", {})[day] = egfr_reading(per, cr, day)
            panel_of.setdefault("eGFR", "pack_renal" if P.name == "p3" else "renal")
    if thyroid_ok:
        for day, tsh in sorted((pack_draws.get("TSH") or {}).items()):
            if tsh and tsh > 0:
                parent = p4_tsh[day] if p4_tsh is not None else tsh
                streams.setdefault("FT4", {})[day] = ft4_reading(cid, parent, day)
                panel_of.setdefault("FT4", "pack_thyroid")
    keep = _view_keeps(raw)
    bearing = item_bearing(raw)
    # FT4 beside the pack's TSH is shown unless FT4 itself bears on the item.
    # Source-backed P4 uses full TSH biology, with FT4's own independent draw.
    visible_panels = ({"pack_thyroid"} if "FT4" not in bearing and (keep is None or "FT4" in keep)
                      else set()) | {
        p for p, row in reg["panels"].items()
                      if not (set(row["members"]) | set(row.get("derived") or ())) & bearing
                      and (keep is None or all(a in keep for a in list(row["members"]) + list(row.get("derived") or ())))}
    ld = raw.longitudinal_data if raw.longitudinal_data is not None else {}
    withheld, visible = {}, []
    for a in sorted(streams):
        pts = [{"ts": d, "value": v, "time": _draw_clock(cid, d)} for d, v in sorted(streams[a].items())]
        if panel_of[a] in visible_panels:
            ld[a] = pts
            visible.append(a)
        else:
            withheld[a] = pts
    # a world stream of a panel analyte is shown with the panels when the view can show it and it is not the
    # item drug's bearing stream; otherwise its points up to T are kept with the withheld panels (same truth)
    for a in sorted(set(ld) & set(reg["analytes"]) - set(visible) - set(withheld)):
        if (keep is None or a in keep) and a not in bearing:
            visible.append(a)
        else:
            withheld[a] = [{k: q[k] for k in ("ts", "value", "time") if k in q} for q in ld[a] if int(q["ts"]) <= T]
    raw.longitudinal_data = ld
    raw.latent_premise[PANELS_KEY] = {"version": reg["version"], "age_years": round(per["age_T"], 4),
                                      "fired": [[d, fired[d]] for d in sorted(fired)],
                                      "visible": visible, "withheld": withheld,
                                      **({"thyroid": truth.thyroid} if truth.thyroid else {})}
    return {"encounters": len(days), "ordering_days": len(fired),
            "draws": {a: len(v) for a, v in sorted(streams.items())},
            "visible": visible, "withheld": sorted(withheld), "owned": sorted(have)}


# ------------------------------------------------------------------ consistency check
def world_streams(raw) -> dict[str, list[dict]]:
    """The case's panel analytes as the world holds them: visible streams plus the withheld record."""
    ld = getattr(raw, "longitudinal_data", None) or {}
    rec = ((getattr(raw, "latent_premise", None) or {}).get(PANELS_KEY) or {})
    out = {a: list(ld[a]) for a in registry()["analytes"] if a in ld}
    for a, pts in (rec.get("withheld") or {}).items():
        out.setdefault(a, [])
        seen = {int(q["ts"]) for q in out[a]}
        out[a] += [q for q in pts if int(q["ts"]) not in seen]
        out[a].sort(key=lambda q: int(q["ts"]))
    return out


def check(raw, T: int | None = None) -> list[str]:
    """Derived-pair violations of a built case: an eGFR without the same day's creatinine (panel or the
    pack's own) or off CKD-EPI 2021 by more than its print step, an FT4 without the same day's TSH, an FT4
    that does not fall as TSH rises across the case's draws. Empty for a coherent record."""
    rec = ((getattr(raw, "latent_premise", None) or {}).get(PANELS_KEY))
    if rec is None:
        return []
    T = int((getattr(raw, "prediction_context", None) or {}).get("prediction_time_T") or 0) if T is None else T
    per = person(raw)
    per["age_T"] = float(rec["age_years"])
    ws = world_streams(raw)
    _, pack_draws, _ = owned(raw, T)
    cr = {int(q["ts"]): float(q["value"]) for q in ws.get("creatinine") or ()}
    cr.update(pack_draws.get("creatinine") or {})
    tsh = {int(q["ts"]): float(q["value"]) for q in ws.get("TSH") or ()}
    tsh.update(pack_draws.get("TSH") or {})
    bad = []
    from haenv.app.packs import owner
    P = owner(raw)
    try:
        p4_tsh = _p4_tsh_truth(raw, T) if P is not None and P.name == "p4" else None
    except ValueError as exc:
        bad.append(f"FT4 source: {exc}")
        return bad
    for q in ws.get("eGFR") or ():
        d = int(q["ts"])
        if d not in cr:
            bad.append(f"eGFR@{d}: no creatinine that day")
        elif abs(egfr_reading(per, cr[d], d) - float(q["value"])) > 10.0 ** -_ndigits("eGFR", per["cid"]) + 1e-9:
            bad.append(f"eGFR@{d}: {q['value']} != CKD-EPI 2021 of creatinine {cr[d]}")
    pairs = []
    for q in ws.get("FT4") or ():
        d = int(q["ts"])
        if d not in tsh:
            bad.append(f"FT4@{d}: no TSH that day")
        else:
            if p4_tsh is not None and d not in p4_tsh:
                bad.append(f"FT4@{d}: no biological TSH source that day")
            else:
                pairs.append((p4_tsh[d] if p4_tsh is not None else tsh[d], float(q["value"])))
    from haenv.world.truth.thyroid import ft4
    a, _b, _s = __import__("haenv.world.truth.thyroid", fromlist=["_relation"])._relation()
    for t, f in pairs:
        # Both shocks are clipped at 3 SD (`build_clinical._z`). With a saved
        # biological parent only FT4's shock and print step remain; the legacy
        # measured-parent reference also needs the existing TSH uncertainty.
        if p4_tsh is not None and (not math.isfinite(f) or f <= 0):
            bad.append(f"FT4 {f} at TSH {t}: off the TSH-FT4 relation")
            continue
        elasticity = 1.0 / (math.log(10) * max(a - math.log10(t), 1e-6))
        bound = (3 * _cv("FT4") + (3 * _cv("TSH") * elasticity if p4_tsh is None else 0)
                 + 10.0 ** -_ndigits("FT4", per["cid"]) / f)
        if abs(math.log(f / ft4(t, per["cid"]))) > bound:
            bad.append(f"FT4 {f} at TSH {t}: off the TSH-FT4 relation")
    return bad
