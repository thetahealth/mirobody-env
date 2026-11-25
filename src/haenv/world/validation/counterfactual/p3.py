"""P3 adapter: pure rule-block operators and production rendering."""
import copy
import math
from functools import lru_cache

from .batch import InvalidTwin, Operator, load_batch
from .trace import RuleTrace

from haenv.p3.gold import g3, accepted, dose_at, historical_limit, adherence_of
from haenv.p3.render import render_block
from haenv.p3.features import rule_reader
from haenv.p3.history import acei_contraindicated, limiting_record
from haenv.p3.interactions import _control
from haenv.p3.policy import absorption_interaction, specification
from haenv.world.labworld.meds import drug as drug_of, axis, band, rule
from haenv.i18n.text import L, is_text


def render(g):
    r = render_block(g)
    if g.get("record_format") == "emr":
        from haenv.p3.emr import source_block
        r = source_block(r, g["current_medications"], g["patient_quote"], reconciliation=g)
    return r


def known_of(g):
    c = g.get("comorbidity")
    return (str(c["code"]),) if isinstance(c, dict) and c.get("code") else ()


def live_symptoms(g):
    rec = specification()["symptom_course"]["recurrent_event_drugs"]
    return [e for e in g["diary"] if e.get("kind") == "drug_symptom" and e.get("cause") != "predating"
            and (g["drug"] in rec or not is_text(e.get("status"), "p3.status.resolved"))]


# ---------------------------------------------------------------- operators: g -> g' or None (not applicable)
def op_control_flip(g):
    o = g3(g)
    if o.get("control") not in ("on", "off", "over"):
        return None
    h = copy.deepcopy(g)
    _control(h, "off" if o["control"] == "on" else "on")
    return h


def op_coverage_full(g):
    if adherence_of(g) >= float(rule()["adherence_line"]):
        return None
    b = copy.deepcopy(g)
    start = min(int(d) for d, _ in b["dose_line"])
    end, supply = int(b["T"]), int(b.get("supply", 30))
    resets = sorted({start, *(int(d) for d, _ in b["dose_line"])})
    picks = []
    for i, day in enumerate(resets):
        stop = resets[i + 1] if i + 1 < len(resets) else end + 1
        picks += list(range(day, stop, supply))
    if picks[-1] == end:
        picks.pop()
    b["refills"] = picks
    b["pill_count"] = supply - (end - picks[-1])
    return b


def op_admin_separate(g):
    if not absorption_interaction(g.get("administration_record")):
        return None
    b = copy.deepcopy(g)
    b["administration_record"]["supplement_hour"] = b["administration_record"]["drug_hour"] + 5
    return b


def op_remove_limit(g):
    x = dose_at(g["dose_line"], g["T"])
    if not historical_limit(g, x):
        return None
    b = copy.deepcopy(g)
    b["adverse_event_record"] = [r for r in b.get("adverse_event_record", ()) if not limiting_record([r], b["drug"], x)]
    return b


def op_acei_remove(g):
    if not acei_contraindicated(g.get("adverse_event_record"), g["drug"]):
        return None
    b = copy.deepcopy(g)
    b["adverse_event_record"] = [r for r in b["adverse_event_record"] if not acei_contraindicated([r], b["drug"])]
    return b


def op_headroom(g):
    lad = drug_of(g["drug"], known_of(g))["ladder"]
    b = copy.deepcopy(g)
    x = float(dose_at(b["dose_line"], b["T"]))
    b["dose_line"][-1][1] = lad[-2] if x == float(lad[-1]) else lad[-1]
    return b


def op_symptom_resolve(g):
    live = live_symptoms(g)
    if not live:
        return None
    b = copy.deepcopy(g)
    ids = {id(e) for e in live}
    for e0, e in zip(g["diary"], b["diary"]):
        if id(e0) in ids:
            e["status"] = L("p3.status.resolved")
    if g["drug"] in specification()["symptom_course"]["recurrent_event_drugs"]:
        return None  # resolved still acts for recurrent-event drugs: no clean course operator
    return b


def op_onset_predating(g):
    live = live_symptoms(g)
    if not live:
        return None
    b = copy.deepcopy(g)
    ids = {id(e) for e in live}
    start = min(int(d) for d, _ in b["dose_line"])
    for e0, e in zip(g["diary"], b["diary"]):
        if id(e0) in ids:
            e.update(onset=start - 30, cause="predating")
    return b


def op_interval_flip(g):
    """Move the last dose increase so that 'just stepped up' flips (off items only)."""
    o = g3(g)
    if o.get("control") != "off":
        return None
    tau = float(drug_of(g["drug"], known_of(g))["tau_rule_days"])
    T = int(g["T"])
    line = g["dose_line"]
    if len(line) < 1:
        return None
    b = copy.deepcopy(g)
    prev = int(line[-2][0]) if len(line) >= 2 else -10 ** 6
    new = T - int(tau) - 2 if "early" in o["states"] else T - max(1, int(0.25 * tau))
    if not prev < new <= T:
        return None
    b["dose_line"][-1][0] = new
    return b


def op_comorbidity_remove(g):
    if not g.get("comorbidity"):
        return None
    b = copy.deepcopy(g)
    b.pop("comorbidity")
    return b


# negative controls: facts the gold rule does not read for the decision
def op_drop_params(g):
    if not g.get("params"):
        return None
    b = copy.deepcopy(g)
    b.pop("params")
    return b


def op_perturb_first_visit(g):
    T = int(g["T"])
    rows = [r for r in g["visits"] if int(r["day"]) < T]
    if len(rows) < 2:
        return None
    b = copy.deepcopy(g)
    a = axis(b["axis"])
    r = min((r for r in b["visits"] if int(r["day"]) < T), key=lambda r: int(r["day"]))
    v = float(r["truth"]) * math.exp(0.5 * band(b["axis"]))
    r.update(truth=v, printed=round(v, int(a["ndigits"])))
    return b


def op_drop_confounders(g):
    if not any(e.get("kind") == "confounder" for e in g["diary"]):
        return None
    b = copy.deepcopy(g)
    b["diary"] = [e for e in b["diary"] if e.get("kind") != "confounder"]
    return b


def op_drop_inert_symptoms(g):
    live = {id(e) for e in live_symptoms(g)}
    inert = [e for e in g["diary"] if e.get("kind") == "drug_symptom" and id(e) not in live]
    if not inert:
        return None
    b = copy.deepcopy(g)
    keep = [i for i, e in enumerate(g["diary"]) if not (e.get("kind") == "drug_symptom" and id(e) not in live)]
    b["diary"] = [b["diary"][i] for i in keep]
    return b


_OPS = {"control_flip": op_control_flip, "coverage_full": op_coverage_full, "admin_separate": op_admin_separate,
       "remove_limit": op_remove_limit, "acei_remove": op_acei_remove, "headroom": op_headroom,
       "symptom_resolve": op_symptom_resolve, "onset_predating": op_onset_predating,
       "interval_flip": op_interval_flip, "comorbidity_remove": op_comorbidity_remove,
       "drop_params": op_drop_params, "perturb_first_visit": op_perturb_first_visit,
       "drop_confounders": op_drop_confounders, "drop_inert_symptoms": op_drop_inert_symptoms}


def _record_operator(name, fn):
    """Project exact source aliases, without claiming a new physiological course.

    R edits hold other measurements/premises fixed. The saved kinetic source is
    evidence for identifying aliases, never a model to refit to the edited record.
    Inputs without that source retain the legacy operator path exactly.
    """
    def intervene(g):
        source = g.get("medication_course")
        if source is None:
            return fn(g)
        from haenv.p3.history import connect_course, _lt4_archive_rows
        from haenv.p3.realize import Unrealizable
        # Validate BEFORE editing: equality after a rewrite cannot establish that
        # independent, conflicting evidence was an alias of the original fact.
        try:
            connect_course(copy.deepcopy(g))
        except Unrealizable as exc:
            raise InvalidTwin(str(exc)) from exc
        b = fn(g)
        if b is None:
            return None
        if name != "interval_flip":
            for row in _lt4_archive_rows(b):
                for rx in row.get("prescriptions") or ():
                    value = dose_at(b["dose_line"], int(rx["day"]))
                    if rx["dose_mg"] != value:
                        rx["dose_mg"] = value
            visits = {int(v["day"]): v for v in b["visits"]}
            for row in b.get("adverse_event_record") or ():
                labs = list(row.get("labs") or ())
                if "tsh_mIU_L" in row:
                    # Legacy flat labs are aliases too, including other-drug notes.
                    value = visits[int(row["day"])]["printed"]
                    if row["tsh_mIU_L"] != value:
                        row["tsh_mIU_L"] = value
                for lab in labs:
                    if "tsh_mIU_L" in lab:
                        value = visits[int(lab["day"])]["printed"]
                        if lab["tsh_mIU_L"] != value:
                            lab["tsh_mIU_L"] = value
        # The original physiology remains replayable under explicit provenance.
        # A local edit must not masquerade as a re-admissible canonical course.
        b.pop("medication_course")
        b["rule_record_intervention"] = dict(operator=name, scope="rule_record",
            physical_rollout=False, original_medication_course=copy.deepcopy(source),
            limitation=("Interval date edit is W-diagnostic only; source reconstruction "
                        "and physical rollout are not repaired." if name == "interval_flip"
                        else "Physical reconstruction requires a W intervention."))
        return b
    return intervene


def _control_target(g):
    return "p3.control_over" if g3(g).get("control") == "over" else "p3.control_off"


def _symptom_target(g):
    return "p3.symptom_unattributed" if g3(g).get("ae") == "unattributed" else "p3.symptom_dose_related"


@lru_cache(maxsize=1)
def trace():
    return RuleTrace(g3, label="p3.g3")


class P3:
    pack = "p3"

    def items(self, batch):
        return load_batch(batch)

    def rule(self):
        return trace()

    def rule_args(self, item):
        return (self.block(item.raw),)

    def block(self, raw):
        return raw if isinstance(raw, dict) else raw.adjudication["meds_p3"]

    def decision(self, result):
        return result["class"]

    def scored(self, raw_like):
        g = self.block(raw_like)
        result = g3(g)
        return sorted(accepted(g, result)) if result["class"] is not None else None

    def visible_answer(self, sp):
        return rule_reader(sp)

    def operators(self):
        targets = dict(control_flip=_control_target, coverage_full="p3.adherence_low", admin_separate="p3.absorption",
            remove_limit="p3.limit", acei_remove="p3.acei_contraindication", headroom="p3.ladder_top",
            symptom_resolve=_symptom_target, onset_predating=_symptom_target, interval_flip="p3.early",
            comorbidity_remove="p3.on_target_titration")
        return {name: Operator(name, targets.get(name), "W-diagnostic" if name == "interval_flip" else "R",
                               _record_operator(name, fn))
                for name, fn in _OPS.items()}

    def scored_metadata(self, item, twin, outcome):
        provenance = twin.get("rule_record_intervention")
        if not provenance:
            return {}
        import hashlib
        import json
        source = provenance["original_medication_course"]
        return {"rule_record_intervention": {
            **{k: v for k, v in provenance.items() if k != "original_medication_course"},
            "original_source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()}}

    def twin_payload(self, item, twin):
        sp = copy.deepcopy(item.sp)
        sp["prediction_context"]["meds_p3"] = render(twin)
        if self.block(item.raw).get("comorbidity") and not twin.get("comorbidity"):
            from haenv.world.labworld.meds import meds
            reg = meds().get("comorbidities") or {}
            up = sp.get("user_profile") or {}
            up["known_conditions"] = [k for k in up.get("known_conditions") or () if k not in reg]
        return sp

    def visible_blocks(self, sp):
        blocks = dict(sp["prediction_context"]["meds_p3"])
        blocks["user_profile"] = sp.get("user_profile") or {}
        return blocks

    def baseline_payload(self, item):
        return self.twin_payload(item, self.block(item.raw))

    def metadata(self, item, result):
        g = self.block(item.raw)
        from haenv.p3.gold import STATES
        return {"subtype": g.get("subtype"), "drug": g["drug"],
                "declared_states": sorted(STATES.get(g.get("subtype"), ())),
                "stored_gold_equal": result["class"] == g.get("class")}


Adapter = P3
