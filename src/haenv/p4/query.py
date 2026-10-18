"""P4 source retrieval: the visit header shows only today's laboratory facts.

The reader identifies the previous report from dated records. Laboratory notices,
specimen procedures, meals, medicines and encounters remain separate sources.
Interpretation knowledge is retrieved from the protocol catalog. Verifier lookups
are private and retain the original gold-side dates for acquisition profiling.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from haenv.interaction.query.adapter import KeyLookup
from haenv.interaction.query.adapter import QueryAdapter
from haenv.interaction.query.adapter import register

_HEAD = ("encounter_type", "today_date", "today", "patient_quote", "analyte", "name", "unit", "reference_range")


def _method_source_side(block, reports):
    """Resolve the non-reference draw from applicable visible laboratory notices."""
    def family(method):
        if "jaffe" in str(method).lower():
            return "jaffe"
        if "酶法" in str(method):
            return "enzymatic"
        raise ValueError("unrecognised comparison reference method")

    references = []
    for report in reports:
        day = int(report["day"])
        applicable = [n for n in block.get("laboratory_method_record") or ()
                      if n.get("scope") == block["analyte"] and
                      n.get("lab") == report.get("lab") and n.get("method") == report.get("method") and
                      int(n["day"]) <= day and int(n["valid_from"]) <= day]
        if not applicable:
            raise ValueError("no applicable comparison reference notice")
        notice = max(applicable, key=lambda n: (int(n["valid_from"]), int(n["day"])))
        if not notice.get("comparison_reference_method"):
            raise ValueError("comparison reference method is not recorded")
        references.append(family(notice["comparison_reference_method"]))
    if len(set(references)) != 1:
        raise ValueError("comparison reference methods disagree")
    non_reference = [i for i, r in enumerate(reports) if family(r["method"]) != references[0]]
    if len(non_reference) != 1:
        raise ValueError("comparison reference does not identify one source side")
    return -1 if non_reference[0] == 0 else 1


class P4Adapter(QueryAdapter):
    def audit_block(self, block):
        from haenv.p4.records import clinical_view
        return {**clinical_view(block), **block}

    def extra_specs(self, store):
        from haenv.p4.protocols import supplement
        supplement(store)
        return []

    def profile_header(self, profile: dict) -> dict:
        return {k: v for k, v in profile.items() if k != "known_conditions"}

    def header(self, block: dict) -> dict:
        out = {k: block[k] for k in _HEAD if k in block}
        reports = [r for r in block.get("report") or [] if int(r["day"]) == int(block["today"])]
        if len(reports) != 1:
            raise ValueError("today's laboratory report is missing or ambiguous")
        out["values"] = {"today": reports[0]["value"]}
        remarks = [r["text"] for r in block.get("patient_remarks") or []
                   if int(r["draw_day"]) == int(block["today"])]
        if remarks:
            out["patient_quote"] = "；".join(remarks)
        return out

    def lab_label(self, block: dict, section: str) -> str:
        return f"{block.get('analyte')} {block.get('name')}" if section == "report" else section

    def guard_terms(self, vp):
        from haenv.p4.gate import EN_TOKENS
        from haenv.p4.gate import ZH_TOKENS
        from haenv.p4.gate import gold_numbers
        g = (getattr(vp, "adjudication", None) or {}).get("followup_p4") or {}
        return EN_TOKENS + ZH_TOKENS, tuple(gold_numbers(g))

    def key_lookups(self, vp, store):
        out = self._decision_lookups(vp, store) + self.protocol_lookups(store)
        for s in ("laboratory_method_record", "specimen_procedure"):
            if s in store.block:
                out.append(KeyLookup(s, lambda _t, r, s=s: r.get("_from") == s))
        return out

    def _decision_lookups(self, vp, store):
        g = (getattr(vp, "adjudication", None) or {}).get("followup_p4") or {}
        if "class" not in g:
            return []
        t0, t1 = int(g["t0"]), int(g["t1"])
        cause = next((int(r["day"]) for r in g.get("records") or ()
                      for p in r.get("perturbations") or () if p.get("role") == "cause"), t1)
        factor = g.get("factor") or ""

        def row(section, day=None, lo=None, hi=None):
            def seen(_tool, r):
                if r.get("_from") != section:
                    return False
                d = r.get("day", r.get("draw_day"))
                if day is not None:
                    return d is not None and int(d) == day
                return lo is None or (d is not None and lo < int(d) <= hi)
            return seen
        if factor in ("delayed_processing", "poc_meter", "jaffe_enzymatic_switch", "hemolysis_note",
                      "lab_switch_zero_bias", "tube_type"):
            return [KeyLookup(f"report@{cause}", row("report", cause))]
        if factor in ("meat_meal", "postprandial_labelled_fasting"):
            return [KeyLookup(f"diet_record@{cause}", row("diet_record", cause))]
        if factor in ("acei_start_or_stop", "glp1_dose_change"):
            return [KeyLookup(f"medication_record@({t0},{t1}]", row("medication_record", lo=t0, hi=t1))]
        if factor == "weight_change":
            return [KeyLookup("weight", lambda _t, r: r.get("_from") == "weight")]
        return [KeyLookup(f"report@{t0}", row("report", t0)), KeyLookup(f"report@{t1}", row("report", t1))]

    def read_answer(self, sp) -> dict:
        from types import SimpleNamespace
        from haenv.p4.records import clinical_view
        block = dict(sp.prediction_context[self.block])
        from haenv.p4.diversity_reader import applies
        from haenv.p4.diversity_reader import answer
        if "today" not in block:
            # A visit can review a completed report/verification series.
            # Rebuild its comparison date from the retrieved routine reports.
            block["today"] = max(int(r["day"]) for r in block["report"])
        if "analyte" not in block:
            current = [r for r in block.get("report") or [] if int(r["day"]) == int(block["today"])]
            if len(current) != 1:
                raise ValueError("today's report does not identify the analyte")
            if current[0].get("analyte"):
                block["analyte"] = current[0]["analyte"]
            else:
                scopes = {r["scope"] for section in ("specimen_procedure", "laboratory_method_record")
                          for r in block.get(section) or [] if r.get("scope")}
                if len(scopes) != 1:
                    raise ValueError("laboratory documents do not identify one analyte")
                block["analyte"] = scopes.pop()
        if applies(block):
            return answer(block)
        sp = SimpleNamespace(**{**vars(sp), "prediction_context": {
            **sp.prediction_context, self.block: clinical_view(block)}})
        from haenv.p4.features import compared
        from haenv.p4.features import visible_class
        cls, cands = visible_class(sp)
        r0, r1 = compared(sp)
        obs = round(100.0 * (math.exp(math.log(float(r1["value"]) / float(r0["value"]))) - 1.0), 1)
        if cls not in ("true_change", "preanalytical", "method_difference"):
            return {"change_source": "analytic_biological_noise", "factor": None, "true_change_pct": 0.0,
                    "next_step": "no_action", "reason": "变化在两次检测的波动范围内"}
        factor = next((c[2] for c in cands if c[0] == cls), None)
        b = sp.prediction_context["followup_p4"]
        action_candidates = cands
        if cls == "method_difference" and b.get("interpretation_rule"):
            # visible_class supplies an observed direction. The action policy
            # requires the draw carrying the non-reference method instead.
            side = _method_source_side(b, (r0, r1))
            action_candidates = [(c, side if c == cls else direction, f) for c, direction, f in cands]
        from haenv.world.protocols.followup import visible_step
        return {"change_source": cls, "factor": factor, "true_change_pct": obs if cls == "true_change" else 0.0,
                "next_step": visible_step(sp, cls, action_candidates),
                "reason": "按记录中与变化方向一致的因素"}

    def blind_answer(self, payload=None) -> dict:
        return {"change_source": "analytic_biological_noise", "factor": None, "true_change_pct": 0.0,
                "next_step": "no_action", "reason": "未查看记录"}


ADAPTER = register(P4Adapter(
    pack="p4", block="followup_p4", header_keys=_HEAD,
    not_patient_data=("text_source", "instruction", "today_date", "encounter_type", "rules_delivery",
                      "comparison_basis", "interpretation_rule", "previous_draw_day", "today",
                      "analyte", "name", "unit", "reference_range"),
    fhir_types={"report": "Observation", "diet_record": "DocumentReference", "medication_record": "MedicationStatement",
                "event_history_record": "DocumentReference", "laboratory_method_record": "DocumentReference", "specimen_procedure": "DocumentReference"},
    sections={
        "report": ("labs", "检验报告，按采样日期"),
        "diet_record": ("record", "饮食记录，按登记日期"),
        "specimen_policy": ("record", "检验科印出的标本稳定性规定"),
        "event_history_record": ("record", "门诊病程与患者自述记录，按就诊日期"),
        "outpatient_record": ("record", "门诊复查病程记录，按就诊日期"),
        "laboratory_method_record": ("record", "检验科方法通知与使用记录，按日期"),
        "specimen_procedure": ("record", "检验科标本规程，按发布日期"),
        "patient_remarks": ("notes", "患者自述，按登记日期"),
        "medication_record": ("meds", "用药记录，按处方日期"),
        "known_comorbidity": ("profile", "已知合并症"),
    }))
