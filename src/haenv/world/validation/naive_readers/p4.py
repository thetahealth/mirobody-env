"""Laboratory-change readers and candidate-rule input ablations.

SYNTHETIC data, evaluation only, not medical advice.
"""
from copy import deepcopy
from functools import partial
import math
from unittest.mock import patch

from haenv.p4 import diversity as D, diversity_reader as R, features as F
from haenv.p4.gold import ingested_interference
from haenv.p4.score import accepted_classes, next_ok_set
from haenv.world.labworld import cv_total, rcv95
from .core import Reader
from .decisive import FactOutcome

NAME = "p4"
FACTS = ("pre", "method", "true_causes", "exposure_course")
FACT_NOTE = "Change magnitude is read by every item and is not a knockout fact. Undetermined knockouts count as decisive."


def block(sp):
    return sp["prediction_context"]["followup_p4"]


def pair(answer):
    return answer["change_source"], answer["next_step"]


def full(sp):
    return pair(R.answer(block(sp)))


def ratio(sp):
    b = block(sp)
    r0, r1 = R.comparison(b)
    methods = ["jaffe" if F.method_family(r["method"]) == "jaffe" else None for r in (r0, r1)]
    return abs(math.log(float(r1["value"]) / float(r0["value"]))) / rcv95(
        cv_total(b["analyte"], methods[0]), cv_total(b["analyte"], methods[1]))


def printed_value_only(sp, threshold=1.0):
    return (("analytic_biological_noise", "no_action") if ratio(sp) < threshold else
            ("true_change", "act_on_change"))


def today_only(sp):
    b = deepcopy(block(sp))
    for section, rows in list(b.items()):
        if not isinstance(rows, list) or section in {"report", "laboratory_method_record", "specimen_procedure"}:
            continue
        b[section] = [r for r in rows if not isinstance(r, dict) or
                      r.get("draw_day", r.get("day", r.get("onset_day", b["today"]))) == b["today"]]
    return pair(R.answer(b))


def keyword_only(sp):
    flags = F.factor_flags(sp)
    first = next((key for key in F.FACTORS if flags[key]), None)
    if first is None:
        return printed_value_only(sp)
    legacy = dict.fromkeys(("glp1_dose_change", "weight_change", "acei_start_or_stop"), "true_change")
    legacy.update(dict.fromkeys(("postprandial_labelled_fasting", "delayed_processing", "meat_meal", "hemolysis_note"), "preanalytical"))
    legacy.update(dict.fromkeys(("poc_meter", "jaffe_enzymatic_switch", "lab_switch_zero_bias"), "method_difference"))
    cls = D.table()["rules"][first]["class"] if first in D.table()["rules"] else legacy[first]
    return cls, "act_on_change" if cls == "true_change" else "repeat_standardized"


READERS = (
    Reader("full", "full", full, "Production source and next-step reader."),
    Reader("printed_value_only", "stub", printed_value_only, "Printed ratio to paired RCV at threshold 1; no protocol."),
    Reader("printed_value_only@0.7", "stub", partial(printed_value_only, threshold=F.READ_NOISE_MAX), "RCV threshold sensitivity at the production noise boundary."),
    Reader("printed_value_only@1.3", "stub", partial(printed_value_only, threshold=F.READ_CHANGE_MIN), "RCV threshold sensitivity at the production change boundary."),
    Reader("today_only", "naive", today_only, "Today's rows plus comparison reports and method/specimen documents."),
    Reader("keyword_only", "stub", keyword_only, "First present factor, ignoring timing and direction; no protocol."),
)
NOT_APPLICABLE = {
    "latest_row_only": "Comparison requires two draws; today_only retains both reports.",
    "summary_sentence_only": "No summary-sentence reader in production.",
    "trust_system_score": "No machine score on laboratory-change items.",
    "checklist_only": "No structured symptom checklist.",
    "latest_dispense_pillbox": "No dispensing and pill-count decision.",
    "anomaly": "keyword_only is the factor-presence anomaly reader here.",
}


def gold(case):
    return case["adjudication"]["followup_p4"]


def ok(answer, g):
    return bool(answer is not None and answer[0] in accepted_classes(g) and answer[1] in next_ok_set(g))


def gold_label(g):
    return f"{g['class']}|{next_ok_set(g)[0]}"


def constant_answers(g):
    return {f"{c}|{n}" for c in accepted_classes(g) for n in next_ok_set(g)}


def answer_from_label(label):
    return tuple(label.split("|"))


def group_key(sp):
    return block(sp)["analyte"]


def metadata(sp):
    return {"ratio": ratio(sp)}


def features(sp):
    s = F.surface(sp, 0)
    naive = {k: s[k] for k in ("ratio", "sign", "interval")}
    naive.update({f"analyte:{group_key(sp)}": 1, **{f"factor:{k}": v for k, v in F.factor_flags(sp).items()}})
    return naive, {k: s[k] for k in ("n_hist", "n_med_rows", "n_notes", "block_chars", "payload_chars")}


def knockout(sp, full_answer):
    b, cfg = block(sp), D.table()
    an = b["analyte"]
    candidates = R.read_class(b)[1]
    result = []
    keys = list(dict.fromkeys(k for section in ("pre", "method", "true_causes") for k in cfg["analytes"][an][section]))
    for key in keys:
        hits = [c for c in candidates if c[2] == key]
        outside = not (any(c[1] == 1 for c in hits) and key in
                       cfg["analytes"][an]["pre"] + cfg["analytes"][an]["method"])
        if not hits:
            result.append(FactOutcome(key, "candidate", None, "same", present=False))
            continue
        cfg2 = deepcopy(cfg)
        for section in ("pre", "method", "true_causes"):
            cfg2["analytes"][an][section] = [k for k in cfg2["analytes"][an][section] if k != key]
        try:
            with patch.object(D, "table", lambda: cfg2):
                answer = pair(R.answer(b))
            outcome, error = ("flip" if answer != full_answer else "same"), None
        except ValueError as exc:
            outcome, error = "undetermined", f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            outcome, error = "error", f"{type(exc).__name__}: {exc}"
        result.append(FactOutcome(key, "candidate", outside, outcome, error=error))
    factor = R.answer(b)["factor"]
    if ingested_interference(factor):
        b2 = deepcopy(b)
        b2["medication_record"] = [r for r in b2.get("medication_record") or [] if "clean_followup" not in r]
        try:
            answer = pair(R.answer(b2))
            outcome, error = ("flip" if answer != full_answer else "same"), None
        except ValueError as exc:
            outcome, error = "undetermined", f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            outcome, error = "error", f"{type(exc).__name__}: {exc}"
        result.append(FactOutcome("exposure_course", "exposure_course", True, outcome, error=error))
    return result
