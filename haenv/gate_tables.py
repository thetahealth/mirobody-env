"""Clinical tables the gates judge against and the generators sample from: weekly dosing
frequency per drug and the words that are anatomically exclusive to one sex.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations



# Administrations per week by drug (weekly injectable = 1, oral daily = 7).
DRUG_DOSES_PER_WEEK: dict[str, int] = {
    "metformin": 7, "semaglutide": 1, "tirzepatide": 1, "liraglutide": 7, "dulaglutide": 1,
    # Indications and dose ladders are in `registry/drug_indications.yaml`.
    "amlodipine": 7, "atorvastatin": 7,
}


# Anatomically exclusive words only, not epidemiological tendencies.
SEX_EXCLUSIVE: dict[str, tuple[str, ...]] = {
    "M": ("月经", "经期", "怀孕", "妊娠", "宫颈", "子宫", "卵巢", "阴道", "绝经", "痛经", "排卵"),
    "F": ("前列腺", "睾丸", "阴茎", "精液", "遗精"),
}


# GEN25: ledger `source_type` -> streams its value must be found on. Separate from
# `DEVICE_SIGNALS`: the kernel emits weight EVs under `wearable`.
LEDGER_VALUE_STREAMS: dict[str, set[str]] = {
    "wearable": {"weight"},  # the kernel's weight-EV convention
    "smart_scale": {"weight"},
    "clinic_scale": {"weight_ref", "weight"},
    "cgm": {"CGM_TIR", "fasting_glucose"},
    "bp_cuff": {"systolic_bp", "diastolic_bp"},
    "lab_panel": {"HbA1c", "LDL", "ALT", "AST", "FIB4",
                  "triglycerides", "fasting_glucose"},
    # Values of these types live in text, so they have no numeric object here.
    "lab_result": set(),
    "patient_reported_symptom": set(),
    "patient_reported_context": set(),
}
