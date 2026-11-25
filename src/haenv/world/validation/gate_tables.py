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
    # DailyMed glimepiride, setid 28eefc95-d92e-4555-b6c5-2933860b0610, 2.1:
    # "The recommended starting dose of glimepiride tablets is 1 mg or 2 mg once daily."
    "glimepiride": 7,
    # DailyMed LANTUS, setid ae27bb66-5dbe-4f01-a89e-f4fa7cb09c99, 2.2:
    # "Administer LANTUS subcutaneously once daily at any time of day but at the same time every day."
    "insulin_glargine": 7,
    # DailyMed JANUVIA, setid f85a48d0-0407-4c50-b0fa-7673a160bf01, 2.1:
    # "The recommended dose of JANUVIA is 100 mg once daily."
    "sitagliptin": 7,
    # DailyMed FARXIGA, setid 01f90c94-71cb-4a1f-81ff-8004b850529b, 2.2:
    # "For additional glycemic control, the dosage can be increased to 10 mg orally once daily."
    "dapagliflozin": 7,
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
