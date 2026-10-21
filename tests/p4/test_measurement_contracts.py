"""Printed laboratory context and change bands agree with the measurement model."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
from haenv.p4 import realize as RZ
from haenv.world.labworld import cv_total, rcv95

DATA = json.loads(Path(__file__).with_name("measurements.json").read_text())


@pytest.mark.parametrize("case", DATA["contexts"], ids=lambda c: c["id"])
def test_reagent_context_agrees_with_the_recorded_lab_method(case, monkeypatch):
    monkeypatch.setattr(RZ, "BG_REAGENT_RATE", 1.0)
    monkeypatch.setattr(RZ, "BG_MEAT_DINNER_RATE", 0.0)
    monkeypatch.setattr(RZ, "jaffe_throughout", lambda *args: False)
    item = deepcopy(DATA["template"])
    if not case["jaffe"]:
        for row in item["records"]:
            row["perturbations"] = []
    RZ._place_context(item, case["case_id"], [], False)
    assert sum(bool((p.get("params") or {}).get("reagent_change"))
               for row in item["records"] for p in row["perturbations"]) == case["reagent_notes"]


@pytest.mark.parametrize("case", DATA["precision"], ids=lambda c: c["id"])
def test_change_band_exceeds_the_alternative_precision_control(case):
    low = RZ.change_band(case["analyte"])[0]
    registered = rcv95(cv_total(case["analyte"], case["m0"]), cv_total(case["analyte"], case["m1"]))
    alternate = 1.96 * math.sqrt(case["cvi"] ** 2 + case["cva0"] ** 2 + case["cvi"] ** 2 + case["cva1"] ** 2)
    assert low * registered / alternate >= 1.0
