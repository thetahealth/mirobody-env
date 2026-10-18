"""Controls for lab audits that must respect the measurement method."""
import copy

import haenv.world.validation.invariants as inv
from haenv.world.observations.findings_render import tc_friedewald_residual


def _lipid_item():
    # A fixed, reproducible positive control whose direct-LDL residual exceeds
    # the old 15% Friedewald equality tolerance. This is a method effect, not a
    # damaged sum or a change to the patient's diagnosis.
    cid = "JD-02v6"
    ldl, hdl, tg = 3.5, 1.2, 1.5
    rho = tc_friedewald_residual(cid, ldl, tg)
    tc = round(ldl + hdl + tg / 2.2 - rho * ldl, 2)
    values = {"LDL": ldl, "HDL": hdl, "TG": tg, "TC": tc}
    return {
        "pack": "m2", "case_id": cid, "case": {},
        "sp": {"prediction_context": {"prediction_time_T": 100},
               "longitudinal_data": {}, "user_profile": {"age_range": "50-54"}},
        "buy": {"world_ld": {}, "results": {
            "lipids": {"observation": [
                {"finding_id": name, "value": value, "ts": 100}
                for name, value in values.items()
            ]}
        }},
    }


def test_direct_ldl_residual_is_not_a_broken_cholesterol_sum():
    item = _lipid_item()
    values = {x["finding_id"]: x["value"]
              for x in item["buy"]["results"]["lipids"]["observation"]}
    estimated = values["LDL"] + values["HDL"] + values["TG"] / 2.2
    assert abs(values["TC"] - estimated) > 0.15 * estimated
    assert inv.c8(item) == []


def test_damaged_tc_still_fails_with_direct_ldl():
    item = copy.deepcopy(_lipid_item())
    for row in item["buy"]["results"]["lipids"]["observation"]:
        if row["finding_id"] == "TC":
            row["value"] *= 2
    assert inv.c8(item)


def _glucose_item(streams, results=None):
    return {"pack": "m2", "case_id": "glucose-screen-control", "case": {},
            "sp": {"prediction_context": {"prediction_time_T": 100}, "user_profile": {}},
            "buy": {"world_ld": streams, "results": results or {}}}


def test_glucose_screen_uses_the_median_not_the_upper_middle():
    item = _glucose_item({"HbA1c": [{"ts": 100, "value": 5.5}],
                          "fasting_glucose": [{"ts": 95, "value": 4.0}, {"ts": 100, "value": 8.0}]})
    assert inv.v4(item) == []


def test_repeated_display_of_a_draw_is_not_a_second_confirmation():
    item = _glucose_item({"fasting_glucose": [{"ts": 100, "value": 7.1}]},
                         {"glucose": {"observation": [{"finding_id": "FBG", "ts": 100, "value": 7.1}]}})
    assert inv.v6(item) == []
    item["buy"]["world_ld"]["fasting_glucose"].append({"ts": 90, "value": 7.1})
    assert inv.v6(item)


def test_statistical_flags_remain_reported_without_claiming_impossibility():
    item = _glucose_item({"HbA1c": [{"ts": 100, "value": 5.5}],
                          "fasting_glucose": [{"ts": 100, "value": 10.0}]})
    report = inv.run([item], "m2", gates=[g for g in inv.GATES if g.id == "V4"])
    assert report["V4"]["pool"] == 1
    assert report["V4"]["evidence_kind"] == "screening"
