"""Bilingual projection of raw clinical samples, without generation targets or gold labels.

SYNTHETIC data, evaluation only, not medical advice.
"""
from haenv.i18n.text import L, T


def visible_rows(item):
    rows = []
    for r in item.get("control_measurements") or ():
        p = {k: v for k, v in r.items() if k != "truth"}
        if item["axis"] == "home_sbp":
            p["sample_note"] = L("p3.measurement.bp.sample")
        elif item["axis"] == "LDL":
            p["sample_note"] = L("p3.measurement.lipid.sample")
        else:
            p["sample_note"] = L("p3.measurement.tsh.sample")
        rows.append(p)
    return rows


def source_rows(b):
    """Place raw samples in the ordinary lab/device records, without a mean."""
    from haenv.p3.policy import identify_drug
    from haenv.world.labworld.meds import drug, axis
    name = drug(identify_drug(b["drug"]))["axis"]
    unit = axis(name)["unit"]
    return [{**{k: v for k, v in r.items() if k != "value"},
             "review_day": int(b["today"]), "test": L("p3.measurement.bp.test") if name == "home_sbp" else b["control_axis"],
             "result": T("p3.lab.result", value=r["value"], unit=unit)}
            for r in b["control_measurements"]]
