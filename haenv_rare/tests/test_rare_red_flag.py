"""Case-level red flag of rare cases: derived from the sampled phenotype, not only from the
condition. A proband-present term listed in `red_flag_hpo` makes the case red-flag / 🔴; the
same term negated or reported for a relative does not. Gate R11 refuses a case whose latent
disagrees, and every shipped rare job agrees with `case_triage`."""
import glob
import pathlib
import types

import pytest
import yaml

from haenv import rng
from haenv_rare.gen import case_triage, check_rare, load_specs, ontology_dir, sample_gold

ROOT = pathlib.Path(__file__).resolve().parents[2]

needs_ontology = pytest.mark.skipif(
    not (ontology_dir() / "hp.obo").exists(),
    reason=f"ontology files absent under {ontology_dir()} (set HAENV_RARE_ONTOLOGY or `rare.ontology_dir` in config.local.yaml); skipped, not passed")


def _term(hp, polarity="present", subject="proband"):
    return {"hpo_id": hp, "label": hp, "polarity": polarity, "subject": subject}


@needs_ontology
def test_proband_present_term_raises_the_case():
    spec = load_specs()["RD-MEN1"]
    assert (spec["red_flag"], spec["urgency"]) == (False, "🟡")
    assert case_triage(spec, [_term("HP:0003072")]) == (False, "🟡")
    assert case_triage(spec, [_term("HP:0003072"), _term("HP:0002248")]) == (True, "🔴")


@needs_ontology
def test_negated_or_family_term_does_not():
    spec = load_specs()["RD-DANON"]
    assert case_triage(spec, [_term("HP:0006543", polarity="absent")]) == (False, "🟡")
    assert case_triage(spec, [_term("HP:0006543", subject="relative")]) == (False, "🟡")


@needs_ontology
def test_descendant_term_counts():
    spec = load_specs()["RD-DRAVET"]
    assert spec["red_flag"] is False
    assert case_triage(spec, [_term("HP:0012847")]) == (True, "🔴")    # partial status epilepticus


@needs_ontology
@pytest.mark.parametrize("cid,want", [("JD-50", True), ("JD-67", True), ("JD-67v2", True), ("JD-71", False),
                                      ("JD-71v2", True), ("JD-63", False), ("JD-54", False)])
def test_sampled_cases(cid, want):
    specs = load_specs()
    spec = next(s for s in specs.values() if s["case_id"] == cid.split("v")[0])
    sex = rng.pick(["F", "M"], cid, "rare", "sex")          # as haenv_rare.gen_job does
    gm = (spec.get("modalities") or {}).get("genome")
    if gm and gm != "none" and gm.get("skeleton") == "trio":
        sex = "M"
    rf, urg = case_triage(spec, sample_gold(spec, cid, sex))
    assert rf is want and urg == ("🔴" if want else spec["urgency"])


@needs_ontology
def test_r11_refuses_a_stale_latent():
    spec = load_specs()["RD-DANON"]
    gold = [_term("HP:0006543")]
    sp = types.SimpleNamespace(evidence_ledger=[])

    def kinds(red_flag, urgency):
        cs = types.SimpleNamespace(latent={"rare_spec_id": "RD-DANON", "rare_hpo_gold": gold,
                                           "ddx_red_flag": red_flag, "ddx_urgency": urgency}, raw={})
        return {h["kind"] for h in check_rare(None, cs, sp, 84)}

    assert spec["red_flag"] is False
    assert "rare_red_flag_gold_mismatch" in kinds(False, "🟡")
    assert "rare_red_flag_gold_mismatch" in kinds(True, "🟡")
    assert "rare_red_flag_gold_mismatch" not in kinds(True, "🔴")


@needs_ontology
def test_shipped_rare_jobs_agree_with_case_triage():
    specs = load_specs()
    jobs = sorted(glob.glob(str(ROOT / "inputs" / "rare_coding-*.job.yaml")))
    assert jobs
    bad, n = [], 0
    for p in jobs:
        for c in yaml.safe_load(pathlib.Path(p).read_text(encoding="utf-8"))["cases"]:
            lat = c["latent"]
            n += 1
            if (lat["ddx_red_flag"], lat["ddx_urgency"]) != case_triage(specs[lat["rare_spec_id"]], lat["rare_hpo_gold"]):
                bad.append((pathlib.Path(p).name, c["case_id"]))
    assert n and not bad, bad
