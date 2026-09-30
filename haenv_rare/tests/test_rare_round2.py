"""Round 2 allocation (`haenv_rare.round2.allocate`): the invariants the round promises, checked
on the real staged data (skipped where it is absent). Nothing is written."""
import collections

import pytest

from haenv_rare import genome_os as GO
from haenv_rare.gen import ontology_dir

# allocation samples gold terms, so it needs the HPO / Orphanet files as well as the staged data
pytestmark = [pytest.mark.skipif(not GO.rare_os_root().exists(),
                                 reason=f"{GO.rare_os_root()} absent: skipped, not passed"),
              pytest.mark.skipif(not (ontology_dir() / "hp.obo").exists(),
                                 reason=f"ontology files absent under {ontology_dir()} (set HAENV_RARE_ONTOLOGY); "
                                        "skipped, not passed")]


@pytest.fixture(scope="module")
def plans():
    from haenv_rare.round2 import allocate
    return allocate()


def test_sixty_new_ids_split_twenty_twenty_twenty(plans):
    ids = [p["case_id"] for p in plans]
    assert ids == [f"JD-{100 + i}" for i in range(60)]
    assert collections.Counter(p["lang"] for p in plans) == {"both": 20, "zh": 20, "en": 20}


def test_each_eeg_recording_goes_to_one_case_and_sets_sex_and_age(plans):
    seen = collections.Counter()
    for p in plans:
        for r in p.get("eeg") or []:
            seen[r["meta"]["file"]] += 1
            if r["kind"] != "control":
                meta = r["meta"]
                assert p["sex"] == meta["patient_sex"] and p["age"] == float(meta["patient_age_years"])
    assert seen and max(seen.values()) == 1
    chb = {r["meta"]["patient"] for p in plans for r in p.get("eeg") or [] if r["dataset"] == "chbmit"}
    assert len(chb) == 16                                       # every CHB-MIT patient, once


def test_about_one_in_ten_recordings_tests_de_identification(plans):
    recs = [r for p in plans for r in p.get("eeg") or []]
    n_phi = sum(r["phi"] for r in recs)
    assert 0.05 <= n_phi / len(recs) <= 0.15


def test_sex_pinned_specs_and_x_linked_males(plans):
    from haenv_rare.gen import load_specs
    specs = load_specs()
    for p in plans:
        s = specs[p["spec_id"]]
        if s.get("sex"):
            assert p["sex"] == s["sex"], p["case_id"]
        elif s["inheritance"] == "XLR":
            assert p["sex"] == "M", p["case_id"]


def test_backgrounds_follow_language_and_sex(plans):
    for p in plans:
        bg = p.get("background")
        if not bg:
            continue
        assert bg["child_sex"] == p["sex"]
        assert (bg["superpopulation"] == "EAS") == (p["lang"] in ("both", "zh")), p["case_id"]


def test_imaging_patients_are_not_reused_and_exact_pairs_match_histology(plans):
    used = [(p["imaging"]["collection"], p["imaging"]["tcia_patient_id"]) for p in plans if p.get("imaging")]
    assert len(used) == len(set(used))
    for p in plans:
        img = p.get("imaging")
        if img and img["tier"] == "exact_class" and img["collection"] == "ReMIND":
            h = img["histology"]["histology"]
            assert (p["spec_id"], h.split()[0]) in {("RD-LFS", "Glioblastoma"), ("RD-LFS", "Astrocytoma"),
                                                   ("RD-NF1", "Astrocytoma")}, (p["case_id"], h)


def test_the_external_case_is_the_exomiser_example(plans):
    ext = [p for p in plans if p.get("external")]
    assert [(p["case_id"], p["spec_id"], p["external"]) for p in ext] == [("JD-159", "RD-PFEIFFER", "exomiser_pfeiffer")]
    assert ext[0]["background"] is None


def test_allocation_is_deterministic(plans):
    from haenv_rare.round2 import allocate, plan_row
    assert [plan_row(p) for p in allocate()] == [plan_row(p) for p in plans]


@pytest.mark.parametrize("job", ["rare_coding-p6-zh", "rare_coding-p7-en", "rare_coding-p8-en-llm"])
def test_every_trio_member_has_a_row_at_every_spiked_locus(job):
    """Joint-call semantics in the shipped packs: the parents of a de novo case carry an explicit
    0/0 row at the true variant, and every member a row at every decoy."""
    import pathlib
    import yaml
    p = pathlib.Path(__file__).resolve().parents[2] / "inputs" / f"{job}.job.yaml"
    if not p.exists():
        pytest.skip(f"{p} absent: skipped, not passed")
    missing, n = [], 0
    for c in yaml.safe_load(p.read_text(encoding="utf-8"))["cases"]:
        g = c["latent"]["rare_attachments"].get("genome") or {}
        if g.get("source") != "rare-os" or len(g["files"]) != 3:
            continue
        for v in [g["variant"], *(g.get("decoys") or [])]:
            for role, f in g["files"].items():
                n += 1
                if GO.readback(f["path"], v["chrom"], v["pos"], v["ref"], v["alt"]) is None:
                    missing.append((c["case_id"], role, v["chrom"], v["pos"]))
    assert n and not missing, missing[:6]


@pytest.mark.parametrize("job", ["rare_coding-p6-zh", "rare_coding-p7-en", "rare_coding-p8-en-llm"])
def test_no_male_member_is_heterozygous_on_a_non_par_sex_chromosome(job):
    """Gold genotypes and file rows: a male member at a non-PAR X / Y locus (truth variant or
    decoy) is haploid -- 0 or 1, never 0/1 or 0/0."""
    import pathlib
    import yaml
    p = pathlib.Path(__file__).resolve().parents[2] / "inputs" / f"{job}.job.yaml"
    if not p.exists():
        pytest.skip(f"{p} absent: skipped, not passed")
    bad, n = [], 0
    for c in yaml.safe_load(p.read_text(encoding="utf-8"))["cases"]:
        g = c["latent"]["rare_attachments"].get("genome") or {}
        if g.get("source") != "rare-os":
            continue
        sex = {"proband": c["raw"]["sex"], "father": "M", "mother": "F"}
        for v in [dict(g["variant"], genotypes=g["genotypes"]), *(g.get("decoys") or [])]:
            for role, f in g["files"].items():
                if sex[role] != "M" or not GO.haploid_site(v["chrom"], v["pos"], "M"):
                    continue
                n += 1
                gold = str((v.get("genotypes") or {}).get(role))
                got = GO.readback(f["path"], v["chrom"], v["pos"], v["ref"], v["alt"])
                if gold not in ("0", "1") or got not in ("0", "1"):
                    bad.append((c["case_id"], role, v["chrom"], v["pos"], gold, got))
    assert n and not bad, bad
