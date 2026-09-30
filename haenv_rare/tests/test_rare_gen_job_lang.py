"""`haenv-rare-gen --lang en --variant 2`: new patients (v2 ids) of the same diseases with English
surface text; `--refresh` of an English pack re-derives it in English."""
import pytest
import yaml

from haenv_rare.gen import ontology_dir

needs_ontology = pytest.mark.skipif(
    not (ontology_dir() / "hp.obo").exists(),
    reason=f"ontology files absent under {ontology_dir()} (set HAENV_RARE_ONTOLOGY); skipped, not passed")


@needs_ontology
def test_variant_only_and_english():
    from haenv_rare.gen_job import rare_case_specs
    cases = rare_case_specs(3, ["RD-LFS", "RD-WILSON"], attachments=False, lang="en", variant_only=2)
    assert [c["case_id"][-2:] for c in cases] == ["v2", "v2"]
    zh = {c["case_id"]: c for c in rare_case_specs(2, ["RD-LFS", "RD-WILSON"], attachments=False)}
    for c in cases:
        lat = c["latent"]
        assert lat["rare_lang"] == "en"
        assert all(s["text"].isascii() for s in c["raw"]["symptoms"])
        z = zh[c["case_id"]]["latent"]
        assert [g["hpo_id"] for g in lat["rare_hpo_gold"]] == [g["hpo_id"] for g in z["rare_hpo_gold"]]
        assert (lat["ddx_red_flag"], lat["ddx_urgency"]) == (z["ddx_red_flag"], z["ddx_urgency"])
        assert "rare_lang" not in z or z["rare_lang"] == "zh"


@needs_ontology
def test_refresh_keeps_the_language_of_the_pack(tmp_path):
    from haenv_rare.gen_job import rare_case_specs, refresh_cases
    cases = rare_case_specs(1, ["RD-LFS"], attachments=False, lang="en", variant_only=2)
    cases[0]["latent"]["rare_attachments"] = {"root": "/kept"}
    p = tmp_path / "x.job.yaml"
    p.write_text(yaml.safe_dump({"job_id": "x", "cases": cases}, allow_unicode=True, sort_keys=False))
    head, fresh = refresh_cases(p)
    assert head["job_id"] == "x"
    assert fresh[0]["latent"]["rare_lang"] == "en"
    assert fresh[0]["raw"]["symptoms"] == cases[0]["raw"]["symptoms"]
    assert fresh[0]["latent"]["rare_attachments"] == {"root": "/kept"}
