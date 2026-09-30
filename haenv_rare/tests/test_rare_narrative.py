"""期刊体叙述病历(本包 narrative.py):结构来自 84 篇开放获取病例报告的章节骨架,
内容全部由金标填充。三条不变量:span 能切回原文、诊断/别名/基因不出现在正文、噪声句有记录。"""
import pytest

from haenv_rare.gen import load_specs, ontology_dir, sample_gold, select_series
from haenv.demographics import sample_profile
from haenv_rare.narrative import leak_terms, pair_imaging, render_narrative

# The tests that sample gold terms need the HPO / Orphanet files (not in git; see the
# rare-disease decision doc). Without them they would fail with FileNotFoundError, which
# reads like a regression. Skipped is not passed: the reason says where the files go.
needs_ontology = pytest.mark.skipif(
    not (ontology_dir() / "hp.obo").exists(),
    reason=f"ontology files absent under {ontology_dir()} (set HAENV_RARE_ONTOLOGY or `rare.ontology_dir` in config.local.yaml); skipped, not passed")

SPECS = ("RD-LFS", "RD-WILSON", "RD-DMD", "RD-RETT")


@pytest.mark.parametrize("sid", SPECS)
@needs_ontology
def test_every_gold_term_has_a_span_that_slices_back(sid):
    spec = load_specs()[sid]
    cid = spec["case_id"]
    gold = sample_gold(spec, cid, "F")
    md, spans = render_narrative(cid, spec, gold, {**sample_profile(cid, "F"), "sex": "F"},
                                 {"collection": "Soft-tissue-Sarcoma", "tier": "exact_class", "series": [{"modality": "MR"}]},
                                 {"files": {"proband": {}}})
    by_idx = {s["idx"]: s for s in spans if s["hpo_id"]}
    assert len(by_idx) == len(gold), f"{sid}: {len(by_idx)} spans for {len(gold)} gold terms"
    for g in gold:
        s = by_idx[g["idx"]]
        frag = md[s["start"]:s["end"]]
        assert frag == frag.strip() and frag, (sid, g)
        assert g["label"] in frag, (sid, g["label"], frag)
        assert s["polarity"] == g["polarity"] and s["subject"] == g["subject"]


@pytest.mark.parametrize("sid", SPECS)
@needs_ontology
def test_diagnosis_alias_and_gene_never_appear(sid):
    spec = load_specs()[sid]
    cid = spec["case_id"]
    md, _ = render_narrative(cid, spec, sample_gold(spec, cid, "M"), {"sex": "M", "age_range": "30-34"}, None, None)
    assert [t for t in leak_terms(spec) if t in md] == []


@needs_ontology
def test_noise_sentences_are_recorded_and_carry_no_term():
    spec = load_specs()["RD-LFS"]
    gold = sample_gold(spec, "JD-55", "F")
    md, spans = render_narrative("JD-55", spec, gold, {"sex": "F", "age_range": "40-44"}, None, None)
    noise = [s for s in spans if not s["hpo_id"]]
    assert noise, "a record with no life-event noise cannot test abstention"
    labels = {g["label"] for g in gold}
    for s in noise:
        frag = md[s["start"]:s["end"]]
        assert not any(l in frag for l in labels), frag


@needs_ontology
def test_sections_follow_the_corpus_skeleton():
    spec = load_specs()["RD-WILSON"]
    md, _ = render_narrative("JD-56", spec, sample_gold(spec, "JD-56", "M"), {"sex": "M", "age_range": "20-24"},
                             {"collection": "UPENN-GBM", "tier": "region_match", "series": [{"modality": "MR"}]},
                             {"files": {"proband": {}, "father": {}, "mother": {}}})
    for h in ("## 主诉", "## 现病史", "## 既往史", "## 家族史", "## 体格检查", "## 实验室检查", "## 影像学检查", "## 基因检测"):
        assert h in md, h
    assert "家系三人" in md                                   # trio stated, result withheld
    assert "结果待回" in md


def test_pair_imaging_never_claims_a_match_it_does_not_have():
    avail = ["Soft-tissue-Sarcoma", "UPENN-GBM", "TCGA-SARC"]
    assert pair_imaging("RD-LFS", 0, avail) == {"collection": "Soft-tissue-Sarcoma", "tier": "exact_class"}
    assert pair_imaging("RD-TSC", 0, avail)["tier"] == "region_match"
    w = pair_imaging("RD-WILSON", 0, avail)                  # no plausible pairing exists
    assert w["tier"] == "unrelated_attachment" and w["collection"] in avail
    assert pair_imaging("RD-TSC", 0, ["TCGA-SARC"])["tier"] == "unrelated_attachment"   # match not downloaded


def test_select_series_prefers_image_series_and_caps_bytes(tmp_path, monkeypatch):
    import haenv_rare.gen as R
    meta = {"rt.zip": ("RTSTRUCT", 1, 300_000), "small.zip": ("MR", 40, 5_000_000),
            "huge.zip": ("CT", 1900, 500_000_000), "mid.zip": ("CT", 200, 40_000_000)}
    for n in meta:
        (tmp_path / n).write_bytes(b"x")
    monkeypatch.setattr(R, "_series_meta", lambda z: {"modality": meta[z.name][0], "n": meta[z.name][1], "bytes": meta[z.name][2]})
    got = [z.name for z, _ in R.select_series(tmp_path)]
    assert "huge.zip" not in got                             # 500 MB alone blows the per-case budget
    assert got[:2] == ["mid.zip", "small.zip"]               # image series, most instances first
    assert "rt.zip" in got                                   # RT object only as the third filler
    assert sum(meta[n][2] for n in got) <= R.SERIES_BYTES_CAP
    tiny = tmp_path / "only"
    tiny.mkdir()
    (tiny / "rt.zip").write_bytes(b"x")
    monkeypatch.setattr(R, "_series_meta", lambda z: {"modality": "RTSTRUCT", "n": 1, "bytes": 300_000})
    assert [z.name for z, _ in R.select_series(tiny)] == ["rt.zip"]     # RT objects only as filler, never dropped to zero


@needs_ontology
def test_style_one_still_writes_the_shipped_p3_reports_byte_for_byte():
    """Round 2 added a style; the default must keep every report of the shipped zh pack."""
    import pathlib
    import yaml
    from haenv_rare.narrative import sha256_text
    job = pathlib.Path(__file__).resolve().parents[2] / "inputs" / "rare_coding-p3.job.yaml"
    if not job.exists():
        pytest.skip(f"{job} absent: skipped, not passed")
    specs = load_specs()
    n = 0
    for c in yaml.safe_load(job.read_text(encoding="utf-8"))["cases"]:
        lat, raw = c["latent"], c["raw"]
        att = lat["rare_attachments"]
        md, _ = render_narrative(c["case_id"], specs[lat["rare_spec_id"]], lat["rare_hpo_gold"],
                                 {**raw, "sex": raw["sex"]}, att.get("imaging"), att.get("genome"))
        assert sha256_text(md) == att["narrative"]["sha256"], c["case_id"]
        n += 1
    assert n == 20


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("sid", SPECS)
@needs_ontology
def test_style_two_keeps_the_span_contract_and_folds_the_tests_into_one_section(sid, lang):
    from haenv_rare.gen import _mentions, gold_in_lang
    spec = load_specs()[sid]
    cid = spec["case_id"] + "v3"
    gold = gold_in_lang(sample_gold(spec, cid, "F"), lang)
    md, spans = render_narrative(cid, spec, gold, {"sex": "F", "age_range": "5-9"},
                                 {"collection": "ReMIND", "tier": "exact_class", "series": [{"modality": "MR"}]},
                                 {"files": {"proband": {}}}, lang=lang, eeg={"recordings": [{}, {}]}, style=2)
    heads = [l for l in md.splitlines() if l.startswith("## ")]
    assert len(heads) == 6 and heads[-1] in ("## 辅助检查", "## Investigations")
    by_idx = {s["idx"]: s for s in spans if s["hpo_id"]}
    assert len(by_idx) == len(gold)
    for g in gold:
        frag = md[by_idx[g["idx"]]["start"]:by_idx[g["idx"]]["end"]]
        assert frag == frag.strip() and g["label"].lower() in frag.lower()
    assert [s for s in spans if not s["hpo_id"]]
    assert ("脑电图" in md) if lang == "zh" else ("EEG has been recorded (2 recordings)" in md)
    assert [t for t in leak_terms(spec) if _mentions(md, t, word=lang == "en", fold=lang == "en")] == []
