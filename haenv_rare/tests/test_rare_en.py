"""English rare-coding packages (`rare_lang: en`): the gold set is sampled exactly as for zh and
only its surface form changes; symptom sentences, the case report, the task note and gates
R1-R3 follow the language. `rare_lang` is a registered latent key (class `knob`), so it reaches
both the gold block and the question block through `haenv.external_gold`.
"""
import pytest

from haenv.demographics import sample_profile
from haenv_rare.gen import (NEG_CUES_EN, RareSpecError, _mentions, gold_in_lang, load_specs,
                            ontology_dir, render_symptoms, sample_gold, surface_en, task_note)
from haenv_rare.narrative import leak_terms, render_narrative

needs_ontology = pytest.mark.skipif(
    not (ontology_dir() / "hp.obo").exists(),
    reason=f"ontology files absent under {ontology_dir()} (set HAENV_RARE_ONTOLOGY); skipped, not passed")

SPECS = ("RD-LFS", "RD-WILSON", "RD-DMD", "RD-RETT")


def test_surface_form_lowercases_a_word_but_keeps_an_acronym():
    assert surface_en("Seizure") == "seizure"
    assert surface_en("EEG abnormality") == "EEG abnormality"
    assert surface_en("X") == "X"


def test_unknown_language_is_refused():
    with pytest.raises(RareSpecError):
        gold_in_lang([{"hpo_id": "HP:1", "label": "x", "polarity": "present", "subject": "proband"}], "fr")


def test_zh_is_the_identity():
    g = [{"hpo_id": "HP:1", "label": "癫痫", "polarity": "present", "subject": "proband"}]
    assert gold_in_lang(g, "zh") is g


@pytest.mark.parametrize("sid", SPECS)
@needs_ontology
def test_english_gold_keeps_the_terms_and_changes_only_the_surface(sid):
    spec = load_specs()[sid]
    cid = spec["case_id"] + "v2"
    zh = sample_gold(spec, cid, "F")
    en = gold_in_lang(zh, "en")
    key = lambda g: (g["idx"], g["hpo_id"], g["polarity"], g["subject"])
    assert [key(g) for g in en] == [key(g) for g in zh]        # same terms, same order
    for a, b in zip(zh, en):
        assert b["label_zh"] == a["label"] and b["label"].isascii(), b
        if a.get("relative"):
            assert b["relative_zh"] == a["relative"] and b["relative"].isascii()


@pytest.mark.parametrize("sid", SPECS)
@needs_ontology
def test_english_symptom_sentences_carry_label_negation_and_relative(sid):
    spec = load_specs()[sid]
    cid = spec["case_id"] + "v2"
    gold = gold_in_lang(sample_gold(spec, cid, "M"), "en")
    rows = render_symptoms(gold, cid, 120, lang="en")
    assert len(rows) == len(gold)
    for g, r in zip(gold, rows):
        text = r["text"]
        assert text.isascii(), text
        assert g["label"].lower() in text.lower(), (g["label"], text)
        if g["subject"] == "relative":
            assert g["relative"] in text
        elif g["polarity"] == "absent":
            assert any(c in text.lower() for c in NEG_CUES_EN), text


@pytest.mark.parametrize("sid", SPECS)
@needs_ontology
def test_english_report_spans_slice_back_and_nothing_leaks(sid):
    spec = load_specs()[sid]
    cid = spec["case_id"] + "v2"
    gold = gold_in_lang(sample_gold(spec, cid, "F"), "en")
    md, spans = render_narrative(cid, spec, gold, {**sample_profile(cid, "F"), "sex": "F"},
                                 {"collection": "UPENN-GBM", "tier": "region_match", "series": [{"modality": "MR"}]},
                                 {"files": {"proband": {}, "father": {}, "mother": {}}}, lang="en")
    for h in ("## Chief Complaint", "## History of Present Illness", "## Family History",
              "## Physical Examination", "## Genetic Testing"):
        assert h in md, h
    by_idx = {s["idx"]: s for s in spans if s["hpo_id"]}
    assert len(by_idx) == len(gold)
    for g in gold:
        frag = md[by_idx[g["idx"]]["start"]:by_idx[g["idx"]]["end"]]
        assert frag == frag.strip() and g["label"].lower() in frag.lower(), (g["label"], frag)
    assert [s for s in spans if not s["hpo_id"]], "no noise sentence: abstention untestable"
    assert [t for t in leak_terms(spec) if _mentions(md, t, word=True)] == []


def test_english_mentions_are_whole_word_for_ascii_terms():
    assert not _mentions("Vital signs were stable", "ALS", word=True)       # "ALS" is not in "vitals"
    assert not _mentions("wears glasses", "GLA", word=True)
    assert _mentions("history of ALS in the family", "ALS", word=True)
    assert _mentions("含GLA基因", "GLA")                                     # zh: substring as before


def test_task_note_follows_the_language_and_has_no_case_content():
    import re
    cjk = re.compile(r"[\u4e00-\u9fff]")
    en, zh = task_note("en"), task_note("zh")
    assert not cjk.search(en) and cjk.search(zh)
    assert task_note() == zh
    for note in (en, zh):
        assert "HP:nnnnnnn" in note and "ORPHA:nnn" in note


def test_rare_lang_is_a_registered_knob_that_reaches_both_blocks():
    import haenv_rare as R
    from haenv import external_gold as EG
    assert "rare_lang" in R.RARE_KEYS and R.RARE_CLASSES["rare_lang"] == "knob"
    meta = {EG.SLOT: {"rare_spec_id": "RD-LFS", "rare_lang": "en", "rare_attachments": {}}}
    assert R.coding_task_probe(meta) == task_note("en")
    assert R.gold_block(meta)["lang"] == "en"
    zh = {EG.SLOT: {"rare_spec_id": "RD-LFS", "rare_attachments": {}}}         # older packs carry no rare_lang
    assert R.coding_task_probe(zh) == task_note("zh")
