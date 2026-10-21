'haenv/world/generation/twins/py: each operation changes only its declared fields, and the check refuses any other change/'
from __future__ import annotations
from _paths import ROOT as _TEST_ROOT

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import haenv.world.generation.twins as T

ROOT = _TEST_ROOT


def test_reagent_note_is_the_renderers_suffix():
    from haenv.p4.render import REAGENT_NOTICE
    assert T.REAGENT_NOTICE == REAGENT_NOTICE


def test_reagent_twin_changes_the_source_notice_and_preserves_report_values():
    row = _p4_case(T.P4_NOISE, ["酶法", "酶法，试剂批号 B"])
    block = row["case"]["prediction_context"]["followup_p4"]
    block["laboratory_method_record"] = [{"day": 19, "valid_from": 20, "text": T.REAGENT_NOTICE}]
    op = T.OPS["drop-reagent-note"]
    assert op.select({}, row, {}) == {"days_with_note": [20]}
    before = json.dumps(block["report"])
    raw = SimpleNamespace(prediction_context=row["case"]["prediction_context"])
    assert op.apply(raw, {}) == {"report_rows_changed": 0, "source_records_changed": 1}
    assert json.dumps(block["report"]) == before
    assert block["laboratory_method_record"][0]["text"] == ""


def _p4_case(cls, methods, days=(10, 20)):
    report = [{"day": d, "method": m, "value": 1.0} for d, m in zip(days, methods)]
    return {"case": {"adjudication": {"followup_p4": {"class": cls}},
                     "prediction_context": {"followup_p4": {"previous_draw_day": days[0], "today": days[-1],
                                                            "report": report}}}}


def test_note_selection_needs_a_noise_item_with_the_note_on_a_compared_draw():
    op = T.OPS["drop-reagent-note"]
    with_note = _p4_case(T.P4_NOISE, ["酶法", "酶法" + T.REAGENT_NOTE])
    assert op.select({}, with_note, {}) == {"days_with_note": [20]}
    assert op.select({}, _p4_case("true_change", ["酶法", "酶法" + T.REAGENT_NOTE]), {}) is None
    assert op.select({}, _p4_case(T.P4_NOISE, ["酶法", "酶法"]), {}) is None


def test_note_removal_touches_only_method_strings():
    row = _p4_case(T.P4_NOISE, ["酶法" + T.REAGENT_NOTE, "酶法" + T.REAGENT_NOTE])
    raw = SimpleNamespace(case_id="x", prediction_context=json.loads(json.dumps(row["case"]["prediction_context"])))
    assert T.OPS["drop-reagent-note"].apply(raw, {}) == {"report_rows_changed": 2}
    rep = raw.prediction_context["followup_p4"]["report"]
    assert [r["method"] for r in rep] == ["酶法", "酶法"] and [r["value"] for r in rep] == [1.0, 1.0]


def test_reworded_notes_keep_the_fact_and_change_only_the_wording():
    for name, text in T.REWORDED_NOTES.items():
        assert text and T.REAGENT_NOTE not in text and "厂家" in text
        row = _p4_case(T.P4_NOISE, ["酶法", "酶法" + T.REAGENT_NOTE])
        raw = SimpleNamespace(case_id="x", prediction_context=json.loads(json.dumps(row["case"]["prediction_context"])))
        assert T.OPS[name].apply(raw, {}) == {"report_rows_changed": 1}
        assert [r["method"] for r in raw.prediction_context["followup_p4"]["report"]] == ["酶法", "酶法" + text]
        assert T.OPS[name].fields == T.OPS["drop-reagent-note"].fields


def test_planted_symptoms_leave_the_ledger_and_nothing_else():
    ledger = [{"evidence_id": "e1", "symptom": "乏力"}, {"evidence_id": "e2", "symptom": "口渴"},
              {"evidence_id": "e3", "symptom": "麻木"}]
    row = {"case": {"evidence_ledger": ledger}, "question": {"injected_manifest": {"real_symptom_evidence_ids": ["e1", "e3"]}}}
    op = T.OPS["drop-planted-symptoms"]
    d = op.select({}, row, {})
    assert d == {"evidence_ids": ["e1", "e3"], "n_removed": 2}
    raw = SimpleNamespace(case_id="x", evidence_ledger=list(ledger))
    rec = op.apply(raw, d)
    assert raw.evidence_ledger == [ledger[1]] and rec["n_removed"] == 2 and rec["n_ledger"] == 3
    with pytest.raises(ValueError):
        op.apply(SimpleNamespace(case_id="x", evidence_ledger=[ledger[0]]), d)   # the build lost an entry
    assert op.select({}, {"case": {"evidence_ledger": ledger}, "question": {}}, {}) is None


def test_unname_drops_only_the_named_explainers():
    raw = SimpleNamespace(case_id="x", user_profile={"known_conditions": ["CKD", "hypertension"], "age": 60})
    rec = T.OPS["unname-explainer"].apply(raw, {"unname": ["CKD"]})
    assert raw.user_profile == {"known_conditions": ["hypertension"], "age": 60} and rec == {"n_unnamed": 1}


def _batch(tmp: Path, name: str, rows: dict) -> Path:
    d = tmp / name
    d.mkdir()
    for f, by_id in rows.items():
        (d / f).write_text("".join(json.dumps({"case_id": k, **v}, ensure_ascii=False) + "\n" for k, v in by_id.items()),
                           encoding="utf-8")
    return d


def test_compare_accepts_the_declared_change_and_refuses_any_other(tmp_path):
    op = T.OPS["unname-explainer"]
    orig = {"cases.jsonl": {"a": {"case": {"user_profile": {"known_conditions": ["CKD"]}, "labs": [1]}}},
            "payloads.jsonl": {"a": {"sp": {"user_profile": {"known_conditions": ["CKD"]}}}}}
    src = _batch(tmp_path, "src", orig)
    same = _batch(tmp_path, "orig", orig)
    good = _batch(tmp_path, "good", {"cases.jsonl": {"a": {"case": {"user_profile": {"known_conditions": []}, "labs": [1]}}},
                                     "payloads.jsonl": {"a": {"sp": {"user_profile": {"known_conditions": []}}}}})
    per, bad = T.compare(op, src, same, good, ["a"])
    assert bad == [] and per["a"]["cases.jsonl"] == ["/case/user_profile/known_conditions"]
    worse = _batch(tmp_path, "worse", {"cases.jsonl": {"a": {"case": {"user_profile": {"known_conditions": []}, "labs": [2]}}},
                                       "payloads.jsonl": {"a": {"sp": {"user_profile": {"known_conditions": []}}}}})
    _, bad = T.compare(op, src, same, worse, ["a"])
    assert bad == ["a cases.jsonl: undeclared change at ['/case/labs/*']"]
    _, bad = T.compare(op, src, same, same, ["a"])
    assert len(bad) == 2 and all("identical to the original" in b for b in bad)
    _, bad = T.compare(op, src, good, good, ["a"])
    assert any("original build differs" in b for b in bad)


def test_the_build_hook_needs_details_for_exactly_the_emitted_cases():
    spec = {"op": "unname-explainer", "items": {"a": {"unname": ["CKD"]}}}
    built = {"a": SimpleNamespace(case_id="a", user_profile={"known_conditions": ["CKD"]})}
    assert T.apply_all(spec, built) == {"a": {"n_unnamed": 1}}
    with pytest.raises(SystemExit):
        T.apply_all(spec, {**built, "b": SimpleNamespace(case_id="b", user_profile={})})


def test_builds_carry_the_audit_marker_only_of_an_audited_source(tmp_path):
    op = T.OPS["drop-reagent-note"]
    rows = {"cases.jsonl": {"a": {"case": {}}}}
    src, b1, b2 = (_batch(tmp_path, n, rows) for n in ("src", "b1", "b2"))
    assert not T.carry_audit(op, src, {b1: "x", b2: "y"})
    assert not (b1 / "P4_AUDIT_OK").exists()
    (src / "P4_AUDIT_OK").write_text("ok\n")
    assert T.carry_audit(op, src, {b1: "unchanged original", b2: "twin"})
    first = (b2 / "P4_AUDIT_OK").read_text().splitlines()
    assert len(first[0]) == 64 and first[1].startswith("twin of the audited batch above")


# HD-COM-78 template symptoms (aldosteronism + Cushing); the aldosteronism line's last symptom is the coverable one
_ALDO_CUSH = [{"day": 14, "text": "难治高血压,严格低盐+三联降压仍≈146/96", "context": "无家族史"},
              {"day": 35, "text": "新发高血压≈148/95,严格低盐一月无改善", "context": ""},
              {"day": 56, "text": "乏力、爬楼腿软,活动后加重 + 手指麻木", "context": ""},
              {"day": 77, "text": "下腹/大腿内侧宽紫红纹 + 易瘀青 + 近端肌无力", "context": "复查提示低钾倾向"}]


def _rendered_case(known):
    from haenv.m2 import core
    _, rec = core.explain_away_render("HD-COM-78", known, _ALDO_CUSH, T=120, case_id="tw-1")
    return {"raw": {"disease": known[0]}, "latent": {"ddx_spec_id": "HD-COM-78", "m2": {"explain_away": rec}}}


def test_unname_counts_only_explainers_that_license_a_clause_left_on_the_record():
    """W7 follow-up. Under known CKD the line's rows are E28 (CKD: 难治高血压) and E31 (the Cushing line); the clue took
    the 难治高血压 symptom, so CKD licenses nothing left and unnaming it would change no explained evidence: not a twin.
    Under known hypothyroidism E29 licenses 乏力、爬楼腿软, which stays: a twin. Mutation: drop the licensing check ->
    CKD becomes removable."""
    ckd = _rendered_case(["obesity", "CKD"])
    lines = ckd["latent"]["m2"]["explain_away"]["lines"]
    assert lines and "E28" in lines[0]["rows"]                                    # the row is recorded on the line ...
    r = T.classify_explainers(ckd, ["obesity", "CKD"], "")
    assert r is not None and not r["include"] and r["why"] == "line_only", r        # ... but licenses nothing left
    hyp = _rendered_case(["obesity", "hypothyroidism"])
    r = T.classify_explainers(hyp, ["obesity", "hypothyroidism"], "")
    assert r["include"] and r["unname"] == ["hypothyroidism"] and r["stays"] == ["JD-CUSH"], r
