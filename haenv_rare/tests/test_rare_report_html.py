"""`haenv-rare-html`: per-sentence narrative errors are grouped by how the sentence opens, so a
phrasing the coder does not read shows up as one row; noise sentences that got coded are listed.
Built on a hand-made batch directory, no ontology or model needed."""
import json

from haenv_rare import report_html as H

MD = "# Case X\n\n## Physical Examination\nNo seizure was observed. The patient's mother has a history of ptosis. A new cat.\n"


def _batch(tmp_path, assertions):
    md = tmp_path / "X.md"
    md.write_text(MD, encoding="utf-8")
    s_abs, s_rel, s_noise = (MD.index(t) for t in ("No seizure", "The patient's mother", "A new cat"))
    spans = [{"hpo_id": "HP:0001250", "polarity": "absent", "subject": "proband", "start": s_abs,
              "end": s_abs + len("No seizure was observed")},
             {"hpo_id": "HP:0000508", "polarity": "present", "subject": "relative", "start": s_rel,
              "end": s_rel + len("The patient's mother has a history of ptosis")},
             {"hpo_id": None, "polarity": None, "subject": None, "start": s_noise, "end": s_noise + len("A new cat")}]
    cases = {"X": {"case": {"adjudication": {"rare": {"lang": "en", "attachments": {
        "narrative": {"path": str(md), "spans": spans}}}}}}}
    raw = "noise before " + json.dumps({"narrative": {"assertions": assertions}}) + " after"
    (tmp_path / "responses.jsonl").write_text(json.dumps({"solver": "mirobody-coding", "case": "X", "raw": raw}) + "\n")
    return cases, (s_abs, s_rel, s_noise)


def test_right_wrong_missed_and_coded_noise(tmp_path):
    cases, (a, r, n) = _batch(tmp_path, [])
    asserts = [{"codes": {"hpo": "HP:0001250"}, "char_span": [a, a + 5], "polarity": "present"},     # wrong polarity
               {"codes": {"hpo": "HP:0000001"}, "char_span": [n, n + 3]}]                         # coded noise
    cases, _ = _batch(tmp_path, asserts)
    err = H.narrative_errors(tmp_path, cases, "mirobody-coding")
    assert dict(err["absent"]) == {"No seizure was": [0, 1, 0]}
    assert dict(err["relative"]) == {"The patient's mother": [0, 0, 1]}                           # missed
    assert [c for c, _, _ in err["noise"]] == ["X"]
    page = H._errors_html(err, __import__("html").escape)
    assert "No seizure was" in page and "HP:0000001" in page


def test_all_right_and_other_solver_ignored(tmp_path):
    cases, (a, r, n) = _batch(tmp_path, [])
    asserts = [{"codes": {"hpo": "HP:0001250"}, "char_span": [a, a + 5], "polarity": "absent"},
               {"codes": {"hpo": "HP:0000508"}, "char_span": [r + 30, r + 40], "subject": "relative"}]
    cases, _ = _batch(tmp_path, asserts)
    err = H.narrative_errors(tmp_path, cases, "mirobody-coding")
    assert dict(err["absent"]) == {"No seizure was": [1, 0, 0]}
    assert dict(err["relative"]) == {"The patient's mother": [1, 0, 0]} and err["noise"] == []
    other = H.narrative_errors(tmp_path, cases, "someone-else")
    assert dict(other["absent"]) == {"No seizure was": [0, 0, 1]}                                 # no response = missed


def test_language_of_a_batch():
    en = {"a": {"case": {"adjudication": {"rare": {"lang": "en"}}}}}
    zh = {"b": {"case": {"adjudication": {"rare": {}}}}}
    assert H._lang_of(en) == "en" and H._lang_of(zh) == "zh" and H._lang_of({**en, **zh}) == "en/zh"


def test_no_responses_file_means_no_section(tmp_path):
    assert H.narrative_errors(tmp_path, {}, "x") == {} and H._errors_html({}, str) == ""
