"""Held-out paraphrased case reports (`haenv-rare-paraphrase`): the LLM only tags sentences with
the finding they carry, the tool computes every span and checks the result. `--cache-only`
rebuilds a pack from the per-case cache and refuses (without opening a client) when a case has
no accepted report, so a rebuild can never spend money or silently change the held-out set."""
import json

import pytest
import yaml

import haenv_rare.paraphrase as P

GOLD = [{"idx": 0, "hpo_id": "HP:0001250", "label": "seizure", "polarity": "present", "subject": "proband"},
        {"idx": 1, "hpo_id": "HP:0001263", "label": "global developmental delay", "polarity": "absent",
         "subject": "proband"},
        {"idx": 2, "hpo_id": "HP:0000729", "label": "autistic behavior", "polarity": "present",
         "subject": "relative", "relative": "brother"}]


def _doc(**over):
    s = [{"text": "She had a first seizure at age three.", "finding": 1, "noise": False},
         {"text": "Her development has been on track.", "finding": 2, "noise": False},
         {"text": "Her brother shows autistic behaviour.", "finding": 3, "noise": False},
         {"text": "She recently took up gardening.", "finding": None, "noise": True},
         {"text": "The family adopted a cat.", "finding": None, "noise": True},
         {"text": "Vital signs were stable.", "finding": None, "noise": False}]
    s = over.get("sentences", s)
    return {"sections": [{"heading": "History of Present Illness", "sentences": s[:4]},
                         {"heading": "Physical Examination", "sentences": s[4:]}]}


def test_a_good_report_passes():
    assert P.check(_doc(), 3, ["Rett syndrome", "MECP2"]) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda s: s[:2] + s[3:], "finding 3 appears 0 times"),
    (lambda s: s + [{"text": "Seizures recurred.", "finding": 1}], "finding 1 appears 2 times"),
    (lambda s: [x for x in s if not x.get("noise")] + [{"text": "A walk.", "finding": None, "noise": True}],
     "only 1 noise"),
    (lambda s: s + [{"text": "Suspected Rett syndrome.", "finding": None}], "names the diagnosis"),
    (lambda s: s + [{"text": "line\nbreak", "finding": None}], "line break"),
    (lambda s: s + [{"text": "x", "finding": 9}], "finding 9 does not exist"),
])
def test_each_defect_is_caught(mutate, needle):
    base = _doc()["sections"][0]["sentences"] + _doc()["sections"][1]["sentences"]
    errs = P.check(_doc(sentences=mutate(base)), 3, ["Rett syndrome", "MECP2"])
    assert any(needle in e for e in errs), errs


def test_leak_check_is_whole_word():
    base = _doc()["sections"][0]["sentences"] + _doc()["sections"][1]["sentences"]
    ok = base + [{"text": "Vitals unremarkable.", "finding": None}]
    assert P.check(_doc(sentences=ok), 3, ["ALS"]) == []


def test_spans_are_computed_here_and_slice_back():
    md, spans = P.assemble("JD-1v2", _doc(), GOLD)
    gold = [s for s in spans if s["hpo_id"]]
    assert [s["idx"] for s in gold] == [0, 1, 2]
    assert len([s for s in spans if not s["hpo_id"]]) == 2
    for s in spans:
        frag = md[s["start"]:s["end"]]
        assert frag and frag == frag.strip() and not frag.endswith(".")
    assert md[gold[0]["start"]:gold[0]["end"]] == "She had a first seizure at age three"


def _src_job(tmp_path, n_cases=2):
    cases = [{"case_id": f"JD-{50 + i}v2", "raw": {"sex": "F", "age_range": "20-24"},
              "latent": {"rare_spec_id": "RD-X", "rare_hpo_gold": GOLD,
                         "rare_attachments": {"root": str(tmp_path / "src"), "genome": None}}}
             for i in range(n_cases)]
    p = tmp_path / "src.job.yaml"
    p.write_text(yaml.safe_dump({"job_id": "src", "report": "eval-src.md", "plugins": ["haenv.judges.rare"],
                                 "cases": cases}, allow_unicode=True, sort_keys=False))
    return p


@pytest.fixture
def no_ontology(monkeypatch):
    monkeypatch.setattr(P, "_leak_terms", lambda spec_id: ["Rett syndrome"])
    monkeypatch.setattr(P, "_cfg", lambda: {"model": "writer/x", "concurrency": 1, "max_retries": 0})


def test_cache_only_refuses_a_missing_case_without_opening_a_client(tmp_path, no_ontology, monkeypatch):
    src = _src_job(tmp_path)
    att = tmp_path / "att"
    (att / "JD-50v2").mkdir(parents=True)
    (att / "JD-50v2" / "llm.json").write_text(json.dumps(_doc()))
    def boom(*a, **k):
        raise AssertionError("a client was opened in --cache-only mode")
    monkeypatch.setattr(P, "_client", boom)
    out = tmp_path / "out.job.yaml"
    rc = P.main(["--src", str(src), "--job-id", "p5", "--attach-root", str(att), "--cache-only", "--out", str(out)])
    assert rc != 0 and not out.exists()


def test_cache_only_rebuilds_the_pack_from_the_cache(tmp_path, no_ontology, monkeypatch):
    src = _src_job(tmp_path)
    att = tmp_path / "att"
    for cid in ("JD-50v2", "JD-51v2"):
        (att / cid).mkdir(parents=True)
        (att / cid / "llm.json").write_text(json.dumps(_doc()))
    monkeypatch.setattr(P, "_client", lambda *a, **k: (_ for _ in ()).throw(AssertionError("client opened")))
    out = tmp_path / "out.job.yaml"
    rc = P.main(["--src", str(src), "--job-id", "p5", "--attach-root", str(att), "--cache-only", "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    from haenv.canary import block
    assert text.startswith(block("# "))                   # answer-bearing file: canary first
    doc = yaml.safe_load(text)
    assert doc["job_id"] == "p5" and doc["report"] == "eval-p5.md"
    nar = doc["cases"][0]["latent"]["rare_attachments"]["narrative"]
    md = (att / "JD-50v2" / "JD-50v2.md").read_text(encoding="utf-8")
    assert nar["path"] == str(att / "JD-50v2" / "JD-50v2.md") and nar["writer"] == "writer/x"
    assert nar["n_gold_spans"] == 3 and nar["n_noise"] == 2
    assert md[nar["spans"][0]["start"]:nar["spans"][0]["end"]] == "She had a first seizure at age three"
    assert doc["cases"][0]["latent"]["rare_attachments"]["root"] == str(tmp_path / "src")   # rest kept


def test_a_cached_report_that_fails_the_check_is_not_reused(tmp_path, no_ontology):
    src = _src_job(tmp_path, 1)
    att = tmp_path / "att"
    (att / "JD-50v2").mkdir(parents=True)
    bad = _doc()
    bad["sections"][0]["sentences"].append({"text": "Rett syndrome suspected.", "finding": None})
    (att / "JD-50v2" / "llm.json").write_text(json.dumps(bad))
    out = tmp_path / "out.job.yaml"
    assert P.main(["--src", str(src), "--job-id", "p5", "--attach-root", str(att), "--cache-only",
                   "--out", str(out)]) != 0


def test_settings_default_from_the_package_and_config_overrides(monkeypatch):
    """The writer's settings ship with the plugin (the public config.yaml must not name a
    held-back plugin); a `rare_paraphrase:` section in config.yaml / config.local.yaml wins."""
    import haenv.cli as cli
    monkeypatch.setattr(cli, "load_cfg", lambda: {})
    base = P._cfg()
    assert base["model"] and base["base_url"] and base["api_key_env"] and int(base["concurrency"]) <= 8
    monkeypatch.setattr(cli, "load_cfg", lambda: {"rare_paraphrase": {"model": "other/m", "concurrency": 2}})
    over = P._cfg()
    assert over["model"] == "other/m" and over["concurrency"] == 2 and over["base_url"] == base["base_url"]



def test_pack_header_names_the_overridden_batch_gate(tmp_path, no_ontology):
    from haenv_rare.gen_job import BATCH_GATE_NOTE
    src = _src_job(tmp_path, 1)
    att = tmp_path / "att"
    (att / "JD-50v2").mkdir(parents=True)
    (att / "JD-50v2" / "llm.json").write_text(json.dumps(_doc()))
    out = tmp_path / "o.job.yaml"
    assert P.main(["--src", str(src), "--job-id", "p5", "--attach-root", str(att), "--cache-only", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert BATCH_GATE_NOTE in text and "footprint_discriminates_real_symptom" in BATCH_GATE_NOTE


def test_model_override_and_holdout_split_in_the_header(tmp_path, no_ontology):
    src = _src_job(tmp_path, 2)
    att = tmp_path / "att"
    for cid in ("JD-50v2", "JD-51v2"):
        (att / cid).mkdir(parents=True)
        (att / cid / "llm.json").write_text(json.dumps(_doc()))
    out = tmp_path / "o.job.yaml"
    assert P.main(["--src", str(src), "--job-id", "p8", "--attach-root", str(att), "--cache-only", "--out", str(out),
                   "--model", "vendor/other-model", "--holdout-split"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "written by vendor/other-model" in text
    assert "# Held-out split by case id: dev = JD-50v2 | test = JD-51v2" in text
    assert "evaluated exactly once" in text
    doc = yaml.safe_load(text)
    assert doc["cases"][0]["latent"]["rare_attachments"]["narrative"]["writer"] == "vendor/other-model"


def test_json_is_read_from_a_fenced_or_prefixed_answer():
    body = json.dumps(_doc())
    assert P.json_from(body) == _doc()
    assert P.json_from("```json\n" + body + "\n```") == _doc()
    assert P.json_from("Here is the report:\n" + body) == _doc()
    with pytest.raises(json.JSONDecodeError):
        P.json_from("no json here")
