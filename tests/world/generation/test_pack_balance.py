"""tools/generation/pack_balance.py: pre-selection exclusion, invariant waiver, shortcut-balance helpers; and the
reported-reading realignment of the build (zero model calls)."""
from _paths import ROOT as _TEST_ROOT
import pathlib
import sys

import pytest
import yaml

ROOT = _TEST_ROOT
sys.path[:0] = [str(ROOT)]

from tools.generation import pack_balance as PB  # noqa: E402
from tools.validation import pack_audit as PA  # noqa: E402
from haenv.world.generation.build_clinical import realign_reported_readings  # noqa: E402


def test_choose_exclusions_keeps_smallest_id_per_group():
    assert PB.choose_exclusions([["PT-b", "PT-a"], ["PT-c", "PT-a", "PT-d"]]) == {"PT-b", "PT-c", "PT-d"}
    assert PB.choose_exclusions([["PT-b", "PT-a"]], already={"PT-a"}) == set()


def test_waive_only_when_published_and_selectable_pass():
    gates = {"diversity": {"pool": 1, "pack": 0}, "C3": {"pool": 1, "pack": 0}}
    pre = {"gates": {"diversity": {"selectable_pass": True}}}
    keep, waived = PB.waive(["diversity: pool 1, pack 0", "C3: pool 1, pack 0"], gates, pre)
    assert waived == ["diversity: pool 1, pack 0"] and keep == ["C3: pool 1, pack 0"]
    # a violation inside the published items is never waived
    keep, waived = PB.waive(["diversity: pool 1, pack 1"], {"diversity": {"pool": 1, "pack": 1}}, pre)
    assert keep and not waived
    # without a logged exclusion, or when the selectable pool still violates, the refusal stands
    assert PB.waive(["diversity: pool 1, pack 0"], gates, None) == (["diversity: pool 1, pack 0"], [])
    bad = {"gates": {"diversity": {"selectable_pass": False}}}
    assert PB.waive(["diversity: pool 1, pack 0"], gates, bad)[1] == []


def test_preselect_skips_fixed_selections():
    assert PB.preselect("m2", "/nonexistent", {"_provenance": {"pack_ids": ["a"]}}) is None
    assert PB.preselect("p3", "/nonexistent", {"_provenance": {"paired_pack_ids": ["a"]}}) is None


class _A:
    def __init__(self, select=None):
        self.select = select


def test_configure_select_excludes_before_selection():
    items = [{"case_id": c} for c in ("a", "b", "c")]
    A = _A(lambda xs: xs[:2])
    PA.configure_select(A, {}, 2, "p4", exclude=["a"])
    assert [x["case_id"] for x in A.select(items)] == ["b", "c"]
    A = _A(None)
    PA.configure_select(A, {"eligible_pack_ids": ["a", "b"]}, 2, "p4", exclude=["b"])
    assert [x["case_id"] for x in A.select(items)] == ["a"]
    A = _A(lambda xs: xs)
    PA.configure_select(A, {}, 3, "p4")
    assert len(A.select(items)) == 3


def test_identifiability_ranks_shortcut_items():
    rows = []
    for i in range(200):
        y = i % 2
        rows.append({"f": float(y if i < 150 else 1 - y), "g": float(i % 7), "label": y, "spec": f"s{i % 20}"})
    ba, hits = PB._identifiability(rows, ("f", "g"), "label", n_orders=2, n_seeds=1, jobs=1)
    assert ba > 0.6
    assert sum(hits[:150]) / 150 > sum(hits[150:]) / 50


def test_drop_cases_records_balance(tmp_path):
    p = tmp_path / "j.job.yaml"
    p.write_text("# head\n" + yaml.safe_dump({"cases": [{"case_id": "a"}, {"case_id": "b"}],
                                             "_provenance": {"eligible_pack_ids": ["a", "b"]}}), encoding="utf-8")
    PB.drop_cases(p, {"b"}, {"target": "class", "line": 0.2, "removed": ["b"], "rounds": []})
    d = yaml.safe_load(p.read_text())
    assert p.read_text().startswith("# head\n")
    assert [c["case_id"] for c in d["cases"]] == ["a"] and d["_provenance"]["eligible_pack_ids"] == ["a"]
    assert d["_provenance"]["shortcut_balance"][0]["removed"] == ["b"]


def _ev(day, text, kind="patient_reported_symptom"):
    return {"evidence_id": f"e{day}", "source_type": kind, "source_timestamp": day, "symptom": text}


def test_reported_reading_is_the_patients_own():
    ld = {"fasting_glucose": [{"ts": 83, "value": 12.91}, {"ts": 120, "value": 10.8}, {"ts": 200, "value": 6.9}]}
    ledger = [_ev(103, "早晨空腹血糖6.9,躺很久才睡着"), _ev(10, "空腹血糖6.9"), _ev(115, "空腹血糖查出11.5"),
              _ev(104, "空腹血糖6.9", kind="clinician_note")]
    moves = realign_reported_readings(ledger, ld, 168)
    assert [m["to"] for m in moves] == ["10.8"]
    assert [e["symptom"] for e in ledger] == ["早晨空腹血糖10.8,躺很久才睡着", "空腹血糖6.9", "空腹血糖查出11.5", "空腹血糖6.9"]


def test_reported_reading_supported_value_unchanged():
    ld = {"HbA1c": [{"ts": 50, "value": 7.0}]}
    ledger = [_ev(60, "糖化7.2,最近口干")]
    assert realign_reported_readings(ledger, ld, 100) == [] and ledger[0]["symptom"] == "糖化7.2,最近口干"


# Inherited bilingual balance: tiny on-disk jobs/receipts, no generation or builds.
@pytest.fixture
def inherited_run(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tools.generation import make_pack as MP
    from haenv.p4 import audit
    import haenv.world.validation.shared_audit as SA

    def write_batch(path, ids, language):
        path.mkdir(parents=True, exist_ok=True)
        (path / "batch.json").write_text(json.dumps({"language": language}))
        for name, field in (("cases.jsonl", "case"), ("payloads.jsonl", "sp")):
            (path / name).write_text("".join(json.dumps({"case_id": cid, field: {}}) + "\n" for cid in ids))
        return path

    def write_job(path, ids):
        path.write_text(yaml.safe_dump({"cases": [{"case_id": cid} for cid in ids]}))
        return path

    reference_job = write_job(tmp_path / "reference.job.yaml", ["a", "c"])
    reference_batch = write_batch(tmp_path / "reference-batch", ["a", "c"], "en")
    reference_pack = write_batch(tmp_path / "reference-pack", ["a"], "en")
    source = tmp_path / "make_pack.json"
    ref = {"pack": "p4", "n_items": 1, "seed_sha256": "seed-hash", "world_set": ["background=true"],
           "steps": dict(gen=0, build=0, audit=0, invariants=0, pack_build=0),
           "job": str(reference_job), "batch": str(reference_batch), "pack_batch": str(reference_pack),
           "shortcut_balance": [{"removed": ["b"]}, {"removed": ["d", "b"]}]}
    source.write_text(json.dumps(ref))
    out = tmp_path / "out"
    out.mkdir()
    job = write_job(out / "p4-followup.job.yaml", ["a", "b", "c", "d"])
    control = write_job(out / "control.job.yaml", ["a", "b", "c", "d"])
    batch = write_batch(out / "pool", ["a", "b", "c", "d"], "zh")
    rec = {k: ref[k] for k in ("pack", "n_items", "seed_sha256", "world_set")}
    rec.update(steps={}, balance_from=str(source), control_job=str(control))
    calls = []

    def build(job, root, env, log):
        calls.append(log.name)
        ids = [c["case_id"] for c in yaml.safe_load(job.read_text())["cases"]]
        write_batch(root / "results/joint_dx/p4-followup/new", ids, "zh")
        return 0

    monkeypatch.setattr(MP, "build", build)
    A = SimpleNamespace(select=lambda items: items[:1], context=None, prepare=None,
                        rows=lambda pool, ctx: pool, surface=("length",), decisive=(),
                        surface_targets=("class",), gate_targets=None, strata=())
    monkeypatch.setattr(audit, "audit", lambda **kw: A)
    monkeypatch.setattr(PB, "_reader_inputs", lambda A, batch, *a: ({}, list(MP.rows_by_id(batch, "cases.jsonl"))))
    monkeypatch.setattr(SA, "surface_classifier", lambda *a, **kw: {"cc": .05, "pass": True})
    monkeypatch.setattr(PB, "shortcut_balance", lambda *a, **kw: pytest.fail("inherited balance filtered again"))
    return SimpleNamespace(mp=MP, out=out, job=job, batch=batch, control=control, rec=rec,
                           source=source, reference=ref, calls=calls, write_batch=write_batch,
                           balance=lambda: MP.balance_step("p4", job, batch, None, "seed", 1, out, {}, rec, 1))


def test_inherit_exact_removals_gate_and_publication(inherited_run, monkeypatch):
    import hashlib
    import json
    import haenv.app.config as config
    import haenv.reporting.pack_readout as readout
    import haenv.world.validation.shared_audit as SA
    r = inherited_run
    batch = r.balance()
    assert batch is not None
    assert r.calls == ["build-balance-inherited.log"]
    assert r.rec["batch"] == str(batch)
    assert r.rec["shortcut_balance_inherited"] == {"cc": {"class": .05}, "refused_cc": {}, "pass": True}
    for job in (r.job, r.control):
        doc = yaml.safe_load(job.read_text())
        assert [c["case_id"] for c in doc["cases"]] == ["a", "c"]
        record = doc["_provenance"]["shortcut_balance"][0]
        assert record["removed"] == ["b", "d"]
        assert "inherited" in record["method"]
        assert record["balance_from"] == str(r.source)
        assert record["balance_from_sha256"] == hashlib.sha256(r.source.read_bytes()).hexdigest()
    (r.out / "audit").mkdir()
    (r.out / "audit" / "pack.json").write_text(json.dumps({"items": ["a"]}))
    monkeypatch.setattr(config, "load_cfg", lambda: {"diversity": {"max_category_share": .5}})
    monkeypatch.setattr(readout, "declaration", lambda _: None)
    monkeypatch.setattr(readout, "diversity", lambda *a: {"warnings": [], "threshold": .5})
    assert r.mp.publish("p4", "p4-followup", r.job, batch, r.out, {}, r.rec) == 0
    assert r.rec["published_ids_equal_reference"] is True
    assert r.rec["steps"]["published_ids_reference"] == 0
    assert (pathlib.Path(r.rec["pack_batch"]) / SA.marker_name("p4")).exists()


@pytest.mark.parametrize("defect, message", [
    ("seed_sha256", "seed_sha256"), ("world_set", "world_set"),
    ("n_items", "n_items"), ("pack", "pack"), ("difficulty_r2", "difficulty_r2"),
    ("ids", "case-ID"), ("failed", "step audit"), ("missing_step", "step pack_build"),
    ("chain", "chains"), ("language", "language equals"), ("missing_language", "language"),
])
def test_inheritance_invalid_reference_fails_before_mutation(inherited_run, defect, message):
    import json
    r = inherited_run
    if defect in ("seed_sha256", "world_set", "n_items", "pack", "difficulty_r2"):
        r.reference[defect] = "different"
    elif defect == "ids":
        r.reference["shortcut_balance"].append({"removed": ["unknown"]})
    elif defect == "failed":
        r.reference["steps"]["audit"] = 1
    elif defect == "missing_step":
        del r.reference["steps"]["pack_build"]
    elif defect == "chain":
        r.reference["balance_from"] = "another.json"
    else:
        metadata = pathlib.Path(r.reference["batch"]) / "batch.json"
        metadata.write_text(json.dumps({"language": "zh"} if defect == "language" else {}))
    r.source.write_text(json.dumps(r.reference))
    before = [p.read_bytes() for p in (r.job, r.control)]
    assert r.balance() is None
    assert r.rec["steps"]["shortcut_balance"] == 1
    assert message in r.rec["shortcut_balance_error"]
    assert [p.read_bytes() for p in (r.job, r.control)] == before
    assert not r.calls


def _rewrite_job(path, ids):
    path = __import__("pathlib").Path(path)
    path.write_text(yaml.safe_dump({"cases": [{"case_id": cid} for cid in ids]}))


def test_inheritance_accepts_planned_cases_the_build_did_not_admit(inherited_run):
    """P2 shape: both jobs plan a case ("e") that neither build admits; the inheritance still applies."""
    r = inherited_run
    _rewrite_job(r.reference["job"], ["a", "c", "e"])
    _rewrite_job(r.job, ["a", "b", "c", "d", "e"])
    r.source.write_text(__import__("json").dumps(r.reference))
    assert r.balance() is not None, r.rec.get("shortcut_balance_error")
    assert r.rec["steps"]["shortcut_balance"] == 0


@pytest.mark.parametrize("defect, message", [
    ("planned", "planned case-ID"), ("admitted", "admitted case-ID")])
def test_inheritance_refuses_planned_or_admitted_mismatch(inherited_run, defect, message):
    r = inherited_run
    if defect == "planned":
        _rewrite_job(r.job, ["a", "b", "c", "d", "f"])
    else:
        r.write_batch(__import__("pathlib").Path(r.reference["batch"]), ["a"], "en")
    r.source.write_text(__import__("json").dumps(r.reference))
    before = r.job.read_text()
    assert r.balance() is None
    assert message in r.rec["shortcut_balance_error"]
    assert r.job.read_text() == before and r.calls == []


def test_inherited_sa6_refusal_never_removes_extra(inherited_run, monkeypatch):
    import haenv.world.validation.shared_audit as SA
    r = inherited_run
    monkeypatch.setattr(SA, "surface_classifier", lambda *a, **kw: {"cc": .4, "pass": False})
    assert r.balance() is None
    assert r.rec["steps"]["shortcut_balance"] == 1
    assert "SA-6 refused" in r.rec["shortcut_balance_error"]
    assert r.rec["shortcut_balance_inherited"] == {"cc": {"class": .4}, "refused_cc": {"class": .4}, "pass": False}
    assert r.calls == ["build-balance-inherited.log"]
    for p in (r.job, r.control):
        assert [c["case_id"] for c in yaml.safe_load(p.read_text())["cases"]] == ["a", "c"]


def test_reference_without_balance_drops_nothing_and_still_gates(inherited_run):
    import json
    r = inherited_run
    del r.reference["shortcut_balance"]
    pathlib.Path(r.reference["job"]).write_bytes(r.job.read_bytes())
    # Without removals the reference admitted the same pool as this run.
    r.write_batch(pathlib.Path(r.reference["batch"]), ["a", "b", "c", "d"], "en")
    r.source.write_text(json.dumps(r.reference))
    before = r.job.read_bytes()
    assert r.balance() == r.batch
    assert r.job.read_bytes() == before
    assert not r.calls
    assert r.rec["shortcut_balance_inherited"]["pass"] is True


@pytest.mark.parametrize("defect", ["ids", "reference_changed", "missing_pack"])
def test_inherited_publication_mismatch_revokes_marker(inherited_run, monkeypatch, defect):
    import json
    import haenv.world.validation.shared_audit as SA
    r = inherited_run
    batch = r.balance()
    (r.out / "audit").mkdir()
    (r.out / "audit" / "pack.json").write_text(json.dumps({"items": ["a"]}))
    if defect == "ids":
        r.write_batch(pathlib.Path(r.reference["pack_batch"]), ["c"], "en")
    elif defect == "reference_changed":
        r.source.write_text(r.source.read_text() + " ")
    else:
        (pathlib.Path(r.reference["pack_batch"]) / "cases.jsonl").unlink()
    assert r.mp.publish("p4", "p4-followup", r.job, batch, r.out, {}, r.rec) == 1
    assert r.rec["published_ids_equal_reference"] is False
    assert r.rec["steps"]["published_ids_reference"] == 1
    assert not (r.out / "audit" / "pack.json").exists()
    assert not (pathlib.Path(r.rec["pack_batch"]) / SA.marker_name("p4")).exists()


def test_balance_without_option_uses_original_path(inherited_run, monkeypatch):
    r = inherited_run
    del r.rec["balance_from"]
    calls = []
    monkeypatch.setattr(PB, "shortcut_balance", lambda *a, **kw: calls.append(kw) or None)
    monkeypatch.setattr(PB, "shortcut_gate", lambda *a, **kw: pytest.fail("unexpected inherited gate"))
    before = r.job.read_bytes()
    assert r.balance() == r.batch
    assert calls == [{"jobs": 1}]
    assert r.job.read_bytes() == before
    assert "shortcut_balance" not in r.rec
    assert not r.calls


def test_p3_difficulty_early_return_stays_first(inherited_run, monkeypatch):
    from haenv.world.validation import difficulty_r2
    r = inherited_run
    monkeypatch.setattr(difficulty_r2, "enabled", lambda: True)
    r.rec["balance_from"] = "missing-reference.json"
    before = r.job.read_bytes()
    assert r.mp.balance_step("p3", r.job, r.batch, None, "seed", 1, r.out, {}, r.rec, 1) == r.batch
    assert r.rec["selection_policy"] == "clinical quotas and patient hashes; shortcut filtering disabled"
    assert r.job.read_bytes() == before
    assert not r.calls


def test_balance_from_cli_reaches_balance_and_publication(inherited_run, monkeypatch):
    from haenv.app.settings import SET_ENV
    r = inherited_run
    mp = r.mp
    monkeypatch.delenv(SET_ENV, raising=False)
    def run(command, env, log):
        if log.name == "gen.log":
            (log.parent / "p4-followup.job.yaml").write_bytes(r.job.read_bytes())
        return 0
    monkeypatch.setattr(mp, "_run", run)
    monkeypatch.setattr(mp, "build", lambda *a: 0)
    monkeypatch.setattr(mp, "_batch", lambda *a: r.batch)
    monkeypatch.setattr(PB, "preselect", lambda *a: None)
    monkeypatch.setattr(mp, "invariants_step", lambda *a: 0)
    seen = []
    def balance(*args):
        rec = args[-2]
        seen.append(dict(rec))
        return r.batch
    def publish(*args):
        seen.append(dict(args[-1]))
        return 0
    monkeypatch.setattr(mp, "balance_step", balance)
    monkeypatch.setattr(mp, "publish", publish)
    monkeypatch.setattr(mp, "_done", lambda out, rec, rc: rc)
    dest = r.out / "cli"
    assert mp.main(["--pack", "p4", "--out", str(dest), "--balance-from", str(r.source)]) == 0
    assert len(seen) == 2
    for rec in seen:
        assert rec["balance_from"] == str(r.source)
        assert len(rec["balance_from_sha256"]) == 64
