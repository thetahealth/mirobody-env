"""Synthetic twin families: regular gold, nonempty evidence, and bounded edits."""
import copy
from dataclasses import asdict
from pathlib import Path

import pytest

import haenv.world.generation.twins as twins

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {}


def remember_question(raw):
    import haenv.world.wq as wq
    MANIFESTS[twins.sha16(asdict(raw))] = copy.deepcopy(wq.injected_manifest(raw.case_id))


def restore_question(raw):
    import haenv.world.wq as wq
    from haenv.world.generation.opaque_ids import register_manifest_alias
    from haenv.world.generation.opaque_ids import stored_solver_id
    wq.register_injection(raw.case_id, MANIFESTS[twins.sha16(asdict(raw))])
    register_manifest_alias(raw.case_id, stored_solver_id(raw))
    return None


def trials_for(raw, pack):
    restore_question(raw)
    return twins.family_candidates(raw, pack)


def test_family_api_is_available():
    assert callable(getattr(twins, "family_candidates", None))
    assert callable(getattr(twins, "apply_family", None))


@pytest.fixture(scope="module")
def real_cases():
    from haenv.app.job import load_job
    from haenv.world.generation.build import build_case
    # Loading jobs registers exactly the production plugins and gates.
    result = {}
    for pack, name in (("p3", "p3-meds"), ("p4", "p4-followup")):
        job = load_job(str(ROOT / 'benchmarks' / 'inputs' / f'{name}.job.yaml'))
        found = {}
        for cs in job.cases:
            plan = cs.latent[pack]
            cls = plan["class"]
            if cls in found or cls not in ({"downtitrate", "maintain"} if pack == "p3" else {"preanalytical", "analytic_biological_noise"}):
                continue
            raw, _ = build_case(cs)
            if raw is not None:
                found[cls] = raw
                remember_question(raw)
            if len(found) == 2:
                break
        assert len(found) == 2
        result[pack] = list(found.values())
    # P2 uses the real measurement hook, which installs its production gold. Its emission gate
    # (`p2.gate.record_vs_world`) refuses a record that lists a condition the world did not draw the
    # labs from, as the P2 generator never writes one (`p2_gen_job.world_case`); so the P2 questions
    # are grafted onto a built case whose record and world agree.
    from haenv.p2.hooks import observe_triage
    from haenv.p2.gate import record_vs_world
    from haenv.contracts.external_gold import SLOT
    base = next((r for r in result["p4"] if not record_vs_world({}, r)), None)
    for cs in ([] if base else load_job(str(ROOT / 'benchmarks' / 'inputs' / 'p4-followup.job.yaml')).cases):
        raw, _ = build_case(cs)
        if raw is not None and not record_vs_world({}, raw):
            base = raw
            remember_question(raw)
            break
    assert base is not None
    p2 = []
    for tier in ("inf_n14", "ben_spo2"):
        plan = {"tier": tier, "stratum": "plain", "wording": "neutral", "beta_blocker": False}
        from haenv.p2.rules import table
        plan.update(scenario=table()["tiers"][tier]["scenario"], floor=table()["tiers"][tier]["floor"])
        raw = copy.deepcopy(base)
        raw.latent_premise["meta"][SLOT].pop("p4", None)
        raw.latent_premise["meta"][SLOT]["p2"] = plan
        raw.adjudication.pop("followup_p4", None)
        raw.prediction_context.pop("followup_p4", None)
        observe_triage(raw, raw.case_id, {}, {})
        remember_question(raw)
        p2.append(raw)
    result["p2"] = p2
    return result


@pytest.mark.parametrize("pack", ["p2", "p3", "p4"])
def test_trials_recompute_gold_and_preserve_every_undeclared_field(real_cases, pack):
    assert callable(getattr(twins, "family_candidates", None))
    trials = [t for raw in real_cases[pack] for t in trials_for(raw, pack)]
    assert trials
    assert any(not t["gold_changed"] for t in trials)
    for t in trials:
        source = next(r for r in real_cases[pack] if r.case_id == t["case_id"] and twins.sha16(asdict(r)) == t["source_sha16"])
        restore_question(source)
        twin = copy.deepcopy(source)
        record = twins.apply_family(twin, t)
        assert record["gold_changed"] == t["gold_changed"]
        assert asdict(source) != asdict(twin)
        if t["kind"] != twins.NULL_KIND:      # the null edit reorders the profile's condition list, nothing else
            assert source.user_profile == twin.user_profile
        assert source.case_id == twin.case_id
        assert twins.family_gold(twin, pack) == t["expected_gold"]


def test_stale_expected_gold_and_undeclared_edit_are_rejected(real_cases):
    raw = real_cases["p3"][0]
    assert callable(getattr(twins, "family_candidates", None))
    t = trials_for(raw, "p3")[0]
    wrong = copy.deepcopy(t)
    wrong["expected_gold"] = {"class": "fabricated"}
    with pytest.raises(ValueError, match="expected gold"):
        twins.apply_family(copy.deepcopy(raw), wrong)
    wrong = copy.deepcopy(t)
    wrong["declared_fields"] = []
    with pytest.raises(ValueError, match="undeclared"):
        twins.apply_family(copy.deepcopy(raw), wrong)
    wrong = copy.deepcopy(t)
    wrong["trial_sha16"] = "0" * 16
    with pytest.raises(ValueError, match="prepared world"):
        twins.apply_family(copy.deepcopy(raw), wrong)
    drift = copy.deepcopy(raw)
    drift.user_profile["known_conditions"].append("synthetic-drift")
    with pytest.raises(ValueError, match="source"):
        twins.apply_family(drift, t)


def test_noop_and_arbitrary_parameter_invention_are_rejected(real_cases):
    assert callable(getattr(twins, "family_candidates", None))
    raw = real_cases["p4"][0]
    t = trials_for(raw, "p4")[0]
    wrong = copy.deepcopy(t)
    wrong["after"] = copy.deepcopy(wrong["before"])
    with pytest.raises(ValueError, match="empty"):
        twins.apply_family(copy.deepcopy(raw), wrong)
    wrong = copy.deepcopy(t)
    wrong["path"] = "adjudication/followup_p4/cause_params/invented"
    with pytest.raises((ValueError, KeyError)):
        twins.apply_family(copy.deepcopy(raw), wrong)


def test_helper_refuses_shortage_without_writing_jobs(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_twins", ROOT / 'tools/generation/prepare_twin_families.py')
    if spec is None or not Path(spec.origin).exists():
        pytest.fail("family artifact helper is missing")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match="shortage"):
        mod.select_families([], 20)
    assert list(tmp_path.iterdir()) == []


def test_public_baseline_uses_only_the_selected_source_questions(tmp_path, monkeypatch):
    import importlib.util
    import json
    import yaml
    spec = importlib.util.spec_from_file_location("family_helper", ROOT / 'tools/generation/prepare_twin_families.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = tmp_path / "source"
    source.mkdir()
    (source / "cases.jsonl").write_text("".join(json.dumps({"case_id": c, "case": {}}) + "\n" for c in ("a", "b")))
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump({"job_id": "synthetic", "cases": [{"case_id": c} for c in ("a", "b", "extra")],
                                   "models": ["unused-model"]}))
    monkeypatch.delenv("HAENV_PACK_SEED", raising=False)
    def check_build(path, out, arm):
        actual = yaml.safe_load(path.read_text())
        assert [c["case_id"] for c in actual["cases"]] == ["a", "b"]
        assert actual["models"] == []
        return source
    monkeypatch.setattr(mod, "_ordinary_build", check_build)
    _, built = mod.rebuild_public_baseline(job, source, tmp_path / "output")
    assert built == source


def test_adherence_intervention_cannot_change_visits_before_that_interval(real_cases):
    tested = 0
    for source in real_cases["p3"]:
        raw = copy.deepcopy(source)
        # Stored nuisance parameters can be less precise than the original
        # generator's floats. A recent intervention cannot rewrite older visits
        # merely because it reconstructs their values from those rounded params.
        raw.adjudication["meds_p3"]["params"]["r"] = round(raw.adjudication["meds_p3"]["params"]["r"], 3)
        MANIFESTS[twins.sha16(asdict(raw))] = MANIFESTS[twins.sha16(asdict(source))]
        for t in trials_for(raw, "p3"):
            if t["kind"] != "p3-adherence":
                continue
            restore_question(raw)
            twin = copy.deepcopy(raw)
            twins.apply_family(twin, t)
            g = raw.adjudication["meds_p3"]
            cutoff = int(g["refills"][-1])
            past = [r for r in g["visits"] if int(r["day"]) < cutoff]
            after = [r for r in twin.adjudication["meds_p3"]["visits"] if int(r["day"]) < cutoff]
            assert past == after
            tested += 1
    assert tested > 0


def test_m2_family_rejects_removing_more_than_one_fact():
    from types import SimpleNamespace
    raw = SimpleNamespace(case_id="synthetic", user_profile={"known_conditions": ["CKD", "hypothyroidism"]})
    details = {"pack": "m2", "kind": "unname-explainer", "path": "user_profile/known_conditions",
               "before": ["CKD", "hypothyroidism"], "after": "remove-declared-evidence",
               "operation_details": {"unname": ["CKD", "hypothyroidism"]}}
    with pytest.raises(ValueError, match="single"):
        twins._mutate_family(raw, details)


@pytest.fixture
def replay_source(tmp_path, monkeypatch):
    import importlib.util
    import hashlib
    import json
    import yaml
    from haenv.world.generation.opaque_ids import solver_case_id
    spec = importlib.util.spec_from_file_location("replay_helper", ROOT / 'tools/generation/prepare_twin_families.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    job = tmp_path / "source.job.yaml"
    job.write_text(yaml.safe_dump({"job_id": "synthetic", "cases": [{"case_id": "synthetic"}]}))
    source = tmp_path / "source"
    source.mkdir()
    row = {"case_id": "synthetic", "case": {"latent_premise": {"meta": {
        "solver_case_id": solver_case_id("synthetic", "")}}}}
    (source / "cases.jsonl").write_text(json.dumps(row) + "\n")
    (source / "payloads.jsonl").write_text(json.dumps({"case_id": "synthetic", "sp": {}, "vp": {}}) + "\n")
    digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:16]
    fingerprints = {"world_sha": "world-v1", "kernel_sha256": "kernel-v1", "world_knobs": "physio=1"}
    meta = {**fingerprints, "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest()[:16],
            "generator": "deterministic", "cases_sha256": {"synthetic": digest}}
    (source / "batch.json").write_text(json.dumps(meta))
    monkeypatch.setenv("HAENV_PACK_SEED", "")
    import haenv.app.job
    monkeypatch.setattr(haenv.app.job, "load_job", lambda *a, **k: None)
    monkeypatch.setattr(mod, "replay_fingerprints", lambda: fingerprints, raising=False)
    import haenv.app.artifacts.store, haenv.app.artifacts.payloads
    monkeypatch.setattr(haenv.app.artifacts.store, "load_cases", lambda p: {"synthetic": object()})
    monkeypatch.setattr(haenv.app.artifacts.payloads, "_payload_line", lambda item: ("unused", (source / "payloads.jsonl").read_text()))
    return mod, job, source, tmp_path / "output", meta


def test_replay_preflight_accepts_matching_seed_code_and_source(replay_source):
    mod, job, source, out, meta = replay_source
    assert callable(getattr(mod, "preflight_source", None))
    mod.preflight_source(job, source, "m2", expected_items=1)
    assert not out.exists()


@pytest.mark.parametrize("reason", ["bounded trial is invalid", "undeclared change: ['/foreign']"])
def test_prepare_records_rejections_even_for_an_otherwise_eligible_question(replay_source, monkeypatch, reason):
    import json
    mod, job, source, out, _ = replay_source
    rejection = {"case_id": "synthetic", "kind": "p3-adherence", "path": "p", "reason": reason}
    def candidates(*args, rejected, **kwargs):
        rejected.append(rejection)
        return [{"case_id": "synthetic", "kind": "p3-observation", "path": "q", "after": 1,
                 "gold_changed": False}]
    monkeypatch.setattr(mod.twins, "family_candidates", candidates)
    if reason.startswith("undeclared"):
        monkeypatch.setattr(mod, "select_families", lambda *a: pytest.fail("dependency drift must stop selection"))
    with pytest.raises(ValueError, match="dependency|shortage"):
        mod.prepare(job, source, "m2", out, 20)
    record = json.loads((out / "family_record.json").read_text())
    assert record["rejected_trials"] == [rejection]
    assert record["eligible_questions"] == 1
    assert not list(out.glob("*.job.yaml"))


@pytest.mark.parametrize("mismatch", ["seed", "code", "job", "count"])
def test_replay_mismatch_fails_before_output_or_build(replay_source, monkeypatch, mismatch):
    import json
    mod, job, source, out, meta = replay_source
    assert callable(getattr(mod, "preflight_source", None))
    if mismatch == "seed":
        monkeypatch.setenv("HAENV_PACK_SEED", "synthetic-wrong-seed")
    elif mismatch == "code":
        meta["world_sha"] = "different-world"
        (source / "batch.json").write_text(json.dumps(meta))
    elif mismatch == "job":
        job.write_text(job.read_text() + "# changed job\n")
    def forbidden(*args, **kwargs):
        pytest.fail("a mismatched source must not select trials or start a build")
    monkeypatch.setattr(mod.twins, "family_candidates", forbidden)
    monkeypatch.setattr(mod, "_ordinary_build", forbidden)
    with pytest.raises(ValueError, match="preflight"):
        mod.prepare(job, source, "m2", out, 20, expected_items=2 if mismatch == "count" else 1)
    assert not out.exists()


@pytest.mark.parametrize("onset", [-1, "future"])
def test_p3_onset_stays_in_actual_nonnegative_window(real_cases, onset):
    raw = copy.deepcopy(real_cases["p3"][0])
    g = raw.adjudication["meds_p3"]
    g["diary"].append({"kind": "drug_symptom", "onset": int(g["T"]) - 10, "status": "持续至今"})
    i = len(g["diary"]) - 1
    onset = int(g["T"]) + 1 if onset == "future" else onset
    d = {"pack": "p3", "kind": "p3-onset", "path": f"adjudication/meds_p3/diary/{i}/onset",
         "before": g["diary"][i]["onset"], "after": onset}
    with pytest.raises(ValueError, match="window"):
        twins._mutate_family(copy.deepcopy(raw), d)


def test_m2_aggregate_count_cannot_replace_support_for_each_line():
    assert callable(getattr(twins, "_check_m2_support", None))
    original = [{"evidence_id": e} for e in ("line-a", "line-b-1", "line-b-2", "line-b-3")]
    remaining = original[1:]
    d = {"support_by_line": {"a": ["line-a"], "b": ["line-b-1", "line-b-2", "line-b-3"]},
         "protected_clue_ids": ["line-a"]}
    with pytest.raises(ValueError, match="line|clue"):
        twins._check_m2_support(remaining, d)
    with pytest.raises(ValueError, match="line|empty"):
        twins._check_m2_support([], {"support_by_line": {"a": ["line-a"]}, "protected_clue_ids": []})



def test_m2_actual_declared_clue_survives_with_original_manifest():
    from types import SimpleNamespace
    from haenv.m2 import core
    comp, phrase = next((core.EA_HIDDEN[k], v[0]) for k, v in core.clues().items() if k in core.EA_HIDDEN and v)
    raw = SimpleNamespace(adjudication={"ddx": {"spec_id": comp, "join_gold": "unified", "aliases": ["synthetic"]}})
    ledger = [{"evidence_id": "e1", "source_timestamp": 10, "symptom": phrase},
              {"evidence_id": "e2", "source_timestamp": 20, "symptom": "synthetic-source-observation"},
              {"evidence_id": "e3", "source_timestamp": 30, "symptom": "synthetic-source-observation"}]
    man = {"real_symptom_evidence_ids": ["e1", "e2", "e3"], "id_map": {}, "event_schedule": [{"evidence_id": "e1"}]}
    row = {"question": {"injected_manifest": man}}
    job = {"latent": {"m2": {"explain_away": {"clues": [{"component": comp, "text": phrase, "day": 10}]}}}}
    preserved = copy.deepcopy(man)
    details = twins._m2_source_support(raw, row, job, {"sp": {"evidence_ledger": ledger}})
    assert details["protected_clue_ids"] == ["e1"]
    with pytest.raises(ValueError, match="clue"):
        twins._check_m2_support(ledger[1:], details)
    twins._check_m2_support(ledger[:2], details)
    assert man == preserved and details["source_manifest_sha16"] == twins.sha16(man)



def test_p4_removal_cannot_delete_medication_history(real_cases):
    raw = copy.deepcopy(real_cases["p4"][0])
    g = raw.adjudication["followup_p4"]
    i = next(i for i, r in enumerate(g["records"]) if int(r["day"]) == int(g["t1"]))
    record = g["records"][i]
    medicine = {"factor": "acei_start_or_stop", "role": "cause", "acts": True, "params": {}}
    record["perturbations"].append(medicine)
    d = {"pack": "p4", "kind": "p4-remove-source", "path": f"adjudication/followup_p4/records/{i}/perturbations",
         "before": copy.deepcopy(record["perturbations"]), "after": copy.deepcopy(record["perturbations"][:-1])}
    history = copy.deepcopy(raw.prediction_context["followup_p4"]["medication_record"])
    with pytest.raises(ValueError, match="medication history"):
        twins._mutate_family(raw, d)
    assert raw.prediction_context["followup_p4"]["medication_record"] == history



def test_m2_regold_cannot_accept_a_stale_stored_frame():
    from types import SimpleNamespace
    raw = SimpleNamespace(latent_premise={"meta": {}},
                          adjudication={"ddx": {"diagnosis": "synthetic stored diagnosis"}, "m2_frame": {"frame": "fabricated"}})
    with pytest.raises(ValueError, match="stored frame"):
        twins.family_gold(raw, "m2")


@pytest.mark.parametrize("changes", [0, 1, 6, 12, 20])
def test_family_selection_uses_available_gold_changes_before_id_order(changes):
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_twins_selection", ROOT / 'tools/generation/prepare_twin_families.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    def t(i, changed):
        return {"case_id": f"C{i:03d}", "gold_changed": changed, "subject": f"S{i}", "kind": "k", "path": "p", "after": i}
    trials = [t(i, False) for i in range(40)]
    trials += [t(i, True) for i in range(40 - changes, 40)]
    before = copy.deepcopy(trials)
    picked = mod.select_families(trials, 20)
    assert len(picked) == len({t["case_id"] for t in picked}) == 20
    assert sum(t["gold_changed"] for t in picked) == min(changes, 10)
    assert mod.select_families(list(reversed(trials)), 20) == picked
    assert trials == before
