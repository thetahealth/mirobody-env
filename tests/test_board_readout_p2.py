"""P2 installed discovery and readout contracts; synthetic fixtures, zero model calls."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tools.evaluation.board_readout as B
from haenv.p2 import score as S


def test_wheel_entry_point_is_declared():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["entry-points"]["haenv.p2"] == {"p2": "haenv.p2.plugin:judges"}


@pytest.mark.slow
def test_real_wheel_loads_p2_without_checkout_pseudo_plugins(tmp_path):
    wheel_dir = tmp_path / "wheels"
    built = subprocess.run(["uv", "build", "--wheel", "--offline", "--out-dir", str(wheel_dir)],
                           cwd=ROOT, text=True, capture_output=True)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(wheel_dir.glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as z:
        assert not any(p.startswith("plugins/") for p in z.namelist())
        z.extractall(installed)
    # A fresh isolated interpreter: no PYTHONPATH, no checkout working directory,
    # no inherited source module imports. Wheel dist-info is the only p2 provider.
    script = '\nimport importlib.metadata as md, pathlib, sys\nroot = pathlib.Path(sys.argv[1]).resolve()\nsys.path.insert(0, str(root))\nimport haenv\nassert pathlib.Path(haenv.__file__).is_relative_to(root)\neps = list(md.entry_points(group="haenv.p2"))\nassert len(eps) == 1 and eps[0].value == "haenv.p2.plugin:judges", eps\nassert pathlib.Path(eps[0].dist.locate_file("")).resolve() == root\nfrom haenv.judges import load_judge_plugins, registry_manifest\nassert load_judge_plugins("haenv.p2") == ["triage_p2"]\nassert "triage_p2" in registry_manifest()["external"]\nimport haenv.contracts.external_gold as external_gold\nimport haenv.app.packs as pack_skeleton\nassert "triage_p2" in external_gold.BLOCKS and "p2" in pack_skeleton.PACKS\nfrom haenv.p2.plugin import probe_dict\nassert probe_dict()["probe_id"] == "p2.triage"\nassert not any(pathlib.Path(p).resolve() == pathlib.Path(sys.argv[2]).resolve() / "plugins" for p in sys.path)\nprint("wheel-only p2 provider loaded triage_p2, block and p2.triage")\n'
    proc = subprocess.run([sys.executable, "-I", "-c", script, str(installed), str(ROOT)],
                          cwd=tmp_path, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr




def p2_rows():
    out = []
    for i, d in enumerate(S.DISPOSITIONS):
        for solver, answer in (("model-a", d), ("model-b", "ed_now"), ("model-c", "watchful_waiting")):
            out.append({"case": f"c{i}", "solver": solver, "rep": 1, "overall": "OK",
                        **S.atoms({"disposition": answer}, {"set": [d], "floor": d, "rule": "R14_nothing"})})
    return out


def test_p2_board_headline_uses_skill_composite_and_separate_diagnostics():
    rows = p2_rows()
    assert B.SF["p2"](rows) == {"model-a": 1.0, "model-b": 0.0, "model-c": -2.5}
    profiles = B.profile("p2", rows)
    b = profiles["model-b"]
    assert b["cost_skill"] == 0 and b["cost_cc"] > 0
    assert b["under_rate"] == 0 and b["over_rate"] == .75
    assert profiles["model-c"]["cost_skill"] == -2.0


def test_p2_unanswered_cells_recompute_cost_from_their_own_gold():
    rows = p2_rows()
    rows = [r for r in rows if not (r["case"] == "c0" and r["solver"] == "model-a")]
    out = B.fill("p2", rows, ["c0", "c1", "c2", "c3"], ["model-a", "model-b", "model-c"], [1])
    assert len(out) == 12
    missing = next(r for r in out if r["case"] == "c0" and r["solver"] == "model-a")
    assert missing["synthetic_no_answer"] is True
    assert missing["p2_said"] is None and missing["p2_cost"] == 9
    assert missing["p2_in_set"] == 0 and missing["p2_missed_emergency"] == 1
    assert B.SF["p2"](out)["model-a"] < 0


def test_p2_abort_row_is_filled_instead_of_dropped():
    rows = p2_rows()
    rows[0] = {**rows[0], "overall": "ABORT(no_answer:budget)"}
    out = B.fill("p2", rows, ["c0", "c1", "c2", "c3"], ["model-a", "model-b", "model-c"], [1])
    missing = next(r for r in out if r["case"] == "c0" and r["solver"] == "model-a")
    assert missing["overall"] == "NO_ANSWER" and missing["p2_cost"] == 9


def test_p2_replaces_old_pack2_without_becoming_a_fifth_pack():
    assert B.board_packs({"m2", "pack2", "p3", "p4"}) == B.PACKS
    new = B.board_packs({"m2", "p2", "p3", "p4"})
    assert new == ("m2", "p2", "p3", "p4")
    with pytest.raises(ValueError):
        B.board_packs({"m2", "pack2", "p2", "p3", "p4"})
    point = {p: {"a": .6, "b": .4} for p in new}
    samples = {p: [{"a": .6 + i * .001, "b": .4 - i * .001} for i in range(10)] for p in new}
    assert set(B.overall(point, samples, ["a", "b"], packs=new)["pack_sd"]) == set(new)


@pytest.fixture
def m2_baseline(tmp_path):
    """Complete same-pack baseline with a freshly derived stamp and source hashes."""
    cases, reps = ["synthetic-a", "synthetic-b"], [1, 2]
    prefix = tmp_path / "batch"
    for rep in reps:
        d = tmp_path / f"batch-pk{rep}"
        d.mkdir()
        (d / "cases.jsonl").write_text("".join(
            json.dumps({"case_id": c, "case": {"synthetic": i}}) + "\n"
            for i, c in enumerate(cases)), encoding="utf-8")
    metadata = B.m2_baseline_metadata([prefix], cases, reps, {})
    rows = [{"case": c, "rep": rep, "solver": solver, "overall": "OK",
             "dx_listed_positions": [1], "dx_listed": .5, "tool_key_covered": .5,
             "m2_judging_sha16": metadata["m2_judging_sha16"]}
            for c in cases for rep in reps for solver in ("S_rand00", "S_rand01")]
    given = tmp_path / "baseline.jsonl"
    sidecar = pathlib.Path(str(given) + ".meta.json")
    out = tmp_path / "readout"
    out.mkdir()

    def write():
        given.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    def read():
        return B.m2_stub_rows([prefix], reps, out, 1, given=given, cases=cases, items={})

    write()
    assert read() == rows  # positive control before each isolated mutation
    return rows, metadata, write, read


def test_m2_baseline_rejects_wrong_case_hash(m2_baseline):
    _, metadata, write, read = m2_baseline
    key = next(iter(metadata["case_sha256"]))
    metadata["case_sha256"][key] = "0" * 64
    write()
    with pytest.raises(ValueError, match="M2 baseline metadata differs"):
        read()


def test_m2_baseline_rejects_wrong_m2_stamp(m2_baseline):
    rows, metadata, write, read = m2_baseline
    rows[0]["m2_judging_sha16"] = "0" * 16
    write()
    with pytest.raises(ValueError, match="different M2 stamp"):
        read()
    rows[0]["m2_judging_sha16"] = metadata["m2_judging_sha16"]
    metadata["m2_judging_sha16"] = "0" * 16
    write()
    with pytest.raises(ValueError, match="M2 baseline metadata differs"):
        read()


def test_m2_baseline_rejects_wrong_rep(m2_baseline):
    rows, metadata, write, read = m2_baseline
    original = rows[0]["rep"]
    rows[0]["rep"] = 99
    write()
    with pytest.raises(ValueError, match="SAME selected cases/reps exactly once"):
        read()
    rows[0]["rep"] = original
    metadata["reps"] = [1]
    write()
    with pytest.raises(ValueError, match="M2 baseline metadata differs"):
        read()


def test_m2_baseline_rejects_missing_cell_grid(m2_baseline):
    rows, _, write, read = m2_baseline
    rows.pop()  # one stub loses one selected case x rep; other stub stays complete
    write()
    with pytest.raises(ValueError, match="SAME selected cases/reps exactly once"):
        read()


def test_public_readout_main_reads_new_p2_stored_fixture(tmp_path):
    """Run the complete four-pack readout, all production scorers, with stored stub rows."""
    models, cases = ["model-a", "model-b"], [f"c{i}" for i in range(12)]
    prefixes = {}
    for pack in ("m2", "p2", "p3", "p4"):
        prefixes[pack] = tmp_path / pack
        for rep in (1, 2):
            d = tmp_path / f"{pack}-pk{rep}"
            d.mkdir()
            rows = []
            for i, case in enumerate(cases):
                for model in models:
                    correct = model == "model-a"
                    r = {"case": case, "solver": model, "overall": "OK"}
                    if pack == "p2":
                        gold = S.DISPOSITIONS[i % 4]
                        r.update(S.atoms({"disposition": gold if correct else "ed_now"},
                                         {"set": [gold], "floor": gold}))
                    elif pack == "m2":
                        value = 1.0 if correct else .25
                        r.update(tests_recall=value, tests_precision=value, tool_key_covered=value,
                                 dx_listed_positions=[1], dx_listed=value,
                                 m2_judging_sha16=B.m2_score_contract()["m2_judging_sha16"])
                    elif pack == "p3":
                        gold = ("maintain", "check_adherence_or_adverse_effect")[i % 2]
                        said = gold if correct else "maintain"
                        r.update(p3_class_gold=gold, p3_class_said=said, p3_class_ok=int(gold == said),
                                 p3_gate_med_change=0, p3_drug="synthetic-fixture")
                    else:
                        gold = ("true_change", "analytic_biological_noise")[i % 2]
                        said = gold if correct else "true_change"
                        r.update(p4_class_gold=gold, p4_class_said=said, p4_class_ok=int(gold == said),
                                 p4_pct_gold=10.0, p4_obs_pct=30.0, p4_pct_abs_err=0.0 if correct else 10.0,
                                 p4_next_ok=int(correct), p4_analyte="synthetic-fixture")
                    rows.append(r)
            (d / "cases.jsonl").write_text("".join(json.dumps({"case_id": c}) + "\n" for c in cases))
            (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    stub = tmp_path / "stored-stub.jsonl"
    metadata = B.m2_baseline_metadata([prefixes["m2"]], cases, [1, 2], {})
    stub_rows = [{"case": c, "rep": rep, "solver": "S_rand00", "tests_recall": .2,
                  "tests_precision": .2, "tool_key_covered": .2, "dx_listed_positions": [1],
                  "dx_listed": .2, "overall": "OK", "m2_judging_sha16": metadata["m2_judging_sha16"]}
                 for c in cases for rep in (1, 2)]
    stub.write_text("".join(json.dumps(r) + "\n" for r in stub_rows), encoding="utf-8")
    pathlib.Path(str(stub) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    out = tmp_path / "readout"
    args = ["--out", str(out), "--reps", "1,2", "--boot", "20", "--workers", "1",
            "--m2-stub-rows", str(stub)]
    for pack, prefix in prefixes.items():
        args += ["--pack", f"{pack}={prefix}"]
    assert B.main(args) == 0
    report = json.loads((out / "readout.json").read_text())
    assert set(report["packs"]) == {"m2", "p2", "p3", "p4"}
    assert report["m2_scoring"] == B.m2_score_contract()
    assert report["m2_baseline"] == metadata
    assert report["m2_stub"]["stub"] == pytest.approx(.2)
    assert report["packs"]["m2"]["profile"]["model-a"]["tests_recall_f1"] == 1
    assert json.loads((out / "m2_stub_rows.jsonl.meta.json").read_text()) == metadata
    p2 = report["packs"]["p2"]
    assert p2["score_metric"].startswith("cost_skill")
    assert {r["model"]: r["score"] for r in p2["board_cc"]} == {"model-a": 1.0, "model-b": 0.0}
    assert p2["profile"]["model-b"]["cost_cc"] > 0
    assert p2["profile"]["model-b"]["under_rate"] == 0
    assert p2["profile"]["model-b"]["over_rate"] == .75


def legacy_readout(tmp_path):
    """Complete stored-row read/last-row/fill/profile/bootstrap snapshot of old pack2."""
    from haenv.pack2.score import atoms
    models, cases, reps = ["model-a", "model-b"], [f"c{i}" for i in range(8)], [1, 2]
    for rep in reps:
        d = tmp_path / f"legacy-pk{rep}"
        d.mkdir()
        rows = []
        for i, c in enumerate(cases):
            gold = {"disposition": S.DISPOSITIONS[i % 4]}
            for solver in models:
                if rep == 2 and c == "c1" and solver == "model-b":
                    continue
                answer = gold["disposition"] if solver == "model-a" else "within_24h"
                rows.append({"case": c, "solver": solver, "overall": "OK",
                             **atoms({"disposition": answer}, gold)})
        (d / "cases.jsonl").write_text("".join(json.dumps({"case_id": c}) + "\n" for c in cases))
        (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [rows[0], *rows]))
    loaded = B.load("pack2", tmp_path / "legacy", reps)
    filled = B.fill("pack2", loaded, cases, models, reps)
    point, samples = B.board("pack2", filled, models, cases, reps, boot=20, workers=1)
    result = {"loaded": loaded, "filled": filled, "scores": B.SF["pack2"](filled),
              "profile": B.profile("pack2", filled), "board": B.table(point, samples, models)}
    return json.dumps(result, sort_keys=True, separators=(",", ":")).encode()


def test_legacy_pack2_readout_bytes_unchanged(tmp_path):
    digest = hashlib.sha256(legacy_readout(tmp_path)).hexdigest()
    # Captured on coordinator base 8319663c, before board_readout edits.
    assert digest == "c941db17e1ee3a48573fc25fc82a83ce5766fee30b892fe02c0686c0c038e689", digest
