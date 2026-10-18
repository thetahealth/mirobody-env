"""make_pack.py -- one command from (pack, N, seed) to an audited pack: write the job, build it with
the deterministic generator, run the shared generation-time audit. Zero model calls.

  PYTHONPATH=<repo> python tools/generation/make_pack.py --pack p4 --n 100 --out <dir> [--seed S] [--py-audit <python>]

The seed is read from `--seed`, else from HAENV_PACK_SEED, else the public sample seed; a private
seed is passed to the child steps through the environment and is never written: the job and the
batch record only its sha256. Outputs under <dir>: `<job_id>.job.yaml` (the pool), `results/...` (the
built pool batch), `audit/` (`audit.json`, `pack.json`), `pack/<job_id>.job.yaml` (the job cut to the N
published items) with its own build under `pack/results/...` (rows checked byte-identical to the audited
pool rows, audit marker written), `make_pack.json` (N, seed hash, paths, exit codes). After the audit the whole-pool invariant gates run on
the pool and the published items (`invariants/`); a fail-closed gate with a violation (`haenv/world/validation/invariants.py:
FAIL_CLOSED`) stops the run before anything is published. Exit 0 when every step passed.

Nested packs (`haenv.pack_size.NEST_SIZES`): for one seed the pool does not depend on N up to the largest
nested size and the N = 50 pack is a subset of the N = 100 pack, its cases byte-identical. A 50-item run
is extended to 100 without re-running the 50: run the 50-item pack job against the 100-item pack batch
(`haenv run <pack50 job> --batch <stamp of the 100 build>`, or `reliability_passk.py <pack50 job>
--from-batch <stamp>`); the 100-item job later resumes the same batch and runs only the cells not yet
answered (a resumed run marks `case|solver` cells done and does not re-run them).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "plugins"))      # the packs' entry points (the M2 purchase loads the job in-process)

import haenv.world.validation.pack_size as PS     # noqa: E402
from tools.generation import pack_balance as PB  # noqa: E402

#: pack -> (generator, job id, needs the deterministic workup batch)
PACKS = {"m2": ('tools/generation/m2_gen_job.py', "m2-pack1", True),
         "pack2": ('tools/generation/pack2_gen_job.py', "pack2-triage", True),
         "p3": ('tools/generation/p3_gen_job.py', "p3-meds", False),
         "p4": ('tools/generation/p4_gen_job.py', "p4-followup", False),
         # pack 2 rebuilt (world 1.4): wearable-trend triage, acceptable sets, expected cost
         "p2": ('tools/generation/p2_gen_job.py', "p2-triage", False)}
WORKUP = 'benchmarks/inputs/ddx-workup.job.yaml'
STEP_SECONDS: dict[str, float] = {}


def _run(cmd: list[str], env: dict, log: pathlib.Path) -> int:
    started = time.monotonic()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    STEP_SECONDS[log.stem] = round(time.monotonic() - started, 3)
    return rc


def _batch(root: pathlib.Path, job_id: str) -> pathlib.Path:
    got = sorted((root / "results").glob(f"*/{job_id}/*/cases.jsonl"))
    if not got:
        raise SystemExit(f"[make_pack] no batch of {job_id} under {root}")
    return got[-1].parent


#: `--set` items of the world switches (`config.synth.world_history` / `.med_dynamics`), for every step
WORLD_SET: list[str] = []
GEN_WORKERS = 0


def build(job: pathlib.Path, root: pathlib.Path, env: dict, log: pathlib.Path) -> int:
    sets = [x for item in ["env_file=/dev/null"] + WORLD_SET for x in ("--set", item)]
    return _run([sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic", "--fresh",
                 *(["--gen-workers", str(GEN_WORKERS)] if GEN_WORKERS else []), *sets],
                {**env, "HAENV_OUTPUT_ROOT": str(root)}, log)


def main(argv=None) -> int:
    global GEN_WORKERS
    STEP_SECONDS.clear()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True, choices=sorted(PACKS))
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="an existing deterministic build of the workup job (else built here)")
    ap.add_argument("--py-audit", default=sys.executable, help="python for the audit (needs scikit-learn, joblib)")
    ap.add_argument("--profile", action="store_true", help="the audit also computes its permutation p readings")
    ap.add_argument("--jobs", type=int, default=None, help="audit worker processes")
    ap.add_argument("--gen-workers", type=int, default=0, help="concurrent deterministic build workers")
    ap.add_argument("--rules-delivery", choices=("printed", "library"), default=None,
                    help="hospital rules: default library in the real world, printed in the short-history control")
    ap.add_argument("--publish-only", action="store_true", help="redo only the cut-and-build step of a passed run in --out")
    ap.add_argument("--paired-with", default=None, help="make_pack.json of the real arm: reuse its patients/plans and selected IDs")
    ap.add_argument("--paired-source", choices=("real", "control"), default="control",
                    help="paired plan source: real uses the original job, control uses its prepared control job")
    ap.add_argument("--paired-controls", action="store_true", help="select patients feasible on both real and three-switch-off worlds")
    ap.add_argument("--background", choices=("on", "off"), default=None,
                    help="the pack's background layer (default: config.background or synth.background)")
    ap.add_argument("--world-history", choices=("long", "short"), default=None,
                    help="years of record before day 0 (default: config.synth.world_history); short = the course only")
    ap.add_argument("--med-dynamics", choices=("on", "off"), default=None,
                    help="background medicines start/stop/change dose (default: config.synth.med_dynamics)")
    ap.add_argument("--history-dependency-rate", type=float, default=None, help="share within clinically eligible history cases; 0 preserves the previous generator")
    ap.add_argument("--difficulty-r2", choices=("on", "off"), default=None,
                    help="clinically matched historical evidence and inactive look-alikes")
    a = ap.parse_args(argv)
    if a.gen_workers < 0:
        ap.error("gen-workers must be nonnegative")
    GEN_WORKERS = a.gen_workers
    if GEN_WORKERS:
        os.environ["HAENV_GEN_WORKERS"] = str(GEN_WORKERS)
    if a.difficulty_r2:
        os.environ["HAENV_DIFFICULTY_R2"] = a.difficulty_r2
    if a.history_dependency_rate is not None:
        if not 0 <= a.history_dependency_rate <= 1:
            ap.error("history-dependency-rate must be in [0, 1]")
        os.environ["HAENV_HISTORY_DEPENDENCY_RATE"] = str(a.history_dependency_rate)
    seed = PS.cli_seed(a.seed)
    WORLD_SET[:] = ([f"world.history={a.world_history}"] if a.world_history else []) + \
        ([f"world.med_dynamics={'true' if a.med_dynamics == 'on' else 'false'}"] if a.med_dynamics else []) + \
        ([f"background={'true' if a.background == 'on' else 'false'}"] if a.background else []) + \
        ([f"synth.rules_delivery={a.rules_delivery}"] if a.rules_delivery else [])
    if a.background:
        os.environ["HAENV_P2_BACKGROUND"] = a.background
        os.environ["HAENV_M2_BACKGROUND"] = a.background
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gen, job_id, needs_ref = PACKS[a.pack]
    # The labels are part of older make_pack records; resolve the CLI separately.
    from tools._bootstrap import command_path
    if WORLD_SET:
        from haenv.app.settings import SET_ENV
        os.environ[SET_ENV] = json.dumps(WORLD_SET)       # the generator and the audit read the same world
    env = {**os.environ, "HAENV_PACK_SEED": seed, "PYTHONPATH": os.pathsep.join(
        [str(ROOT), str(ROOT / "plugins"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    if a.publish_only:
        rec = json.loads((out / "make_pack.json").read_text(encoding="utf-8"))
        if "difficulty_r2" in rec:
            os.environ["HAENV_DIFFICULTY_R2"] = env["HAENV_DIFFICULTY_R2"] = rec["difficulty_r2"]
        if "history_dependency_rate" in rec:
            os.environ["HAENV_HISTORY_DEPENDENCY_RATE"] = env["HAENV_HISTORY_DEPENDENCY_RATE"] = str(rec["history_dependency_rate"])
        WORLD_SET[:] = list(rec.get("world_set") or WORLD_SET)       # the world the audited pool was built with
        if WORLD_SET:
            from haenv.app.settings import SET_ENV
            os.environ[SET_ENV] = env[SET_ENV] = json.dumps(WORLD_SET)
        if rec.get("pack") != a.pack or any(rec["steps"].get(k) != 0 for k in ("gen", "build", "audit", "invariants")):
            rec["steps"]["invariants"] = 1
            rec["invariant_error"] = "--publish-only requires completed gen/build/audit/invariants for this pack"
            _refuse_pack(rec.get("pack") or a.pack, out, rec)
            _done(out, rec, 1)
            raise SystemExit(f"[make_pack] {rec['invariant_error']}")
        if invariants_step(a.pack, pathlib.Path(rec["job"]), pathlib.Path(rec["batch"]), out, rec, a.jobs):
            return _done(out, rec, 1)
        return _done(out, rec, publish(a.pack, job_id, pathlib.Path(rec["job"]), pathlib.Path(rec["batch"]), out, env, rec))
    if (out / "make_pack.json").exists() or (out / "audit").exists() or (out / "pack").exists():
        raise SystemExit("[make_pack] use a fresh output directory for a new run")
    rec = {"pack": a.pack, "n_items": a.n, "seed_sha256": PS.seed_sha256(seed), "steps": {},
           "gen_workers": GEN_WORKERS,
           **({"world_set": list(WORLD_SET)} if WORLD_SET else {})}
    if a.history_dependency_rate is not None:
        rec["history_dependency_rate"] = a.history_dependency_rate
    if a.difficulty_r2:
        rec["difficulty_r2"] = a.difficulty_r2
    ref = pathlib.Path(a.ref).resolve() if a.ref else None
    if needs_ref and ref is None:
        rc = build(ROOT / WORKUP, out / "ref", env, out / "ref.log")
        rec["steps"]["ref"] = rc
        if rc:
            return _done(out, rec, 1)
        ref = _batch(out / "ref", "ddx-workup")
    job = out / f"{job_id}.job.yaml"
    cmd = [sys.executable, str(command_path(gen, root=ROOT)), "--n", str(a.n), "--out", str(job)]
    if a.pack == "pack2":
        cmd += ["--ref-batch", str(ref), "--work", str(out / "pack2-gen")]
    if a.paired_with:
        import yaml
        paired = json.loads(pathlib.Path(a.paired_with).read_text(encoding="utf-8"))
        if (paired["pack"], paired["n_items"], paired["seed_sha256"]) != (a.pack, a.n, PS.seed_sha256(seed)) or any(paired["steps"].values()):
            raise ValueError("paired arm requires a successful manifest with identical pack, size and seed")
        saved_rate = paired.get("history_dependency_rate")
        saved_r2 = paired.get("difficulty_r2")
        if a.difficulty_r2 is not None and a.difficulty_r2 != saved_r2:
            raise ValueError("paired arm requires the same matched history configuration")
        if saved_r2 is not None:
            rec["difficulty_r2"] = saved_r2
            os.environ["HAENV_DIFFICULTY_R2"] = env["HAENV_DIFFICULTY_R2"] = saved_r2
        if a.history_dependency_rate is not None and a.history_dependency_rate != saved_rate:
            raise ValueError("paired arm requires the same history dependency configuration")
        if saved_rate is not None:
            rec["history_dependency_rate"] = saved_rate
            os.environ["HAENV_HISTORY_DEPENDENCY_RATE"] = env["HAENV_HISTORY_DEPENDENCY_RATE"] = str(saved_rate)
        source = pathlib.Path(paired["job"] if a.paired_source == "real" else paired.get("control_job") or paired["job"])
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        # Patient facts and gold plans stay fixed. Only the background switch
        # stored by the two plugins follows this arm's requested world switch.
        for c in document["cases"]:
            if a.pack == "p2":
                c["latent"]["p2"]["background"] = a.background or "on"
            if a.pack == "m2" and c["latent"].get("m2"):
                c["latent"]["m2"]["background"] = a.background != "off"
                from haenv.m2.core import scene_text
                c["latent"]["m2"]["scene"] = scene_text(c["latent"]["m2"]["frame"], c["raw"], c["latent"], c["case_id"])
        ids = [json.loads(line)["case_id"] for line in pathlib.Path(paired["pack_batch"], "cases.jsonl").read_text().splitlines()]
        if a.pack == "m2":
            document["_provenance"]["pack_ids"] = ids
        document["_provenance"]["paired_pack_ids"] = ids
        job.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
        rec["paired_with"] = str(pathlib.Path(a.paired_with).resolve())
        rec["paired_source"] = a.paired_source
        rec["paired_source_sha256"] = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        rc = 0
    else:
        rc = _run(cmd, env, out / "gen.log")
    rec["steps"]["gen"] = rc
    if rc:
        return _done(out, rec, 1)
    rc = build(job, out, env, out / "build.log")
    rec["steps"]["build"] = rc
    if rc:
        return _done(out, rec, 1)
    batch = _batch(out, job_id)
    if a.paired_controls:
        if a.paired_with:
            raise ValueError("--paired-controls belongs to the real arm; the other arm uses --paired-with")
        paired_eligibility(a.pack, job_id, job, batch, out, env, rec)
        # The eligibility/provenance update changes the job hash. Bind the
        # official pool receipt to that exact job, without restamping old rows.
        rc = build(job, out, env, out / "build-paired.log")
        rec["steps"]["paired_source_rebuild"] = rc
        if rc:
            return _done(out, rec, 1)
        batch = _batch(out, job_id)
    batch = balance_step(a.pack, job, batch, ref, seed, a.n, out, env, rec, a.jobs)
    if batch is None:
        return _done(out, rec, 1)
    pre = PB.preselect(a.pack, batch, __import__("yaml").safe_load(job.read_text(encoding="utf-8")))
    if pre:
        rec["preselect"] = pre
        (out / "preselect.json").write_text(json.dumps(pre, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cmd = [a.py_audit, 'tools/validation/pack_audit.py', "--pack", a.pack, "--batch", str(batch), "--out", str(out / "audit"),
           "--job", str(job)] + (["--ref", str(ref)] if ref else []) + (["--exclude", str(out / "preselect.json")] if pre else [])
    if a.profile:
        cmd.append("--profile")
    if a.jobs is not None:
        cmd += ["--jobs", str(a.jobs)]
    rc = _run(cmd, {**env, "OMP_NUM_THREADS": "1"}, out / "audit.log")
    rec["steps"]["audit"] = rc
    rec.update({"job": str(job), "batch": str(batch), "ref": str(ref) if ref else None})
    if rc:
        return _done(out, rec, 1)
    if invariants_step(a.pack, job, batch, out, rec, a.jobs):
        return _done(out, rec, 1)
    return _done(out, rec, publish(a.pack, job_id, job, batch, out, env, rec))


def balance_step(pack, job, batch, ref, seed, n, out, env, rec, jobs):
    """SA-6 shortcut balance of the pool (`tools/generation/pack_balance.py`): the job (and a paired control job)
    lose the removed cases and the pool is rebuilt; None when the balance cannot reach the line."""
    import importlib
    import yaml
    from tools.validation import pack_audit as PA
    os.environ.setdefault("HAENV_PACK_SEED", seed)
    for attempt in range(3):
        A = importlib.import_module(PA.PACKS[pack]).audit(n_items=n, seed=seed)
        prov = (yaml.safe_load(job.read_text(encoding="utf-8")) or {}).get("_provenance") or {}
        res = PB.shortcut_balance(A, lambda A_: PA.configure_select(A_, prov, n, pack), batch, job, ref, n, jobs=jobs or 32)
        if res is None:
            return batch
        rec.setdefault("shortcut_balance", []).append(res)
        if not res["pass"] or not res["removed"]:
            rec["steps"]["shortcut_balance"] = 1
            return None
        for j in [job] + ([pathlib.Path(rec["control_job"])] if rec.get("control_job") else []):
            PB.drop_cases(j, set(res["removed"]), res)
        rc = build(job, out, env, out / f"build-balance-{attempt}.log")
        rec["steps"][f"balance_rebuild_{attempt}"] = rc
        if rc:
            return None
        batch = _batch(out, pathlib.Path(job).name.replace(".job.yaml", ""))
        rec["batch"] = str(batch)
    rec["steps"]["shortcut_balance"] = 1
    return None


def paired_eligibility(pack, job_id, job, batch, out, env, rec):
    """Prospective two-world eligibility, before selection; never remove a named item.

    Both pools remain on disk and are audited in full. The common eligible IDs
    restrict only the selector. M2's cached first readings are re-derived for the
    control world, before writing its job, as in its ordinary generation stage.
    """
    import copy
    import yaml
    from types import SimpleNamespace
    document = yaml.safe_load(job.read_text(encoding="utf-8"))
    control = copy.deepcopy(document)
    for c in control["cases"]:
        if pack == "p2":
            c["latent"]["p2"]["background"] = "off"
        if pack == "m2" and c["latent"].get("m2"):
            c["latent"]["m2"]["background"] = False
            c["latent"]["m2"].pop("f2", None)
    dest = out / "paired-control.job.yaml"
    dest.write_text(yaml.safe_dump(control, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    from haenv.app.settings import SET_ENV
    switches = ["background=false", "world.history=short", "world.med_dynamics=false"]
    previous = list(WORLD_SET)
    WORLD_SET[:] = switches
    try:
        rc = build(dest, out / "paired-control", {**env, SET_ENV: json.dumps(switches),
                   "HAENV_P2_BACKGROUND": "off", "HAENV_M2_BACKGROUND": "off"}, out / "paired-control.log")
    finally:
        WORLD_SET[:] = previous
    if rc not in (0, 4):
        raise ValueError(f"paired feasibility build failed: exit {rc}")
    other = _batch(out / "paired-control", job_id)
    read = lambda b: {r["case_id"]: r["case"] for r in map(json.loads, (b / "cases.jsonl").read_text().splitlines())}
    real, off = read(batch), read(other)
    ids = sorted(set(real) & set(off))
    if not ids:
        raise ValueError("paired worlds have no common eligible patients")
    if pack == "m2":
        from haenv.m2.core import f2_direction
        by_id = {c["case_id"]: c for c in control["cases"]}
        for cid, raw in off.items():
            c = by_id[cid]
            m2 = c["latent"].get("m2") or {}
            if m2.get("frame") == "F2":
                m2["f2"] = f2_direction(SimpleNamespace(**raw), c["raw"]["disease"], c["latent"]["index_time_T"],
                                        c["raw"]["drug"], c["latent"]["ddx_spec_id"] if m2["arm"] == "positive" else None)
        from tools.generation.m2_gen_job import reselect
        picked = reselect(document, ids)
        document["_provenance"]["pack_ids"] = picked
        control["_provenance"]["pack_ids"] = picked
    for d in (document, control):
        d["_provenance"]["eligible_pack_ids"] = ids
    job.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    dest.write_text(yaml.safe_dump(control, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    rec["control_job"], rec["paired_control_batch"] = str(dest), str(other)
    rec["paired_eligibility"] = {"real": len(real), "control": len(off), "common": len(ids),
                                 "control_build_exit": rc}


def cut_job(job: pathlib.Path, items: list[str], dest: pathlib.Path) -> int:
    """The job with only the published cases (job order); header comments and `_provenance` kept."""
    import yaml
    src = job.read_text(encoding="utf-8")
    j = yaml.safe_load(src)
    want = set(items)
    j["cases"] = [c for c in j["cases"] if c["case_id"] in want]
    if len(j["cases"]) != len(want):
        raise SystemExit(f"[make_pack] {len(want) - len(j['cases'])} published ids are not in {job}")
    head = "".join(ln for ln in src.splitlines(True) if ln.startswith("#"))
    head += f"# The {len(want)} published items of the pool job above (tools/generation/make_pack.py).\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(head + yaml.safe_dump(j, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    return len(j["cases"])


def rows_by_id(batch: pathlib.Path, name: str) -> dict:
    out = {}
    for line in (batch / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out[json.loads(line)["case_id"]] = line
    return out


def published_ids(job: pathlib.Path, out: pathlib.Path) -> list[str]:
    """The published items: the audit's `pack.json`, or the job's `_provenance.pack_ids` (pack 1: its F0 bridge
    items carry no pack gold block, so the audit's item list holds only the F1-F3 part of it)."""
    import yaml
    items = json.loads((out / "audit" / "pack.json").read_text(encoding="utf-8"))["items"]
    ids = (yaml.safe_load(job.read_text(encoding="utf-8")).get("_provenance") or {}).get("pack_ids")
    if ids:
        if not set(items) <= set(ids):
            raise ValueError(f"[make_pack] audited items outside the job's pack_ids: {sorted(set(items) - set(ids))[:5]}")
        items = sorted(ids)
    return items


def invariants_step(pack: str, job: pathlib.Path, batch: pathlib.Path, out: pathlib.Path, rec: dict,
                    jobs: int | None = None) -> int:
    """The whole-pool invariant gates (`haenv/world/validation/invariants.py`) on the audited pool and its published items
    (M2: after a scripted full-menu purchase). A gate of `invariants.FAIL_CLOSED` with a violation refuses the
    pack: its audit manifest is retained with a refused suffix and nothing is published."""
    sys.path.insert(0, str(ROOT / "tools"))
    from tools.validation import invariant_gates as IG
    import haenv.world.validation.invariants as INV
    out = pathlib.Path(out)
    if pack == "pack2":  # legacy triage has a different schema; the inventory covers m2/p2/p3/p4
        rec["steps"]["invariants"] = 0
        rec["invariants_status"] = "not_applicable: legacy pack2"
        return 0
    try:
        started = time.monotonic()
        doc = IG.report(pack, batch, out / "invariants", set(published_ids(job, out)), job=job, jobs=jobs or 32)
        STEP_SECONDS["invariants"] = round(time.monotonic() - started, 3)
    except Exception as exc:
        rec["steps"]["invariants"] = 1
        rec["invariant_error"] = f"{type(exc).__name__}: {exc}"
        _refuse_pack(pack, out, rec)
        print(f"[make_pack] invariant check failed: {rec['invariant_error']}", file=sys.stderr)
        return 1
    refusals, waived = PB.waive(doc["refusals"], doc["gates"], rec.get("preselect"))
    if waived:
        doc["refusals"] = refusals
        rec["invariant_waived_preselect"] = waived
    rc = 1 if doc["refusals"] else 0
    rec["steps"]["invariants"] = rc
    rec["invariants"] = {gid: [r["pool"], r["pack"]] for gid, r in doc["gates"].items()}
    rec["invariant_refusals"] = doc["refusals"]
    if rc:
        _refuse_pack(pack, out, rec)
        for x in doc["refusals"]:
            print(f"[make_pack] invariant gate refused the pack: {x} (fail-closed: {', '.join(INV.FAIL_CLOSED)})",
                  file=sys.stderr)
    return rc


def _refuse_pack(pack: str, out: pathlib.Path, rec: dict) -> None:
    import haenv.world.validation.shared_audit as SA
    manifest = out / "audit" / "pack.json"
    if manifest.exists():
        manifest.replace(manifest.with_name("pack.refused.json"))
    if rec.get("pack_batch"):
        batch = pathlib.Path(rec["pack_batch"]).resolve()
        if not batch.is_relative_to(out.resolve()):
            raise ValueError("published batch is outside the output directory")
        marker = batch / SA.marker_name(pack)
        if marker.exists():
            marker.replace(marker.with_name(marker.name + ".refused"))
    for key in ("pack_build", "pack_rows_identical"):
        rec["steps"].pop(key, None)


def publish(pack: str, job_id: str, job: pathlib.Path, batch: pathlib.Path, out: pathlib.Path, env: dict,
            rec: dict) -> int:
    """Cut the job to the published items, build it as its own batch, check its rows byte-identical to
    the audited pool rows, and mark the batch audited (the marker records the audited pool's sha256)."""
    import hashlib
    import haenv.world.validation.shared_audit as SA
    items = published_ids(job, out)
    pjob = out / "pack" / f"{job_id}.job.yaml"
    cut_job(job, items, pjob)
    rc = build(pjob, out / "pack", env, out / "pack-build.log")
    rec["steps"]["pack_build"] = rc
    if rc:
        return 1
    pb = _batch(out / "pack", job_id)
    diff = {}
    for name in ("cases.jsonl", "payloads.jsonl"):
        a, b = rows_by_id(batch, name), rows_by_id(pb, name)
        bad = sorted(k for k in items if a.get(k) is None or a.get(k) != b.get(k))
        if bad or set(b) != set(items):
            diff[name] = bad[:5] or sorted(set(b) ^ set(items))[:5]
    rec["steps"]["pack_rows_identical"] = 0 if not diff else 1
    rec.update({"pack_job": str(pjob), "pack_batch": str(pb), "pack_rows_diff": diff})
    if diff:
        return 1
    (pb / SA.marker_name(pack)).write_text(hashlib.sha256((batch / "cases.jsonl").read_bytes()).hexdigest() + "\n",
                                           encoding="utf-8")
    from haenv.reporting.pack_readout import declaration
    from haenv.reporting.pack_readout import diversity
    from haenv.app.config import load_cfg
    cases = {cid: json.loads(row)["case"] for cid, row in rows_by_id(pb, "cases.jsonl").items()}
    reading = diversity(declaration(pack), cases, load_cfg()["diversity"]["max_category_share"])
    rec["diversity"] = reading
    (out / "diversity.json").write_text(json.dumps(reading, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[make_pack] diversity: " + json.dumps(reading, ensure_ascii=False), flush=True)
    for dimension in reading["warnings"]:
        print(f"WARNING [make_pack] {dimension}: maximum category share exceeds {reading['threshold']}", flush=True)
    # Publish-only may rebuild at a new stamp. The pack is a single release;
    # retain its chosen batch only after every publication check has succeeded.
    import shutil
    pack_root = (out / "pack").resolve()
    if not pb.resolve().is_relative_to(pack_root / "results"):
        raise ValueError("Published batch is outside the pack results directory")
    for previous in (pack_root / "results").glob("*/*/*"):
        if previous.is_dir() and previous.resolve() != pb.resolve():
            shutil.rmtree(previous)
    return 0


def _done(out: pathlib.Path, rec: dict, rc: int) -> int:
    from haenv.app.config import load_cfg
    from haenv.reporting.pack_readout import seed_readout
    directory = rec.get("pack_batch") or rec.get("batch")
    rec["seed"] = seed_readout([directory] if directory else [], load_cfg())
    print(f"[make_pack] {rec['seed']['kind']} seed", flush=True)
    for warning in rec["seed"]["warnings"]:
        print(f"WARNING [make_pack] {warning}", flush=True)
    rec["step_seconds"] = dict(STEP_SECONDS)
    (out / "make_pack.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
