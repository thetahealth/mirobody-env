"""make_pack.py -- one command from (pack, N, seed) to an audited pack: write the job, build it with
the deterministic generator, run the shared generation-time audit. Zero model calls.

  PYTHONPATH=<repo> python tools/make_pack.py --pack p4 --n 100 --out <dir> [--seed S] [--py-audit <python>]

The seed is read from `--seed`, else from HAENV_PACK_SEED, else the public sample seed; a private
seed is passed to the child steps through the environment and is never written: the job and the
batch record only its sha256. Outputs under <dir>: `<job_id>.job.yaml`, `results/...` (the built
batch), `audit/` (`audit.json`, `pack.json`), `make_pack.json` (N, seed hash, paths, exit codes).
Exit 0 when every step passed.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haenv import pack_size as PS     # noqa: E402

#: pack -> (generator, job id, needs the deterministic workup batch)
PACKS = {"m2": ("tools/m2_gen_job.py", "m2-pack1", True),
         "pack2": ("tools/pack2_gen_job.py", "pack2-triage", True),
         "p3": ("tools/p3_gen_job.py", "p3-meds", False),
         "p4": ("tools/p4_gen_job.py", "p4-followup", False)}
WORKUP = "inputs/ddx-workup.job.yaml"


def _run(cmd: list[str], env: dict, log: pathlib.Path) -> int:
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def _batch(root: pathlib.Path, job_id: str) -> pathlib.Path:
    got = sorted((root / "results").glob(f"*/{job_id}/*/cases.jsonl"))
    if not got:
        raise SystemExit(f"[make_pack] no batch of {job_id} under {root}")
    return got[-1].parent


def build(job: pathlib.Path, root: pathlib.Path, env: dict, log: pathlib.Path) -> int:
    return _run([sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic", "--fresh",
                 "--set", "env_file=/dev/null"], {**env, "HAENV_OUTPUT_ROOT": str(root)}, log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True, choices=sorted(PACKS))
    ap.add_argument("--n", type=int, default=PS.DEFAULT_N)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="an existing deterministic build of the workup job (else built here)")
    ap.add_argument("--py-audit", default=sys.executable, help="python for the audit (needs scikit-learn, joblib)")
    ap.add_argument("--profile", action="store_true", help="the audit also computes its permutation p readings")
    ap.add_argument("--jobs", type=int, default=None, help="audit worker processes")
    a = ap.parse_args(argv)
    seed = PS.cli_seed(a.seed)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gen, job_id, needs_ref = PACKS[a.pack]
    env = {**os.environ, "HAENV_PACK_SEED": seed, "PYTHONPATH": os.pathsep.join(
        [str(ROOT), str(ROOT / "plugins"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    rec = {"pack": a.pack, "n_items": a.n, "seed_sha256": PS.seed_sha256(seed), "steps": {}}
    ref = pathlib.Path(a.ref).resolve() if a.ref else None
    if needs_ref and ref is None:
        rc = build(ROOT / WORKUP, out / "ref", env, out / "ref.log")
        rec["steps"]["ref"] = rc
        if rc:
            return _done(out, rec, 1)
        ref = _batch(out / "ref", "ddx-workup")
    job = out / f"{job_id}.job.yaml"
    cmd = [sys.executable, gen, "--n", str(a.n), "--out", str(job)]
    if a.pack == "pack2":
        cmd += ["--ref-batch", str(ref), "--work", str(out / "pack2-gen")]
    rc = _run(cmd, env, out / "gen.log")
    rec["steps"]["gen"] = rc
    if rc:
        return _done(out, rec, 1)
    rc = build(job, out, env, out / "build.log")
    rec["steps"]["build"] = rc
    if rc:
        return _done(out, rec, 1)
    batch = _batch(out, job_id)
    cmd = [a.py_audit, "tools/pack_audit.py", "--pack", a.pack, "--batch", str(batch), "--out", str(out / "audit"),
           "--job", str(job)] + (["--ref", str(ref)] if ref else [])
    if a.profile:
        cmd.append("--profile")
    if a.jobs is not None:
        cmd += ["--jobs", str(a.jobs)]
    rc = _run(cmd, {**env, "OMP_NUM_THREADS": "1"}, out / "audit.log")
    rec["steps"]["audit"] = rc
    rec.update({"job": str(job), "batch": str(batch), "ref": str(ref) if ref else None})
    return _done(out, rec, 1 if rc else 0)


def _done(out: pathlib.Path, rec: dict, rc: int) -> int:
    (out / "make_pack.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
