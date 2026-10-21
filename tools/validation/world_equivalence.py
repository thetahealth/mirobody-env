"""Regenerate the certified world comparison without dropping output fields."""
from __future__ import annotations
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
import hashlib
import json
import runpy
from haenv.provenance.anchor import ANCHOR_DIR

ROOT = _repository_root(__file__)


def proof():
    return json.loads((ANCHOR_DIR / "world-equivalence-1.3.1.json").read_text())


def job_output(task):
    pack, seed, force = task[:3]
    from haenv.world.validation import pack_size as PS
    seed = PS.PUBLIC_SEED if seed == "PUBLIC" else seed
    if pack == "p3":
        from tools.generation import p3_gen_job as G
        original = G.WINDOW_FLOOR, G.WINDOW_EXTRA_MAX
        try:
            if force:
                G.WINDOW_FLOOR, G.WINDOW_EXTRA_MAX = 100.0, 1
            return G.build(PS.DEFAULT_N, seed)[0]
        finally:
            G.WINDOW_FLOOR, G.WINDOW_EXTRA_MAX = original
    if pack == "p2":
        from tools.generation import p2_gen_job as G
        return G.build(PS.DEFAULT_N, seed)[0]
    if pack == "p4":
        from tools.generation import p4_gen_job as G
        return G.build(**G.pool_size(PS.pool_n(PS.DEFAULT_N)), n=PS.DEFAULT_N, seed=seed)[0]
    from tools.generation import m2_gen_job as G
    return G.build(PS.DEFAULT_N, seed)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", choices=("p2", "p3", "p4", "m2"), required=True)
    args = parser.parse_args()
    data = proof()
    # Source-revision identifiers are the only controlled build inputs. Native
    # hashes and native world identities remain separately recorded in the proof.
    from haenv.p2 import hooks as p2
    from haenv.p3 import hooks as p3
    labels = data["fixed_revision_labels"]
    p2.p2_world_fingerprint = lambda: labels["P2_WORLD"]
    p3.p3_world_fingerprint = lambda: labels["P3_WORLD"]
    jobs = {"p2": "pack2-triage", "p3": "p3-meds", "p4": "p4-followup", "m2": "m2-pack1"}
    sys = __import__("sys")
    sys.argv = ["haenv", "build", str(ROOT / f"benchmarks/inputs/{jobs[args.build]}.job.yaml"),
                "--gen", "deterministic", "--fresh", "--workers", "64", "--set", "env_file=/dev/null"]
    runpy.run_module("haenv", run_name="__main__")


if __name__ == "__main__":
    main()
