"""Select component contracts and gates directly from repository paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# Each test directory has one owner; the most specific source prefix wins.
COMPONENTS = {
    "app": ("src/haenv/app", "src/haenv/cli.py", "src/haenv/job.py"),
    "contracts": ("src/haenv/contracts",),
    "guards": (),
    "interaction": ("src/haenv/interaction",),
    "interaction/query": ("src/haenv/interaction/query",),
    "judges": ("src/haenv/judges",),
    "kernel": ("src/haenv_kernel", "src/verifier_core"),
    "m2": ("src/haenv/m2", "tools/generation/m2", "tools/generation/m2_gen_job.py"),
    "p2": ("src/haenv/p2", "tools/generation/p2_gen_job.py"),
    "p3": ("src/haenv/p3", "tools/generation/p3_gen_job.py"),
    "p4": ("src/haenv/p4", "tools/generation/p4_gen_job.py"),
    "pack2": ("src/haenv/pack2",),
    "paper": ("paper",),
    "provenance": ("src/haenv/provenance",),
    "release": ("tools/release", "web", "docs", "README.md", "README.zh-CN.md", "CHANGELOG.md"),
    "reporting": ("src/haenv/reporting",),
    "resources": ("src/haenv/resources",),
    "runtime": ("src/haenv/runtime",),
    "scoring": ("src/haenv/scoring",),
    "scoring/semantic": ("src/haenv/scoring/semantic",),
    "serve": ("src/haenv/serve",),
    "testing": ("tools/dev", "tools/testing", "tools/validation"),
    "world": ("src/haenv/world", "tools/generation"),
    "world/clinical": ("src/haenv/world/clinical",),
    "world/generation": ("src/haenv/world/generation",),
    "world/history": ("src/haenv/world/history",),
    "world/observations": ("src/haenv/world/observations",),
    "world/overlay": ("src/haenv/world/overlay.py",),
    "world/physio": ("src/haenv/world/physio",),
    "world/protocols": ("src/haenv/world/protocols",),
    "world/validation": ("src/haenv/world/validation",),
    "world/wq": ("src/haenv/world/wq.py",),
}


def _under(path, prefix):
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def owners(test):
    """Parent directories own only files outside separately listed child components."""
    path = Path(test)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    matches = [name for name in COMPONENTS if _under(path.as_posix(), "tests/" + name)]
    return [name for name in matches if not any(name != other and _under(other, name) for other in matches)]


def select(paths, *, public=False):
    selected = {"tests/guards"} if (ROOT / "tests/guards").is_dir() else set()
    for item in paths:
        path = Path(item)
        path = path.relative_to(ROOT) if path.is_absolute() else path
        label = path.as_posix()
        if label.startswith("tests/"):
            matches = [(len(name), name) for name in owners(label)]
        else:
            matches = [(len(prefix), name) for name, prefixes in COMPONENTS.items()
                       for prefix in prefixes if _under(label, prefix)]
        if matches:
            selected.add("tests/" + max(matches)[1])
        else:
            selected.add("tests")
    if public:
        selected.update(json.loads((ROOT / "tools/dev/public_tests.json").read_text()))
    selected = {p for p in selected if (ROOT / p).exists()}
    return sorted(p for p in selected if not any(p != parent and _under(p, parent) for parent in selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="changed source/test paths; default: git diff HEAD")
    parser.add_argument("--public", action="store_true", help="also select the exported public surface")
    parser.add_argument("--run", action="store_true", help="run the selection with 64 workers")
    args = parser.parse_args()
    paths = args.paths or subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()
    tests = select(paths, public=args.public)
    if not args.run:
        print("\n".join(tests))
        return 0
    env = dict(os.environ, UV_NO_SYNC="1", UV_FROZEN="1",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT / "src", ROOT, ROOT / "plugins"))))
    command = [sys.executable, "-m", "pytest", *tests, "-n", "64", "--ff", "--durations=30"]
    if args.public:
        command += ["-m", "not disk and not internal"]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
