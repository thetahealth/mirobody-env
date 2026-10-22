"""Select component contracts and gates directly from repository paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import re
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
    "release": ("tools/release", "tools/presentation/make_readme_task_scores.py",
                "tools/presentation/make_readme_board.py",
                "web", "docs", "README.md", "README.zh-CN.md", "CHANGELOG.md", "CITATION.cff"),
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


#: Files that otherwise select the whole tree; a diff that only moves haenv's own version
#: selects the release contracts instead.
VERSION_FILES = ("pyproject.toml", "uv.lock", "src/haenv/__init__.py")
_VERSION_LINE = re.compile(r'^[+-]\s*(?:version|__version__)\s*=\s*"[^"]*"\s*$')
#: The default tier's marker plus the exported surface; world contracts run only with -m world.
PUBLIC_MARKER = "not disk and not internal and not world"


def is_version_only(path, diff):
    """True when every changed line of `diff` (git diff -U2) is a haenv version line."""
    hunks = re.split(r"^@@[^\n]*\n?", diff, flags=re.M)[1:]
    changed = [line for h in hunks for line in h.splitlines() if line[:1] in "+-"]
    if not changed or not all(_VERSION_LINE.match(line) for line in changed):
        return False
    return path != "uv.lock" or all(' name = "haenv"' in h for h in hunks)


def _diff(path, base):
    return subprocess.run(["git", "diff", "--no-renames", "-U2", base, "--", path], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


def _under(path, prefix):
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def owners(test):
    """Parent directories own only files outside separately listed child components."""
    path = Path(test)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    matches = [name for name in COMPONENTS if _under(path.as_posix(), "tests/" + name)]
    return [name for name in matches if not any(name != other and _under(other, name) for other in matches)]


def select(paths, *, public=False, base="HEAD"):
    selected = {"tests/guards"} if (ROOT / "tests/guards").is_dir() else set()
    for item in paths:
        path = Path(item)
        path = path.relative_to(ROOT) if path.is_absolute() else path
        label = path.as_posix()
        if label in VERSION_FILES and is_version_only(label, _diff(label, base)):
            selected.add("tests/release")
            continue
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
    parser.add_argument("--base", default="HEAD", help="revision the changed paths are compared with")
    parser.add_argument("--allow-slow", action="store_true",
                        help="permit a selection that falls back to the whole test tree")
    args = parser.parse_args()
    paths = args.paths or subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", args.base], cwd=ROOT, text=True).splitlines()
    tests = select(paths, public=args.public, base=args.base)
    if "tests" in tests and not args.allow_slow:
        wide = [p for p in paths if "tests" in select([p], base=args.base)]
        print("refusing: these paths select the whole test tree: " + ", ".join(wide)
              + "\nmap them to a component, run the default suite once, or pass --allow-slow",
              file=sys.stderr)
        return 2
    if not args.run:
        print("\n".join(tests))
        return 0
    env = dict(os.environ, UV_NO_SYNC="1", UV_FROZEN="1",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT / "src", ROOT, ROOT / "plugins"))))
    command = [sys.executable, "-m", "pytest", *tests, "-n", "64", "--ff", "--durations=30"]
    if args.public:
        command += ["-m", PUBLIC_MARKER]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
