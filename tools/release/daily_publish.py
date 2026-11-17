#!/usr/bin/env python3
"""Gate a public snapshot and prepare one daily commit per destination.

Dry run is the default. Supply --scratch outside the checkout to retain logs,
pack hashes and the plan. Publication requires --push and successful checks.
The source ref must include the development export policy and its audit tools.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
GITHUB = "git@github.com:thetahealth/mirobody-env.git"
ARTIFACTS = ("cases.jsonl", "payloads.jsonl", "audit.jsonl", "message_sources.jsonl")
FINGERPRINT_PROBE = """
import json
from haenv.provenance import anchor as A, declarations as D
from haenv.provenance.judging import criterion_fingerprints
segments = {k: A._fingerprint_of(v) for k,v in vars(D).items()
            if k in ('GENERATION', 'JUDGING', 'RENDER') or k.endswith(('_WORLD', '_JUDGING'))}
print(json.dumps({'segments': segments, 'criteria': criterion_fingerprints()}, sort_keys=True))
"""


class Refused(RuntimeError):
    pass


def run(cmd, *, cwd=None, env=None, log=None, input=None, timeout=7200):
    """Checked subprocess failures stop publication; full diagnostics stay on disk."""
    cwd = cwd or ROOT
    label = list(map(str, cmd))
    if len(label) > 2 and label[1] == "-c":
        label[2] = "<probe>"
    if log:
        print(f"check: {log.parent.name}/{log.name}", flush=True)
        with log.open("w") as out:
            p = subprocess.run(cmd, cwd=cwd, env=env, input=input, text=True,
                               stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        if p.returncode:
            raise Refused(f"{shlex.join(label)}: exit {p.returncode}; see {log}")
        return log.read_text()
    p = subprocess.run(cmd, cwd=cwd, env=env, input=input, text=True,
                       capture_output=True, timeout=timeout)
    if p.returncode:
        raise Refused(f"{shlex.join(label)}: exit {p.returncode}: {p.stderr[-2000:]}")
    return p.stdout.strip()


def git(*args, **kw):
    return run(["git", *args], **kw)


def resolve(ref):
    return git("rev-parse", "--verify", ref + "^{commit}")


def fetch(url, ref):
    # A login shell selects the operator's SSH configuration. No remote ref is written.
    return run(["bash", "-lc", shlex.join([
        "git", "fetch", "--no-tags", "--no-auto-maintenance", url,
        f"refs/heads/main:{ref}"])], timeout=900)


def remote_head(url):
    out = run(["bash", "-lc", shlex.join(["git", "ls-remote", url, "refs/heads/main"])], timeout=120)
    rows = out.splitlines()
    if len(rows) != 1 or not re.fullmatch(r"[0-9a-f]{40}\s+refs/heads/main", rows[0]):
        raise Refused(f"cannot identify main at {url}")
    return rows[0].split()[0]


def archive(ref, target):
    target.mkdir()
    tar = target.parent / (target.name + ".tar")
    git("archive", "--format=tar", "--output=" + str(tar), ref)
    with tarfile.open(tar) as stream:
        # Refuse links and traversal before extracting an arbitrary source ref.
        members = stream.getmembers()
        if any(m.issym() or m.islnk() or not (target / m.name).resolve().is_relative_to(target)
               for m in members):
            raise Refused("source archive contains a link or an escaping path")
        stream.extractall(target, members=members)
    tar.unlink()
    # The exporter intentionally reads tracked names; give this isolated snapshot an index.
    git("init", "-q", cwd=target)
    git("add", "--force", "--all", cwd=target)
    # Preserve the selected revision for provenance probes without another checkout.
    objects = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "objects"
    (target / ".git/objects/info/alternates").write_text(str(objects) + "\n")
    git("update-ref", "refs/heads/snapshot", resolve(ref), cwd=target)
    git("symbolic-ref", "HEAD", "refs/heads/snapshot", cwd=target)


def files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                  and not {".git", ".venv", "__pycache__"}.intersection(p.relative_to(root).parts)
                  and p.suffix != ".pyc")


def tree_for(export, index, paths=None):
    names = files(export) if paths is None else sorted(paths)
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    git("read-tree", "--empty", env=env)
    git("--work-tree=" + str(export), "add", "--force", "--pathspec-from-file=-",
        "--pathspec-file-nul", env=env, input="\0".join(names) + "\0")
    tree = git("write-tree", env=env)
    tracked = [p for p in git("ls-tree", "-r", "--name-only", "-z", tree).split("\0") if p]
    if tracked != names:
        raise Refused("staged file list differs from export")
    for name in names:
        if git("rev-parse", f"{tree}:{name}") != git("hash-object", str(export / name)):
            raise Refused(f"staged bytes differ: {name}")
    return tree


def readme_version_only(old, new):
    """Only the BibTeX version value may change, in either language."""
    pattern = r"(?m)^(\s*version\s*=\s*\{)[^}\n]+(\}.*)$"
    if len(re.findall(pattern, old)) != 1 or len(re.findall(pattern, new)) != 1:
        raise Refused("README must have exactly one BibTeX version")
    if re.sub(pattern, r"\1<VERSION>\2", old) != re.sub(pattern, r"\1<VERSION>\2", new):
        raise Refused("README changes exceed the version string")


def unreleased(text):
    match = re.search(r"(?ms)^## \[Unreleased\]\s*\n(.*?)(?=^## |\Z)", text)
    return re.findall(r"(?m)^- (.+)$", match[1]) if match else []


def commit_message(today, old, new):
    previous = unreleased(old)
    delta = [line for line in unreleased(new) if line not in previous]
    if not delta:
        raise Refused("changed public tree has no CHANGELOG Unreleased delta")
    return f"chore: daily update {today}\n\n" + "\n".join("- " + line for line in delta) + "\n"


def already_published(parent, subject):
    return subject in git("log", "--format=%s", "--fixed-strings", "--grep=" + subject, parent).splitlines()


def clean_env(tmp):
    # Builds/tests get no solver credentials, source redirects or world overrides.
    transport = {"PATH", "HOME", "LANG", "LC_ALL", "SSL_CERT_FILE", "HTTP_PROXY", "HTTPS_PROXY",
                 "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k in transport} | {
        "TMPDIR": str(tmp), "UV_INDEX_URL": os.environ.get("UV_INDEX_URL", "https://pypi.org/simple/"), "HAENV_ENV_FILE": "/dev/null",
        "UV_CACHE_DIR": str(tmp.parent / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(tmp.parent / "python"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "HAENV_SCAN_WORKERS": "16"}


def private_material(rel, text, views):
    """Private publication content, with exact exceptions for two exclusion guards."""
    prohibited = ("Anon" + "Env", "www" + "2027", "physician" + "-review")
    if any(term.casefold() in view.casefold() for view in [rel, *views] for term in prohibited):
        return [rel + ": private publication material"]
    held = "paper" + "/"
    guards = {
        "tools/generation/p4_gen_job.py": 'name.startswith(("tests/", "docs/", "' + held + '"))',
        "tests/release/test_link_targets.py": "not p.startswith('" + held + "')",
    }
    if held in rel.casefold() or any(held in view.casefold() for view in views):
        # These literals remove private paths from traversal; they contain no
        # publication data. Bind the exact snippet and occurrence count so the
        # exception cannot hide additional references in either file.
        guard = guards.get(rel)
        if held in rel.casefold() or not guard or text.casefold().count(held) != 1 or guard not in text:
            return [rel + ": private publication path"]
    return []


def privacy(source, export, env, log):
    # Reuse both existing detectors without rewriting any exported bytes.
    probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.release import make_public_tree as M, scrub_local_paths as S, release_check as R
from tools.release.daily_publish import private_material
base = Path(sys.argv[2])
paths = [p.relative_to(base).as_posix() for p in base.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts]
deferred = []
bad = M.scan(paths, base, deferred, workers=16) + M.prototype_path_findings(paths)
bad += deferred
ok, detail = R.check_gate()
if ok is not True:
    bad.append('release export gate: ' + detail)
for rel in paths:
    text = (base / rel).read_bytes().decode('utf-8', errors='ignore')
    if S.PROBE.search(text):
        bad.append(rel + ': local path')
    bad += private_material(rel, text, M.views(rel, text))
print('\\n'.join(bad) if bad else 'privacy passed: zero findings or deferrals')
raise SystemExit(bool(bad))
"""
    run([sys.executable, "-c", probe, str(source), str(export)], cwd=source, env=env, log=log)


def setup(export, version, env, logs):
    run(["uv", "sync", "--project", str(export), "--python", version,
         "--group", "test", "--frozen"], env=env, log=logs / f"setup-{export.name}-{version}.log")
    python = export / ".venv/bin/python"
    # A public test builds a wheel offline. Warm the exact isolated build
    # environment; installing the backend alone can reuse packages whose wheel
    # URLs were never cached for this index or interpreter.
    run(["uv", "build", "--wheel", "--python", version, "--out-dir",
         str(logs / f"wheel-warmup-{export.name}-{version}")], cwd=export, env=env,
        log=logs / f"build-backend-{export.name}-{version}.log")
    return python


def suites(export, python, version, workers, env, logs, scope="public"):
    for language in ("en", "zh"):
        tiers = (("release", "not disk and not internal"),) if scope == "release" else (
            ("unit", "not disk and not internal and not world"), ("world", "world"))
        for tier, marks in tiers:
            run([str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-n", str(workers), "--dist", "worksteal", "-m", marks,
                 *(["tests/release"] if scope == "release" else [])],
                cwd=export, env=env | {"HAENV_LANG": language},
                log=logs / f"tests-{version}-{language}-{tier}.log")


def fingerprints(export, python, env):
    return json.loads(run([str(python), "-c", FINGERPRINT_PROBE], cwd=export, env=env))


def prepare_pack_checkout(export, tree):
    """Give preview caches an exact tracked inventory without creating a commit."""
    if (export / ".git").exists():
        raise Refused("public snapshot already has Git metadata")
    git("init", "-q", cwd=export)
    objects = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "objects"
    (export / ".git/objects/info/alternates").write_text(str(objects) + "\n")
    # Read the already verified public tree, never dependency installs or test outputs.
    git("read-tree", tree, cwd=export)


def audit_dependencies(python, version, env, logs):
    run(["uv", "pip", "install", "--python", str(python), "scikit-learn", "joblib"],
        env=env, log=logs / f"pack-deps-{version}.log")


def run_pack(cmd, *, cwd, env, log, stop, allow_failure=False):
    """Cancel this owned process group when another pack check fails."""
    if stop.is_set():
        raise Refused("pack check cancelled after an earlier failure")
    print(f"check: {log.parent.parent.parent.name}/{log.parent.parent.name}/{log.parent.name}", flush=True)
    with log.open("w") as out:
        child = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 7200
        try:
            while child.poll() is None:
                if stop.wait(0.2):
                    raise Refused("pack check cancelled after an earlier failure")
                if time.monotonic() >= deadline:
                    raise Refused(f"pack check timed out; see {log}")
            if child.returncode and not allow_failure:
                raise Refused(f"{shlex.join(map(str, cmd))}: exit {child.returncode}; see {log}")
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

    return child.returncode


def pack_inputs(root):
    """Conservative build inputs, including tools, data, configuration and dependency lock.

    Only release/dev tooling, tests and prose are omitted: they are never read by
    default pack generation. Code additions/deletions invalidate reuse too.
    """
    names = []
    for folder in ("src", "plugins", "tools", "benchmarks", "registry", "config"):
        for base, dirs, leaves in os.walk(root / folder):
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", ".venv"}
                       and Path(base, d).relative_to(root).as_posix() not in {"tools/release", "tools/dev"}]
            names.extend((Path(base) / name).relative_to(root).as_posix()
                         for name in leaves if not name.endswith(".pyc"))
    names += [name for name in ("pyproject.toml", "uv.lock", "config.yaml") if (root / name).is_file()]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}


def pack_snapshot(out, returncode, reports=None):
    """Hash exact pool/published outputs and location-normalized diagnostic reports.

    Only known absolute location fields in JSON reports are relocated. Case rows,
    generated YAML, report findings and every other byte remain significant.
    """
    record = json.loads((out / "make_pack.json").read_text())
    steps = record["steps"]
    if not steps or any(type(code) is not int or code < 0 for code in steps.values()):
        raise Refused(f"invalid pack step statuses: {out}")
    if type(returncode) is not int or returncode < 0 or bool(returncode) != any(steps.values()):
        raise Refused(f"pack exit/status mismatch: {out}")
    required = {"gen", "build", "audit", "invariants", "pack_build", "pack_rows_identical"}
    if not returncode and not required.issubset(steps):
        raise Refused(f"incomplete successful pack: {out}")
    paths = {}
    for field in ("batch", "pack_batch"):
        if record.get(field):
            batch = Path(record[field])
            for name in ARTIFACTS:
                path = batch / name
                if not path.is_file():
                    raise Refused(f"missing persisted pack output: {path}")
                paths[field + "/" + name] = path
    if not record.get("job") or not record.get("batch"):
        early_failure = any(steps.get(step, 0) != 0 for step in ("ref", "gen", "build", "shortcut_balance"))
        if not early_failure:
            raise Refused(f"missing generated job or persisted pool: {out}")
        # Early failures may persist only a partial job/batch. Compare the exact
        # available inventory and failure diagnostics instead of inventing outputs.
        for path in sorted(out.rglob("*")):
            if path.is_file() and (path.name in ARTIFACTS or path.name.endswith(".job.yaml")):
                name = re.sub(r"/\d{8}-\d{6}/", "/<batch>/", path.relative_to(out).as_posix())
                if name in paths:
                    raise Refused(f"ambiguous partial pack outputs: {out}")
                paths[name] = path
        for step in ("ref", "gen", "build"):
            if steps.get(step):
                paths[step + ".log"] = out / (step + ".log")
    for field in ("job", "pack_job"):
        if record.get(field):
            paths[field] = Path(record[field])
    for folder in ("audit", "invariants"):
        for path in sorted((out / folder).rglob("*")):
            if path.is_file():
                paths[path.relative_to(out).as_posix()] = path
    # An audit can stop before writing JSON (the released P2 failure does this).
    # Its diagnostic remains a required report, never an unrecorded exception.
    if "audit" in steps:
        paths["audit.log"] = out / "audit.log"
    if steps.get("audit") == 0 and not (out / "audit" / "audit.json").is_file():
        raise Refused(f"missing audit report: {out}")
    if steps.get("invariants") == 0 and not list((out / "invariants").glob("invariants-*.json")):
        raise Refused(f"missing invariant report: {out}")
    hashes, raw = {}, {}
    locations = {key: record.get(key) for key in ("batch", "pack_batch")}
    locations["purchases"] = str(out / "invariants" / "m2_purchases.jsonl")
    for name, path in paths.items():
        data = path.read_bytes()
        raw[name] = hashlib.sha256(data).hexdigest()
        if name.startswith(("audit/", "invariants/")) and path.suffix == ".json":
            # Replace exactly the top-level location value, preserving JSON byte
            # formatting and all other content, including unexpected path values.
            report = json.loads(data)
            for field, location in locations.items():
                if location and report.get(field) == location:
                    pattern = (rb'("' + field.encode() + rb'"\s*:\s*)'
                               + re.escape(json.dumps(location).encode()))
                    data = re.sub(pattern, lambda match: match[1] + json.dumps("<" + field + ">").encode(),
                                  data, count=1)
        if reports is not None and (name == "audit.log" or (
                name.startswith(("audit/", "invariants/")) and path.suffix in {".json", ".md"})):
            compared = reports / name
            compared.parent.mkdir(parents=True, exist_ok=True)
            compared.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"returncode": returncode, "steps": steps, "artifacts": hashes,
            "errors": {key: value for key, value in record.items() if key.endswith("_error")}}, raw


def pack_identity(baseline, candidate, python, env, work, reuse=None):
    """Rebuild the four released packs in both languages with default flags.

    Certify equivalence to the released code, including inherited failures.
    Compare step statuses, pool/published rows, generated jobs and reports.
    Semantic stamps must match independently without any restamping.
    """
    protected = ("benchmarks/frozen/", "benchmarks/results/", "registry/anchors/")
    frozen = [p for p in files(baseline) if p.startswith(protected)]
    for name in frozen:
        if not (candidate / name).is_file() or (baseline / name).read_bytes() != (candidate / name).read_bytes():
            raise Refused(f"frozen artifact or anchor changed: {name}")
    if [p for p in files(candidate) if p.startswith(protected)] != frozen:
        raise Refused("frozen artifact or anchor file set changed")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    stop = threading.Event()
    work.mkdir(parents=True, exist_ok=True)
    packs = ("m2", "p2", "p3", "p4")
    def compare(task):
        if stop.is_set():
            raise Refused("pack check cancelled after an earlier failure")
        pack, language = task
        outputs, provenance = [], {}
        for label, root in (("baseline", baseline), ("candidate", candidate)):
            reports = work / "reports" / label / language / pack
            cached = (reuse or {}).get(f"{label}/{language}/{pack}")
            if cached:
                if cached["inputs"] != pack_inputs(root):
                    raise Refused(f"reused pack inputs changed: {label}/{language}/{pack}")
                out = Path(cached["out"])
                snapshot, raw = pack_snapshot(out, cached["returncode"], reports)
                if snapshot != cached["snapshot"] or raw != cached["raw"]:
                    raise Refused(f"reused pack outputs changed: {out}")
                print(f"reuse: {label}/{language}/{pack}: {out}", flush=True)
            else:
                out = work / label / language / pack
                out.mkdir(parents=True)
                code = run_pack([str(python), str(root / "tools/generation/make_pack.py"),
                     "--pack", pack, "--out", str(out), "--jobs", "8", "--gen-workers", "8"],
                    cwd=root, env=env | {"PYTHONPATH": str(root / "src"), "HAENV_LANG": language},
                    log=out / "make-pack.log", stop=stop, allow_failure=True)
                if not (out / "make_pack.json").is_file():
                    raise Refused(f"pack exited {code} without a completed record: {out}")
                snapshot, raw = pack_snapshot(out, code, reports)
            outputs.append(snapshot)
            provenance[label] = {"out": str(out), "reports": str(reports), "reused": bool(cached), "raw": raw}
        if outputs[0] != outputs[1]:
            raise Refused(f"rebuilt public pack differs: {pack}@{language}: {outputs}")
        inherited = {step: code for step, code in outputs[0]["steps"].items() if code}
        result = {"status": "inherited_failure" if inherited else "passed",
                  **outputs[0], "inherited_failure": inherited, "evidence": provenance}
        return f"{pack}@{language}", result
    # Two pack builds, each with eight workers: at most sixteen pack workers.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(compare, (pack, lang)) for lang in ("en", "zh") for pack in packs]
        evidence = {}
        try:
            for result in as_completed(pending):
                key, value = result.result()
                evidence[key] = value
        except BaseException:
            stop.set()
            for result in pending:
                result.cancel()
            raise
    (work / "identity.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def push_command(url, commit):
    return ["bash", "-lc", shlex.join(["git", "push", url, commit + ":refs/heads/main"])]


def execute(a, work):
    logs = work / "logs"
    logs.mkdir()
    tmp = a.scratch / "tmp"
    tmp.mkdir(exist_ok=True)
    env = clean_env(tmp)
    source_sha = resolve(a.source)
    fetch(a.github_url, "refs/daily-publish/github-main")
    parent = resolve("refs/daily-publish/github-main")
    if a.github_main and resolve(a.github_main) != parent:
        raise Refused("GitHub main changed from the supplied last-main ref")
    archive(source_sha, work / "source")
    source, export = work / "source", work / "public"
    exporter = source / "tools/release/make_public_tree.py"
    if not exporter.is_file():
        raise Refused("source ref has no public export policy")
    run([sys.executable, str(exporter), "--out", str(export)], cwd=source, env=env,
        log=logs / "export.log")
    privacy(source, export, env, logs / "privacy.log")
    tree = tree_for(export, work / "public.index")
    if tree == git("rev-parse", parent + "^{tree}"):
        print("nothing to publish", flush=True)
        return {"status": "nothing to publish", "tree": tree, "parent": parent}
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    subject = f"chore: daily update {today}"
    if already_published(parent, subject):
        raise Refused("today already has a daily commit; wait until the next UTC day")
    for name in ("README.md", "README.zh-CN.md"):
        readme_version_only(git("show", f"{parent}:{name}") + "\n", (export / name).read_text())
    message = commit_message(today, git("show", f"{parent}:CHANGELOG.md"), (export / "CHANGELOG.md").read_text())
    (work / "candidate.json").write_text(json.dumps({
        "status": "checks pending", "source": source_sha, "tree": tree, "parent": parent,
        "message": message, "files": git("diff", "--name-status", parent, tree).splitlines()
    }, indent=2) + "\n")
    baseline_sha = resolve(a.baseline)
    if baseline_sha != parent:
        raise Refused("pack baseline must equal fetched GitHub main")
    if not git("show", f"{baseline_sha}:pyproject.toml").count('version = "1.4.0"'):
        raise Refused("pack baseline must be the 1.4.0 release")
    archive(baseline_sha, work / "baseline")
    baseline = work / "baseline"
    for version in ("3.10", "3.12"):
        python = setup(export, version, env, logs)
        suites(export, python, version, a.workers, env, logs, getattr(a, "test_scope", "public"))
        # Install baseline in its own fresh environment with the same public lock.
        base_python = setup(baseline, version, env, logs)
        before, after = fingerprints(baseline, base_python, env), fingerprints(export, python, env)
        if before != after:
            raise Refused(f"1.4.0 semantic fingerprints changed on Python {version}: {before} != {after}")
        (logs / f"fingerprints-{version}.json").write_text(json.dumps(after, indent=2) + "\n")
    # Both interpreter suites finish before the expensive identity gate. The
    # default pack rebuild compares old/new code in one Python 3.12 environment.
    audit_dependencies(python, "3.12", env, logs)
    prepare_pack_checkout(export, tree)
    try:
        reuse_path = getattr(a, "reuse_packs", None)
        reuse = json.loads(reuse_path.read_text()) if reuse_path else None
        evidence = pack_identity(baseline, export, python, env, work / "packs-3.12", reuse)
    finally:
        shutil.rmtree(export / ".git")
    # Detect any tracked-byte mutation by tests or setup before creating the commit.
    manifest = [p for p in git("ls-tree", "-r", "--name-only", "-z", tree).split("\0") if p]
    if tree_for(export, work / "verified.index", paths=manifest) != tree:
        raise Refused("checks changed the public tree")
    gitlab_url = a.gitlab_url or git("remote", "get-url", "origin")
    fetch(gitlab_url, "refs/daily-publish/gitlab-main")
    gitlab_parent = resolve("refs/daily-publish/gitlab-main")
    if already_published(gitlab_parent, subject):
        raise Refused("GitLab main already has today's daily commit")
    # Check both remote heads for dry runs too, before staging a reviewable commit.
    if remote_head(a.github_url) != parent or remote_head(gitlab_url) != gitlab_parent:
        raise Refused("a remote main advanced during checks; rerun before publishing")
    commit = git("commit-tree", tree, "-p", parent, input=message)
    gitlab_commit = commit if gitlab_parent == parent else git(
        "commit-tree", tree, "-p", gitlab_parent, input=message)
    git("update-ref", f"refs/staged/daily-github/{today}", commit)
    git("update-ref", f"refs/staged/daily-gitlab/{today}", gitlab_commit)
    plan = {"status": "dry run", "source": source_sha, "baseline": baseline_sha,
            "tree": tree, "parent": parent, "commit": commit, "message": message,
            "gitlab_parent": gitlab_parent, "gitlab_commit": gitlab_commit,
            "files": git("diff", "--name-status", parent, commit).splitlines(),
            "packs": evidence,
            "reuse_manifest": ({"path": str(reuse_path),
                                "sha256": hashlib.sha256(reuse_path.read_bytes()).hexdigest()}
                               if reuse_path else None),
            "inherited_failure": {key: value["inherited_failure"] for key, value in evidence.items()
                                  if value.get("inherited_failure")},
            "push_commands": [shlex.join(push_command(a.github_url, commit)),
                              shlex.join(push_command(gitlab_url, gitlab_commit))],
            "checks": {"python": ["3.10", "3.12"], "languages": ["en", "zh"],
                       "tiers": ["release"] if getattr(a, "test_scope", "public") == "release" else ["unit", "world"], "pack_jobs": len(evidence), "pack_python": "3.12",
                       "fingerprints": "identical", "privacy": "passed", "readme": "version only"}}
    receipt = work / "receipt.json"
    receipt.write_text(json.dumps(plan, indent=2) + "\n")
    if a.push:
        if remote_head(a.github_url) != parent or remote_head(gitlab_url) != gitlab_parent:
            raise Refused("a remote main advanced during checks; rerun before publishing")
        plan["status"] = "pushing"
        receipt.write_text(json.dumps(plan, indent=2) + "\n")
        for label, url, sha in (("github", a.github_url, commit), ("gitlab", gitlab_url, gitlab_commit)):
            run(push_command(url, sha), timeout=900)
            plan[label + "_pushed"] = sha
            receipt.write_text(json.dumps(plan, indent=2) + "\n")
        git("update-ref", "refs/github-main", commit)
        plan["status"] = "pushed"
        receipt.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="public/next")
    ap.add_argument("--github-main", help="expected last GitHub main ref (checked against remote)")
    ap.add_argument("--baseline", default="7a5b29b8743bcd839a90e33e9e0e5b06875e0544", help="public 1.4.0 commit")
    ap.add_argument("--github-url", default=GITHUB)
    ap.add_argument("--gitlab-url", help="defaults to origin's URL")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--workers", type=int, choices=range(1, 17), default=16)
    ap.add_argument("--test-scope", choices=("public", "release"), default="public")
    ap.add_argument("--reuse-packs", type=Path, help="input/output-bound manifest of completed local pack builds")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    a.scratch = a.scratch.resolve()
    if a.scratch.is_relative_to(ROOT):
        ap.error("scratch must be outside the checkout")
    a.scratch.mkdir(parents=True, exist_ok=True)
    with (a.scratch / "daily.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error("another daily publisher holds this scratch lock")
        work = Path(tempfile.mkdtemp(prefix="run-", dir=a.scratch))
        print(f"evidence: {work}", flush=True)
        try:
            execute(a, work)
        except (Refused, subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError) as exc:
            (work / "failure.json").write_text(json.dumps({"status": "failed", "reason": str(exc)}, indent=2) + "\n")
            print(f"publication refused: {exc}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
