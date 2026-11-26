"""Daily refreshes gate the exported bytes and never publish during a dry run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pytest

from tools.release import daily_publish as D


@pytest.mark.parametrize("suffix", ["", ","])
def test_readme_allows_only_one_bibtex_version(suffix):
    old = "# Example\n\n  version = {1.6.0}" + suffix + "\n"
    D.readme_version_only(old, old.replace("1.6.0", "1.6.1.dev0"))
    for new in (old + "New paragraph\n", old.replace("Example", "Changed"), old.replace("version", "release")):
        with pytest.raises(D.Refused, match="README"):
            D.readme_version_only(old, new)


def test_message_contains_only_new_unreleased_bullets():
    old = "## [Unreleased]\n\n- Earlier change.\n\n## [1.6.0]\n- Released.\n"
    new = old.replace("- Earlier change.", "- Earlier change.\n- New change.")
    assert D.commit_message("2026-11-17", old, new) == "chore: daily update 2026-11-17\n\n- New change.\n"
    with pytest.raises(D.Refused, match="no CHANGELOG"):
        D.commit_message("2026-11-17", old, old)


def test_build_environment_removes_keys_world_overrides_and_redirects(tmp_path, monkeypatch):
    for key in ("RELAY_API_KEY", "HAENV_WORLD_HISTORY", "HAENV_DATA_ROOT", "PYTHONPATH"):
        monkeypatch.setenv(key, "must not reach a public build")
    monkeypatch.setenv("UV_INDEX_URL", "https://packages.example/simple/")
    env = D.clean_env(tmp_path)
    assert not set(env) & {"RELAY_API_KEY", "HAENV_WORLD_HISTORY", "HAENV_DATA_ROOT", "PYTHONPATH"}
    assert env["UV_INDEX_URL"] == "https://packages.example/simple/"
    assert env["TMPDIR"] == str(tmp_path)


def test_publication_scan_distinguishes_exact_exclusion_guards():
    held = "paper" + "/"
    snippet = 'name.startswith(("tests/", "docs/", "' + held + '"))'
    rel = "tools/generation/p4_gen_job.py"
    assert D.private_material(rel, snippet, [snippet]) == []
    for name, text in ((rel, snippet + held + "draft.md"), (held + "draft.md", "draft"),
                       ("CONTRIBUTING.md", "tests/" + held), (rel, "Anon" + "Env")):
        assert D.private_material(name, text, [text])


@pytest.fixture
def daily_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(D, "ROOT", repo)
    D.git("init", "-q")
    D.git("config", "user.name", "Test")
    D.git("config", "user.email", "test@example.com")
    readme = "# Example\n\n  version = {1.6.0}\n"
    for name in ("README.md", "README.zh-CN.md"):
        (repo / name).write_text(readme)
    (repo / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "1.6.0"\n')
    (repo / "CHANGELOG.md").write_text("## [1.6.0]\n\n- Released.\n")
    D.git("add", ".")
    D.git("commit", "-qm", "release")
    baseline = D.resolve("HEAD")
    gitlab = baseline  # GitLab main is an ancestor of the source branch.
    for name in ("README.md", "README.zh-CN.md", "pyproject.toml"):
        p = repo / name
        p.write_text(p.read_text().replace("1.6.0", "1.6.1.dev0"))
    (repo / "CHANGELOG.md").write_text("## [Unreleased]\n\n- New feature.\n\n## [1.6.0]\n")
    exporter = repo / "tools/release" / ("make_" + "public_tree.py")
    exporter.parent.mkdir(parents=True)
    exporter.write_text("import pathlib, shutil, sys\n"
                        "out = pathlib.Path(sys.argv[2]); out.mkdir()\n"
                        "for name in ('README.md','README.zh-CN.md','pyproject.toml','CHANGELOG.md'):\n"
                        "    shutil.copy2(name, out / name)\n")
    D.git("add", ".")
    D.git("commit", "-qm", "new feature")
    calls = []
    def fetch(url, ref):
        calls.append(("fetch", url))
        D.git("update-ref", ref, baseline if url == "github" else gitlab)
    monkeypatch.setattr(D, "fetch", fetch)
    monkeypatch.setattr(D, "remote_head", lambda url: baseline if url == "github" else gitlab)
    monkeypatch.setattr(D, "privacy", lambda *args: calls.append(("privacy",)))
    monkeypatch.setattr(D, "setup", lambda root, version, *args: calls.append(("setup", root.name, version)) or Path(sys.executable))
    monkeypatch.setattr(D, "suites", lambda root, py, version, *args: calls.append(("suite", version)))
    monkeypatch.setattr(D, "fingerprints", lambda *args: {"world": "unchanged"})
    monkeypatch.setattr(D, "audit_dependencies", lambda *args: calls.append(("audit dependencies",)))
    monkeypatch.setattr(D, "pack_identity", lambda *args: calls.append(("packs",)) or {"job@en": {"sha": "same"}})
    for var in D.IDENTITY_VARS:
        monkeypatch.setenv(var, "public@example.org" if var.endswith("EMAIL") else "Public Name")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    args = argparse.Namespace(source="HEAD", github_url="github", github_main=baseline, baseline=baseline,
                              gitlab_url="gitlab", workers=2, push=False, scratch=scratch)
    return repo, scratch, args, calls


def test_dry_run_builds_one_commit_per_parent_after_all_gates(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    original = D.run
    def run(cmd, **kwargs):
        assert not (cmd[:2] == ["bash", "-lc"] and "git push " in cmd[2])
        return original(cmd, **kwargs)
    monkeypatch.setattr(D, "run", run)
    result = D.execute(args, work)
    assert result["status"] == "dry run"
    assert D.git("rev-parse", result["commit"] + "^@") == args.github_main
    assert result["gitlab_commit"] == result["source"] and result["gitlab_mode"] == "fast-forward"
    assert D.git("rev-parse", result["commit"] + "^{tree}") == result["tree"]
    assert D.git("log", "-1", "--format=%an <%ae>|%cn <%ce>", result["commit"]) == \
        "Public Name <public@example.org>|Public Name <public@example.org>"
    assert [("suite", "3.10"), ("suite", "3.12")] == [c for c in calls if c[0] == "suite"]
    assert len([c for c in calls if c[0] == "packs"]) == 1
    assert json.loads((work / "receipt.json").read_text()) == result
    assert "refs/github-main" not in D.git("for-each-ref", "--format=%(refname)")
    assert D.resolve("HEAD") == result["source"]


def test_equal_public_tree_skips_suites_and_commit_creation(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    tree = D.git("rev-parse", args.github_main + "^{tree}")
    monkeypatch.setattr(D, "tree_for", lambda *args: tree)
    result = D.execute(args, work)
    assert result["status"] == "nothing to publish"
    assert not any(c[0] in ("suite", "setup", "packs") for c in calls)


def test_daily_limit_survives_an_intervening_commit(daily_repo):
    repo, scratch, args, calls = daily_repo
    subject = "chore: daily update 2026-11-17"
    daily = D.git("commit-tree", "HEAD^{tree}", "-p", "HEAD", input=subject + "\n")
    later = D.git("commit-tree", "HEAD^{tree}", "-p", daily, input="Another change\n")
    assert D.already_published(later, subject)
    assert not D.already_published(args.github_main, subject)


def test_pack_checkout_tracks_only_the_verified_tree(daily_repo):
    repo, scratch, args, calls = daily_repo
    public = scratch / "public"
    public.mkdir()
    (public / "README.md").write_text("public\n")
    tree = D.tree_for(public, scratch / "index")
    extra = public / ".venv" / "private.txt"
    extra.parent.mkdir()
    extra.write_text("test dependency output")
    D.prepare_pack_checkout(public, tree)
    assert D.git("ls-files", cwd=public) == "README.md"


def test_verified_manifest_keeps_nested_assets_and_excludes_generated_outputs(daily_repo):
    repo, scratch, args, calls = daily_repo
    public = scratch / "public"
    asset = public / "examples" / "artifacts" / "sample.txt"
    asset.parent.mkdir(parents=True)
    asset.write_text("approved fixture")
    manifest = D.files(public)
    tree = D.tree_for(public, scratch / "before.index")
    for parts in ((".tmpwork", "contracts"), ("artifacts",), ("dist",)):
        generated = public.joinpath(*parts) / "generated.txt"
        generated.parent.mkdir(parents=True)
        generated.write_text("test output")
    assert D.tree_for(public, scratch / "after.index", paths=manifest) == tree
    assert D.git("ls-tree", "-r", "--name-only", tree) == asset.relative_to(public).as_posix()


@pytest.mark.parametrize("remote_moved", [False, True])
def test_push_requires_all_checks_and_unchanged_remote_heads(daily_repo, monkeypatch, remote_moved):
    repo, scratch, args, calls = daily_repo
    args.push = True
    work = scratch / "run"
    work.mkdir()
    original = D.run
    sent = []
    def run(cmd, **kwargs):
        if cmd[:2] == ["bash", "-lc"] and cmd[2].startswith("git push "):
            assert len([c for c in calls if c[0] == "suite"]) == 2
            assert len([c for c in calls if c[0] == "packs"]) == 1
            sent.append(cmd)
            return "mocked: no network mutation"
        return original(cmd, **kwargs)
    monkeypatch.setattr(D, "run", run)
    def head(url):
        if remote_moved:
            return "0" * 40
        return D.resolve("refs/daily-publish/" + url + "-main")
    monkeypatch.setattr(D, "remote_head", head)
    if remote_moved:
        with pytest.raises(D.Refused, match="remote main advanced"):
            D.execute(args, work)
        assert not sent
    else:
        result = D.execute(args, work)
        assert len(sent) == 2
        assert result["status"] == "pushed"
        assert D.resolve("refs/github-main") == result["github_pushed"] == result["commit"]
        assert result["gitlab_pushed"] == result["gitlab_commit"]


def test_failed_suite_prevents_commit_creation_and_push(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    args.push = True
    def refuse(*args):
        raise D.Refused("test failure")
    monkeypatch.setattr(D, "suites", refuse)
    original = D.git
    def git(*args, **kwargs):
        assert "commit-tree" not in args and "push" not in args
        return original(*args, **kwargs)
    monkeypatch.setattr(D, "git", git)
    with pytest.raises(D.Refused, match="test failure"):
        D.execute(args, work)
    assert not (work / "receipt.json").exists()
    assert not any(call[0] == "packs" for call in calls)


@pytest.fixture
def pack_roots(tmp_path):
    roots = [tmp_path / name for name in ("old", "new")]
    maker = r'''import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--pack'); p.add_argument('--out')
p.add_argument('--jobs'); p.add_argument('--gen-workers')
a = p.parse_args(); out = Path(a.out); batch = out / 'published'; batch.mkdir()
root = Path(__file__).resolve().parents[2]
config = json.loads((root / 'config.json').read_text())
for name in ('cases.jsonl', 'payloads.jsonl', 'audit.jsonl', 'message_sources.jsonl'):
    (batch / name).write_text(json.dumps({'gold': config.get(name, 'fixed')}) + '\n')
job = out / 'job.yaml'; job.write_text(config.get('job', 'fixed job'))
steps = dict.fromkeys(('gen','build','audit','invariants','pack_build','pack_rows_identical'), 0)
failed = config.get('failure', False)
if failed: steps = {'gen': 0, 'build': 0, 'audit': 1}
(out / 'audit.log').write_text('audit failed' if failed else 'audit passed')
if not failed:
    (out / 'audit').mkdir(); (out / 'invariants').mkdir()
    (out / 'audit' / 'audit.json').write_text(json.dumps({'batch': str(batch), 'pass': True}))
    (out / 'invariants' / 'invariants-test.json').write_text(json.dumps({'batch': str(batch), 'findings': config.get('report', [])}))
record = {'job': str(job), 'batch': str(batch), 'steps': steps}
if not failed: record.update(pack_batch=str(batch), pack_job=str(job))
(out / 'make_pack.json').write_text(json.dumps(record))
sys.exit(int(failed))
'''
    for root in roots:
        tool = root / "tools/generation/make_pack.py"
        tool.parent.mkdir(parents=True)
        tool.write_text(maker)
        (root / "config.json").write_text("{}")
    return roots


def test_pack_identity_compares_complete_persisted_outputs(tmp_path, pack_roots):
    result = D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), tmp_path / "evidence")
    assert set(result) == {p + "@" + lang for p in ("m2", "p2", "p3", "p4") for lang in ("en", "zh")}
    assert all(value["status"] == "passed" and not value["inherited_failure"] for value in result.values())
    reports = result["m2@en"]["evidence"]
    before = Path(reports["baseline"]["reports"]) / "audit" / "audit.json"
    after = Path(reports["candidate"]["reports"]) / "audit" / "audit.json"
    assert before.read_bytes() == after.read_bytes()
    assert json.loads(before.read_text())["batch"] == "<batch>"
    assert set(D.ARTIFACTS) == {name.split("/")[-1] for name in result["p3@zh"]["artifacts"] if name.startswith("pack_batch/")}


def test_identical_inherited_failure_passes_and_is_recorded(tmp_path, pack_roots):
    for root in pack_roots:
        (root / "config.json").write_text(json.dumps({"failure": True}))
    result = D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), tmp_path / "evidence")
    assert all(value["status"] == "inherited_failure" and value["inherited_failure"] == {"audit": 1}
               for value in result.values())
    assert json.loads((tmp_path / "evidence" / "identity.json").read_text()) == result


def test_identical_early_failure_compares_partial_outputs_and_diagnostics(tmp_path):
    snapshots = []
    for label in ("baseline", "candidate"):
        out = tmp_path / label; out.mkdir()
        (out / "make_pack.json").write_text(json.dumps({"steps": {"gen": 1}}))
        (out / "gen.log").write_text("pre-existing generator failure")
        snapshots.append(D.pack_snapshot(out, 1)[0])
    assert snapshots[0] == snapshots[1]
    (tmp_path / "candidate/gen.log").write_text("new failure")
    assert D.pack_snapshot(tmp_path / "candidate", 1)[0] != snapshots[0]


def test_candidate_only_failure_fails(tmp_path, pack_roots):
    (pack_roots[1] / "config.json").write_text(json.dumps({"failure": True}))
    with pytest.raises(D.Refused, match="rebuilt public pack differs"):
        D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), tmp_path / "evidence")


@pytest.mark.parametrize("artifact", [*D.ARTIFACTS, "job", "report"])
def test_differing_bytes_with_same_status_fail(tmp_path, pack_roots, artifact):
    (pack_roots[1] / "config.json").write_text(json.dumps({artifact: ["changed"] if artifact == "report" else "changed"}))
    with pytest.raises(D.Refused, match="rebuilt public pack differs"):
        D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), tmp_path / "evidence")


def test_inherited_failure_is_in_commit_receipt(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    monkeypatch.setattr(D, "pack_identity", lambda *args: {"p2@en": {
        "status": "inherited_failure", "inherited_failure": {"audit": 1}}})
    work = scratch / "run"; work.mkdir()
    result = D.execute(args, work)
    receipt = json.loads((work / "receipt.json").read_text())
    assert receipt["commit"] == result["commit"]
    assert receipt["inherited_failure"] == {"p2@en": {"audit": 1}}


def test_reused_pack_is_bound_to_inputs_and_output_bytes(tmp_path, pack_roots):
    first = tmp_path / "first"
    result = D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), first)
    reuse = {}
    for key, value in result.items():
        pack, lang = key.split("@")
        for label, root in zip(("baseline", "candidate"), pack_roots):
            out = first / label / lang / pack
            snapshot, raw = D.pack_snapshot(out, 0)
            reuse[f"{label}/{lang}/{pack}"] = dict(out=str(out), inputs=D.pack_inputs(root),
                                                    returncode=0, snapshot=snapshot, raw=raw)
    second = tmp_path / "second"
    reused = D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), second, reuse)
    assert all(arm["reused"] for value in reused.values() for arm in value["evidence"].values())
    tool = pack_roots[1] / "tools/generation/make_pack.py"
    original = tool.read_bytes()
    tool.write_bytes(original + b"\n# changed input\n")
    with pytest.raises(D.Refused, match="reused pack inputs changed"):
        D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), tmp_path / "third", reuse)
    # Restore every arm's input identity before testing output tampering: any
    # concurrent pack can finish first, so no stale input rejection may remain.
    tool.write_bytes(original)
    (first / "baseline" / "en" / "m2" / "published" / "cases.jsonl").write_text("changed")
    with pytest.raises(D.Refused, match="reused pack outputs changed"):
        D.pack_identity(*pack_roots, Path(sys.executable), D.clean_env(tmp_path), tmp_path / "fourth", reuse)


def test_failed_pack_cancels_a_slow_parallel_builder(tmp_path):
    roots = [tmp_path / name for name in ("old", "new")]
    maker = '''import argparse, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--pack'); p.add_argument('--out')
p.add_argument('--jobs'); p.add_argument('--gen-workers')
a = p.parse_args(); out = Path(a.out)
if a.pack == 'm2':
    (out / 'started.json').write_text(str(os.getpid()))
    time.sleep(60)
else:
    for _ in range(50):
        if (out.parent / 'm2' / 'started.json').exists(): break
        time.sleep(.1)
    sys.exit(7)
'''
    for root in roots:
        tool = root / "tools/generation/make_pack.py"
        tool.parent.mkdir(parents=True)
        tool.write_text(maker)
    evidence = tmp_path / "evidence"
    start = time.monotonic()
    with pytest.raises(D.Refused, match="exited 7"):
        D.pack_identity(*roots, Path(sys.executable), D.clean_env(tmp_path), evidence)
    assert time.monotonic() - start < 15
    pid = int((evidence / "baseline" / "en" / "m2" / "started.json").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_gitlab_main_off_the_source_history_is_refused_before_commit(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    stray = D.git("commit-tree", "HEAD^{tree}", input="Unrelated history\n")
    def fetch(url, ref):
        D.git("update-ref", ref, args.github_main if url == "github" else stray)
    monkeypatch.setattr(D, "fetch", fetch)
    with pytest.raises(D.Refused, match="fast-forward only"):
        D.execute(args, work)
    assert not (work / "receipt.json").exists()


def test_missing_operator_identity_is_refused_before_commit(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    for var in D.IDENTITY_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(D.Refused, match="set the public commit identity"):
        D.execute(args, work)
    assert not (work / "receipt.json").exists()


def test_later_daily_update_uses_the_released_commit_as_pack_baseline(daily_repo, monkeypatch):
    """After the first daily commit, GitHub main is no longer 1.6.0; the reference is a fixed commit."""
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    args.baseline = None
    monkeypatch.setattr(D, "BASELINE", args.github_main)
    result = D.execute(args, work)
    assert result["baseline"] == args.github_main


def test_pack_baseline_off_the_head_history_is_refused(daily_repo, monkeypatch):
    repo, scratch, args, calls = daily_repo
    work = scratch / "run"
    work.mkdir()
    stray = D.git("commit-tree", "HEAD^{tree}", input="Unrelated release" + chr(10))
    args.baseline = None
    monkeypatch.setattr(D, "BASELINE", stray)
    with pytest.raises(D.Refused, match="pack baseline"):
        D.execute(args, work)
    assert not (work / "receipt.json").exists()
