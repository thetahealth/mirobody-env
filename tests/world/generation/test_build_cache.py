"""Identical deterministic inputs reuse complete results; ambiguity always rebuilds."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from _paths import ROOT
from haenv.app.job import CaseSpec
from haenv.i18n.text import Msg
from haenv.world.generation.build_cache import CaseCache, decode, digest, encode, fingerprint_tree, packed


def _case():
    return CaseSpec("cache-case", {"a": 1, "b": 2}, {"index_time_T": 84}, {"name": "single"})


def _cache(tmp_path, context=None, plan=None):
    return CaseCache(tmp_path, context or {"language": "en", "world": "w", "judging": "j"},
                     {"cache-case": plan or asdict(_case())})


def _result():
    from haenv_kernel.schema import RawCase
    text = Msg("saved text", "literal", "cache-test", {"x": (1, 2)})
    raw = RawCase("cache-case", {text: text}, {"prediction_time_T": 84}, {}, [], "no", {}, [], {})
    return (raw, {"case_id": raw.case_id, "emitted": True, "pair": (1, 2)}), None, [(raw.case_id, {"ids": [text]})]


@pytest.mark.parametrize("field", ["sources", "world", "judging", "runtime", "environment", "config",
                                  "language", "max_rounds", "job", "knobs", "version"])
def test_every_context_input_changes_key(tmp_path, field):
    context = {k: ["old"] for k in ("sources", "world", "judging", "runtime", "environment", "config",
                                    "language", "max_rounds", "job", "knobs", "version")}
    old = _cache(tmp_path, context).key_for(_case())
    changed = copy.deepcopy(context)
    changed[field] = ["new"]
    assert _cache(tmp_path, changed).key_for(_case()) != old


@pytest.mark.parametrize("field", ["case_id", "raw", "latent", "geometry"])
def test_every_effective_case_field_changes_key(tmp_path, field):
    cache = _cache(tmp_path)
    case = _case()
    old = cache.key_for(case)
    setattr(case, field, "changed" if field == "case_id" else {"changed": True})
    assert cache.key_for(case) != old


def test_serialized_plan_and_mapping_order_are_bound(tmp_path):
    case = _case()
    original = {"case_id": case.case_id, "raw": {"a": 1, "b": 2}, "extra": None}
    first = _cache(tmp_path, plan=original).key_for(case)
    for plan in ({**original, "extra": False}, {**original, "raw": {"b": 2, "a": 1}}):
        assert _cache(tmp_path, plan=plan).key_for(case) != first
    assert digest([1, 2]) != digest((1, 2))
    assert digest(1) != digest(1.0) != digest(True)
    assert _cache(tmp_path).key_for(CaseSpec("unknown", {}, {})) is None


def test_source_content_names_additions_and_deletions_are_bound(tmp_path):
    path = tmp_path / "one.py"
    path.write_text("one")
    first = fingerprint_tree(tmp_path)
    path.write_text("two")
    assert fingerprint_tree(tmp_path) != first
    path.write_text("one")
    assert fingerprint_tree(tmp_path) == first
    path.rename(tmp_path / "two.py")
    assert fingerprint_tree(tmp_path) != first
    path.write_text("one")
    added = fingerprint_tree(tmp_path)
    path.unlink()
    assert fingerprint_tree(tmp_path) != added


def test_collected_context_binds_environment_config_language_and_runtime_knobs(monkeypatch):
    from types import SimpleNamespace
    from haenv.world.generation import build_cache as C, world_knobs
    from haenv.provenance import anchor
    from haenv.app import job as J
    monkeypatch.setattr(J, "_LOADED_GROUPS", {"haenv.p3"})
    monkeypatch.setattr(C, "fingerprint_tree", lambda root: str(root))
    monkeypatch.setattr(anchor, "world_fingerprint", lambda: "world")
    monkeypatch.setattr(anchor, "judging_fingerprint", lambda: "judging")
    monkeypatch.delenv("HAENV_DATA_ROOT", raising=False)
    monkeypatch.delenv("HAENV_KERNEL_PATH", raising=False)
    job = SimpleNamespace(plugins=["haenv.p3"], world_plugins=[], language="en", job_id="p3", task_type="joint_dx",
                          findings=False, gated=False, multiround=False, cadence_days=28, slices="",
                          probe_id="p3.meds", allow_retired=False)
    baseline = digest(C.context_for(job, {"synth": {"max_rounds": 6}}, 6))
    monkeypatch.setenv("HAENV_NEW_BUILD_SWITCH", "future knob")
    assert digest(C.context_for(job, {"synth": {"max_rounds": 6}}, 6)) != baseline
    monkeypatch.delenv("HAENV_NEW_BUILD_SWITCH")
    assert digest(C.context_for(job, {"synth": {"max_rounds": 5}}, 6)) != baseline
    assert digest(C.context_for(job, {"synth": {"max_rounds": 6}}, 5)) != baseline
    job.language = "zh"
    assert digest(C.context_for(job, {"synth": {"max_rounds": 6}}, 6)) != baseline
    job.language = "en"
    monkeypatch.setattr(world_knobs, "PHYSIO_ENABLED", [not world_knobs.PHYSIO_ENABLED[0]])
    assert digest(C.context_for(job, {"synth": {"max_rounds": 6}}, 6)) != baseline


def test_external_or_ambiguous_plugin_context_disables_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from haenv.world.generation import build_cache as C
    from haenv.app import job as J
    monkeypatch.delenv("HAENV_DATA_ROOT", raising=False)
    monkeypatch.setattr(J, "_LOADED_GROUPS", {"haenv.p3"})
    job = SimpleNamespace(plugins=["custom.p3"], world_plugins=[])
    with pytest.raises(ValueError, match="only builtin"):
        C.context_for(job, {}, 6)
    job.plugins = ["haenv.p3"]
    monkeypatch.setattr(C.importlib.metadata, "entry_points", lambda **kw: [
        SimpleNamespace(name="p3", value="haenv.p3.plugin:judges"),
        SimpleNamespace(name="rival", value="external.plugin:judges")])
    with pytest.raises(ValueError, match="ambiguous"):
        C.context_for(job, {}, 6)


def test_roundtrip_preserves_messages_keys_order_tuples_and_injections(tmp_path):
    cache = _cache(tmp_path)
    key = cache.key_for(_case())
    result = _result()
    cache.write(key, result)
    got = cache.read(key, "cache-case")
    assert got is not None
    assert encode(asdict(got[0][0])) == encode(asdict(result[0][0]))
    assert got[0][1] == result[0][1] and got[2] == result[2]
    assert isinstance(next(iter(got[0][0].user_profile)), Msg)
    assert isinstance(got[0][1]["pair"], tuple)


@pytest.mark.parametrize("damage", ["truncate", "checksum", "key", "version", "payload", "case_id", "extra",
                                   "duplicate", "boolean_version"])
def test_corrupt_or_partial_entries_miss(tmp_path, damage):
    cache = _cache(tmp_path)
    key = cache.key_for(_case())
    cache.write(key, _result())
    path = cache.path_for(key)
    data = json.loads(path.read_bytes())
    if damage == "truncate":
        path.write_bytes(path.read_bytes()[:60])
    elif damage == "duplicate":
        path.write_bytes(b'{"key":"ambiguous",' + path.read_bytes()[1:])
    else:
        if damage in ("checksum", "key", "version"):
            data["sha256" if damage == "checksum" else damage] = "bad"
        elif damage == "extra":
            data["extra"] = "bad"
        elif damage == "boolean_version":
            data["version"] = True
        else:
            value = decode(data["payload"])
            if damage == "case_id":
                value["audit"]["case_id"] = "other"
            else:
                value["injections"] = [[3, "bad"]]
            data["payload"] = encode(value)
            data["sha256"] = hashlib.sha256(packed(data["payload"])).hexdigest()
        path.write_bytes(packed(data))
    assert cache.read(key, "cache-case") is None
    cache.write(key, _result())
    assert cache.read(key, "cache-case") is not None


def test_pool_replays_hits_and_rebuilds_corruption(tmp_path):
    from haenv.world.generation.build_pool import build_all
    import haenv.world.wq as wq
    cache = _cache(tmp_path)
    case = _case()
    def build(cs):
        ra, _, inj = _result()
        for k, v in inj:
            wq.register_injection(k, v)
        return ra
    try:
        wq.enter_scope("cache-test")
        first = build_all([case], build, 2, cache=cache)
        assert cache.hits == 0 and cache.misses == 1 and first[0][2] is None
        wq.enter_scope("cache-test")
        def forbidden(cs):
            raise AssertionError("hit rebuilt")
        second = build_all([case], forbidden, 1, cache=cache)
        assert cache.hits == 1 and second[0][2] is None
        assert wq.injected_manifest(case.case_id) == _result()[2][0][1]
        cache.path_for(cache.key_for(case)).write_text("partial")
        third = build_all([case], build, 1, cache=cache)
        assert cache.misses == 2 and third[0][2] is None
    finally:
        wq.enter_scope("")


def test_exceptions_and_unsupported_results_are_never_cached(tmp_path):
    cache = _cache(tmp_path)
    key = cache.key_for(_case())
    cache.write(key, (None, ("ValueError", "bad"), []))
    assert cache.read(key, "cache-case") is None
    result = _result()
    result[0][1]["ambiguous"] = object()
    cache.write(key, result)
    assert cache.read(key, "cache-case") is None


@pytest.mark.integration
def test_real_p3_twelve_plans_both_arms_cached_uncached_byte_identity(tmp_path):
    """Exercise CLI, rendering, all serializers and batch gates in the session language."""
    document = yaml.safe_load((ROOT / "benchmarks/inputs/p3-meds.job.yaml").read_text())
    document["cases"] = document["cases"][:12]
    job = tmp_path / "p3.job.yaml"
    job.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False))
    env = {**os.environ, "HAENV_DIFFICULTY_R2": "on", "OMP_NUM_THREADS": "1"}
    env.pop("HAENV_BUILD_CACHE", None)
    env.pop("HAENV_CONFIG_SET", None)
    cache = tmp_path / "cache"
    for arm, settings in (("real", ["world.history=long", "world.med_dynamics=true", "background=true"]),
                           ("control", ["world.history=short", "world.med_dynamics=false", "background=false"])):
        batches = []
        for mode in ("off", "fill", "hit"):
            root = tmp_path / f"{arm}-{mode}"
            cmd = [sys.executable, "-m", "haenv", "build", str(job), "--gen", "deterministic", "--fresh",
                   "--gen-workers", "2", "--set", "env_file=/dev/null", "--set", "world.course_coupling=true",
                   "--set", "world.routine_panels=true", "--set", "world.emr_params=calibration15",
                   "--set", "synth.rules_delivery=library"]
            for item in settings:
                cmd.extend(["--set", item])
            if mode != "off":
                cmd.extend(["--build-cache", str(cache)])
            proc = subprocess.run(cmd, env={**env, "HAENV_OUTPUT_ROOT": str(root)}, capture_output=True, text=True)
            (tmp_path / f"{arm}-{mode}.log").write_text(proc.stdout + proc.stderr)
            assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
            if mode == "fill":
                assert "build cache: 0 hits, 12 misses" in proc.stdout
            if mode == "hit":
                assert "build cache: 12 hits, 0 misses" in proc.stdout
            batches.append(next(root.glob("results/*/*/*/cases.jsonl")).parent)
        for name in ("cases.jsonl", "payloads.jsonl", "audit.jsonl", "message_sources.jsonl"):
            assert len({(b / name).read_bytes() for b in batches}) == 1, (arm, name)
        # Only batch timestamps/output locations are volatile across these builds.
        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items() if k not in ("created_at", "last_run_at", "batch")}
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value
        metas = [stable(json.loads((b / "batch.json").read_bytes())) for b in batches]
        assert metas[0] == metas[1] == metas[2]


def test_session_only_environment_does_not_enter_the_key():
    from haenv.world.generation import build_cache as C
    base = {"HAENV_DIFFICULTY_R2": "on", "PATH": "/bin"}
    noisy = {**base, "SSH_CONNECTION": "192.0.2.1 51234 192.0.2.2 22", "CODEX_TOOL_CALL_ID": "call_1",
             "PWD": "/elsewhere", "SHLVL": "3", "TMPDIR": "/scratch/tmp", "CLAUDECODE": "1"}
    assert C.keyed_env(noisy) == C.keyed_env(base)
    assert C.keyed_env({**base, "HAENV_NEW_BUILD_SWITCH": "x"}) != C.keyed_env(base)
    assert C.keyed_env({**base, "PATH": "/usr/bin"}) != C.keyed_env(base)
    assert C.keyed_env({**base, "HAENV_OUTPUT_ROOT": "/a", "HAENV_GEN_WORKERS": "8"}) == C.keyed_env(base)
