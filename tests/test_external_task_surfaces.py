"""The three entry points an external task type needs besides its gold block:
emission gates, provenance classes for its latent keys, and a place in the world
stamp.

Each entry point is checked in both directions: with nothing registered the
repo's behaviour is unchanged, and once something is registered it takes
effect. Either check alone passes on a mechanism that has been removed.

Registration is process-global, so every test snapshots the registries, clears
them, and restores them in a `finally`.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import pathlib
import sys

import pytest

import haenv                                              # noqa: F401  mounts the kernel
from haenv import anchor, provenance
from haenv import external_gold as EG
from haenv import job as J
from haenv import world_plugins as WP

ROOT = pathlib.Path(haenv.__file__).resolve().parent.parent


@contextlib.contextmanager
def clean_registries():
    """Empty external registries for the duration of a test, restored afterwards."""
    saved = {name: dict(getattr(EG, name))
             for name in ("BLOCKS", "GATES", "FRAMINGS", "PROBES", "LATENT_CLASSES")}
    saved_reg = dict(J.LATENT_REGISTRY)
    try:
        for k in EG.passthrough_keys():
            J.LATENT_REGISTRY.pop(k, None)
        for name in saved:
            getattr(EG, name).clear()
        yield
    finally:
        for name, d in saved.items():
            getattr(EG, name).clear()
            getattr(EG, name).update(d)
        J.LATENT_REGISTRY.clear()
        J.LATENT_REGISTRY.update(saved_reg)


def _a_case():
    """First case of a job in the repo; a fabricated spec could not be built."""
    return J.load_job(ROOT / "inputs" / "example-ew.job.yaml").cases[0]


# ---------------------------------------------------------------- gates

def test_gates_for_is_empty_when_nothing_is_registered():
    with clean_registries():
        assert EG.gates_for(None, None, None, 0) == []


def test_a_registered_gate_blocks_emission():
    """Reverse control: a gate-severity hit must stop the case at the emission
    gate, and the gate must see the solver payload and the case's own T."""
    from haenv.build import build_case
    seen = {}

    def always(raw, cs, sp, T):
        seen.update(case_id=cs.case_id, T=T, has_ledger=hasattr(sp, "evidence_ledger"))
        return [{"kind": "probegate_always", "severity": "gate", "detail": "reverse control"}]

    with clean_registries():
        cs = _a_case()
        _, before = build_case(cs)
        EG.register_gate("probegate", always, source="tests", why="reverse control")
        raw, after = build_case(_a_case())
    assert "probegate_always" not in before.get("post_noise_conflicts", [])
    assert raw is None and "probegate_always" in after["post_noise_conflicts"]
    assert seen == {"case_id": cs.case_id, "T": int(cs.index_time_T), "has_ledger": True}


def test_a_warn_hit_is_logged_not_blocking():
    from haenv.build import build_case
    with clean_registries():
        _, before = build_case(_a_case())
        EG.register_gate("probewarn", lambda raw, cs, sp, T: [
            {"kind": "probewarn_note", "severity": "warn", "detail": "x"}],
            source="tests", why="warn path")
        _, after = build_case(_a_case())
    assert after["emitted"] == before["emitted"]
    assert "probewarn_note:x" in after["gate_warnings"]


@pytest.mark.parametrize("hit, err", [
    ({"kind": "other_prefix", "severity": "gate", "detail": "x"}, ValueError),
    ({"kind": "probe_x", "severity": "fatal", "detail": "x"}, ValueError),
    ({"kind": "probe_x", "severity": "gate"}, TypeError),
])
def test_malformed_hits_are_rejected(hit, err):
    with clean_registries():
        EG.register_gate("probe", lambda *a: [hit], source="tests", why="shape")
        with pytest.raises(err):
            EG.gates_for(None, None, None, 0)


def test_gate_registration_is_fail_closed():
    with clean_registries():
        EG.register_gate("dup", lambda *a: [], source="tests", why="x")
        with pytest.raises(ValueError):
            EG.register_gate("dup", lambda *a: [], source="tests", why="x")
        with pytest.raises(ValueError):
            EG.register_gate("nowhy", lambda *a: [], source="tests", why="")
        with pytest.raises(TypeError):
            EG.register_gate("notfn", 42, source="tests", why="x")


# ---------------------------------------------------------------- question side

def test_a_string_question_side_lands_at_the_top_of_prediction_context():
    """A block with no gold side and a string question side puts a solver-visible field at
    `prediction_context.<name>`; the gold side writes nothing, and a non-dict, non-string
    value is refused."""
    from haenv.build import build_case
    with clean_registries():
        EG.register_gold_block("probenote", lambda s: None, probe_fn=lambda s: "code every line",
                               latent_keys=(), latent_classes={}, source="tests", why="x")
        EG.register_gold_block("probeptr", lambda s: None, probe_fn=lambda s: {"path": "p"},
                               latent_keys=(), latent_classes={}, source="tests", why="x")
        raw, _ = build_case(_a_case())
        assert raw.prediction_context["probenote"] == "code every line"
        assert raw.prediction_context["probeptr"] == {"path": "p"}
        assert "probenote" not in raw.adjudication and "probeptr" not in raw.adjudication
        EG.register_gold_block("probebad", lambda s: None, probe_fn=lambda s: 3,
                               latent_keys=(), latent_classes={}, source="tests", why="x")
        with pytest.raises(TypeError):
            EG.probe_blocks_for({})


# ---------------------------------------------------------------- classes

def test_latent_keys_without_a_class_are_rejected_and_leave_nothing_behind():
    with clean_registries():
        before = dict(J.LATENT_REGISTRY)
        for classes in (None, {"probe_k1": "gold"}, {"probe_k1": "gold", "probe_k2": "nope"},
                        {"probe_k1": "gold", "probe_k2": "knob", "stray": "gold"}):
            with pytest.raises(ValueError):
                EG.register_gold_block("probe", lambda s: None,
                                       latent_keys=("probe_k1", "probe_k2"),
                                       latent_classes=classes, source="tests", why="x")
        assert J.LATENT_REGISTRY == before and "probe" not in EG.BLOCKS
        assert not EG.LATENT_CLASSES


def test_declared_classes_reach_provenance():
    with clean_registries():
        EG.register_gold_block("probe", lambda s: None, latent_keys=("probe_k1", "probe_k2"),
                               latent_classes={"probe_k1": "gold", "probe_k2": "knob"},
                               source="tests", why="x")
        assert provenance.class_of("probe_k1") == "gold"
        assert provenance.class_of("probe_k2") == "knob"
        assert provenance.check_covers_registry(J.LATENT_REGISTRY) == []
    assert provenance.class_of("probe_k1") is None


# ---------------------------------------------------------------- world stamp

def _old_formula() -> str:
    """The world stamp as computed when no external task type is registered."""
    base = anchor._fingerprint_of(anchor.generation_files())
    m = WP.manifest_sha()
    return base if not m else hashlib.sha256(f"{base}|{m}".encode()).hexdigest()[:16]


def _plugin(tmp_path: pathlib.Path, modname: str, body: str):
    """Write and import a one-file package `modname` whose `gold` returns `body`."""
    pkg = tmp_path / modname / modname
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    sys.path.insert(0, str(pkg.parent))
    try:
        return importlib.import_module(modname)
    finally:
        sys.path.remove(str(pkg.parent))


def test_world_stamp_unchanged_when_nothing_is_registered():
    with clean_registries():
        assert EG.manifest_sha() == ""
        assert anchor.world_fingerprint() == _old_formula()


def test_world_stamp_follows_plugin_code_not_its_comments(tmp_path):
    a = "def gold(spec):\n    return {'v': 1}\n"
    b = "def gold(spec):\n    return {'v': 2}\n"
    c = '"""Docstring added."""\n# a comment\ndef gold(spec):\n    return {\'v\': 1}\n'
    mods = {m: _plugin(tmp_path, m, src) for m, src in
            (("tmpplug_a", a), ("tmpplug_b", b), ("tmpplug_c", c))}
    try:
        stamps = {}
        for m, mod in mods.items():
            with clean_registries():
                EG.register_gold_block("probe", mod.gold, latent_keys=(), source="tests", why="x")
                stamps[m] = anchor.world_fingerprint()
        with clean_registries():
            empty = anchor.world_fingerprint()
    finally:
        for m in mods:
            sys.modules.pop(m, None)
    assert stamps["tmpplug_a"] != empty
    assert stamps["tmpplug_a"] != stamps["tmpplug_b"]      # code changed
    assert stamps["tmpplug_a"] == stamps["tmpplug_c"]      # comments only


def test_cached_world_stamp_sees_a_later_registration():
    """A second job in the same process registers a plugin after the cache was
    filled; the cached stamp must follow."""
    with clean_registries():
        first = anchor.world_fp_cached()
        EG.register_gate("probe", lambda *a: [], source="tests", why="x")
        second = anchor.world_fp_cached()
        EG.GATES.clear()
        third = anchor.world_fp_cached()
    assert first != second and third == first
