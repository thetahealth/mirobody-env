"""The world-course prompt sees premise `meta` through an allow-list.

`premise_spec` puts generation intent and delivery flags side by side in the
premise's `meta`. Only the former may shape the patient's course; a delivery flag
in the world prompt would change the same patient's weight, blood pressure and lab
history. The allow-list makes a new key invisible to the world until someone
classifies it.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import haenv                                                        # noqa: E402

sys.path.insert(0, str(haenv.kernel_path()))

from haenv_kernel.latent import LatentPremise                                    # noqa: E402

import haenv.app.job as J                                          # noqa: E402
from haenv.world.generation.build import OBSERVATION_ONLY_META
from haenv.world.generation.build import WORLD_PROMPT_META
from haenv.world.generation.build import _world_prompt_premise
from haenv.world.generation.build import premise_spec

JOBS = ("ddx-timeline.job.yaml", "ddx-workup.job.yaml", "example-ew.job.yaml")


def _premise(cs) -> LatentPremise:
    # Built without `make_premise`: its validation needs the comorbidity domain
    # that `build_case` scopes in, and this test is about serialization only.
    spec = premise_spec(cs)
    return LatentPremise(**{k: spec[k] for k in ("patient_basics", "event_density",
                                                  "device_signals", "adherence")},
                         source="human", meta=spec.get("meta", {}))


def _cases():
    for name in JOBS:
        for cs in J.load_job(ROOT / 'benchmarks' / 'inputs' / name).cases:
            yield name, cs


def test_every_meta_key_is_classified_once():
    assert not set(WORLD_PROMPT_META) & set(OBSERVATION_ONLY_META)
    known = set(WORLD_PROMPT_META) | set(OBSERVATION_ONLY_META)
    n = 0
    for name, cs in _cases():
        unclassified = set(premise_spec(cs)["meta"]) - known
        assert not unclassified, f"{name}:{cs.case_id} emits unclassified meta keys {unclassified}"
        n += 1
    assert n >= 150, f"scan surface looks wrong: {n} cases"


def test_prompt_preserves_world_intent_without_delivery_bookkeeping(monkeypatch):
    """Preserve patient-course inputs; omit delivery IDs and an empty plugin slot."""
    import haenv.contracts.external_gold as EG
    # A registered plugin may cause premise_spec to emit an empty external slot even
    # for these legacy cases. Exercise that state explicitly, regardless of test order.
    monkeypatch.setattr(EG, "passthrough_keys", lambda: ("fixture_absent_latent",))
    for name, cs in _cases():
        p = _premise(cs)
        old = p.dumps()
        old["meta"] = {k: v for k, v in old["meta"].items()
                       if k not in (*OBSERVATION_ONLY_META, "solver_case_id")
                       and not (k == EG.SLOT and not v)}
        assert (json.dumps(_world_prompt_premise(p), ensure_ascii=False)
                == json.dumps(old, ensure_ascii=False)), f"{name}:{cs.case_id}"


def test_opaque_solver_id_is_private_and_does_not_change_the_world_prompt():
    _, cs = next(_cases())
    p = _premise(cs)
    before = _world_prompt_premise(p)
    opaque = "C0123456789"
    p.meta["solver_case_id"] = opaque
    prompt = _world_prompt_premise(p)
    assert prompt == before
    assert "solver_case_id" not in prompt["meta"]
    assert opaque not in json.dumps(prompt)
    assert prompt["meta"]["case_id"] == cs.case_id  # world RNG identity is retained
    assert p.meta["solver_case_id"] == opaque       # verifier bookkeeping is intact
    p.meta["solver_case_id"] = "C9876543210"
    assert _world_prompt_premise(p) == before


def test_unlisted_key_is_withheld():
    """Negative control: a key nobody classified does not reach the prompt."""
    _, cs = next(_cases())
    p = _premise(cs)
    p.meta["probe_unlisted_key"] = "leak"
    p.meta["rhythm_gap"] = True
    meta = _world_prompt_premise(p)["meta"]
    assert "probe_unlisted_key" not in meta and "rhythm_gap" not in meta
    assert "case_id" in meta


def test_external_gold_and_knob_keys_never_reach_the_prompt(monkeypatch):
    """The external slot carries a plugin's latent keys into the premise; the world prompt shows
    only those classed `patient_fact`. An empty slot is dropped, so a case with no plugin latent keeps
    its prompt (and its cache key) byte for byte."""
    import haenv.contracts.external_gold as EG
    _, cs = next(_cases())
    p = _premise(cs)
    monkeypatch.setitem(EG.LATENT_CLASSES, "x_gold", "gold")
    monkeypatch.setitem(EG.LATENT_CLASSES, "x_knob", "knob")
    monkeypatch.setitem(EG.LATENT_CLASSES, "x_fact", "patient_fact")
    p.meta[EG.SLOT] = {"x_gold": {"dx": "SECRET"}, "x_knob": {"tier": "SECRET"}, "x_fact": {"age": 61}}
    meta = _world_prompt_premise(p)["meta"]
    assert meta[EG.SLOT] == {"x_fact": {"age": 61}} and "SECRET" not in json.dumps(meta)
    p.meta[EG.SLOT] = {"x_gold": {"dx": "SECRET"}}
    assert EG.SLOT not in _world_prompt_premise(p)["meta"]
    p.meta[EG.SLOT] = {}
    assert EG.SLOT not in _world_prompt_premise(p)["meta"]
