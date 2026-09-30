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

from latent import LatentPremise                                    # noqa: E402

from haenv import job as J                                          # noqa: E402
from haenv.build import (OBSERVATION_ONLY_META, WORLD_PROMPT_META,  # noqa: E402
                         _world_prompt_premise, premise_spec)

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
        for cs in J.load_job(ROOT / "inputs" / name).cases:
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


def test_prompt_is_the_premise_minus_observation_only_keys():
    """Positive control: on the frozen packs the allow-list changes no prompt byte
    relative to dropping `OBSERVATION_ONLY_META` alone."""
    for name, cs in _cases():
        p = _premise(cs)
        old = p.dumps()
        old["meta"] = {k: v for k, v in old["meta"].items() if k not in OBSERVATION_ONLY_META}
        assert (json.dumps(_world_prompt_premise(p), ensure_ascii=False)
                == json.dumps(old, ensure_ascii=False)), f"{name}:{cs.case_id}"


def test_unlisted_key_is_withheld():
    """Negative control: a key nobody classified does not reach the prompt."""
    _, cs = next(_cases())
    p = _premise(cs)
    p.meta["probe_unlisted_key"] = "leak"
    p.meta["rhythm_gap"] = True
    meta = _world_prompt_premise(p)["meta"]
    assert "probe_unlisted_key" not in meta and "rhythm_gap" not in meta
    assert "case_id" in meta
