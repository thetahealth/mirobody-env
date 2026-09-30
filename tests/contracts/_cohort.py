"""Synthetic cohorts for the world-layer contracts.

Cases are built in-process with the deterministic generator (`build.build_case`), from the
shipped job files with one field changed at a time. Nothing calls a model. Job files written
here go to `.tmpwork/contracts/` (git-ignored), never to the system temp directory.

`build(spec, capture=True)` also records the case at the exit of each pipeline layer, so a
contract can assert that a property which holds after one layer still holds after every later
one. The capture wraps production functions and returns their results unchanged.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass, field

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import build as B                                  # noqa: E402
from haenv import job as J                                    # noqa: E402

WORK = ROOT / ".tmpwork" / "contracts"
T2D_JOB = ROOT / "inputs" / "early_warning-t2d-glp1.job.yaml"
EW20_JOB = ROOT / "inputs" / "early_warning-20.job.yaml"
EXAMPLE_JOB = ROOT / "inputs" / "example-ew.job.yaml"

#: Pipeline layers in the order `build._build_case_inner` runs them after the course is
#: rendered: kernel noise that changes the patient or drops readings, day-by-day events and
#: physiology, reading artifacts on the observed series, post-injection.
LAYERS = ("noise", "events", "artifacts", "post_inject", "final")


def spec_of(job: pathlib.Path, case_id: str) -> dict:
    data = yaml.safe_load(job.read_text(encoding="utf-8"))
    return copy.deepcopy(next(c for c in data["cases"] if c["case_id"] == case_id))


def variant(base: dict, case_id: str | None = None, **latent) -> dict:
    """A copy of `base` with `latent` fields replaced (None deletes the field)."""
    c = copy.deepcopy(base)
    if case_id:
        c["case_id"] = case_id
    lat = dict(c.get("latent") or {})
    for k, v in latent.items():
        if v is None:
            lat.pop(k, None)
        else:
            lat[k] = v
    c["latent"] = lat
    return c


def _job_file(case: dict, template: pathlib.Path) -> pathlib.Path:
    base = yaml.safe_load(template.read_text(encoding="utf-8"))
    job = {k: v for k, v in base.items() if k != "cases"}
    job["cases"] = [case]
    # the name covers the template header as well as the case, so an edited header never
    # reuses a stale file
    key = hashlib.sha256(json.dumps(job, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
    job["job_id"] = f"contract-{key}"
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"{key}.job.yaml"
    text = "# haenv-canary-v1\n" + yaml.safe_dump(job, allow_unicode=True, sort_keys=False)
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        p.write_text(text, encoding="utf-8")
    return p


@dataclass
class Built:
    case_id: str
    raw: object | None                      # RawCase, or None when not emitted
    audit: dict
    layers: dict = field(default_factory=dict)   # layer -> deep copy of longitudinal_data etc.

    @property
    def emitted(self) -> bool:
        return self.raw is not None


def _snapshot(raw) -> dict:
    return {"longitudinal_data": copy.deepcopy(getattr(raw, "longitudinal_data", {}) or {}),
            "reversal_points": copy.deepcopy(getattr(raw, "reversal_points", []) or []),
            "adjudication": copy.deepcopy(getattr(raw, "adjudication", {}) or {}),
            "evidence_ledger": copy.deepcopy(getattr(raw, "evidence_ledger", []) or []),
            "outcome_label": copy.deepcopy(getattr(raw, "outcome_label", None)),
            "gold_drivers": copy.deepcopy(getattr(raw, "gold_drivers", None))}


def build(case: dict, template: pathlib.Path = T2D_JOB, capture: bool = False) -> Built:
    return _build_cached(json.dumps(case, sort_keys=True, ensure_ascii=False), str(template), capture)


@functools.lru_cache(maxsize=None)
def _build_cached(case_json: str, template: str, capture: bool) -> Built:
    case = json.loads(case_json)
    job = J.load_job(_job_file(case, pathlib.Path(template)))
    cs = job.cases[0]
    layers: dict = {}
    if not capture:
        raw, audit = B.build_case(cs, T=cs.index_time_T)
        return Built(cs.case_id, raw, audit, layers)

    import pytest                                           # noqa: PLC0415
    mp = pytest.MonkeyPatch()
    try:
        orig_events = B._inject_events_verified
        orig_artifacts = B._inject_observation_artifacts
        orig_post = B._post_inject.apply_post_injection

        def spy_events(raw, *a, **k):
            layers["noise"] = _snapshot(raw)                # the pre-physiology noise ran just before
            out = orig_events(raw, *a, **k)
            layers["events"] = _snapshot(out[0])
            # the physiology layer's ground-truth series (`vrep["physio_clean_ld"]`), which the
            # slope and anchor gates judge and the post-injection backfill samples from
            layers["truth"] = copy.deepcopy((out[1] or {}).get("physio_clean_ld") or {})
            return out

        def spy_artifacts(raw, *a, **k):
            out = orig_artifacts(raw, *a, **k)
            layers["artifacts"] = _snapshot(out[0])
            return out

        def spy_post(raw, **k):
            res = orig_post(raw, **k)
            layers["post_inject"] = _snapshot(raw)
            return res

        mp.setattr(B, "_inject_events_verified", spy_events)
        mp.setattr(B, "_inject_observation_artifacts", spy_artifacts)
        mp.setattr(B._post_inject, "apply_post_injection", spy_post)
        raw, audit = B.build_case(cs, T=cs.index_time_T)
    finally:
        mp.undo()
    if raw is not None:
        layers["final"] = _snapshot(raw)
    return Built(cs.case_id, raw, audit, layers)


def series(ld: dict, name: str) -> dict[int, float]:
    return {int(p["ts"]): float(p["value"]) for p in (ld.get(name) or [])}


def build_uncached(case: dict, template: pathlib.Path = T2D_JOB) -> Built:
    """Same as `build(..., capture=False)` without the cache: for negative controls that
    monkeypatch a mechanism, whose result must not be served from, or leak into, the cache."""
    job = J.load_job(_job_file(case, template))
    cs = job.cases[0]
    raw, audit = B.build_case(cs, T=cs.index_time_T)
    return Built(cs.case_id, raw, audit, {})
