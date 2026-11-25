"""The value model of a printed reading (world 1.5, B3: fidelity F4 and F6).

    ln observed = ln person_baseline + effects(t) + CVI z_draw + CVA z_draw'

`registry/biological_variation.yaml` holds every number: EFLM within- and between-subject variation,
EuBIVAS analytical CVs, the IDHOCO home-BP spread and the NHANES potassium/creatinine centres. Read only
when `synth.course_coupling` is on; callers check `coupling_enabled()` before they call in here, so the
uncoupled world never reads this file.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from haenv.resources.regpath import load_registry, registry_cached

_KEYS = {"version", "sources", "analytes", "home_bp", "person_baseline", "checks"}
_ANALYTE_KEYS = {"cvi", "cvg", "cva", "cva_method", "source"}


def _sources_of(row) -> list[str]:
    s = row.get("source")
    return [s] if isinstance(s, str) else list(s or ())


def validate(doc: dict) -> dict:
    """Refuse an unknown key, an unsourced or unverified number, or a CV outside (0, 1)."""
    if set(doc) - _KEYS:
        raise ValueError(f"unknown biological_variation keys: {sorted(set(doc) - _KEYS)}")
    sources = doc.get("sources") or {}

    def sourced(row, path):
        names = _sources_of(row)
        if not names:
            raise ValueError(f"number without source at {path}")
        for name in names:
            if name not in sources:
                raise ValueError(f"unresolved source at {path}: {name}")
            if sources[name].get("status") != "verified":
                raise ValueError(f"source not verified at {path}: {name}")

    for name, row in (doc.get("analytes") or {}).items():
        if set(row) - _ANALYTE_KEYS:
            raise ValueError(f"unknown analyte fields at {name}: {sorted(set(row) - _ANALYTE_KEYS)}")
        sourced(row, f"analytes.{name}")
        for key in ("cvi", "cvg"):
            v = row.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < float(v) < 1.0:
                raise ValueError(f"analytes.{name}.{key} must be a fraction in (0, 1)")
        cva = row.get("cva")
        if cva is not None and (isinstance(cva, bool) or not 0.0 < float(cva) < 1.0):
            raise ValueError(f"analytes.{name}.cva must be null or a fraction in (0, 1)")
    for section in ("home_bp", "person_baseline", "checks"):
        for name, row in (doc.get(section) or {}).items():
            if not isinstance(row, dict):
                raise ValueError(f"{section}.{name} must be a mapping")
            sourced(row, f"{section}.{name}")
    for name, row in (doc.get("person_baseline") or {}).items():
        if name not in (doc.get("analytes") or {}):
            raise ValueError(f"person_baseline.{name} has no analyte row (its spread is the analyte's CVG)")
        if set(row.get("median") or {}) != {"M", "F"} or any(float(v) <= 0 for v in row["median"].values()):
            raise ValueError(f"person_baseline.{name}.median needs positive M and F centres")
    for name in ("systolic", "diastolic"):
        row = (doc.get("home_bp") or {}).get(name)
        if row is not None and not (float(row["mean"]) > 0 and float(row["sd"]) > 0):
            raise ValueError(f"home_bp.{name}: mean and sd must be positive")
    return doc


@registry_cached("biological_variation.yaml")
def table() -> dict:
    return validate(load_registry("biological_variation.yaml"))


def analyte(name: str) -> dict:
    return table()["analytes"][name]


def draw_cv(name: str) -> float:
    """The per-draw total on the log scale, sqrt(CVI^2 + CVA^2) (CVA omitted when none is published)."""
    row = analyte(name)
    cva = float(row["cva"] or 0.0)
    return math.sqrt(float(row["cvi"]) ** 2 + cva * cva)


def _z(*path: str) -> float:
    """A deterministic standard normal clipped to +-3 (Box-Muller on the world's hash RNG)."""
    from haenv.world.generation.rng import unit
    u1 = max(1e-12, unit(*path, "bm0"))
    u2 = unit(*path, "bm1")
    return min(3.0, max(-3.0, math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)))


def person_z(case_id: str, name: str) -> float:
    """The person's standard score on `name`, drawn once per case (its own RNG key)."""
    return _z("b3-person", str(case_id), str(name))


def person_baseline(case_id: str, name: str, sex: str) -> float:
    """NHANES centre for the sex, spread by the analyte's EFLM CVG."""
    row = table()["person_baseline"][name]
    centre = float(row["median"]["M" if str(sex) == "M" else "F"])
    return centre * math.exp(float(analyte(name)["cvg"]) * person_z(case_id, name))


def home_bp_between_cv(kind: str) -> float:
    """Between-person spread of a home BP mean as a log-scale CV (IDHOCO: sd / mean)."""
    row = table()["home_bp"][kind]
    return float(row["sd"]) / float(row["mean"])


def home_mean_repeat_sd() -> float:
    """Within-person SD (mmHg) of one systolic home mean: Stergiou 2002 SD of differences / sqrt 2."""
    return float(table()["home_bp"]["repeat_sdd_systolic"]["value"]) / math.sqrt(2.0)
