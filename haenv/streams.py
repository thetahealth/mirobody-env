"""The stream manifest (`registry/streams.yaml`) and the per-stream tables derived from it
(daily metrics, auxiliary signals, physio exclusions, the GEN7 device map, derived-stream
bindings, gated pricing). Adding a stream means editing only the manifest.

Gated pricing, the one table scoring reads, is written to
`registry/gated_pricing_streams.yaml` by `tools/gen_stream_tables.py`, so the judging
fingerprint moves only when a price does.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

from typing import Any

FILENAME = "streams.yaml"
PRICING_FILENAME = "gated_pricing_streams.yaml"

_FIELDS = {"aux", "devices", "proves_device", "pricing", "render", "physio",
           "derived_from", "grid_parent"}
_REQUIRED = ("aux", "pricing", "physio")
_RENDER_FIELDS = {"base", "amp", "period", "hard_range", "tol", "vital_key", "tags"}
_RENDER_REQUIRED = ("base", "amp", "period", "hard_range", "tol")
#: Profile fields a derived stream may be a function of, besides other streams.
PROFILE_FIELDS = ("bmi", "age", "sex")
TIERS = ("vitals", "basic_lab", "advanced_lab", "imaging")
#: The kernel's device vocabulary (`latent.KNOWN_DEVICES`), repeated here so the
#: manifest can be validated without mounting the kernel.
DEVICES = ("smart_scale", "clinic_scale", "wearable", "cgm", "bp_cuff", "lab_panel")


class ManifestError(ValueError):
    """The stream manifest is malformed. Raised at load, never skipped."""


def manifest() -> dict[str, dict[str, Any]]:
    """All entries, in file order, validated."""
    from .regpath import load_registry
    doc = load_registry(FILENAME) or {}
    ent = doc.get("streams")
    if not isinstance(ent, dict) or not ent:
        raise ManifestError(f"{FILENAME}: needs a non-empty `streams` mapping")
    for name, e in ent.items():
        _validate(str(name), e, ent)
    return ent


def _validate(name: str, e: Any, ent: dict) -> None:
    if not isinstance(e, dict):
        raise ManifestError(f"{name}: entry must be a mapping")
    unknown = set(e) - _FIELDS
    if unknown:
        raise ManifestError(f"{name}: unknown fields {sorted(unknown)}")
    missing = [k for k in _REQUIRED if k not in e]
    if missing:
        raise ManifestError(f"{name}: missing {missing}; there are no defaults")
    if not isinstance(e["aux"], bool):
        raise ManifestError(f"{name}: aux must be true or false")
    if e["pricing"] not in TIERS:
        raise ManifestError(f"{name}: pricing {e['pricing']!r} is not one of {TIERS}")
    if e["physio"] not in ("render", "exclude"):
        raise ManifestError(f"{name}: physio must be render or exclude")
    for k in ("devices", "proves_device"):
        bad = [d for d in e.get(k) or [] if d not in DEVICES]
        if bad:
            raise ManifestError(f"{name}: {k} names unknown devices {bad}")
    r = e.get("render")
    if r is not None:
        if not isinstance(r, dict) or set(r) - _RENDER_FIELDS:
            raise ManifestError(f"{name}: render takes only {sorted(_RENDER_FIELDS)}")
        miss = [k for k in _RENDER_REQUIRED if k not in r]
        if miss:
            raise ManifestError(f"{name}: render is missing {miss}")
        if "devices" not in e:
            raise ManifestError(f"{name}: a rendered metric must list its devices "
                                "(an empty list means self-report)")
    if ("derived_from" in e) != ("grid_parent" in e):
        raise ManifestError(f"{name}: derived_from and grid_parent come together")
    if "derived_from" in e:
        bad = [x for x in e["derived_from"] if x not in ent and x not in PROFILE_FIELDS]
        if bad:
            raise ManifestError(f"{name}: derived_from names neither a stream nor a "
                                f"profile field: {bad}")
        gp = e["grid_parent"]
        if gp not in ent or "render" not in ent[gp]:
            raise ManifestError(f"{name}: grid_parent {gp!r} must be a rendered stream")


def metric_fields() -> list[dict[str, Any]]:
    """Arguments for `events.MetricSpec`, one per rendered stream, in file order.

    Unit and precision come from the stream's dossier: they are properties of the
    indicator, and the dossier is where every prompt reads them.
    """
    from .indicators import of
    out = []
    for name, e in manifest().items():
        r = e.get("render")
        if r is None:
            continue
        d = of(name)
        out.append({
            "name": name, "devices": tuple(e["devices"]), "unit": d["unit"],
            "base": r["base"], "amp": r["amp"], "period": r["period"],
            "ndigits": d["ndigits"], "hard_range": tuple(r["hard_range"]), "tol": r["tol"],
            "vital_key": r.get("vital_key"), "tags": tuple(r.get("tags") or ()),
        })
    return out


def aux_signals() -> frozenset[str]:
    """Streams that are not predicted clinical signals (kernel `AUX_SIGNALS`)."""
    return frozenset(n for n, e in manifest().items() if e["aux"])


def aux_metrics() -> frozenset[str]:
    """Rendered daily metrics that are auxiliary (`events.AUX_WHITELIST`)."""
    return frozenset(n for n, e in manifest().items() if e["aux"] and "render" in e)


def physio_excluded() -> frozenset[str]:
    return frozenset(n for n, e in manifest().items() if e["physio"] == "exclude")


def physio_rendered() -> frozenset[str]:
    return frozenset(n for n, e in manifest().items() if e["physio"] == "render")


def device_signals() -> dict[str, set[str]]:
    """Launch gate GEN7: device -> the streams that show it is in use."""
    out: dict[str, set[str]] = {}
    for name, e in manifest().items():
        for dev in e.get("proves_device") or []:
            out.setdefault(dev, set()).add(name)
    return out


def derived_bindings() -> dict[str, tuple[str, ...]]:
    return {n: tuple(e["derived_from"]) for n, e in manifest().items() if "derived_from" in e}


def grid_parents() -> dict[str, str]:
    return {n: e["grid_parent"] for n, e in manifest().items() if "grid_parent" in e}


def pricing() -> dict[str, str]:
    """Stream -> gated price tier. Scoring reads the generated file, not this."""
    return {n: e["pricing"] for n, e in manifest().items()}


def register_with_kernel() -> None:
    """Hand the manifest's auxiliary signals to the kernel.

    The kernel keeps only the auxiliary signals its own generators emit; the rest
    come from here, so adding a stream does not mean editing the kernel.
    """
    try:
        import synth                                          # kernel
    except ImportError:
        return
    register = getattr(synth, "register_aux_signals", None)
    if register is not None:
        register(aux_signals())
    elif isinstance(getattr(synth, "AUX_SIGNALS", None), set):
        synth.AUX_SIGNALS.update(aux_signals())        # a kernel checkout without the hook
