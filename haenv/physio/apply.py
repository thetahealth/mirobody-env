"""apply.py -- the physiology layer's assembly point: noise, bounds and audit
for one case.

    apply_physio(case_id, longitudinal_data, registry)
        -> (new longitudinal_data, AuditSummary)

Called by `events.inject` when the `physio` switch is on. Streams in `exclude`
(treatment, adherence, low-frequency scales) pass through unchanged; only
`value` changes, never `ts`; noise is deterministic given `case_id` (no RNG
state). The caller decides whether a nonzero clip rate fails the run
(`audit.assert_clean`).

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from ..yamlcache import load_yaml as _cached_yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import audit, bounds, kernel, noise, superpose


class StreamRegistryError(ValueError):
    """A stream's spec is missing or invalid. Deliberately has no default value."""


@dataclass(frozen=True)
class StreamSpec:
    """A stream's noise and bounds, registered together."""

    indicator: str
    noise: noise.NoiseSpec
    bound: bounds.BoundSpec


@dataclass(frozen=True)
class StreamRegistry:
    specs: dict[str, StreamSpec]
    excluded: frozenset[str]
    #: Observation-noise AR(1) coefficient per indicator; absent = none.
    ar_phi: Mapping[str, float] = field(default_factory=dict)
    #: `weight` noise marginal sd as a fraction of body mass; `None` = absolute kg.
    weight_sd_frac: float | None = None
    #: Distribution-audit thresholds, loaded from the same table as `specs`.
    audit_policy: "audit.AuditPolicy | None" = None

    def pending(self) -> list[str]:
        """Entries under review (all of them: loadings are engineering estimates)."""
        return sorted(self.specs)


_REQUIRED = ("group", "loading", "idio_sd", "lo", "hi", "max_step", "transform",
             "source", "review")


def load_stream_registry(path: str | Path) -> StreamRegistry:
    """Read the stream-spec registry; any invalid entry raises."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise StreamRegistryError(f"stream registry not found: {p}")
    from ..regpath import overlaid_yaml as _overlaid_yaml
    doc = _overlaid_yaml(p)          # the in-repo table carries the world-plugin overlay
    if not isinstance(doc, Mapping) or "streams" not in doc:
        raise StreamRegistryError(f"{p}: top level must be a mapping with `streams`")

    from ..regpath import registry_path as _registry_path
    if "exclude" in doc or p.resolve() != _registry_path(p.name).resolve():
        excluded = frozenset(str(x) for x in (doc.get("exclude") or []))
    else:
        # For the repository table, exclusions come from the stream manifest
        # (`physio: exclude`), and every `physio: render` stream needs a block here.
        from .. import streams as _streams
        excluded = _streams.physio_excluded()
        unblocked = sorted(_streams.physio_rendered() - {str(n) for n in (doc["streams"] or {})})
        if unblocked:
            raise StreamRegistryError(
                f"{unblocked}: registry/streams.yaml says physio: render, but {p.name} "
                "has no parameter block for them")
    specs: dict[str, StreamSpec] = {}
    for name, raw in (doc["streams"] or {}).items():
        name = str(name)
        if name in excluded:
            raise StreamRegistryError(
                f"{name}: listed in both `exclude` and `streams`; "
                "keep it in one of them"
            )
        if not isinstance(raw, Mapping):
            raise StreamRegistryError(f"{name}: entry must be a mapping")
        missing = [k for k in _REQUIRED if k not in raw]
        if missing:
            raise StreamRegistryError(f"{name}: missing fields {missing}; no defaults are provided")
        if str(raw["review"]).strip() not in ("pending", "done"):
            raise StreamRegistryError(f"{name}: review must be 'pending' or 'done'")
        _tf = str(raw["transform"])
        _at = raw.get("at")
        _bd = bounds.BoundSpec(
            indicator=name, lo=float(raw["lo"]), hi=float(raw["hi"]),
            max_step=float(raw["max_step"]), transform=_tf,
            source=str(raw["source"]),
            at=None if _at is None else float(_at),
        )
        # `loading`/`idio_sd` are registered in raw units and converted to the
        # transform domain here (a non-identity transform requires `at`).
        def _sc(v: float, _b: bounds.BoundSpec = _bd) -> float:
            if _b.transform == "identity":
                return v
            # A declared zero (calibrated stream, noise supplied externally) stays zero.
            if v == 0.0:
                return 0.0
            s = bounds.raw_sd_to_transform_sd(abs(v), float(_b.at), _b)
            return s if v >= 0 else -s
        specs[name] = StreamSpec(
            indicator=name,
            noise=noise.NoiseSpec(
                indicator=name,
                loadings={str(raw["group"]): _sc(float(raw["loading"]))},
                idio_sd=_sc(float(raw["idio_sd"])),
                external=str(raw.get("noise_external") or ""),
            ),
            bound=_bd,
        )
    if not specs:
        raise StreamRegistryError(f"{p}: `streams` is empty")

    # Invariant: `max_step` (raw units) must cover both tails of the bounded
    # noise, folded back to raw units via `bounds.induced_step`.
    tight = [(n, sp.bound.max_step, 2 * bounds.induced_step(sp.noise.bound, sp.bound))
             for n, sp in specs.items()
             if sp.bound.max_step < 2 * bounds.induced_step(sp.noise.bound, sp.bound)]
    if tight:
        raise StreamRegistryError(
            "max_step of these streams cannot absorb both noise tails (needs >= 2x the noise budget in raw units): "
            + "; ".join(f"{n}: max_step={m} < {need:.2f}" for n, m, need in tight)
            + ". The hard clamp is a safety net; hitting it means the soft-mechanism parameters are wrong. "
            "For log/logit streams, `idio_sd`/`loading` are in transform-domain units; "
            "convert with `bounds.raw_sd_to_transform_sd`"
        )
    # ---- observation noise's time structure and relative magnitude (optional) ----
    _ts = doc.get("noise_time_structure") or {}
    _phi: dict[str, float] = {}
    if _ts:
        _p = _ts.get("ar_phi")
        if _p is None:
            raise StreamRegistryError("noise_time_structure is present "
                                      "but has no ar_phi")
        for _n in (_ts.get("applies_to") or []):
            _phi[str(_n)] = float(_p)
    _wn = doc.get("weight_noise_relative") or {}
    _wfrac = _wn.get("marginal_sd_frac_of_body_mass")
    return StreamRegistry(specs=specs, excluded=excluded, ar_phi=_phi,
                          audit_policy=audit.AuditPolicy.from_registry(doc, required=False),
                          weight_sd_frac=(float(_wfrac) if _wfrac is not None else None))


def _points_to_days(points: Sequence[Mapping[str, Any]]) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for p in points:
        ts, v = p.get("ts"), p.get("value")
        if not isinstance(ts, int) or isinstance(ts, bool):
            raise StreamRegistryError(f"point ts must be an integer day index, got {ts!r}")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise StreamRegistryError(f"point value must be numeric, got {v!r}")
        out.append((int(ts), float(v)))
    return out


def event_offsets(
    events: Sequence[Mapping[str, Any]],
    kreg: "kernel.PhysioRegistry",
    indicator: str,
    day: int,
) -> float:
    """Offset for one indicator on one day from every active event, through
    `superpose.saturate`. Events attach to kernels by `topic`; an event without
    a registered `(topic, indicator)` kernel contributes nothing."""
    contribs: list[float] = []
    for ev in events:
        topic = ev.get("topic")
        if not topic:
            continue
        p = kreg.kernels.get((str(topic), indicator))
        if p is None:
            continue
        d = float(day) - float(ev.get("day", 0))
        if d < 0:                       # the event has not happened yet -- no contribution before its onset
            continue
        contribs.append(kernel.contribution(p, d))
    if not contribs:
        return 0.0
    return superpose.saturate(contribs, kreg.m_for(indicator))


def apply_physio(
    case_id: str,
    longitudinal_data: Mapping[str, Any],
    reg: StreamRegistry,
    *,
    strict_unknown: bool = True,
    events: Sequence[Mapping[str, Any]] | None = None,
    kernels: "kernel.PhysioRegistry | None" = None,
) -> tuple[dict[str, Any], audit.AuditSummary]:
    """Add correlated noise and event footprints to one case's streams and
    project them into bounds; returns `(new longitudinal_data, AuditSummary)`.

    `strict_unknown` (default True) raises on a stream in neither `streams` nor
    `exclude`. `events` and `kernels` must be given together.
    """
    if (events is None) != (kernels is None):
        raise StreamRegistryError(
            "`events` and `kernels` must be given together or not at all; one alone would ignore the events.")

    report = bounds.ProjectionReport()
    out: dict[str, Any] = {}
    unknown: list[str] = []

    # Expand each stream by day so every indicator on a day shares the same factors.
    prepared: dict[str, list[tuple[int, float]]] = {}
    for name, points in longitudinal_data.items():
        if not isinstance(points, list) or not points or not isinstance(points[0], Mapping):
            out[name] = points          # a field not shaped like [{ts,value}] passes through unchanged
            continue
        if name in reg.excluded:
            out[name] = points
            continue
        if name not in reg.specs:
            unknown.append(name)
            out[name] = points
            continue
        prepared[name] = _points_to_days(points)

    if unknown and strict_unknown:
        raise StreamRegistryError(
            f"streams with no spec and not in exclude: {sorted(unknown)}. "
            "Add a spec or list them in exclude."
        )

    all_days = sorted({d for pts in prepared.values() for d, _ in pts})
    prev: dict[str, float] = {}
    new_vals: dict[str, dict[int, float]] = {n: {} for n in prepared}

    # Noise per day sequence via an AR(1) recursion (`correlated_noise_series`).
    # `weight` noise is scaled to a fraction of the case's first weight reading.
    _all_names = sorted(prepared)
    _scale: dict[str, float] = {}
    if reg.weight_sd_frac is not None and "weight" in prepared and prepared["weight"]:
        _ref_mass = float(prepared["weight"][0][1])
        _cur_sd = reg.specs["weight"].noise.marginal_sd
        if _cur_sd > 0:
            _scale["weight"] = (reg.weight_sd_frac * _ref_mass) / _cur_sd
    _eps_all = noise.correlated_noise_series(
        case_id, all_days, [reg.specs[n].noise for n in _all_names],
        phi=reg.ar_phi, scale=_scale)
    if "weight" in _eps_all and reg.weight_sd_frac:
        _w_mass = float(prepared["weight"][0][1]) if prepared.get("weight") else 0.0
        _eps_all["weight"] = {
            d: noise.weight_noise_shaped(case_id, d, e, _w_mass, reg.weight_sd_frac)
            for d, e in _eps_all["weight"].items()}

    for day in all_days:
        active = [n for n in prepared if any(d == day for d, _ in prepared[n])]
        if not active:
            continue
        eps = {n: _eps_all[n][day] for n in active}
        for n in active:
            base = next(v for d, v in prepared[n] if d == day)
            spec = reg.specs[n].bound
            delta = 0.0
            if events is not None and kernels is not None:
                delta = event_offsets(events, kernels, n, day)
            # The event footprint changes the true value (raw units); noise is
            # added in the transform domain, which keeps log/logit outputs in range.
            proposal = bounds.from_transform(
                bounds.to_transform(base + delta, spec) + eps[n], spec)
            y = bounds.project(proposal, prev.get(n), spec, report)
            new_vals[n][day] = y
            prev[n] = y

    for name, pts in prepared.items():
        out[name] = [{"ts": d, "value": round(new_vals[name][d], 4)} for d, _ in pts]

    dists: list[audit.DistributionReport] = []
    for name in sorted(prepared):
        days = sorted(new_vals[name])
        if not days:
            continue
        lo, hi = days[0], days[-1]
        series: list[float | None] = [None] * (hi - lo + 1)
        for d in days:
            series[d - lo] = new_vals[name][d]
        try:
            dists.append(audit.describe(name, series))
        except audit.AuditError:
            # Not computable: left out rather than recorded as 0.0.
            continue

    # `severity: warn` only reports violations; `gate` is enforced by the caller.
    # Missing thresholds raise here, at use time.
    if reg.audit_policy is None:
        raise audit.AuditError(
            "stream registry has no `distribution_audit:` section; distribution audit thresholds are undefined.")
    viol: list[str] = []
    for d in dists:
        viol += d.check_policy(reg.audit_policy, float(reg.ar_phi.get(d.indicator, 0.0)))
    return out, audit.AuditSummary(projection=report, distributions=dists, violations=viol)
