"""constraints.py -- the constraint registry, a pre-generation feasibility
check, and three tightness entry points.

A world is generated from a set of constraints, and tightness is a dial on the
same generator:

    tight   every observable surface of one case  -> one determined world
    mid     a population profile                  -> a cohort of similar patients
    loose   only a disease or condition           -> the possible worlds for it

`check_feasible` rejects constraint sets that cannot generate, or that would
generate without ever checking a constraint: inconsistent weight anchors,
anchors on unsampled days, unregistered drugs or conditions, devices that
produce no signal, event density above the event pool, symptoms after T.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------- 1. Constraint registry
#
# Default-deny: fields not listed here are rejected. `kind`: sampled (from a
# domain), derived (from another field; not independently constrainable),
# anchor (numeric, loosened to a range), listed (a list, fixed or loosened whole).
CONSTRAINABLE: dict[str, str] = {
    "age_range": "sampled",
    "sex": "sampled",
    "devices": "sampled",
    "drug": "sampled",
    "dose_steps": "derived",          # derived from drug
    "disease": "sampled",
    "start_weight": "anchor",
    "nadir_weight": "anchor",
    "symptoms": "listed",
    # The diagnosis condition (JD-PHEO / JD-PCOS / ...): the only degree of
    # freedom for gold, which is derived from it through the condition registry.
    "ddx_condition": "sampled",
}

DERIVED_FROM: dict[str, str] = {"dose_steps": "drug"}

# Once `ddx_condition` is fixed, these gold fields are derived from it and may
# not be independently constrained.
GOLD_FROM_CONDITION: tuple[str, ...] = (
    "diagnosis", "aliases", "tests", "specialty", "urgency", "red_flag_present",
    "threads", "join_gold", "outcome_label", "clinician_action_warranted")


@dataclass(frozen=True)
class Fix:
    """Fixed at this value (tightest)."""
    value: object


@dataclass(frozen=True)
class Range:
    """A numeric range (medium tightness)."""
    lo: float
    hi: float


@dataclass(frozen=True)
class OneOf:
    """An enumerated domain (medium tightness)."""
    options: tuple


@dataclass(frozen=True)
class Free:
    """Fully open (loosest) -- the sampler picks a value from the registered
    domain."""


@dataclass
class ConstraintSpec:
    """A set of constraints; `tightness` is a label only."""
    fields: dict = field(default_factory=dict)
    tightness: str = "mid"
    note: str = ""

    def of(self, name: str):
        return self.fields.get(name, Free())


class Infeasible(ValueError):
    """This constraint set cannot be solved into a world."""


# ------------------------------------------------------- 2. Feasibility check
def check_feasible(cs: ConstraintSpec, *, T: int = 84,
                   sampling_days: int = 1, event_density: dict | None = None,
                   facts=None, driver: str = "") -> list[str]:
    """Check, before generation, whether this constraint set is solvable.
    Returns a list of problems; empty means feasible.
    """
    bad: list[str] = []

    # default-deny: unregistered field
    unknown = sorted(set(cs.fields) - set(CONSTRAINABLE))
    if unknown:
        bad.append(f"unregistered constraint field(s) {unknown} — a misspelled field name fails "
                   f"silently, hence default-deny (registered: {sorted(CONSTRAINABLE)})")

    # a derived field may not be independently constrained
    for f_, src in DERIVED_FROM.items():
        if isinstance(cs.of(f_), (Fix, Range, OneOf)):
            bad.append(f"{f_} is derived from {src} and may not be independently constrained — "
                       f"giving it its own value would conflict with {src}")

    # 1. weight anchors are mutually consistent
    sw, nw = cs.of("start_weight"), cs.of("nadir_weight")
    if isinstance(sw, Fix) and isinstance(nw, Fix):
        if float(nw.value) > float(sw.value):
            bad.append(f"nadir_weight {nw.value} > start_weight {sw.value} —— "
                       f"the nadir cannot be higher than the starting point")
    if isinstance(sw, Range) and sw.lo > sw.hi:
        bad.append(f"start_weight range is inverted [{sw.lo}, {sw.hi}]")

    # 2. anchors must land on a sampled day, or they are never checked
    if sampling_days and int(sampling_days) > 1:
        bad.append(f"sampling interval {sampling_days} days > 1: an anchor may land on an "
                   f"unsampled day and never be validated; "
                   f"loosening the sampling rate requires also declaring anchors aligned to a sampled slot")

    # 2b. the condition must be registered (gold is derived from it)
    cond = cs.of("ddx_condition")
    if isinstance(cond, (Fix, OneOf)):
        from .overlay import condition_registry
        reg = condition_registry()
        want = [cond.value] if isinstance(cond, Fix) else list(cond.options)
        unknown = sorted({str(x) for x in want} - set(reg))
        if unknown:
            bad.append(f"ddx_condition={unknown} is not in the condition registry — "
                       f"gold is derived from it via the registry, so a miss means the entire "
                       f"gold set is missing ({len(reg)} entries total, e.g. {sorted(reg)[:3]}…)")

    # 3. drug and dosing frequency
    drug = cs.of("drug")
    if isinstance(drug, Fix):
        from .gate_tables import DRUG_DOSES_PER_WEEK
        if str(drug.value) not in DRUG_DOSES_PER_WEEK:
            bad.append(f"drug={drug.value!r} is not in the dosing-frequency registry — "
                       f"frequency is derived from the drug, and an unregistered drug can't "
                       f"derive one (available: {sorted(DRUG_DOSES_PER_WEEK)})")

    # 4. a device must actually be able to produce a signal
    dev = cs.of("devices")
    if isinstance(dev, (Fix, OneOf)):
        vals = dev.value if isinstance(dev, Fix) else [x for o in dev.options for x in o]
        from .events_streams import METRICS
        servable = {d for m in METRICS for d in (m.devices or ())}
        idle = sorted({str(d) for d in (vals or [])} - servable - {"smart_scale"})
        if idle:
            bad.append(f"device(s) {idle} cannot produce any signal in this environment — "
                       f"declaring a device that has never produced a reading is a factual "
                       f"error in the question (device_inventory_idle)")

    # 5. event density must not exceed event pool capacity. With `facts` and
    # `driver` the patient's effective pool is used; otherwise the nominal pool,
    # which is only a necessary condition.
    ed = event_density or {}
    if ed:
        from .events import effective_pool_size, event_pools
        from .events_pools import event_weeks, expected_event_counts
        BENIGN_EVENTS, LIFE_EVENTS = event_pools()
        weeks = event_weeks(T)
        # shares its default rates with the injector
        _want = expected_event_counts(ed, T)
        eff = effective_pool_size(facts, driver or "") if facts is not None else None
        for key, pool in (("symptom_rate", BENIGN_EVENTS), ("life_event_rate", LIFE_EVENTS)):
            want = _want[key]
            cap, basis = ((eff[key], "this patient's effective pool") if eff is not None
                          else (len(pool), "the pool's nominal size (the effective pool is "
                                "usually smaller; this check is only a necessary condition)"))
            if want > cap:
                bad.append(f"{key} × {weeks:.1f} weeks = {want}, but {basis} only has {cap} — "
                           f"'inject as many as declared' is guaranteed to fall short")

    # 6. symptom timestamps must fall inside the observation window
    sym = cs.of("symptoms")
    if isinstance(sym, Fix) and isinstance(sym.value, (list, tuple)):
        days = [int(s["day"]) if isinstance(s, dict) else int(s[0]) for s in sym.value]
        late = [d for d in days if d > T]
        if late:
            bad.append(f"symptom timestamp(s) {late} fall after T={T} — the solver can't see "
                       f"them, which means the question asks something the question text can't "
                       f"answer")
        if len(days) != len(set(days)):
            bad.append(f"symptom timestamps have duplicates {days} — two real symptoms on the "
                       f"same day would compete with benign events for that day")

    return bad


def require_feasible(cs: ConstraintSpec, **kw) -> None:
    """`check_feasible`, raising `Infeasible` on any problem."""
    bad = check_feasible(cs, **kw)
    if bad:
        raise Infeasible("; ".join(bad))


# ------------------------------------------------------- 3. Three tightness entry points
def tight_from_case(raw: dict) -> ConstraintSpec:
    """Tight: fix every observable surface of a specific case -> one determined
    world.
    """
    return ConstraintSpec(
        fields={k: Fix(v) for k, v in raw.items()
                if k in CONSTRAINABLE and k not in DERIVED_FROM},
        tightness="tight", note="从具体病例提取:每一维都固定")


def cohort(*, age_range=None, sex=None, drug=None, weight=None) -> ConstraintSpec:
    """Mid: a population profile -> a family of similar patients. Dimensions
    given are tightened; the rest are left open."""
    f: dict = {}
    if age_range is not None:
        f["age_range"] = OneOf(tuple(age_range)) if isinstance(age_range, (list, tuple)) \
            else Fix(age_range)
    if sex is not None:
        f["sex"] = Fix(sex)
    if drug is not None:
        f["drug"] = OneOf(tuple(drug)) if isinstance(drug, (list, tuple)) else Fix(drug)
    if weight is not None:
        f["start_weight"] = Range(float(weight[0]), float(weight[1]))
    return ConstraintSpec(fields=f, tightness="cohort",
                          note="队列:给定维收紧,其余按登记域采样")


def condition(disease: str) -> ConstraintSpec:
    """Loose: only a metabolic-baseline disease -> the possible worlds for that
    disease. All demographics are left open."""
    return ConstraintSpec(fields={"disease": Fix(disease)}, tightness="condition",
                          note="病种:除主线病外全部放开")


def ddx_condition(spec_id: str) -> ConstraintSpec:
    """Loose (for diagnostic questions): only a condition identity -> the
    possible worlds for that condition, with all of gold determined by it.

    Gold fields are derived from the condition through the registry (see
    `GOLD_FROM_CONDITION` and `wq.REGISTRY_DERIVERS`).
    """
    return ConstraintSpec(fields={"ddx_condition": Fix(spec_id)}, tightness="condition",
                          note="诊断条件:金标由它推出,其余全部放开")


def resolve(cs: ConstraintSpec, case_id: str) -> dict:
    """Resolve a constraint set into a `raw`. Takes only case_id and constraints
    as input, never the diagnosis (answer-neutral).

    Unconstrained dimensions come from `demographics.sample_profile`
    (counter-based), so loosening one dimension never moves another.
    """
    from . import rng
    from .demographics import doses_per_week, sample_profile
    sex = cs.of("sex")
    base = sample_profile(case_id, sex.value if isinstance(sex, Fix) else "F")
    out = dict(base)
    for name, c in cs.fields.items():
        if name in DERIVED_FROM:
            continue
        if isinstance(c, Fix):
            out[name] = c.value
        elif isinstance(c, OneOf) and c.options:
            out[name] = rng.pick(list(c.options), case_id, "constraint", name)
        elif isinstance(c, Range):
            u = rng.unit(case_id, "constraint", name)
            out[name] = round(c.lo + u * (c.hi - c.lo), 1)
    out["dose_steps"] = list(base["dose_steps"]) if out.get("drug") == base.get("drug") \
        else _steps_for(out.get("drug"))
    from .job import raw_field as _raw_field
    out.setdefault("disease", _raw_field(base, "disease"))
    # frequency is derived from the drug (see DERIVED_FROM)
    out["_doses_per_week"] = doses_per_week(str(out["drug"]))
    return out


def _steps_for(drug: str | None) -> list[float]:
    from .demographics import DRUGS
    for d, steps in DRUGS:
        if d == drug:
            return list(steps)
    raise Infeasible(f"drug={drug!r} is not in the registry, cannot derive a dosing schedule")
