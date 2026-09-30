"""coupling.py -- the discrete comorbidity layer: rules read and write per-patient
attributes on a blackboard (e.g. CKD stage 3 => metformin reduced => a different diabetes
branch), changing which path is taken, not daily values.

`apply_couplings` returns a `CouplingTrace` of the rules that fired, so every coupling is
observable in the artifact.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from ..yamlcache import load_yaml as _cached_yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "in": lambda a, b: a in b,
}


class CouplingError(ValueError):
    """A coupling rule is missing, invalid, or conflicting. Deliberately no
    default value is provided."""


# Blackboard attributes are derived from `events.Facts` (read by duck typing; `events`
# imports this package). A missing attribute is either a typo, rejected at load by
# `validate_rule_attrs`, or a value this patient lacks, which skips the rule per case.

#: How diabetes is written in the comorbidity table (`disease` only ever holds one
#: value => a second disease can only be expressed through `comorbidities`).
_DM_TERMS = ("t2d", "t2dm", "diabetes", "type2_diabetes", "糖尿病")
_HTN_TERMS = ("hypertension", "htn", "高血压")


def _has(facts, disease_name: str, terms: Sequence[str]) -> bool:
    """True if the disease is the primary `disease` or appears in `comorbidities` (the only
    way a second disease can be expressed)."""
    if str(getattr(facts, "disease", "") or "").strip().lower() == disease_name:
        return True
    return any(str(c).strip().lower() in terms
               for c in (getattr(facts, "comorbidities", ()) or ()))


#: attribute -> (derivation, every patient has a value, description). Only attributes a
#: rule reads are registered.
DERIVERS: dict[str, tuple[Any, bool, str]] = {
    "has_diabetes": (lambda f: _has(f, "t2d", _DM_TERMS), True,
                     "主病位 T2D 或共病表里有糖尿病"),
    "has_hypertension": (lambda f: _has(f, "hypertension", _HTN_TERMS), True,
                         "主病位 hypertension 或共病表里有高血压"),
    "has_hypothyroidism": (lambda f: bool(getattr(f, "hypothyroid", False)), True,
                           "`Facts.hypothyroid`(共病名以 hypothyroid 开头)"),
    "has_osa": (lambda f: bool(getattr(f, "osa", False)), True,
                "`Facts.osa`(共病名等于 OSA)"),
    # Partial: absent BMI is omitted, never given a sentinel.
    "bmi": (lambda f: getattr(f, "bmi", None), False, "`Facts.bmi`,缺失时不写"),
}

def blackboard_of(facts) -> Blackboard:
    """Derives the blackboard from `Facts` per `DERIVERS`. A partial attribute
    that's missing has its key omitted entirely."""
    bb = Blackboard()
    for name, (fn, _total, _desc) in DERIVERS.items():
        v = fn(facts)
        if v is None:
            continue
        bb.set(name, v)
    return bb


def validate_rule_attrs(rules: Sequence["CouplingRule"]) -> None:
    """Load-time check: every attribute an active rule reads is derivable or written by
    another rule; otherwise the rule can never fire."""
    produced = set(DERIVERS) | {k for r in rules for k in r.then}
    bad: list[str] = []
    for r in rules:
        if r.status != "active":
            continue
        miss = sorted(set(r.when) - produced)
        if miss:
            bad.append(f"{r.rule_id} reads {miss}")
    if bad:
        raise CouplingError(
            "coupling rules read blackboard attributes that cannot be derived: " + "; ".join(bad)
            + f". Derivable: {sorted(DERIVERS)}, plus attributes written by rules' `then`. "
            "Add a derivation to `DERIVERS`, or mark the rule "
            "`status: no_attribute` and give the reason"
        )


@dataclass
class Blackboard:
    """Per-patient shared attributes (scalars and strings only, so they stay traceable)."""

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        if key not in self.values:
            raise CouplingError(
                f"blackboard has no {key!r}: a rule reads an attribute that was never written"
            )
        return self.values[key]

    def set(self, key: str, value: Any) -> None:
        if not isinstance(value, (int, float, str, bool)) or isinstance(value, bytes):
            raise CouplingError(f"{key}: blackboard accepts only scalars/strings, got {type(value).__name__}")
        self.values[key] = value

    def has(self, key: str) -> bool:
        return key in self.values


@dataclass(frozen=True)
class CouplingRule:
    """One coupling: `when` is a conjunction `{attribute: [op, value]}`, `then` the attributes
    written back. `source` (clinical basis or design decision) is required."""

    rule_id: str
    when: Mapping[str, tuple[str, Any]]
    then: Mapping[str, Any]
    source: str
    review: str = "pending"     # `pending` must propagate all the way to the report
    # `active`, or an explicit declaration that the rule cannot fire on the current corpus:
    # `no_attribute` (the attribute is not derivable) or `no_subject` (no matching patient).
    # Both need `status_reason`.
    status: str = "active"
    status_reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.rule_id).strip():
            raise CouplingError("rule_id must not be empty")
        if str(self.status) not in ("active", "no_attribute", "no_subject"):
            raise CouplingError(
                f"{self.rule_id}: status must be active / no_attribute / no_subject, "
                f"got {self.status!r}"
            )
        if self.status != "active" and not str(self.status_reason).strip():
            raise CouplingError(
                f"{self.rule_id}: status={self.status} requires status_reason "
                "(why the rule never fires: missing attribute or missing patient)"
            )
        if not self.when:
            raise CouplingError(f"{self.rule_id}: when must not be empty (an unconditional rule is a constant, not a coupling)")
        if not self.then:
            raise CouplingError(f"{self.rule_id}: then must not be empty")
        if not str(self.source).strip():
            raise CouplingError(f"{self.rule_id}: source must not be empty")
        if str(self.review).strip() not in ("pending", "done"):
            raise CouplingError(
                f"{self.rule_id}: review must be 'pending' or 'done', got {self.review!r}"
            )
        for attr, cond in self.when.items():
            if not (isinstance(cond, (list, tuple)) and len(cond) == 2):
                raise CouplingError(f"{self.rule_id}/{attr}: condition must be [operator, value]")
            if cond[0] not in _OPS:
                raise CouplingError(
                    f"{self.rule_id}/{attr}: unknown operator {cond[0]!r}; allowed {sorted(_OPS)}"
                )

    def matches(self, bb: Blackboard) -> bool:
        """Whether every condition holds; referencing an absent attribute raises."""
        for attr, (op, want) in self.when.items():
            if not bb.has(attr):
                raise CouplingError(
                    f"{self.rule_id}: condition reads attribute {attr!r}, which is not on the blackboard "
                    "(a typo would be indistinguishable from a false condition)"
                )
            if not _OPS[op](bb.get(attr), want):
                return False
        return True


@dataclass
class CouplingTrace:
    """Which rules fired and what they wrote, recorded in the artifact."""

    fired: list[str] = field(default_factory=list)
    writes: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Plain-data shape for writing to the artifact."""
        return {"fired_rules": list(self.fired), "coupled_attrs": dict(self.writes)}


def apply_couplings(
    bb: Blackboard,
    rules: Sequence[CouplingRule],
    max_rounds: int | None = None,
) -> CouplingTrace:
    """Apply rules to a fixed point and return the firing trace. Every round re-evaluates all
    rules, so conflicting writes oscillate and raise after `max_rounds` (default
    `len(rules) + 1`, enough for any acyclic chain)."""
    limit = len(rules) + 1 if max_rounds is None else max_rounds
    if limit <= 0:
        raise CouplingError(f"max_rounds must be > 0, got {limit}")

    trace = CouplingTrace()
    for _ in range(limit):
        changed = False
        for r in rules:
            if not r.matches(bb):
                continue
            if r.rule_id not in trace.fired:
                trace.fired.append(r.rule_id)
            for k, v in r.then.items():
                if bb.has(k) and bb.get(k) == v:
                    continue
                bb.set(k, v)
                trace.writes[k] = v
                changed = True
        if not changed:
            return trace
    raise CouplingError(
        f"coupling rules did not converge within {limit} rounds: the rule set has a cycle "
        "(two rules write conflicting values to the same attribute), so output would depend on evaluation order"
    )


def load_coupling_rules(path: str | Path) -> list[CouplingRule]:
    """Reads the coupling registry. Any single non-conforming entry `raise`s —
    no skipping, no warn-and-continue."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise CouplingError(f"coupling registry not found: {p}")
    doc = _cached_yaml(p)
    if not isinstance(doc, Mapping) or "rules" not in doc:
        raise CouplingError(f"{p}: top level must be a mapping with `rules`")

    rules: list[CouplingRule] = []
    seen: set[str] = set()
    for raw in doc["rules"] or []:
        if not isinstance(raw, Mapping):
            raise CouplingError(f"{p}: each rule must be a mapping")
        for k in ("id", "when", "then", "source"):
            if k not in raw:
                raise CouplingError(f"{p}: rule missing field {k!r}; no defaults are provided")
        rid = str(raw["id"])
        if rid in seen:
            raise CouplingError(f"{p}: duplicate rule id {rid!r}")
        seen.add(rid)
        when = {str(a): tuple(c) for a, c in (raw["when"] or {}).items()}
        rules.append(CouplingRule(
            rule_id=rid, when=when, then=dict(raw["then"]),
            source=str(raw["source"]), review=str(raw.get("review", "pending")),
            status=str(raw.get("status", "active")),
            status_reason=str(raw.get("status_reason", "")),
        ))
    if not rules:
        raise CouplingError(f"{p}: registry is empty")
    validate_rule_attrs(rules)
    return rules


def active_rules(rules: Sequence[CouplingRule]) -> list[CouplingRule]:
    """Only `status: active` rules participate in evaluation. The other two
    statuses are explicit declarations that the rule cannot fire."""
    return [r for r in rules if r.status == "active"]


def couple_case(facts, rules: Sequence[CouplingRule]) -> dict[str, Any]:
    """Production entry: `Facts` -> blackboard -> rules -> a record for the Q-side ledger
    (`fired_rules`, `coupled_attrs`, `skipped_rules`, `derived_attrs`)."""
    bb = blackboard_of(facts)
    derived = dict(bb.values)      # snapshot before the rules write
    runnable, skipped = [], {}
    for r in active_rules(rules):
        miss = sorted(a for a in r.when if not bb.has(a))
        if not miss:
            runnable.append(r)
            continue
        # A skip is recorded only when every other condition holds, i.e. the missing
        # attribute alone decides whether the rule fires.
        if all(_OPS[op](bb.get(a), want) for a, (op, want) in r.when.items() if bb.has(a)):
            skipped[r.rule_id] = miss
    trace = apply_couplings(bb, runnable)
    rec = trace.as_record()
    rec["skipped_rules"] = skipped
    rec["derived_attrs"] = derived
    return rec


def pending_rules(rules: Sequence[CouplingRule]) -> list[str]:
    """Rule ids with `review: pending`, for the report."""
    return sorted(r.rule_id for r in rules if str(r.review).strip() == "pending")
