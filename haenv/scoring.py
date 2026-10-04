"""scoring.py -- the scoring profile (`registry/scoring.yaml`): roles, validity, and drift checks.

Roles: `gate` (non-compensatory multiplier), `dim` (counts toward the core
mean), `diagnostic` (reported, never scored), `anchor` (normalization only;
forces the item-blind floor to be printed). The scored set is pinned in the
profile so batches stay comparable; `profile_drift` reports when a batch's
measurements disagree with it. A `dim` of judgment class needs blind-label
agreement >= `validity_threshold` (0.70); `unmeasured` dimensions are admitted
provisionally, `failing` ones of kind `construct` are barred.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import hashlib

import logging
from pathlib import Path

import yaml

from .yamlcache import load_yaml as _cached_yaml
from haenv import data_root as _dr   # resource root: source tree = repo root . wheel = haenv/_data

log = logging.getLogger("haenv.scoring")

#: `retired`: no longer computed into any readout of the board (dead or superseded); the
#: entry stays so historical readings keep a documented meaning.
ROLES = frozenset({"gate", "dim", "diagnostic", "anchor", "retired"})
#: The legal values of `direction`. The composite is a higher-is-better mean, so
#: every `role: dim` metric must declare its direction.
DIRECTIONS = frozenset({"higher", "lower", "non_monotone"})

# Fields allowed on each metric in the profile (default-deny).
METRIC_FIELDS = frozenset({
    "role", "why", "source_metric", "validity", "pending_decision",
    "paired_with", "belongs_to", "forces_floor_print", "superseded_reason",
    # `direction` -- required for `role: dim`.
    "direction",
    # `pair_aggregate` -- `harmonic` combines a recall/precision pair into one
    # slot via F1, so maxing recall alone cannot game the composite.
    "pair_aggregate",
    # `decided` -- a settled decision and its basis (vs. `pending_decision`).
    "decided",
    # `failing_kind` -- `defect` (fixable; stays in score, not publishable),
    # `construct` (measures the wrong thing; barred), `disputed` (awaiting clinical
    # review; stays in score, not publishable).
    "failing_kind",
    # `effective_set` -- the difficulty tier, pre-registered before annotation.
    "effective_set",
    # `applies_to` -- geometries under which this dimension has a value; elsewhere it is not applicable, not zero.
    "applies_to",
    # `scored_on` -- geometries on which a `role: dim` enters the composite; elsewhere it is
    # still computed and reported, not scored (undeclared = every geometry).
    "scored_on",
    # `not_scored_on_kinds` -- gold kinds on whose cells a scored dimension is not applicable
    # (neither 0 nor in the denominator); the per-cell value is still recorded and reported.
    "not_scored_on_kinds",
    # `retired_on` / `retired_why` -- date and reason of `role: retired`.
    "retired_on", "retired_why",
    # `profile_since` / `profile_why` / `profile_flag` -- A4 2026-10-01: a `role: diagnostic` metric
    # shown as a profile reading, with the reason it is not scored and any validity flag.
    "profile_since", "profile_why", "profile_flag",
    # `provisional` -- why a scored dimension is admitted provisionally only.
    "provisional",
    # `requires_gold` -- gold key the dimension needs (`ddx` / `warranted_two_sided`, see `Profile.gold_present`).
    "requires_gold",
    # `aggregate: per_solver` -- not a row field; computed per solver.
    "aggregate",
    # `metric_class` -- `computable` (the criterion is an identity, e.g. "cited
    # evidence is in the ledger"; skips the validity precondition) or `judgment`
    # (default; needs a blind-label anchor).
    "metric_class",
    # `revisit_when` -- condition under which the role should be reconsidered; checked every batch.
    "revisit_when",
    # `metric_type` -- `score01` / `binary` / `count` / `error`; only the first two
    # enter `score_full`.
    "metric_type",
})


class ScoringError(RuntimeError):
    pass


class Profile:
    """A scoring profile. Read-only: change it by editing the yaml."""

    def __init__(self, doc: dict, path: Path):
        self.path = path
        # `content_sha256` changes with the profile content even when `profile_id` does not.
        self.content_sha256 = hashlib.sha256(
            path.read_bytes()).hexdigest() if path.is_file() else ""
        self.profile_id = str(doc.get("profile_id") or "")
        self.task_type = str(doc.get("task_type") or "")
        self.threshold = float(doc.get("validity_threshold") or 0.0)
        self.sources = tuple(doc.get("validity_sources") or ())
        self.metrics: dict[str, dict] = dict(doc.get("metrics") or {})
        self._oor: dict = {}                      # out-of-range ledger, see `note_out_of_range`
        if not self.profile_id:
            raise ScoringError(f"{path}: missing profile_id -- a profile with no id cannot be referenced in an artefact")
        if not (0.0 < self.threshold <= 1.0):
            raise ScoringError(f"{path}: validity_threshold={self.threshold!r} is not in (0,1]")
        for name, m in self.metrics.items():
            if not isinstance(m, dict):
                raise ScoringError(f"{path}:{name} must be a mapping")
            bad = set(m) - METRIC_FIELDS
            if bad:
                raise ScoringError(f"{path}:{name} has unregistered fields {sorted(bad)}"
                                   f"(allowed: {sorted(METRIC_FIELDS)})")
            if str(m.get("role")) not in ROLES:
                raise ScoringError(f"{path}:{name} role={m.get('role')!r} is invalid (allowed: {sorted(ROLES)})")
            if not str(m.get("why") or "").strip():
                raise ScoringError(f"{path}:{name} is missing `why` -- "
                                   f"a role assignment that cannot state its reason cannot be reviewed by the next person")
            _dir = m.get("direction")
            if _dir is not None and str(_dir) not in DIRECTIONS:
                raise ScoringError(f"{path}:{name} direction={_dir!r} is invalid"
                                   f"(allowed: {sorted(DIRECTIONS)})")
            if str(m.get("role")) == "dim" and _dir is None:
                raise ScoringError(
                    f"{path}:{name} is role: dim but is missing `direction` -- "
                    f"the composite score is a higher-is-better arithmetic mean, guessing the direction can flip its sign, "
                    f"and after flipping the reading still looks completely normal (allowed: {sorted(DIRECTIONS)})")

    # ---------------------------------------------------------------- role queries
    def by_role(self, role: str) -> list[str]:
        return [k for k, m in self.metrics.items() if m.get("role") == role]

    def harmonic_pairs(self) -> list[tuple[str, str]]:
        """Pairs declaring `pair_aggregate: harmonic`, as `[(primary, paired), ...]` in profile order.

        A `pair_aggregate` without a valid `paired_with` raises.
        """
        order = list(self.metrics)
        out: list[tuple[str, str]] = []
        for name, m in self.metrics.items():
            if str(m.get("pair_aggregate") or "") != "harmonic":
                continue
            other = str(m.get("paired_with") or "")
            if not other or other not in self.metrics:
                raise ScoringError(
                    f"{name}: pair_aggregate=harmonic but paired_with={other!r} does not hold -- "
                    f"a paired aggregate with no partner is a typo, silently ignoring it would let the composite score change its rule unnoticed")
            a, b = sorted((name, other), key=order.index)
            if (a, b) not in out:
                out.append((a, b))
        return out

    def scored_on(self, name: str, geometry: str | None) -> bool:
        """Whether a scored dimension enters the composite on `geometry`. Undeclared = everywhere."""
        a = (self.metrics.get(name) or {}).get("scored_on")
        return True if (not a or geometry is None) else (str(geometry) in [str(x) for x in a])

    def scored_on_kind(self, name: str, gold_kind: str | None) -> bool:
        """Whether a scored dimension is applicable on a cell of `gold_kind`. Undeclared = every kind."""
        a = (self.metrics.get(name) or {}).get("not_scored_on_kinds")
        return True if (not a or gold_kind is None) else (str(gold_kind) not in [str(x) for x in a])

    def retired(self) -> list[str]:
        return self.by_role("retired")

    def columns(self) -> dict:
        """A4 (2026-10-01): the three public columns of a board.

        `scored` -- `role: dim` (enters the composite on the geometries it lists);
        `profile` -- `role: diagnostic`: computed, stored and shown, not scored ({name: flag or None});
        `retired` -- `role: retired`, only dimensions that judge wrongly ({name: retired_why}).
        """
        return {"scored": self.scored_dims,
                "profile": {k: (m.get("profile_flag") or None) for k, m in self.metrics.items()
                            if m.get("role") == "diagnostic"},
                "retired": {k: str(m.get("retired_why") or "") for k, m in self.metrics.items()
                            if m.get("role") == "retired"}}

    def applies_on(self, name: str, geometry: str) -> bool:
        """Whether this dimension can have a value under `geometry`. Undeclared = all geometries."""
        a = (self.metrics.get(name) or {}).get("applies_to")
        return True if not a else (str(geometry) in [str(x) for x in a])

    @staticmethod
    def gold_present(cases: list[dict]) -> dict[str, int]:
        """Which gold classes are present in `cases` (parsed `cases.jsonl` rows), keyed like `requires_gold`."""
        n_ddx, warr = 0, set()
        for c in cases:
            inner = c.get("case") if isinstance(c.get("case"), dict) else c
            adj = (inner or {}).get("adjudication") or {}
            if adj.get("ddx"):
                n_ddx += 1
            if "clinician_action_warranted" in adj:
                warr.add(bool(adj["clinician_action_warranted"]))
        return {"ddx": n_ddx, "warranted_two_sided": 1 if len(warr) >= 2 else 0}

    def gold_ok(self, name: str, gold: dict[str, int] | None) -> bool:
        """Whether this dimension's gold is present in this batch. Undeclared `requires_gold` => present."""
        need = (self.metrics.get(name) or {}).get("requires_gold")
        if not need or gold is None:
            return True
        return int(gold.get(str(need), 0)) > 0

    def is_aggregate(self, name: str) -> bool:
        """Whether this dimension is aggregated per solver (no field in `eval.jsonl`)."""
        return str((self.metrics.get(name) or {}).get("aggregate") or "") == "per_solver"

    @property
    def scored_dims(self) -> list[str]:
        """Dimensions that count toward the composite score, in profile order."""
        return self.by_role("dim")

    #: The only `failing_kind` that bars entry to the composite score.
    BARRING_KINDS = ("construct",)

    def failing_kind(self, name: str) -> str | None:
        """The unresolved validity issue kind (`defect` / `construct` / `disputed`), or `None`.

        Independent of whether aggregate agreement passes the threshold.
        """
        v = (self.metrics.get(name) or {}).get("validity") or {}
        k = v.get("failing_kind")
        return str(k) if k else None

    def publishable_dims(self) -> list[str]:
        """Scored dimensions that may be published: `passing` validity and no `failing_kind`."""
        return [k for k in self.scored_dims
                if self.validity_state(k)[0] == "passing" and not self.failing_kind(k)]

    def blocked_from_publish(self) -> list[dict]:
        """Scored dimensions blocked from publication, with the reason and who can resolve it."""
        out = []
        for k in self.scored_dims:
            st, a = self.validity_state(k)
            if st == "passing" and not self.failing_kind(k):
                continue
            v = (self.metrics.get(k) or {}).get("validity") or {}
            out.append({"metric": k, "state": st, "agreement": a,
                        "failing_kind": self.failing_kind(k),
                        "who": v.get("who_can_resolve")
                        or {"defect": "ourselves", "disputed": "clinical review",
                            "construct": "fix the judge"}.get(self.failing_kind(k) or "", "pending")})
        return out

    def metric_class(self, name: str) -> str:
        """`computable` / `judgment`; defaults to judgment."""
        return str((self.metrics.get(name) or {}).get("metric_class") or "judgment")

    def direction(self, name: str) -> str:
        """`higher` / `lower` / `non_monotone`.

        Declared in the profile (required for `role: dim`), else taken from
        `separation.PROCESS_DIMS`, else `higher`.
        """
        _d = (self.metrics.get(name) or {}).get("direction")
        if _d:
            return str(_d)
        try:
            from .separation import PROCESS_DIMS as _PD
        except Exception:                                    # noqa: BLE001
            return "higher"
        _s = _PD.get(name)
        return str(_s.direction) if _s else "higher"

    def scored_dims_signed(self) -> list[tuple[str, str]]:
        """Scored dimensions with their directions. `non_monotone` ones are passed through; the aggregator excludes and records them."""
        return [(k, self.direction(k)) for k in self.scored_dims]

    def computable_dims(self) -> list[str]:
        """Computable scored dimensions (no blind-label anchor needed)."""
        return [k for k in self.scored_dims if self.metric_class(k) == "computable"]

    def judgment_dims(self) -> list[str]:
        """Judgment scored dimensions (subject to the validity precondition)."""
        return [k for k in self.scored_dims if self.metric_class(k) != "computable"]

    def validity_state(self, name: str) -> tuple[str, float | None]:
        """`(passing|failing|unmeasured, agreement)`. No reading means unmeasured, not 0."""
        if self.metric_class(name) == "computable":
            # A computable criterion is an identity, so it passes by definition.
            return "passing", None
        v = (self.metrics.get(name) or {}).get("validity") or {}
        a = v.get("agreement")
        if a is None:
            return "unmeasured", None
        return ("passing" if float(a) >= self.threshold else "failing"), float(a)

    def provisional_dims(self) -> list[str]:
        """Scored dimensions with `unmeasured` validity: admitted provisionally, flagged in the report, not for published boards."""
        return [k for k in self.scored_dims if self.validity_state(k)[0] == "unmeasured"]

    #: Out-of-range ledger `{dim: {"n", "min", "max"}}` filled by `note_out_of_range`.
    _oor: dict

    def note_out_of_range(self, name: str, v) -> bool:
        """Record a `score01` value outside [0, 1]; return whether it was out of range.

        Records only, never raises: sources clamp already, so a reading here means
        source clamping did not take effect.
        """
        m = self.metrics.get(name) or {}
        if str(m.get("metric_type") or "") != "score01":
            return False
        try:
            x = float(v)
        except (TypeError, ValueError):
            return False
        if 0.0 <= x <= 1.0:
            return False
        e = self._oor.setdefault(name, {"n": 0, "min": x, "max": x})
        e["n"] += 1
        e["min"], e["max"] = min(e["min"], x), max(e["max"], x)
        log.warning("[scoring] `%s` declared score01 but got %.4f -- the out-of-range value "
                    "entered the average; clamping at the source did not take effect", name, x)
        return True

    def out_of_range_report(self) -> dict:
        """`score01` dimensions that went out of range during this render; `{}` = clean."""
        return {k: dict(v) for k, v in sorted(self._oor.items())}

    #: Metric types accepted into `score_full`.
    FULL_TYPES = ("score01", "binary")

    def full_dims(self) -> list[str]:
        """Dimensions in `score_full`: every `metric_type in FULL_TYPES`, regardless of role.

        Comparing its ranking with the scored composite shows whether excluded
        dimensions would change the conclusion.
        """
        return [k for k, m in self.metrics.items()
                if str(m.get("metric_type") or "") in self.FULL_TYPES
                and m.get("role") != "retired"]

    def revisit_conditions(self) -> list[dict]:
        """Dimensions with a `revisit_when` condition, with their current role."""
        return [{"metric": k, "role": m.get("role"), "when": str(m.get("revisit_when"))}
                for k, m in self.metrics.items() if m.get("revisit_when")]

    def illegal_dims(self) -> list[dict]:
        """Scored dimensions whose `failing_kind` is in `BARRING_KINDS`."""
        out = []
        for k in self.scored_dims:
            st, a = self.validity_state(k)
            if self.failing_kind(k) in self.BARRING_KINDS:
                out.append({"metric": k, "agreement": a, "threshold": self.threshold,
                            "failing_kind": self.failing_kind(k)})
        return out

    def unmet_validity(self) -> list[dict]:
        """Scored dimensions that do not meet the validity precondition."""
        out = []
        for k in self.scored_dims:
            state, a = self.validity_state(k)
            if state != "passing":
                out.append({"metric": k, "state": state, "agreement": a,
                            "threshold": self.threshold,
                            "pending": (self.metrics[k].get("pending_decision") or "")})
        return out


def load_profile(root: Path | None = None, task_type: str = "joint_dx") -> Profile:
    root_dir = (root or _dr())
    if task_type == "joint_dx":
        p = root_dir / "registry/scoring.yaml"
    else:
        p = root_dir / f"registry/scoring_{task_type}.yaml"
        if not p.is_file():
            p_fallback = root_dir / "registry/scoring.yaml"
            if p_fallback.is_file():
                doc = _cached_yaml(p_fallback) or {}
                if doc.get("task_type") == task_type:
                    p = p_fallback
    if not p.is_file():
        raise ScoringError(f"scoring profile not found: {p} (task_type={task_type!r}) -- profile for this item type is missing or not covered")
    doc = _cached_yaml(p) or {}
    prof = Profile(doc, p)
    if prof.task_type and task_type and prof.task_type != task_type:
        raise ScoringError(f"[scoring] profile {p.name} is for {prof.task_type}, current task is {task_type} (fail-closed)")
    # LLM-judged dimensions may only be `diagnostic`, so judge changes stay
    # recomputable without new model calls.
    from .llm_rubric import DIM as _LR_DIM, assert_role_is_diagnostic as _assert_lr_role
    if ((doc.get("metrics") or {}).get(_LR_DIM)) is not None:
        _assert_lr_role(doc, _LR_DIM)
    return prof


# ---------------------------------------------------------------- drift check
def profile_drift(prof: Profile, health: dict[str, dict]) -> list[dict]:
    """Compare the profile's pinned roles with this batch's `dimension_health` readings.

    Reports `stale_scored` (scored but no longer healthy) and `stale_excluded`
    (excluded but usable again); changes no judgment.
    """
    out = []
    for name, h in (health or {}).items():
        if not isinstance(h, dict):
            continue
        role = str((prof.metrics.get(name) or {}).get("role") or "")
        ok = bool(h.get("usable_as_capability"))
        if role == "dim" and not ok:
            out.append({"kind": "stale_scored", "metric": name,
                        "detail": str(h.get("verdict") or "this batch's measurement is not passing")})
        elif role == "diagnostic" and ok:
            out.append({"kind": "stale_excluded", "metric": name,
                        "detail": str(h.get("verdict") or "usable as capability on this batch")})
    return out
