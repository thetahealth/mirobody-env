"""Pack data and callback signatures. Imports only the Python standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: weight-forecast scaffold of the base world: a frame that asks no weight forecast does not show it
WEIGHT_FORECAST_CONTEXT = ("target_event_type", "prediction_window", "available_history_window")
WEIGHT_FORECAST_PROFILE = ("treatment_goals",)


@dataclass(frozen=True)
class PayloadView:
    """What a pack's frame reads off the solver payload.

    `streams`: the longitudinal streams the frame reads (None keeps every stream); a stream an
    evidence-ledger value is read from (`gate_tables.LEDGER_VALUE_STREAMS`) is always kept, so a
    shown value stays traceable. `drop_context` / `drop_profile`: keys removed from
    `prediction_context` / `user_profile`. `applies`: raw -> bool, the cases of the pack the view
    covers (None: all of them)."""
    streams: tuple | None = None
    drop_context: tuple = WEIGHT_FORECAST_CONTEXT
    drop_profile: tuple = WEIGHT_FORECAST_PROFILE
    applies: Callable | None = None


@dataclass
class Pack:
    name: str                                   # marker / stamp prefix ("p4", "pack2", "m2")
    owns: Callable                              # raw case -> bool: a case of this pack
    blocks: tuple = ()                          # (name, gold_fn, probe_fn, latent_keys, latent_classes, why)
    emission: object | None = None              # shared_audit.Emission
    framing: tuple | None = None                # (name, template, why)
    probe: Callable | None = None               # () -> probe dict
    judge: tuple | None = None                  # (name, fn, kinds, when)
    mounts: dict = field(default_factory=dict)  # geometry -> "out"
    why_not: dict = field(default_factory=dict)
    prompt_gold_meta: tuple = ()                # extra premise `meta` keys a pack case must not show
    judging_segment: str | None = None          # make_freeze segment stamped on rows
    source: str = ""
    view: PayloadView | None = None             # solver view of the pack's cases (None: the whole payload)
    acute: Callable | None = None               # raw -> [{"day", "temp_c", "hr_bpm", "spo2_pct"}] acute vitals
    age_from_conditions: bool = True            # the builder redraws the age band from the conditions
    base_labels: tuple | None = None            # base-world labels (driver/outcome/join_gold) the pack's score
                                                # reads; None: all. The build's A5 scan grades only these
    history_dependency: object | None = None     # optional HistoryDependency; no-op when unset or rate zero
    answers_action: bool = True                 # False: the pack's answer has no action class, review flag,
                                                # tests or action text, so the kernel gates that read them
                                                # (`ACTION_FIELD_GATES`) would fire on the item alone; dropped
    gated_row: Callable | None = None           # (base_row, cid, sname, raw, T, solver, *, ctx) -> row
    gold_derivations: tuple["GoldDerivation", ...] = ()


@dataclass(frozen=True)
class GoldDerivation:
    """A named, ordered registry policy. No authored-gold recovery is implied.

    Applies to a world/field pair; derive receives the registry value and its
    provenance and returns a replacement (value, provenance). Exceptions propagate.
    """
    name: str
    applies: Callable
    derive: Callable


@dataclass
class PackRegistry:
    """Ordered declarations only; registering never imports or replaces an executor.

    `packs` may be supplied by a caller to keep independent runs isolated.
    The first declaration for a name wins, matching historical pack registration.
    """
    packs: dict[str, Pack] = field(default_factory=dict)

    def register(self, pack: Pack) -> Pack:
        return self.packs.setdefault(pack.name, pack)

    def owner(self, raw) -> Pack | None:
        return next((p for p in self.packs.values() if p.owns(raw)), None)
