"""Optional gold-layer history, dispatched through Pack without clinical branches.

A zero rate leaves plans, records and random draws untouched. Clinical adapters
prepare hidden inputs before realization and finalize the paired visible records
before the existing emission gates. History quantity is a separate world knob.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Callable

from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached


@registry_cached("history_dependency.yaml")
def table():
    return load_registry("history_dependency.yaml")


def uniform(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:13], 16) / 16 ** 13


def rate(pack, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import config
    if enabled(plan) and "history_dependency_rate" not in plan and "HAENV_HISTORY_DEPENDENCY_RATE" not in os.environ:
        return float(config(pack)["rate"]["value"])
    value = plan.get("history_dependency_rate", os.environ.get("HAENV_HISTORY_DEPENDENCY_RATE"))
    r = float(table()["packs"][pack]["rate"]["value"] if value is None else value)
    if not 0 <= r <= 1:
        raise ValueError("history_dependency_rate must be in [0, 1]")
    return r


def old_day(pack, case_id, T):
    cfg = table()["timing"]
    lo, hi = cfg["age_days"]["value"]
    age = int(lo) + int(uniform("history-date", case_id) * (int(hi) - int(lo) + 1))
    return int(T) - int(cfg["recent_days"]["value"][pack]) - age


@dataclass(frozen=True)
class HistoryDependency:
    block: str
    plan_of: Callable
    prepare: Callable
    finalize: Callable
    answer: Callable
    twin: Callable
    ready: Callable | None = None


_INSTALLED = set()


def install(pack):
    """Register history then its declared extensions, without touching build functions."""
    from functools import partial
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    from haenv.app.pack_hooks import PACK_EXTENSIONS
    register_hook(Stage.OBSERVE, "history:" + pack.name, partial(_observe_history, pack=pack))
    for callback in tuple(PACK_EXTENSIONS.values()):
        callback(pack)


def _observe_history(next_call, raw, case_id, *, pack):
    p = pack.history_dependency
    plan = p.plan_of(raw)
    from haenv.world.validation.difficulty_r2 import enabled as r2_enabled
    enabled = plan is not None and (rate(pack.name, plan) > 0 or r2_enabled(plan))
    if enabled:
        from haenv.world.generation.build_clinical import _LAB_CTX
        ctx = _LAB_CTX.get(str(case_id))
        draw_policy = table()["packs"][pack.name].get("additional_draw_policy")
        if ctx is not None and draw_policy:
            ctx["additional_draw_policy"] = draw_policy["value"]
        p.prepare(raw, str(case_id), plan)
    truth, audit = next_call(raw, case_id)
    complete = p.ready(raw) if p.ready else p.block in (raw.prediction_context or {})
    if enabled and complete:
        p.finalize(raw, str(case_id), plan)
    return truth, audit


def observe(raw, case_id, base_observer):
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import call_hook
    return call_hook(Stage.OBSERVE, base_observer, raw, case_id)


def policy(pack):
    from haenv.app.packs import PACKS
    if pack not in PACKS:
        import importlib
        importlib.import_module("haenv." + pack + ".plugin").register()
    return PACKS[pack].history_dependency


def metadata(raw, pack):
    p = policy(pack)
    g = (raw.get("adjudication") if isinstance(raw, dict) else raw.adjudication) or {}
    return (g.get(p.block) or {}).get("history_dependency") or g.get("history_dependency") or {}


def remove_early(raw, lower, day=None):
    """Delete dated record objects, never truth labels or assumptions in a plan."""
    from copy import deepcopy
    def visit(value):
        if isinstance(value, list):
            kept = []
            for row in value:
                d = next((row[k] for k in ("day", "draw_day", "ts", "source_timestamp", "onset_day") if isinstance(row, dict) and isinstance(row.get(k), (int, float))), None)
                if d is not None and int(d) < lower and (day is None or int(d) == day):
                    continue
                kept.append(visit(row))
            return kept
        if isinstance(value, dict):
            return {k: (v if k in ("plan", "history_dependency", "latent_premise") else visit(v)) for k, v in value.items()}
        return value
    return visit(raw)


def early_days(raw, lower):
    def collect(value):
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    d = next((row[k] for k in ("day", "draw_day", "ts", "source_timestamp", "onset_day") if isinstance(row.get(k), (int, float))), None)
                    if d is not None and int(d) < lower:
                        yield int(d)
                yield from collect(row)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in ("plan", "history_dependency", "latent_premise", "longitudinal_data"):
                    yield from collect(v)
    return sorted(set(collect(raw)))
