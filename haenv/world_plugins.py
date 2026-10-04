"""Registration surface for world-side plugins.

An out-of-tree package can contribute streams, event pools, drug effects and post-kernel
injectors. Plugin output still passes the generation gates, no plugin may write a
label-bearing stream, and key collisions raise unless `replace=True`. The plugin manifest
enters the world fingerprint, each entry with a hash of its value; with no plugins attached
the fingerprint is unchanged.

Registration happens before generation starts and on one thread (`job.load_job` runs before
the build thread pool); consumers read the overlay without locks.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import logging
import threading
from typing import Any, Callable

log = logging.getLogger("haenv.world_plugins")

#: Label-bearing streams (weight feeds `build.label_rule`). No plugin may write them.
LABEL_BEARING: tuple[str, ...] = ("weight", "weight_ref")

#: Target (table, section) of each registration kind.
TARGETS: dict[str, tuple[str, str]] = {
    "stream": ("physio_streams.yaml", "streams"),
    "kernel": ("physio_kernels.yaml", "kernels"),
    "indicator": ("indicators.yaml", "indicators"),
    "benign_event": ("benign_events.yaml", "benign_events"),
    "life_event": ("benign_events.yaml", "life_events"),
    "symptom": ("symptom_topics.yaml", "symptoms"),
    "lookalike": ("lookalikes.yaml", "lookalikes"),
    "effect": ("drug_effects.yaml", "drugs"),
    # The section must be the one the consumer reads, or the injector is registered but never called.
    "artifact": ("artifact_rates.yaml", "post_injectors"),
}


class WorldPluginError(ValueError):
    """A plugin registration is invalid. Raised at load time."""


_OVERLAY: dict[str, dict[str, dict]] = {}
#: (table, section, key) -> the plugin it came from.
_SOURCES: dict[tuple[str, str, str], str] = {}
_POST_INJECTORS: dict[str, tuple[Callable, str]] = {}
_LOADED: dict[str, int] = {}


def reset() -> None:
    """Clear all registrations. For tests only."""
    _OVERLAY.clear()
    _SOURCES.clear()
    _POST_INJECTORS.clear()
    _LOADED.clear()


# ---------------------------------------------------------------- the gate


def _repo_table(table: str) -> dict:
    """The in-tree table without the overlay, which collision checks compare against."""
    from .regpath import registry_path
    from .yamlcache import load_yaml
    p = registry_path(table)
    if not p.is_file():
        return {}
    return load_yaml(p) or {}


def _guard_label_bearing(kind: str, key: str, value: Any) -> None:
    """No label-bearing stream may be written, neither as a key nor as a value a spec points at."""
    if key in LABEL_BEARING:
        raise WorldPluginError(
            f"[{kind}] `{key}` carries the gold label (`label_rule` derives `outcome_label` from it); "
            f"world plugins cannot write it: relations validate, they do not set the gold")
    if isinstance(value, dict):
        hit = sorted(set(value) & set(LABEL_BEARING))
        if hit:
            raise WorldPluginError(
                f"[{kind}] `{key}` points at label-bearing streams {hit}; "
                f"plugins cannot set the gold")


def _register(kind: str, key: str, value: Any, *, source: str, replace: bool) -> None:
    if kind not in TARGETS:
        raise WorldPluginError(f"unknown registration kind {kind!r}; expected one of {sorted(TARGETS)}")
    if not isinstance(source, str) or not source.strip():
        raise WorldPluginError(
            f"[{kind}] `{key}` has no `source`; every registration must name its source")
    _guard_label_bearing(kind, key, value)

    table, section = TARGETS[kind]
    repo = (_repo_table(table).get(section) or {})
    if key in repo and not replace:
        raise WorldPluginError(
            f"[{kind}] `{key}` collides with the repository entry in `{table}:{section}`. "
            f"Pass `replace=True` to override it (overrides are recorded in the manifest)")
    slot = _OVERLAY.setdefault(table, {}).setdefault(section, {})
    if key in slot:
        raise WorldPluginError(
            f"[{kind}] `{key}` is registered by two plugins"
            f" ({_SOURCES.get((table, section, key))} and {source}); the result would be ambiguous, rejected")
    # A private copy: a plugin editing its spec afterwards cannot change the world behind the
    # manifest's back, and cached consumers keyed on `overlay_token` never share it.
    slot[key] = copy.deepcopy(value)
    _SOURCES[(table, section, key)] = source


# ------------------------------------------------- the four registration points


def register_stream(name: str, spec: dict, *, source: str, replace: bool = False,
                    kind: str = "stream") -> None:
    """Devices, indicator streams and physiological kernels. `kind` is one of
    `stream`, `kernel`, `indicator`.
    """
    if kind not in ("stream", "kernel", "indicator"):
        raise WorldPluginError(f"register_stream kind must be stream/kernel/indicator, got {kind!r}")
    _register(kind, name, spec, source=source, replace=replace)


def register_event_pool(name: str, spec: dict, *, source: str, replace: bool = False,
                        kind: str = "benign_event") -> None:
    """An event pool. `kind` is one of `benign_event`, `life_event`, `symptom`,
    `lookalike`.
    """
    if kind not in ("benign_event", "life_event", "symptom", "lookalike"):
        raise WorldPluginError(
            f"register_event_pool kind must be benign_event/life_event/symptom/lookalike, got {kind!r}")
    _register(kind, name, spec, source=source, replace=replace)


def register_effect(drug: str, spec: dict, *, source: str, replace: bool = False) -> None:
    """A drug-effect entry; it must pass the same field checks as the in-repo table."""
    _guard_label_bearing("effect", drug, spec)
    from .drug_schema import DrugEffectsError, check_entry
    try:
        check_entry(drug, spec)
    except DrugEffectsError as e:
        raise WorldPluginError(f"[effect] `{drug}` from {source}: {e}") from None
    _register("effect", drug, spec, source=source, replace=replace)


def register_post_injector(name: str, fn: Callable, *, source: str,
                           params: dict | None = None, replace: bool = False) -> None:
    """A dirty-data injector that runs after the kernel.

    `fn(points, *, rng_key, params) -> points` transforms the points of one stream. The kernel's
    own injector cannot take new categories, so new ones run here.
    """
    if not callable(fn):
        raise WorldPluginError(f"[post_injector] `{name}`: fn is not callable")
    if name in _POST_INJECTORS and not replace:
        raise WorldPluginError(
            f"[post_injector] `{name}` is already registered by {_POST_INJECTORS[name][1]}; "
            f"pass `replace=True` to override it")
    _register("artifact", name, dict(params or {}), source=source, replace=replace)
    _POST_INJECTORS[name] = (fn, source)


# ---------------------------------------------------------------- consumer side


def apply_overlay(table: str, doc: dict) -> dict:
    """Merge plugin-registered entries into the in-tree table. With no overlay it returns `doc`
    itself.

    A mapping section takes the entries by key. A list section (`benign_events.yaml`'s event
    pools) takes the registered specs appended after the in-tree items.
    """
    ov = _OVERLAY.get(table)
    if not ov:
        return doc
    if not isinstance(doc, dict):
        return doc
    out = dict(doc)
    for section, items in ov.items():
        base = out.get(section)
        if isinstance(base, list):
            out[section] = [*base, *items.values()]
        elif isinstance(base, dict):
            out[section] = {**base, **items}
        else:
            out[section] = dict(items)
    return out


def overlay_token(table: str) -> tuple:
    """Identity of the overlay entries for `table`, for consumers that cache the merged table.

    A cache entry keyed on it holds the very entries it merged, so their ids cannot be reused
    while the entry lives. Registration, `reset()` and a restore that swaps the tables all change
    it; an in-place edit of an already registered spec does not. `manifest_sha()` cannot serve:
    it records keys and sources, not values.
    """
    return tuple((section, key, id(value))
                 for section, items in (_OVERLAY.get(table) or {}).items()
                 for key, value in items.items())


def overlay_values(table: str) -> tuple:
    """The overlay value objects of `table`, in the order `overlay_token` lists them."""
    return tuple(value for items in (_OVERLAY.get(table) or {}).values() for value in items.values())


_CACHES: list = []


def overlay_cached(*tables: str, key: Callable[[], Any] | None = None):
    """Single-slot cache for a zero-argument builder that reads registry tables.

    The slot is keyed on the overlay tokens of `tables` (and on `key()` when given, e.g. the
    registry directory), so a registration, `reset()` or a restore that swaps the overlay
    tables rebuilds it. The slot also holds the overlay value objects it was built from, which
    keeps their ids from being reused while it lives (the reason `overlay_token` can use `id`).
    `cache_clear()` empties the slot; `clear_overlay_caches()` empties all of them.
    """
    def deco(fn):
        lock = threading.RLock()
        slot: list = [None]

        def _key():
            return (tuple(overlay_token(t) for t in tables), key() if key else None)

        @functools.wraps(fn)
        def wrapper():
            k = _key()
            cur = slot[0]
            if cur is not None and cur[0] == k:
                return cur[1]
            with lock:
                cur = slot[0]
                if cur is not None and cur[0] == k:
                    return cur[1]
                hold = tuple(overlay_values(t) for t in tables)
                value = fn()
                slot[0] = (k, value, hold)
                return value

        def cache_clear():
            with lock:
                slot[0] = None

        wrapper.cache_clear = cache_clear
        _CACHES.append(cache_clear)
        return wrapper
    return deco


def clear_overlay_caches() -> None:
    """Empty every `overlay_cached` slot (for `regpath.registry_dir_override`)."""
    for c in _CACHES:
        c()


def source_of(table: str, section: str, key: str) -> str | None:
    """The plugin that registered this entry, or `None` for an in-repo entry."""
    return _SOURCES.get((table, section, key))


def post_injectors() -> dict[str, tuple[Callable, str]]:
    return dict(_POST_INJECTORS)


def _canon(v: Any) -> Any:
    """JSON-ready form with string keys, so specs keyed by numbers (dose ladders) hash stably."""
    if isinstance(v, dict):
        return {k if isinstance(k, str) else repr(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v


def _value_sha(v: Any) -> str:
    blob = json.dumps(_canon(v), ensure_ascii=False, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def manifest() -> dict:
    """Registered entries with their source and a hash of their value: the same key from the same
    source with a different value is a different world."""
    entries = [{"table": t, "section": s, "key": k, "source": src,
                "value_sha": _value_sha(((_OVERLAY.get(t) or {}).get(s) or {}).get(k))}
               for (t, s, k), src in sorted(_SOURCES.items())]
    return {
        "groups": dict(sorted(_LOADED.items())),
        "n_entries": len(_SOURCES),
        "entries": entries,
        "post_injectors": {n: src for n, (_, src) in sorted(_POST_INJECTORS.items())},
    }


def manifest_sha() -> str:
    """Fingerprint of the manifest, or `""` when no plugin is attached so existing world stamps
    are unchanged.
    """
    if not _SOURCES and not _LOADED:
        return ""
    blob = json.dumps(manifest(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_world_plugins(group: str) -> int:
    """Load world-side plugins from an entry-point group; returns the number of entries attached.

    Each entry point is a `() -> None` callable that calls the registration functions. A load
    failure raises rather than being swallowed.
    """
    from importlib.metadata import entry_points
    before = len(_SOURCES)
    n_ep = 0
    for ep in entry_points(group=group):
        ep.load()()                  # let a broken plugin raise; see the docstring
        n_ep += 1
    added = len(_SOURCES) - before
    _LOADED[group] = _LOADED.get(group, 0) + added
    if n_ep and not added:
        log.warning("[world_plugins] group %s has %d entry point(s), but none registered anything -- "
                    "does the factory call the registration hook?", group, n_ep)
    return added
