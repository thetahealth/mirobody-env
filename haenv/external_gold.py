"""Registration surface for gold produced outside this package.

``register_gold_block(name, fn, *, latent_keys, source, why, probe_fn=None, latent_classes=None)``:
``fn`` fills ``adjudication.<name>`` (verifier only) and ``probe_fn`` fills
``prediction_context.<name>`` (visible to the solver) with a dict or a string; either side
may return ``None`` for a case, and then writes no key. ``register_gate`` adds emission
gates for an external task type's own cases. With nothing registered the surface is a
no-op, and ``manifest_sha`` is ``""`` so the world stamp of other jobs does not move.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import typing
from typing import Callable
from typing import Literal

#: Built-in `adjudication` keys; an external block may not reuse them.
_RESERVED = frozenset({
    "adjudication_protocol_present", "primary_driver", "red_flag_present",
    "clinician_action_warranted", "ddx",
})

#: name -> (gold_fn, probe_fn, latent_keys, source, why). Both sides are registered in one call
#: so every item gets an explicit visibility decision; `probe_fn` output still passes `leakage_probe`.
BLOCKS: dict[str, tuple[Callable, Callable | None, tuple[str, ...], str, str]] = {}

#: Latent key -> provenance class, declared by the registering plugin (`provenance.class_of`).
LATENT_CLASSES: dict[str, str] = {}

#: name -> (fn, source, why). Emission gates of external task types, run after the built-in chain.
GATES: dict[str, tuple[Callable, str, str]] = {}

#: Severities a gate hit may carry; only `gate` blocks emission.
_SEVERITIES = frozenset({"gate", "warn", "info"})

#: External question templates, merged by `evaluate._framings()`.
FRAMINGS: dict[str, tuple[str, str, str]] = {}      # name -> (template, source, why)


def register_framing(name: str, template: str, *, source: str, why: str) -> None:
    """Register an external question template. Duplicate or built-in names are rejected; the
    probe yaml must carry its `framing_sha256` like a built-in template.
    """
    if name in _BUILTIN_FRAMING_NAMES:
        raise ValueError(f"{name!r} is a built-in question-framing template -- may not be replaced at runtime.")
    if name in FRAMINGS:
        raise ValueError(f"{name!r} is already registered (source {FRAMINGS[name][1]}). Duplicate name rejected.")
    if not template or not source or not why:
        raise ValueError("register_framing: template / source / why are all required")
    FRAMINGS[name] = (template, source, why)


#: External probes, merged by `evaluate.load_probes`. They must not be dropped into `probes/`:
#: `load_probes` validates the whole directory, so a probe whose template is not registered in a
#: job would break every job.
PROBES: dict[str, tuple[dict, str]] = {}      # probe_id -> (probe_dict, source)


def register_probe(probe: dict, *, source: str) -> None:
    """Register an external probe. A missing `probe_id` or a duplicate name is rejected."""
    pid = (probe or {}).get("probe_id")
    if not pid:
        raise ValueError("register_probe: probe is missing `probe_id`")
    if pid in PROBES:
        raise ValueError(f"probe_id {pid!r} is already registered (source {PROBES[pid][1]}). "
                         f"Duplicate name rejected.")
    if not source:
        raise ValueError("register_probe: `source` is required")
    PROBES[pid] = (dict(probe), source)


def external_probes() -> dict[str, dict]:
    return {pid: dict(d) for pid, (d, _) in PROBES.items()}


def external_framings() -> dict[str, str]:
    return {n: tpl for n, (tpl, _, _) in FRAMINGS.items()}


#: External latents live in their own spec slot, never flattened into the built-in fields,
#: so it stays clear which keys are haenv's and which a plugin's.
SLOT = "_external"


def register_gold_block(name: str, fn: Callable, *, latent_keys: tuple[str, ...],
                        source: str, why: str, probe_fn: Callable | None = None,
                        latent_classes: dict[str, str] | None = None) -> None:
    """Register one external gold block, optionally with its question side.

    Duplicate names, built-in keys and a missing `source`/`why` are rejected. Without `probe_fn`
    the block has an answer but no question. `latent_classes` must give every latent key a
    provenance class (`patient_fact`, `gold` or `knob`). Everything is validated before
    anything is written.
    """
    if not name or not isinstance(name, str):
        raise ValueError("register_gold_block: name must be a non-empty string")
    if name in _RESERVED:
        raise ValueError(
            f"{name!r} collides with a built-in `adjudication` key -- not allowed. "
            f"A collision silently rewrites the gold, and the artifact shows only that a score moved. "
            f"Built-in keys: {sorted(_RESERVED)}")
    if name in BLOCKS:
        raise ValueError(f"{name!r} is already registered (source {BLOCKS[name][3]}). "
                         f"Duplicate name rejected -- otherwise \"which block runs\" would depend on "
                         f"registration order.")
    if not callable(fn):
        raise TypeError("register_gold_block: fn must be callable")
    if not source or not why:
        raise ValueError("register_gold_block: `source` and `why` are both required -- "
                         "gold with no traceable origin can't have its score audited")
    keys = tuple(latent_keys or ())
    classes = dict(latent_classes or {})
    valid = set(typing.get_args(FieldClass))
    missing = [k for k in keys if k not in classes]
    extra = sorted(set(classes) - set(keys))
    bad = {k: c for k, c in classes.items() if c not in valid}
    if missing or extra or bad:
        raise ValueError(
            f"register_gold_block({name!r}): latent_classes must give each latent key "
            f"exactly one class out of {sorted(valid)}; missing {missing}, "
            f"not a latent key {extra}, invalid {bad}")

    # Registering the latent keys here keeps block and keys from being half-registered.
    from .job_schema import LATENT_REGISTRY
    taken = [k for k in keys if k in LATENT_REGISTRY]
    if taken:
        raise ValueError(f"latent keys {taken} are already in LATENT_REGISTRY; "
                         f"an external block may not reuse them")
    for k in keys:
        LATENT_REGISTRY[k] = (f"external_gold[{name}] -> adjudication.{name}",
                              f"registered by {source}: {why}")
    LATENT_CLASSES.update({k: classes[k] for k in keys})
    BLOCKS[name] = (fn, probe_fn, keys, source, why)


def latent_classes() -> dict[str, str]:
    """Provenance classes declared for external latent keys (`provenance.class_of`)."""
    return dict(LATENT_CLASSES)


def register_gate(name: str, fn: Callable, *, source: str, why: str) -> None:
    """Register an emission gate for an external task type. Fail-closed: a
    duplicate name, a non-callable, or a missing source or rationale is rejected.

    `fn(raw, cs, sp, T) -> list[dict]` sees a case the way the built-in gates do:
    the assembled case `raw` (gold included), the case spec `cs`, the solver
    payload `sp` and the prediction time `T`. A hit with severity `gate` blocks
    emission; `warn` and `info` hits are logged.
    """
    if not name or not isinstance(name, str):
        raise ValueError("register_gate: name must be a non-empty string")
    if name in GATES:
        raise ValueError(f"register_gate: {name!r} is already registered "
                         f"(by {GATES[name][1]}); duplicates are rejected")
    if not callable(fn):
        raise TypeError("register_gate: fn must be callable")
    if not source or not why:
        raise ValueError("register_gate: `source` and `why` are both required")
    GATES[name] = (fn, source, why)


def gates_for(raw, cs, sp, T: int) -> list[dict]:
    """Run the external emission gates on one case; `[]` when none is registered,
    so the built-in gate chain is unchanged.

    A hit must carry `kind`, `severity` and `detail` like a built-in one, and its
    `kind` must start with `<gate name>_` so that a blocked case in the batch
    record points at the plugin that blocked it. A gate that raises is an
    incident, not a pass: the exception propagates.
    """
    if not GATES:
        return []
    out: list[dict] = []
    for name, (fn, source, _) in GATES.items():
        try:
            hits = fn(raw, cs, sp, T)
        except Exception as e:                      # noqa: BLE001
            raise RuntimeError(f"external gate {name!r} (from {source}) raised: {e!r}") from e
        if not isinstance(hits, list):
            raise TypeError(f"external gate {name!r} must return a list, got {type(hits).__name__}")
        for h in hits:
            if not isinstance(h, dict) or not {"kind", "severity", "detail"} <= set(h):
                raise TypeError(f"external gate {name!r}: a hit needs kind/severity/detail, got {h!r}")
            if not str(h["kind"]).startswith(f"{name}_"):
                raise ValueError(f"external gate {name!r}: hit kind {h['kind']!r} "
                                 f"must start with {name + '_'!r}")
            if h["severity"] not in _SEVERITIES:
                raise ValueError(f"external gate {name!r}: severity {h['severity']!r} "
                                 f"is not one of {sorted(_SEVERITIES)}")
            out.append(h)
    return out


def passthrough_keys() -> tuple[str, ...]:
    out: list[str] = []
    for _, _, keys, _, _ in BLOCKS.values():
        out.extend(keys)
    return tuple(out)


def blocks_for(spec: dict) -> dict:
    """Materialise all external blocks; `{}` when empty. A block returning `None` writes no key."""
    if not BLOCKS:
        return {}
    out: dict = {}
    for name, (fn, _, _, source, _) in BLOCKS.items():
        try:
            got = fn(spec)
        except Exception as e:                      # noqa: BLE001
            raise RuntimeError(
                f"external_gold[{name}] (source {source}) raised while building its block: {e!r} -- "
                f"not swallowed: gold that fails to build is an incident; silently skipping it "
                f"would make this dimension look like \"not applicable\" in the report") from e
        if got is None:
            continue
        if not isinstance(got, dict):
            raise TypeError(f"external_gold[{name}] must return dict or None, got {type(got).__name__}")
        out[name] = got
    return out


def probe_blocks_for(spec: dict) -> dict:
    if not BLOCKS:
        return {}
    out: dict = {}
    for name, (_, pfn, _, source, _) in BLOCKS.items():
        if pfn is None:
            continue
        try:
            got = pfn(spec)
        except Exception as e:                      # noqa: BLE001
            raise RuntimeError(
                f"external_gold[{name}]'s question side (source {source}) raised: {e!r} -- "
                f"not swallowed: a question that fails to build is an incident") from e
        if got is None:
            continue
        if not isinstance(got, (dict, str)):
            raise TypeError(f"external_gold[{name}].probe_fn must return dict, str or None")
        out[name] = got
    return out


#: (root, file stats) -> fingerprint; see `_code_fingerprint`.
_FP_CACHE: dict[tuple, str] = {}


def _code_root(fn: Callable) -> pathlib.Path | None:
    """Where the code behind `fn` lives: the directory of its top-level package,
    or its module file when it is not in a package. Code inside `haenv` is
    already covered by the generation stamp, so only its own file is used.
    """
    target = getattr(fn, "func", fn)                # functools.partial
    modname = getattr(target, "__module__", None) or ""
    mod = sys.modules.get(modname)
    if mod is None or not getattr(mod, "__file__", None):
        return None
    path = pathlib.Path(mod.__file__).resolve()
    top = modname.split(".")[0]
    top_mod = sys.modules.get(top)
    if top == "haenv" or top_mod is None or not getattr(top_mod, "__file__", None):
        return path
    top_path = pathlib.Path(top_mod.__file__).resolve()
    return top_path.parent if top_path.name == "__init__.py" else top_path


def _code_fingerprint(fn: Callable) -> str:
    """Semantic hash of every file under `_code_root(fn)` (data files included),
    so editing the plugin's code or its tables moves the world stamp while
    editing a comment or docstring does not (same rule as `anchor.semantic_bytes`).
    """
    root = _code_root(fn)
    if root is None:
        target = getattr(fn, "func", fn)
        return f"unlocated:{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', '?')}"
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        and not any(part.startswith(".") for part in p.relative_to(root).parts))
    key = (str(root), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files))
    if key not in _FP_CACHE:
        from .semantic_source import semantic_bytes
        h = hashlib.sha256()
        for p in files:
            rel = p.name if root.is_file() else str(p.relative_to(root))
            h.update(rel.encode("utf-8") + b"\0" + semantic_bytes(p) + b"\0")
        _FP_CACHE[key] = h.hexdigest()[:16]
    return _FP_CACHE[key]


def manifest() -> dict:
    """What external task types have attached: gold blocks, gates, question
    templates and probes, each with who registered it and a fingerprint of the
    code behind it. It goes into the artefacts so that "which external pieces
    shaped this batch" stays checkable.
    """
    return {
        "blocks": [{"name": n, "latent_keys": list(k),
                    "latent_classes": {x: LATENT_CLASSES.get(x) for x in k},
                    "source": s, "why": w, "has_probe_side": pf is not None,
                    "code": sorted({_code_fingerprint(fn)}
                                   | ({_code_fingerprint(pf)} if pf else set()))}
                   for n, (fn, pf, k, s, w) in sorted(BLOCKS.items())],
        "gates": [{"name": n, "source": s, "why": w, "code": _code_fingerprint(fn)}
                  for n, (fn, s, w) in sorted(GATES.items())],
        "framings": {n: hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
                     for n, (t, _, _) in sorted(FRAMINGS.items())},
        "probes": {pid: hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                                  default=str).encode("utf-8")).hexdigest()[:16]
                   for pid, (d, _) in sorted(PROBES.items())},
    }


def manifest_sha() -> str:
    """The part of `manifest()` that shapes the world, hashed; `""` when nothing is
    registered, so the world stamp of a job without external task types stays
    byte-identical. `source` and `why` are provenance text and stay out of the
    hash, like comments stay out of the semantic fingerprint.
    """
    if not (BLOCKS or GATES or FRAMINGS or PROBES):
        return ""
    m = manifest()
    shaping = {
        "blocks": [{k: b[k] for k in ("name", "latent_keys", "has_probe_side", "code")}
                   for b in m["blocks"]],
        "gates": [{k: g[k] for k in ("name", "code")} for g in m["gates"]],
        "framings": m["framings"],
        "probes": m["probes"],
    }
    blob = json.dumps(shaping, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: Field class; see the table in the module docstring.
FieldClass = Literal["patient_fact", "gold", "knob"]


_BUILTIN_FRAMING_NAMES = frozenset({
    "PROMPT", "DDX_PROMPT", "DDX_SCOPE_PROMPT", "DDX_SCOPE2_PROMPT", "DDX_TRACE_PROMPT"})
