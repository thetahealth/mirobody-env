"""Run-local, fail-closed memoization of deterministic P3 case builds.

Only the build_case return value and its Q injection registrations are retained.
Rendering, persistence, batch gates and provenance always run again. No pickle is
read: the tagged JSON codec retains message descriptors, mapping order and tuples.
Unsupported inputs/results simply disable reuse. External plugins are excluded.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

VERSION = 1
# These select output locations or concurrency only. Every other environment
# value participates (hashed, never written to the cache as plaintext).
EXECUTION_ENV = frozenset({"HAENV_OUTPUT_ROOT", "HAENV_GEN_WORKERS", "HAENV_BUILD_CACHE", "_"})
# Shell, terminal and agent-session values that change between invocations without
# reaching the generator (connection ports, tool-call IDs, working directory). Keeping
# them in the key made every new shell miss the cache.
SESSION_ENV = frozenset({"PWD", "OLDPWD", "SHLVL", "TMPDIR", "TERM", "COLUMNS", "LINES", "LS_COLORS",
                         "MAIL", "STY", "WINDOW", "HISTFILE"})
SESSION_ENV_PREFIXES = ("SSH_", "CODEX_", "CLAUDE", "TMUX", "XDG_SESSION_", "TERM_")


def keyed_env(env=None):
    """The environment values that enter the key: everything except location, concurrency
    and session-only values."""
    env = os.environ if env is None else env
    return sorted((k, v) for k, v in env.items() if k not in EXECUTION_ENV and k not in SESSION_ENV
                  and not k.startswith(SESSION_ENV_PREFIXES))


def encode(value):
    from haenv.i18n.text import Msg
    if isinstance(value, Msg):
        return ["message", str(value), value.kind, value.key, encode(value.slots)]
    if type(value) in (type(None), bool, int, float, str):
        return ["scalar", value]
    if type(value) is dict:
        return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
    if type(value) in (list, tuple):
        return ["list" if type(value) is list else "tuple", [encode(v) for v in value]]
    if type(value) in (date, datetime):
        return [type(value).__name__, value.isoformat()]
    raise TypeError(f"unsupported cache value: {type(value).__name__}")


def decode(value):
    from haenv.i18n.text import Msg
    tag, *parts = value
    if tag == "scalar" and len(parts) == 1 and type(parts[0]) in (type(None), bool, int, float, str):
        return parts[0]
    if tag == "message" and len(parts) == 4:
        return Msg(*parts[:3], decode(parts[3]))
    if tag == "dict" and len(parts) == 1:
        pairs = [(decode(k), decode(v)) for k, v in parts[0]]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate cache key")
        return result
    if tag in ("tuple", "list") and len(parts) == 1:
        seq = [decode(v) for v in parts[0]]
        return tuple(seq) if tag == "tuple" else seq
    if tag in ("date", "datetime") and len(parts) == 1:
        return (date if tag == "date" else datetime).fromisoformat(parts[0])
    raise ValueError("invalid cache value")


def packed(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(packed(encode(value))).hexdigest()


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate cache envelope field")
    return result


def fingerprint_tree(root: Path) -> str:
    """Content and relative names, including additions/deletions; never mtime-only."""
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
    if not files:
        raise ValueError(f"empty cache source: {root}")
    return digest(files)


def context_for(job, cfg, max_rounds: int) -> dict:
    """Conservative supported boundary: builtin P3 in an unredirected checkout.

    All code/resources are hashed in addition to the world/judging stamps. We
    deliberately over-key config/env, rather than maintain an allowlist of knobs.
    The job's other cases and pool provenance never enter build_case; the exact
    serialized case and effective CaseSpec are bound separately in key_for.
    """
    from haenv.locations import repository_root, source_root, data_root
    from haenv.provenance.anchor import world_fingerprint, judging_fingerprint
    from haenv.world.generation import world_knobs
    from haenv.app.job import _LOADED_GROUPS
    root = repository_root()
    if root is None or data_root().resolve() != root.resolve():
        raise ValueError("cache requires unredirected repository resources")
    if job.plugins != ["haenv.p3"] or job.world_plugins:
        raise ValueError("cache supports only builtin P3 without world plugins")
    if _LOADED_GROUPS != {"haenv.p3"}:
        raise ValueError("cache requires an isolated builtin P3 job")
    # Editable haenv and plugins/ both declare this exact builtin callback.
    # Identical (name, import target) declarations load the same verified module;
    # any distinct target remains ambiguous and disables the cache.
    entries = {(e.name, e.value)
               for e in importlib.metadata.entry_points(group="haenv.p3")}
    if len(entries) != 1 or next(iter(entries))[1] != "haenv.p3.plugin:judges":
        raise ValueError("ambiguous P3 entry point")
    from haenv.p3 import plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source_root().resolve()):
        raise ValueError("external P3 plugin")
    # External resource redirects cannot be proved by the repository manifest.
    if os.environ.get("HAENV_KERNEL_PATH"):
        raise ValueError("external kernel comparison source")
    sources = {name: fingerprint_tree(root / name) for name in
               ("src", "registry", "tools", "plugins", "benchmarks/inputs", "benchmarks/probes")}
    # Bind the actually installed dependency implementations, including native code.
    for name in ("yaml", "pydantic", "pydantic_core", "numpy", "scipy"):
        if importlib.util.find_spec(name) is None:
            sources[name] = "absent"
        else:
            module = importlib.import_module(name)
            sources[name] = fingerprint_tree(Path(module.__file__).resolve().parent)
    sources["python"] = hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
    return {"version": VERSION, "sources": sources,
            "world": world_fingerprint(), "judging": judging_fingerprint(),
            "runtime": [sys.version, sys.executable, platform.platform(),
                        sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())],
            "environment": keyed_env(),
            "config": cfg, "language": job.language, "max_rounds": max_rounds,
            "job": {k: getattr(job, k) for k in ("job_id", "task_type", "plugins", "world_plugins",
                    "findings", "gated", "multiround", "cadence_days", "slices", "probe_id", "allow_retired")},
            "knobs": [world_knobs.PHYSIO_ENABLED, world_knobs.FINDINGS_ENABLED]}


class CaseCache:
    def __init__(self, directory: Path, context: dict, plans: dict):
        self.directory = Path(directory)
        self.context = digest(context)
        self.plans = plans
        self.hits = self.misses = 0

    def key_for(self, case) -> str | None:
        try:
            return digest([VERSION, self.context, self.plans[case.case_id], asdict(case)])
        except (KeyError, TypeError, ValueError):
            return None

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / (key + ".json")

    def read(self, key: str | None, case_id: str):
        if key is None:
            return None
        try:
            entry = json.loads(self.path_for(key).read_bytes(), object_pairs_hook=_unique_object)
            if (set(entry) != {"version", "key", "payload", "sha256"}
                    or type(entry["version"]) is not int or entry["version"] != VERSION or entry["key"] != key):
                return None
            payload = entry["payload"]
            if hashlib.sha256(packed(payload)).hexdigest() != entry["sha256"]:
                return None
            value = decode(payload)
            if set(value) != {"raw", "audit", "injections"} or value["audit"]["case_id"] != case_id:
                return None
            if encode(value) != payload:  # reject lossy/ambiguous decoding
                return None
            raw = value["raw"]
            if raw is not None:
                from haenv_kernel.schema import RawCase
                raw = RawCase(**raw)
                if raw.case_id != case_id:
                    return None
            inj = value["injections"]
            if not isinstance(inj, list) or any(not isinstance(k, str) or not isinstance(v, dict) for k, v in inj):
                return None
            return (raw, value["audit"]), None, inj
        except Exception:  # any ambiguity/corruption/I/O failure is a miss
            return None

    def write(self, key: str | None, result) -> None:
        ra, err, inj = result
        if key is None or err is not None:  # exceptions are retried, never cached
            return
        temp = None
        try:
            raw, audit = ra
            payload = encode({"raw": asdict(raw) if raw is not None else None,
                              "audit": audit, "injections": inj})
            entry = packed({"version": VERSION, "key": key, "payload": payload,
                            "sha256": hashlib.sha256(packed(payload)).hexdigest()})
            path = self.path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as f:
                temp = Path(f.name)
                f.write(entry)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)  # reader sees one complete envelope or a miss
        except Exception:  # cache failures must never change build behaviour
            pass
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass


def optional_cache(directory, job, cfg, max_rounds):
    if not directory:
        return None
    try:
        from haenv.resources.yamlcache import load_yaml
        plans = {c["case_id"]: c for c in load_yaml(Path(job.path))["cases"]}
        return CaseCache(Path(directory), context_for(job, cfg, max_rounds), plans)
    except Exception as exc:
        print(f"[haenv] build cache disabled: {type(exc).__name__}: {exc}")
        return None
