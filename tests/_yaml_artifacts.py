"""Reuse native safe-YAML parses without sharing mutable values or hiding probes."""
from collections import OrderedDict
import fcntl
import hashlib
import os
from pathlib import Path
import pickle
import sys

import pytest
import yaml

_native_load = yaml.load
_safe_loader = yaml.SafeLoader
_cache = OrderedDict()
_shared = None


def _frozen(value):
    if isinstance(value, dict):
        return tuple((key, _frozen(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(map(_frozen, value))
    if isinstance(value, set):
        return frozenset(map(_frozen, value))
    return value


def _loader_state():
    return tuple((base, _frozen(dict(vars(base)))) for base in _safe_loader.__mro__)


_original_state = _loader_state()


def load(stream, Loader):
    # Custom/modified loaders and the explicit parser opt-out must execute their
    # original callbacks. Streams also keep their native consumption semantics.
    if (Loader is not _safe_loader or not isinstance(stream, (str, bytes))
            or os.environ.get("HAENV_NO_YAML_CACHE") == "1" or _loader_state() != _original_state):
        return _native_load(stream, Loader)
    key = (type(stream), stream)
    if key in _cache:
        _cache.move_to_end(key)
        return pickle.loads(_cache[key])
    if _shared is not None and len(stream) >= 16384:
        encoded = stream.encode() if isinstance(stream, str) else stream
        digest = hashlib.sha256(type(stream).__name__.encode() + encoded).hexdigest()
        path = _shared / (digest + ".pickle")
        if not path.exists():
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not path.exists():
                    value = _native_load(stream, Loader)
                    pending = path.with_suffix(".tmp")
                    pending.write_bytes(pickle.dumps(value, protocol=5))
                    pending.replace(path)
        payload = path.read_bytes()
    else:
        payload = pickle.dumps(_native_load(stream, Loader), protocol=5)
    _cache[key] = payload
    if len(_cache) > 32:
        _cache.popitem(last=False)
    return pickle.loads(payload)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    yaml.load = load


def configure_shared(config):
    global _shared
    from _artifact_cache import cache_directory
    version = hashlib.sha256(Path(__file__).read_bytes() + sys.version.encode()
                             + (Path(__file__).parents[1] / "uv.lock").read_bytes()).hexdigest()
    _shared = cache_directory(config, "yaml-parses") / version
    _shared.mkdir(exist_ok=True)
    os.environ["HAENV_TEST_YAML_CACHE"] = str(_shared)
    root = Path(__file__).resolve().parent
    paths = [str(root / "_audit_bootstrap"), str(root)]
    os.environ["PYTHONPATH"] = os.pathsep.join([
        *paths, *(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p not in paths)])

def _prepare_file(path):
    load(Path(path).read_text(), _safe_loader)


def prebuild(config):
    """Parse large immutable inputs before workers, using the native safe loader."""
    from concurrent.futures import ProcessPoolExecutor
    root = Path(__file__).parents[1]
    paths = [str(path) for directory in ("registry", "benchmarks/inputs", "benchmarks/derived",
                                        "fixtures", "examples")
             for path in (root / directory).rglob("*")
             if path.is_file() and path.suffix in {".yaml", ".yml"}
             and path.stat().st_size >= 16384]
    with ProcessPoolExecutor(max_workers=64) as pool:
        list(pool.map(_prepare_file, sorted(paths)))
