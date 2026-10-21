"""Record repository reads in builders, including forked and Python subprocesses."""
from contextlib import contextmanager
import functools
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
_local = threading.local()
_installed = False
_log = os.environ.get("HAENV_TEST_AUDIT_LOG")
_seen = set()

_call_log = os.environ.get("HAENV_TEST_CALL_LOG")
_call_codes = {}
_call_sources = set()


def _call_hook(frame, event, arg):
    if event != "call" or not _call_log:
        return
    code = frame.f_code
    identity = id(code)
    if identity in _call_codes:
        return
    _call_codes[identity] = code
    label = frame.f_globals.get("__file__", code.co_filename)
    if not isinstance(label, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(label)).absolute()
    if not path.is_relative_to(ROOT):
        return
    relative = path.relative_to(ROOT)
    if not relative.parts or relative.parts[0] not in {"src", "tools", "plugins"}:
        return
    rel = relative.as_posix()
    if rel in _call_sources:
        return
    _call_sources.add(rel)
    fd = os.open(_call_log, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps([rel, os.getpid()]) + "\n").encode())
    finally:
        os.close(fd)


@contextmanager
def calling(path):
    """Track executed source, including modules inherited through fork/import caches."""
    global _call_log
    if _call_log:
        yield Path(_call_log)
        return
    previous = sys.getprofile()
    _call_log = str(path)
    _call_codes.clear()
    _call_sources.clear()
    Path(path).touch()
    sys.setprofile(_call_hook)
    try:
        yield Path(path)
    finally:
        sys.setprofile(previous)
        _call_log = None


def called_sources(path):
    with paused():
        return sorted({json.loads(line)[0] for line in Path(path).read_text().splitlines()})


def _hook(event, args):
    if event != "open" or not _log or getattr(_local, "paused", False):
        return
    path, mode, flags = args
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    _local.paused = True
    try:
        path = Path(os.fsdecode(path)).absolute()
        if not path.is_relative_to(ROOT) or str(path) == _log:
            return
        identity = (str(path), flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR), os.getpid(), _log)
        if identity in _seen:
            return
        _seen.add(identity)
        row = [str(path.relative_to(ROOT)), flags, os.getpid()]
        fd = os.open(_log, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(row) + "\n").encode())
        finally:
            os.close(fd)
    finally:
        _local.paused = False


@contextmanager
def paused():
    previous = getattr(_local, "paused", False)
    _local.paused = True
    try:
        yield
    finally:
        _local.paused = previous


def install():
    global _installed
    if _installed:
        return
    _installed = True
    sys.addaudithook(_hook)
    original = subprocess.Popen.__init__

    @functools.wraps(original)
    def spawn(self, *args, **kwargs):
        # Builders often supply a complete PYTHONPATH. Keep instrumentation even
        # for those commands; do not rely on their inherited environment.
        if _log:
            env = dict(kwargs.get("env") or os.environ)
            env["HAENV_TEST_AUDIT_LOG"] = _log
            if _call_log:
                env["HAENV_TEST_CALL_LOG"] = _call_log
            paths = [str(ROOT / "tests/_audit_bootstrap"), str(ROOT / "tests")]
            env["PYTHONPATH"] = os.pathsep.join([*paths, env.get("PYTHONPATH", "")])
            kwargs["env"] = env
        return original(self, *args, **kwargs)

    subprocess.Popen.__init__ = spawn


@contextmanager
def capture(path):
    global _log
    install()
    if _log:
        yield Path(_log)
        return
    _log = str(path)
    _seen.clear()
    Path(_log).touch()
    try:
        yield Path(_log)
    finally:
        _log = None


def reads(log):
    """Preserve every event; map bytecode reads to the source that produced it."""
    with paused():
        events = [json.loads(line) for line in Path(log).read_text().splitlines()]
    paths = set()
    for rel, flags, _pid in events:
        if flags & os.O_WRONLY and not flags & os.O_RDWR:
            continue
        if rel.endswith((".pyc", ".pyo")):
            try:
                rel = str(Path(importlib.util.source_from_cache(str(ROOT / rel))).relative_to(ROOT))
            except ValueError:
                pass
        paths.add(rel)
    return sorted(paths), events


if _log:
    install()

if _call_log:
    sys.setprofile(_call_hook)
