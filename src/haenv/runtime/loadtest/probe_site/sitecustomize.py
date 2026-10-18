"""Timing probes for a rehearsal (`haenv loadtest`), loaded through `PYTHONPATH`.

`HAENV_LOADTEST_PROBES` is a JSON object `{probe name: ["module:Qualified.name", ...]}`; a
target ending in `@N` also records positional argument N as the event's label;
`HAENV_LOADTEST_PROBE_LOG` is the directory for the events; `HAENV_LOADTEST_PROBE_TAGS`
(JSON `{tag: module}`) tags a call made with that module on the stack. Each named function is
wrapped once its module is imported: a call records its duration, and a call that returns a
context manager (a slot) records the time its `__enter__` took. One JSONL file per process,
`{"p": probe, "f": function, "t": start, "dt": seconds, "cpu": thread CPU seconds, "tags": [...],
"label": ...}` per call, flushed every second. `dt - cpu` is time the thread spent waiting (locks, I/O, the
interpreter lock).
Standalone on purpose: it runs at interpreter start, before any haenv import.
"""
import importlib.abc
import json
import os
import sys
import threading
import time

_SPEC = json.loads(os.environ.get("HAENV_LOADTEST_PROBES") or "{}")
_DIR = os.environ.get("HAENV_LOADTEST_PROBE_LOG")
_TAGS = json.loads(os.environ.get("HAENV_LOADTEST_PROBE_TAGS") or "{}")
#: `HAENV_LOADTEST_TODAY` (ISO date): the day tariff checks see (`native_accounting.utc_today`).
_TODAY = os.environ.get("HAENV_LOADTEST_TODAY")


def _tags():
    found, frame = [], sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__")
        found += [tag for tag, name in _TAGS.items() if name == module and tag not in found]
        frame = frame.f_back
    return found


class _Log:
    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.file = open(os.path.join(directory, f"probe-{os.getpid()}.jsonl"), "a")
        self.lock, self.last = threading.Lock(), time.monotonic()

    def write(self, probe, name, start, dt, tags=(), label=None, cpu=None):
        line = json.dumps({"p": probe, "f": name, "t": round(start, 4), "dt": round(dt, 5),
                           "cpu": None if cpu is None else round(cpu, 5),
                           "tags": list(tags), "label": label})
        with self.lock:
            self.file.write(line + "\n")
            if time.monotonic() - self.last > 1.0:
                self.file.flush()
                self.last = time.monotonic()


def _wrap(fn, probe, name, log, label_arg=None):
    def timed(*args, **kwargs):
        start, cpu0, tags = time.time(), time.thread_time(), _tags() if _TAGS else ()
        label = str(args[label_arg]) if label_arg is not None and len(args) > label_arg else None
        result = fn(*args, **kwargs)
        if hasattr(result, "__enter__") and hasattr(result, "__exit__"):
            return _TimedContext(result, probe, name, log, start, tags, label)
        log.write(probe, name, start, time.time() - start, tags, label, time.thread_time() - cpu0)
        return result
    timed.__wrapped__, timed._loadtest_probe = fn, True
    timed.__name__ = getattr(fn, "__name__", name)
    return timed


class _TimedContext:
    def __init__(self, inner, probe, name, log, start, tags, label):
        self.inner, self.probe, self.name, self.log = inner, probe, name, log
        self.start, self.tags, self.label = start, tags, label

    def __enter__(self):
        value = self.inner.__enter__()
        self.log.write(self.probe, self.name, self.start, time.time() - self.start, self.tags,
                       self.label)
        return value

    def __exit__(self, *exc):
        return self.inner.__exit__(*exc)


def _patch(module, targets, log):
    for probe, target in targets:
        qualname, _, arg = target.partition("@")
        owner, parts = module, qualname.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
        fn = getattr(owner, parts[-1], None) if owner is not None else None
        if fn is not None and not getattr(fn, "_loadtest_probe", False):
            setattr(owner, parts[-1], _wrap(fn, probe, f"{module.__name__}:{qualname}", log,
                                            int(arg) if arg else None))


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, by_module, log):
        self.by_module, self.log = by_module, log

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.by_module:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = getattr(finder, "find_spec", lambda *a: None)(fullname, path, target)
            if spec is not None and spec.loader is not None:
                exec_module, targets, log = spec.loader.exec_module, self.by_module[fullname], self.log

                def exec_and_patch(module, _exec=exec_module):
                    _exec(module)
                    _patch(module, targets, log)
                    if _TODAY and module.__name__ == "haenv.runtime.native_accounting":
                        import datetime
                        module.utc_today = lambda day=datetime.date.fromisoformat(_TODAY): day
                spec.loader.exec_module = exec_and_patch
                return spec
        return None


if (_SPEC and _DIR) or _TODAY:
    _by_module = {"haenv.runtime.native_accounting": []} if _TODAY else {}
    for _probe, _paths in _SPEC.items():
        for _path in _paths:
            _module, _qual = _path.split(":", 1)
            _by_module.setdefault(_module, []).append((_probe, _qual))
    sys.meta_path.insert(0, _Finder(_by_module, _Log(_DIR) if _DIR else None))
