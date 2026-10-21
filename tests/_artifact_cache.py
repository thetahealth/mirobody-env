"""Content-addressed, cross-worker test artifacts. Readers receive fresh JSON values.

Only build inputs enter the key. Builders name additional fixture/script files
explicitly; editing tests or committing the same inputs never invalidates a pack.
"""
from __future__ import annotations

import fcntl
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
_BUILD_HELPERS = {"_artifact_cache.py", "_artifact_audit.py", "_component_artifacts.py",
                  "_build_cli.py", "_query_build.py", "_query_helpers.py", "_shared_build.py",
                  "_module_family.py", "_p3_samples.py", "_paths.py"}
HELPERS = tuple(str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("_*.py"))
                if p.name in _BUILD_HELPERS or p.name.endswith("_artifacts.py")) + (
    "tests/_audit_bootstrap/sitecustomize.py",)
DATA_INPUTS = ("registry", "benchmarks/inputs", "benchmarks/derived", "plugins", "config.yaml", "uv.lock")
DEPENDENCIES = ("src", "tools", *DATA_INPUTS, *HELPERS)
WORLD_RECIPES = frozenset({
    "world-equivalence", "p3-seed-pool", "p3-review-records", "p3-synthetic-ledger", "p3-background-hash",
    "query-batch", "m2-twin-source", "m2-twin-rebuild", "m2-line-trial",
    "component-twin-families", "m2-confirmation-record", "world-plugin-demo",
    "idempotency-source", "m2-patient-prompts", "m2-patient-medication",
})
_world_token = None
_session_digests = {}


def file_digest(path):
    if not path.is_file():
        return "<missing>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_digest(path):
    """Controller inputs are immutable during prebuild; unit-test files stay live."""
    return _session_digests[str(path)] if str(path) in _session_digests else file_digest(path)


def dependency_digest(root=ROOT, paths=DEPENDENCIES):
    """Include paths and bytes, so additions, removals and same-size edits invalidate."""
    files = set()
    for rel in paths:
        path = root / rel
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file()
                         and not {"__pycache__", ".pytest_cache", ".venv"} & set(p.parts)
                         and p.suffix not in {".pyc", ".pyo"})
        elif path.is_file():
            files.add(path)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def world_token():
    global _world_token
    if _world_token is None:
        from haenv.provenance import anchor, declarations
        segments = {name: anchor._fingerprint_of(paths) for name, paths in vars(declarations).items()
                    if name.endswith("_WORLD")}
        _world_token = hashlib.sha256(json.dumps([
            anchor.world_fingerprint(), segments,
            dependency_digest(paths=(*DATA_INPUTS, *HELPERS)), sys.version,
        ]).encode()).hexdigest()
    return _world_token


def inherited_inputs(world):
    """Include declared data and source already loaded before a recipe opens files.

    Registry values may be inherited through parser caches. Source actually used
    by a builder is recorded separately, including calls into preloaded modules.
    Test assertions are excluded; build helpers are declared separately.
    """
    prefixes = (*DATA_INPUTS, *HELPERS) if world else DEPENDENCIES
    paths = set()
    for label in _session_digests:
        path = Path(label)
        if path.is_relative_to(ROOT):
            rel = path.relative_to(ROOT).as_posix()
            if any(rel == item or rel.startswith(item.rstrip("/") + "/") for item in prefixes):
                paths.add(path)
    if not paths:  # Isolated cache-unit roots do not share the controller snapshot.
        for rel in prefixes:
            path = ROOT / rel
            if path.is_dir():
                paths.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            elif path.is_file():
                paths.add(path)
    return {path.relative_to(ROOT).as_posix(): snapshot_digest(path) for path in paths}


def world_sources():
    from haenv.provenance import declarations
    labels = set(declarations.GENERATION)
    for name, paths in vars(declarations).items():
        if name.endswith("_WORLD"):
            labels.update(paths)
    return sorted((Path("src") / label).as_posix()
                  if label.startswith(("haenv/", "haenv_kernel/", "verifier_core/")) else label
                  for label in labels)


def _cached_reads(task):
    metadata_path, cache_parent = task
    metadata = json.loads(Path(metadata_path).read_text())
    return [rel for rel in metadata.get("key", {}).get("read_inputs", {})
            if (ROOT / rel).is_relative_to(Path(cache_parent))]


def snapshot_cache_reads(cache_root):
    """Hash immutable published values once, rather than once per fixture lookup."""
    from concurrent.futures import ProcessPoolExecutor
    cache_root = Path(cache_root)
    indexes = list(cache_root.rglob("*.inputs.json")) + list((cache_root / "read-audits").glob("*.json"))
    shared = cache_root.parent / "yaml-parses"
    paths = {p for p in shared.rglob("*.pickle") if p.is_file()}
    with ProcessPoolExecutor(max_workers=64) as pool:
        for reads in pool.map(_cached_reads, ((str(p), str(cache_root.parent)) for p in indexes), chunksize=8):
            paths.update(ROOT / rel for rel in reads if (ROOT / rel).is_relative_to(cache_root.parent))
        paths = sorted(p for p in paths if p.is_file())
        _session_digests.update(zip(map(str, paths), pool.map(file_digest, paths, chunksize=8)))


def validate_key(metadata):
    """A published receipt may neither omit a read nor change its dependency set."""
    covered = set(metadata["key"]["read_inputs"]) | set(metadata["key"].get("world_sources", []))
    assert not metadata["derived_reads"], "all repository reads must appear explicitly in the key"
    missing = set(metadata["read_paths"]) - covered - set(metadata["derived_reads"])
    assert not missing, f"repository reads missing from artifact key: {sorted(missing)}"
    expected = hashlib.sha256(json.dumps(metadata["key"], sort_keys=True).encode()).hexdigest()
    assert expected == metadata["key_sha256"], "artifact dependency key was changed"


class ArtifactCache:
    def __init__(self, root, fingerprint, world=None):
        self.base = Path(root)
        self.fingerprint = fingerprint
        self.world = world
        self.root = self.base / fingerprint
        self.builds = 0
        self.entries = []
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, name, inputs, build, *, dependencies=()):
        from _artifact_audit import capture, paused, reads, calling, called_sources
        with paused():
            # Executable helpers belong to the common harness inputs. Recipe
            # declarations describe fixture data rather than import graphs.
            assert all(str(p).startswith("tests/") and not str(p).endswith(".py")
                       for p in dependencies), "recipe declarations must contain only test fixture data"
            token = self.world or world_token() if name in WORLD_RECIPES else self.fingerprint
            namespace = self.base / ("world-" + token) if name in WORLD_RECIPES else self.root
            namespace.mkdir(exist_ok=True)
            base = {"recipe": name, "inputs": inputs, "snapshot": token,
                    "declared_data": dependency_digest(paths=dependencies)}
            recipe = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
            index = namespace / f"{recipe}.inputs.json"
            self.entries.append(str(index))

        def lookup():
            with paused():
                if not index.is_file():
                    return None
                metadata = json.loads(index.read_text())
                validate_key(metadata)
                if any(snapshot_digest(ROOT / rel) != digest for rel, digest in metadata["key"]["read_inputs"].items()):
                    return None
                dest = namespace / metadata["directory"]
                return dest if (namespace / (metadata["key_sha256"] + ".ready")).is_file() else None

        dest = lookup()
        if dest is not None:
            return dest
        with (namespace / f"{recipe}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            dest = lookup()
            if dest is not None:
                return dest
            # Receipts embed absolute paths. Keep the build directory immutable;
            # the content key indexes it after the complete read set is known.
            dest = namespace / (recipe + "-" + uuid.uuid4().hex)
            dest.mkdir()
            log = namespace / (dest.name + ".opens.jsonl")
            try:
                with capture(log) as audit_log:
                    self.builds += 1
                    context = calling(log.with_suffix(".calls.jsonl")) if name in WORLD_RECIPES else nullcontext(None)
                    with context as call_log:
                        build(dest)
                    sources = called_sources(call_log) if call_log is not None else []
                    paths, events = reads(audit_log)
                with paused():
                    # Cache files are repository reads too. Hash them explicitly;
                    # a path under a cache directory is never an exemption.
                    derived = []
                    read_inputs = inherited_inputs(name in WORLD_RECIPES)
                    parse_root = self.base.parent / "yaml-parses"
                    read_inputs.update({Path(p).relative_to(ROOT).as_posix(): digest
                                        for p, digest in _session_digests.items()
                                        if Path(p).is_relative_to(parse_root) and Path(p).is_relative_to(ROOT)})
                    read_inputs.update({rel: snapshot_digest(ROOT / rel) for rel in (*paths, *sources)})
                    # Fixture files have explicit owners. Source reads are always
                    # part of the key, even when not in a predeclared segment.
                    fixture_reads = [rel for rel in read_inputs if rel.startswith("tests/") and not rel.endswith(".py")]
                    undeclared = [rel for rel in fixture_reads if not any(rel == str(d) or rel.startswith(str(d).rstrip("/")+"/") for d in dependencies)]
                    assert not undeclared, f"undeclared test data read: {undeclared}"
                    key = {**base, "read_inputs": read_inputs,
                           "world_sources": world_sources() if name in WORLD_RECIPES else []}
                    key_sha256 = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
                    metadata = {"key": key, "key_sha256": key_sha256, "directory": dest.name,
                                "cache_root": str(self.base.absolute()),
                                "read_paths": paths, "called_sources": sources, "derived_reads": derived, "open_events": events}
                    validate_key(metadata)
                    (dest / "read-receipt.json").write_text(json.dumps(metadata, sort_keys=True))
                    pending = index.with_suffix(".tmp")
                    pending.write_text(json.dumps(metadata, sort_keys=True))
                    pending.replace(index)
                    (namespace / (key_sha256 + ".ready")).touch()
            except BaseException:
                shutil.rmtree(dest, ignore_errors=True)
                log.unlink(missing_ok=True)
                log.with_suffix(".calls.jsonl").unlink(missing_ok=True)
                raise
        return dest

    def json(self, name, inputs, build, *, dependencies=()):
        def write(dest):
            (dest / "value.json").write_text(json.dumps(build(), ensure_ascii=False), encoding="utf-8")
        dest = self.directory(name, inputs, write, dependencies=dependencies)
        return json.loads((dest / "value.json").read_text(encoding="utf-8"))


def cache_directory(config, name):
    """Use persistent pytest storage or a controller-owned private session directory."""
    if hasattr(config, "cache"):
        return Path(config.cache.makedir(name))
    if not hasattr(config, "_artifact_temp_root"):
        from tempfile import TemporaryDirectory
        temporary = TemporaryDirectory(prefix="haenv-test-cache-")
        config._artifact_temp_root = Path(temporary.name)
        config.add_cleanup(temporary.cleanup)
    path = config._artifact_temp_root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_cache(config, tmp_path_factory):
    """The controller and workers always use the same storage, with or without pytest's cache."""
    return ArtifactCache(cache_directory(config, "component-artifacts"),
                         config._artifact_fingerprint, config._world_fingerprint)


def fingerprint():
    for rel in DEPENDENCIES:
        path = ROOT / rel
        files = path.rglob("*") if path.is_dir() else [path]
        for item in files:
            if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}:
                _session_digests[str(item)] = file_digest(item)
    return hashlib.sha256(json.dumps([dependency_digest(), sys.version]).encode()).hexdigest()


def pytest_configure(config):
    from _yaml_artifacts import pytest_configure as configure_yaml
    configure_yaml(config)
    global _world_token
    if hasattr(config, "workerinput"):
        config._artifact_fingerprint = config.workerinput["artifact_fingerprint"]
        config._world_fingerprint = config.workerinput["world_fingerprint"]
        _session_digests.update(config.workerinput["source_digests"])
        if config.workerinput.get("artifact_temp_root"):
            config._artifact_temp_root = Path(config.workerinput["artifact_temp_root"])
        config._artifact_prebuild = config.workerinput["artifact_prebuild"]
    else:
        config._artifact_fingerprint = fingerprint()
        config._world_fingerprint = world_token()
    _world_token = config._world_fingerprint


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    node.workerinput["artifact_fingerprint"] = node.config._artifact_fingerprint
    node.workerinput["world_fingerprint"] = node.config._world_fingerprint
    node.workerinput["source_digests"] = _session_digests
    node.workerinput["artifact_temp_root"] = str(node.config._artifact_temp_root) if hasattr(node.config, "_artifact_temp_root") else None
    node.workerinput["artifact_storage"] = str(cache_directory(node.config, "component-artifacts"))
    node.workerinput["artifact_prebuild"] = getattr(node.config, "_artifact_prebuild", {"jobs": []})


@pytest.fixture(scope="session")
def artifact_cache(pytestconfig, tmp_path_factory):
    return session_cache(pytestconfig, tmp_path_factory)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Finish shared builds before xdist launches its workers (also on a cold cache)."""
    config = session.config
    from _yaml_artifacts import configure_shared, prebuild as prebuild_yaml
    configure_shared(config)
    if hasattr(config, "workerinput") or config.option.collectonly:
        return
    prebuild_yaml(config)
    from concurrent.futures import ProcessPoolExecutor
    try:
        from _component_artifacts import recipes, prebuild
    except ModuleNotFoundError as error:
        if error.name != "_component_artifacts":
            raise
        return
    jobs = recipes(config.args, world=config.option.markexpr == "world")
    cache = session_cache(config, config._tmp_path_factory)
    snapshot_cache_reads(cache.base)
    if not jobs:
        config._artifact_prebuild = {"jobs": [], "builds": 0, "seconds": 0}
        if hasattr(config, "cache"):
            config.cache.set("component-artifacts/prebuild", config._artifact_prebuild)
        return
    tasks = [(str(cache.root.parent), config._artifact_fingerprint, config._world_fingerprint, *job) for job in jobs]
    import time
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=64) as pool:
        results = list(pool.map(prebuild, tasks))
    config._artifact_prebuild = {"seconds": time.monotonic() - started, "jobs": results,
                                "builds": sum(row["builds"] for row in results)}
    snapshot_cache_reads(cache.base)
    if hasattr(config, "cache"):
        config.cache.set("component-artifacts/prebuild", config._artifact_prebuild)


def pytest_terminal_summary(terminalreporter):
    facts = getattr(terminalreporter.config, "_artifact_prebuild", None)
    if facts:
        terminalreporter.write_line(f"component artifacts: {facts['builds']} built / {len(facts['jobs'])} recipes; "
                                    f"controller prebuild {facts['seconds']:.2f}s (included in wall time)")
