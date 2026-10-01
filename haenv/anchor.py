"""anchor.py — resolves the freeze anchor and computes the judging and world fingerprints.

The freeze snapshot is `docs/anchor/freeze-*.json`; there must be exactly one
(a new version edits it in place). `judging_sha16` covers `make_freeze.JUDGING`
and is stamped on every eval row; `world_sha` covers `make_freeze.GENERATION`
plus installed world plugins and is stamped on every batch. Both are computed
over semantic content, so comment and docstring edits do not move them.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import threading

from haenv import data_root as _data_root
ROOT = _data_root()
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
ANCHOR_DIR = ROOT / "docs" / "anchor"
#: Freeze snapshot filename pattern. The date is the file's creation date; the
#: version is its `revision` field.
GLOB = "freeze-*.json"


class AnchorError(RuntimeError):
    """Freeze anchor is unreadable, or has split into multiple copies.
    No fallback — this raises directly."""


def _candidates() -> list[pathlib.Path]:
    return sorted(p for p in ANCHOR_DIR.glob(GLOB) if p.is_file())


def split_anchors() -> list[pathlib.Path]:
    """All freeze snapshots if there is more than one (a split), else `[]`."""
    c = _candidates()
    return c if len(c) > 1 else []


def freeze_path() -> pathlib.Path:
    """Path to the current freeze snapshot; on a split, logs an error and picks the
    highest `revision`.
    """
    c = _candidates()
    if not c:
        raise AnchorError(f"no {GLOB} under {ANCHOR_DIR} -- freeze anchor is missing, refusing to continue")
    _sp = split_anchors()
    if not _sp:
        return c[0]
    _log = __import__("logging").getLogger("haenv.anchor")
    _log.error("the freeze anchor has split into %d copies: %s -- different tools will "
               "each read whichever one they happen to. Keep only one authoritative copy (a "
               "new version edits that same file directly). Picking the one with the highest "
               "revision for now; that does not resolve the split.",
               len(c), [x.name for x in c])

    def _rev(p: pathlib.Path) -> int:
        try:
            return int(json.loads(p.read_text(encoding="utf-8")).get("revision") or -1)
        except Exception:                                      # noqa: BLE001
            return -1
    return max(c, key=_rev)


def load_freeze() -> dict:
    p = freeze_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                     # noqa: BLE001
        raise AnchorError(f"{p} failed to parse: {type(e).__name__} {e}") from e


def frozen_packs() -> dict[str, str]:
    snap = load_freeze()
    return {job: str(v["batch"]) for job, v in (snap.get("packs") or {}).items()}


# ==========================================================================
# Judging-fingerprint — a board is only allowed one judging version
# ==========================================================================

#: Cache parsing by actual content: edits can preserve size and mtime, and even
#: ctime has limited resolution on some filesystems. File reads stay live.
#:
#: Semantic content: `.py` via AST with docstrings stripped, `.yaml` via
#: `safe_load` to JSON with key order preserved; a parse failure falls back to
#: raw bytes. Pack fingerprints (`cases.jsonl`, job files) stay byte-exact.
_SEM_CACHE: dict[tuple, bytes] = {}
_SOURCE_CACHE: dict[tuple, bytes] = {}
_SOURCE_FILE_STATES: dict[str, str] = {}


def _fingerprint_path(rel: str) -> pathlib.Path:
    """Resolve a canonical source label in the selected layout, without searching.

    A wheel puts code, kernel and resources in separate locations. The labels
    fed into the hash stay source-relative, so identical shipped content has
    identical stamps. An explicit alternate ROOT remains a source-layout root;
    missing files there must not be silently filled from the installed package.
    """
    if ROOT == _PACKAGE_ROOT / "_data":
        head, _, tail = rel.partition("/")
        bases = {"haenv": _PACKAGE_ROOT, "core": _PACKAGE_ROOT / "_kernel",
                 "verifier_core": _PACKAGE_ROOT.parent / "verifier_core"}
        if head in bases:
            return bases[head] / tail
    return ROOT / rel


def semantic_bytes(path: pathlib.Path) -> bytes:
    """A file's semantic content (comments/docstrings/formatting excluded).
    Falls back to the raw bytes if parsing fails."""
    raw = path.read_bytes()
    suf = path.suffix.lower()
    key = (suf, hashlib.sha256(raw).digest())
    hit = _SEM_CACHE.get(key)
    if hit is not None:
        return hit
    out = _semantic_of(raw, suf)
    if len(_SEM_CACHE) > 4096:                                   # simple cap, avoid unbounded growth
        _SEM_CACHE.clear()
    _SEM_CACHE[key] = out
    return out


def _semantic_cached(raw: bytes, suf: str) -> bytes:
    """`_semantic_of`, memoised on content like `semantic_bytes`."""
    key = (suf, hashlib.sha256(raw).digest())
    hit = _SEM_CACHE.get(key)
    if hit is None:
        hit = _semantic_of(raw, suf)
        if len(_SEM_CACHE) > 4096:
            _SEM_CACHE.clear()
        _SEM_CACHE[key] = hit
    return hit


def _semantic_of(raw: bytes, suf: str) -> bytes:
    try:
        if suf == ".py":
            import ast
            tree = ast.parse(raw.decode("utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
            return _unparse(ast.fix_missing_locations(tree)).encode("utf-8")
        if suf in (".yaml", ".yml"):
            import json

            import yaml
            # Key order is preserved: some registry tables are first-match-wins.
            return json.dumps(yaml.safe_load(raw.decode("utf-8")),
                              ensure_ascii=False, sort_keys=False,
                              default=str).encode("utf-8")
    except Exception:                                            # noqa: BLE001
        return raw                                               # fail-closed
    return raw


def _unparse(tree) -> str:
    """`ast.unparse` as Python 3.11 and later write it, on every supported version."""
    import ast
    if sys.version_info >= (3, 11):
        return ast.unparse(tree)
    from ._unparse310 import unparse
    return unparse(tree)


def _semantic_source(src: str) -> bytes:
    import textwrap
    return _semantic_of(textwrap.dedent(src).encode("utf-8"), ".py")


def _fingerprint_of(rels) -> str:
    """Semantic fingerprint over a set of files; a missing file hashes as `<missing>`."""
    import hashlib
    h = hashlib.sha256()
    for rel in rels:
        p = _fingerprint_path(rel)
        h.update(rel.encode("utf-8"))
        h.update(semantic_bytes(p) if p.is_file() else b"<missing>")
    return h.hexdigest()[:16]

#: Files on the judging path, read live from `tools/make_freeze.py:JUDGING`.
def judging_files() -> tuple[str, ...]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mf_for_judging", ROOT / "tools" / "make_freeze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                # noqa: S102
    return tuple(mod.JUDGING)


def infra_files() -> tuple[str, ...]:
    """`tools/make_freeze.py:INFRA`: recorded, never part of `judging_sha16`."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mf_for_infra", ROOT / "tools" / "make_freeze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                # noqa: S102
    return tuple(mod.INFRA)


class EquivalenceError(ValueError):
    """A `judging_sha_equivalence` row does not reproduce from the repository."""


def _git_blob(rev: str, rel: str) -> bytes | None:
    import re
    import subprocess
    immutable = re.fullmatch(r"[0-9a-f]{7,40}", rev) is not None
    if immutable and (rev, rel) in _BLOBS:
        return _BLOBS[rev, rel]
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
                       capture_output=True, timeout=60)
    blob = r.stdout if r.returncode == 0 else None
    if immutable:                      # a commit id names fixed content; a ref such as HEAD moves
        _BLOBS[rev, rel] = blob
    return blob


_BLOBS: dict = {}


def _segment_at(rev: str, name: str) -> tuple[str, ...]:
    src = _git_blob(rev, "tools/make_freeze.py")
    if src is None:
        raise EquivalenceError(f"revision {rev} has no tools/make_freeze.py")
    ns = {"__file__": str(ROOT / "tools" / "make_freeze.py"), "__name__": "_mf_at_rev"}
    exec(compile(src, f"make_freeze@{rev}", "exec"), ns)             # noqa: S102
    return tuple(ns[name])


def fingerprint_at(rev: str, rels) -> str:
    """`_fingerprint_of` over the files as committed at `rev` (git objects, not the tree)."""
    h = hashlib.sha256()
    for rel in rels:
        blob = _git_blob(rev, rel)
        h.update(rel.encode("utf-8"))
        h.update(_semantic_cached(blob, pathlib.Path(rel).suffix.lower()) if blob is not None
                 else b"<missing>")
    return h.hexdigest()[:16]


def verify_judging_equivalence(row: dict, *, infra=None) -> dict:
    """Recompute one `old judging_sha16 -> new` row; raise `EquivalenceError` if it fails.

    At `row["rev"]`: the old segment (that revision's own `JUDGING`) must hash to `old`;
    the same files minus the current `INFRA` must hash to `new`; and every file dropped
    must be an `INFRA` member (a judging file cannot be laundered out by relabelling it).
    """
    infra = set(infra_files() if infra is None else infra)
    rev = row["rev"]
    old_seg = _segment_at(rev, "JUDGING")
    if fingerprint_at(rev, old_seg) != row["old"]:
        raise EquivalenceError(f"{row['old']} does not reproduce at {rev}")
    kept = tuple(f for f in old_seg if f not in infra)
    dropped = sorted(set(old_seg) - set(kept))
    current = set(judging_files())
    stray = sorted(f for f in dropped if f in current or f not in infra)
    if stray:
        raise EquivalenceError(f"files dropped from judging are not INFRA: {stray}")
    got = fingerprint_at(rev, kept)
    if got != row["new"]:
        raise EquivalenceError(f"{row['old']}@{rev} maps to {got}, not {row['new']}")
    return {**row, "dropped_to_infra": dropped, "kept_n": len(kept)}


def judging_fingerprint() -> str:
    """Content fingerprint (16 hex) over every file on the judging path. Aggregation
    fails if one batch carries more than one value.
    """
    return _fingerprint_of(judging_files())


_JUDGING_FP: list[str] = []
_JUDGING_FP_LOCK = threading.Lock()


def judging_fp_cached() -> str:
    """In-process cache — the judging code can't change mid-run, so the
    fingerprint is not recomputed on every row. The lock keeps the first rows of a
    parallel run from each computing it."""
    if _JUDGING_FP:
        return _JUDGING_FP[0]
    with _JUDGING_FP_LOCK:
        if not _JUDGING_FP:
            try:
                _JUDGING_FP.append(judging_fingerprint())
            except Exception:                                   # noqa: BLE001
                _JUDGING_FP.append("unknown")
    return _JUDGING_FP[0]


# ==========================================================================
# World-version fingerprint
# ==========================================================================

def generation_files() -> tuple[str, ...]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mf_for_generation", ROOT / "tools" / "make_freeze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                # noqa: S102
    return tuple(mod.GENERATION)


def world_fingerprint() -> str:
    """Content fingerprint (16 hex) over the generation path plus installed world
    plugins; with no plugin installed it equals the file fingerprint.
    """
    base = _fingerprint_of(generation_files())
    m, e = _plugin_manifest_shas()
    if not m and not e:
        return base
    # With no external task type registered the stamp is `base|world`, unchanged.
    blob = f"{base}|{m}" if not e else f"{base}|{m}|external:{e}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _plugin_manifest_shas() -> tuple[str, str]:
    """(world-side plugins, external task types) manifest hashes; `""` when empty.

    External task types (`external_gold`: gold blocks, gates, templates, probes)
    shape the world the same way world-side plugins do, and they also live
    outside the repo, so both enter the stamp.
    """
    from haenv.world_plugins import manifest_sha
    from haenv.external_gold import manifest_sha as external_manifest_sha
    return manifest_sha(), external_manifest_sha()


_WORLD_FP: list = []          # [file-content key, fingerprint] — see `world_fp_cached`


def world_fp_cached() -> str:
    try:
        files = []
        for rel in generation_files():
            path = _fingerprint_path(rel)
            files.append((rel, str(path), _raw_file_sha(path)))
        # The plugin manifests are part of the key: a plugin registered after the
        # cache was filled (a second job in the same process) must move the stamp.
        key = (tuple(files), _plugin_manifest_shas())
    except Exception:                                           # noqa: BLE001
        key = None
    if _WORLD_FP and key is not None and _WORLD_FP[0] == key:
        return _WORLD_FP[1]
    try:
        fp = world_fingerprint()
    except Exception:                                           # noqa: BLE001
        fp = "unknown"
    _WORLD_FP[:] = [key, fp]
    return fp


def world_vintages(rows) -> dict[str, int]:
    """World version → row count. The grouping key is `world_sha`, plus
    `world_knobs` and the first 8 chars of `job_sha256` when present (runtime knobs
    and the job's condition set change the world without moving `world_sha`).
    A missing stamp is counted as `"<unstamped>"`.
    """
    out: dict[str, int] = {}
    for r in rows:
        ref = r.get("world_ref") or {}
        sha = r.get("world_sha") or (ref.get("world_sha") if isinstance(ref, dict) else None)
        knobs = r.get("world_knobs") or (ref.get("world_knobs") if isinstance(ref, dict) else None)
        jsha = r.get("job_sha256") or (ref.get("job_sha256") if isinstance(ref, dict) else None)
        k = str(sha or "<unstamped>")
        if knobs:
            k = f"{k}·{knobs}"
        if jsha:
            k = f"{k}·job={str(jsha)[:8]}"
        out[k] = out.get(k, 0) + 1
    return out


# --------------------------------------------------------------------------
# Per-part fingerprint: one sha per judge and per non-judge file, so a batch is
# stale only if a part its rows used changed. Consulted only when the coarse
# fingerprint differs.
# --------------------------------------------------------------------------

#: Judging files stamped per file. The judge modules (the `haenv/judges/` package)
#: are left out: their bodies are stamped one at a time from `judges._CORE`'s
#: semantic source, so a comment edit there must not flip every row's part stamp.
_JUDGES_PACKAGE = "haenv/judges/"


def _non_judge_files() -> tuple[str, ...]:
    return tuple(f for f in judging_files() if not f.startswith(_JUDGES_PACKAGE))


def _sha16(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()[:16]


def _raw_file_sha(path: pathlib.Path) -> str:
    return _sha16(path.read_bytes() if path.is_file() else b"<missing>")


def _judge_source(fn, file_shas: dict[str, str]) -> bytes:
    """Cache source parsing, not the registry or the metadata of a judge."""
    import inspect
    import linecache
    try:
        fn = inspect.unwrap(fn)
        source_file = inspect.getsourcefile(fn)
        if source_file is None:
            return b"<nosource>"
        if source_file not in file_shas:
            file_shas[source_file] = _sha16(pathlib.Path(source_file).read_bytes())
        state = file_shas[source_file]
        key = (fn, getattr(fn, "__code__", None), state)
        if key in _SOURCE_CACHE:
            return _SOURCE_CACHE[key]
        # inspect's linecache only compares mtime and size; a content change
        # must also evict its potentially stale source text.
        if _SOURCE_FILE_STATES.get(source_file) != state:
            linecache.cache.pop(source_file, None)
            _SOURCE_FILE_STATES[source_file] = state
        src = _semantic_source(inspect.getsource(fn))
    except (OSError, TypeError):
        return b"<nosource>"
    if len(_SOURCE_CACHE) > 4096:
        _SOURCE_CACHE.clear()
        _SOURCE_FILE_STATES.clear()
    _SOURCE_CACHE[key] = src
    return src


def judge_parts() -> dict[str, str]:
    """`{part name: sha16}` with `file:<path>` for non-judge judging files and
    `judge:<name>` for built-in judges (source + kinds + `when` + category).
    """
    parts: dict[str, str] = {}
    file_shas: dict[str, str] = {}
    for rel in _non_judge_files():
        path = _fingerprint_path(rel)
        sha = _raw_file_sha(path)
        parts[f"file:{rel}"] = sha
        if path.is_file():
            file_shas[str(path)] = sha
    from .judges import _CORE
    for j in _CORE:
        src = _judge_source(j.fn, file_shas)
        meta = (f"{j.name}|{','.join(j.kinds)}|{j.when is not None}"
                f"|{getattr(j, 'category', 'deterministic')}").encode("utf-8")
        parts[f"judge:{j.name}"] = _sha16(src + b"\x00" + meta)
    return parts


def parts_used_by(kind: str, geometry: str) -> tuple[str, ...]:
    """Parts one eval row (`kind` × `geometry`) uses: every non-judge file plus the
    judges mounted for that cell.
    """
    from .judges import _CORE
    from .mount_table import NONE as _NONE
    from .mount_table import subject_of as _subject_of
    used = [f"file:{rel}" for rel in _non_judge_files()]
    used += [f"judge:{j.name}" for j in _CORE
             if _subject_of(j.name, geometry) != _NONE
             and ("*" in j.kinds or kind in j.kinds)]
    return tuple(sorted(used))


def parts_compatible(rows, before: dict[str, str],
                     after: dict[str, str] | None = None) -> tuple[bool, list[str]]:
    """`(compatible, changed parts)` between `before` and `after` (default: current
    code). A part missing from `before` counts as changed.
    """
    after = after if after is not None else judge_parts()
    need: set[str] = set()
    for r in rows:
        k = str(r.get("gold_kind") or "")
        g = str(r.get("geometry") or r.get("_geometry") or "")
        if not k and str(r.get("overall") or "").startswith("ABORT"):
            # An aborted cell (no answer, leak, ...) was never scored: it used no judging
            # part, so no judging change can make it stale.
            continue
        if not k or not g:
            return False, ["<row has no recorded gold_kind/geometry => cannot tell what it used>"]
        need |= set(parts_used_by(k, g))
    changed = sorted(n for n in need
                     if n not in before or before[n] != after.get(n))
    return (not changed), changed


def judging_vintages(rows) -> dict[str, int]:
    """Judging fingerprint → row count; a missing stamp is counted as `"<unstamped>"`."""
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get("judging_sha16") or "<unstamped>")
        out[k] = out.get(k, 0) + 1
    return out
