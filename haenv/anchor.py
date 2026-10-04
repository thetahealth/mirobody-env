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
import threading

from haenv import data_root as _data_root
from .semantic_source import (  # noqa: F401
    _SEM_CACHE,
    _semantic_cached,
    _semantic_of,
    _semantic_source,
    _unparse,
    semantic_bytes,
)
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
        bases = {"haenv": _PACKAGE_ROOT, "haenv_kernel": _PACKAGE_ROOT.parent / "haenv_kernel",
                 "verifier_core": _PACKAGE_ROOT.parent / "verifier_core"}
        if head in bases:
            return bases[head] / tail
    return ROOT / rel


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


def _reader(rev: str | None):
    """Source access at a revision (git objects), or the working tree for `None`."""
    if rev is None:
        return lambda rel: (ROOT / rel).read_bytes() if (ROOT / rel).is_file() else None
    return lambda rel: _git_blob(rev, rel)


def _segments(rev: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if rev is None:
        return judging_files(), infra_files()
    return _segment_at(rev, "JUDGING"), _segment_at(rev, "INFRA")


def _commit_of(rev: str | None) -> str | None:
    """Full commit id of `rev`, so a moving ref such as HEAD is never a cache key."""
    if rev is None:
        return None
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{rev}^{{commit}}"],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or None


def verify_judging_move(row: dict, *, infra=None) -> dict:
    """Recompute one `old judging_sha16 -> new` row that records moved code.

    `old` must reproduce at `rev` and `new` at `new_rev` (`None` = the working tree, for a
    check before committing); between the two the judging segment may only move code
    (`anchor_moves.certify`), apart from the definitions `changed` names with a reason.
    `infra`, when given, limits the `INFRA` files code may leave the segment for (an anchor
    passes the ones a previous anchor already recorded). Raises `EquivalenceError`.
    """
    from .anchor_moves import MoveError, certify
    rev, new_rev = row["rev"], row.get("new_rev")
    j_old, i_old = _segments(rev)
    j_new, i_new = _segments(new_rev)
    if infra is not None:
        i_new = tuple(f for f in i_new if f in set(infra))
    if fingerprint_at(rev, j_old) != row["old"]:
        raise EquivalenceError(f"{row['old']} does not reproduce at {rev}")
    got = _fingerprint_of(j_new) if new_rev is None else fingerprint_at(new_rev, j_new)
    if got != row["new"]:
        raise EquivalenceError(f"judging at {new_rev or 'the working tree'} is {got}, not {row['new']}")
    try:
        cert = certify((_reader(rev), j_old, i_old), (_reader(new_rev), j_new, i_new),
                       row.get("changed"), keys=(_commit_of(rev), _commit_of(new_rev)))
    except MoveError as e:
        raise EquivalenceError(f"{row['old']}@{rev} -> {row['new']}: {e}") from e
    return {**row, **cert}


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
