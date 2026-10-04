"""Certify that two revisions of the judging segment differ only by moving code.

`judging_sha16` hashes each file under its path, so moving a function to another file,
renaming a module or turning a lazy import into a top-level one changes it although no
score can change. `move_digest` describes the segment one level down, where those edits
are invisible: one entry per top-level statement, made of

* the statement's semantic source with every import inside it removed, and
* for every name it reads, the definition that name is bound to, followed through
  imports and re-exports to the statement that defines it and identified by that
  statement's own semantic source.

File paths, the module a definition sits in, and where an import is written do not
enter. Two revisions with equal digests run the same definitions bound to the same
definitions, so `verify_judging_move` lets an anchor declare the old `judging_sha16`
equivalent to the new one.

Definitions that leave the segment for an `INFRA` file are dropped from both sides, as
`verify_judging_equivalence` drops whole files; a name bound to one is identified by name
only, because `INFRA` content is outside the judging fingerprint by design.

A name some function rebinds with `global` also records whether a reader sees its current
value (same module, an import at call time, an attribute of the module) or the copy a
module-level import took at load time: moving the definition away from its readers turns
the first into the second, and that is refused.

Not seen, so refused rather than certified: a statement that looks names up dynamically
(`globals()`, `vars()`, `sys.modules`, `__file__`) changing module; a star import.
The import-time order of module-level statements is not compared.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import pathlib
from typing import Callable

_DYNAMIC = ("globals", "vars")
_DYNAMIC_ATTR = (("sys", "modules"),)
_DYNAMIC_NAMES = ("__name__", "__file__", "__package__", "__spec__")
#: Definitions of the segments themselves: outside every segment, identified by name.
_FREEZE_MANIFEST = "tools/make_freeze.py"
_KERNEL_DIRS = ("haenv_kernel", "core")


class MoveError(ValueError):
    """The two revisions differ by more than moved code, or the digest cannot see a change."""


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]


class _DropImports(ast.NodeTransformer):
    """Remove every import inside a statement, keeping the removed nodes."""

    def __init__(self):
        self.imports: list[ast.stmt] = []

    def _drop(self, node):
        self.imports.append(node)
        return None

    visit_Import = visit_ImportFrom = _drop

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(node, field, None)
            if field == "body" and isinstance(seq, list) and not seq:
                node.body = [ast.Pass()]
        return node


def _only_imports(stmt: ast.stmt) -> bool:
    """A module-level statement that only binds imports (`try: import x except ...`)."""
    for n in ast.walk(stmt):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                          ast.AnnAssign, ast.AugAssign, ast.Call, ast.Return)):
            return False
    return True


class _Layout:
    """The modules of one revision, read through `read(rel) -> bytes | None`."""

    def __init__(self, read: Callable[[str], bytes | None], judging, infra):
        self.read = read
        self.judging = tuple(judging)
        self.infra = set(infra)
        self._trees: dict[str, ast.Module | None] = {}
        self._ns: dict[str, dict] = {}
        self._path_of: dict[str, str | None] = {}
        self._resolved: dict[tuple, tuple] = {}
        self._rebound: dict[str, set[str]] = {}
        self.src: dict[tuple, tuple[str, dict]] = {}
        self.kernel_dir = next((d for d in _KERNEL_DIRS
                                if any(f.startswith(d + "/") for f in self.judging)), "haenv_kernel")

    # ---- module names and files
    def path_of(self, mod: str) -> str | None:
        if mod in self._path_of:
            return self._path_of[mod]
        parts = mod.split(".")
        cands = [f"{'/'.join(parts)}.py", f"{'/'.join(parts)}/__init__.py"]
        if len(parts) == 1:                                   # bare kernel import on sys.path
            cands += [f"{self.kernel_dir}/{mod}.py"]
        hit = next((c for c in cands if self.read(c) is not None), None)
        self._path_of[mod] = hit
        return hit

    @staticmethod
    def module_of(rel: str) -> str:
        p = pathlib.PurePosixPath(rel).with_suffix("")
        parts = list(p.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def canonical(self, rel: str) -> str:
        """Module identity that survives the kernel's move from `core/` to `haenv_kernel/`."""
        head, _, tail = rel.partition("/")
        if head in _KERNEL_DIRS:
            return "kernel:" + self.module_of(tail)
        return self.module_of(rel)

    def tree(self, rel: str) -> ast.Module | None:
        if rel not in self._trees:
            raw = self.read(rel)
            try:
                t = ast.parse(raw.decode("utf-8")) if raw is not None else None
            except SyntaxError:
                t = None
            if t is not None:
                _strip_docstrings(t)
            self._trees[rel] = t
        return self._trees[rel]

    def rebound(self, rel: str) -> set[str]:
        """Module-level names some function in `rel` rebinds with `global`."""
        if rel not in self._rebound:
            t = self.tree(rel)
            self._rebound[rel] = {x for n in (ast.walk(t) if t is not None else ())
                                  if isinstance(n, ast.Global) for x in n.names}
        return self._rebound[rel]

    def _package(self, rel: str) -> str:
        mod = self.module_of(rel)
        return mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]

    def absolute(self, rel: str, level: int, module: str | None) -> str:
        if not level:
            return module or ""
        pkg = self._package(rel)
        if level > 1:
            pkg = pkg.rsplit(".", level - 1)[0]
        if rel.startswith(self.kernel_dir + "/") and not pkg.startswith(self.kernel_dir):
            pkg = self.kernel_dir
        return ".".join(x for x in (pkg, module or "") if x)

    # ---- bindings
    def import_bindings(self, rel: str, nodes) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for n in nodes:
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        out[a.asname] = ("module", a.name)
                    else:
                        out[a.name.split(".")[0]] = ("module", a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                base = self.absolute(rel, n.level, n.module)
                for a in n.names:
                    if a.name == "*":
                        raise MoveError(f"{rel}: star import from {base} is not certifiable")
                    out[a.asname or a.name] = ("from", base, a.name)
        return out

    def namespace(self, rel: str) -> dict[str, tuple]:
        """Module-level name -> ("def", rel, index) | import binding."""
        if rel in self._ns:
            return self._ns[rel]
        self._ns[rel] = {}
        t = self.tree(rel)
        ns: dict[str, tuple] = {}
        if t is not None:
            top_imports = []
            for s in t.body:
                if isinstance(s, (ast.Import, ast.ImportFrom)):
                    top_imports.append(s)
                elif not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    top_imports += [n for n in ast.walk(s)
                                    if isinstance(n, (ast.Import, ast.ImportFrom))]
            ns.update(self.import_bindings(rel, top_imports))
            for i, s in enumerate(t.body):
                for name in _bound_names(s):
                    ns[name] = ("def", rel, i)
        self._ns[rel] = ns
        return ns

    def resolve(self, binding: tuple, seen=None) -> tuple:
        """Follow a binding to ("def", rel, i) | ("module", rel) | ("ext", dotted)."""
        if seen is None:
            if binding not in self._resolved:
                self._resolved[binding] = self.resolve(binding, set())
            return self._resolved[binding]
        if binding in seen:
            return ("cycle",) + binding
        seen.add(binding)
        kind = binding[0]
        if kind == "def":
            return binding
        if kind == "module":
            p = self.path_of(binding[1])
            return ("module", p) if p else ("ext", binding[1])
        _, base, name = binding
        p = self.path_of(base) if base else None
        if p is None:
            return ("ext", f"{base}.{name}")
        ns = self.namespace(p)
        if name in ns:
            return self.resolve(ns[name], seen)
        sub = self.path_of(f"{base}.{name}")
        if sub:
            return ("module", sub)
        return ("ext", f"{base}.{name}")


def _bound_names(s: ast.stmt) -> list[str]:
    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [s.name]
    if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = s.targets if isinstance(s, ast.Assign) else [s.target]
        out = []
        for t in targets:
            out += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        return out
    return []


def _dynamic(stmt: ast.stmt, nodes=None) -> bool:
    """Whether the statement looks itself up by module identity."""
    for n in (nodes if nodes is not None else ast.walk(stmt)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _DYNAMIC:
            return True
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and (n.value.id, n.attr) in _DYNAMIC_ATTR):
            return True
        if isinstance(n, ast.Name) and n.id in _DYNAMIC_NAMES:
            return True
    return False


def _function_locals(stmt: ast.stmt, nodes=None) -> set[str]:
    """Names a function or class statement binds inside itself (arguments, assignments,
    loop and `with` targets), which therefore never read the module namespace.
    Approximate in one direction only: a name bound in any nested scope is treated as
    local everywhere in the statement, so the digest may see fewer bindings, never wrong ones."""
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return set()
    out: set[str] = set()
    declared: set[str] = set()
    for n in (nodes if nodes is not None else ast.walk(stmt)):
        if n is stmt:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            declared.update(n.names)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = stmt.args
        out.update(x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs))
        out.update(x.arg for x in (a.vararg, a.kwarg) if x is not None)
    return out - declared


def _is_declared(rel: str, names, declared) -> bool:
    """Whether every name a statement of `rel` binds is declared, bare (`name`, every
    definition of that name) or qualified by its file (`rel:name`, that one only)."""
    return bool(names) and all(n in declared or f"{rel}:{n}" in declared for n in names)


class _Digest:
    def __init__(self, lay: _Layout, infra_names: set[str], declared: set[str]):
        self.lay = lay
        self.infra_names = infra_names
        self.declared = declared
        self._src = lay.src

    def nodes(self, rel: str, i: int) -> list:
        """Every node of statement `i` of `rel`, walked once."""
        key = ("nodes", rel, i)
        if key not in self._src:
            self._src[key] = list(ast.walk(self.lay.tree(rel).body[i]))
        return self._src[key]

    def statement(self, rel: str, i: int) -> tuple[str, dict]:
        """Semantic source of statement `i` of `rel` without its imports, and the import
        bindings those imports made."""
        key = (rel, i)
        if key not in self._src:
            from .semantic_source import _unparse
            node = self.lay.tree(rel).body[i]
            if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in self.nodes(rel, i)):
                drop = _DropImports()
                node = drop.visit(copy.deepcopy(node))
                imports = drop.imports
            else:
                imports = []
            src = _unparse(ast.fix_missing_locations(node)) if node is not None else ""
            self._src[key] = (src, self.lay.import_bindings(rel, imports))
        return self._src[key]

    def identity(self, target: tuple) -> str:
        if target[0] == "def":
            _, rel, i = target
            names = _bound_names(self.lay.tree(rel).body[i])
            if rel == _FREEZE_MANIFEST:
                return "freeze:" + ",".join(sorted(names))
            if _is_declared(rel, names, self.declared):
                return "declared:" + ",".join(sorted(names))
            if rel in self.lay.infra or (names and set(names) <= self.infra_names):
                return "infra:" + ",".join(sorted(names))
            src, _ = self.statement(rel, i)
            return "def:" + hashlib.sha256(src.encode()).hexdigest()[:16]
        if target[0] == "module":
            return "module:" + self.lay.canonical(target[1])
        return ":".join(target)

    def _targets(self, rel: str, i: int) -> tuple[str, dict, bool]:
        """(source hash, name -> resolved target, dynamic): independent of the declared set,
        so cached on the layout."""
        key = ("targets", rel, i)
        if key in self._src:
            return self._src[key]
        stmt = self.lay.tree(rel).body[i]
        nodes = self.nodes(rel, i)
        src, local = self.statement(rel, i)
        ns = self.lay.namespace(rel)
        private = _function_locals(stmt, nodes) - set(local)
        out: dict[str, tuple] = {}

        def bound(name):
            """(target, how): `live` reads the module's current value (same module, an
            import run at call time, an attribute of the module); `copy` reads the value a
            module-level import took when this module was loaded."""
            if name in private:
                return None
            if name in local:
                return self.lay.resolve(local[name]), "live"
            if name in ns:
                return self.lay.resolve(ns[name]), ("live" if ns[name][0] == "def" else "copy")
            return None

        # A module name read only as `mod.attr` is identified by what each attribute resolves
        # to, so moving those definitions to another module is a move.
        attr_base = {id(n.value) for n in nodes
                     if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        for n in nodes:
            if isinstance(n, ast.Name) and n.id not in out:
                b = bound(n.id)
                if b is not None and not (b[0][0] == "module" and id(n) in attr_base):
                    out[n.id] = b
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                b = bound(n.value.id)
                if b is not None and b[0][0] == "module":
                    held = self.lay.namespace(b[0][1]).get(n.attr)
                    out[f"{n.value.id}.{n.attr}"] = (self.lay.resolve(
                        ("from", self.lay.module_of(b[0][1]), n.attr)),
                        "live" if held is None or held[0] == "def" else "copy")
        hit = (hashlib.sha256(src.encode()).hexdigest()[:16], out, _dynamic(stmt, nodes))
        self._src[key] = hit
        return hit

    def entry(self, rel: str, i: int) -> dict:
        src, targets, dynamic = self._targets(rel, i)
        binds = {}
        for k, (t, how) in sorted(targets.items()):
            ident = self.identity(t)
            # A name some function rebinds with `global` is only current where it is read
            # live; a module-level import elsewhere keeps the value of load time.
            if t[0] == "def" and set(_bound_names(self.lay.tree(t[1]).body[t[2]])) \
                    & self.lay.rebound(t[1]):
                ident += "|" + how
            binds[k] = ident
        e = {"src": src, "binds": binds}
        if dynamic:
            e["module"] = self.lay.canonical(rel)
        return e


def move_digest(read: Callable[[str], bytes | None], judging, infra,
                infra_names: set[str] = frozenset(),
                declared: set[str] = frozenset()) -> tuple[str, list]:
    """(digest, entries) of the judging segment as `read` sees it; see the module docstring.

    `declared` names definitions the caller states have changed: they are left out, and a
    name bound to one is identified by name only. Each entry also carries `_where` and
    `_names`, which do not enter the digest.
    """
    return _digest(_Layout(read, judging, infra), infra_names, declared)


def _digest(lay: "_Layout", infra_names, declared) -> tuple[str, list]:
    read = lay.read
    dg = _Digest(lay, set(infra_names), set(declared))
    entries: list = []
    for rel in lay.judging:
        if rel.endswith(".py"):
            t = lay.tree(rel)
            if t is None:
                raw = read(rel)
                entries.append({"raw": hashlib.sha256(raw or b"<missing>").hexdigest()[:16],
                                "_where": rel})
                continue
            for i, s in enumerate(t.body):
                if isinstance(s, (ast.Import, ast.ImportFrom)) or _only_imports(s):
                    continue
                names = _bound_names(s)
                if names and set(names) <= set(infra_names):
                    continue
                e = dg.entry(rel, i)
                e["_where"] = f"{rel}:{s.lineno}"
                e["_names"] = names
                entries.append(e)
        else:
            key = ("data", rel)
            if key not in lay.src:
                from .semantic_source import _semantic_of
                raw = read(rel)
                sem = _semantic_of(raw, pathlib.PurePosixPath(rel).suffix.lower()) \
                    if raw is not None else b"<missing>"
                lay.src[key] = hashlib.sha256(sem).hexdigest()[:16]
            entries.append({"data": lay.src[key], "_where": rel, "_names": [rel]})
    kept = [e for e in entries
            if not _is_declared(e["_where"].rsplit(":", 1)[0], e.get("_names"), declared)]
    kept.sort(key=lambda e: json.dumps(_bare(e), sort_keys=True))
    blob = json.dumps([_bare(e) for e in kept], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], entries


#: One layout per fixed revision whose caches later layouts of that revision share; a
#: working tree is never cached.
_LAYOUTS: dict[tuple, "_Layout"] = {}


def _layout(side: tuple, key) -> "_Layout":
    """A layout for `side`; for a fixed revision the parses and resolutions, which do not
    depend on the segment lists, are shared by every layout of that revision."""
    lay = _Layout(*side)
    if key is not None:
        base = _LAYOUTS.setdefault(key, lay)
        for cache in ("_trees", "_ns", "_path_of", "_resolved", "_rebound", "src"):
            setattr(lay, cache, getattr(base, cache))
    return lay


def certify(old: tuple, new: tuple, changed: dict[str, str] | None = None, *,
            keys: tuple = (None, None)) -> dict:
    """Certify that segment `old` becomes segment `new` by moving code, apart from `changed`.

    `old` and `new` are `(read, judging, infra)`. `changed` maps definition names to the
    reason they changed; every one must really differ, and nothing else may. A name
    written `path:name` declares only the definition in that file, so a common name such
    as `ROOT` does not hide the other modules' definitions of it. `keys` names the
    revision of each side (`None` for a working tree); a fixed revision's parse is reused.
    Raises `MoveError`; returns a summary.
    """
    changed = dict(changed or {})
    infra_names = infra_only_names(*new)
    lay_old, lay_new = _layout(old, keys[0]), _layout(new, keys[1])
    d_old, e_old = _digest(lay_old, infra_names, set(changed))
    d_new, e_new = _digest(lay_new, infra_names, set(changed))
    if d_old != d_new:
        keep = set(changed)
        raise MoveError("the segments differ by more than moved code: "
                        + json.dumps(diff([e for e in e_old if not set(e.get("_names") or ()) & keep],
                                          [e for e in e_new if not set(e.get("_names") or ()) & keep]),
                                     ensure_ascii=False))
    stale = [name for name in changed
             if _digest(lay_old, infra_names, set(changed) - {name})[0]
             == _digest(lay_new, infra_names, set(changed) - {name})[0]]
    if stale:
        raise MoveError(f"declared as changed but certify without the declaration: {stale}")
    return {"move_digest": d_new, "statements_old": len(e_old), "statements_new": len(e_new),
            "infra_dropped": sorted(infra_names & _defined(lay_old, lay_old.judging)),
            "changed": changed}


def _bare(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}


def _defined(lay: "_Layout", files) -> set[str]:
    out: set[str] = set()
    for rel in files:
        t = lay.tree(rel) if rel.endswith(".py") else None
        for s in (t.body if t is not None else ()):
            out.update(_bound_names(s))
    return out


def infra_only_names(read: Callable[[str], bytes | None], judging, infra) -> set[str]:
    """Names defined at module level in an `INFRA` file and in no judging file."""
    lay = _Layout(read, judging, infra)
    return _defined(lay, infra) - _defined(lay, lay.judging)


def diff(old: list, new: list) -> dict:
    """Entries only one side has, for an error message."""
    o = {json.dumps(_bare(e), sort_keys=True): e.get("_where") for e in old}
    n = {json.dumps(_bare(e), sort_keys=True): e.get("_where") for e in new}
    return {"only_old": sorted(o[k] or "?" for k in set(o) - set(n)),
            "only_new": sorted(n[k] or "?" for k in set(n) - set(o))}
