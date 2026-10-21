"""Criterion identities from source imports and registry reads, including Git history.

The legacy judging stamp remains readable. These identities decide which saved
columns need recomputing; the sorted identities also name a release's ruler.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import subprocess

import haenv.provenance.anchor as anchor
from haenv.scoring.semantic.source import _semantic_cached, _source_tree
from haenv.locations import source_path, resource_path
from haenv.provenance.criteria import PACK_SCORING, DISPATCH_BOUNDARIES, CRITERIA
from haenv.provenance.history import (
    SEGMENT_NAMES, contract_tables as _contract_tables, contract_tables_at,
    declarations_at, literal_tables, revision_path,
)


@lru_cache(maxsize=256)
def _registry_module_names(raw: bytes) -> tuple[str, ...]:
    """Cache only immutable names from content, never paths or file existence."""
    import re
    import yaml
    doc = yaml.load(raw.decode("utf-8"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    todo, names = [doc], []
    while todo:
        value = todo.pop()
        if isinstance(value, dict):
            todo.extend(value.values())
        elif isinstance(value, list):
            todo.extend(value)
        elif isinstance(value, str) and re.fullmatch(r"(?:haenv|haenv_kernel|verifier_core|tools)(?:\.[A-Za-z_]\w*)+", value):
            names.append(value)
    return tuple(names)


def criterion_fingerprints(rev: str | None = None, *, root: Path | None = None,
                           overrides: dict[str, bytes] | None = None) -> dict:
    """Compute semantic identities and dependency receipts without importing judges.

    `rev` reads immutable Git objects, never checks out or executes old judging
    code. Existing import exemptions stop code traversal, but their registry
    reads still enter the closure, as in the freeze-discipline data-edge guard.
    A criterion whose entry point did not exist is absent, never fabricated.
    """
    root = Path(root) if root is not None else anchor.ROOT
    overrides = dict(overrides or {})
    if rev is not None and overrides:
        raise ValueError("A historical commit cannot be combined with source overrides")
    commit = None
    committed_paths = None
    if rev is not None:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", f"{rev}^{{commit}}"], text=True).strip()
        committed_paths = set(subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--full-tree", "--name-only", "-z", commit]).decode().split("\0"))
        physical_paths = committed_paths
        anchor._prime_blobs(commit, [path for path in committed_paths
                                   if path.endswith((".yaml", ".yml", ".json", ".toml"))], root=root)

    @lru_cache(maxsize=None)
    def raw_read(rel):
        if rel in overrides:
            return overrides[rel]
        if commit is not None:
            if rel not in physical_paths:
                return None
            return anchor._git_blob(commit, rel, root=root)
        path = anchor._fingerprint_path(rel) if root == anchor.ROOT else source_path(rel, root=root)
        return path.read_bytes() if path.is_file() else None

    if commit is not None:
        ns = declarations_at(raw_read)
        contracts, dispatch_boundaries = contract_tables_at(raw_read, ns)
        physical_paths = committed_paths
        prefix = "src/" if ns["SOURCE_LAYOUT"] == "src" else ""
        committed_paths = {rel[len(prefix):] if prefix and rel.startswith(prefix) else rel
                           for rel in physical_paths}
    else:
        manifest = raw_read("haenv/provenance/declarations.py")
        if manifest is None:
            raise ValueError("Missing haenv/provenance/declarations.py; cannot establish the judging boundary")
        ns = literal_tables(manifest, SEGMENT_NAMES)
        criterion_entry = ns["CRITERION_DECLARATIONS"]
        if criterion_entry != "haenv/provenance/criteria.py":
            raise ValueError(f"Unsupported current criterion declaration entry: {criterion_entry!r}")
        criterion_source = raw_read(criterion_entry)
        if criterion_source is None:
            raise ValueError(f"Missing criterion declarations: {criterion_entry}")
        contracts, dispatch_boundaries = _contract_tables(criterion_source)
        ns["_entry"] = "haenv/provenance/declarations.py"

    @lru_cache(maxsize=None)
    def read(rel):
        if commit is not None:
            physical = revision_path(rel, ns)
            if physical not in physical_paths:
                return None
            return raw_read(physical)
        return raw_read(rel)

    allow = set(ns["JUDGING_IMPORT_ALLOW"])
    provenance = set(ns.get("PROVENANCE_MODULES", {}))
    firstparty = {"haenv", "haenv_kernel", "verifier_core", "tools"}
    declaration_module = ns["_entry"].removeprefix("src/").removesuffix(".py").replace("/", ".")
    modern_namespace = ns.get('CRITERION_DECLARATIONS') == 'haenv/provenance/criteria.py'
    registry_owner = 'haenv/resources/registry.py' if modern_namespace else 'haenv/registry.py'
    pack_owner = 'haenv/app/packs.py' if modern_namespace else 'haenv/pack_skeleton.py'

    @lru_cache(maxsize=None)
    def tree(rel):
        blob = read(rel)
        if blob is None:
            raise ValueError(f"Missing dependency: {rel}")
        return _source_tree(blob)

    @lru_cache(maxsize=None)
    def levels(node):
        out, level = [], (node,)
        while level:
            out.append(level)
            level = tuple(child for n in level for child in ast.iter_child_nodes(n))
        return tuple(out)

    @lru_cache(maxsize=None)
    def walk(node):
        # Source ASTs are immutable here; this cache ends with this build.
        if not isinstance(node, ast.Module):
            return tuple(n for level in levels(node) for n in level)
        # Selected modules share their statements. Reuse each statement's walk
        # while preserving ast.walk's exact breadth-first order.
        groups = [levels(child) for child in ast.iter_child_nodes(node)]
        return (node, *(n for depth in range(max(map(len, groups), default=0))
                        for group in groups if depth < len(group) for n in group[depth]))

    @lru_cache(maxsize=None)
    def bindings(node):
        """Module bindings, without confusing a function's locals with exports."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {node.name}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {a.asname or a.name.split(".")[0] for a in node.names}
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            return {n.id for target in targets for n in walk(target) if isinstance(n, ast.Name)}
        return set().union(*(bindings(n) for n in ast.iter_child_nodes(node)))

    @lru_cache(maxsize=None)
    def references(node):
        return {n.id for n in walk(node) if isinstance(n, ast.Name)}

    @lru_cache(maxsize=None)
    def statement_index(rel):
        available, users = set(), {}
        for i, statement in enumerate(tree(rel).body):
            names = bindings(statement)
            available.update(names)
            # Unbound module initialization matters when it touches a wanted name.
            for name in names or references(statement):
                users.setdefault(name, set()).add(i)
        return available, users

    @lru_cache(maxsize=None)
    def selected_tree(rel, symbols):
        """Follow a named scoring export and its local helpers, hash its whole file.

        Ordinary module roots retain every import. A named callback/import does
        not execute the bodies of unrelated registration/generation functions.
        Re-exports, constants, decorators and local helper references are followed.
        """
        node = tree(rel)
        if symbols is None:
            return node
        if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in node.body):
            # A wildcard can supply any local helper/global; retain the whole
            # module rather than drop an import with no explicit binding name.
            return node
        available, users = statement_index(rel)
        missing = set(symbols) - available
        if missing:
            raise ValueError(f"Unresolved scoring export in {rel}: {sorted(missing)}")
        pending, kept = set(symbols), set()
        while pending:
            name = pending.pop()
            for i in users.get(name, ()):
                if i not in kept:
                    kept.add(i)
                    pending.update(references(node.body[i]))
        return ast.Module(body=[n for i, n in enumerate(node.body) if i in kept], type_ignores=[])

    def module(parts):
        return module_cached(tuple(parts))

    @lru_cache(maxsize=None)
    def module_cached(parts):
        if not parts or parts[0] not in firstparty:
            return None
        if ".".join(parts) == declaration_module:
            # This is the segment declaration already interpreted above, not
            # a scoring dependency. Hashing its whole registry roster would
            # couple every criterion to every table, including management.
            return None
        path = "/".join(parts)
        for rel in (path + ".py", path + "/__init__.py"):
            if read(rel) is not None:
                return rel
        live_path = anchor._fingerprint_path(path) if root == anchor.ROOT else source_path(path, root=root)
        if ((committed_paths is not None and any(rel.startswith(path + "/") for rel in committed_paths))
                or (committed_paths is None and live_path.is_dir())):
            return None                    # namespace package: no initializer to hash
        raise ValueError(f"Unresolved first-party import: {'.'.join(parts)}")

    @lru_cache(maxsize=None)
    def imports(rel, node):
        out = []
        here = rel.split("/")[:-1]
        for n in walk(node):
            if isinstance(n, ast.ImportFrom):
                parts = (here[:len(here) - n.level + 1] if n.level else [])
                parts += n.module.split(".") if n.module else []
                if not parts or parts[0] not in firstparty:
                    continue
                if ".".join(parts) == declaration_module:
                    unknown = {a.name for a in n.names} - (set(ns) & SEGMENT_NAMES)
                    if unknown:
                        raise ValueError(f"Undeclared source declaration symbols: {sorted(unknown)}")
                owner = module(parts) if n.module else None
                for a in n.names:
                    # A package import can name a submodule or a package symbol.
                    child = "/".join([*parts, a.name])
                    is_module = read(child + ".py") is not None or read(child + "/__init__.py") is not None
                    target = module([*parts, a.name]) if is_module else owner or module(parts)
                    if target:
                        out.append((target, a.asname or a.name, "*" if is_module else a.name))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    dep = module(a.name.split("."))
                    if dep:
                        # Without an alias a dotted import binds the top-level
                        # package, not this module. Keep that target unscoped.
                        alias = a.asname or (a.name if "." not in a.name else "")
                        out.append((dep, alias, "*"))
            elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
                function = n.func.id if isinstance(n.func, ast.Name) else (
                    n.func.attr if isinstance(n.func, ast.Attribute) else None)
                value = n.args[0].value
                if function in {"import_module", "__import__"} and isinstance(value, str):
                    dep = module(value.split("."))
                    if dep:
                        out.append((dep, "", "*"))
        return out

    registry_files = (committed_paths if committed_paths is not None else
                      {"registry/" + p.name for p in (resource_path("registry") if root == anchor.ROOT else root / "registry").glob("*.yaml")}) | set(overrides)

    @lru_cache(maxsize=None)
    def yaml_refs(node):
        refs = {"registry/" + n.value.rsplit("/", 1)[-1] for n in walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.endswith((".yaml", ".yml"))
                and read("registry/" + n.value.rsplit("/", 1)[-1]) is not None}
        # A formatted registry path (e.g. scoring_{task_type}.yaml) reads a
        # family of tables. Match that family from the selected source tree.
        for n in walk(node):
            if isinstance(n, ast.JoinedStr):
                pattern = "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "*"
                                  for v in n.values)
                if pattern.startswith("registry/") and pattern.endswith((".yaml", ".yml")):
                    refs |= {rel for rel in registry_files if fnmatch.fnmatchcase(rel, pattern)}
        return refs

    # The registry loader -> loader closure, including aliased consumers.
    reg = tree(registry_owner)
    fns = {n.name: n for n in reg.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    loaders = {name: yaml_refs(node) for name, node in fns.items()}
    calls = {name: {n.func.id for n in walk(node) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id in fns}
             for name, node in fns.items()}
    while True:
        grown = {name: refs | set().union(*(loaders[c] for c in calls[name]))
                 for name, refs in loaders.items()}
        if grown == loaders:
            break
        loaders = grown

    @lru_cache(maxsize=None)
    def dispatch_tree(rel, node):
        if rel not in {path for path, _ in dispatch_boundaries}:
            return node, []
        import copy
        receipts = []

        def visit(n, scope=""):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope = f"{scope}.{n.name}" if scope else n.name
                reason = dispatch_boundaries.get((rel, scope))
                if reason:
                    stopped = imports(rel, n)
                    receipts.append({"module": rel, "scope": scope, "reason": reason,
                                     "imports": sorted({dep for dep, _, _ in stopped})})
                    return None
            # Preserve the AST shape and literals: path assignments outside a
            # call are data dependencies too. Never mutate the cached source AST.
            out = copy.copy(n)
            for key, value in ast.iter_fields(n):
                if isinstance(value, ast.AST):
                    setattr(out, key, visit(value, scope))
                elif isinstance(value, list):
                    setattr(out, key, [v for item in value
                                      if (v := visit(item, scope) if isinstance(item, ast.AST) else item) is not None])
            return out

        return visit(node), receipts

    @lru_cache(maxsize=None)
    def name_uses(node):
        parents, uses = {}, {}
        for n in walk(node):
            parents.update((child, n) for child in ast.iter_child_nodes(n))
            if isinstance(n, ast.Name):
                uses.setdefault(n.id, []).append(n)
        return parents, uses

    @lru_cache(maxsize=None)
    def imported_symbols(node, alias, symbol):
        if symbol != "*":
            return (symbol,)
        # A module alias can be scoped when every use is an explicit attribute;
        # passing the module object/getattr/dynamic imports retain the whole module.
        parents, names = name_uses(node)
        uses = names.get(alias, ())
        if uses and all(isinstance(parents.get(n), ast.Attribute) and parents[n].value is n for n in uses):
            return tuple(sorted({parents[n].attr for n in uses}))
        return None

    def registered_judges(rel):
        node = tree(rel)
        aliases = {alias for dep, alias, symbol in imports(rel, node)
                   if dep == pack_owner and symbol == "Pack"}
        callbacks = set()
        for n in walk(node):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name) or n.func.id not in aliases:
                continue
            for kw in n.keywords:
                if kw.arg != "judge":
                    continue
                if not isinstance(kw.value, ast.Tuple) or len(kw.value.elts) != 4:
                    raise ValueError(f"Unresolved judge registration in {rel}")
                for value in (kw.value.elts[1], kw.value.elts[3]):
                    if isinstance(value, ast.Name):
                        callbacks.add(value.id)
                    elif not isinstance(value, ast.Constant) or value.value is not None:
                        raise ValueError(f"Unresolved judge callback in {rel}")
        if not callbacks:
            raise ValueError(f"No registered scoring callbacks in {rel}")
        return (rel, tuple(sorted(callbacks)))

    @lru_cache(maxsize=None)
    def file_edges(rel):
        # The entire hashed file is subject to the illegal-import guard, even
        # when the import lives in an unused callback or a dispatch function.
        forbidden = ({dep for dep, _, _ in imports(rel, tree(rel))} & provenance)
        # Code is hashed as a complete file. Its data reads keep that same
        # conservative surface, even when executable imports are symbol-scoped.
        data_node = tree(rel)
        refs = yaml_refs(data_node).copy() if rel != registry_owner else set()
        aliases = {alias: symbol for dep, alias, symbol in imports(rel, data_node)
                   if dep == registry_owner}
        for n in walk(data_node):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else (
                    n.func.attr if isinstance(n.func, ast.Attribute) else None)
                refs |= loaders.get(aliases.get(name, name), set())
        return refs, forbidden

    @lru_cache(maxsize=None)
    def edges(rel, symbols=None, scoped=False):
        selected = selected_tree(rel, symbols)
        node, stopped = dispatch_tree(rel, selected)
        imps = imports(rel, node)
        refs, forbidden = file_edges(rel)
        deps = {(dep, imported_symbols(selected, alias, symbol) if scoped else None)
                for dep, alias, symbol in imps}
        return deps, refs, stopped, forbidden

    @lru_cache(maxsize=None)
    def data_modules(rel):
        # Registry-declared adapters are executable dependencies too. This
        # handles importlib.import_module(policy['adapters'][pack]) generically.
        out = set()
        for value in _registry_module_names(read(rel)):
            parts = value.split(".")
            target = module(parts)
            if target:
                out.add(target)
            # Registry adapters are loaded dynamically: their package is
            # itself a declared part of that executable data dependency.
            parent = "/".join(parts[:-1]) + "/__init__.py"
            if read(parent) is not None:
                out.add(parent)
        return out

    criteria, dependencies, boundary_receipts = {}, {}, {}
    for name, spec in sorted(contracts.items()):
        roots = [rel for entry in spec["roots"] for rel in ns.get(entry, (entry,))]
        roots.extend(rel for rel in spec.get("optional_roots", ()) if read(rel) is not None)
        registry = spec.get("judge_registry")
        own_roots = roots + ([registry] if registry else [])
        if not any(read(rel) is not None for rel in own_roots):
            continue
        judge_roots = [registered_judges(registry)] if registry and read(registry) is not None else []
        judge_roots.extend((rel, symbols) for rel, symbols in
                           (*spec.get("shared_symbol_roots", ()), *spec.get("optional_symbol_roots", ()))
                           if read(rel) is not None)
        roots.extend(spec.get("shared_roots", ()))
        all_roots = roots + [rel for rel, _ in judge_roots] + ([registry] if registry else [])
        existing = [rel for rel in all_roots if read(rel) is not None]
        if len(existing) != len(all_roots):
            raise ValueError(f"Incomplete criterion {name}: {sorted(set(all_roots) - set(existing))}")
        scoped = spec.get("import_scope") == "symbols"
        seen, hashed = set(), set()
        todo = [(rel, None) for rel in roots] + judge_roots
        receipts = {}
        while todo:
            rel, symbols = todo.pop()
            if (rel, symbols) in seen:
                continue
            seen.add((rel, symbols))
            if rel in provenance:
                raise ValueError(f"Provenance import on judging path: {name} -> {rel}")
            if read(rel) is None:
                raise ValueError(f"Missing dependency of {name}: {rel}")
            if rel not in allow or rel in all_roots:
                hashed.add(rel)
            if rel.endswith((".yaml", ".yml")):
                todo.extend((dep, None) for dep in data_modules(rel))
            elif rel.endswith(".py"):
                deps, data, stopped, forbidden = edges(rel, symbols, scoped)
                for receipt in stopped:
                    receipts[(receipt["module"], receipt["scope"])] = receipt
                todo.extend((dep, None) for dep in data)
                if rel not in allow or rel in all_roots:
                    if forbidden:
                        raise ValueError(f"Provenance import on judging path: {name}: {rel} -> {sorted(forbidden)}")
                    todo.extend(deps)
        rels = sorted(hashed)
        h = hashlib.sha256()
        for rel in rels:
            h.update(rel.encode("utf-8"))
            h.update(_semantic_cached(read(rel), Path(rel).suffix.lower()))
        criteria[name] = h.hexdigest()[:16]
        dependencies[name] = rels
        boundary_receipts[name] = [receipts[key] for key in sorted(receipts)]
    total = hashlib.sha256(json.dumps(sorted(criteria.items()), ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()[:16]
    return {"criteria": criteria, "total_sha16": total, "dependencies": dependencies,
            "dispatch_boundaries": boundary_receipts,
            "source_commit": commit,
            "source_overrides": {rel: hashlib.sha256(blob).hexdigest() for rel, blob in overrides.items()}}


def compare_criteria(before: dict, after: dict, names=None) -> dict:
    """Compare selected columns; missing identities are explicitly incomparable."""
    old, new = before.get("criteria", before), after.get("criteria", after)
    names = sorted(set(names) if names is not None else set(old) | set(new))
    missing = [name for name in names if not old.get(name) or not new.get(name)]
    changed = [name for name in names if name not in missing and old[name] != new[name]]
    unchanged = [name for name in names if name not in missing and name not in changed]
    return {"comparable": not missing and not changed, "changed": changed,
            "missing": missing, "unchanged": unchanged}


@lru_cache(maxsize=1)
def current_fingerprints() -> dict:
    """One code version per process; historical/inspection calls remain uncached."""
    return criterion_fingerprints()


def criteria_used(row: dict) -> list[str]:
    return [name for name, spec in CRITERIA.items()
            if any(fnmatch.fnmatchcase(key, pattern) for key in row for pattern in spec["fields"])]


def stamp_row(row: dict, snapshot: dict | None = None, *, names=None) -> None:
    snapshot = current_fingerprints() if snapshot is None else snapshot
    names = criteria_used(row) if names is None else names
    row["judging_criteria"] = {name: snapshot["criteria"][name] for name in names}
    row["judging_total_sha16"] = snapshot["total_sha16"]


def recorded_criteria(row: dict) -> dict | None:
    """Read live rows (solve provenance) and offline/recomputed top-level stamps."""
    if "judging_criteria" in row:
        return row["judging_criteria"]
    return (row.get("world_ref") or {}).get("judging_criteria")


def merge_recorded_criteria(row: dict, updates: dict) -> None:
    """Stamp only newly measured columns; an unknown overlay stays unversioned."""
    stamps = dict(recorded_criteria(row) or {})
    for name, value in updates.items():
        if value:
            stamps[name] = value
        else:
            stamps.pop(name, None)
    row["judging_criteria"] = stamps
    row["judging_total_sha16"] = hashlib.sha256(json.dumps(
        sorted(stamps.items()), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:16]


def batch_fingerprint_fields(rows: list[dict]) -> dict:
    """Summarize measured versions without assigning current identities to old rows."""
    vintages = {name: sorted({recorded_criteria(row)[name] for row in rows
                             if name in (recorded_criteria(row) or {})})
                for name in sorted({name for row in rows for name in (recorded_criteria(row) or {})})}
    totals = sorted({row.get("judging_total_sha16") or (row.get("world_ref") or {}).get("judging_total_sha16")
                     for row in rows if row.get("judging_total_sha16") or
                     (row.get("world_ref") or {}).get("judging_total_sha16")})
    return {"judging_criteria_vintages": vintages,
            "judging_criteria": {name: values[0] for name, values in vintages.items() if len(values) == 1},
            "judging_total_vintages": totals,
            "judging_total_sha16": totals[0] if len(totals) == 1 else None}


def row_comparison(row: dict, current: dict, names=None) -> dict:
    names = criteria_used(row) if names is None else names
    if not names:
        names = ["<unknown-scoring-columns>"]
    return compare_criteria(recorded_criteria(row) or {}, current, names)


def criteria_provenance(rows: list[dict], *, names=None) -> dict:
    current = current_fingerprints()
    comparisons = [row_comparison(row, current, names) for row in rows]
    changed = sorted({n for result in comparisons for n in result["changed"]})
    missing = sorted({n for result in comparisons for n in result["missing"]})
    return {"changed_criteria": changed, "missing_criteria": missing,
            "stale": bool(changed or missing),
            "why": (["Changed criteria: " + ", ".join(changed)] if changed else []) +
                   (["Unrecorded criteria: " + ", ".join(missing)] if missing else []),
            "judging_total_current": current["total_sha16"]}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", help="Compute from this commit's Git objects")
    parser.add_argument("--batch", type=Path, help="Inspect/backfill using each row's recorded source commit")
    parser.add_argument("--write", action="store_true", help="Persist historical identities; never score or replace answers")
    parser.add_argument("--changed-file", help="List criteria whose dependency closure contains this source path")
    parser.add_argument("--criterion", action="append", help="Compare only this criterion (repeatable)")
    args = parser.parse_args(argv)
    if args.write and not args.batch:
        parser.error("--write requires --batch")
    current = criterion_fingerprints(args.rev)
    if args.batch:
        meta_path, eval_path = args.batch / "batch.json", args.batch / "eval.jsonl"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError("No evaluation rows; no historical judging identities to backfill")
        comparisons, historical = [], {}
        for row in rows:
            before = recorded_criteria(row)
            if before is None:
                rev = args.rev or row.get("haenv_git_sha") or (row.get("world_ref") or {}).get("haenv_git_sha") or meta.get("haenv_git_sha")
                if not rev or "+dirty" in rev:
                    raise ValueError("Missing/dirty source commit; historical scoring code cannot be reconstructed")
                if rev not in historical:
                    historical[rev] = criterion_fingerprints(rev)
                    legacy = anchor.fingerprint_at(rev, anchor._segment_at(rev, "JUDGING"))
                    historical[rev]["legacy_judging_sha16"] = legacy
                old = historical[rev]
                if row.get("judging_sha16") != old["legacy_judging_sha16"]:
                    raise ValueError("Recorded judging_sha16 does not reproduce at the source commit")
                for name in CRITERIA:
                    if name.endswith("_JUDGING"):
                        field = name.lower() + "_sha16"
                        if row.get(field):
                            expected = anchor.fingerprint_at(rev, anchor._segment_at(rev, name))
                            if row[field] != expected:
                                raise ValueError(f"Recorded {field} does not reproduce at the source commit")
                stamp_row(row, old, names=sorted(old["criteria"]))
                before = row["judging_criteria"]
            elif args.write:
                row.setdefault("judging_criteria", dict(before))
            comparisons.append(compare_criteria(before, current, args.criterion or criteria_used(row)))
        result = {"current": current, "historical": historical, "rows": len(rows),
                  "changed": sorted({name for c in comparisons for name in c["changed"]}),
                  "missing": sorted({name for c in comparisons for name in c["missing"]}),
                  "comparable": all(c["comparable"] for c in comparisons)}
        if args.write:
            meta["judging_criteria_backfill"] = {rev: {k: v for k, v in snap.items() if k != "dependencies"}
                                               for rev, snap in historical.items()}
            meta.update(batch_fingerprint_fields(rows))
            eval_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.changed_file:
        result = {"file": args.changed_file,
                  "criteria": [name for name, deps in current["dependencies"].items() if args.changed_file in deps]}
    else:
        result = current
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
