"""Read a revision's declarations as data, without executing its code.

Legacy revisions declare segments literally in tools/make_freeze.py. New
revisions record PROVENANCE_DECLARATIONS there, naming one supported entry.
Missing or malformed declared entries are errors, never a reason to use the
current installation's rules.
"""
from __future__ import annotations

import ast
import json
from pathlib import PurePosixPath

DECLARATION_ENTRIES = frozenset({
    "haenv/provenance/declarations.py", "src/haenv/provenance/declarations.py",
})


def literal_tables(source: bytes, names: set[str]) -> dict:
    """The declaration language: literals, names, concatenation, frozen sets.

    Imports, functions, calls outside that language and unrelated statements
    are never evaluated. Both regular and annotated declarations are accepted.
    """
    values = {}

    def literal(node):
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Dict):
            out = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    out.update(literal(value))
                else:
                    out[literal(key)] = literal(value)
            return out
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)](literal(n) for n in node.elts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "frozenset" and len(node.args) <= 1 and not node.keywords):
            return frozenset(literal(node.args[0])) if node.args else frozenset()
        # Early move certificates expanded groups of identical reasons this
        # way. Interpret the data operation; never call historical functions.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "dict"
                and node.func.attr == "fromkeys" and len(node.args) in {1, 2} and not node.keywords):
            return dict.fromkeys(literal(node.args[0]), literal(node.args[1]) if len(node.args) == 2 else None)
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError) as error:
            raise ValueError("Source declarations must be literal data") from error

    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id in names:
            values[targets[0].id] = literal(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            if not isinstance(node.op, ast.Add) or node.target.id not in values:
                raise ValueError("Unsupported source declaration update")
            values[node.target.id] += literal(node.value)
    return values


SEGMENT_NAMES = {
    "GENERATION", "JUDGING", "INFRA", "MANAGEMENT_JUDGING", "M2_JUDGING",
    "PACK2_WORLD", "PACK2_JUDGING", "P2_WORLD", "P2_JUDGING", "P3_WORLD", "P3_JUDGING",
    "P4_WORLD", "P4_JUDGING", "PROTOCOL_WORLD", "INFRA_PATH_MOVES", "INFRA_FORBIDDEN_IMPORTS",
    "JUDGING_SHA_EQUIVALENCE", "JUDGING_SHA_MOVES", "JUDGING_IMPORT_ALLOW", "PROVENANCE_MODULES",
    "REGISTRY_YAML_OUT_OF_SCOPE", "PENDING_ANCHOR_REVISION", "PENDING_ANCHOR",
    "ALLOW_SYMBOL_PIN", "PAYLOAD_YAML_ALLOW", "SOURCE_LAYOUT", "CRITERION_DECLARATIONS",
}


def declarations_at(read) -> dict:
    """Read the selected revision's explicit schema; never substitute current rules."""
    header = read("registry/source_layout.json")
    declared_layout = None
    if header is not None:
        try:
            description = json.loads(header)
        except (ValueError, TypeError) as error:
            raise ValueError("Malformed source layout manifest") from error
        if (not isinstance(description, dict)
                or set(description) != {"schema", "declarations", "source_layout"}
                or type(description.get("schema")) is not int or description["schema"] != 1
                or description.get("source_layout") not in {"flat", "src"}):
            raise ValueError("Unsupported source layout manifest")
        entry = description["declarations"]
        declared_layout = description["source_layout"]
        if entry not in DECLARATION_ENTRIES:
            raise ValueError(f"Unsupported provenance declaration entry: {entry!r}")
        raw = read(entry)
        if raw is None:
            raise ValueError(f"Missing declared source entry: {entry}")
        modern = True
    else:
        # Revisions predating the manifest use their fixed historical entry.
        entry = "tools/make_freeze.py"
        raw = read(entry)
        if raw is None:
            raise ValueError(f"Missing {entry}; cannot establish the source boundary")
        pointer = literal_tables(raw, {"PROVENANCE_DECLARATIONS"})
        modern = "PROVENANCE_DECLARATIONS" in pointer
        if modern:
            entry = pointer["PROVENANCE_DECLARATIONS"]
            if entry not in DECLARATION_ENTRIES:
                raise ValueError(f"Unsupported provenance declaration entry: {entry!r}")
            raw = read(entry)
            if raw is None:
                raise ValueError(f"Missing declared source entry: {entry}")
    values = literal_tables(raw, SEGMENT_NAMES)
    if modern and "SOURCE_LAYOUT" not in values:
        raise ValueError(f"Missing SOURCE_LAYOUT in declared source entry: {entry}")
    layout = values.get("SOURCE_LAYOUT", "flat")
    if layout not in {"flat", "src"}:
        raise ValueError(f"Unsupported declared source layout: {layout!r}")
    if declared_layout is not None and declared_layout != layout:
        raise ValueError(f"Manifest and source layout disagree: {declared_layout}, {layout}")
    if modern and entry.startswith("src/") != (layout == "src"):
        raise ValueError(f"Declaration entry and source layout disagree: {entry}, {layout}")
    values["SOURCE_LAYOUT"] = layout
    values["_entry"] = entry
    return values


def revision_path(label: str, declarations: dict) -> str:
    """Physical Git path of a canonical label under this revision's layout."""
    path = PurePosixPath(label)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid source label: {label!r}")
    if declarations["SOURCE_LAYOUT"] == "src" and path.parts[0] in {"haenv", "haenv_kernel", "verifier_core"}:
        return "src/" + label
    return label


def contract_tables_at(read, declarations: dict) -> tuple[dict, dict]:
    if declarations.get("_entry") != "tools/make_freeze.py" and "CRITERION_DECLARATIONS" not in declarations:
        raise ValueError("Missing explicit criterion declaration entry")
    entry = declarations.get("CRITERION_DECLARATIONS", "haenv/judging.py")
    if entry not in {"haenv/judging.py", "haenv/provenance/criteria.py"}:
        raise ValueError(f"Unsupported criterion declaration entry: {entry!r}")
    raw = read(revision_path(entry, declarations))
    if raw is None:
        raise ValueError(f"Missing historical criterion declarations: {entry}")
    return contract_tables(raw)


def contract_tables(source: bytes) -> tuple[dict, dict]:
    values = literal_tables(source, {"PACK_SCORING", "DISPATCH_BOUNDARIES", "CRITERIA"})
    if not values.get("CRITERIA"):
        raise ValueError("Historical source has no complete criterion declarations")
    return values["CRITERIA"], values.get("DISPATCH_BOUNDARIES", {})
