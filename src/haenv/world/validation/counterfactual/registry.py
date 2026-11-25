"""Versioned premise declarations and production condition drift detection."""
from copy import deepcopy
from functools import lru_cache

from haenv.resources.regpath import load_registry
from .trace import atoms_of


@lru_cache(maxsize=1)
def load():
    data = load_registry("counterfactual_premises.yaml")
    if data.get("version") != 1 or not data.get("source"):
        raise ValueError("invalid premise registry version/source")
    for pack, entry in data["packs"].items():
        ids = set()
        for row in entry["premises"]:
            if row["id"] in ids or not row["id"].startswith(pack + "."):
                raise ValueError("duplicate or unqualified premise id")
            ids.add(row["id"])
            for key in ("label", "reads", "visible", "operators", "source"):
                if key not in row or (key in ("label", "source") and not row[key]):
                    raise ValueError(f"premise missing {key}")
        for value in entry["atoms"].values():
            if isinstance(value, str):
                if value not in ids:
                    raise ValueError(f"unknown premise {value}")
            elif value.get("kind", "premise") not in ("premise", "table", "validity", "internal"):
                raise ValueError("unknown atom kind")
            elif value.get("kind", "premise") == "premise" and value.get("id") not in ids:
                raise ValueError("unknown premise atom")
    return data


def premises(pack):
    return deepcopy(load()["packs"][pack]["premises"])


def atom_map(pack):
    return deepcopy(load()["packs"][pack]["atoms"])


def premise_id(value):
    return value if isinstance(value, str) else value.get("id") if value.get("kind", "premise") == "premise" else None


def rule_functions(pack):
    from . import adapter
    return adapter(pack).rule().sources


def check_complete(pack):
    actual = {text for func in rule_functions(pack) for text in atoms_of(func)}
    declared = set(atom_map(pack))
    return ([{"kind": "missing", "text": text} for text in sorted(actual - declared)]
            + [{"kind": "stale", "text": text} for text in sorted(declared - actual)])
