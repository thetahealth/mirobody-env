"""Optional clinically matched histories; disabled leaves the earlier pipeline untouched."""
from __future__ import annotations
import os
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached


def enabled(plan=None):
    value = (plan or {}).get("difficulty_r2", os.environ.get("HAENV_DIFFICULTY_R2", "off"))
    if value not in (True, False, "on", "off"):
        raise ValueError("difficulty_r2 must be on or off")
    return value is True or value == "on"


@registry_cached("difficulty_r2.yaml")
def table():
    return load_registry("difficulty_r2.yaml")


def config(pack):
    return table()["difficulty_r2"][pack]


def register(gold, section, day, acts, *, recent_lower, deletion="different"):
    """Common truth-side annotation; neither roles nor this manifest reach the solver."""
    refs = [{"section": section, "day": int(day)}]
    gold["look_alike"] = {"records": [{**refs[0], "acts": bool(acts)}], "recent_lower": int(recent_lower)}
    gold["history_dependency"] = {"active": bool(acts), "recent_lower": int(recent_lower),
                                  "records": refs, "deletion": deletion}
