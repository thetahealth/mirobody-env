"""Evidence sources for knockout and declared premise roles.

SYNTHETIC data, evaluation only, not medical advice.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FactOutcome:
    fact: str
    group: str
    outside_today: bool | None
    outcome: Literal["flip", "same", "undetermined", "error"]
    source: str = "knockout"
    present: bool = True
    per_fact: bool = True
    error: str | None = None


class KnockoutSource:
    name = "knockout"

    def __init__(self, pack):
        self.pack = pack

    def read(self, item, full_answer):
        if getattr(self.pack, "NAME", None) == "p2":
            return self.pack.knockout(item.sp, full_answer, accepted=item.gold["set"])
        return self.pack.knockout(item.sp, full_answer)


class PremiseRoleSource:
    name = "premise"

    def __init__(self, pack):
        self.pack = pack

    def read(self, item, full_answer=None):
        roles = item.gold.get("premises")
        if roles is None:
            return None
        outside = getattr(self.pack, "premise_outside_today", lambda _: None)
        result = []
        for key, role in sorted(roles.items()):
            if role not in {"acting", "overridden", "context", "guard", "unread"}:
                raise ValueError(f"unknown premise role: {role}")
            result.append(FactOutcome(key, key.rsplit(".", 1)[0], outside(key),
                                      "flip" if role == "acting" else "same", "premise"))
        return result


class AutoSource:
    def __init__(self, pack):
        self.pack = pack

    def for_item(self, item):
        cls = PremiseRoleSource if item.gold.get("premises") is not None else KnockoutSource
        return cls(self.pack)


def select(pack, mode="auto"):
    return {"auto": AutoSource, "knockout": KnockoutSource, "premise": PremiseRoleSource}[mode](pack)
