"""Built-batch loading and the adapter contract for report-only audits."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Hashable, Iterable, Protocol

from .trace import RuleTrace


@dataclass(frozen=True)
class Item:
    case_id: str
    raw: object
    sp: dict
    language: str | None = None


@dataclass(frozen=True)
class Operator:
    id: str
    target: str | Callable | None
    tier: str
    fn: Callable

    def target_for(self, raw):
        return self.target(raw) if callable(self.target) else self.target


class InvalidTwin(ValueError):
    """A requested edit cannot be realized through production derivation."""


class Adapter(Protocol):
    pack: str
    def items(self, batch) -> Iterable[Item]: ...
    def rule(self) -> RuleTrace: ...
    def rule_args(self, item) -> tuple: ...
    def decision(self, result) -> Hashable: ...
    def scored(self, raw_like) -> list[str] | None: ...
    def visible_answer(self, sp): ...
    def operators(self) -> dict[str, Operator]: ...
    def twin_payload(self, item, twin) -> dict: ...
    def visible_blocks(self, sp) -> dict: ...


def language(batch_dir):
    """Require an explicit language marker; never infer it from directory names."""
    marker = json.loads((Path(batch_dir) / "batch.json").read_text()).get("language")
    if marker not in ("zh", "en"):
        raise ValueError(f"unknown batch language: {marker!r}")
    return marker


def load_batch(batch_dir, *, _m2_loaded=False):
    from haenv.app.artifacts.store import load_cases
    path = Path(batch_dir)
    lang = language(path)
    if not _m2_loaded and len(path.parents) > 3 and (path.parents[3] / "m2-pack1.job.yaml").is_file():
        from .m2 import load_batch as load_m2
        return load_m2(path)
    cases = load_cases(path / "cases.jsonl")
    payloads = {}
    for line in (path / "payloads.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["case_id"] in payloads:
            raise ValueError("duplicate payload case id")
        payloads[row["case_id"]] = row["sp"]
    if set(cases) != set(payloads):
        raise ValueError("case and payload ids differ")
    return [Item(cid, cases[cid], payloads[cid], lang) for cid in sorted(cases)]
