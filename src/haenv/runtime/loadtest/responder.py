"""What the fake provider answers to one request.

* A request with a JSON-schema `response_format` (a judge) gets an instance of that schema:
  the first enum value for a label, a short string for free text, an empty list for arrays.
* Any other request gets a recorded answer from the replay library. The answer kind is the
  library group whose answer keys the prompt names most completely (a prompt spells out the
  JSON it wants); groups that tie are drawn in proportion to how often they were recorded,
  so a multi-round protocol (query rounds vs. the final answer) keeps its measured mix.
  Answers of the requested model are preferred; any model's answer of that kind is next.
* No library group matches: the answer is `{}`, which the client rejects as unparseable.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

from haenv.runtime.loadtest.profile import JUDGE_KIND

#: Usage of a synthesized judge answer when the library has no judge entries.
DEFAULT_JUDGE_USAGE = {"prompt": 6000, "completion": 1500, "cached": 0, "reasoning": 1000}


@dataclass
class Answer:
    kind: str
    text: str
    usage: dict


def schema_instance(schema: dict, defs: dict | None = None):
    """A minimal instance of a JSON schema (objects, enums, strings, arrays, `$ref`)."""
    defs = defs if defs is not None else schema.get("$defs") or {}
    if "$ref" in schema:
        return schema_instance(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties") or {}
        return {k: schema_instance(props.get(k) or {"type": "string"}, defs)
                for k in schema.get("required") or list(props)}
    if kind == "array":
        return []
    if kind in ("integer", "number"):
        return 0
    if kind == "boolean":
        return False
    return "rehearsal"


class Responder:
    def __init__(self, entries: list[dict], seed: int = 0):
        self.rng = random.Random(seed)
        self.groups: dict[tuple, list[dict]] = defaultdict(list)
        self.judge_usage: list[dict] = []
        for e in entries:
            if e["kind"] == JUDGE_KIND:
                self.judge_usage.append(e["usage"])
            else:
                self.groups[(e["kind"], tuple(e["keys"]))].append(e)

    def answer(self, model: str, prompt: str, response_format: dict | None) -> Answer:
        schema = ((response_format or {}).get("json_schema") or {}).get("schema")
        if schema is not None:
            usage = (self.rng.choice(self.judge_usage) if self.judge_usage
                     else DEFAULT_JUDGE_USAGE)
            return Answer(JUDGE_KIND, json.dumps(schema_instance(schema), ensure_ascii=False),
                          dict(usage))
        group = self._group(model, prompt)
        if group is None:
            return Answer("unknown", "{}", {"prompt": len(prompt) // 3, "completion": 2,
                                            "cached": 0, "reasoning": 0})
        own = [e for e in self.groups[group] if e["model"] == model]
        entry = self.rng.choice(own or self.groups[group])
        return Answer(group[0], entry["text"], dict(entry["usage"]))

    def _group(self, model: str, prompt: str):
        def score(keys) -> float:
            return sum(f'"{k}"' in prompt or f"{k}:" in prompt for k in keys) / len(keys)
        scored = [(score(g[1]), g) for g in self.groups if g[1]]
        best = max((s for s, _ in scored), default=0.0)
        if best <= 0:
            return None
        tied = [g for s, g in scored if s == best]
        weights = []
        for g in tied:
            own = [e["weight"] for e in self.groups[g] if e["model"] == model]
            weights.append(sum(own) if own else sum(e["weight"] for e in self.groups[g]) * 1e-3)
        return self.rng.choices(tied, weights=weights)[0]
