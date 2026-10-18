"""Provider profile and replay library, measured from finished batches.

A **profile** (YAML) describes how a provider behaves, per model as named on the wire
(`openai/gpt-6-sol`, `gemini-3.7-flash`, `qwen3.7-flash`):

* `service_s` -- service-time samples in seconds, per answer kind (`judge`, a job id, or
  `*` for any kind); the fake provider draws one per request;
* `errors` -- per-attempt probabilities of `rate_limited` (HTTP 429 before generation),
  `extra_data` (a second JSON object after the answer) and `disconnect` (the connection
  closes after the service time, no response);
* `served_by` -- the upstream named in OpenRouter responses.

It also carries the OpenRouter `catalog` entries (pricing, endpoints) the client verifies
before a paid run, and per-backend `max_concurrency` (requests in flight beyond it get 429).

The **library** (JSONL) holds recorded provider responses, one per line, with the answer
kind and the top-level keys of the answer JSON; the responder replays them.

`build` measures both from batch directories (see its docstring). Client-side latencies of a
loaded run are not used as service times: they include the wait for a paid slot.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Answer kind of a judge request (`response_format` named `haenv_semantic_verdicts`).
JUDGE_KIND = "judge"
#: Kind used when no sample of the request's own kind exists.
ANY_KIND = "*"
#: Error classes the fake provider injects, matched against recorded attempt errors.
ERROR_PATTERNS = {
    "rate_limited": re.compile(r"\b429\b|rate.?limit|RejectedBeforeGeneration", re.I),
    "extra_data": re.compile(r"Extra data", re.I),
    "disconnect": re.compile(r"timed? ?out|stall|IncompleteRead|RemoteDisconnected|"
                             r"ConnectionReset|截断", re.I),
}
#: Recorded responses kept per (model, kind) in the library.
LIBRARY_PER_KIND = 40


@dataclass
class ModelProfile:
    service_s: dict[str, list[float]] = field(default_factory=dict)
    errors: dict[str, float] = field(default_factory=dict)
    served_by: str | None = None

    def draw_service_s(self, kind: str, rng: random.Random) -> float:
        samples = self.service_s.get(kind) or self.service_s.get(ANY_KIND) or [1.0]
        return rng.choice(samples)


@dataclass
class Profile:
    models: dict[str, ModelProfile] = field(default_factory=dict)
    catalog: dict[str, dict] = field(default_factory=dict)
    max_concurrency: dict[str, int] = field(default_factory=dict)

    def model(self, wire_model: str) -> ModelProfile:
        return self.models.get(wire_model) or self.models.get(ANY_KIND) or ModelProfile()

    @classmethod
    def load(cls, path: Path) -> "Profile":
        from haenv.resources.yamlcache import load_yaml
        doc = load_yaml(Path(path)) or {}
        return cls(models={k: ModelProfile(**v) for k, v in (doc.get("models") or {}).items()},
                   catalog=doc.get("catalog") or {},
                   max_concurrency=doc.get("max_concurrency") or {})

    def dump(self, path: Path, *, header: str = "") -> None:
        doc = {"models": {k: {"service_s": v.service_s, "errors": v.errors,
                              "served_by": v.served_by} for k, v in sorted(self.models.items())},
               "max_concurrency": self.max_concurrency, "catalog": self.catalog}
        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120)
        Path(path).write_text(header + text, encoding="utf-8")


def answer_keys(text: str) -> list[str]:
    """Top-level keys of the first JSON object in `text` (empty when there is none)."""
    start = text.find("{") if isinstance(text, str) else -1
    if start < 0:
        return []
    try:
        doc, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return []
    return sorted(doc) if isinstance(doc, dict) else []


def response_text(response: dict) -> str:
    """Answer text of an OpenAI-compatible or Google response."""
    if "candidates" in response:
        parts = ((response["candidates"] or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)
    return (((response.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""


def _jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _batch_job(batch: Path) -> str:
    return json.loads((batch / "batch.json").read_text())["job_id"]


def _solver_receipts(batch: Path):
    for f in sorted((batch / "solver-accounting" / "receipts").glob("*.json")):
        if f.name.endswith((".failed.json", ".inflight.json")):
            continue
        try:
            yield json.loads(f.read_text())
        except ValueError:
            continue


def _judge_receipts(batch: Path):
    for f in sorted(batch.glob("semantic/*/receipts/*.json")):
        if f.name.endswith((".failed.json", ".inflight.json")):
            continue
        try:
            yield json.loads(f.read_text())
        except ValueError:
            continue


def usage_tokens(response: dict) -> dict:
    """Token counts of a recorded response, provider-neutral."""
    if "usageMetadata" in response:
        um = response["usageMetadata"] or {}
        return {"prompt": um.get("promptTokenCount") or 0,
                "completion": (um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0),
                "cached": um.get("cachedContentTokenCount") or 0,
                "reasoning": um.get("thoughtsTokenCount") or 0}
    usage = response.get("usage") or {}
    return {"prompt": usage.get("prompt_tokens") or 0,
            "completion": usage.get("completion_tokens") or 0,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}


def build(batch_dirs: list[Path], *, service_batches: list[Path] | None = None,
          generation_samples: Path | None = None,
          service_like: dict[str, tuple[str, float]] | None = None,
          max_concurrency: dict[str, int] | None = None,
          seed: int = 0) -> tuple[Profile, list[dict]]:
    """Measure a profile and a replay library.

    `batch_dirs` give answers, prices, served upstreams, error rates and judge service times;
    `service_batches` (runs with no wait for a paid slot) and `generation_samples` (OpenRouter
    `generation_time`, `{model: [{generation_time, pack}]}`) give solver service times;
    `service_like` fills a model with none from another model's samples times a factor.
    """
    rng = random.Random(seed)
    wire_of: dict[str, str] = {}                       # config model name -> wire model id
    served: dict[str, str] = {}
    catalog: dict[str, dict] = {}
    library: dict[tuple, list[dict]] = defaultdict(list)
    latency: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    attempts: dict[str, int] = defaultdict(int)
    errors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stamps: dict[str, str] = {}                        # batch dir name -> job id

    for batch in batch_dirs:
        job = _batch_job(batch)
        stamps[batch.name] = job
        manifest = batch / "solver-accounting" / "manifest.json"
        if manifest.is_file():
            for meta in (json.loads(manifest.read_text()).get("price_metadata") or {}).values():
                if isinstance(meta, dict) and meta.get("id"):
                    catalog[meta["id"]] = meta
                tariff = (meta or {}).get("native_tariff") if isinstance(meta, dict) else None
                if tariff:
                    catalog[f"{tariff['backend']}:{tariff['model']}"] = {
                        "inputTokenLimit": tariff["max_input_tokens"],
                        "outputTokenLimit": tariff["max_output_tokens"]}
        for meta_file in batch.glob("semantic/*/price-metadata.json"):
            meta = json.loads(meta_file.read_text())
            if meta.get("id"):
                catalog[meta["id"]] = meta
        for receipt in _solver_receipts(batch):
            route, cell = receipt.get("route") or {}, receipt.get("cell") or {}
            response = receipt.get("response")
            if not isinstance(response, dict) or not route.get("model"):
                continue
            wire = route["model"]
            wire_of[cell.get("solver")] = wire
            if (receipt.get("served") or {}).get("provider"):
                served[wire] = receipt["served"]["provider"]
            text = response_text(response)
            keys = answer_keys(text)
            if keys:
                library[(wire, job, tuple(keys))].append(
                    {"model": wire, "kind": job, "keys": keys, "text": text,
                     "usage": usage_tokens(response)})
        for receipt in _judge_receipts(batch):
            response, transport = receipt.get("response") or {}, receipt.get("transport") or {}
            wire = response.get("model") or (receipt.get("request") or {}).get("model")
            if wire and transport.get("seconds") is not None:
                latency[wire][JUDGE_KIND].append(float(transport["seconds"]))
                served.setdefault(wire, response.get("provider"))
                library[(wire, JUDGE_KIND, ())].append(
                    {"model": wire, "kind": JUDGE_KIND, "keys": [], "text": "",
                     "usage": usage_tokens(response)})
        rows = list(_jsonl(batch / "responses.jsonl")) + list(_jsonl(batch / "gated_rounds.jsonl"))
        for row in rows:
            solver = row.get("solver")
            if row.get("rounds") is None and (row.get("raw") or row.get("n_chars")):
                attempts[solver] += 1                 # the successful attempt
            failed = row.get("failed_attempts")
            for attempt in failed if isinstance(failed, list) else []:
                attempts[solver] += 1
                for name, pattern in ERROR_PATTERNS.items():
                    if pattern.search(str(attempt.get("error") or "")):
                        errors[solver][name] += 1
                        break

    for batch in service_batches or []:
        # A receipt's lock is created when its request starts and the receipt is written when
        # the answer arrives; in a run with no wait for a paid slot the gap is the service time.
        job = _batch_job(batch)
        for path in (batch / "solver-accounting" / "receipts").glob("*.json"):
            lock = path.with_suffix(".lock")
            if path.name.endswith((".failed.json", ".inflight.json")) or not lock.exists():
                continue
            solver = (json.loads(path.read_text()).get("cell") or {}).get("solver")
            latency[solver][job].append(path.stat().st_mtime - lock.stat().st_mtime)

    if generation_samples is not None and Path(generation_samples).is_file():
        for solver, samples in json.loads(Path(generation_samples).read_text()).items():
            for s in samples:
                if s.get("generation_time") is not None:
                    kind = stamps.get(str(s.get("pack")), ANY_KIND)
                    latency[solver][kind].append(s["generation_time"] / 1000.0)

    models: dict[str, ModelProfile] = {}
    for name, kinds in latency.items():
        wire = wire_of.get(name, name)
        mp = models.setdefault(wire, ModelProfile(served_by=served.get(wire)))
        for kind, values in kinds.items():
            mp.service_s.setdefault(kind, []).extend(round(v, 2) for v in values)
        n = attempts.get(name, 0)
        if n:
            mp.errors = {k: round(errors[name].get(k, 0) / n, 5) for k in ERROR_PATTERNS}
    for wire, (like, factor) in (service_like or {}).items():
        source = models.get(like)
        if source is not None:
            models.setdefault(wire, ModelProfile(served_by=served.get(wire))).service_s = {
                k: [round(v * factor, 2) for v in vs] for k, vs in source.service_s.items()
                if k != JUDGE_KIND}
    for mp in models.values():
        if not mp.service_s.get(ANY_KIND):
            mp.service_s[ANY_KIND] = [v for k, vs in mp.service_s.items()
                                      if k not in (JUDGE_KIND, ANY_KIND) for v in vs]
    entries: list[dict] = []
    for group in library.values():
        kept = rng.sample(group, min(LIBRARY_PER_KIND, len(group)))
        entries.extend({**e, "weight": round(len(group) / len(kept), 3)} for e in kept)
    return Profile(models=models, catalog=catalog,
                   max_concurrency=dict(max_concurrency or {})), entries
