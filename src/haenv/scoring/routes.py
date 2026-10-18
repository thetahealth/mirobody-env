"""Equivalent judge routes: credentials, dated tariffs, paid receipts and admission.

This execution module is only reached when a route pool is explicitly configured.
Judge prompts, parsing and scores remain owned by the sealed judging policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import urllib.request

from haenv.runtime.judge_scheduler import RouteLanes
from haenv.runtime.judge_scheduler import RoutesUnavailable
from haenv.runtime.completion import AccountedCompletion, never_sent
from haenv.runtime.budget import BudgetExceeded
from haenv.app.settings import JudgeRouteSpec

log = logging.getLogger("haenv.judge_routes")
_expired_warned: set = set()


def model_identity(model):
    """Normalize only vendor prefixes and ISO/compact snapshot dates, never fuzzy aliases."""
    model = str(model).rsplit("/", 1)[-1]
    match = re.fullmatch(r"(.+)-(\d{4})-?(\d{2})-?(\d{2})", model)
    if not match:
        return model, None
    name, year, month, day = match.groups()
    return name, date.fromisoformat(f"{year}-{month}-{day}").isoformat()


def configured_routes(cfg, policy):
    routes = {name: JudgeRouteSpec.model_validate(spec) for name, spec in (cfg.get("judge_routes") or {}).items()}
    if not routes:
        raise ValueError("No judge_routes configured")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in routes):
        raise ValueError("Judge route names must contain only letters, digits, dot, dash or underscore")
    first = next(iter(routes.values()))
    identity = {"canonical_model": first.canonical_model, "snapshot": first.snapshot.isoformat()}
    if cfg.get("judge_identity"):
        from haenv.app.settings import JudgeIdentity
        identity = JudgeIdentity.model_validate(cfg["judge_identity"]).model_dump(mode="json")
    sealed_name, sealed_snapshot = model_identity(policy["judge"]["model_id"])
    if identity["canonical_model"] != sealed_name or sealed_snapshot and identity["snapshot"] != sealed_snapshot:
        raise ValueError("Judge identity differs from the sealed model")
    for route in routes.values():
        if {"canonical_model": route.canonical_model, "snapshot": route.snapshot.isoformat()} != identity:
            raise ValueError("Every judge route must declare the same canonical model and snapshot")
    return routes, identity


class SnapshotMismatch(ValueError):
    def __init__(self, expected, returned, *, catalog=None):
        super().__init__("Model evidence does not match the declared judge snapshot")
        self.snapshot_check = {"expected": expected, "returned": str(returned)[:160], "matched": False,
                               "snapshot_evidence": "catalog" if catalog is not None else "response"}
        if catalog is not None:
            self.snapshot_check["catalog"] = catalog


def check_catalog_snapshot(catalog, identity):
    """Require a dated canonical slug, never infer a snapshot from an undated alias."""
    slug = catalog.get("canonical_slug")
    try:
        actual = model_identity(slug) if isinstance(slug, str) else (None, None)
    except ValueError:
        actual = (None, None)
    if actual != (identity["canonical_model"], identity["snapshot"]):
        raise SnapshotMismatch(identity, slug, catalog=catalog)


def check_snapshot(response, identity, *, catalog=None):
    returned = response.get("model") or response.get("modelVersion") if isinstance(response, dict) else None
    try:
        actual = model_identity(returned) if isinstance(returned, str) else (None, None)
    except ValueError:
        actual = (None, None)
    if catalog is not None:
        check_catalog_snapshot(catalog, identity)
        # OpenRouter may echo its undated request alias. The catalog establishes
        # the date; a different response name or an explicitly wrong date still fails.
        matched = actual[0] == identity["canonical_model"] and actual[1] in (None, identity["snapshot"])
    else:
        matched = actual == (identity["canonical_model"], identity["snapshot"])
    if not matched:
        raise SnapshotMismatch(identity, returned, catalog=catalog)
    return {"expected": identity, "returned": returned, "matched": True,
            "snapshot_evidence": "catalog" if catalog is not None else "response",
            **({"catalog": catalog} if catalog is not None else {})}


@dataclass(frozen=True)
class DeclaredJudgePrices:
    model: str
    price: object
    expiry: str

    def validate_dispatch(self):
        from haenv.runtime.native_accounting import VALID_DAYS, utc_today
        age = (utc_today() - self.price.tariff_checked_on).days
        if age < 0:
            raise ValueError("Tariff verification date is in the future")
        if age >= VALID_DAYS:
            if self.expiry == "refuse":
                raise ValueError("Tariff verification expired")
            key = (self.model, self.price.source, self.price.tariff_checked_on)
            if key not in _expired_warned:
                _expired_warned.add(key)
                log.warning("Judge route tariff verification expired; using the declared tariff")

    def upper_bound(self, prompt, max_output_tokens):
        rates = [self.price, *self.price.prompt_tiers]
        inputs = [r for p in rates for r in (p.input_per_million_usd,
                  p.cached_input_per_million_usd, p.cache_creation_per_million_usd) if r is not None]
        outputs = [r for p in rates for r in (p.output_per_million_usd, p.reasoning_per_million_usd) if r is not None]
        output = max(Decimal(str(r)) for r in outputs)
        return ((len(prompt.encode()) + 2048) * max(Decimal(str(r)) for r in inputs)
                + max_output_tokens * output) / Decimal(1_000_000)

    def cost_receipt(self, response):
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("No token usage receipt")
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = prompt_details.get("cached_tokens", usage.get("cache_read_input_tokens", 0))
        created = prompt_details.get("cache_creation_tokens", usage.get("cache_creation_input_tokens", 0))
        if any(type(n) is not int or n < 0 for n in (prompt, completion, reasoning, cached, created)):
            raise ValueError("Invalid token usage receipt")
        if cached + created > prompt:
            raise ValueError("Cached input counts exceed the inclusive prompt count")
        if self.price.completion_includes_reasoning:
            if reasoning > completion:
                raise ValueError("Reasoning tokens exceed inclusive completion count")
            visible = completion - reasoning
        else:
            visible = completion
        p = self.price
        threshold = 0
        for tier in self.price.prompt_tiers:
            if prompt > tier.prompt_tokens_above:
                p, threshold = tier, tier.prompt_tokens_above
        rr = p.reasoning_per_million_usd if p.reasoning_per_million_usd is not None else p.output_per_million_usd
        cache_rate = p.cached_input_per_million_usd if p.cached_input_per_million_usd is not None else p.input_per_million_usd
        create_rate = p.cache_creation_per_million_usd if p.cache_creation_per_million_usd is not None else p.input_per_million_usd
        actual = ((prompt - cached - created) * Decimal(str(p.input_per_million_usd))
                  + cached * Decimal(str(cache_rate)) + created * Decimal(str(create_rate))
                  + visible * Decimal(str(p.output_per_million_usd))
                  + reasoning * Decimal(str(rr))) / Decimal(1_000_000)
        return {"kind": "declared_usage_receipt", "actual_usd": str(actual),
                "tokens": {"prompt": prompt, "completion": completion, "reasoning": reasoning,
                           "cached_input": cached, "cache_creation": created},
                "prompt_tier_above": threshold, "tariff": self.price.model_dump(mode="json")}


def route_environment(cfg, spec):
    from haenv.runtime.solvers import load_env_file
    env = dict(os.environ)
    env.update(load_env_file(cfg.get("env_file")))
    if spec.env_file:
        env.update(load_env_file(spec.env_file))
    return env


def route_endpoint(cfg, spec, env):
    from haenv.runtime.solvers import BACKENDS
    bdef = {**BACKENDS.get(spec.backend, {}), **(cfg.get("backends", {}).get(spec.backend) or {})}
    if spec.base_url_env:
        base = env.get(spec.base_url_env)
        if not base:
            raise ValueError("Judge route base URL environment variable is missing")
        endpoint = base.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions"
    else:
        endpoint = spec.url or bdef.get("url")
    # Revalidate expanded URLs too: a URL must never embed a key.
    JudgeRouteSpec.valid_url(endpoint)
    if not endpoint:
        raise ValueError("Judge route endpoint is not configured")
    return endpoint, bdef


def route_keys(spec, env, bdef):
    from haenv.runtime.solvers import _KeyPool
    prefix = spec.key_env_pool or (bdef.get("key_env_pool") if not spec.key_env else None)
    names = sorted((n for n in env if prefix and n.startswith(prefix) and n[len(prefix):].isdigit()),
                   key=lambda n: int(n[len(prefix):]))
    keys = [env[n] for n in names if env[n]]
    ordinals = [int(n[len(prefix):]) for n in names if env[n]]
    single = spec.key_env or bdef.get("key_env")
    if not keys and single and env.get(single):
        keys, ordinals = [env[single]], [0]
    return _KeyPool(keys, spec.backend, ordinals)


class JudgeRoute:
    def __init__(self, name, spec, cfg, policy, ledger, receipts, *, price_check=True):
        self.name, self.spec, self.cfg, self.policy = name, spec, cfg, policy
        env = route_environment(cfg, spec)
        self.endpoint, bdef = route_endpoint(cfg, spec, env)
        self.keys = route_keys(spec, env, bdef)
        self.ledger, self.receipts = ledger, receipts
        self.metadata = {"source": "not checked"}
        self.snapshot_catalog = None
        self.prices = None
        if price_check:
            self.preflight()

    def get_json(self, path):
        from haenv.runtime.transport import model_urlopen
        base = self.endpoint.rsplit("/chat/completions", 1)[0]
        request = urllib.request.Request(base + path,
            headers={"Authorization": f"Bearer {self.keys.current()}"})
        with model_urlopen(request, proxy=self.spec.proxy, timeout=30) as response:
            return json.loads(response.read())

    def preflight(self):
        if not self.keys.n_alive:
            raise ValueError("Judge route credentials are not configured")
        if self.spec.backend == "relay":
            prices = DeclaredJudgePrices(self.spec.model, self.spec.price,
                                         self.cfg.get("paid", {}).get("tariff_expiry", "warn"))
            prices.validate_dispatch()
            self.metadata = {"source": "configured tariff", "price": self.spec.price.model_dump(mode="json"),
                "tariff_check": {"checked_on": self.spec.price.tariff_checked_on.isoformat(), "expiry": prices.expiry}}
        elif self.spec.backend == "google":
            from haenv.runtime.native_accounting import NativePrices, configure, tariff_check
            configure(self.cfg)
            prices = NativePrices.verified(self.spec.model, "google")
            from dataclasses import asdict
            self.metadata = {"source": "native tariff", "price": asdict(prices), "tariff_check": tariff_check()}
        else:
            from haenv.runtime.pricing import PriceSchedule
            models = self.get_json("/models")["data"]
            matches = [m for m in models if m.get("id") == self.spec.model]
            if len(matches) != 1:
                raise ValueError("No unique exact-model price record")
            endpoint_path = f"/models/{self.spec.model}/endpoints"
            endpoint_data = self.get_json(endpoint_path)["data"]
            identity = {"canonical_model": self.spec.canonical_model, "snapshot": self.spec.snapshot.isoformat()}
            catalogs = [{"model_id": self.spec.model, "canonical_slug": data["canonical_slug"], "sources": [path]}
                        for path, data in (("/models", matches[0]), (endpoint_path, endpoint_data))
                        if "canonical_slug" in data]
            if not catalogs:
                check_catalog_snapshot({}, identity)
            for catalog in catalogs:
                check_catalog_snapshot(catalog, identity)
            self.snapshot_catalog = {**catalogs[0], "sources": [c["sources"][0] for c in catalogs]}
            endpoints = endpoint_data["endpoints"]
            if not isinstance(endpoints, list) or not endpoints:
                raise ValueError("No endpoint price list")
            metadata = {**matches[0], "endpoints": endpoints}
            prices = PriceSchedule.from_metadata(metadata, self.spec.model,
                require_reasoning=bool(self.policy["judge"].get("reasoning_effort")))
            self.metadata = {"source": "OpenRouter model and endpoint catalog", **metadata,
                             "snapshot_catalog": self.snapshot_catalog}
        self.prices = prices

    def solver(self):
        from haenv.runtime.solvers import OpenAICompatSolver
        judge, spec = self.policy["judge"], self.spec
        key = self.keys.current()
        if not key:
            raise ValueError("Judge route credentials are unavailable")
        if spec.backend == "google":
            from haenv.scoring.management.transport import ManagementGoogleSolver
            solver = ManagementGoogleSolver(self.name, spec.model, key, 900, retries=0,
                                             max_tokens=judge["max_tokens"])
            solver.URL = self.endpoint
        else:
            solver = OpenAICompatSolver(self.name, spec.model, key, 900, retries=0,
                max_tokens=judge["max_tokens"], url=self.endpoint, backend=spec.backend,
                max_tokens_field=spec.max_tokens_field, reasoning_effort=judge.get("reasoning_effort"),
                provider=spec.provider, sampling=judge.get("sampling"))
            solver.pool = None  # Every send has its own reservation; no hidden rotation.
        solver.proxy = spec.proxy
        solver.sampling = judge.get("sampling")
        if spec.upstream:
            solver.upstream = spec.upstream
        return solver

    def response(self, request, *, before_dispatch=None):
        from haenv.runtime.solver_accounting import wire_settings
        solver = self.solver()
        solver.response_format = request.response_format
        accountant = AccountedCompletion(self.prices, self.ledger, self.receipts,
            backend=self.spec.backend,
            cost_reader=None if self.spec.backend == "openrouter" else self.prices.cost_receipt,
            role="judge", global_pool=True,
            request_limit=self.cfg.get("paid", {}).get("global_request_limit", 48))
        # Alias and upstream names may differ across equivalent routes. Per-route attempt
        # IDs preserve every receipt and prevent any failed attempt from being re-bought.
        request_id = request.sample_id + ":route:" + self.name
        settings = wire_settings(solver)
        if before_dispatch:
            before_dispatch(settings)
        try:
            return accountant.request(request_id, request.prompt, settings, solver._post_wire)
        except Exception as exc:
            if getattr(exc, "status", None) in (401, 403) or getattr(exc, "quota_exhausted", False):
                self.keys.mark_dead(solver.api_key)
                exc.route_has_keys = bool(self.keys.n_alive)
            raise

    def provenance(self):
        return {"name": self.name, "backend": self.spec.backend, "model": self.spec.model,
                "proxy": self.spec.proxy, "endpoint_sha256": hashlib.sha256(self.endpoint.encode()).hexdigest(),
                "snapshot_evidence": "catalog" if self.spec.backend == "openrouter" else "response",
                **({"snapshot_catalog": self.snapshot_catalog} if self.spec.backend == "openrouter" else {})}


class RoutedJudges:
    def __init__(self, cfg, policy, ledger, receipts):
        self.cfg, self.policy, self.ledger, self.receipts = cfg, policy, ledger, Path(receipts)
        self.specs, self.identity = configured_routes(cfg, policy)
        self.routes, self.unavailable = {}, {}
        for name, spec in self.specs.items():
            try:
                self.routes[name] = JudgeRoute(name, spec, cfg, policy, ledger, receipts)
            except Exception as exc:
                # One offline/expired/unfunded route must not prevent other routes from running.
                self.unavailable[name] = type(exc).__name__
                log.warning("Judge route %s preflight failed (%s)", name, type(exc).__name__)
        if not self.routes:
            raise RoutesUnavailable("No judge route passed preflight; no paid calls started")
        self.lanes = RouteLanes({n: self.specs[n] for n in self.routes})

    def _replay(self, entry, request):
        """Recover an accounted attempt before choosing any new route, without credentials."""
        request_id = request.sample_id + ":route:" + entry["name"]
        receipt = self.receipts / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
        prior = self.ledger.record(request_id)
        if not receipt.exists() and (prior is None or never_sent(prior, receipt)):
            return None  # The affinity was written before a send; no purchased vote exists.
        spec = JudgeRouteSpec.model_validate(entry["spec"])
        if spec.backend == "relay":
            prices = DeclaredJudgePrices(spec.model, spec.price, self.cfg.get("paid", {}).get("tariff_expiry", "warn"))
        elif spec.backend == "google":
            from haenv.runtime.native_accounting import NativePrices
            prices = NativePrices(**entry["metadata"]["price"])
        else:
            from haenv.runtime.pricing import PriceSchedule
            prices = PriceSchedule.from_metadata(entry["metadata"], spec.model,
                require_reasoning=bool(self.policy["judge"].get("reasoning_effort")))
        accountant = AccountedCompletion(prices, self.ledger, self.receipts, backend=spec.backend,
            cost_reader=None if spec.backend == "openrouter" else prices.cost_receipt,
            role="judge", global_pool=True)
        def never_dispatch(_):
            raise RuntimeError("A purchased route attempt must never be sent during replay")
        return accountant.request(request_id, request.prompt, entry["settings"], never_dispatch)

    def _reply(self, response, spec, record, metadata):
        from haenv.scoring.management.judge import transport_reply
        from haenv.scoring.management.judge import EmptyJudgeResponse
        from haenv.scoring.semantic.judge import JudgeReply
        if isinstance(response, dict) and response.get("error") and not response.get("choices"):
            # A provider refusal is a route failure, not evidence about the model snapshot.
            error = response["error"]
            exc = RuntimeError("Provider returned an error object")
            code = error.get("code") if isinstance(error, dict) else None
            exc.status = code if type(code) is int else None
            raise exc
        catalog = None
        if spec.backend == "openrouter":
            catalog = metadata.get("snapshot_catalog")
            if catalog is None:
                # Older journals already saved the unmodified model catalog.
                catalog = {"model_id": spec.model, "canonical_slug": metadata.get("canonical_slug"),
                           "sources": ["/models"]}
            record["snapshot_catalog"] = catalog
        record["snapshot_evidence"] = "catalog" if catalog is not None else "response"
        record["snapshot_check"] = check_snapshot(response, self.identity, catalog=catalog)
        if spec.backend == "google":
            from haenv.runtime.solvers import _google_usage
            usage = _google_usage(response)
            candidates = response.get("candidates") or []
            if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
                raise ValueError("Google judge did not finish one complete answer")
            parts = (candidates[0].get("content") or {}).get("parts") or []
            raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if not raw.strip():
                raise EmptyJudgeResponse(raw, usage)
            reply = JudgeReply(raw=raw, cached=False, usage=usage)
        else:
            reply = transport_reply(response, response.get("usage"))
        if spec.provider:
            from types import SimpleNamespace
            from haenv.runtime.transport import check_served
            check_served(SimpleNamespace(provider=spec.provider), response)
        return reply

    def __call__(self, request):
        from haenv.scoring.management.judge import EmptyJudgeResponse
        from haenv.scoring.semantic.parallel import atomic_json
        journal_path = self.receipts.parent / "route-attempts" / (hashlib.sha256(request.sample_id.encode()).hexdigest() + ".json")
        fingerprint = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"request_sha256": fingerprint, "attempts": []}
        if journal["request_sha256"] != fingerprint:
            raise ValueError("Route journal request identity differs")
        attempted, records = set(), []
        for entry in journal["attempts"]:
            record = dict(entry["route"])
            try:
                response = self._replay(entry, request)
                if response is None:
                    continue
                attempted.add(entry["name"])
                reply = self._reply(response, JudgeRouteSpec.model_validate(entry["spec"]), record, entry["metadata"])
                records.append(record)
                return reply, records
            except BudgetExceeded:
                raise
            except Exception as exc:
                record["error_class"] = type(exc).__name__
                records.append(record)
                attempted.add(entry["name"])
                if isinstance(exc, (SnapshotMismatch, EmptyJudgeResponse)):
                    if isinstance(exc, SnapshotMismatch):
                        record["snapshot_check"] = exc.snapshot_check
                    exc.route_attempts = records
                    raise
        while True:
            try:
                name = self.lanes.acquire(exclude=attempted)
            except RoutesUnavailable as exc:
                exc.route_attempts = records
                raise
            route = self.routes[name]
            attempted.add(name)
            record = {**route.provenance(), "request_id": request.sample_id + ":route:" + name}
            failed, disable = False, False
            try:
                def remember(settings):
                    journal["attempts"] = [e for e in journal["attempts"] if e["name"] != name]
                    journal["attempts"].append({"name": name, "spec": route.spec.model_dump(mode="json"),
                        "metadata": route.metadata, "settings": settings, "route": record})
                    journal_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(journal_path, journal)
                response = route.response(request, before_dispatch=remember)
                reply = self._reply(response, route.spec, record, route.metadata)
                records.append(record)
                return reply, records
            except BudgetExceeded:
                raise
            except SnapshotMismatch as exc:
                failed, disable = True, True
                record.update(error_class=type(exc).__name__, snapshot_check=exc.snapshot_check)
                records.append(record)
                exc.route_attempts = records
                raise
            except Exception as exc:
                failed = True
                status = getattr(exc, "status", None)
                disable = ((status in (401, 403) or bool(getattr(exc, "quota_exhausted", False)))
                           and not getattr(exc, "route_has_keys", False))
                record.update(error_class=type(exc).__name__, http_status=status)
                records.append(record)
                if isinstance(exc, EmptyJudgeResponse):
                    exc.route_attempts = records
                    raise
                # Transport and provider failures can move this request to another route.
                # A new route ID is a new accounted attempt, never a reissue of a paid ID.
                log.warning("Judge route %s attempt failed (%s); trying another route", name, type(exc).__name__)
            finally:
                self.lanes.release(name, failed=failed, disable=disable)


def compare_saved(left, right, key_file):
    """Compare a fixed saved cohort without querying a model or rewriting either run."""
    left, right = Path(left), Path(right)
    keys = json.loads(Path(key_file).read_text())
    if (not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys)
            or len(set(keys)) != len(keys) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", k) for k in keys)):
        raise ValueError("Key file must list unique saved cell keys")
    pairs, quality_pairs, missing, mismatches, label_matches, label_total = [], [], [], [], 0, 0
    for key in keys:
        paths = [root / "results" / f"{key}.json" for root in (left, right)]
        if not all(p.is_file() for p in paths):
            missing.append(key)
            continue
        a, b = [json.loads(p.read_text()) for p in paths]
        for field in ("key", "source", "plan_sha256", "answer_sha256", "facts_sha256", "core_decision", "rubric_version"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Compared cell {key} differs in saved input {field}")
        if a["summary"]["status"] != "scored" or b["summary"]["status"] != "scored":
            missing.append(key)
            continue
        prompts = []
        for root, row in ((left, a), (right, b)):
            prompts.append([(s["request"]["prompt"], s["request"]["response_format"])
                            for name in row["raw_judge_files"]
                            for s in [json.loads((root / name).read_text())]])
        if prompts[0] != prompts[1]:
            raise ValueError(f"Compared cell {key} has different judge prompts or response schemas")
        x, y = a["summary"]["rubric_score"], b["summary"]["rubric_score"]
        qx, qy = a["summary"].get("quality_score"), b["summary"].get("quality_score")
        if qx is not None and qy is not None:
            quality_pairs.append((qx, qy))
        different = []
        for item in sorted(set(a["verdicts"]) | set(b["verdicts"])):
            match = a["verdicts"].get(item, {}).get("label") == b["verdicts"].get(item, {}).get("label")
            label_total += 1
            label_matches += match
            if not match:
                different.append(item)
        pairs.append((x, y))
        if x != y or qx != qy or different:
            mismatches.append({"key": key, "left_score": x, "right_score": y,
                               "left_quality_score": qx, "right_quality_score": qy, "different_items": different})
    n = len(pairs)
    return {"requested_cells": len(keys), "paired_scored_cells": n, "missing_or_unscored": missing,
            "exact_score_matches": sum(x == y for x, y in pairs),
            "score_agreement": sum(x == y for x, y in pairs) / n if n else None,
            "mean_absolute_score_difference": sum(abs(x - y) for x, y in pairs) / n if n else None,
            "paired_quality_cells": len(quality_pairs),
            "quality_score_agreement": sum(x == y for x, y in quality_pairs) / len(quality_pairs) if quality_pairs else None,
            "mean_absolute_quality_score_difference": sum(abs(x - y) for x, y in quality_pairs) / len(quality_pairs) if quality_pairs else None,
            "verdict_label_matches": label_matches, "verdict_labels": label_total,
            "verdict_label_agreement": label_matches / label_total if label_total else None,
            "differences": mismatches, "paid_calls": 0}


def main(argv=None):
    """`haenv routes check`: no network at all unless --paid is present."""
    import argparse
    from haenv.app.config import load_cfg
    from haenv.scoring.management.rubric import load_policy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["check", "compare"])
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--limit-usd")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "compare":
        if args.paid or args.left is None or args.right is None or args.key_file is None:
            parser.error("compare requires --left, --right and --key-file; it is always offline")
        result = compare_saved(args.left, args.right, args.key_file)
        if args.out:
            from haenv.scoring.semantic.parallel import atomic_json
            destination = args.out.resolve()
            if any(destination == root.resolve() or root.resolve() in destination.parents
                   for root in (args.left, args.right)):
                parser.error("comparison output must be outside both source runs")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if not result["missing_or_unscored"] else 2
    if args.paid and (args.ledger is None or args.limit_usd is None or args.out is None):
        parser.error("--paid requires --ledger, --limit-usd and --out for durable receipts")
    cfg, policy = load_cfg(), load_policy()
    if not cfg.get("judge_routes"):
        print(json.dumps({"routes": [], "paid": False, "status": "no_routes_configured"}))
        return 2
    specs, identity = configured_routes(cfg, policy)
    ledger = None
    if args.paid:
        from haenv.runtime.budget import BudgetLedger
        ledger = BudgetLedger(args.ledger, limit_usd=args.limit_usd)
    reports = []
    for name, spec in specs.items():
        report = {"route": name, "paid": args.paid, "expected_model": identity,
                  "credentials": "missing", "price_source": "configured tariff" if spec.backend == "relay"
                  else "native tariff" if spec.backend == "google" else "OpenRouter live catalog (unchecked offline)"}
        try:
            route = JudgeRoute(name, spec, cfg, policy, ledger,
                               args.out / "receipts" if args.out else None, price_check=False)
            report["credentials"] = "********" if route.keys.n_alive else "missing"
            report["credential_count"] = route.keys.n_alive
            if spec.price:
                prices = DeclaredJudgePrices(spec.model, spec.price, cfg.get("paid", {}).get("tariff_expiry", "warn"))
                prices.validate_dispatch()
                report["tariff_checked_on"] = spec.price.tariff_checked_on.isoformat()
                report["tariff_source"] = spec.price.source
            if args.paid:
                import uuid
                from haenv.scoring.semantic.judge import JudgeRequest
                route.preflight()
                # A connectivity check needs only a tiny completion. No scoring policy is changed.
                route.policy = {**policy, "judge": {**policy["judge"], "max_tokens": 16}}
                request = JudgeRequest(sample_id="routes-check:" + uuid.uuid4().hex,
                    prompt="Reply OK.", index=1, fresh=True, response_format=None)
                response = route.response(request)
                report["snapshot_evidence"] = "catalog" if spec.backend == "openrouter" else "response"
                report["snapshot_check"] = check_snapshot(response, identity, catalog=route.snapshot_catalog)
                report["status"] = "connected"
            else:
                report["status"] = "offline_valid" if route.keys.n_alive else "credentials_missing"
        except Exception as exc:
            report.update(status="failed", error_class=type(exc).__name__)
            if getattr(exc, "status", None) is not None:
                report["http_status"] = exc.status
            if isinstance(exc, SnapshotMismatch):
                report["snapshot_check"] = exc.snapshot_check
        reports.append(report)
    result = {"routes": reports, "paid": args.paid}
    if ledger is not None:
        result["budget"] = ledger.summary()
    if args.out:
        from haenv.scoring.semantic.parallel import atomic_json
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out / "routes-check.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if all(r["status"] in {"connected", "offline_valid"} for r in reports) else 2
