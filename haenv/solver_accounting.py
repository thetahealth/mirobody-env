"""Bind solver HTTP attempts to the same durable dollar ledger as the judge.

OpenRouter uses usage.cost; a verified New API relay uses exact gateway log
receipts. Google/Dashscope use dated tariff upper bounds, never fake USD invoices.
"""
from __future__ import annotations

import fcntl
from dataclasses import asdict
import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

from .paid_completion import AccountedCompletion
from .semantic_budget import BudgetExceeded, BudgetLedger
from .semantic_transport import PriceSchedule


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def wire_settings(solver) -> dict:
    """Non-secret identity of every option that currently affects a solver call."""
    out = {"backend": solver.backend, "model": solver.model,
           "max_tokens": solver.max_tokens, "max_tokens_field": solver.max_tokens_field,
           "reasoning_effort": solver.reasoning_effort, "stream": solver.stream,
           "response_format": solver.response_format, "retries": solver.retries,
           "endpoint_sha256": _hash(solver.URL)}
    if getattr(solver, "provider", None):          # only when set: identities of unrouted solvers are unchanged
        out["provider"] = solver.provider
    if getattr(solver, "upstream", None):          # declared upstream (config `upstream`), only when set
        out["upstream"] = solver.upstream
    return out


def answer_settings(solver) -> dict:
    """Batch identity of one solver: canonical model, upstream, max_tokens, effort,
    response format, sampling. Backend, endpoint, keys, stream and retries are route."""
    from .transport import answer_view
    return answer_view(wire_settings(solver))


#: Rate-limit backoff (HTTP 429 / upstream rate-limited): exponential with full jitter, each
#: retry a new request identity. Overridable per run via `config.rate_limit_backoff`.
#: Occupied request ids one call may step over before the run stops (a resumed cell holds at
#: most its earlier runs' requests, a few dozen; the cap only guards against a loop).
MAX_SEQUENCE_SKIPS = 1000
BACKOFF_DEFAULTS = {"max_retries": 6, "base_s": 2.0, "max_delay_s": 60.0, "max_total_s": 300.0}


class RequestIndexConflict(BudgetExceeded):
    """Legacy receipts disagree about one request; stop rather than guess which was paid."""


def cell_v2(batch_uid: str, case_id: str, solver_name: str) -> str:
    """Content-addressed cell id: independent of the batch identity (it survives migrations)."""
    return _hash(["cell-v2", batch_uid, case_id, solver_name])


def v2_request_id(cell: str, prompt_sha256: str, n: int) -> str:
    return f"solver:v2:{cell}:{prompt_sha256[:16]}:{n}"


class RequestIndex:
    """(cell_v2, prompt_sha256, n) -> legacy sequence request id, built once from receipts.

    Ledger keys are never renamed; the index only points a content address at the id a
    legacy (sequence-numbered) run already used, so that request replays instead of being
    bought again.
    """

    FILE = "request_index.jsonl"

    def __init__(self, entries: dict | None = None):
        self.entries = dict(entries or {})

    def get(self, cell: str, prompt_sha256: str, n: int) -> str | None:
        return self.entries.get((cell, prompt_sha256, n))

    @staticmethod
    def legacy_cells(root: Path, manifest: dict, case_ids, *, legacy_roots=None) -> dict:
        """legacy cell hash -> (case, solver, epoch) for every run id a sequence-numbered
        request of this batch can have been issued under, read from the manifest alone.

        - A v1 identity (`identity_v1` after the v1 -> v2 migration, `identity` before it),
          replayed through the `code_upgrades` / `limit_changes` recorded up to the
          migration: under the v1 code each of them re-derived `hash(root, identity)`. The
          root is the one the migration recorded (`legacy_root`), so a copied batch still
          maps; `legacy_roots` overrides it for a copy of a not yet migrated batch.
        - The pinned `run_id` (a migrated batch pins its last v1 run id, a v2 batch its
          creation run id): the only run id sequence ids were derived from afterwards.
        Epochs order the run ids in time; a cell's requests continue across epochs.
        """
        ident = manifest["identity"]
        migrations = manifest.get("identity_migrations") or []
        v1 = manifest.get("identity_v1") or (ident if ident.get("protocol") != PROTOCOL_V2 else None)
        run_ids: list[str] = []
        if v1 is not None:
            cutoff = migrations[0].get("at") if migrations else None
            before = lambda e: cutoff is None or (e.get("at") or "") <= cutoff  # noqa: E731
            ups = [u for u in manifest.get("code_upgrades") or [] if before(u)]
            lims = [c for c in manifest.get("limit_changes") or [] if before(c)]
            sha = ups[0]["from"] if ups else (v1.get("source") or {}).get("haenv_git_sha")
            limit = lims[0]["from"] if lims else v1.get("limit_usd")
            states = [(sha, limit)]
            for _, kind, value in sorted([(u.get("at") or "", "sha", u.get("to")) for u in ups]
                                         + [(c.get("at") or "", "limit", c.get("to")) for c in lims]):
                sha, limit = (value, limit) if kind == "sha" else (sha, value)
                states.append((sha, limit))
            extra = [(s_, l_) for s_ in {sha for sha, _ in states} for l_ in {l for _, l in states}]
            order = states + [st for st in extra if st not in states]
            candidates = [{**v1, "limit_usd": l_,
                           "source": {**(v1.get("source") or {}), "haenv_git_sha": s_}}
                          for s_, l_ in order]
            candidates += manifest.get("identity_history") or []
            roots = (list(legacy_roots or [])
                     or [m["legacy_root"] for m in migrations if m.get("legacy_root")]
                     or [str(Path(root).resolve())])
            run_ids += [_hash([root_str, c]) for root_str in roots for c in candidates]
        if manifest.get("run_id"):
            run_ids.append(manifest["run_id"])
        solvers = sorted((v1 or {}).get("settings") or ident.get("answer_settings") or {})
        out = {}
        for epoch, run_id in enumerate(run_ids):
            for case in case_ids:
                for solver in solvers:
                    out.setdefault(_hash([run_id, case, solver]), (case, solver, epoch))
        return out

    @classmethod
    def build(cls, root: Path, manifest: dict, case_ids, batch_uid: str, *, legacy_roots=None):
        receipts = Path(root) / "receipts"
        cells = cls.legacy_cells(root, manifest, case_ids, legacy_roots=legacy_roots)
        seen: dict[str, str] = {}
        groups: dict[tuple, set] = {}
        unmapped = []
        for path in sorted(receipts.glob("*.json")) if receipts.is_dir() else []:
            if path.name.endswith((".billing.json", ".inflight.json")):
                continue
            request = (json.loads(path.read_text()).get("request") or {})
            rid, psha = request.get("request_id"), request.get("prompt_sha256")
            if not isinstance(rid, str) or not rid.startswith("solver:") or rid.startswith("solver:v2:"):
                continue
            if rid in seen and seen[rid] != psha:
                raise RequestIndexConflict(f"Legacy request {rid} carries two different prompts")
            seen[rid] = psha
            legacy_cell, _, seq = rid[len("solver:"):].rpartition(":")
            if legacy_cell not in cells or not seq.isdigit():
                unmapped.append(rid)
                continue
            case, solver, epoch = cells[legacy_cell]
            groups.setdefault((case, solver, psha), set()).add((epoch, int(seq), rid))
        if unmapped:
            raise RequestIndexConflict(
                f"{len(unmapped)} legacy receipt(s) belong to no recorded identity; content "
                "addressing would buy them again")
        entries = {}
        for (case, solver, psha), members in groups.items():
            cell = cell_v2(batch_uid, case, solver)
            for n, (_, _, rid) in enumerate(sorted(members), 1):
                key = (cell, psha, n)
                if key in entries and entries[key] != rid:
                    raise RequestIndexConflict("Two legacy ids for one content address")
                entries[key] = rid
        return cls(entries)

    def save(self, root: Path) -> None:
        lines = [json.dumps({"cell_v2": c, "prompt_sha256": p, "n": n, "request_id": r}, sort_keys=True)
                 for (c, p, n), r in sorted(self.entries.items())]
        path = Path(root) / self.FILE
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
            f.write("".join(line + "\n" for line in lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(f.name, path)

    @classmethod
    def load(cls, root: Path):
        path = Path(root) / cls.FILE
        entries = {}
        for line in path.read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                entries[(row["cell_v2"], row["prompt_sha256"], row["n"])] = row["request_id"]
        return cls(entries)


class AllKeysExhausted(RuntimeError):
    """Every key of this route is out of quota or disabled."""


class KeyRotation:
    """Pick one key per attempt (round robin over live, enabled keys); never inside one."""

    def __init__(self, pool, backend: str, ledger_path):
        self.keys = list(getattr(pool, "_keys", []) or [])
        self.ordinals = list(getattr(pool, "ordinals", range(len(self.keys))))
        self.backend, self.ledger_path = backend, ledger_path
        self.pool = pool
        self._rr = 0
        self._lock = threading.Lock()

    def _dead(self) -> set:
        return set(getattr(self.pool, "_dead", set()) or set())

    def pick(self) -> tuple[str, int]:
        from .ops.runtime import for_ledger
        disabled = set((for_ledger(self.ledger_path).current()["keys"].get(self.backend) or {})
                       .get("disabled", []))
        dead = self._dead()
        with self._lock:
            for _ in range(len(self.keys)):
                i = self._rr % len(self.keys)
                self._rr += 1
                if self.keys[i] not in dead and self.ordinals[i] not in disabled:
                    return self.keys[i], self.ordinals[i]
        raise AllKeysExhausted(f"no usable {self.backend} key")

    def mark_dead(self, ordinal: int) -> None:
        from .ops.runtime import ops_log
        i = self.ordinals.index(ordinal)
        if hasattr(self.pool, "_dead"):
            with self.pool._lock:
                self.pool._dead.add(self.keys[i])
        ops_log(self.ledger_path, "key_dead", backend=self.backend, key_ordinal=ordinal,
                reason="quota_exhausted")

    def alive(self) -> int:
        try:
            self.pick()
        except AllKeysExhausted:
            return 0
        return 1


class Route:
    """One way to reach a model: its own solver (payload), priced completion and keys."""

    def __init__(self, name: str, paid: AccountedCompletion, solver, keys: KeyRotation | None = None):
        self.name, self.paid, self.solver, self.keys = name, paid, solver, keys


#: Process-wide route health per model: forward-only active index and 5xx streaks.
_CHAINS: dict[str, dict] = {}
_CHAINS_LOCK = threading.Lock()


def _chain_locked(model: str) -> dict:
    """`_chain` for a caller already holding `_CHAINS_LOCK`."""
    return _CHAINS.setdefault(model, {"active": 0, "streak": {}})


def _chain(model: str) -> dict:
    with _CHAINS_LOCK:
        return _chain_locked(model)


def _saved_attempt(paid: AccountedCompletion, request_id: str) -> dict | None:
    """The `attempt` record of a durable receipt of `request_id` ({} when the receipt names
    none), or None when the request has no receipt yet (it will be sent)."""
    key = hashlib.sha256(request_id.encode()).hexdigest()
    for suffix in (".json", ".failed.json"):
        path = paid.receipts / f"{key}{suffix}"
        if path.is_file():
            try:
                return dict(json.loads(path.read_text()).get("attempt") or {})
            except (OSError, ValueError):
                return {}
    return None


class SolverRequestAccountant:
    """One mutable request cursor per cell, never shared across solver copies.

    A rate-limited attempt (429) is retried here with exponential backoff under a new
    request identity; the wait happens outside the paid request slot. Each retry is
    recorded (`drain_backoff`) and goes onto the cell's row.

    With `cell_v2` the request id is content-addressed:
    `solver:v2:{cell_v2}:{prompt_sha256[:16]}:{n}`, n = the how-many-th request of this
    prompt in this cell. A resumed run asking prompt P for the k-th time maps to the
    k-th record of (cell, P), whatever other prompts were answered from saved rows.
    `index` points content addresses at ids a legacy sequence-numbered run used.
    """

    def __init__(self, paid: AccountedCompletion, cell_id: str, stop: threading.Event,
                 backoff: dict | None = None, *, sleep=None, rng=None, cell_v2: str | None = None,
                 index: RequestIndex | None = None, routes: list | None = None,
                 model: str | None = None, keys: KeyRotation | None = None):
        import random
        import time
        self.paid, self.cell_id, self.stop = paid, cell_id, stop
        self.sequence = 0
        self.backoff = {**BACKOFF_DEFAULTS, **(backoff or {})}
        self._sleep, self._rng = sleep or time.sleep, rng or random.random
        self._events: list[dict] = []
        self.sequence_skips: list[str] = []
        self.cell_v2, self.index = cell_v2, index or RequestIndex()
        self._counts: dict[str, int] = {}
        self.routes, self.model, self.keys = routes, model, keys
        self.last_request_id = None

    def drain_backoff(self) -> list[dict]:
        events, self._events = self._events, []
        return events

    def _next_id(self, prompt: str) -> str:
        if self.cell_v2 is None:
            self.sequence += 1
            return f"solver:{self.cell_id}:{self.sequence}"
        psha = hashlib.sha256(prompt.encode()).hexdigest()
        n = self._counts.get(psha, 0) + 1
        self._counts[psha] = n
        return self.index.get(self.cell_v2, psha, n) or v2_request_id(self.cell_v2, psha, n)

    def _route_for(self, request_id: str):
        """The route an already-recorded request went out on (replay), else the active one."""
        if not self.routes:
            return None
        key = hashlib.sha256(request_id.encode()).hexdigest()
        for route in self.routes:
            for suffix in (".json", ".failed.json"):
                path = route.paid.receipts / f"{key}{suffix}"
                if path.is_file():
                    from .transport import route_view
                    saved = json.loads(path.read_text())
                    # Routes share one receipts directory: the receipt names its route
                    # (`attempt.route`), else its route view, else (pre-split) full settings.
                    name = (saved.get("attempt") or {}).get("route")
                    for candidate in self.routes:
                        if name is not None and candidate.name == name:
                            return candidate
                    wire = (saved.get("request") or {}).get("settings")
                    for candidate in self.routes:
                        ws = wire_settings(candidate.solver)
                        got = saved.get("route")
                        if got is not None and {k: v for k, v in route_view(ws).items() if k != "key_ordinal"} \
                                == {k: v for k, v in got.items() if k != "key_ordinal"}:
                            return candidate
                        if ws == wire:
                            return candidate
                    return route
        return self.routes[min(_chain(self.model)["active"], len(self.routes) - 1)]

    def _one(self, solver, prompt, dispatch):
        from .paid_completion import ReceiptSequenceMismatch
        from . import relay_accounting
        from .transport import WireMismatch, verify_wire
        for _ in range(MAX_SEQUENCE_SKIPS + 1):
            request_id = self._next_id(prompt)
            self.last_request_id = request_id
            route = self._route_for(request_id)
            paid, target, send, keys = self.paid, solver, dispatch, self.keys
            if route is not None:
                paid, target, keys = route.paid, route.solver, route.keys
                send = dispatch if route.solver is solver else route.solver._post_wire
            body_of = getattr(target, "_wire_body", None)
            if callable(body_of):
                # Pre-send wire check before any reservation: the body the transport will
                # send (on the route actually taken) must carry exactly this prompt and the
                # accounted answer settings.
                try:
                    verify_wire(target, body_of(prompt), prompt)
                except WireMismatch:
                    self.stop.set()
                    raise
            attempt = {}
            if route is not None:
                attempt["route"] = route.name
            saved_attempt = _saved_attempt(paid, request_id)
            if keys is not None and saved_attempt is not None:
                # Replay (a receipt exists, nothing is sent): no rotation. The key that served
                # the request is the one its receipt names (billing lookups need it).
                ordinal = saved_attempt.get("key_ordinal")
                if ordinal in keys.ordinals:
                    attempt.update(api_key=keys.keys[keys.ordinals.index(ordinal)], key_ordinal=ordinal)
            elif keys is not None:
                key, ordinal = keys.pick()
                attempt.update(api_key=key, key_ordinal=ordinal)
                base = send
                send = lambda p, _b=base, _k=key: _b(p, api_key=_k)
            relay_accounting.ATTEMPT.key = attempt.get("api_key")
            try:
                result = paid.request(request_id, prompt, wire_settings(target), send,
                                      attempt=attempt or None)
                if route is not None:
                    with _CHAINS_LOCK:
                        _chain_locked(self.model)["streak"][route.name] = 0
                    if route is not self.routes[0]:
                        self._events.append({"request_id": request_id, "route": route.name})
                return result
            except ReceiptSequenceMismatch:
                # Held by another request of this cell (see ReceiptSequenceMismatch): next id.
                self.sequence_skips.append(request_id)
            except BudgetExceeded:
                self.stop.set()
                raise
            except Exception as error:  # noqa: BLE001 -- annotate for key/route decisions
                error.haenv_attempt = {**{k: v for k, v in attempt.items() if k != "api_key"},
                                       "request_id": request_id, "_route": route, "_keys": keys}
                raise
            finally:
                relay_accounting.ATTEMPT.key = None
        self.stop.set()
        raise BudgetExceeded("Too many occupied request ids for one cell; stopping")

    def _after_failure(self, error) -> bool:
        """Key death and forward-only route fallback; True = try again at once."""
        from .ops.runtime import for_ledger, ops_log
        info = getattr(error, "haenv_attempt", None) or {}
        route, keys = info.get("_route"), info.get("_keys")
        retry = False
        if getattr(error, "quota_exhausted", False) and keys is not None and info.get("key_ordinal") is not None:
            keys.mark_dead(info["key_ordinal"])
            self._events.append({"request_id": info["request_id"], "key_dead": info["key_ordinal"]})
            retry = keys.alive() > 0
        if route is None or len(self.routes) < 2:
            return retry
        ledger_path = route.paid.ledger.path
        policy = for_ledger(ledger_path).current()["fallback_on"]
        chain = _chain(self.model)
        status = getattr(error, "status", None)
        reason = None
        if getattr(error, "quota_exhausted", False) and not retry and policy.get("quota_exhausted"):
            reason = "all_keys_quota_exhausted"
        elif status in (401, 403) and not getattr(error, "quota_exhausted", False) and policy.get("auth_failed"):
            reason = f"auth_failed_{status}"
        elif (type(status) is int and status >= 500) or getattr(error, "stream_cut", False):
            with _CHAINS_LOCK:          # read-modify-write: concurrent 5xx must not lose a count
                streak = chain["streak"][route.name] = chain["streak"].get(route.name, 0) + 1
            limit = policy.get("consecutive_5xx") or 0
            if limit and streak >= limit:
                reason = f"consecutive_5xx_{streak}"
        if reason is None:
            return retry
        with _CHAINS_LOCK:
            here = self.routes.index(route)
            if chain["active"] == here and here + 1 < len(self.routes):
                chain["active"] = here + 1
                event = {"model": self.model, "from": route.name, "to": self.routes[here + 1].name,
                         "reason": reason, "request_id": info["request_id"]}
                ops_log(ledger_path, "route_fallback", **event)
                self._events.append({"route_fallback": event})
        return chain["active"] > self.routes.index(route) or retry

    def request(self, solver, prompt: str, dispatch) -> dict:
        import time
        if self.stop.is_set():
            raise BudgetExceeded("Solver run stopped; no further requests may start")
        if solver.pool is not None:
            raise ValueError("Budgeted solver must not rotate keys inside a single attempt")
        cfg, started, retries, switches = self.backoff, time.monotonic(), 0, 0
        while True:
            try:
                return self._one(solver, prompt, dispatch)
            except BudgetExceeded:
                raise
            except AllKeysExhausted:
                raise
            except Exception as error:  # noqa: BLE001 -- only rate limits, dead keys, fallbacks retry
                if not getattr(error, "rate_limited", False):
                    if switches < 64 and self._after_failure(error):
                        switches += 1
                        if self.stop.is_set():
                            raise BudgetExceeded("Solver run stopped; no further requests may start") from None
                        continue
                    raise
                if retries >= int(cfg["max_retries"]):
                    raise
                replayed = getattr(error, "replayed", False)
                delay = 0.0
                if not replayed:
                    delay = min(float(cfg["max_delay_s"]), float(cfg["base_s"]) * 2 ** retries)
                    delay *= self._rng()                       # full jitter
                    try:
                        delay = max(delay, min(float(error.retry_after), float(cfg["max_delay_s"])))
                    except (AttributeError, TypeError, ValueError):
                        pass
                    if time.monotonic() - started + delay > float(cfg["max_total_s"]):
                        raise
                retries += 1
                self._events.append({"request_id": self.last_request_id,
                                     "http_status": getattr(error, "status", None),
                                     "settled": type(error).__name__ == "RejectedBeforeGeneration",
                                     "backoff_s": round(delay, 3), "replayed": replayed})
                if self.stop.is_set():
                    raise BudgetExceeded("Solver run stopped; no further requests may start") from None
                if delay:
                    self._sleep(delay)


def generation_cost_lookup(cfg: dict, *, attempts: int = 6, delay_s: float = 5.0):
    """Exact per-generation USD from OpenRouter's generation record (free GET).

    Used only when a completion lost its `usage.cost` (e.g. a cut stream). The
    record's `total_cost` equals `usage.cost` on complete responses (checked on a
    live judge receipt). No record => cost stays unknown.
    """
    import time
    import urllib.error
    from .evaluate import BACKENDS, _ensure_backends_registered, load_env_file
    _ensure_backends_registered(cfg)
    backend = BACKENDS["openrouter"]
    key = load_env_file(cfg.get("env_file")).get(backend["key_env"])
    base = backend["url"].rsplit("/chat/completions", 1)[0]

    def lookup(generation_id: str) -> dict:
        from urllib.parse import quote
        not_found = 0
        started = time.monotonic()
        for attempt in range(attempts):
            request = urllib.request.Request(f"{base}/generation?id={quote(generation_id)}",
                                             headers={"Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read()).get("data") or {}
            except urllib.error.HTTPError as error:
                not_found += error.code == 404
                data = {}
            except (urllib.error.URLError, OSError, ValueError):
                data = {}
            if data.get("id") == generation_id and data.get("total_cost") is not None:
                return {"kind": "openrouter_generation_record", "generation_id": generation_id,
                        "actual_usd": str(data["total_cost"]),
                        "tokens_prompt": data.get("tokens_prompt"),
                        "tokens_completion": data.get("tokens_completion"),
                        "native_tokens_reasoning": data.get("native_tokens_reasoning"),
                        "streamed": data.get("streamed"), "cancelled": data.get("cancelled"),
                        "finish_reason": data.get("finish_reason"), "model": data.get("model")}
            time.sleep(delay_s)
        if not_found == attempts:
            raise NoGenerationRecord(generation_id, attempts, round(time.monotonic() - started, 1))
        raise ValueError("No generation cost record")
    return lookup


class NoGenerationRecord(ValueError):
    """Every lookup answered 404: OpenRouter holds no billed generation under this id."""

    def __init__(self, generation_id: str, lookups: int, waited_s: float):
        super().__init__("No generation record")
        self.generation_id, self.lookups, self.waited_s = generation_id, lookups, waited_s


def fetch_endpoints(cfg: dict, model: str) -> list[dict]:
    """Free per-endpoint price list for one exact OpenRouter model id."""
    from .evaluate import BACKENDS, _ensure_backends_registered, load_env_file
    _ensure_backends_registered(cfg)
    backend = BACKENDS["openrouter"]
    key = load_env_file(cfg.get("env_file")).get(backend["key_env"])
    url = backend["url"].rsplit("/chat/completions", 1)[0] + f"/models/{model}/endpoints"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read())["data"]
    except Exception:
        raise ValueError("Could not verify endpoint prices; no paid calls started") from None
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError(f"No endpoint price list for {model}")
    return [{"provider_name": e.get("provider_name"), "pricing": e.get("pricing") or {}} for e in endpoints]


def fetch_catalog(cfg: dict) -> list[dict]:
    """Free exact-model price lookup; credentials and failure URLs are not logged."""
    from .evaluate import BACKENDS, _ensure_backends_registered, load_env_file
    _ensure_backends_registered(cfg)
    backend = BACKENDS["openrouter"]
    key = load_env_file(cfg.get("env_file")).get(backend["key_env"])
    if not key:
        raise ValueError("OpenRouter credentials are required for solver pricing")
    url = backend["url"].rsplit("/chat/completions", 1)[0] + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            catalog = json.loads(response.read())["data"]
    except Exception:
        raise ValueError("Could not verify solver price metadata; no paid calls started") from None
    if not isinstance(catalog, list):
        raise ValueError("Invalid solver model catalog")
    return catalog


class BatchAccounting:
    def __init__(self, ledger, prices, settings, root, run_id, cfg):
        self.ledger, self.prices, self.settings = ledger, prices, settings
        self.answer = {name: answer_settings_of(v) for name, v in settings.items()}
        self.root, self.run_id = root, run_id
        self.cfg = cfg
        self.stop = threading.Event()
        self.batch_uid = None                 # set by `enable_content_keys`
        self.index = RequestIndex()
        #: solver name -> [(route name, make_solver, prices)] fallback routes after the primary
        self.fallbacks: dict[str, list] = {}

    def enable_content_keys(self, case_ids, *, legacy_roots=None) -> dict:
        """Give the batch a stable `batch_uid` and build the legacy request index once."""
        manifest = self.root / "manifest.json"
        with (self.root / "manifest.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            saved = json.loads(manifest.read_text())
            # The batch identity has one source: the run id pinned at creation (or at the
            # v1 -> v2 migration). Content addresses derive from it, so they survive every
            # accepted upgrade exactly as the pinned run id does.
            if saved.get("run_id") != self.run_id:
                raise ValueError("Manifest run id differs from the accounting run id")
            if saved.get("batch_uid") not in (None, self.run_id):
                raise ValueError("Manifest batch_uid is not the pinned run id; use a new batch")
            if saved.get("batch_uid") is None:
                saved["batch_uid"] = self.run_id
                AccountedCompletion._persist(manifest, saved)
            self.batch_uid = saved["batch_uid"]
            if (self.root / RequestIndex.FILE).is_file():
                self.index = RequestIndex.load(self.root)
            else:
                self.index = RequestIndex.build(self.root, saved, list(case_ids), self.batch_uid,
                                                legacy_roots=legacy_roots)
                self.index.save(self.root)
        return {"batch_uid": self.batch_uid, "indexed": len(self.index.entries)}

    def _declared(self, solver_name: str) -> frozenset:
        """`(from, to)` upstream declarations the manifest recorded for this solver (both the
        v2 resume record and the v1 -> v2 migration record)."""
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            return frozenset()
        saved = json.loads(manifest.read_text())
        rows = [d for rec in saved.get("upstream_declarations") or [] for d in rec.get("declared") or []]
        rows += [d for rec in saved.get("identity_migrations") or []
                 for d in rec.get("upstream_declared") or []]
        return frozenset((d["from"], d["to"]) for d in rows
                         if d.get("solver") == solver_name and str(d.get("from")).endswith(":unpinned"))

    def _completion(self, prices, solver, solver_name: str | None = None) -> AccountedCompletion:
        reader, probe = None, None
        if solver.backend == "relay":
            from .relay_accounting import RelayBilling, usage_probe
            from .evaluate import BACKENDS, _ensure_backends_registered
            reader = RelayBilling.for_solver(prices, solver, self.cfg)
            _ensure_backends_registered(self.cfg)
            base = BACKENDS["relay"].get("quota_url")
            probe = usage_probe(base) if base else None
        elif solver.backend in {"google", "dashscope"}:
            reader = prices.cost_receipt
        lookup = generation_cost_lookup(self.cfg) if solver.backend == "openrouter" else None
        return AccountedCompletion(prices, self.ledger, self.root / "receipts",
                                   backend=solver.backend, cost_reader=reader,
                                   generation_lookup=lookup, usage_probe=probe,
                                   upstream_declared=(self._declared(solver_name)
                                                      if solver_name else frozenset()))

    @staticmethod
    def _keys(solver, ledger_path):
        pool = getattr(solver, "pool", None)
        if solver.backend == "relay" and pool is not None and len(getattr(pool, "_keys", [])) > 1:
            return KeyRotation(pool, solver.backend, ledger_path)
        return None

    def bind(self, solver, case_id: str, solver_name: str) -> None:
        from .baselines import BASELINE_NAMES
        if solver_name in BASELINE_NAMES:
            return
        if (solver_name not in self.prices or answer_settings(solver) != self.answer[solver_name]
                or solver.backend != self.settings[solver_name]["backend"]):
            raise ValueError("Solver settings differ from the verified accounting preflight")
        keys = self._keys(solver, self.ledger.path)
        solver.pool = None                   # never rotated inside an attempt; see KeyRotation
        cell = _hash([self.run_id, case_id, solver_name])
        paid = self._completion(self.prices[solver_name], solver, solver_name)
        routes = None
        if self.fallbacks.get(solver_name):
            routes = [Route("primary", paid, solver, keys)]
            for name, make, prices in self.fallbacks[solver_name]:
                alt = make()
                alt_keys = self._keys(alt, self.ledger.path)
                alt.pool = None
                routes.append(Route(name, self._completion(prices, alt, solver_name), alt, alt_keys))
        solver._accounting = SolverRequestAccountant(
            paid, cell, self.stop, self.cfg.get("rate_limit_backoff"),
            cell_v2=None if self.batch_uid is None else cell_v2(self.batch_uid, case_id, solver_name),
            index=self.index, routes=routes, model=solver_name, keys=keys)


def answer_settings_of(settings: dict) -> dict:
    from .transport import answer_view
    return answer_view(settings)


def _infra_segment() -> frozenset:
    import importlib.util
    from . import data_root
    path = Path(data_root()) / "tools" / "make_freeze.py"
    spec = importlib.util.spec_from_file_location("_mf_for_infra_upgrade", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                # noqa: S102
    return frozenset(mod.INFRA)


#: Files whose change cannot alter what a solver is shown, how it is asked or how a
#: cell is scored: the `INFRA` segment (`tools/make_freeze.py`, the single list), the
#: freeze anchor's vintage bookkeeping (`anchor.py` is outside both frozen segments) and
#: the untracked-by-intent deployment config. A resume across a commit that touches
#: anything else must use a new batch.
ACCOUNTING_ONLY = _infra_segment() | {"haenv/anchor.py", "config.local.yaml",
                                       "config.local.example.yaml"}
NON_RUNTIME_PREFIXES = ("tools/", "tests/", "docs/", "sources/", "web/", ".github/")
#: The semantic judge layer executes only after solving (the default judging step and
#: report views), on saved answers, under its own sealed manifest. On the solving path it
#: is at most imported (cli's post-solve steps; semantic_transport imports the judge's
#: request/reply types), never executed: no solver prompt, payload or price bound
#: depends on it. semantic_transport itself prices solver requests and is accounting.
JUDGE_LAYER = frozenset({
    "haenv/semantic_judge.py", "haenv/semantic_pipeline.py", "haenv/semantic_report.py",
    "haenv/semantic_rubric.py", "haenv/semantic_lean.py", "haenv/semantic_inputs.py",
    "haenv/semantic_corrections.py", "haenv/semantic_visibility.py", "haenv/semantic_parallel.py",
    "haenv/judge_evidence.py", "haenv/judge_compaction.py",
    "haenv/judge_scheduler.py", "registry/semantic_judging.yaml", "registry/semantic_judging_v4.yaml"})
#: Package metadata; admitted only while uv.lock (the snapshot environment) is unchanged.
METADATA = frozenset({"CITATION.cff", "pyproject.toml"})


def accounting_only_upgrade(saved: dict, current: dict) -> dict | None:
    """v1 form: identities that differ only in `source.haenv_git_sha` (see `infra_only_upgrade`)."""
    old_sha = (saved.get("source") or {}).get("haenv_git_sha")
    new_sha = (current.get("source") or {}).get("haenv_git_sha")
    strip = lambda ident: {**ident, "source": {k: v for k, v in ident["source"].items() if k != "haenv_git_sha"}}
    if strip(saved) != strip(current):
        return None
    return infra_only_upgrade(old_sha, new_sha)


def infra_only_upgrade(old_sha, new_sha, membership: dict | None = None) -> dict | None:
    """Allow resuming a batch after a commit that changed infrastructure code only.

    The git diff between the two revisions (in this code tree) may touch only `INFRA`
    files, the semantic judge layer, package metadata or non-runtime paths. The upgrade
    is recorded, never silent.

    Membership (which files count as INFRA, judge layer, metadata, non-runtime, budget
    gate) is never taken from the code being checked alone: `membership` is the table the
    batch pinned (default: the tables as committed at `old_sha`), intersected with the
    running code's own tables. A file relabelled in the same commit that changes it is
    therefore a runtime change.
    """
    import subprocess
    from datetime import datetime, timezone
    from . import data_root
    if not old_sha or not new_sha or "+dirty" in old_sha + new_sha:
        return None
    try:
        # -z: raw paths (git otherwise quotes non-ASCII names, which then match no prefix).
        # --no-renames: a runtime file moved out of the runtime tree is a deletion.
        changed = [f for f in subprocess.run(
            ["git", "-C", str(data_root()), "diff", "-z", "--no-renames", "--name-only",
             old_sha, new_sha],
            capture_output=True, text=True, timeout=30, check=True).stdout.split("\0") if f]
        head = subprocess.run(["git", "-C", str(data_root()), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not new_sha.startswith(head[:len(new_sha)]) and not head.startswith(new_sha):
        return None                    # the recorded revision is not the code actually running
    if "uv.lock" in changed:
        return None
    pinned = membership if membership is not None else membership_at(old_sha)
    if pinned is None:
        return None                    # no trusted table to judge the change by
    t = trusted_membership(pinned)
    runtime = [f for f in changed if not f.startswith(t["non_runtime_prefixes"])
               and not (f.endswith(".md") and "/" not in f) and f not in t["metadata"]]
    gated = [f for f in runtime if f in t["budget_gate_only"]
             and _changed_only_in(data_root(), old_sha, new_sha, f, t["budget_gate_only"][f])]
    if any(f not in t["infra"] and f not in t["judge_layer"] and f not in gated for f in runtime):
        return None
    return {"from": old_sha, "to": new_sha, "changed_runtime_files": runtime,
            **({"budget_gate_only": gated} if gated else {}),
            "membership_from": pinned.get("pinned_from"),
            "at": datetime.now(timezone.utc).isoformat()}


#: Functions that only decide whether a run may start spending (budget gates) and live in a
#: file that also scores. A change confined to them is accounting: it cannot alter what a
#: solver is shown, how it is asked or how a cell is scored.
BUDGET_GATE_ONLY = {"haenv/evaluate.py": frozenset({"_preflight_quota"})}


#: Inside an exempted budget-gate body: statements/calls that could change module state
#: seen by the solving path. A body holding any of them is not a budget-gate-only change.
_GATE_BODY_FORBIDDEN_CALLS = frozenset({"globals", "vars", "setattr", "delattr", "exec", "eval",
                                        "__import__", "compile", "locals"})


def _gate_body_is_inert(body: list) -> bool:
    import ast
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            return False
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return False
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in _GATE_BODY_FORBIDDEN_CALLS:
                return False
    return True


def _changed_only_in(root, old_sha: str, new_sha: str, path: str, functions) -> bool:
    """True when `path` differs between the revisions only inside the bodies of the named
    top-level functions. Decorators, arguments (defaults run at import), return annotations
    and everything else are compared as ASTs; the new bodies must not write module state."""
    import ast
    import subprocess

    def stripped(rev):
        try:
            text = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{path}"],
                                  capture_output=True, text=True, timeout=30, check=True).stdout
            tree = ast.parse(text)
        except (OSError, subprocess.SubprocessError, SyntaxError):
            return None, []
        bodies = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in functions:
                bodies.append(n.body)
                n.body = [ast.Pass()]
        return ast.dump(tree), bodies

    a, _ = stripped(old_sha)
    b, new_bodies = stripped(new_sha)
    if a is None or a != b:
        return False
    return all(_gate_body_is_inert(body) for body in new_bodies)


class AccountingRefused(ValueError):
    """`prepare_accounting` refused the run; no billed request was sent."""


def _refusals(prepare):
    """Re-raise a plain `ValueError` from `prepare` as `AccountingRefused`, so the command line
    can print the refusal as one line; subclasses keep their own type."""
    @functools.wraps(prepare)
    def checked(*args, **kwargs):
        try:
            return prepare(*args, **kwargs)
        except ValueError as error:
            if type(error) is not ValueError:
                raise
            raise AccountingRefused(str(error)) from error
    return checked


@_refusals
def prepare_accounting(solvers: list, cfg: dict, batch_dir: Path, *,
                       ledger_path: Path, limit_usd: str, identity: dict,
                       legacy_roots: list[str] | None = None) -> BatchAccounting:
    """Validate the complete requested set before any solver can spend money.

    The manifest pins settings, price data, source provenance and the shared
    ledger. Resume validates identity; it never changes the existing cap or
    silently reissues a request with a changed payload.
    """
    from .baselines import BASELINE_NAMES
    from .evaluate import OpenAICompatSolver, GoogleSolver
    live = {name: make() for name, make in solvers if name not in BASELINE_NAMES}
    unsupported = [name for name, s in live.items()
                   if not isinstance(s, (OpenAICompatSolver, GoogleSolver))
                   or s.backend not in {"openrouter", "relay", "google", "dashscope"}]
    if unsupported:
        raise ValueError(
            "Cannot budget unverified solver route(s): "
            + ", ".join(f"{name} (backend {getattr(live[name], 'backend', '?')})" for name in unsupported)
            + "; a billed run is metered against --judge-budget-usd, and verified price data "
              "exists for the openrouter, relay, google and dashscope backends only")
    ledger = BudgetLedger(Path(ledger_path).resolve(), limit_usd=limit_usd)
    if ledger.snapshot()["halt_reason"]:
        raise BudgetExceeded(ledger.snapshot()["halt_reason"])
    root = Path(batch_dir) / "solver-accounting"
    root.mkdir(parents=True, exist_ok=True)
    settings = {name: wire_settings(s) for name, s in live.items()}
    source_hashes = {name: hashlib.sha256((Path(batch_dir) / name).read_bytes()).hexdigest()
                     for name in ("cases.jsonl", "slices.json") if (Path(batch_dir) / name).is_file()}
    code_rev = (identity or {}).get("haenv_git_sha")
    manifest = root / "manifest.json"
    route_changes = []
    with (root / "manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        membership = {**solving_membership(), "pinned_from": f"created@{code_rev}"}
        if manifest.is_file():
            saved = json.loads(manifest.read_text())
            if saved.get("solving_membership") is None:
                # Pin the membership tables of the code this batch last ran under (never the
                # running code's own: those are what is being checked).
                was = saved.get("code_rev") or ((saved["identity"].get("source") or {})
                                                .get("haenv_git_sha"))
                pinned = membership_at(was)
                if pinned is None:
                    raise ValueError("Cannot read the membership tables of this batch's code "
                                     f"revision {was!r}; use a new batch")
                saved["solving_membership"] = pinned
                AccountedCompletion._persist(manifest, saved)
            membership = saved["solving_membership"]
        run_identity = run_identity_v2(settings, identity, source_hashes, ledger.path, membership)
        if manifest.is_file():
            if saved["identity"].get("protocol") == PROTOCOL_V1:
                migrate_v1_manifest(saved, run_identity, root, code_rev,
                                    legacy_root=(legacy_roots or [None])[0])
                AccountedCompletion._persist(manifest, saved)
            declared = upstream_declarations(saved["identity"], run_identity)
            if declared is not None:
                # Same answer, new upstream claim: recorded; run id and request ids unchanged.
                saved.setdefault("upstream_declarations", []).append(
                    {"declared": declared, "at": _now()})
                saved["identity"] = run_identity
                AccountedCompletion._persist(manifest, saved)
            if saved["identity"] != run_identity:
                upgrade = None
                if strip_code(saved["identity"]) == strip_code(run_identity):
                    upgrade = infra_only_upgrade(saved.get("code_rev"), code_rev,
                                                 saved["solving_membership"])
                if upgrade is None:
                    raise ValueError("Solver accounting run identity changed; use a new batch")
                saved.setdefault("code_upgrades", []).append(upgrade)
                saved["identity"] = run_identity
                AccountedCompletion._persist(manifest, saved)
            if saved.get("code_rev") != code_rev:
                # Identity unchanged (same solving code by content): the revision is recorded.
                saved.setdefault("code_revisions", []).append(
                    {"from": saved.get("code_rev"), "to": code_rev, "at": _now()})
                saved["code_rev"] = code_rev
                AccountedCompletion._persist(manifest, saved)
            metadata = saved["price_metadata"]
            route_meta = saved.setdefault("route_price_metadata", {})
            catalog = None
            for name, s in live.items():
                # A resumed batch may be served by another route than the one it was
                # verified on (only the route: the answer view is the identity). The new
                # route gets its own free price preflight, kept beside the original.
                was = metadata[name].get("route_backend") or (
                    (saved.get("identity_v1") or {}).get("settings", {}).get(name, {}).get("backend"))
                if was is not None and was != s.backend:
                    key = f"{name}@{s.backend}"
                    if key not in route_meta:
                        if catalog is None and s.backend == "openrouter":
                            catalog = fetch_catalog(cfg)
                        route_meta[key] = _route_metadata(cfg, s, name, catalog)
                        route_changes.append({"solver": name, "from": was, "to": s.backend,
                                              "at": _now()})
                    metadata = {**metadata, name: route_meta[key]}
            if route_changes:
                saved.setdefault("route_changes", []).extend(route_changes)
                AccountedCompletion._persist(manifest, saved)
            added = []
            for name, s in live.items():
                if s.backend == "openrouter" and "endpoints" not in metadata[name]:
                    metadata[name]["endpoints"] = fetch_endpoints(cfg, s.model)
                    added.append(name)
            if added:
                from datetime import datetime, timezone
                saved["price_metadata"].update({n: metadata[n] for n in added})
                saved.setdefault("price_upgrades", []).append({
                    "added_endpoint_price_bounds": added, "at": datetime.now(timezone.utc).isoformat(),
                    "why": "model-level price is not an upper bound over OpenRouter endpoints"})
                AccountedCompletion._persist(manifest, saved)
            for name, s in live.items():
                if s.backend in {"google", "dashscope"}:
                    from .native_accounting import preflight_native
                    # A resumed batch may hold native models with no work left; the
                    # tariff date is enforced at dispatch (`validate_dispatch`).
                    current = asdict(preflight_native(s, s.backend, s.URL, False))
                    if metadata[name].get("native_tariff") != current:
                        raise ValueError("Native tariff metadata changed; use a newly verified batch")
        else:
            catalog = fetch_catalog(cfg) if any(s.backend == "openrouter" for s in live.values()) else []
            metadata = {name: _route_metadata(cfg, s, name, catalog) for name, s in live.items()}
        resumed = manifest.exists()
        prices = {}
        for name, s in live.items():
            if s.backend in {"google", "dashscope"}:
                from .native_accounting import NativePrices
                prices[name] = NativePrices(**metadata[name]["native_tariff"])
                if not resumed:
                    prices[name].validate_date()
                continue
            if s.backend == "relay":
                from .relay_accounting import RelayPrices
                prices[name] = RelayPrices.from_metadata(metadata[name], s.model)
                continue
            prices[name] = PriceSchedule.from_metadata(metadata[name], s.model, require_reasoning=False)
            maximum = (metadata[name].get("top_provider") or {}).get("max_completion_tokens")
            if type(maximum) is not int or maximum <= 0:
                raise ValueError(f"Provider output limit is unverified for {name}")
            if s.max_tokens > maximum:
                raise ValueError(f"Configured output cap for {name} exceeds provider limit {maximum}")
        if not manifest.exists():
            saved = {"identity": run_identity, "run_id": _hash([str(root.resolve()), run_identity]),
                     "code_rev": code_rev, "price_metadata": metadata,
                     "solving_membership": membership}
            AccountedCompletion._persist(manifest, saved)
    # The run id (hence every cell id and request id) is fixed when the batch is created
    # (or, for a v1 batch, when it is migrated) and never re-derived: an accepted identity
    # upgrade must not renumber requests whose receipts already exist.
    accounting = BatchAccounting(ledger, prices, settings, root, saved["run_id"], cfg)
    _prepare_fallbacks(accounting, live, cfg, root, ledger.path)
    accounting.enable_content_keys(_case_ids(Path(batch_dir)), legacy_roots=legacy_roots)
    return accounting


def _route_prices(s, name: str, meta: dict):
    """Verified price schedule of one route (the same checks as the primary's)."""
    if s.backend in {"google", "dashscope"}:
        from .native_accounting import NativePrices
        return NativePrices(**meta["native_tariff"])
    if s.backend == "relay":
        from .relay_accounting import RelayPrices
        return RelayPrices.from_metadata(meta, s.model)
    prices = PriceSchedule.from_metadata(meta, s.model, require_reasoning=False)
    maximum = (meta.get("top_provider") or {}).get("max_completion_tokens")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError(f"Provider output limit is unverified for {name}@{s.backend}")
    if s.max_tokens > maximum:
        raise ValueError(f"Configured output cap for {name}@{s.backend} exceeds provider limit {maximum}")
    return prices


def verify_upstream(s, meta: dict) -> bool:
    """Is the declared upstream the one this route is served by? True only with evidence.

    OpenRouter: the route must pin exactly the declared provider with fallbacks off, and
    the model's endpoint list (free GET, in `meta["endpoints"]`) must hold that provider.
    A declared-but-unpinned OpenRouter route is refused (the router may serve it from any
    provider). relay: the gateway gives no upstream evidence => False (recorded, not refused).
    """
    declared = getattr(s, "upstream", None)
    if s.backend in {"google", "dashscope"}:
        return True
    if s.backend != "openrouter" or not declared:
        return False
    pin = getattr(s, "provider", None) or {}
    if pin.get("only") != [declared] or pin.get("allow_fallbacks") is not False:
        raise ValueError(f"{s.name}: OpenRouter route declares upstream {declared!r} but does not "
                         "pin it (provider.only == [upstream], allow_fallbacks: false)")
    names = {e.get("provider_name") for e in meta.get("endpoints") or []}
    if declared not in names:
        raise ValueError(f"{s.name}: OpenRouter lists no {declared!r} endpoint for {s.model}")
    return True


def _prepare_fallbacks(accounting, live: dict, cfg: dict, root: Path, ledger_path) -> None:
    """Fallback routes (config `fallback_routes` / runtime.yaml `routes`): each is built,
    held to the primary's answer view, price-verified by a free preflight and written to
    the manifest (`route_price_metadata`, `fallback_routes`) before any request."""
    from .evaluate import fallback_solver_factories
    manifest = root / "manifest.json"
    plans = {}
    for name, primary in live.items():
        routes = fallback_solver_factories(cfg, name, primary, ledger_path)
        if routes:
            plans[name] = (primary, routes)
    if not plans:
        return
    with (root / "manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        saved = json.loads(manifest.read_text())
        route_meta = saved.setdefault("route_price_metadata", {})
        recorded = saved.setdefault("fallback_routes", {})
        catalog, changed = None, False
        for name, (primary, routes) in plans.items():
            chain = []
            for rname, make, _spec in routes:
                alt = make()
                if alt is None:
                    raise ValueError(f"fallback route {rname} for {name} could not be built")
                if answer_settings(alt) != answer_settings(primary):
                    raise ValueError(f"fallback route {rname} for {name} changes the answer view "
                                     f"({answer_settings(alt)} != {answer_settings(primary)})")
                key = f"{name}@{rname}"
                if key not in route_meta:
                    if catalog is None and alt.backend == "openrouter":
                        catalog = fetch_catalog(cfg)
                    route_meta[key] = _route_metadata(cfg, alt, name, catalog)
                    changed = True
                meta = route_meta[key]
                prices = _route_prices(alt, name, meta)
                entry = {"route": rname, "backend": alt.backend, "model": alt.model,
                         "upstream": answer_settings(alt)["upstream"],
                         "upstream_verified": verify_upstream(alt, meta), "price_key": key}
                chain.append(entry)
                accounting.fallbacks.setdefault(name, []).append((rname, make, prices))
            if recorded.get(name) != chain:
                saved.setdefault("fallback_route_changes", []).append(
                    {"solver": name, "from": recorded.get(name), "to": chain, "at": _now()})
                recorded[name] = chain
                changed = True
        if changed:
            AccountedCompletion._persist(manifest, saved)


PROTOCOL_V1 = "solver-cost-receipts-v1"
PROTOCOL_V2 = "solver-cost-receipts-v2"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _route_metadata(cfg: dict, s, name: str, catalog) -> dict:
    """Free price preflight for one solver on its current route."""
    if s.backend in {"google", "dashscope"}:
        from .native_accounting import preflight_native
        out = {"native_tariff": asdict(preflight_native(s, s.backend, s.URL))}
    elif s.backend == "relay":
        from .relay_accounting import fetch_metadata
        out = fetch_metadata(cfg, s)
    else:
        matches = [m for m in (catalog or []) if m.get("id") == s.model]
        if len(matches) != 1:
            raise ValueError(f"No unique exact-model price record for {name}")
        out = {**matches[0], "endpoints": fetch_endpoints(cfg, s.model)}
    return {**out, "route_backend": s.backend}


#: Runtime files on the default-in scope (`solving_candidates`) that are left out of the
#: solving fingerprint, each with the reason it cannot change what a solver is shown, how it
#: is asked or how a cell is scored. Everything else under `haenv/`, `verifier_core/` and
#: `registry/*.yaml` is in the fingerprint by default, including modules added later.
SOLVING_FP_EXCLUDED: dict[str, str] = {
    "haenv/spend.py": "ledger reports and after-the-fact reconciliation of unknown charges; "
                      "runs on the ledger file, never on the solving path",
    "haenv/ops/__init__.py": "empty package marker of the operator runtime knobs",
    "haenv/ops/runtime.py": "operator knobs (drain, budget pause, disabled keys, route fallback "
                            "policy): whether and on which route a request goes out; the answer "
                            "view is held by the pre-send wire check",
    "haenv/ops/cli.py": "operator command that edits the runtime knobs file; not imported by solving",
    "haenv/board_freeze.py": "run gate: refuses real-model runs while BOARD_FREEZE exists; decides "
                             "only whether a run may start",
    "haenv/inputs.py": "read-only `haenv inputs` listing of knobs.yaml; not on the solving path",
    "haenv/provenance_report.py": "read-only report over saved provenance rows",
    "haenv/verify_report.py": "read-only verification report over a finished batch",
    "registry/knobs.yaml": "documentation of module constants; only consumer is haenv/inputs.py",
    "registry/reachability_baseline.yaml": "baseline of a maintainer tool (tools/reachability_check.py); "
                                           "no runtime consumer",
    "registry/direct_read_baseline.yaml": "baseline of a maintainer tool (tools/measure_copied_enums.py); "
                                          "no runtime consumer",
}


def solving_candidates(root=None) -> list[str]:
    """Default-in scope of the solving fingerprint: every `haenv/**.py`,
    `verifier_core/**.py` and `registry/*.yaml` of the tree."""
    from . import data_root
    root = Path(root if root is not None else data_root())
    out = set()
    for base, pattern in (("haenv", "*.py"), ("verifier_core", "*.py"), ("registry", "*.yaml")):
        for path in (root / base).rglob(pattern) if (root / base).is_dir() else ():
            if "__pycache__" not in path.parts:
                out.add(path.relative_to(root).as_posix())
    return sorted(out)


def solving_membership() -> dict:
    """The running code's own membership tables (what a new batch pins)."""
    return {"infra": sorted(ACCOUNTING_ONLY), "judge_layer": sorted(JUDGE_LAYER),
            "excluded": sorted(SOLVING_FP_EXCLUDED), "metadata": sorted(METADATA),
            "non_runtime_prefixes": list(NON_RUNTIME_PREFIXES),
            "budget_gate_only": {k: sorted(v) for k, v in BUDGET_GATE_ONLY.items()}}


def _table_value(node, infra_at_rev):
    """Evaluate one membership-table assignment of an older solver_accounting.py without
    running it: literals, frozenset/set/tuple of literals, `|` unions, `_infra_segment()`."""
    import ast
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return set(_table_value(node.left, infra_at_rev)) | set(_table_value(node.right, infra_at_rev))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "_infra_segment" and not node.args:
            return set(infra_at_rev)
        if node.func.id in ("frozenset", "set", "tuple") and len(node.args) <= 1:
            return set(_table_value(node.args[0], infra_at_rev)) if node.args else set()
    if isinstance(node, ast.Dict):
        return {ast.literal_eval(k): sorted(_table_value(v, infra_at_rev))
                for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def membership_at(rev: str) -> dict | None:
    """Membership tables as committed at `rev` (git objects, never the running code):
    `INFRA` from that revision's tools/make_freeze.py, the rest from its
    haenv/solver_accounting.py. A table the revision does not have is empty (so the check
    it feeds is stricter, never looser). None when the revision cannot be read."""
    import ast
    import subprocess
    from . import data_root
    base = str(rev or "").split("+", 1)[0]
    if not base:
        return None

    def show(rel):
        r = subprocess.run(["git", "-C", str(data_root()), "show", f"{base}:{rel}"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else None
    try:
        mf = show("tools/make_freeze.py")
        if mf is None:
            return None
        infra = ()
        for node in ast.parse(mf).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "INFRA" for t in node.targets):
                infra = tuple(ast.literal_eval(node.value))
        tables = {}
        sa = show("haenv/solver_accounting.py")
        for node in ast.parse(sa).body if sa else ():
            target = node.target if isinstance(node, ast.AnnAssign) else (
                node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
            name = getattr(target, "id", None)
            if name in ("ACCOUNTING_ONLY", "JUDGE_LAYER", "SOLVING_FP_EXCLUDED", "METADATA",
                        "NON_RUNTIME_PREFIXES", "BUDGET_GATE_ONLY") and node.value is not None:
                tables[name] = _table_value(node.value, infra)
    except (OSError, subprocess.SubprocessError, SyntaxError, ValueError, TypeError):
        return None
    return {"infra": sorted(set(tables.get("ACCOUNTING_ONLY") or ()) | set(infra)),
            "judge_layer": sorted(tables.get("JUDGE_LAYER") or ()),
            "excluded": sorted(tables.get("SOLVING_FP_EXCLUDED") or ()),
            "metadata": sorted(tables.get("METADATA") or ()),
            "non_runtime_prefixes": list(tables.get("NON_RUNTIME_PREFIXES") or ()),
            "budget_gate_only": {k: sorted(v) for k, v in (tables.get("BUDGET_GATE_ONLY") or {}).items()},
            "pinned_from": base}


def trusted_membership(pinned: dict) -> dict:
    """Pinned tables intersected with the running code's: a relabelling counts only once
    both the batch's revision and the running code agree on it."""
    cur = solving_membership()
    both = lambda k: frozenset(pinned.get(k) or ()) & frozenset(cur[k])
    gates = {f: frozenset(fns) & frozenset(cur["budget_gate_only"].get(f, ()))
             for f, fns in (pinned.get("budget_gate_only") or {}).items()}
    return {"infra": both("infra") | both("excluded"), "judge_layer": both("judge_layer"),
            "metadata": both("metadata"),
            "non_runtime_prefixes": tuple(p for p in pinned.get("non_runtime_prefixes") or ()
                                          if p in cur["non_runtime_prefixes"]),
            "budget_gate_only": {f: v for f, v in gates.items() if v}}


def solving_code_files(membership: dict | None = None, *, root=None, extra=None) -> list[str]:
    """Files of the solving fingerprint: the default-in scope plus `GENERATION` and
    `JUDGING`, minus the membership's INFRA, judge layer and reasoned exclusions.
    `membership` is the batch's pinned table (default: the running code's)."""
    from .anchor import generation_files, judging_files
    m = membership or solving_membership()
    out = set(m.get("infra") or ()) | set(m.get("judge_layer") or ()) | set(m.get("excluded") or ())
    if extra is None:
        extra = set(generation_files()) | set(judging_files())
    return sorted((set(solving_candidates(root)) | set(extra)) - out)


def _fingerprint_of_files(files, root=None) -> str:
    from .anchor import _fingerprint_of
    return _fingerprint_of(files)


def solving_code_fingerprint(membership: dict | None = None) -> str:
    """Content hash of every runtime file that can decide what a solver is shown, how it is
    asked or how a cell is scored (`solving_code_files`)."""
    return _fingerprint_of_files(solving_code_files(membership))


def run_identity_v2(settings: dict, source: dict, source_files: dict, ledger_path,
                    membership: dict | None = None) -> dict:
    """What a batch is: what the model sees, how it is asked, who serves it.

    Not in it: `limit_usd` (the cap lives in the ledger), `haenv_git_sha` (recorded as
    `code_rev`; the code is pinned by content in `solving_code_sha16`), backend,
    endpoint, keys, stream and retries (recorded per receipt as `route`).
    """
    src = {k: v for k, v in (source or {}).items() if k != "haenv_git_sha"}
    src["solving_code_sha16"] = solving_code_fingerprint(membership)
    return {"protocol": PROTOCOL_V2,
            "answer_settings": {n: answer_settings_of(v) for n, v in sorted(settings.items())},
            "source": src, "source_files": source_files, "ledger": str(ledger_path)}


def upstream_declarations(saved: dict, current: dict) -> list | None:
    """The upstream declarations that are the only difference between two v2 identities.

    An upstream that was `<backend>:unpinned` (nothing said who serves the weights) may be
    declared later (config `upstream`, e.g. relay kimi-k3 = "Moonshot AI"); the
    answer is unchanged, the claim is new and recorded. Any other difference => None.
    """
    a, b = saved.get("answer_settings") or {}, current.get("answer_settings") or {}
    if {k: v for k, v in saved.items() if k != "answer_settings"} != \
            {k: v for k, v in current.items() if k != "answer_settings"} or set(a) != set(b):
        return None
    out = []
    for name in sorted(a):
        x, y = a[name], b[name]
        if x == y:
            continue
        if ({k: v for k, v in x.items() if k != "upstream"} != {k: v for k, v in y.items() if k != "upstream"}
                or not str(x.get("upstream")).endswith(":unpinned")):
            return None
        out.append({"solver": name, "from": x.get("upstream"), "to": y.get("upstream")})
    return out or None


def strip_code(ident: dict) -> dict:
    return {**ident, "source": {k: v for k, v in ident["source"].items() if k != "solving_code_sha16"}}


def project_v1(v1: dict, solving_code_sha16: str) -> dict:
    """v2 identity of a v1 identity: answer view of its settings, source without the git
    revision, no cap. The v1 identity has no content pin for the code; the current one is
    taken (the migration record names the v1 revision)."""
    src = {k: v for k, v in (v1.get("source") or {}).items() if k != "haenv_git_sha"}
    src["solving_code_sha16"] = solving_code_sha16
    return {"protocol": PROTOCOL_V2,
            "answer_settings": {n: answer_settings_of(v) for n, v in sorted(v1["settings"].items())},
            "source": src, "source_files": v1.get("source_files") or {}, "ledger": v1["ledger"]}


def migrate_v1_manifest(saved: dict, run_identity: dict, root: Path, code_rev, *,
                        legacy_root: str | None = None) -> None:
    """One-time v1 -> v2 migration of a batch manifest, in place (the caller persists it).

    The v1 run id is `hash(root, v1 identity)`; it is pinned here as `run_id`, so every
    cell id and request id of the batch stays what the v1 code derived and its receipts
    keep replaying. Accepted only when the v1 identity projects to the current v2 one.
    """
    v1 = saved["identity"]
    projected = project_v1(v1, run_identity["source"]["solving_code_sha16"])
    declared = None
    if projected != run_identity:
        declared = upstream_declarations(projected, run_identity)
        if declared is None:
            raise ValueError("Solver accounting run identity changed; use a new batch")
    # `legacy_root`: the path the v1 code ran under, when this is a copy of the batch.
    root_str = legacy_root or str(Path(root).resolve())
    legacy_run_id = _hash([root_str, v1])
    saved["identity_v1"] = v1
    saved["identity"] = run_identity
    saved["run_id"] = legacy_run_id
    saved["code_rev"] = v1["source"].get("haenv_git_sha")
    saved.setdefault("identity_migrations", []).append({
        "from": PROTOCOL_V1, "to": PROTOCOL_V2, "at": _now(),
        "v1_identity_sha256": _hash(v1), "run_id_pinned": legacy_run_id, "legacy_root": root_str,
        "v1_haenv_git_sha": v1["source"].get("haenv_git_sha"), "migrated_by_code_rev": code_rev,
        "dropped_from_identity": ["limit_usd", "source.haenv_git_sha", "settings.backend",
                                  "settings.endpoint_sha256", "settings.max_tokens_field",
                                  "settings.stream", "settings.retries", "settings.provider"],
        "why": "batch identity pins what the model sees and how it is asked; route, cap and "
               "code revision are recorded, not pinned (P1 identity split)",
        **({"upstream_declared": declared} if declared else {})})


def _case_ids(batch_dir: Path) -> list[str]:
    path = Path(batch_dir) / "cases.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line)["case_id"] for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]
