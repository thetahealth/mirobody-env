"""Verified New API gateway tariffs and request-ID-specific USD receipts."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import fnmatch
import json
import threading
import time
import urllib.request

from .semantic_budget import _amount
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data
from .run_state import log
from .call_log import (  # noqa: F401
    billed_tokens,
)

#: The key the current thread's attempt was sent with (set by the request accountant around
#: one attempt). The bill of that attempt must be read with that same key.
ATTEMPT = threading.local()


def attempt_key(default: str) -> str:
    return getattr(ATTEMPT, "key", None) or default


def usage_probe(base: str):
    """key -> used quota (`/dashboard/billing/usage` total_usage, free GET); raises when unreadable."""
    def probe(key: str):
        result = get_json(base, key, "/dashboard/billing/usage")
        used = result.get("total_usage")
        if not isinstance(used, (int, float)):
            raise ValueError("Relay usage unreadable")
        return _amount(used)
    return probe


def get_json(base: str, key: str, path: str) -> dict:
    request = urllib.request.Request(base.rstrip("/") + path,
        headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except Exception:
        raise ValueError("Relay billing metadata could not be read") from None
    if not isinstance(result, dict):
        raise ValueError("Relay billing endpoint returned no JSON object")
    return result


@dataclass(frozen=True)
class RelayPrices:
    model: str
    input_rate: Decimal
    output_rate: Decimal
    quota_per_usd: Decimal

    @classmethod
    def from_metadata(cls, data: dict, model: str):
        item = data["model"]
        if item.get("model_name") != model or item.get("quota_type") != 0:
            raise ValueError("Relay exact-model token pricing is required")
        if data.get("quota_display_type") != "USD":
            raise ValueError("Relay quota must be explicitly denominated in USD")
        unit = _amount(data["quota_per_unit"], positive=True)
        rules = data.get("billing_rule_summaries")
        if not isinstance(rules, dict) or any(fnmatch.fnmatchcase(model, pattern) for pattern in rules):
            raise ValueError("Unsupported relay billing-rule override")
        groups = data.get("group_ratio")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Relay group rate upper bound is missing")
        group = max(_amount(r, positive=True) for r in groups.values())
        base = _amount(item["model_ratio"], positive=True) * group / unit
        cache = _amount(item["cache_ratio"])
        output = _amount(item["completion_ratio"], positive=True)
        return cls(model, base * max(Decimal(1), cache), base * output, unit)

    def upper_bound(self, prompt: str, max_output_tokens: int) -> Decimal:
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("A positive solver output cap is required")
        # Bound all UTF-8 bytes as input tokens, plus envelope and rounding.
        # No model context limit or actual USD charge is invented here.
        return ((len(prompt.encode()) + 2048) * self.input_rate
                + max_output_tokens * self.output_rate + 1 / self.quota_per_usd)


def fetch_metadata(cfg: dict, solver) -> dict:
    from .solvers import BACKENDS, _ensure_backends_registered
    _ensure_backends_registered(cfg)
    base = BACKENDS["relay"].get("quota_url")
    if not base:
        raise ValueError("Relay route has no billing endpoint")
    prices = get_json(base, solver.api_key, "/api/pricing")
    status = get_json(base, solver.api_key, "/api/status")
    if prices.get("success") is not True or status.get("success") is not True:
        raise ValueError("Relay price/status lookup did not succeed")
    matches = [row for row in prices.get("data", []) if row.get("model_name") == solver.model]
    if len(matches) != 1:
        raise ValueError("Relay has no unique exact-model tariff")
    item = matches[0]
    result = {"model": {k: item.get(k) for k in ("model_name", "quota_type", "model_ratio",
                                                "completion_ratio", "cache_ratio")},
              "group_ratio": prices.get("group_ratio"),
              "billing_rule_summaries": prices.get("billing_rule_summaries"),
              "pricing_version": prices.get("pricing_version"),
              "quota_per_unit": (status.get("data") or {}).get("quota_per_unit"),
              "quota_display_type": (status.get("data") or {}).get("quota_display_type")}
    RelayPrices.from_metadata(result, solver.model)
    return result


class RelayBilling:
    def __init__(self, prices: RelayPrices, fetch_logs):
        self.prices, self.fetch_logs = prices, fetch_logs

    @classmethod
    def for_solver(cls, prices, solver, cfg):
        from .solvers import BACKENDS, _ensure_backends_registered
        _ensure_backends_registered(cfg)
        base = BACKENDS["relay"].get("quota_url")
        if not base:
            raise ValueError("Relay route has no billing endpoint")
        def logs():
            # The attempt's own key: a rotated key's bill is not in another key's log.
            result = get_json(base, attempt_key(solver.api_key), "/api/log/token")
            if result.get("success") is not True or not isinstance(result.get("data"), list):
                raise ValueError("Relay log lookup did not succeed")
            return result["data"]
        return cls(prices, logs)

    def __call__(self, response: dict) -> dict:
        request_id = response.get("_gateway_request_id") if isinstance(response, dict) else None
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Response lacks the gateway request ID; cannot match its bill")
        matches = []
        for attempt in range(3):
            matches = [r for r in self.fetch_logs() if r.get("request_id") == request_id]
            if matches:
                break
            if attempt < 2:
                time.sleep(1)  # Only free log reads retry; never a new completion.
        if len(matches) != 1:
            raise ValueError("No unique billing log for this gateway request ID")
        row = matches[0]
        if row.get("model_name") != self.prices.model or row.get("type") != 2:
            raise ValueError("Billing log does not describe this model's completed request")
        quota = _amount(row["quota"])
        if quota != quota.to_integral_value():
            raise ValueError("Invalid non-integral relay quota deduction")
        # Exact gateway bill, sanitized to exclude user/key/IP and raw content.
        return {"kind": "relay_quota_receipt", "request_id": request_id,
                "model": self.prices.model, "quota": str(quota),
                "quota_per_usd": str(self.prices.quota_per_usd),
                "actual_usd": str(quota / self.prices.quota_per_usd)}


# The relay balance gate. It lives here (INFRA) so that changing how much money a resumed run
# asks for never moves the solving-code or judging fingerprints; `evaluate._preflight_quota`
# only delegates.
def preflight_quota(job, cfg, solvers, n_cases: int, n_done: int, *, resp_path=None) -> None:
    """Check the relay balance before a batch starts and refuse to start if it is short.

    Covers only `relay` models, and only when `config.backends.relay.quota_url` is set
    (OneAPI-style billing endpoints); other backends are listed in a warning. If the balance
    cannot be read, it warns and continues. `resp_path` is the run's `responses.jsonl`; the
    cells already done are read next to it.
    """
    from .solvers import BACKENDS
    from .run_state import log
    from .resume import _done_keys
    from . import data_root as _dr
    _nb = [n for n, _ in solvers
           if (cfg.get("models", {}).get(n) or {}).get("backend") == "relay"]
    _uncov: dict[str, list[str]] = {}
    for _n, _ in solvers:
        _bk = (cfg.get("models", {}).get(_n) or {}).get("backend")
        if _bk and _bk != "relay":
            _uncov.setdefault(_bk, []).append(_n)
    if _uncov:
        log.warning("[preflight] balance is checked for the relay only; not covered: %s -- "
                    "running out of quota on these backends shows up as 4xx mid-run",
                    {k: sorted(v) for k, v in _uncov.items()})
    if not _nb:
        return
    _rdef = BACKENDS.get("relay") or {}
    _qurl = str(_rdef.get("quota_url") or "").rstrip("/")
    if not _qurl:
        log.info("[quota] no billing endpoint configured for the relay "
                 "(config.backends.relay.quota_url) -- balance not checked")
        return
    _est, _basis = 0.0, "per-cell constant table (rough)"
    # a resumed batch only needs money for the relay cells it has not answered yet
    done = _done_keys(resp_path.with_name("eval.jsonl")) if resp_path else set()
    _todo = {n: max(0, n_cases - sum(1 for k in done if str(k).rsplit("|", 1)[-1] == n)) for n in _nb}
    n_cases = max(_todo.values())
    _tok = _measured_tokens_per_grid(job, _nb)
    if _tok:
        _est = sum(_tok.get(n, 0.0) * _todo[n] for n in _nb) / 1e6 * RELAY_USD_PER_MTOK
        _basis = (f"measured tokens/cell {({k: int(v) for k, v in _tok.items()})} x "
                  f"${RELAY_USD_PER_MTOK}/M")
    if _est <= 0:
        _est = sum(RELAY_COST_PER_GRID.get(n, RELAY_COST_DEFAULT) * _todo[n] for n in _nb)
    _need = _est * RELAY_COST_HEADROOM
    try:
        from pathlib import Path as _P
        _ensure_on_path(str(_dr()))
        from tools.relay_quota import _keys, _get
        import os as _os
        _envf = _P(_os.path.expanduser(str(cfg.get("env_file", ""))))
        _host = str(_rdef.get("no_proxy_host") or "")
        if _host:
            _bypass_proxy(_host)
        _prefix = str(_rdef.get("key_env_pool") or "RELAY_KEY_")
        _left, _alive = 0.0, 0
        for _nm, _k in _keys(_envf, _prefix):
            _s = _get(_qurl, _k, "/v1/dashboard/billing/subscription")
            _u = _get(_qurl, _k, "/dashboard/billing/usage")
            _hl, _tu = _s.get("hard_limit_usd"), _u.get("total_usage")
            if isinstance(_hl, (int, float)) and isinstance(_tu, (int, float)):
                _r = _hl - _tu / 100.0
                if _r > 1.0:
                    _left += _r
                    _alive += 1
    except Exception as e:                                   # noqa: BLE001
        log.warning("[quota] could not read the relay balance (%s) -- warning only; "
                    "the usual cause is a dead key, which the key pool rotates past", e)
        return
    log.info("[quota] relay models %s · %d cases => estimated $%.2f (basis: %s; with %.1fx "
             "retry headroom $%.2f) · usable keys %d · balance $%.2f",
             _nb, n_cases, _est, _basis, RELAY_COST_HEADROOM, _need, _alive, _left)
    if _left < _need:
        raise RuntimeError(
            f"Relay balance too low; refusing to start. Balance ${_left:.2f} < estimated "
            f"need ${_need:.2f} (unit cost x {n_cases} cases x {len(_nb)} models x "
            f"{RELAY_COST_HEADROOM:.1f} retry headroom).\n"
            f"   A run that breaks part-way leaves a half-finished reading that looks the "
            f"same as a complete one.\n"
            f"   Options: add keys / switch backend / narrow with --models; "
            f"per-key balance: `uv run python tools/relay_quota.py`.")


def _ensure_on_path(path: str) -> None:
    """Put `path` on sys.path once (repeated preflights must not grow it)."""
    import sys
    if path not in sys.path:
        sys.path.insert(0, path)


def _bypass_proxy(host: str) -> None:
    """Add `host` to no_proxy / NO_PROXY once (repeated preflights must not grow them)."""
    import os
    items = [h for h in os.environ.get("no_proxy", "").split(",") if h]
    if host not in items:
        items.append(host)
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(items)


#: Per-cell unit cost in USD (~20k-token prompts) for relay-billed models; a rough
#: affordability check, not billing.
RELAY_COST_PER_GRID = {"kimi-k3": 0.2374, "terra": 0.0592}


#: Unit cost for unlisted relay-billed models (errs high).
RELAY_COST_DEFAULT = 0.15


RELAY_COST_HEADROOM = 2.0


#: Normalized rate (USD per million tokens) for the affordability check; errs high, since a
#: run that breaks midway leaves a partial reading.
RELAY_USD_PER_MTOK = 4.0


def _measured_tokens_per_grid(job, models: list[str]) -> dict[str, float]:
    """Each model's measured tokens per cell from this job's earlier batches (from `billed`,
    else `usage`); empty if there are none.
    """
    try:
        import json as _j
        _root =_dr()
        base = _root / "results" / getattr(job, "task_type", "") / getattr(job, "job_id", "")
        if not base.is_dir():
            return {}
        # The four most recent batches that contain the estimated models (offline smoke batches
        # would otherwise crowd them out).
        cands = [d for d in base.iterdir()
                 if d.is_dir() and (d / "responses.jsonl").is_file()]
        agg: dict[str, dict] = {}
        _want = {str(m) for m in (models or [])}
        _with_usage = []
        for _d in sorted(cands, key=lambda x: x.name, reverse=True):
            try:
                with (_d / "responses.jsonl").open(encoding="utf-8", errors="ignore") as _fh:
                    if any(_l.strip() and any(f'"{m}"' in _l for m in _want) for _l in _fh):
                        _with_usage.append(_d)
            except Exception:                          # noqa: BLE001
                continue
            if len(_with_usage) >= 4:
                break
        for d in _with_usage:
            rp = d / "responses.jsonl"
            for line in rp.read_text(encoding="utf-8", errors="ignore").split("\n"):
                if not line.strip():
                    continue
                try:
                    r = _j.loads(line)
                except Exception:                               # noqa: BLE001
                    continue
                if r.get("solver") not in models:
                    continue
                b = r.get("billed") or billed_tokens(r.get("usage"))
                e = agg.setdefault(r["solver"], {"tok": 0, "grids": set()})
                e["tok"] += int(b.get("in_total", 0)) + int(b.get("out", 0))
                e["grids"].add((r.get("case"), d.name))
        return {m: e["tok"] / max(1, len(e["grids"])) for m, e in agg.items() if e["tok"]}
    except Exception as e:                                      # noqa: BLE001
        log.warning("[quota] failed to compute tokens live from historical batches (%s) —— "
                    "falling back to the unit-price table", e)
        return {}
