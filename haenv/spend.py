"""`haenv spend`: read-only spend dashboard over a shared budget ledger, and `haenv spend reconcile`.

The dashboard never writes the ledger and never takes its lock: the ledger is replaced
atomically (`BudgetLedger._write`: temp file + `os.replace`), so a plain read is a consistent
snapshot. Attribution of ledger entries to provider / model / batch / route comes from the
receipts on disk (`solver-accounting/receipts`, `semantic*/receipts`), keyed by the request id
(solver receipts are named `sha256(request_id)`, judge receipts carry `request.sample_id`); a
ledger entry may also carry `tags` (backend, model, batch_uid, route, key_ordinal, kind), which
win when present. Every ledger entry lands in exactly one group, entries without a receipt in
an `(unattributed)` group, so each grouping sums to the ledger's committed total exactly.

`reconcile` turns an upper bound into a paid amount only when evidence exists, and only through
`BudgetLedger.settle_unknown_later`; without `--apply` it writes nothing.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ZERO = Decimal("0")
UNATTRIBUTED = "(unattributed)"
STATUS_ORDER = ("settled", "tariff_capped", "unknown_capped", "unknown", "reserved")
DIMENSIONS = ("provider", "model", "batch", "key", "route", "kind")
DEFAULT_BY = ("provider", "model", "batch", "route")
JUDGE_BACKEND = "openrouter"      # semantic judge receipts do not record a backend


def _dec(value) -> Decimal:
    return Decimal(str(value)) if value is not None else ZERO


# --------------------------------------------------------------------------- ledger (read only)

def read_ledger(path: Path) -> dict:
    """Plain read of the ledger file: no lock, no lock file, no write."""
    return json.loads(Path(path).read_text())


def counted(rec: dict) -> Decimal:
    """What the budget counts for one entry (mirrors `BudgetLedger._committed`)."""
    return _dec(rec["actual_usd"] if rec.get("actual_usd") is not None else rec["reserved_usd"])


def committed_total(state: dict) -> Decimal:
    return sum((counted(r) for r in state["requests"].values()), ZERO)


# --------------------------------------------------------------------------- attribution index

@dataclass
class Info:
    batch: str = UNATTRIBUTED
    kind: str = "?"                 # solver | judge | ?
    backend: str = UNATTRIBUTED
    model: str = UNATTRIBUTED
    provider: str | None = None     # upstream pin, e.g. "Moonshot AI"
    key: str = "-"
    receipt: Path | None = None
    failed: Path | None = None
    failure: dict | None = None     # parsed .failed.json
    source: str = "none"            # tags | receipt | failed | lock | id-prefix

    @property
    def route(self) -> str:
        if self.backend == UNATTRIBUTED:
            return UNATTRIBUTED
        return self.backend + (f"/{self.provider}" if self.provider else "")


@dataclass
class BatchScan:
    name: str
    path: Path
    kind: str = "?"                 # timeline | workup | ?
    models: list = field(default_factory=list)
    n_receipts: int = 0


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _settings_info(settings: dict) -> tuple[str, str, str | None]:
    prov = settings.get("provider")
    only = prov.get("only") if isinstance(prov, dict) else None
    return (settings.get("backend") or UNATTRIBUTED, settings.get("model") or UNATTRIBUTED,
            only[0] if isinstance(only, list) and only else None)


def _solver_cell(rid: str) -> str | None:
    """Cell of a solver request id: `solver:{cell}:{n}` or `solver:v2:{cell}:{prompt16}:{n}`."""
    parts = rid.split(":")
    if parts[0] != "solver":
        return None
    if len(parts) == 3:
        return parts[1]
    if len(parts) == 5 and parts[1] == "v2":
        return parts[2]
    return None


def scan_batch(batch: Path, index: dict, locks: dict, cells: dict) -> BatchScan:
    """Add one batch's receipts to `index` (request id -> Info); lock-only files to `locks`."""
    batch = Path(batch)
    scan = BatchScan(name=f"{batch.parent.name}/{batch.name}", path=batch,
                     kind="timeline" if (batch / "slices.json").is_file() else "workup")
    meta = _load(batch / "batch.json") or {}
    scan.models = list(meta.get("models") or [])
    solver_dir = batch / "solver-accounting" / "receipts"
    if solver_dir.is_dir():
        for entry in os.scandir(solver_dir):
            name = entry.name
            if name.endswith(".lock"):
                locks[name[:-5]] = scan.name
                continue
            if name.endswith(".billing.json") or not name.endswith(".json"):
                continue
            failed = name.endswith(".failed.json")
            data = _load(Path(entry.path))
            request = (data or {}).get("request") or {}
            rid = request.get("request_id")
            if not rid:
                continue
            # Receipts since the identity split keep the answer view in `request.settings`
            # and the route (backend, wire model, provider pin) beside it in `route`.
            backend, model, provider = _settings_info({**(request.get("settings") or {}),
                                                       **((data or {}).get("route") or {})})
            info = index.get(rid) or Info()
            if info.source in ("receipt",) and not failed:
                continue
            info.batch, info.kind, info.backend, info.model, info.provider = (
                scan.name, "solver", backend, model, provider)
            if failed:
                info.failed, info.failure = Path(entry.path), data
                info.source = info.source if info.source == "receipt" else "failed"
            else:
                info.receipt, info.source = Path(entry.path), "receipt"
                scan.n_receipts += 1
            index[rid] = info
            cell = _solver_cell(rid)
            if cell is not None:
                cells[cell] = (scan.name, backend, model, provider)
    for rdir in sorted(batch.glob("semantic*/**/receipts")):
        for entry in os.scandir(rdir):
            if not entry.name.endswith(".json") or entry.name.endswith(".billing.json"):
                continue
            data = _load(Path(entry.path))
            request = (data or {}).get("request") or {}
            rid = request.get("sample_id")
            if not rid or rid in index:
                continue
            index[rid] = Info(batch=scan.name, kind="judge", backend=JUDGE_BACKEND,
                              model=request.get("model") or UNATTRIBUTED,
                              receipt=Path(entry.path), source="receipt")
            scan.n_receipts += 1
    return scan


def discover_batches(root: Path) -> list[Path]:
    """Batch directories (holding solver receipts or semantic receipts) under `root`."""
    root, found = Path(root), []
    for dirpath, dirnames, _files in os.walk(root):
        p = Path(dirpath)
        if (p / "solver-accounting").is_dir() or any(True for _ in p.glob("semantic*/**/receipts")):
            found.append(p)
            dirnames[:] = []
    return sorted(found)


def build_index(batches: list[Path]) -> tuple[dict, dict, list[BatchScan]]:
    index, locks, cells, scans = {}, {}, {}, []
    for b in batches:
        scans.append(scan_batch(b, index, locks, cells))
    index["__cells__"] = cells      # popped by attribute()
    index["__locks__"] = locks
    return index, {"cells": cells, "locks": locks}, scans


def attribute(rid: str, rec: dict, index: dict) -> Info:
    """Where did this ledger entry's money go: tags, else receipts, else lock file, else prefix."""
    tags = rec.get("tags")
    base = index.get(rid)
    if isinstance(tags, dict) and tags:
        info = Info(batch=tags.get("batch_uid") or (base.batch if base else UNATTRIBUTED),
                    kind=tags.get("kind") or (base.kind if base else "?"),
                    backend=tags.get("backend") or (base.backend if base else UNATTRIBUTED),
                    model=tags.get("model") or (base.model if base else UNATTRIBUTED),
                    provider=(base.provider if base else None),
                    key=str(tags["key_ordinal"]) if tags.get("key_ordinal") is not None else "-",
                    receipt=base.receipt if base else None, failed=base.failed if base else None,
                    failure=base.failure if base else None, source="tags")
        if tags.get("route") and "/" in str(tags["route"]):
            info.provider = str(tags["route"]).split("/", 1)[1]
        return info
    if base is not None:
        return base
    parts = rid.split(":")
    if _solver_cell(rid) is not None:
        sha = hashlib.sha256(rid.encode()).hexdigest()
        batch = index["__locks__"].get(sha)
        if batch is not None:
            _b, backend, model, provider = index["__cells__"].get(_solver_cell(rid), (batch, UNATTRIBUTED, UNATTRIBUTED, None))
            return Info(batch=batch, kind="solver", backend=backend, model=model, provider=provider,
                        source="lock")
        return Info(kind="solver", source="id-prefix")
    if len(parts[0]) == 64 and len(parts) == 2:
        return Info(kind="judge", source="id-prefix")
    return Info(source="none")


# --------------------------------------------------------------------------- aggregation

def _new_row() -> dict:
    return {"n": 0, "settled_n": 0, "settled_usd": ZERO, "tariff_n": 0, "tariff_usd": ZERO,
            "unknown_n": 0, "unknown_usd": ZERO, "reserved_n": 0, "reserved_usd": ZERO,
            "committed_usd": ZERO, "settled_reserved_usd": ZERO}


def _add(row: dict, rec: dict) -> None:
    status = rec["status"]
    row["n"] += 1
    row["committed_usd"] += counted(rec)
    if status == "settled":
        row["settled_n"] += 1
        row["settled_usd"] += _dec(rec["actual_usd"])
        row["settled_reserved_usd"] += _dec(rec["reserved_usd"])
    elif status == "tariff_capped":
        row["tariff_n"] += 1
        row["tariff_usd"] += _dec(rec["reserved_usd"])
    elif status in ("unknown_capped", "unknown"):
        row["unknown_n"] += 1
        row["unknown_usd"] += _dec(rec["reserved_usd"])
    elif status == "reserved":
        row["reserved_n"] += 1
        row["reserved_usd"] += _dec(rec["reserved_usd"])


def _dim_value(dim: str, info: Info) -> str:
    return {"provider": info.backend, "model": info.model, "batch": info.batch,
            "key": info.key, "route": info.route, "kind": info.kind}[dim]


def _reason_of(rec: dict, info: Info) -> str:
    text = rec.get("reconciliation") or ""
    reason = text.split(";", 1)[0] if text else "(no reason recorded)"
    fail = info.failure
    if fail:
        cls = fail.get("error_class") or "?"
        status = fail.get("http_status")
        return f"{reason} / {cls}" + (f" {status}" if status else "")
    return reason


def aggregate(state: dict, index: dict, by=DEFAULT_BY) -> dict:
    total = _new_row()
    groups = {d: defaultdict(_new_row) for d in by}
    reasons = defaultdict(lambda: {"n": 0, "bound_usd": ZERO})
    already = {"settled_after_unknown_n": 0, "settled_after_unknown_usd": ZERO,
               "settled_after_unknown_bound_usd": ZERO, "rejected_zero_n": 0}
    unattributed = _new_row()
    for rid, rec in state["requests"].items():
        info = attribute(rid, rec, index)
        _add(total, rec)
        if info.batch == UNATTRIBUTED:
            _add(unattributed, rec)
        for d in by:
            _add(groups[d][_dim_value(d, info)], rec)
        if rec["status"] in ("unknown_capped", "unknown"):
            r = reasons[_reason_of(rec, info)]
            r["n"] += 1
            r["bound_usd"] += _dec(rec["reserved_usd"])
        if rec.get("settled_after_unknown"):
            already["settled_after_unknown_n"] += 1
            already["settled_after_unknown_usd"] += _dec(rec["actual_usd"])
            already["settled_after_unknown_bound_usd"] += _dec(rec["reserved_usd"])
        if rec.get("zero_cost_basis"):
            already["rejected_zero_n"] += 1
    return {"total": total, "groups": {d: dict(g) for d, g in groups.items()},
            "unknown_reasons": dict(reasons), "already_reconciled": already,
            "unattributed": unattributed}


# --------------------------------------------------------------------------- balances

def _http_get(url: str, key: str, *, proxy: str | None, timeout: int = 30):
    import urllib.request
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.build_opener(handler).open(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_balances(env: dict | None = None, get=None, relay_base: str | None = None,
                   relay_key_prefix: str | None = None,
                   or_base: str = "https://openrouter.ai/api/v1", or_proxy: str | None = None) -> dict:
    """Free read-only balance GETs. Key values are only ever used as headers, never returned.

    The relay's billing base and key pool prefix come from the deployment config
    (`backends.relay.quota_url` / `key_env_pool`); without them relay balances are skipped."""
    env = os.environ if env is None else env
    or_proxy = or_proxy or env.get("HAENV_OPENROUTER_PROXY") or env.get("https_proxy") or env.get("HTTPS_PROXY")
    get = get or (lambda url, key, via_proxy: _http_get(url, key, proxy=or_proxy if via_proxy else None))
    out = {"openrouter": {}, "relay": {}, "google": "n/a (tariff bound)", "dashscope": "n/a (tariff bound)"}
    key = env.get("OPENROUTER_API_KEY")
    if key:
        try:
            credits = get(f"{or_base}/credits", key, True)["data"]
            total, used = _dec(credits["total_credits"]), _dec(credits["total_usage"])
            out["openrouter"]["account"] = {"credits_usd": str(total), "used_usd": str(used),
                                            "left_usd": str(total - used)}
        except Exception as e:  # noqa: BLE001
            out["openrouter"]["account"] = {"error": type(e).__name__}
        try:
            kd = get(f"{or_base}/key", key, True)["data"]
            out["openrouter"]["key"] = {"limit_usd": kd.get("limit"), "used_usd": kd.get("usage"),
                                        "left_usd": kd.get("limit_remaining"),
                                        "used_daily_usd": kd.get("usage_daily")}
        except Exception as e:  # noqa: BLE001
            out["openrouter"]["key"] = {"error": type(e).__name__}
    else:
        out["openrouter"] = "n/a (no key in environment)"
    if not relay_base or not relay_key_prefix:
        out["relay"] = "n/a (relay billing not configured)"
        return out
    pre = relay_key_prefix
    for name in sorted(k for k in env if k.startswith(pre) and k[len(pre):].isdigit()):
        ordinal = name[len(pre):]
        try:
            sub = get(relay_base.rstrip("/") + "/v1/dashboard/billing/subscription", env[name], False)
            use = get(relay_base.rstrip("/") + "/v1/dashboard/billing/usage", env[name], False)
            limit, used = _dec(sub["hard_limit_usd"]), _dec(use["total_usage"]) / 100
            out["relay"][ordinal] = {"limit_usd": str(limit), "used_usd": str(used),
                                     "left_usd": str(limit - used)}
        except Exception as e:  # noqa: BLE001
            out["relay"][ordinal] = {"error": type(e).__name__}
    return out


# --------------------------------------------------------------------------- forecast

def _match_model(provider_model: str, names: list[str]) -> str | None:
    for n in sorted(names, key=len, reverse=True):
        if provider_model == n or provider_model.endswith("/" + n) or provider_model.startswith(n) \
                or provider_model.split("/")[-1].startswith(n):
            return n
    return None


def _batch_progress(scan: BatchScan, window_min: float, now: datetime) -> dict:
    """Scope, cells done and recent completion rate of one batch (files read only)."""
    b = scan.path
    models = scan.models
    rows = []
    try:
        with open(b / "responses.jsonl") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except OSError:
        pass
    real = set(models)
    gated = [r for r in rows if str(r.get("solver", "")).startswith("gated_")]
    slices = (_load(b / "slices.json") or {}).get("by_case") or {}
    if scan.kind == "timeline":
        scope_cases = {r["case"] for r in gated} or set(slices)
        units_per_model = sum(len(slices.get(c, [])) for c in scope_cases)
        done = {(r["case"], r["solver"], r.get("slice_t")) for r in rows if r.get("solver") in real}
    else:
        scope_cases = {r["case"] for r in gated} or ({json.loads(l)["case_id"] for l in open(b / "cases.jsonl")}
                                                      if (b / "cases.jsonl").is_file() else set())
        units_per_model = len(scope_cases)
        done = {(r["case"], r["solver"]) for r in rows if r.get("solver") in real}
    done_by_model = defaultdict(int)
    for d in done:
        done_by_model[d[1]] += 1
    cutoff = now - timedelta(minutes=window_min)
    recent = 0
    try:
        with open(b / "trace.jsonl") as fh:
            for line in fh:
                if '"case/end"' not in line:
                    continue
                ev = json.loads(line)
                if ev.get("solver") in real and datetime.fromisoformat(ev["ts"]) >= cutoff:
                    recent += 1
    except OSError:
        pass
    return {"kind": scan.kind, "models": models, "scope_units_per_model": units_per_model,
            "expected_units": units_per_model * len(models), "done_units": len(done),
            "done_by_model": dict(done_by_model), "recent_completions": recent,
            "rate_per_min": recent / window_min if window_min else 0.0}


def forecast(state: dict, index: dict, scans: list[BatchScan], window_min: float = 30.0,
             now: datetime | None = None) -> dict:
    """Estimated cost/time to finish: remaining cells x pooled per-cell price. Labelled estimate."""
    now = now or datetime.now(timezone.utc)
    progress = {s.name: _batch_progress(s, window_min, now) for s in scans}
    spend = defaultdict(lambda: {"solver": ZERO, "judge": ZERO})     # (task kind, model) -> counted
    for rid, rec in state["requests"].items():
        if rec["status"] == "reserved":
            continue
        info = attribute(rid, rec, index)
        sc = next((s for s in scans if s.name == info.batch), None)
        if sc is None:
            continue
        model = _match_model(info.model, sc.models) if info.kind == "solver" else "*judge*"
        if model:
            spend[(sc.kind, model)][info.kind] += counted(rec)
    pooled_done = defaultdict(int)          # (task kind, model) -> done cells
    judge_done = defaultdict(int)           # task kind -> done cells
    for s in scans:
        for m, n in progress[s.name]["done_by_model"].items():
            pooled_done[(s.kind, m)] += n
            judge_done[s.kind] += n
    judge_cost = defaultdict(lambda: ZERO)
    for (k, m), v in spend.items():
        if m == "*judge*":
            judge_cost[k] += v["judge"]
    out = []
    for s in scans:
        p = progress[s.name]
        rows, batch_est = [], ZERO
        for m in s.models:
            remaining = max(p["scope_units_per_model"] - p["done_by_model"].get(m, 0), 0)
            done = pooled_done.get((s.kind, m), 0)
            unit = (spend[(s.kind, m)]["solver"] / done) if done else None
            judge_unit = (judge_cost[s.kind] / judge_done[s.kind]) if judge_done[s.kind] else ZERO
            est = (remaining * (unit + judge_unit)) if unit is not None else None
            if est is not None:
                batch_est += est
            rows.append({"model": m, "remaining_cells": remaining, "unit_solver_usd": unit,
                         "unit_judge_usd": judge_unit, "est_remaining_usd": est})
        remaining_total = max(p["expected_units"] - p["done_units"], 0)
        eta = (remaining_total / p["rate_per_min"]) if p["rate_per_min"] else None
        out.append({"batch": s.name, "kind": s.kind, **{k: p[k] for k in (
            "expected_units", "done_units", "recent_completions", "rate_per_min")},
            "remaining_cells": remaining_total, "eta_min": eta, "est_remaining_usd": batch_est,
            "models": rows})
    return {"window_min": window_min, "batches": out,
            "est_remaining_usd": sum((b["est_remaining_usd"] for b in out), ZERO),
            "note": "estimate: remaining cells x pooled per-cell price of finished cells (solver + judge, "
                    "counted at bound where unknown); in-flight reservations are already inside committed"}


# --------------------------------------------------------------------------- report

def build_report(ledger: Path, batches: list[Path], *, by=DEFAULT_BY, balances: dict | None = None,
                 window_min: float = 30.0, with_forecast: bool = True, now: datetime | None = None) -> dict:
    state = read_ledger(ledger)
    index, _aux, scans = build_index(batches)
    agg = aggregate(state, index, by=by)
    committed = committed_total(state)
    for d, g in agg["groups"].items():           # the invariant: each grouping sums to committed
        s = sum((r["committed_usd"] for r in g.values()), ZERO)
        if s != committed:
            raise AssertionError(f"grouping by {d} sums to {s}, ledger committed is {committed}")
    report = {"ledger": str(ledger), "limit_usd": str(state["limit_usd"]), "halt_reason": state["halt_reason"],
              "n_requests": len(state["requests"]), "committed_usd": committed,
              "headroom_usd": _dec(state["limit_usd"]) - committed, "batches_scanned": [s.name for s in scans],
              **agg}
    if balances is not None:
        report["balances"] = balances
    if with_forecast and scans:
        report["forecast"] = forecast(state, index, scans, window_min, now)
    return report


def _usd(x, nd=2) -> str:
    return "-" if x is None else f"${_dec(x):,.{nd}f}"


def _table(title: str, header: list[str], rows: list[list]) -> list[str]:
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines + [""]


def render_markdown(rep: dict) -> str:
    t = rep["total"]
    out = [f"# haenv spend", "",
           f"- ledger: `{rep['ledger']}` (read only, no lock taken)",
           f"- limit {_usd(rep['limit_usd'])} | committed **{_usd(rep['committed_usd'], 4)}** | "
           f"headroom {_usd(rep['headroom_usd'], 2)} | halt: {rep['halt_reason']} | entries {rep['n_requests']}",
           f"- batches scanned: {', '.join(rep['batches_scanned']) or '(none)'}", "",
           "Columns: paid = settled actual; bound = tariff_capped + unknown_capped counted at their upper "
           "bound; in flight = reserved; reserved/paid = over-reservation on settled entries.", ""]
    header = ["group", "n", "paid (n)", "paid $", "tariff bound (n)", "tariff $", "unknown (n)",
              "unknown bound $", "in flight (n)", "in flight $", "committed $", "reserved/paid"]

    def row(name, r):
        ratio = (r["settled_reserved_usd"] / r["settled_usd"]) if r["settled_usd"] else None
        return [name, r["n"], r["settled_n"], _usd(r["settled_usd"], 4), r["tariff_n"], _usd(r["tariff_usd"], 4),
                r["unknown_n"], _usd(r["unknown_usd"], 4), r["reserved_n"], _usd(r["reserved_usd"], 4),
                _usd(r["committed_usd"], 4), "-" if ratio is None else f"{ratio:.1f}x"]
    for dim, g in rep["groups"].items():
        rows = [row(k, v) for k, v in sorted(g.items(), key=lambda kv: -kv[1]["committed_usd"])]
        rows.append(row("**total**", t))
        out += _table(f"By {dim}", header, rows)
    ua = rep["unattributed"]
    out += [f"Unattributed (no receipt/tag found in the scanned batches): {ua['n']} entries, "
            f"{_usd(ua['committed_usd'], 4)} of {_usd(rep['committed_usd'], 4)}", ""]
    ar = rep["already_reconciled"]
    out += [f"Already reconciled: {ar['settled_after_unknown_n']} unknown entries settled afterwards "
            f"(paid {_usd(ar['settled_after_unknown_usd'], 4)} against bound "
            f"{_usd(ar['settled_after_unknown_bound_usd'], 4)}); {ar['rejected_zero_n']} rejected attempts settled at $0.", ""]
    out += _table("Unknown-cost entries by reason / failure class", ["reason / class", "n", "bound $"],
                  [[k, v["n"], _usd(v["bound_usd"], 4)] for k, v in sorted(rep["unknown_reasons"].items(),
                                                                          key=lambda kv: -kv[1]["bound_usd"])])
    if "balances" in rep:
        b = rep["balances"]
        rows = []
        orr = b["openrouter"]
        if isinstance(orr, dict):
            a, k = orr.get("account", {}), orr.get("key", {})
            rows.append(["openrouter account", a.get("credits_usd", "-"), a.get("used_usd", "-"), a.get("left_usd", a.get("error", "-"))])
            rows.append(["openrouter key", k.get("limit_usd", "-"), k.get("used_usd", "-"), k.get("left_usd", k.get("error", "-"))])
        else:
            rows.append(["openrouter", "-", "-", orr])
        for ordinal, v in b["relay"].items():
            rows.append([f"relay key {ordinal}", v.get("limit_usd", "-"), v.get("used_usd", "-"), v.get("left_usd", v.get("error", "-"))])
        rows += [["google", "-", "-", b["google"]], ["dashscope", "-", "-", b["dashscope"]]]
        out += _table("Balances (USD)", ["account", "limit / credits", "used", "left"], rows)
    if "forecast" in rep:
        f = rep["forecast"]
        out += [f"## Forecast (estimate, rate window {f['window_min']:.0f} min)", "", f["note"], ""]
        out += _table("Per batch", ["batch", "kind", "cells done/expected", "remaining", f"cells/min (last {f['window_min']:.0f} min)",
                                    "ETA (min)", "est. remaining $"],
                      [[b["batch"], b["kind"], f"{b['done_units']}/{b['expected_units']}", b["remaining_cells"],
                        f"{b['rate_per_min']:.2f}", "-" if b["eta_min"] is None else f"{b['eta_min']:.0f}",
                        _usd(b["est_remaining_usd"], 2)] for b in f["batches"]])
        out += [f"Estimated remaining spend: **{_usd(f['est_remaining_usd'], 2)}**; projected total "
                f"{_usd(_dec(rep['committed_usd']) + f['est_remaining_usd'], 2)} against limit {_usd(rep['limit_usd'])}.", ""]
    return "\n".join(out)


def _jsonable(x):
    if isinstance(x, Decimal):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# --------------------------------------------------------------------------- reconcile

@dataclass
class Action:
    request_id: str
    decision: str               # settle | refuse_over_bound | skip
    bound_usd: Decimal
    actual_usd: Decimal | None = None
    kind: str = ""
    evidence: str = ""
    evidence_raw: bytes | None = None
    reason: str = ""
    model: str = ""
    batch: str = ""


class GenerationLookup:
    """OpenRouter GET /api/v1/generation?id= ; returns (status, raw bytes)."""

    def __init__(self, key: str, proxy: str | None, base="https://openrouter.ai/api/v1", attempts=6, delay_s=5.0):
        self.key, self.proxy, self.base, self.attempts, self.delay_s = key, proxy, base, attempts, delay_s

    def __call__(self, generation_id: str):
        import urllib.error
        import urllib.parse
        import urllib.request
        handler = urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy} if self.proxy else {})
        opener = urllib.request.build_opener(handler)
        url = f"{self.base}/generation?id={urllib.parse.quote(generation_id)}"
        misses = 0
        for _ in range(self.attempts):
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.key}"})
                with opener.open(req, timeout=30) as resp:
                    return 200, resp.read()
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    return e.code, b""
                misses += 1
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(self.delay_s)
        return (404 if misses == self.attempts else 0), b""


def _generation_id(info: Info) -> tuple[str | None, str]:
    if info.failure and info.failure.get("generation_id"):
        return info.failure["generation_id"], "failed.json"
    if info.receipt is not None and info.backend == "openrouter":
        saved = _load(info.receipt) or {}
        resp = saved.get("response")
        if isinstance(resp, dict):
            gid = resp.get("_generation_id") or resp.get("id")
            if isinstance(gid, str) and gid:
                return gid, "receipt"
    return None, ""


def plan_reconcile(state: dict, index: dict, lookup=None, *, key_usage_evidence=None) -> list[Action]:
    """One Action per unknown_capped entry: settle only with evidence, else skip; over bound -> refuse."""
    actions = []
    for rid, rec in state["requests"].items():
        if rec["status"] != "unknown_capped":
            continue
        info = attribute(rid, rec, index)
        bound = _dec(rec["reserved_usd"])
        base = dict(request_id=rid, bound_usd=bound, model=info.model, batch=info.batch)
        # 1. OpenRouter generation record
        gid, where = _generation_id(info)
        if gid and info.backend == "openrouter" and lookup is not None:
            status, raw = lookup(gid)
            if status == 200:
                data = (json.loads(raw).get("data") or {}) if raw else {}
                if data.get("id") == gid and data.get("total_cost") is not None:
                    actual = _dec(data["total_cost"])
                    dec = "refuse_over_bound" if actual > bound else "settle"
                    actions.append(Action(**base, decision=dec, actual_usd=actual, kind="openrouter_generation",
                                          evidence_raw=raw, reason=f"generation id from {where}",
                                          evidence=f"openrouter /api/v1/generation id={gid} total_cost={actual}"))
                    continue
            elif status == 404:
                saved = (_load(info.receipt) or {}).get("response") if info.receipt else None
                if isinstance(saved, dict) and saved.get("error") and not saved.get("choices"):
                    actions.append(Action(**base, decision="settle", actual_usd=ZERO, kind="zero_completion",
                                          evidence_raw=b"", reason="404 x6 and the saved response is an error object",
                                          evidence=f"openrouter generation lookup 404 for id={gid} on every attempt; "
                                                   "saved response is an error object without choices "
                                                   "(Zero Completion Insurance)"))
                    continue
            actions.append(Action(**base, decision="skip", kind="openrouter_generation",
                                  reason=f"generation lookup status {status}: no billable record found"))
            continue
        # 2. relay quota refusal with an unchanged per-key usage counter
        if key_usage_evidence is not None and info.backend == "relay" and info.failure:
            proof = key_usage_evidence(rid, info)
            if proof:
                actions.append(Action(**base, decision="settle", actual_usd=ZERO, kind="relay_key_usage_unchanged",
                                      evidence=proof, reason="quota refusal; key usage unchanged around the attempt"))
                continue
        why = "no generation id and no receipt" if not gid else "no lookup available"
        if info.backend == "relay":
            why = "relay: no key-usage evidence (needs tags.key_ordinal and usage snapshots around the attempt)"
        actions.append(Action(**base, decision="skip", reason=why))
    return actions


def apply_reconcile(actions: list[Action], ledger_path: Path, evidence_dir: Path) -> dict:
    """Write evidence files, then settle through BudgetLedger.settle_unknown_later. Over-bound: refuse."""
    from .semantic_budget import BudgetLedger
    limit = json.loads(Path(ledger_path).read_text())["limit_usd"]
    led = BudgetLedger(Path(ledger_path), limit_usd=str(limit))
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    done, refused = [], []
    from .semantic_budget import BudgetExceeded
    for a in actions:
        if a.decision == "refuse_over_bound":
            # Evidence of a charge above the bound: book the actual charge (never keep the lower
            # bound) and halt, exactly as `settle` / `settle_unknown_later` do.
            path = evidence_dir / (hashlib.sha256(a.request_id.encode()).hexdigest() + ".generation.json")
            raw = a.evidence_raw or json.dumps({"kind": a.kind, "evidence": a.evidence}).encode()
            path.write_bytes(raw)
            try:
                led.settle_unknown_later(a.request_id, str(a.actual_usd),
                                         evidence=f"{a.evidence}; file={path}; "
                                                  f"sha256={hashlib.sha256(raw).hexdigest()}")
            except BudgetExceeded:
                pass                                   # halt is set on the ledger
            refused.append(a.request_id)
            continue
        if a.decision != "settle":
            continue
        assert a.actual_usd is not None and a.actual_usd <= a.bound_usd
        path = evidence_dir / (hashlib.sha256(a.request_id.encode()).hexdigest() + ".generation.json")
        raw = a.evidence_raw or json.dumps({"kind": a.kind, "evidence": a.evidence}).encode()
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        led.settle_unknown_later(a.request_id, str(a.actual_usd),
                                 evidence=f"{a.evidence}; file={path}; sha256={digest}")
        done.append(a.request_id)
    return {"settled": done, "refused_over_bound": refused}


def render_reconcile(actions: list[Action], applied: dict | None, dry: bool) -> str:
    n_settle = [a for a in actions if a.decision == "settle"]
    freed = sum((a.bound_usd - a.actual_usd for a in n_settle), ZERO)
    out = [f"# haenv spend reconcile ({'DRY RUN, nothing written' if dry else 'APPLIED'})", "",
           f"- unknown_capped entries examined: {len(actions)}",
           f"- would settle: {len(n_settle)} (paid {_usd(sum((a.actual_usd for a in n_settle), ZERO), 6)} "
           f"replacing bound {_usd(sum((a.bound_usd for a in n_settle), ZERO), 4)}; frees {_usd(freed, 4)})",
           f"- refused, evidence above the bound (needs an operator): "
           f"{sum(a.decision == 'refuse_over_bound' for a in actions)}",
           f"- kept at bound (no evidence): {sum(a.decision == 'skip' for a in actions)}", ""]
    by = defaultdict(lambda: [0, ZERO])
    for a in actions:
        k = (a.decision, a.kind or "-", a.reason.split(":")[0][:70])
        by[k][0] += 1
        by[k][1] += a.bound_usd
    out += _table("Outcome by evidence class", ["decision", "kind", "reason", "n", "bound $"],
                  [[*k, v[0], _usd(v[1], 4)] for k, v in sorted(by.items())])
    if n_settle:
        out += _table("Settlements", ["request id", "batch", "model", "bound $", "paid $", "evidence"],
                      [[a.request_id[:44], a.batch, a.model, _usd(a.bound_usd, 5), _usd(a.actual_usd, 6), a.evidence[:90]]
                       for a in n_settle])
    if applied:
        out += [f"Applied: settled {len(applied['settled'])}, refused over bound {len(applied['refused_over_bound'])}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI

def _resolve_batches(args) -> list[Path]:
    found = []
    for b in args.batch or []:
        p = Path(b)
        if not p.is_dir():
            matches = sorted(Path.cwd().glob(f"results/*/*/{b}"))
            if not matches:
                raise SystemExit(f"batch not found: {b}")
            p = matches[0]
        found.append(p)
    for root in args.scan_root or []:
        found += discover_batches(Path(root))
    seen, uniq = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "spend"
    if argv and argv[0] == "reconcile":
        mode, argv = "reconcile", argv[1:]
    ap = argparse.ArgumentParser(prog="haenv spend" + (" reconcile" if mode == "reconcile" else ""))
    ap.add_argument("--ledger", type=Path, required=True, help="shared budget ledger (read only, no lock)")
    ap.add_argument("--batch", action="append", metavar="DIR", help="batch directory (repeatable)")
    ap.add_argument("--scan-root", action="append", metavar="DIR", help="attribute every batch found under DIR")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, help="write the report here instead of stdout")
    if mode == "spend":
        ap.add_argument("--by", action="append", choices=DIMENSIONS, help="grouping (repeatable)")
        ap.add_argument("--balances", action="store_true", help="query provider balances (free GETs)")
        ap.add_argument("--window-min", type=float, default=30.0, help="throughput window for the forecast")
        ap.add_argument("--no-forecast", action="store_true")
    else:
        g = ap.add_mutually_exclusive_group()
        g.add_argument("--dry-run", action="store_true", help="default: print what would be settled")
        g.add_argument("--apply", action="store_true", help="settle through BudgetLedger.settle_unknown_later")
        ap.add_argument("--evidence-dir", type=Path, help="where raw evidence is stored on --apply")
        ap.add_argument("--no-network", action="store_true", help="skip OpenRouter generation lookups")
    args = ap.parse_args(argv)
    batches = _resolve_batches(args)

    if mode == "spend":
        by = tuple(args.by) if args.by else DEFAULT_BY
        bal = None
        if args.balances:
            from .cli import load_cfg
            relay = ((load_cfg().get("backends") or {}).get("relay") or {})
            bal = fetch_balances(relay_base=relay.get("quota_url"),
                                 relay_key_prefix=relay.get("key_env_pool"))
        rep = build_report(args.ledger, batches, by=by, balances=bal, window_min=args.window_min,
                           with_forecast=not args.no_forecast)
        text = json.dumps(_jsonable(rep), indent=2, ensure_ascii=False) if args.json else render_markdown(rep)
        rc = 0
    else:
        state = read_ledger(args.ledger)
        index, _aux, _scans = build_index(batches)
        lookup = None
        if not args.no_network and os.environ.get("OPENROUTER_API_KEY"):
            lookup = GenerationLookup(os.environ["OPENROUTER_API_KEY"],
                                      os.environ.get("HAENV_OPENROUTER_PROXY") or os.environ.get("https_proxy"))
        actions = plan_reconcile(state, index, lookup)
        applied = None
        if args.apply:
            if args.evidence_dir is None:
                raise SystemExit("--apply needs --evidence-dir")
            applied = apply_reconcile(actions, args.ledger, args.evidence_dir)
        if args.json:
            text = json.dumps(_jsonable([a.__dict__ | {"evidence_raw": None} for a in actions]), indent=2,
                              ensure_ascii=False)
        else:
            text = render_reconcile(actions, applied, dry=not args.apply)
        rc = 2 if any(a.decision == "refuse_over_bound" for a in actions) else 0
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
