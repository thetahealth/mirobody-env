"""Build `registry/solver_timing.json` from the traces of finished batches.

The scheduler reads it for two things (`haenv/runtime/scheduler.py`): the expected duration
of a cell (`median_cell_call_s`: the median over cells of the summed call seconds of one
cell, per model and geometry; older tables without it: median call seconds x median calls
per cell) to run the longest cells first, and the longest successful call per model, which
sets that model's stall deadline. Neither changes a score. A product of two medians
misplaces a model whose cells mostly hold one long call, so the per-cell sum is used.

A call is one `assistant/message` in `trace.jsonl`. Its duration is the service time of
the successful attempt: the recorded `latency_s` minus `slot_wait_s`, the seconds the
request waited for a paid lane (a queue on our side, not the provider's time). When a
solver records no latency, the time from `step/start` to `assistant/message` stands in,
counted only for steps without a failed attempt (so the span is one successful attempt),
minus the same wait. A trace written before `slot_wait_s` was recorded gives latency
including any lane wait; `calls_without_slot_wait` counts those calls per row. Baseline
stubs are skipped, and so are durations under `MIN_CALL_S` (a trace rebuilt after the run
carries no live timestamps: every span reads 0).

    .venv/bin/python tools/calibration/build_solver_timing.py <batch dir> [<batch dir> ...]
    .venv/bin/python tools/calibration/build_solver_timing.py --check <batch dir> ...   # exit 1 if stale

The output is generated: do not edit it by hand.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = _repository_root(__file__)
sys.path.insert(0, str(ROOT))
OUT = ROOT / "registry" / "solver_timing.json"
#: No model call returns this fast; a shorter duration means the timestamp is not live.
MIN_CALL_S = 0.5


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _batch_name(d: Path) -> str:
    parts = d.resolve().parts
    if "results" in parts:
        k = len(parts) - 1 - parts[::-1].index("results")
        return "/".join(parts[k + 1:])
    return d.name


def collect(batch_dirs: list[Path]) -> dict:
    from haenv.world.generation.baselines import BASELINE_NAMES
    calls: dict[tuple, list[float]] = {}
    success: dict[tuple, list[float]] = {}
    per_cell: dict[tuple, dict[tuple, int]] = {}
    cell_call_s: dict[tuple, dict[tuple, float]] = {}
    cell_span: dict[tuple, dict[tuple, list]] = {}
    no_wait: dict[tuple, int] = {}
    sources = []
    for d in batch_dirs:
        tp = Path(d) / "trace.jsonl"
        raw = tp.read_bytes()
        sources.append({"batch": _batch_name(Path(d)), "path": str(d),
                        "trace_sha16": hashlib.sha256(raw).hexdigest()[:16]})
        step_start: dict[tuple, float] = {}
        failed_steps: set = set()
        name = _batch_name(Path(d))
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s = str(r.get("solver"))
            if s in BASELINE_NAMES:
                continue
            g = str(r.get("geometry"))
            cell = (name, r.get("case"), s, r.get("turn"))
            key = (s, g)
            t = r.get("type")
            if t == "case/start":
                cell_span.setdefault(key, {}).setdefault(cell, [None, None])[0] = _ts(r["ts"])
            elif t == "case/end":
                cell_span.setdefault(key, {}).setdefault(cell, [None, None])[1] = _ts(r["ts"])
            elif t == "step/start":
                step_start[(cell, r.get("step"))] = _ts(r["ts"])
            elif t == "assistant/attempt":
                failed_steps.add((cell, r.get("step")))
            elif t == "assistant/message":
                data = r.get("data") or {}
                per_cell.setdefault(key, {})
                per_cell[key][cell] = per_cell[key].get(cell, 0) + 1
                lat = data.get("latency_s")
                span = None
                if (cell, r.get("step")) in step_start:
                    span = _ts(r["ts"]) - step_start[(cell, r.get("step"))]
                    if span < MIN_CALL_S:
                        span = None
                if lat is not None and float(lat) < MIN_CALL_S:
                    lat = None
                if lat is None and (cell, r.get("step")) not in failed_steps and span is not None:
                    lat = span
                wait = data.get("slot_wait_s")
                if wait is None:
                    no_wait[key] = no_wait.get(key, 0) + 1
                elif lat is not None:
                    lat = max(float(lat) - float(wait), 0.0)
                if span is not None:
                    calls.setdefault(key, []).append(span)
                if lat is not None:
                    success.setdefault(key, []).append(float(lat))
                took = lat if lat is not None else span
                if took is not None:
                    cc = cell_call_s.setdefault(key, {})
                    cc[cell] = cc.get(cell, 0.0) + float(took)
    models: dict[str, dict] = {}
    for key in sorted(set(calls) | set(success)):
        s, g = key
        ok = sorted(success.get(key, []))
        if not ok:
            continue
        n_per = sorted(per_cell.get(key, {}).values())
        spans = sorted(b - a for a, b in cell_span.get(key, {}).values()
                       if a is not None and b is not None and b - a >= MIN_CALL_S)
        models.setdefault(s, {})[g] = {
            "n_calls": len(ok), "n_cells": len(n_per),
            "median_call_s": round(statistics.median(ok), 2),
            "p90_call_s": round(ok[min(len(ok) - 1, int(0.9 * len(ok)))], 2),
            "max_success_call_s": round(ok[-1], 2),
            "median_calls_per_cell": statistics.median(n_per) if n_per else None,
            "median_cell_call_s": (round(statistics.median(cell_call_s[key].values()), 1)
                                   if cell_call_s.get(key) else None),
            "median_cell_s": round(statistics.median(spans), 1) if spans else None,
            "calls_without_slot_wait": no_wait.get(key, 0)}
    return {"_generated": "Generated by tools/calibration/build_solver_timing.py; do not edit by hand. "
                          "Rebuild: .venv/bin/python tools/calibration/build_solver_timing.py "
                          + " ".join(x["path"] for x in sources),
            "sources": sources, "models": models}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    got = json.dumps(collect(a.batches), ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if a.check:
        same = a.out.is_file() and a.out.read_text(encoding="utf-8") == got
        print("up to date" if same else f"stale: {a.out}")
        return 0 if same else 1
    a.out.write_text(got, encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
