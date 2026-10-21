"""Re-ask archived diagnosis responses with controlled evidence injection.

``prepare`` is entirely offline. ``run`` uses the normal model runtime,
request accountant and scheduler, writing results to a separate directory.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from haenv.interaction.reask import ARMS, Batch, build_unit, digest, final_split, read_rows, tool_items


@dataclass
class Unit:
    """A verified plan plus the original scorer payloads; no model client is created."""

    plan: dict
    solver_payload: object
    verifier_payload: object
    source: Path | None = None


def score(text, solver_payload, verifier_payload):
    """D from the shipped diagnosis scorer, with the experiment's parse-failure policy."""
    from haenv.judges import gold_kind, judge_dx_comorbidity, judge_dx_unified
    from haenv.runtime.solvers import _to_output
    from haenv_kernel.solver import _extract_json
    try:
        data = _extract_json(text) if (text or "").strip() else {}
    except Exception:  # malformed model text is a scored outcome
        data = {}
    if not isinstance(data, dict) or not data:
        return {"D": 0, "parse": False, "dx_listed": None}
    kind_name = gold_kind(verifier_payload)
    judge = {"ddx:unified": judge_dx_unified, "ddx:comorbidity": judge_dx_comorbidity}.get(kind_name)
    if judge is None:
        return {"D": None, "parse": True, "dx_kind": kind_name.removeprefix("ddx:")}
    row = judge(_to_output(data, solver_payload), verifier_payload, {})
    kind, listed = row.get("dx_kind"), row.get("dx_listed")
    return {"D": int(listed is not None and float(listed) == 1) if kind in ("unified", "comorbidity") else None,
            "parse": True, "dx_kind": kind, "dx_listed": listed}


def batch_paths(source):
    """A batch directory, response file, or run tree containing batches."""
    path = Path(source)
    if path.is_file() and path.name in ("responses.jsonl", "query_trace.jsonl"):
        path = path.parent
    if (path / "query_trace.jsonl").is_file():
        return [path]
    found = sorted(p.parent for p in path.rglob("query_trace.jsonl")) if path.is_dir() else []
    if not found:
        raise ValueError(f"no stored query conversations found in {source}")
    return found


def repeat_of(batch, trace):
    if trace.get("repeat") is not None:
        return int(trace["repeat"])
    match = re.search(r"-(?:pk|rep)(\d+)$", batch.path.name)
    return int(match[1]) if match else 1


def prepare(source, *, sources=(), questions=None, repeats=None, models=None,
            arms=tuple(ARMS), seed=20261019, half_seed=20261020, limit=None):
    """Select failed diagnosis units and rebuild requests without any model calls.

    Source candidates are traversed in sorted batch-path order, then stored trace
    order. Supply the same result corpus and seeds to reproduce an archived choice.
    ``models`` selects original response models; it never substitutes another model.
    Half is omitted for units with fewer than two missed results.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError("arms must be unique names from resample,decisive,placebo,half")
    paths = batch_paths(source)
    all_paths = sorted(set(paths) | {p for root in sources for p in batch_paths(root)})
    batches = {p: Batch(p) for p in all_paths}
    worlds = {}
    for path in paths:
        for question, model in batches[path].traces:
            if ((questions is None or question in questions) and (models is None or model in models)):
                worlds.setdefault(question, batches[path].world_identity(question))
    items = defaultdict(list)
    returns = {}
    for batch in batches.values():
        for (question, model), trace in batch.traces.items():
            if question not in worlds:
                continue
            target, source_world = worlds[question], batch.world_identity(question)
            common = target["fingerprints"].keys() & source_world["fingerprints"].keys()
            if (target["payload_sha256"] != source_world["payload_sha256"] or target["T"] != source_world["T"]
                    or any(target["fingerprints"][k] != source_world["fingerprints"][k] for k in common)):
                raise ValueError(f"{question}: source world mismatch in {batch.path}")
            for item in tool_items(trace):
                call = question, item["name"], json.dumps(item["arguments"], sort_keys=True, ensure_ascii=False)
                content = json.dumps(item["content"], sort_keys=True, ensure_ascii=False)
                if call in returns and returns[call] != content:
                    raise ValueError(f"{question}: conflicting tool returns for {item['name']} {call[-1]} in {batch.path}")
                returns[call] = content
                items[question].append({**item, "src": f"{batch.path}|{model}"})
    units, seen = [], set()
    for path in paths:
        batch = batches[path]
        for (question, model), trace in batch.traces.items():
            repeat = repeat_of(batch, trace)
            if ((questions is not None and question not in questions)
                    or (models is not None and model not in models)
                    or (repeats is not None and repeat not in repeats)):
                continue
            if trace.get("stop") not in ("answered", "forced_final"):
                continue
            _, answer, _ = final_split(trace)
            sp, vp = batch.payloads[question]
            baseline = score(answer.get("content") or "", sp, vp)
            if baseline["D"] != 0:
                continue
            from haenv.m2.query import ADAPTER
            if not ADAPTER.key_lookups(vp, None):
                continue
            key = model, question, repeat
            if key in seen:
                raise ValueError(f"ambiguous stored unit {key}: select one response batch")
            seen.add(key)
            plan = build_unit(batch, question, model, repeat, items[question], arms=arms,
                              seed=seed, half_seed=half_seed)
            plan["baseline_D"] = baseline["D"]
            plan["payload_sha256"] = digest(json.dumps([asdict(sp), asdict(vp)], ensure_ascii=False, sort_keys=True))
            plan["source_world"] = batch.world_identity(question)
            plan["source_answer_settings"] = batch.original_answer_settings(question, model, plan["prefix_sha"])
            units.append(Unit(plan, sp, vp, path))
            if limit is not None and len(units) >= limit:
                return units
    return units


def dry_rows(units):
    """Per-unit/arm operator view; keys here are never part of model messages."""
    return [{"model": u.plan["model"], "question": u.plan["question"], "repeat": u.plan["repeat"],
             "kind": u.plan["kind"], "arm": arm, "arm_code": ARMS[arm],
             "prefix_sha": u.plan["prefix_sha"],
             "injected": [{"test": c["arguments"].get("target", c["name"]),
                           "name": c["name"], "arguments": c["arguments"], "keys": c["keys"]}
                          for c in req["items"]]}
            for u in units for arm, req in u.plan["requests"].items()]


def tool_choice_for(model, cfg):
    """The historical GLM route requires auto; tools returned here are never executed."""
    spec = cfg.get("models", {}).get(model, {})
    return "auto" if model == "glm-5.3-flash" or spec.get("model") == "z-ai/glm-5.3-flash" else "none"


def run(units, output, *, cfg=None, workers=64, model_caps=None,
        ledger_path=None, budget_usd=None, tool_choice=None, allow_settings_override=False):
    """Send one final request per planned arm, scored by the stored question's gold.

    Requires a shared budget ledger and its approved cap. Completed rows are
    resumed; a changed plan refuses reuse of an output directory. Transport errors
    remain D=null and are eligible for a later retry. Scoring errors retain the raw
    answer with D=null and are complete outcomes, never fresh paid attempts.
    """
    from haenv.app.config import load_cfg
    from haenv.interaction.query.solver import as_query_solver
    from haenv.interaction.query.tools import ToolSpec
    from haenv.runtime import scheduler
    from haenv.runtime.solver_accounting import answer_settings, prepare_accounting
    from haenv.runtime.solvers import build_solvers
    if workers < 1:
        raise ValueError("workers must be positive")
    if ledger_path is None or budget_usd is None:
        raise ValueError("model calls require a budget ledger and budget_usd")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("tool_choice must be auto or none")
    if not units:
        raise ValueError("no eligible failed diagnosis units selected")
    cfg = cfg if cfg is not None else load_cfg()
    output = Path(output)
    if any(u.source is not None and output.resolve() == u.source.resolve() for u in units):
        raise ValueError("output must be separate from the stored response directory")
    names = sorted({u.plan["model"] for u in units})
    factories = dict(build_solvers(SimpleNamespace(models=names, include_baseline=False), cfg))
    live = {m: factories[m]() for m in names}
    actual = {m: answer_settings(solver) for m, solver in live.items()}
    replay = {}
    for unit in units:
        p = unit.plan
        if "source_answer_settings" not in p:
            p["source_answer_settings"] = (Batch(unit.source).original_answer_settings(
                p["question"], p["model"], p["prefix_sha"]) if unit.source is not None else None)
        source_settings = p["source_answer_settings"]
        state = "unavailable" if source_settings is None else (
            "verified" if source_settings == actual[p["model"]] else "mismatch")
        if state != "verified" and not allow_settings_override:
            reason = "source answer settings unavailable" if state == "unavailable" else "answer settings mismatch"
            raise ValueError(f"{p['question']}/{p['model']}: {reason}; use --allow-settings-override "
                             "to proceed without claiming exact settings replay")
        replay[id(unit)] = state
    manifest = {"version": 2, "units": [u.plan for u in units], "answer_settings": actual,
                "models": {m: {"model": s.model, "backend": s.backend} for m, s in live.items()},
                "allow_settings_override": bool(allow_settings_override),
                "tool_choices": {m: tool_choice or tool_choice_for(m, cfg) for m in names}}
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "reask-manifest.json"
    if manifest_path.is_file() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise ValueError("re-ask plan changed; use a new output directory")
    manifest_path.write_text(manifest_text, encoding="utf-8")
    done = {(r["source_model"], r["question"], r["repeat"], r["arm"])
            for r in read_rows(output / "injection-rows.jsonl")
            if r["status"] != "error" or r.get("error_stage") == "scoring"}
    tasks = []
    for unit in units:
        for arm in unit.plan["requests"]:
            p = unit.plan
            if (p["model"], p["question"], p["repeat"], ARMS[arm]) not in done:
                cell = f"{p['question']}|{p['repeat']}|{ARMS[arm]}"
                tasks.append((cell, unit, arm))
    if not tasks:
        return []
    accounting = prepare_accounting(list(factories.items()), cfg, output,
                                    ledger_path=Path(ledger_path), limit_usd=str(budget_usd),
                                    identity={"reask_manifest_sha256": digest(manifest_text)})
    accounting.enable_content_keys([t[0] for t in tasks])
    lock = threading.Lock()

    def one(task):
        cell, unit, arm = task
        plan, req = unit.plan, unit.plan["requests"][arm]
        model = plan["model"]
        inner = factories[model]()
        accounting.bind(inner, cell, model)
        solver = as_query_solver(inner, cfg["query"])
        specs = [ToolSpec(t["name"], t["description"], t["parameters"], lambda: {}) for t in req["tools"]]
        started = time.monotonic()
        turn = solver.chat(req["system"], req["messages"], specs, manifest["tool_choices"][model])
        text = turn.content or ""
        failed = getattr(solver, "_failed_turns", None) or []
        if turn.tool_calls:
            status, scored = "unanswered", {"D": 0}
        elif text.strip():
            try:
                status, scored = "answered", score(text, unit.solver_payload, unit.verifier_payload)
            except Exception as error:  # a scorer defect is local to this saved answer
                status, scored = "error", {"D": None, "error_stage": "scoring",
                                           "error": f"{type(error).__name__}: {error}"}
        elif failed and all(str(f.get("error", "")).startswith((
                "HTTPError", "URLError", "RuntimeError", "TimeoutError", "ConnectionError", "RemoteDisconnected"))
                            for f in failed):
            status, scored = "error", {"D": None}
        else:
            status, scored = "empty", {"D": 0}
        scalar = {"model": live[model].model, "source_model": model, "answer_settings": actual[model],
                  "settings_replay": replay[id(unit)], "question": plan["question"], "repeat": plan["repeat"],
                  "kind": plan["kind"], "arm": ARMS[arm], "status": status,
                  "D": scored["D"], "n_added": len(req["items"])}
        if scored.get("error_stage"):
            scalar.update({k: scored[k] for k in ("error_stage", "error")})
        # Runtime cell ids distinguish repeats and arms for accounting and recovery.
        return {"case": cell, "solver": model, "overall": "ERROR" if status == "error" else status,
                "scalar": scalar, "answer": text, "finish": turn.finish, "usage": turn.usage,
                "failed_attempts": failed, "score": scored, "tool_calls": turn.tool_calls,
                "latency_s": round(time.monotonic() - started, 3)}

    def emit(row):
        with lock:
            for name, data in (("answers.jsonl", row), ("injection-rows.jsonl", row["scalar"])):
                with (output / name).open("a", encoding="utf-8") as file:
                    file.write(json.dumps(data, ensure_ascii=False) + "\n")
        return row

    # Raise historical small caps to the requested concurrency; explicit model caps
    # take precedence and backend/paid ceilings still bound both levels.
    with scheduler.ledger_context(ledger_path), scheduler.model_cap_context(model_caps, workers):
        rows = scheduler.run(one, tasks, emit, workers=workers, ledger_path=Path(ledger_path),
                             batch_dir=output, key_of=lambda t: t[1].plan["model"],
                             backend_of=lambda m: cfg["models"][m].get("backend", "openrouter"),
                             pooling="model", per_model={}, wall_budget={},
                             backend_limits=scheduler._load_backend_limits(cfg))
    return [r["scalar"] for r in rows]
