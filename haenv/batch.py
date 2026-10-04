"""batch.py -- resolving and registering evaluation batches.

Artifacts land under a timestamped batch directory:

    results/<task_type>/<job_id>/<YYYYmmdd-HHMMSS>/{eval.jsonl, verify.jsonl, batch.json}
    reports/<job_id>/<YYYYmmdd-HHMMSS>/{eval-<job_id>.md, verify-<job_id>.md}

By default a run reuses the job's most recent batch (so a rerun only fills in
missing cells); `--fresh` starts a new batch and `--batch <stamp>` selects one.
`batch.json` records provenance, usage, sampling conditions and the
declared-vs-ran solver reconciliation.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data

log = logging.getLogger("haenv.batch")

#: Batch name: timestamp, optionally with `-pk<n>` or `-rep<n>` (the nth repeat of a
#: reliability sample).
STAMP_RE = re.compile(r"^\d{8}-\d{6}(?:-(?:pk|rep)\d{1,2})?$")

#: The repeat suffix alone, for telling a sample directory from its source batch.
SAMPLE_SUFFIX_RE = re.compile(r"^-(?:pk|rep)\d{1,2}$")


def sample_dirs(batch: Path) -> list[Path]:
    """The repeat-sample directories `<batch>-pk<n>` / `<batch>-rep<n>` next to `batch`."""
    batch = Path(batch)
    if not batch.parent.is_dir():
        return []
    return sorted(p for p in batch.parent.iterdir()
                  if p.is_dir() and p.name.startswith(batch.name)
                  and SAMPLE_SUFFIX_RE.match(p.name[len(batch.name):]))


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _existing(root: Path, task_type: str, job_id: str) -> list[str]:
    d = root / f"results/{task_type}/{job_id}"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and STAMP_RE.match(p.name))


def resolve(root: Path, task_type: str, job_id: str, fresh: bool = False,
            batch: str | None = None) -> tuple[str, bool]:
    """Determine which batch stamp to use for this run. Returns (stamp, is_new).

    Precedence: explicit `--batch` > `--fresh` (new stamp) > reuse the most
    recent batch > create new if there is no history.
    """
    if batch:
        if not STAMP_RE.match(batch):
            raise ValueError(f"--batch must be in YYYYmmdd-HHMMSS form, got {batch!r}")
        return batch, batch not in _existing(root, task_type, job_id)
    if fresh:
        return now_stamp(), True
    _prior = latest(root, task_type, job_id)
    if _prior:
        log.info("[batch] reusing the most recent batch %s (resuming; use --fresh to start a new batch)", _prior)
        return _prior, False
    return now_stamp(), True


def fingerprint_dir(kp: Path) -> str:
    """Fingerprint of every *.py file in the directory, sorted by filename with the
    filename folded in (renames and swapped contents both change it).
    """
    h = hashlib.sha256()
    for f in sorted(Path(kp).glob("*.py"), key=lambda p: p.name):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def kernel_fingerprint() -> tuple[str, str]:
    """Returns (kernel absolute path, fingerprint) of the kernel package actually imported.

    Read from the loaded `haenv_kernel` module rather than from `kernel_path()`: since the
    kernel became a package, the executed copy is the one on the import path, and a
    `HAENV_KERNEL_PATH` checkout is only a comparison source (see `haenv.kernel_path`).
    """
    import haenv_kernel.schema
    kp = Path(haenv_kernel.schema.__file__).resolve().parent
    return str(kp), fingerprint_dir(kp)


def provenance_fields(cfg: dict, job_path: str | None = None,
                      root: Path | None = None) -> dict:
    """Per-row provenance: `haenv_git_sha` (with `+dirty`), `job_sha256`,
    `world_knobs`, `generator_sha`, `world_sha`, `kernel_sha256`.
    """
    import subprocess
    root = Path(root) if root else _dr()
    out: dict = {}
    try:                                        # haenv's own code version
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        out["haenv_git_sha"] = (r.stdout or "").strip() or "unknown"
        d = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10)
        if (d.stdout or "").strip():
            out["haenv_git_sha"] += "+dirty"    # uncommitted changes -> the conclusion can't be reproduced exactly, must stay visible
    except Exception:
        out["haenv_git_sha"] = "unknown"
    if job_path:                                # content digest of job.yaml (should change if the item changed)
        try:
            out["job_sha256"] = hashlib.sha256(
                Path(job_path).read_bytes()).hexdigest()[:16]
        except OSError:
            out["job_sha256"] = "unknown"
    # Runtime knobs such as `--no-physio` change the world without changing any file,
    # so they are recorded as `world_knobs` next to `world_sha`.
    try:
        from . import world_knobs as _ev_knob
        out["world_knobs"] = f"physio={int(bool(_ev_knob.PHYSIO_ENABLED[0]))}"
    except Exception:                                    # noqa: BLE001
        out["world_knobs"] = "unknown"

    # `generator_sha` covers only three generation files and is kept for reading
    # existing batches; `world_sha` is the world version.
    try:
        gen = b"".join(sorted(
            (root / "haenv" / f).read_bytes() for f in ("build.py", "events.py", "overlay.py")))
        out["generator_sha"] = hashlib.sha256(gen).hexdigest()[:16]
    except OSError:
        out["generator_sha"] = "unknown"
    try:
        from .anchor import world_fp_cached
        out["world_sha"] = world_fp_cached()
    except Exception:                                  # noqa: BLE001
        out["world_sha"] = "unknown"
    _abs, _fp = kernel_fingerprint()
    out["kernel_sha256"] = _fp
    return out


def usage_fields(job, rows: list[dict] | None = None) -> dict:
    """Usage record: per-cell evaluation usage rolled up by
    `evaluate.usage_coverage`, plus generation-side usage. Missing rows are
    recorded as `absent:no_eval_rows`, never as 0.
    """
    from .row_store import load_rows
    from .run_ledger import usage_coverage
    out: dict = {}
    if rows is None:
        p = getattr(job, "results_file", None)
        try:
            rows = load_rows(Path(p)) if p and Path(p).is_file() else None
        except (OSError, json.JSONDecodeError) as e:
            rows, out["eval_usage_error"] = None, f"{type(e).__name__}: {e}"[:200]
    out["eval_usage"] = (usage_coverage(rows) if rows is not None
                         else {"status": "absent:no_eval_rows"})
    try:                                        # item-generation side (did this process go through LLM generation)
        from .llm import gen_usage_snapshot
        out["gen_usage"] = gen_usage_snapshot()
    except Exception as e:                                      # noqa: BLE001
        out["gen_usage"] = {"status": "error", "why": f"{type(e).__name__}: {e}"[:200]}
    return out


#: Explanation written into `batch.json` next to the `solvers` column.
SOLVER_RECONCILE_NOTE = (
    "Reconciliation between this batch's declared model set (`models`) and the solver set "
    "that actually produced cells. Three states, kept separate: `declared_and_ran` / "
    "`declared_not_run` / `ran_not_declared`. "
    "`declared_not_run` is not a defect by itself: it equals the full set right after "
    "a batch is created, and the same is true when a resumable run stops partway; no "
    "conclusion is drawn in this column. "
    "`ran_not_declared` is the dangerous state: a real model produced cells in this batch "
    "without being declared for it. "
    "A batch without this column has not recorded it (\"unmeasured\"); that does not mean "
    "declared matches ran. "
    "Unreadable eval rows are not \"nothing ran\": that is \"not evaluated yet\", recorded as "
    "`absent:`, never folded to 0.")


def reconcile_solvers(declared, rows: list[dict] | None) -> dict:
    """The declared model set vs. the solvers that produced cells.

    States: `declared_and_ran`, `declared_not_run` (normal right after a batch is
    created or when a resumed run stops partway) and `ran_not_declared` (a real
    model produced cells without being declared). Offline stubs are listed
    separately in `stubs_ran`. Real models are identified by
    `report.real_solver_pool`, stubs by `baselines.BASELINE_NAMES`.
    """
    from .baselines import BASELINE_NAMES
    from .report import real_solver_pool

    decl = [str(m) for m in (declared or [])]
    out: dict = {
        "declared": sorted(set(decl)),
        "n_declared": len(set(decl)),
        "note": SOLVER_RECONCILE_NOTE,
        "source": ("haenv/batch.py::reconcile_solvers + haenv/report.py::real_solver_pool"
                   " + haenv/baselines.py::BASELINE_NAMES"),
    }
    if rows is None:
        out.update({"status": "absent:no_eval_file", "n_eval_rows": None,
                    "why": "the batch directory has no `eval.jsonl` -- this batch has not been evaluated yet, that is not the same as \"nothing ran\""})
        return out
    ran_all = {str(r.get("solver")) for r in rows if r.get("solver") is not None}
    pool, unregistered = real_solver_pool(rows)
    ran_real = {str(x) for x in pool} & ran_all
    out.update({
        "n_eval_rows": len(rows),
        "ran": sorted(ran_all),
        "ran_real": sorted(ran_real),
        "stubs_ran": sorted(ran_all - ran_real),
        "unregistered_in_config": sorted(str(x) for x in (unregistered or [])),
        "declared_and_ran": sorted(set(decl) & ran_all),
        "declared_not_run": sorted(set(decl) - ran_all),
        "ran_not_declared": sorted(ran_real - set(decl)),
    })
    if not rows:
        out["status"] = "absent:no_eval_rows"
        out["why"] = "`eval.jsonl` exists but has 0 rows -- this batch has not been evaluated yet, that is not the same as \"nothing ran\""
        return out
    if out["ran_not_declared"]:
        out["status"] = "mismatch:ran_not_declared"
        out["ran_not_declared_kind"] = ("no_declaration_at_all" if not decl
                                        else "extra_beyond_declaration")
    elif not decl:
        out["status"] = "absent:no_models_declared"     # offline/stub batch: no models to check, not "nothing ran"
    elif out["declared_not_run"] and not out["declared_and_ran"]:
        out["status"] = "declared_none_ran"
    elif out["declared_not_run"]:
        out["status"] = "partial:some_declared_not_run"
    else:
        out["status"] = "consistent"
    return out


def solver_fields(job, declared, rows: list[dict] | None = None) -> dict:
    if rows is None:
        p = getattr(job, "results_file", None)
        try:
            from .row_store import load_rows
            rows = load_rows(Path(p)) if p and Path(p).is_file() else None
        except (OSError, json.JSONDecodeError):
            rows = None
    return reconcile_solvers(declared, rows)


#: Written into `batch.json` when no sampling parameter is sent.
SAMPLING_UNSPECIFIED_NOTE = (
    "Unspecified -- the payload haenv sends carries no sampling parameters at all, so the "
    "effective value is whatever the provider defaults to and is unknown. Do not "
    "read this as temperature=0: that would be \"absence folded into 0\". Leaving temperature "
    "unpinned is deliberate (temperature 0 degenerates into loops on long generations, "
    "reasoning models often ignore temperature, and no real deployment runs temperature 0), "
    "so this column records a condition, not a constraint.")

SAMPLING_DECLARED_NOTE = (
    "`params` is what each model is sent, from `config.models.<m>.sampling` and "
    "`reasoning_effort`: a numeric temperature is in the request body; `unsupported:<why>` "
    "means the provider takes no temperature and none is sent (checked against the route's "
    "`supported_parameters` before any request). A model absent from `params` runs at the "
    "provider default.")

#: Explanation written into `batch.json` next to `solving_sha16`.
SOLVING_FINGERPRINT_NOTE = (
    "A fingerprint of this batch's solving conditions (per-model backend/model/max_tokens/"
    "stream/sampling keys). It covers the declared call conditions, not the code that sends "
    "the request -- the latter is covered by `world_ref.haenv_git_sha` and the freeze "
    "anchor. `config.yaml` is not in any segment of the freeze anchor, so changing "
    "`max_tokens` changes the measurement conditions of every cell without moving "
    "`judging_sha16`; this fingerprint makes that visible. A batch without this column has "
    "not recorded it (\"unmeasured\"); that does not mean the conditions were the same.")


def solving_fingerprint(per_model: dict) -> str | None:
    """16-character fingerprint of the declared per-model call conditions
    (backend, model, max_tokens, stream, sampling keys, status, argv); `None` when
    no model is declared. Covers the declaration, not the request code.
    """
    if not per_model:
        return None
    canon = {str(n): {k: v for k, v in sorted((m or {}).items())
                      if k in ("backend", "model", "max_tokens", "stream",
                               "sampling_in_config", "status", "argv", "provider",
                               "sampling", "reasoning_effort", "upstream")}
             for n, m in sorted(per_model.items())}
    blob = json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def sampling_fields(cfg: dict, models=None) -> dict:
    """Sampling conditions: `params` (what haenv puts in the payload, currently
    `{}`), `declared_in_config` (sampling keys in `config.models` that never reach
    the payload), `per_model` (models declared in this batch) and `observed`
    (what providers echo back). A backend with no request in this process is
    `absent:not_observed`, not "does not echo".
    """
    from .solvers import BACKENDS
    from .llm import (SAMPLING_PARAM_NAMES, budget_audit_snapshot as _budget_audit_snapshot,
                      effective_transport as _effective_transport, sampling_snapshot)

    specs = (cfg or {}).get("models") or {}
    # `declared_in_config` scans the whole config; `per_model` covers only the
    # models declared in this batch.
    declared = {}
    for n, spec in sorted(specs.items()):
        if isinstance(spec, dict):
            d = {k: spec[k] for k in SAMPLING_PARAM_NAMES if k in spec}
            if d:
                declared[str(n)] = d
    names = [str(m) for m in (models or [])]
    per_model = {}
    for n in names:
        spec = specs.get(n)
        if spec is None:
            per_model[n] = {"status": "absent:not_in_config_models"}
            continue
        if not isinstance(spec, dict):
            per_model[n] = {"status": "absent:cli_dispatch_params_not_visible",
                            "argv": [str(x) for x in spec]}
            continue
        d = {k: spec[k] for k in SAMPLING_PARAM_NAMES if k in spec}
        # Records the effective `stream` value (`llm.effective_transport`), which
        # feeds `solving_sha16`.
        _eff = _effective_transport(n, spec.get("backend", "openrouter"),
                                    int(spec.get("max_tokens") or 0),
                                    bool(spec.get("stream", False)))
        per_model[n] = {"backend": spec.get("backend", "openrouter"),
                        "model": spec.get("model"),
                        "max_tokens": spec.get("max_tokens"),
                        "stream": bool(_eff["stream"]),
                        "stream_from": _eff["stream_from"],
                        "sampling_in_config": d,
                        **({"provider": spec["provider"]} if spec.get("provider") else {}),
                        # Declared sampling block and effort (`config.models.<m>.sampling`),
                        # only when set, so undeclared models keep their fingerprint.
                        **({"sampling": spec["sampling"]} if spec.get("sampling") else {}),
                        **({"reasoning_effort": spec["reasoning_effort"]}
                           if spec.get("reasoning_effort") else {}),
                        **({"upstream": spec["upstream"]} if spec.get("upstream") else {})}
    obs = sampling_snapshot()
    by_model = obs.get("by_model") or {}
    backends = {}
    for b in sorted(BACKENDS):
        seen = [v for v in by_model.values() if v.get("backend") == b]
        backends[b] = ({"echo_status": "absent:not_observed",
                        "why": "this process has not sent a request through this backend -- "
                               "whether it echoes sampling parameters back is untested"}
                       if not seen else
                       {"echo_status": sorted({str(v.get("echo_status")) for v in seen}),
                        "n_calls": sum(int(v.get("n_calls") or 0) for v in seen),
                        "echo_distinct": [e for v in seen
                                          for e in (v.get("echo_distinct") or [])]})
    _real = {n: m for n, m in per_model.items() if "backend" in m}
    _pinned = {n: {"sampling": m.get("sampling"), "reasoning_effort": m.get("reasoning_effort")}
               for n, m in _real.items() if m.get("sampling")}
    status = ("unspecified:provider_default" if not _pinned
              else "declared:config.models.sampling" if len(_pinned) == len(_real)
              else f"partial:{len(_pinned)}/{len(_real)}_models_declared")
    return {
        "status": status,
        "params": _pinned,
        "note": SAMPLING_UNSPECIFIED_NOTE if not _pinned else SAMPLING_DECLARED_NOTE,
        "solving_sha16": solving_fingerprint(per_model),
        "solving_status": ("absent:no_models_declared_in_batch" if not per_model
                           else f"{len(per_model)} model(s) in the fingerprint"),
        "solving_note": SOLVING_FINGERPRINT_NOTE,
        "declared_in_config": declared,
        "declared_not_forwarded": sorted(declared),
        "n_config_models_scanned": len(specs),     # scan scope: 0 scanned must not look the same as "all clean"
        "per_model": per_model,
        "per_model_status": ("absent:no_models_declared_in_batch" if not names
                             else f"{len(names)} model(s)"),
        "backends": backends,
        "observed": obs,
        "echo_capture": {
            "wired": ["haenv/llm.py::_MappingDispatcher (generation, direct backend)",
                      "haenv/llm.py::Dispatcher (generation, CLI dispatch, always absent:no_response_object)",
                      "haenv/evaluate.py::OpenAICompatSolver._post (evaluation, OpenAI-compatible)",
                      "haenv/evaluate.py::GoogleSolver._post (evaluation, direct Google)"],
            "not_wired": [],
            "why_not": "",
            "payload_capture": ["haenv/evaluate.py::OpenAICompatSolver._post",
                                "haenv/evaluate.py::GoogleSolver._post"],
        },
        "budget_audit": _budget_audit_snapshot(),
        "source": "haenv/batch.py::sampling_fields + haenv/llm.py::sampling_snapshot"
                  " + haenv/llm.py::budget_audit_snapshot",
    }


def _put_sampling(meta: dict, samp: dict, meta_was_empty: bool) -> None:
    """Write sampling conditions into `meta`. Sticky: a process that sent no
    requests does not overwrite recorded observations. `applies_to` is
    `this_batch` only for a freshly created batch.
    """
    prev = meta.get("sampling")
    fresh_obs = not str((samp.get("observed") or {}).get("status", "")
                        ).startswith("absent:no_real_calls")
    # A model routed to a different backend/model id than the recorded one (per-run config
    # overlay) must not keep the old route in the batch record.
    _old, _new = (prev or {}).get("per_model") or {}, samp.get("per_model") or {}
    route_changed = any(isinstance(_old.get(n), dict) and isinstance(_new.get(n), dict)
                        and (_old[n].get("backend"), _old[n].get("model"))
                        != (_new[n].get("backend"), _new[n].get("model")) for n in _new)
    if prev and not fresh_obs and not route_changed:
        return                                   # sticky: no requests sent, must not erase the prior observation
    samp = dict(samp)
    samp["applies_to"] = ("this_batch" if meta_was_empty
                          else "current_code_declaration_only")
    if prev and (prev.get("observed") or {}).get("by_model"):
        samp["observed_prior"] = prev["observed"]   # don't lose the conditions observed in the prior round
    meta["sampling"] = samp


def _refresh_sampling_observed(meta: dict) -> None:
    """Refresh only `sampling.observed` at the end of a run, and only if this
    process sent requests.
    """
    from .llm import sampling_snapshot
    obs = sampling_snapshot()
    if str(obs.get("status", "")).startswith("absent:no_real_calls"):
        return
    samp = dict(meta.get("sampling") or {})
    if (samp.get("observed") or {}).get("by_model"):
        samp["observed_prior"] = samp["observed"]
    samp["observed"] = obs
    meta["sampling"] = samp


def record_usage(job, rows: list[dict] | None = None) -> Path | None:
    """Back-fill usage, sampling observations and the solver reconciliation into an
    existing `batch.json` after evaluation. `gen_usage` is sticky: it is only
    overwritten by a process that made generation calls.
    """
    p = job.results_dir / "batch.json"
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    _u = usage_fields(job, rows)
    meta["eval_usage"] = _u["eval_usage"]
    if "eval_usage_error" in _u:
        meta["eval_usage_error"] = _u["eval_usage_error"]
    _g = _u.get("gen_usage") or {}
    if str(_g.get("status", "")).startswith("absent:no_llm_generation"):
        meta.setdefault("gen_usage", _g)        # sticky: must not erase the item-generation cost recorded earlier
    else:
        meta["gen_usage"] = _g
    try:
        _refresh_sampling_observed(meta)
    except Exception as e:                                      # noqa: BLE001
        meta["sampling_error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        meta["solvers"] = solver_fields(job, meta.get("models"), rows)
    except Exception as e:                                      # noqa: BLE001
        meta["solvers_error"] = f"{type(e).__name__}: {e}"[:200]
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def register(job, cfg: dict, n_cases: int, n_emitted: int, cmd: str,
             case_digests: dict | None = None, generator: str | None = None,
             gate_report: dict | None = None) -> Path:
    _kp_abs, _kp_fp = kernel_fingerprint()
    job.results_dir.mkdir(parents=True, exist_ok=True)
    p = job.results_dir / "batch.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    # Must be read before any `setdefault` fills `meta`.
    _meta_was_empty = not meta
    meta.setdefault("batch", job.batch)
    meta.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))
    if gate_report is not None:
        meta["emission_gates"] = gate_report
    # `emission_gates_status` distinguishes not run / empty / malformed / recorded.
    meta["emission_gates_status"] = (
        "recorded" if isinstance(gate_report, dict) and gate_report
        else "empty" if isinstance(gate_report, dict)
        else "malformed" if gate_report is not None
        else "not_run")
    # Provenance that describes the generated items (`generator_sha`, `world_sha`,
    # `world_knobs`, `job_sha256`) is sticky, so batches derived by `restamp`
    # keep the version that generated their `cases.jsonl`; code-version fields are
    # refreshed on every call.
    _prov_all = provenance_fields(cfg, job_path=str(getattr(job, "path", "") or ""),
                                  root=job.root)
    _STICKY = ("generator_sha", "world_sha", "world_knobs", "job_sha256")
    _sticky = {k: _prov_all.pop(k) for k in _STICKY if k in _prov_all}
    # `models` is the union over the batch's calls: set when the batch is created,
    # extended only by `run`. `n_cases`/`n_emitted` refresh only when this call
    # wrote `cases.jsonl`. The last command's own view is kept in `last_cmd_*`.
    _DECLARING = ("run",)
    _decl_now = [str(m) for m in (job.models or cfg.get("default_models", []) or [])]
    _may_declare = _meta_was_empty or cmd in _DECLARING
    _decl = list(dict.fromkeys([str(m) for m in (meta.get("models") or [])]
                               + (_decl_now if _may_declare else [])))
    _regen = bool(case_digests)
    _n_cases = n_cases if (_regen or meta.get("n_cases") is None) else meta["n_cases"]
    _n_emitted = n_emitted if (_regen or meta.get("n_emitted") is None) else meta["n_emitted"]
    meta.update({
        "job_id": job.job_id, "task_type": job.task_type,
        "last_cmd": cmd,
        "last_run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_cases": _n_cases, "n_emitted": _n_emitted,
        "models": _decl,
        "last_cmd_models": _decl_now,
        "last_cmd_n_cases": n_cases, "last_cmd_n_emitted": n_emitted,
        "last_cmd_regenerated_cases": _regen,
        "include_baseline": job.include_baseline, "multiround": job.multiround,
        "kernel_path": cfg.get("kernel_path"), "timeout_s": cfg.get("timeout_s"),
        "kernel_resolved": _kp_abs, "kernel_sha256": _kp_fp,
        **_prov_all,                               # describes the conclusion, refreshed every time
    })
    for _k, _v in _sticky.items():
        meta.setdefault(_k, _v)
    # haenv version at item-generation time: sticky, unlike `haenv_git_sha`
    # (the version of this run).
    meta.setdefault("generation_git_sha", meta.get("haenv_git_sha", "unknown"))
    _prov = dict(getattr(job, "provenance", None) or {})
    meta["gold_provenance"] = _prov
    meta["gold_provenance_recorded"] = bool(_prov)
    from .pack_size import batch_fields
    _pack = batch_fields(getattr(job, "path", None), job.job_id)
    if _pack:
        meta["pack"] = _pack
    # Per-part judging fingerprint as of this run (see `anchor.parts_compatible`).
    try:
        from .anchor import judge_parts
        meta["judge_parts"] = judge_parts()
    except Exception as e:                                     # noqa: BLE001
        meta["judge_parts_error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        from .gold_kinds import rules_manifest
        meta["kind_rules"] = rules_manifest()
    except Exception as e:                                     # noqa: BLE001
        meta["kind_rules_error"] = f"{type(e).__name__}: {e}"[:200]
    _u = usage_fields(job)
    meta["eval_usage"] = _u["eval_usage"]
    if "eval_usage_error" in _u:
        meta["eval_usage_error"] = _u["eval_usage_error"]
    _gu = _u.get("gen_usage") or {}
    if str(_gu.get("status", "")).startswith("absent:no_llm_generation"):
        meta.setdefault("gen_usage", _gu)      # sticky: describes how the items were generated
    else:
        meta["gen_usage"] = _gu
    try:
        _put_sampling(meta, sampling_fields(cfg, models=_decl),
                      meta_was_empty=_meta_was_empty)
    except Exception as e:                                     # noqa: BLE001
        meta["sampling_error"] = f"{type(e).__name__}: {e}"[:200]
    # Written here too, because `build` / `verify` / `report` never reach
    # `record_usage`.
    try:
        meta["solvers"] = solver_fields(job, meta.get("models"))
    except Exception as e:                                     # noqa: BLE001
        meta["solvers_error"] = f"{type(e).__name__}: {e}"[:200]
    if generator:
        meta["generator"] = generator          # who generated the items: llm:<model> / deterministic
    if case_digests:                           # per-case sha256: a content fingerprint for the batch, so later changes can be detected
        meta["cases_sha256"] = case_digests
        meta["n_cases_saved"] = len(case_digests)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def latest(root: Path, task_type: str, job_id: str) -> str | None:
    prior = _existing(root, task_type, job_id)
    return prior[-1] if prior else None


#: What a rebuild must share with an earlier batch before any per-case difference
#: counts as nondeterminism rather than a legitimate change.
REPRODUCIBLE_KEYS = ("world_sha", "world_knobs", "job_sha256", "generator")


def same_generation_inputs(prev_meta: dict, job, cfg: dict, generator: str) -> tuple[bool, str]:
    """Whether rebuilding `job` now must reproduce the batch `prev_meta` describes
    byte for byte: same generation code, world knobs, job file and generator.

    Returns ``(same, why_not)``. An unknown value on either side counts as
    different, so the caller stays in report-only mode when it cannot tell.
    """
    cur = provenance_fields(cfg, job_path=str(getattr(job, "path", "") or ""))
    cur["generator"] = generator
    for k in REPRODUCIBLE_KEYS:
        a, b = prev_meta.get(k), cur.get(k)
        if not a or a == "unknown" or a != b:
            return False, f"{k}: {a} -> {b}"
    return True, ""


def previous_digests(job) -> tuple[str, dict] | None:
    """Per-case sha256 of the previous batch (for the idempotency-gate
    comparison). Takes the most recent one before the current batch that has
    a recorded fingerprint."""
    base = job.results_dir.parent
    if not base.is_dir():
        return None
    for name in sorted((d.name for d in base.iterdir() if d.is_dir()), reverse=True):
        if name >= job.batch:                 # skip the current batch and anything newer
            continue
        p = base / name / "batch.json"
        if not p.is_file():
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("cases_sha256"):
            return name, meta["cases_sha256"]
    return None
