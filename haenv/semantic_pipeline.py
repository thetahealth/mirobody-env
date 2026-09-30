"""Prepare, seal and resume semantic judging without overwriting saved evaluations."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import urllib.request

from .semantic_budget import BudgetExceeded, BudgetLedger
from .semantic_inputs import ANSWER_FIELDS, anonymous_prompt, gated_observations, select_responses
from .judge_evidence import evidence_catalogue
from .semantic_judge import JudgeTask, evaluate_consensus, task_record
from .semantic_rubric import REGISTRY, load_policy, make_rubric, summarize_verdicts
from .semantic_transport import PriceSchedule, PricedJudge
from .semantic_parallel import MAX_JUDGE_LANES

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as file:
        h = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def code_state() -> dict:
    """A pilot seals its complete judging segment plus not-yet-mounted semantic modules.

    `files` (raw bytes) and `world_sha` are provenance; `semantic` (comments and
    docstrings excluded, INFRA files left out) with `judging_sha16` is what
    `same_judging_code` compares.
    """
    from .anchor import infra_files, judging_fingerprint, semantic_bytes, world_fingerprint
    from tools.make_freeze import JUDGING
    files = sorted(set(JUDGING) | {
        "haenv/semantic_judge.py",
        "haenv/semantic_inputs.py", "haenv/semantic_rubric.py", "haenv/semantic_pipeline.py",
        "registry/semantic_judging.yaml", "haenv/semantic_lean.py", "registry/semantic_judging_v4.yaml",
        "haenv/semantic_atom_correction.py", "registry/semantic_judging_second.yaml",
        "registry/semantic_judging_why.yaml", "registry/semantic_judging_why_lean.yaml",
        "registry/semantic_judging_source.yaml",
    })
    infra = set(infra_files())
    return {"files": {name: digest(ROOT / name) for name in files},
            "semantic": {name: hashlib.sha256(semantic_bytes(ROOT / name)).hexdigest()
                         for name in files if name not in infra},
            "judging_sha16": judging_fingerprint(), "world_sha": world_fingerprint()}


def same_judging_code(recorded: dict, current: dict | None = None) -> bool:
    """Whether the code that decides a score is unchanged since `recorded`.

    A comment, an INFRA module or a generation-side change does not stop a run: the
    world stamp is not compared because a reconstructed slice is checked against its
    saved canonical payload (`_context_for_cell`). A state recorded before `semantic`
    existed compares raw bytes and the world stamp, as it did when it was sealed.
    """
    current = code_state() if current is None else current
    if "files" not in recorded:                  # a state of another shape: exact comparison
        return recorded == current
    if "semantic" not in recorded:
        return all(current[k] == recorded[k] for k in ("files", "judging_sha16", "world_sha"))
    return (current.get("semantic") == recorded["semantic"]
            and current["judging_sha16"] == recorded["judging_sha16"])


def _context_for_cell(row: dict, raw_cases: dict, payloads: dict,
                      batch_metadata: dict, current_kernel_sha: str,
                      verified_canonical: set[str]):
    """Reconstruct a saved slice only if the source's payload builder is proven identical.

    A later emission-gate change moves the broad world stamp without changing
    `core.build_instance`. The saved canonical payload is an independent roundtrip
    control; a changed kernel or changed canonical output refuses reconstruction.
    """
    from build import build_instance
    sp, vp = payloads[row["case"]]
    if row["geometry"] == "slices":
        endpoint = max(part["t"] for part in row["slice_rows"])
        if endpoint != vp.T:
            if type(endpoint) is not int or endpoint < 0 or endpoint > vp.T:
                raise ValueError("Saved slice endpoint is outside the canonical observation window")
            if not batch_metadata.get("world_sha") or row.get("world_sha") != batch_metadata["world_sha"]:
                raise ValueError("Saved row and source batch have different generation worlds")
            if not batch_metadata.get("kernel_sha256") or batch_metadata["kernel_sha256"] != current_kernel_sha:
                raise ValueError("Cannot reconstruct a saved slice with a different kernel")
            if row["case"] not in verified_canonical:
                check_sp, check_vp = build_instance(raw_cases[row["case"]], vp.T)
                if asdict(check_sp) != asdict(sp) or asdict(check_vp) != asdict(vp):
                    raise ValueError("Current builder does not reproduce the saved canonical payload")
                verified_canonical.add(row["case"])
            sp, vp = build_instance(raw_cases[row["case"]], endpoint)
    return sp, vp


def _reference(row: dict, sp, vp) -> dict:
    from .judges._helpers import _gold, _rivals_of, gold_kind
    from .judges.differential import _gold_test_optional, _gold_test_segs
    tests = [str(t) for t in (_gold(vp, "tests", None) or []) if _gold_test_segs(str(t))]
    probe = None
    if type(row.get("noop_truth_present")) is bool:
        probe = {"target": row["noop_target"], "window": row.get("noop_window"),
                 "truth_present": row["noop_truth_present"]}
    visible = asdict(sp)
    visible.pop("case_id", None)
    reference = {"kind": gold_kind(vp), "diagnosis": _gold(vp, "diagnosis", None),
            "aliases": _gold(vp, "aliases", None), "threads": _gold(vp, "threads", None),
            "join_gold": _gold(vp, "join_gold", None),
            "required_tests": [t for t in tests if not _gold_test_optional(t)],
            "optional_tests": [t for t in tests if _gold_test_optional(t)],
            "rivals": [{"name": r.get("name"), "discriminator": r.get("discriminator")}
                       for r in (_rivals_of(vp) or [])],
            "noop": probe, "visible_case": visible}
    from .semantic_visibility import correct_reference
    return correct_reference(reference)[0]


def prepare_batches(batches: list[Path], out: Path, *, run_id: str,
                    limit_per_model: int | None = None, policy_path: Path | None = None,
                    only_keys: set[str] | None = None,
                    include_baselines: tuple[str, ...] = ()) -> dict:
    """Read-only source preparation; sample by a fixed case hash before any judging.

    `policy_path` selects a candidate protocol (default: the approved registry);
    `only_keys` restricts preparation to a preregistered cell sample;
    `include_baselines` names constructed stubs to judge as upper-bound controls.
    """
    from .baselines import BASELINE_NAMES
    from .evaluate import _extract_json, load_rows
    from .payloads import load_payloads
    from .store import load_cases
    from .batch import kernel_fingerprint
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation output must be new; resume the existing manifest instead")
    if limit_per_model is not None and limit_per_model <= 0:
        raise ValueError("Pilot limit must be positive")
    from .semantic_rubric import LEAN_LAYOUT, rubric_atoms
    policy = load_policy(policy_path)
    lean = policy.get("prompt_layout") == LEAN_LAYOUT
    tasks, sources, states = [], [], Counter()
    for batch in batches:
        batch = batch.resolve()
        required = [batch / name for name in ("eval.jsonl", "responses.jsonl", "cases.jsonl", "payloads.jsonl", "batch.json")]
        if not all(path.is_file() for path in required):
            raise ValueError(f"Batch requires complete saved responses and payloads: {batch.name}")
        batch_metadata = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
        current_kernel_sha = kernel_fingerprint()[1]
        verified_canonical: set[str] = set()
        rows = [r for r in load_rows(batch / "eval.jsonl")
                if r["solver"] not in BASELINE_NAMES or r["solver"] in include_baselines]
        if not rows:
            raise ValueError("No real-model rows to judge")
        trace_by_cell = {}
        if any(row.get("geometry") == "gated" for row in rows):
            trace_path = batch / "trace.jsonl"
            if not trace_path.is_file():
                raise ValueError("Gated rejudging requires saved tool observations")
            required.append(trace_path)
            for event in _read_jsonl(trace_path):
                if event["type"] in {"case/start", "tool/call", "tool/result",
                                      "request/header", "assistant/message"}:
                    trace_by_cell.setdefault((event["case"], event["solver"]), []).append(event)
        sources.append({"batch": str(batch), "files": {p.name: digest(p) for p in required},
                        "source_world_sha": batch_metadata.get("world_sha"),
                        "source_kernel_sha256": batch_metadata.get("kernel_sha256"),
                        "slice_reprojection": "matching_kernel_plus_saved_canonical_roundtrip"})
        by_identity = {(r["case"], r["solver"]): r for r in rows}
        selected = select_responses(rows, _read_jsonl(batch / "responses.jsonl"))
        selected.sort(key=lambda c: (c["solver"], hashlib.sha256(
            ("semantic-pilot-v1|" + c["case"]).encode()).hexdigest()))
        if lean:
            # One case's answers stay adjacent so their shared prompt prefix is reused.
            selected.sort(key=lambda c: (c["case"], c["solver"]))
        taken = Counter()
        raw_cases = load_cases(batch / "cases.jsonl")
        payloads = load_payloads(batch / "payloads.jsonl")
        if not {r["case"] for r in rows} <= raw_cases.keys() & payloads.keys():
            raise ValueError("A saved case or payload is missing; do not regenerate it")
        for cell in selected:
            if limit_per_model is not None and taken[cell["solver"]] >= limit_per_model:
                continue
            taken[cell["solver"]] += 1
            identity = {"batch": str(batch), "case": cell["case"], "solver": cell["solver"],
                        "geometry": cell["geometry"], "endpoint": cell["endpoint"],
                        "raw_sha256": cell["raw_sha256"]}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if only_keys is not None and key not in only_keys:
                continue
            record = {"key": key, "source": identity, "status": cell["status"]}
            if cell["status"] != "ready":
                states[cell["status"]] += 1; tasks.append(record); continue
            try:
                answer = _extract_json(cell["response"]["raw"])
                if not isinstance(answer, dict) or not answer:
                    raise ValueError("No structured answer")
            except (ValueError, TypeError):
                record["status"] = "unparseable_response"
                states[record["status"]] += 1; tasks.append(record); continue
            row = by_identity[(cell["case"], cell["solver"])]
            sp, vp = _context_for_cell(row, raw_cases, payloads, batch_metadata,
                                       current_kernel_sha, verified_canonical)
            if row["geometry"] == "gated":
                sp, bought = gated_observations(sp, trace_by_cell.get(
                    (cell["case"], cell["solver"]), []), cell["response"]["raw"])
                # These are verified executed tool choices, not model self-report.
                answer = {**answer, "executed_investigations": bought}
            if policy.get("reason_slot") == "tool_calls_why":
                from .semantic_inputs import investigation_reasons
                answer = {**answer, "investigation_reasons": investigation_reasons(answer)}
            if not any(field in ANSWER_FIELDS for field in answer):
                # An answer with nothing the judge may read is unjudgeable: same status as an
                # unextractable one. The whole-batch prepare and the follow stream both land here.
                record["status"] = "unparseable_response"
                states[record["status"]] += 1; tasks.append(record); continue
            ref = _reference(row, sp, vp)
            use_ids = policy["judge"]["evidence_mode"] == "pointer_ids"
            if lean:
                from .semantic_lean import lean_prompt
                from .semantic_visibility import visible_probe
                atoms, rubric = rubric_atoms(ref, answer, policy)
                prompt, answer_text, template = lean_prompt(
                    atoms, ref, answer, policy, proposal_cap=rubric["proposal_cap"],
                    n_proposed=rubric["n_proposed"])
                item_ids = tuple(item for item, _, _ in atoms)
                # The delivered-window check is recorded here because the lean prompt no
                # longer carries the visible case it was computed from.
                record["reference_check"] = visible_probe(ref)
            else:
                criteria, rubric = make_rubric(ref, answer, policy)
                prompt, answer_text = anonymous_prompt(criteria, ref, answer, evidence_ids=use_ids)
                item_ids, template = tuple(criteria), None
            task = JudgeTask(run_id=run_id, item_ids=item_ids, prompt=prompt,
                             answer_text=answer_text, judge_model=policy["judge"]["model_id"],
                             rubric_version=policy["version"],
                             reasoning_effort=policy["judge"]["reasoning_effort"],
                             evidence_catalog=evidence_catalogue(answer_text)[0] if use_ids else None,
                             atom_template=template)
            record.update(task=task_record(task), rubric=rubric, task_sha256=task.fingerprint)
            states["ready"] += 1; tasks.append(record)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "policy": policy, "sources": sources, "code": code_state(),
                "tasks_sha256": digest(out / "tasks.jsonl"), "states": dict(states),
                "sampling": {"limit_per_model": limit_per_model, "rule": "case SHA256, model blind",
                             **({"only_keys_sha256": hashlib.sha256(json.dumps(sorted(only_keys)).encode()).hexdigest(),
                                 "n_only_keys": len(only_keys)} if only_keys is not None else {})},
                "paid_calls": False, "sealed": False}
    _write_json(out / "manifest.json", manifest)
    return manifest


def seal_run(out: Path) -> dict:
    """Lock the tested pilot code without restamping any existing batch or freeze anchor."""
    manifest = json.loads((out / "manifest.json").read_text())
    if "correction" in manifest:
        from .semantic_corrections import validate_correction
        validate_correction(out)
    if not same_judging_code(manifest["code"]) or manifest["tasks_sha256"] != digest(out / "tasks.jsonl"):
        raise ValueError("Prepared inputs or judging code changed; create a new run")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, text=True)
    if dirty:
        raise ValueError("Commit tested changes before sealing a paid pilot")
    manifest["sealed"] = True
    manifest["sealed_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    _write_json(out / "manifest.json", manifest)
    return manifest


def validate_run(out: Path) -> dict:
    manifest = json.loads((out / "manifest.json").read_text())
    if "correction" in manifest:
        from .semantic_corrections import validate_correction
        validate_correction(out)
    if manifest.get("sealed") is not True:
        raise ValueError("Paid judging requires a sealed, tested version")
    if not same_judging_code(manifest["code"]) or manifest["tasks_sha256"] != digest(out / "tasks.jsonl"):
        raise ValueError("Sealed code or tasks drifted; do not mix judging versions")
    for source in manifest["sources"]:
        for name, expected in source["files"].items():
            if digest(Path(source["batch"]) / name) != expected:
                raise ValueError("Original answers or gold changed after preparation")
    return manifest


def fetch_prices(cfg: dict, policy: dict) -> dict:
    from .evaluate import BACKENDS, _ensure_backends_registered, load_env_file
    _ensure_backends_registered(cfg)
    judge = policy["judge"]
    spec = cfg["models"][judge["model_key"]]
    if spec["model"] != judge["model_id"] or spec["backend"] != judge["backend"]:
        raise ValueError("Judge route does not match the declared semantic policy")
    backend = BACKENDS[spec["backend"]]
    env = load_env_file(cfg.get("env_file"))
    key = env.get(backend["key_env"])
    if not key:
        raise ValueError("Judge backend credentials are not configured")
    url = backend["url"].rsplit("/chat/completions", 1)[0] + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.loads(response.read())["data"]
    matches = [m for m in models if m.get("id") == judge["model_id"]]
    if len(matches) != 1:
        raise ValueError("The exact judge model has no unique provider price record")
    # Reserve at the maximum over every endpoint OpenRouter may route to.
    from .solver_accounting import fetch_endpoints
    return {**matches[0], "endpoints": fetch_endpoints(cfg, judge["model_id"])}


def execute_run(out: Path, cfg: dict, ledger_path: Path, *, limit_usd: str,
                max_cells: int | None = None, concurrency: int | None = None) -> dict:
    """Resume per-sample journals; all cost belongs to the shared run-wide ledger."""
    import fcntl
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _execute_run(out, cfg, ledger_path, limit_usd=limit_usd,
                                max_cells=max_cells, concurrency=concurrency)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _execute_run(out: Path, cfg: dict, ledger_path: Path, *, limit_usd: str,
                 max_cells: int | None = None, concurrency: int | None = None) -> dict:
    from .evaluate import load_env_file, solver_for_spec
    from .semantic_parallel import run_cells
    import copy
    if max_cells is not None and max_cells <= 0:
        raise ValueError("max_cells must be positive")
    manifest = validate_run(out)
    policy = manifest["policy"]
    from .semantic_rubric import default_lanes
    concurrency = default_lanes(policy) if concurrency is None else concurrency
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_JUDGE_LANES:
        raise ValueError("Concurrency must be between 1 and %d" % MAX_JUDGE_LANES)
    metadata = fetch_prices(cfg, policy)
    if not {"response_format", "structured_outputs"} <= set(metadata.get("supported_parameters", [])):
        raise ValueError("Judge metadata does not advertise the required structured output support")
    _write_json(out / "price-metadata.json", metadata)
    prices = PriceSchedule.from_metadata(metadata, policy["judge"]["model_id"])
    ledger = BudgetLedger(ledger_path, limit_usd=limit_usd)
    spec = dict(cfg["models"][policy["judge"]["model_key"]])
    spec.update(max_tokens=policy["judge"]["max_tokens"], reasoning_effort="high", retries=0)
    solver = solver_for_spec(policy["judge"]["model_key"], spec,
                             load_env_file(cfg.get("env_file")), timeout=900, retries=0)
    if solver is None:
        raise ValueError("Configured judge could not be constructed")
    solver.pool = None
    def call_factory():
        # Per-cell state (response schema, SSE buffers) must not be shared by threads.
        return PricedJudge(copy.copy(solver), prices, ledger, receipts=out / "receipts")
    return run_cells(_read_jsonl(out / "tasks.jsonl"), out, manifest["run_id"], ledger,
                     call_factory, concurrency=concurrency, max_cells=max_cells)


def default_after_run(batch: Path, cfg: dict, *, offline: bool,
                      budget_ledger: Path | None, budget_usd: str | None) -> dict:
    """Online runs require semantic judging; offline runs explicitly report no judgment."""
    if offline:
        return {"status": "not_run_offline", "semantic_scores_available": False}
    if budget_ledger is None or budget_usd is None:
        raise ValueError("Online semantic judging requires an explicit budget ledger and limit")
    policy = load_policy()
    out = batch / "semantic" / policy["version"]
    if not (out / "manifest.json").exists():
        prepare_batches([batch], out, run_id=f"{batch.parent.name}-{batch.name}-{policy['version']}")
        seal_run(out)
    report = execute_run(out, cfg, budget_ledger, limit_usd=budget_usd)
    return {"status": "completed" if not (report["budget"]["halt_reason"] or report.get("stop_reason"))
            and report["states"].get("resolved", 0) > 0
            and not any(n for state, n in report["states"].items() if state != "resolved")
            else "incomplete",
            "output": str(out), "summary": report}
