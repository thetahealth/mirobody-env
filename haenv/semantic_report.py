"""Read-only semantic score views; raw solver results and old proxy scores are preserved."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .quantities import row_names

#: Semantic dimensions a composite needs. The composite's diagnosis dimension is the
#: code-side dx_listed, so a missing semantic dx_hit does not void it; dx_hit is an
#: auxiliary reading.
PRIMARY = ("noop_ok", "tests_recall", "tests_precision")
AUX_DIMS = ("dx_hit",)
REPLACED = PRIMARY + AUX_DIMS + ("disc_recall",)


def held_out_dims() -> dict[str, str]:
    """Scored dimensions whose LLM atom type failed the cross-protocol stability gate."""
    from .semantic_rubric import load_policy
    gate = load_policy().get("scoring_gate") or {}
    return {spec["dim"]: (f"{atom} failed the cross-protocol stability gate "
                          f"(cross {spec['cross']} > within-pair {spec['within']}, n={spec['n_atoms']})")
            for atom, spec in (gate.get("atom_types") or {}).items()
            if spec.get("dim") and not spec.get("passed")}
AUXILIARY = ("action_consistency", "join_reason_specific", "test_reasoning", "exclusion_reason")


def overlay_rows(rows: list[dict], results: dict, *, run_id: str, judging_sha: str) -> list[dict]:
    """Copy rows into a new measurement version, including explicit absent LLM values.

    The compound stamp records both the unchanged code measurements' source and
    the semantic judgment source; it never claims all fields were recomputed.
    """
    from .anchor import judging_fingerprint
    view_sha = judging_fingerprint()
    held = held_out_dims()
    viewed = []
    for source in rows:
        row = deepcopy(source)
        cell = results.get((row["case"], row["solver"]))
        status = cell["status"] if cell is not None else "pending"
        summary = cell.get("summary", {}) if cell is not None else {}
        metrics = summary.get("metrics", {})
        na = summary.get("not_applicable", {})
        coverage = summary.get("coverage", {})
        states = {}
        for name in set(REPLACED) | set(metrics):
            value = metrics.get(name)
            if value is not None and (type(value) not in (int, float, bool) or not 0 <= value <= 1):
                raise ValueError("Semantic values must be bounded or explicitly unresolved")
            if name in na:
                states[name] = "not_applicable"
                if value is not None:
                    raise ValueError("A not-applicable semantic metric cannot carry a score")
            elif value is not None:
                states[name] = "resolved"
            elif name in coverage:
                states[name] = "unresolved"
            else:
                states[name] = status if status != "resolved" else "missing_metric"
        withheld = {}
        for name in held:
            if states.get(name) not in (None, "not_applicable"):
                withheld[name] = metrics.get(name)
                states[name] = "held_out"
        proxies = {}
        for name in REPLACED:
            for key in row_names(name):
                if key in row:
                    proxies[key] = row.pop(key)
            row[name] = None if name in held else metrics.get(name)
        old_sha = row.get("judging_sha16")
        cell_sha = cell.get("judging_sha16", judging_sha) if cell else judging_sha
        cell_run = cell.get("run_id", run_id) if cell else run_id
        compound = hashlib.sha256(json.dumps(
            ["semantic-overlay-v1", old_sha, cell_sha, view_sha], separators=(",", ":")).encode()).hexdigest()[:16]
        row["code_proxy"] = proxies
        row["judging_sha16"] = compound
        row["semantic"] = {"run_id": cell_run, "status": status, "metrics": dict(metrics),
                           "metric_states": states, "coverage": coverage,
                           "not_applicable": na, "source_code_judging_sha16": old_sha,
                           "semantic_judging_sha16": cell_sha,
                           "view_judging_sha16": view_sha,
                           "clinical_review": "not_clinician_reviewed", "final": False}
        if withheld:
            row["semantic"]["held_out"] = {"metrics": withheld,
                                           "why": {name: held[name] for name in withheld}}
        if cell and cell.get("correction"):
            row["semantic"]["correction"] = deepcopy(cell["correction"])
        if cell and cell.get("atom_corrections"):
            row["semantic"]["atom_corrections"] = deepcopy(cell["atom_corrections"])
        if cell and cell.get("proposal_cap"):
            row["semantic"]["proposal_cap"] = deepcopy(cell["proposal_cap"])
        if cell and cell.get("reference_issue"):
            row["semantic"]["reference_issue"] = deepcopy(cell["reference_issue"])
        viewed.append(row)
    return viewed


def protect_composite(rec: dict, rows: list[dict]) -> None:
    """A missing measurement is neither an incorrect answer nor a removable dimension."""
    semantic = [r["semantic"] for r in rows if isinstance(r.get("semantic"), dict)]
    if not semantic:
        return
    missing = sum(any(s.get("metric_states", {}).get(name) not in
                      ("resolved", "not_applicable") for name in PRIMARY) for s in semantic)
    missing += len(rows) - len(semantic)
    held = {name: why for s in semantic for name, why in ((s.get("held_out") or {}).get("why") or {}).items()}
    if held:
        rec["held_out_dims"] = held          # reported separately; no slot in the composite
    rec["semantic_coverage"] = {"cells": len(rows), "primary_missing_cells": missing,
                                "states": dict(Counter(s["status"] for s in semantic)),
                                "clinical_review": "not_clinician_reviewed", "final": False}
    if missing:
        rec["score"] = None
        rec["score_full"] = None
        rec["score_none_reason"] = (f"LLM primary judgments missing or unresolved on {missing} cells; "
                                    "no proxy substitution, zero fill or reduced denominator")


PROPOSAL_FIELDS = ("tests_to_order", "executed_investigations")


def apply_proposal_cap(task, consensus: dict) -> tuple[dict, list[str]]:
    """Required-test yes votes must cite a proposal inside the counted proposal cap.

    The judge sees every proposal but only `counted_proposals` (the registered cap,
    the code proxy's rule) may earn recall. A yes vote whose cited proposal
    pointers all lie beyond the cap is read as no; votes citing no proposal field
    are left as the judge gave them. The majority is then re-taken exactly as in
    `evaluate_consensus`. Raw votes on disk are not changed.
    """
    from collections import Counter as _Counter
    from .judge_evidence import evidence_catalogue
    from .semantic_visibility import parse_reference
    items = [k for k in task.item_ids if k.startswith("required_test_")]
    if not items or task.evidence_catalog is None or task.atom_template is not None:
        return consensus["verdicts"], []
    _, criteria = parse_reference(task.prompt)
    answer = json.loads(task.answer_text)
    offsets, start = {}, 0
    for field in PROPOSAL_FIELDS:
        value = answer.get(field) or []
        offsets[field] = start
        start += len(value) if isinstance(value, list) else 0
    _, pointers = evidence_catalogue(task.answer_text)

    def index_of(evidence_id):
        pointer = pointers.get(evidence_id) or ""
        parts = pointer.split("/")
        if len(parts) >= 3 and parts[1] in offsets and parts[2].isdigit():
            return offsets[parts[1]] + int(parts[2])
        return None

    verdicts, changed = dict(consensus["verdicts"]), []
    for item in items:
        text, sep, context = criteria[item].partition("\nSpecific reference: ")
        cap = len(json.loads(context)["counted_proposals"]) if sep else None
        if cap is None:
            continue
        labels = []
        for sample in consensus["samples"]:
            if sample.get("status") != "ok" or item not in (sample.get("verdicts") or {}):
                continue
            vote = sample["verdicts"][item]
            label = vote["label"]
            cited = [i for i in (index_of(e) for e in vote.get("evidence_ids") or []) if i is not None]
            if label == "yes" and cited and min(cited) >= cap:
                label = "no"
            labels.append(label)
        counts = _Counter(labels)
        matches = [label for label in ("yes", "no") if counts[label] >= 2]
        new = matches[0] if matches else None
        if new != verdicts[item]:
            verdicts[item] = new
            changed.append(item)
    return verdicts, changed


#: Files that identify a judged batch by content: the answers, cases and payloads the
#: judge read, and the tool trace of gated answers. A source is found by these digests,
#: never by its recorded path, which is only a record (batches are copied and moved).
ANSWER_FILES = ("responses.jsonl", "cases.jsonl", "payloads.jsonl", "trace.jsonl")
#: `batch.json` fields a judgment depends on. Every other field is rewritten by later
#: commands (`report` registers the batch again: last command, run time, code versions).
BATCH_FIELDS = {"world_sha": "source_world_sha"}
_DIGESTS: dict = {}


def _file_digest(path: Path) -> str | None:
    from .semantic_pipeline import digest
    if not path.is_file():
        return None
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = digest(path)
    return _DIGESTS[key]


def _real_lines(path: Path) -> list[str]:
    """Lines of real models: offline stubs (BASELINE_NAMES) are dropped and re-run by a restamp."""
    from .baselines import BASELINE_NAMES
    stubs, kept = set(BASELINE_NAMES), []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            solver = json.loads(line).get("solver")
        except (json.JSONDecodeError, AttributeError):
            solver = None
        if solver not in stubs:
            kept.append(line)
    return kept


def _content_matches(source: dict, batch: Path) -> list[str]:
    """Answer files of `batch` whose digest differs from the source's record (empty = same content)."""
    return [name for name in ANSWER_FILES
            if name in source["files"] and _file_digest(batch / name) != source["files"][name]]


def _lineage_parent(batch: Path) -> Path | None:
    """The batch this one was restamped from: `derived_from_path` if present, else the sibling
    directory named `derived_from` in the same job directory."""
    meta_path = batch / "batch.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hint = meta.get("derived_from_path")
    if hint and Path(hint).is_dir():
        return Path(hint).resolve()
    if meta.get("derived_from"):
        return (batch.parent / str(meta["derived_from"])).resolve()
    return None


def source_batch(manifest: dict, batch: Path) -> tuple[dict, Path]:
    """The run's source whose recorded answer files equal `batch`'s, or an ancestor's.

    `batch` itself is tried first, then its restamp lineage (`_lineage_parent`). Exactly one
    source may match a batch; for an ancestor, the real-model lines of every answer file of
    `batch` must equal the ancestor's. Returns (source record, directory holding its content).
    """
    batch = Path(batch).resolve()
    recorded = [s for s in manifest["sources"] if any(n in s["files"] for n in ANSWER_FILES)]
    unrecorded = [s for s in manifest["sources"] if s not in recorded]
    # A source that recorded no answer file cannot be recognised by content; only its path names it.
    by_path = next((s for s in unrecorded if Path(s["batch"]).resolve() == batch), None)
    if by_path is not None:
        return by_path, batch
    chain, current = [], batch
    while current is not None and current.is_dir():
        if current in chain:
            raise ValueError(f"Semantic run source lineage loops at {current.name}; not a judged batch")
        chain.append(current)
        hits = [s for s in recorded if not _content_matches(s, current)]
        if len(hits) > 1:
            raise ValueError(f"Semantic run has {len(hits)} sources with the content of {current.name}: "
                             "ambiguous, refused")
        if hits:
            if current != batch:
                for name in ANSWER_FILES:
                    if name in hits[0]["files"] and (not (batch / name).is_file()
                                                     or _real_lines(batch / name) != _real_lines(current / name)):
                        raise ValueError(f"Restamped batch {batch.name}: real-model lines of {name} "
                                         f"differ from the judged batch {current.name}")
            return hits[0], current
        current = _lineage_parent(current)
    named = next((s for s in manifest["sources"] if Path(s["batch"]).resolve() == batch), None)
    if named is not None:
        raise ValueError("Semantic report source changed since judging: "
                         + ", ".join(_content_matches(named, batch)))
    if len(chain) == 1 and current is None:
        raise ValueError("Semantic run does not contain this source batch")
    raise ValueError(f"Semantic run does not contain this source batch: no source matches the content "
                     f"of {' <- '.join(p.name for p in chain)}"
                     + (f"; restamp lineage ends at {current}" if current is not None else ""))


def _verify_cells(source: dict, batch: Path, records: list[dict]) -> None:
    """The batch read still holds each judged cell: its eval row, the same answer, the same items."""
    for field, key in BATCH_FIELDS.items():
        meta = json.loads((batch / "batch.json").read_text(encoding="utf-8")) if (batch / "batch.json").is_file() else {}
        if key in source and meta.get(field) != source[key]:
            raise ValueError(f"Semantic report source changed since judging: batch.json {field} "
                             f"{meta.get(field)} != {source[key]}")
    if "eval.jsonl" in source["files"]:
        present = set()
        for line in (batch / "eval.jsonl").read_text(encoding="utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                present |= {(row.get("case"), row.get("solver"), row.get("geometry")),
                            (row.get("case"), row.get("solver"), None)}
        missing = sorted({(r["source"]["case"], r["source"]["solver"], r["source"].get("geometry"))
                          for r in records} - present)
        if missing:
            raise ValueError(f"Semantic report source changed since judging: eval.jsonl lacks "
                             f"{len(missing)} judged cells, e.g. {missing[0]}")
    judged = [r for r in records if r["source"].get("raw_sha256")]
    if judged and (batch / "responses.jsonl").is_file():
        import hashlib
        last = {}
        for line in (batch / "responses.jsonl").read_text(encoding="utf-8").split("\n"):
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(response, dict):
                last[(response.get("case"), response.get("solver"), response.get("slice_t"))] = response
        for r in judged:
            src = r["source"]
            raw = (last.get((src["case"], src["solver"], src.get("endpoint"))) or {}).get("raw")
            if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != src["raw_sha256"]:
                raise ValueError(f"Semantic report source changed since judging: the answer of "
                                 f"{src['case']} x {src['solver']} differs from the judged one")


def read_run(out: Path, batch: Path) -> tuple[dict, dict]:
    """Verify sources and independently recompute persisted votes without any dispatch.

    The source is found by content (`source_batch`): `batch` may be a copy of the judged
    batch elsewhere, or a batch restamped from it. Results are keyed by (case, solver)
    and overlay `batch`'s rows.
    """
    from .semantic_pipeline import digest
    from .semantic_judge import JudgeTask, evaluate_consensus
    from .semantic_rubric import summarize_verdicts
    manifest = json.loads((out / "manifest.json").read_text())
    if not manifest.get("sealed") or digest(out / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("Semantic report requires sealed, unmodified tasks")
    source, _ = source_batch(manifest, batch)
    records = [json.loads(line) for line in (out / "tasks.jsonl").read_text().split("\n") if line.strip()]
    records = [r for r in records if r["source"]["batch"] == source["batch"]]
    _verify_cells(source, Path(batch), records)
    results = {}
    for record in records:
        key = (record["source"]["case"], record["source"]["solver"])
        if key in results:
            raise ValueError("Duplicate semantic cell")
        path = out / "results" / f"{record['key']}.json"
        if not path.exists():
            results[key] = {"status": "pending" if record["status"] == "ready" else record["status"],
                            "run_id": manifest["run_id"],
                            "judging_sha16": manifest["code"]["judging_sha16"]}
            continue
        result = json.loads(path.read_text())
        if result["source"] != record["source"]:
            raise ValueError("Semantic result belongs to a different source answer")
        task = JudgeTask(**{**record["task"], "item_ids": tuple(record["task"]["item_ids"])})
        if task.fingerprint != record["task_sha256"]:
            raise ValueError("Use the original protocol implementation to read this historical run")
        journal = out / "samples" / f"{record['key']}.jsonl"
        if not journal.is_file():
            raise ValueError("Semantic report requires the independent sample journal")
        samples = [json.loads(line) for line in journal.read_text().split("\n") if line.strip()]
        if samples != result["consensus"]["samples"]:
            raise ValueError("Semantic result samples disagree with the independent journal")
        def no_dispatch(request):
            raise AssertionError("A report cannot purchase or invent missing votes")
        consensus = evaluate_consensus(task, no_dispatch, prior_samples=samples)
        summary = summarize_verdicts(consensus["verdicts"], record["rubric"])
        if consensus != result["consensus"] or summary != result["summary"]:
            raise ValueError("Persisted semantic result disagrees with its raw votes")
        capped, cap_changed = apply_proposal_cap(task, consensus)
        status = consensus["status"]
        if cap_changed:
            summary = summarize_verdicts(capped, record["rubric"])
            status = "unresolved" if any(v is None for v in capped.values()) else "resolved"
        results[key] = {"status": status, "summary": summary,
                        "verdicts": dict(capped),
                        "run_id": manifest["run_id"],
                        "judging_sha16": manifest["code"]["judging_sha16"]}
        if cap_changed:
            results[key]["proposal_cap"] = {"atoms": cap_changed,
                                            "before": {k: consensus["verdicts"][k] for k in cap_changed},
                                            "after": {k: capped[k] for k in cap_changed}}
        if "data_availability" in task.item_ids and task.atom_template is None:
            # Lean tasks were built on the delivered-window truth (record["reference_check"]).
            from .semantic_visibility import parse_reference, visible_probe
            reference, _ = parse_reference(task.prompt)
            availability = visible_probe(reference)
            if availability is not None and availability["mismatch"]:
                # Preserve the source consensus/summary on disk. A known wrong
                # reference is not a valid measurement, even after two votes.
                results[key]["summary"] = deepcopy(summary)
                results[key]["summary"]["metrics"]["noop_ok"] = None
                results[key]["status"] = "needs_visibility_correction"
                results[key]["reference_issue"] = availability
    return manifest, results


COMPOSITE_KIND = "semantic-composite-view-v1"


def _subset_chain(tip: Path) -> list[Path]:
    """`tip` followed by every run it was cut from; each cut is re-verified here."""
    from .semantic_pipeline import digest
    runs, seen, current = [], set(), Path(tip).resolve()
    while True:
        if current in seen:
            raise ValueError("Cyclic semantic subset chain")
        seen.add(current)
        runs.append(current)
        manifest = json.loads((current / "manifest.json").read_text())
        if "subset_of" not in manifest:
            return runs
        from .semantic_runref import subset_base
        from .semantic_rubric import judging_policy
        base = subset_base(current, manifest)
        base_manifest = json.loads((base / "manifest.json").read_text())
        if (digest(base / "manifest.json") != manifest["sampling"]["base_manifest_sha256"]
                or base_manifest["tasks_sha256"] != manifest["sampling"]["base_tasks_sha256"]
                or manifest["sources"] != base_manifest["sources"]
                or judging_policy(manifest["policy"]) != judging_policy(base_manifest["policy"])):
            raise ValueError("A semantic subset no longer matches the run it was cut from")
        base_rows = {}
        for line in (base / "tasks.jsonl").read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                base_rows[row["key"]] = row
        for line in (current / "tasks.jsonl").read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                if base_rows.get(row["key"]) != row:
                    raise ValueError("A subset task differs from its base task")
                if (base / "results" / f"{row['key']}.json").exists():
                    raise ValueError("A subset re-judges a cell its base already finished")
        current = base


def read_chain(tip: Path, batch: Path) -> tuple[dict, dict, dict]:
    """Union of one subset chain: every finished cell comes from exactly one run.

    Returns (tip manifest, results, origin run id per cell). Unfinished cells keep
    the newest run's pending state; 'empty'/'unparseable' states stay missing.
    """
    runs = _subset_chain(tip)
    merged, origin = {}, {}
    tip_manifest = None
    for run in reversed(runs):
        manifest, results = read_run(run, batch)
        tip_manifest = manifest
        for key, cell in results.items():
            finished = "summary" in cell
            if key not in merged or (finished and "summary" not in merged[key]):
                merged[key], origin[key] = cell, manifest["run_id"]
            elif finished:
                raise ValueError("A semantic cell was finished in two runs of one chain")
            elif merged[key].get("status") in ("pending",) and cell.get("status") == "pending":
                origin[key] = manifest["run_id"]
    return tip_manifest, merged, origin


def read_composite(spec_path: Path, batch: Path) -> tuple[dict, dict]:
    """A declared view: one subset chain plus whole-set corrective runs on its root."""
    from .semantic_corrections import validate_correction
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("kind") != COMPOSITE_KIND:
        raise ValueError("Unrecognized semantic view declaration")
    from .semantic_runref import view_run
    tip = view_run(spec_path, spec["chain_tip"])
    chain = [str(run) for run in _subset_chain(tip)]
    manifest, results, _ = read_chain(tip, batch)
    for correction in spec.get("corrections", []):
        from .semantic_runref import correction_base
        correction = view_run(spec_path, correction)
        metadata = json.loads((correction / "manifest.json").read_text())
        base_manifest, _, records = validate_correction(correction)
        if str(correction_base(correction, metadata["correction"])) not in chain:
            raise ValueError("A correction must apply to a run in the declared chain")
        corrected_manifest, corrections = read_run(correction, batch)
        judged = source_batch(corrected_manifest, batch)[0]["batch"]
        expected = {(r["source"]["case"], r["source"]["solver"]) for r in records
                    if r["source"]["batch"] == judged}
        if set(corrections) != expected or not expected <= results.keys():
            raise ValueError("Correction view must cover exactly the affected base cells")
        _apply_correction(results, corrections, metadata, base_manifest, corrected_manifest)
    return manifest, results


def _apply_correction(results, corrections, metadata, base_manifest, corrected_manifest):
    """Whole-cell corrections replace the cell; atom corrections replace one group only."""
    from .semantic_atom_correction import KINDS as ATOM_KINDS, SOURCE_KIND, merge_atom, merge_atoms
    for key, cell in corrections.items():
        note = {"kind": metadata["correction"]["kind"], "base_run_id": base_manifest["run_id"],
                "base_tasks_sha256": metadata["correction"]["base_tasks_sha256"]}
        if metadata["correction"]["kind"] == SOURCE_KIND:
            merged = merge_atoms(results[key], cell, "tests_precision", cell.get("verdicts", {}))
        if metadata["correction"]["kind"] in ATOM_KINDS:
            if metadata["correction"]["kind"] != SOURCE_KIND:
                merged = merge_atom(results[key], cell, ATOM_KINDS[metadata["correction"]["kind"]]["group"])
            if "summary" in results[key]:
                # A distinct judging vintage: base votes plus the corrected atom's votes.
                merged["judging_sha16"] = hashlib.sha256(json.dumps(
                    [results[key].get("judging_sha16"), corrected_manifest["code"]["judging_sha16"],
                     metadata["correction"]["atom"]]).encode()).hexdigest()[:16]
            merged.setdefault("atom_corrections", []).append(
                {**note, "atom": metadata["correction"]["atom"], "run_id": corrected_manifest["run_id"],
                 "judging_sha16": corrected_manifest["code"]["judging_sha16"],
                 "status": cell.get("status")})
            results[key] = merged
            continue
        cell.update(run_id=corrected_manifest["run_id"],
                    judging_sha16=corrected_manifest["code"]["judging_sha16"], correction=note)
        results[key] = cell


def view_for_batch(rows: list[dict], batch: Path, out: Path | None) -> list[dict]:
    if out is None:
        viewed = overlay_rows(rows, {}, run_id="not_run", judging_sha="not_run")
    elif Path(out).is_file():
        manifest, results = read_composite(Path(out), batch)
        viewed = overlay_rows(rows, results, run_id=manifest["run_id"],
                              judging_sha=manifest["code"]["judging_sha16"])
    else:
        metadata = json.loads((out / "manifest.json").read_text())
        if "correction" in metadata:
            from .semantic_corrections import validate_correction
            base_manifest, _, correction_records = validate_correction(out)
            from .semantic_runref import correction_base
            base = correction_base(out, metadata["correction"])
            _, results = read_run(base, batch)
            manifest, corrections = read_run(out, batch)
            judged = source_batch(manifest, batch)[0]["batch"]
            expected = {(r["source"]["case"], r["source"]["solver"]) for r in correction_records
                        if r["source"]["batch"] == judged}
            if set(corrections) != expected or not expected <= results.keys():
                raise ValueError("Correction view must cover exactly the affected base cells")
            _apply_correction(results, corrections, metadata, base_manifest, manifest)
        else:
            manifest, results = read_run(out, batch)
        viewed = overlay_rows(rows, results, run_id=manifest["run_id"],
                              judging_sha=manifest["code"]["judging_sha16"])
    enrich_urgency(viewed, batch)
    return viewed


def urgency_from_saved_actions(slices: list[dict], gold_urgency: str) -> dict:
    """Read the existing categorical rule, separating endpoint and trajectory.

    This is a deterministic observation under the saved canonical reference,
    not a fresh time-dependent clinical adjudication.
    """
    from .judges.differential import URGENCY_ACTION
    if gold_urgency not in URGENCY_ACTION:
        values = []
    else:
        accepted, _ = URGENCY_ACTION[gold_urgency]
        values = [None if (r.get("raw_empty") or r.get("raw_unparseable"))
                  else r.get("action") in accepted for r in sorted(slices, key=lambda r: r["t"])]
    known = [v for v in values if v is not None]
    return {"wk_urgency_ok_last": values[-1] if values else None,
            "wk_urgency_ok_rate": sum(known) / len(known) if known else None,
            "wk_n_slices_urgency_judged": len(known)}


def enrich_urgency(rows: list[dict], batch: Path) -> None:
    """Add observations from saved inputs only; missing evidence remains explicit."""
    selected = [r for r in rows if r.get("geometry") == "slices"]
    if not selected:
        return
    if not all((batch / name).is_file() for name in ("cases.jsonl", "payloads.jsonl")):
        for row in selected:
            row["urgency_observation_status"] = "not_recomputed:missing_saved_payloads"
        return
    from .payloads import load_payloads
    from .store import load_cases
    from .judges._helpers import _gold, gold_kind
    from .semantic_pipeline import digest
    load_cases(batch / "cases.jsonl")
    payloads = load_payloads(batch / "payloads.jsonl")
    payload_sha = digest(batch / "payloads.jsonl")
    for row in selected:
        _, vp = payloads[row["case"]]
        if not gold_kind(vp).startswith("ddx:"):
            row["urgency_observation_status"] = "not_applicable:no_diagnostic_reference"
            continue
        row.update(urgency_from_saved_actions(row["slice_rows"], _gold(vp, "urgency", None)))
        row["urgency_observation_status"] = "recomputed:canonical_reference"
        row["urgency_observation_source"] = {"payload_sha256": payload_sha,
            "basis": "saved actions against canonical-T reference; not time-dependent clinical review"}


def profile_for_rows(profile, rows: list[dict]):
    """Old proxy validation cannot certify a replacement LLM judgment."""
    if not any(isinstance(r.get("semantic"), dict) for r in rows):
        return profile
    profile = deepcopy(profile)
    for name in REPLACED:
        if name in profile.metrics:
            profile.metrics[name].pop("validity", None)
            profile.metrics[name]["pending_decision"] = (
                "LLM judging is new and preliminary; old proxy validation does not transfer")
    return profile


def report_section(rows: list[dict]) -> str:
    """Separate primary coverage and non-scoring observations; no ranking claims."""
    selected = [r for r in rows if isinstance(r.get("semantic"), dict)]
    if not selected:
        return ""
    by = defaultdict(list)
    for row in selected:
        by[row["solver"]].append(row)
    lines = ["## LLM semantic judging — preliminary, not final", "",
             "Semantic dimensions replace their code proxies. Other measurements retain their code origin. "
             "Judge agreement is not clinician approval or independent human validation. "
             "An unresolved applicable primary judgment withholds the composite score and is never replaced by a value. "
             "A case with no answer is a different state: the default board rule (`rank_rule.fill_unanswered`) scores it 0.", "",
             "| Model | Dimension | Mean of resolved cells | Resolved / applicable |", "|---|---|---|---|"]
    for model, cells in sorted(by.items()):
        for name in PRIMARY + AUX_DIMS:
            applicable = [r for r in cells if r["semantic"]["metric_states"][name] != "not_applicable"]
            values = [r[name] for r in applicable if r[name] is not None]
            mean = f"{sum(values)/len(values):.3f}" if values else "—"
            states = Counter(r["semantic"]["metric_states"][name] for r in applicable)
            state_text = ", ".join(f"{key}={n}" for key, n in sorted(states.items()))
            lines.append(f"| {model} | {name} | {mean} | {len(values)} / {len(applicable)} ({state_text}) |")
    lines += ["", "<details>", "<summary>Auxiliary observations — not included in the composite</summary>", "",
              "| Model | Observation | Mean | Measured cells |", "|---|---|---|---|"]
    from .scoring import load_profile
    observations = tuple(dict.fromkeys((*AUXILIARY,
        *(name for name, spec in load_profile().metrics.items() if spec["role"] == "diagnostic"),
        "wk_urgency_ok_last", "wk_urgency_ok_rate", "wk_n_slices_urgency_judged")))
    for model, cells in sorted(by.items()):
        for name in observations:
            values = [r["semantic"]["metrics"].get(name) if name in AUXILIARY else r.get(name) for r in cells]
            values = [v for v in values if type(v) in (int, float, bool)]
            mean = f"{sum(values)/len(values):.3f}" if values else "—"
            lines.append(f"| {model} | {name} | {mean} | {len(values)} / {len(cells)} |")
    lines += ["", "Unmeasured and not-applicable observations are not imputed. "
              "Urgency endpoint, trajectory mean and judged-slice count are distinct quantities. "
              "Recomputed urgency uses the saved canonical reference, not a new time-dependent clinical review.",
              "", "</details>", "", "---", ""]
    return "\n".join(lines)


# ---- Board rule for a semantic track ------------------------------------------------
#: Preregistered before any board after the proposed_test source correction was read.
#: R0: the composite on the common complete cases (every real model has every applicable
#: primary dimension resolved); at least `min_common_cases` of them => composite with a
#: 95% case-bootstrap interval. R1, only when R0 has fewer: every case every real model
#: answered, each missing primary cell filled with 0 and with 1; if any model's two fills
#: differ by more than `r1_max_width` the track gets no ranking, otherwise the fill
#: interval is reported and a separation must hold under both fills. Tier = 1 + the number
#: of models significantly better (paired case bootstrap, Holm, alpha); ties share a tier.
#: Phase 5 compares composites only on at least `phase5_min_cases` cases complete in
#: every repeat.
BOARD_RULE = {"min_common_cases": 20, "boot": 10000, "seed": 20260929,
              "r1_max_width": 0.05, "alpha": 0.05, "phase5_min_cases": 10}
MISSING_ANSWER = ("missing_response", "empty_response", "unparseable_response")
OK_STATES = ("resolved", "not_applicable")


def phase5_composite_allowed(n_complete: int) -> bool:
    return n_complete >= BOARD_RULE["phase5_min_cases"]


def _dim_state(row: dict, dim: str) -> str:
    return row["semantic"]["metric_states"].get(dim, row["semantic"]["status"])


def common_complete_cases(rows: list[dict]) -> tuple[list[str], dict]:
    """Cases on which every model has every applicable primary dimension resolved."""
    models = sorted({r["solver"] for r in rows})
    cells = {(r["case"], r["solver"]): r for r in rows}
    keep, excluded = [], {}
    for case in sorted({r["case"] for r in rows}):
        reasons = []
        for model in models:
            row = cells.get((case, model))
            if row is None:
                reasons.append(f"{model}:no_row")
                continue
            bad = [d for d in PRIMARY if _dim_state(row, d) not in OK_STATES]
            if bad:
                reasons.append(f"{model}:{row['semantic']['status']}:{'+'.join(bad)}")
        if reasons:
            excluded[case] = reasons
        else:
            keep.append(case)
    return keep, excluded


def fill_missing(rows: list[dict], value: float) -> tuple[list[dict], Counter]:
    """Set each missing applicable primary cell to `value`; returns rows and fills per model.

    A dimension every other model has as not applicable on that case is not applicable here too.
    """
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    out, fills = [], Counter()
    for source in rows:
        row = deepcopy(source)
        sem = row["semantic"]
        for dim in PRIMARY:
            if _dim_state(row, dim) in OK_STATES:
                continue
            others = [_dim_state(o, dim) for o in by_case[row["case"]] if o["solver"] != row["solver"]]
            if others and all(s == "not_applicable" for s in others):
                sem["metric_states"][dim] = "not_applicable"
                continue
            row[dim] = value
            sem["metric_states"][dim] = "resolved"
            fills[row["solver"]] += 1
        if all(_dim_state(row, d) in OK_STATES for d in PRIMARY):
            sem["status"] = "resolved"
        out.append(row)
    return out, fills


def composite_scores(rows: list[dict]) -> dict:
    """The production composite (`analytics.rank_ddx`) per model; None = unscored."""
    from .analytics import rank_ddx
    return {r["model"]: r.get("score") for r in rank_ddx(rows)}


_BOOT = {}


def _boot_one(draw):
    rows = []
    for i, case in enumerate(draw):
        for row in _BOOT["by_case"][case]:
            rows.append({**row, "case": f"{case}#{i}"})
    return _BOOT["score_fn"](rows)


def bootstrap_scores(rows: list[dict], cases: list[str], *, score_fn, boot: int, seed: int,
                     workers: int = 1) -> list[dict]:
    """Case-bootstrap composites: `boot` resamples of `cases`, same draws for every model."""
    import random
    rng = random.Random(seed)
    draws = [[cases[rng.randrange(len(cases))] for _ in cases] for _ in range(boot)]
    by_case = defaultdict(list)
    for row in rows:
        if row["case"] in set(cases):
            by_case[row["case"]].append(row)
    _BOOT.update(by_case=by_case, score_fn=score_fn)
    try:
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                return pool.map(_boot_one, draws, chunksize=max(1, boot // (workers * 8)))
        return [_boot_one(d) for d in draws]
    finally:
        _BOOT.clear()


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def _interval(xs: list[float]) -> list[float] | None:
    xs = sorted(xs)
    if not xs:
        return None
    return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]


def _significant(point: dict, samples: list[dict], alpha: float) -> dict:
    """(better, worse) -> Holm-adjusted two-sided paired-bootstrap p, for point-better pairs."""
    import itertools
    raw = {}
    for a, b in itertools.combinations(sorted(point), 2):
        diffs = [s[a] - s[b] for s in samples if s.get(a) is not None and s.get(b) is not None]
        n = len(diffs)
        le = (sum(d <= 0 for d in diffs) + 1) / (n + 1)
        ge = (sum(d >= 0 for d in diffs) + 1) / (n + 1)
        better, worse = (a, b) if point[a] >= point[b] else (b, a)
        raw[(better, worse)] = min(1.0, 2 * min(le, ge))
    adjusted = holm(raw)
    return {pair: p for pair, p in adjusted.items() if p < alpha and point[pair[0]] > point[pair[1]]}


def track_board(rows: list[dict], *, score_fn=None, boot: int | None = None, seed: int | None = None,
                workers: int = 1, rule: dict | None = None) -> dict:
    """Apply the board rule to one track's real-model rows."""
    rule = {**BOARD_RULE, **(rule or {})}
    score_fn = composite_scores if score_fn is None else score_fn
    boot = rule["boot"] if boot is None else boot
    seed = rule["seed"] if seed is None else seed
    models = sorted({r["solver"] for r in rows})
    keep, excluded = common_complete_cases(rows)
    out = {"n_common_complete": len(keep), "excluded_cases": excluded, "boot": boot, "seed": seed,
           "rule_parameters": rule}
    if len(keep) >= rule["min_common_cases"]:
        cases = set(keep)
        scored = [r for r in rows if r["case"] in cases]
        point = {m: s for m, s in score_fn(scored).items() if s is not None}
        samples = bootstrap_scores(scored, keep, score_fn=score_fn, boot=boot, seed=seed, workers=workers)
        sig = _significant(point, samples, rule["alpha"])
        out.update(rule="R0", ranked=True, n_cases=len(keep),
                   why=f"{len(keep)} common complete cases >= {rule['min_common_cases']}")
        out["models"] = [{"model": m, "score": None if m not in point else round(point[m], 4),
                          "ci95": _interval([s[m] for s in samples if s.get(m) is not None]) if m in point else None,
                          "tier": 1 + sum(w == m for _, w in sig) if m in point else None}
                         for m in models]
        out["significant_pairs"] = [{"better": a, "worse": b, "diff": round(point[a] - point[b], 4),
                                     "p_holm": round(p, 4)} for (a, b), p in sorted(sig.items())]
        return out
    cells = {(r["case"], r["solver"]): r for r in rows}
    unanswered = sorted({r["case"] for r in rows if any(
        (r["case"], m) not in cells or cells[(r["case"], m)]["semantic"]["status"] in MISSING_ANSWER
        for m in models)})
    cases = sorted({r["case"] for r in rows} - set(unanswered))
    answered = [r for r in rows if r["case"] in set(cases)]
    low_rows, fills = fill_missing(answered, 0.0)
    high_rows, _ = fill_missing(answered, 1.0)
    low, high = score_fn(low_rows), score_fn(high_rows)
    widths = {m: abs(high[m] - low[m]) for m in models if low.get(m) is not None and high.get(m) is not None}
    entries = {m: {"model": m, "imputed_cells": fills.get(m, 0),
                   "fill_interval": ([round(min(low[m], high[m]), 4), round(max(low[m], high[m]), 4)]
                                     if m in widths else None),
                   "fill_width": round(widths[m], 4) if m in widths else None} for m in models}
    out.update(rule="R1", n_cases=len(cases), unanswered_cases=unanswered)
    wide = sorted(m for m, w in widths.items() if w > rule["r1_max_width"])
    if wide or not widths:
        out.update(ranked=False, models=[entries[m] for m in models],
                   why=(f"{len(keep)} common complete cases < {rule['min_common_cases']}; fill interval "
                        f"wider than {rule['r1_max_width']} for {', '.join(wide) or 'every model'}: "
                        "no ranking, per-dimension readings only"))
        return out
    sig_by_fill = {}
    for name, filled_rows, point in (("fill_0", low_rows, low), ("fill_1", high_rows, high)):
        point = {m: s for m, s in point.items() if s is not None}
        samples = bootstrap_scores(filled_rows, cases, score_fn=score_fn, boot=boot, seed=seed, workers=workers)
        sig_by_fill[name] = _significant(point, samples, rule["alpha"])
        for m in models:
            entries[m].setdefault("ci95", {})[name] = (_interval([s[m] for s in samples if s.get(m) is not None])
                                                       if m in point else None)
    both = set(sig_by_fill["fill_0"]) & set(sig_by_fill["fill_1"])
    for m in models:
        entries[m]["tier"] = 1 + sum(w == m for _, w in both) if m in widths else None
    out.update(ranked=True, models=[entries[m] for m in models],
               why=(f"{len(keep)} common complete cases < {rule['min_common_cases']}; every fill interval "
                    f"<= {rule['r1_max_width']}; a separation must hold under both fills"),
               significant_pairs=[{"better": a, "worse": b,
                                   "fill_0": (a, b) in sig_by_fill["fill_0"], "fill_1": (a, b) in sig_by_fill["fill_1"],
                                   "p_holm": {k: round(v[(a, b)], 4) for k, v in sig_by_fill.items() if (a, b) in v}}
                                  for (a, b) in sorted(both)],
               one_fill_only=[{"better": a, "worse": b, "fill": name}
                              for name, other in (("fill_0", "fill_1"), ("fill_1", "fill_0"))
                              for (a, b) in sorted(set(sig_by_fill[name]) - set(sig_by_fill[other]))])
    return out
