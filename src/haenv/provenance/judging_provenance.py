"""Outer provenance facade; frozen scoring modules never import this module.

Private function namespaces bind existing orchestration to metadata decorators.
No shared module is patched, and score calculations and their inputs are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from types import FunctionType

from haenv import analytics
import haenv.scoring.semantic.report as semantic_report
import haenv.scoring.semantic.seal as semantic_seal
import haenv.scoring.semantic.pipeline as semantic_pipeline
import haenv.scoring.management.pipeline as management_pipeline
from haenv.provenance.judging import criteria_provenance
from haenv.provenance.judging import criteria_used
from haenv.provenance.judging import current_fingerprints
from haenv.provenance.judging import recorded_criteria
from haenv.provenance.judging import merge_recorded_criteria
from haenv.provenance.judging import criterion_fingerprints

NotPublishable = analytics.NotPublishable


def _bind(function, **bindings):
    """Use an isolated namespace for outer metadata calls, without monkeypatching."""
    namespace = {**function.__globals__, **bindings}
    wrapped = FunctionType(function.__code__, namespace, function.__name__,
                           function.__defaults__, function.__closure__)
    wrapped.__kwdefaults__ = function.__kwdefaults__
    return wrapped


def judging_provenance(rows: list[dict], *, criteria=None) -> dict:
    """The judging version this batch's readings were measured under, and whether it is stale.

    Returns `{"measured_under", "judging_current", "stale", "why"}`; every
    aggregated reading carries it.
    """
    result = analytics.judging_provenance(rows)
    if criteria is not None or any(recorded_criteria(row) is not None for row in rows):
        result.update(criteria_provenance(rows, names=criteria))
    return result


def assert_publishable(rows: list[dict], where: str = "", *, criteria=None) -> None:
    """Publish gate: raise `NotPublishable` for a reading with a mixed or stale judging version.

    Unversioned batches are rejected at the publish layer but still render. Called
    only on paths that produce external material, such as `make_freeze` (which may
    pass `--allow-stale`).
    """
    # The judgment lives in `verifier_core.vintage`; this is the single entry point
    # that maps haenv's names onto it.
    from verifier_core.vintage import NotPublishable as _VCNotPublishable
    from verifier_core.vintage import assert_publishable as _vc_assert
    prov = judging_provenance(rows, criteria=criteria)
    try:
        if rows and (criteria is not None or any(recorded_criteria(row) is not None for row in rows)):
            current = current_fingerprints()
            names = sorted(set(criteria) if criteria is not None else
                           {name for row in rows for name in criteria_used(row)})
            if not names:
                raise _VCNotPublishable("No scoring criterion can be identified for these rows")
            for name in names:
                selected = [row for row in rows if criteria is not None or name in criteria_used(row)]
                projected = [{**row, "judging_sha16": (recorded_criteria(row) or {}).get(name)} for row in selected]
                _vc_assert(projected, current["criteria"].get(name), where=f"{where} · {name}")
        else:
            _vc_assert(rows, prov["judging_current"], where=where)
    except _VCNotPublishable as e:
        raise NotPublishable(str(e)) from e


def code_state():
    state = semantic_pipeline.code_state()
    ruler = current_fingerprints()
    return {**state, "judging_criteria": {"semantic": ruler["criteria"]["semantic"]},
            "judging_total_sha16": ruler["total_sha16"]}


def overlay_rows(rows, results, *, run_id, judging_sha, criteria=None):
    viewed = semantic_report.overlay_rows(rows, results, run_id=run_id, judging_sha=judging_sha)
    for row in viewed:
        cell = results.get((row["case"], row["solver"]))
        stamps = (cell.get("judging_criteria") or {}) if cell is not None and "judging_criteria" in cell else (criteria or {})
        stamp = stamps.get("semantic") if row["semantic"]["status"] == "resolved" else None
        if recorded_criteria(row) is not None or stamp:
            merge_recorded_criteria(row, {"semantic": stamp})
    return viewed


def read_run(out, batch):
    manifest, results = semantic_report.read_run(out, batch)
    stamps = _run_criteria(manifest)
    for cell in results.values():
        cell["judging_criteria"] = dict(stamps)
    return manifest, results


def _run_criteria(manifest):
    code = manifest["code"]
    if "judging_criteria" in code:
        return code["judging_criteria"]
    rev = manifest.get("sealed_commit")
    if not manifest.get("sealed") or not rev or not code.get("semantic"):
        return {}                    # no verifiable historical identity
    if not re.fullmatch(r"[0-9a-f]{7,40}", rev):
        raise ValueError("Semantic seal requires an immutable clean source commit")
    return _criteria_at_seal(rev, code["judging_sha16"], tuple(sorted(code["semantic"].items())))


@lru_cache(maxsize=32)
def _criteria_at_seal(rev, legacy, semantic_items):
    """Recover a clean sealed scorer from Git, verifying its existing receipts."""
    import haenv.provenance.anchor as anchor
    from haenv.scoring.semantic.source import _semantic_cached
    segment = anchor._segment_at(rev, "JUDGING")
    if anchor.fingerprint_at(rev, segment) != legacy:
        raise ValueError("Semantic seal judging_sha16 does not reproduce at its commit")
    recorded = dict(semantic_items)
    required = set(segment) - set(anchor._segment_at(rev, "INFRA"))
    source_declarations = anchor._declarations_at(rev)
    from haenv.provenance.history import revision_path
    if source_declarations["_entry"] != "tools/make_freeze.py" and "CRITERION_DECLARATIONS" not in source_declarations:
        raise ValueError("Missing explicit criterion entry at the sealed commit")
    declarations = anchor._git_blob(rev, revision_path(source_declarations.get("CRITERION_DECLARATIONS", "haenv/judging.py"), source_declarations))
    if declarations is None and "CRITERION_DECLARATIONS" in source_declarations:
        raise ValueError("Missing declared criterion source at the sealed commit")
    if declarations is None:
        # Seals predating per-criterion declarations remain verifiable legacy
        # artifacts, without inventing a current criterion identity for them.
        required |= {"haenv/semantic_pipeline.py", "haenv/semantic_report.py"}
    else:
        from haenv.provenance.history import contract_tables_at
        contracts, _ = contract_tables_at(lambda rel: anchor._git_blob(rev, rel), source_declarations)
        required |= set(contracts["semantic"]["roots"])
    if not required <= recorded.keys():
        raise ValueError("Semantic seal has incomplete source receipts")
    for rel, digest in semantic_items:
        blob = anchor._git_blob(rev, revision_path(rel, source_declarations))
        if blob is None or hashlib.sha256(_semantic_cached(blob, Path(rel).suffix.lower())).hexdigest() != digest:
            raise ValueError(f"Semantic seal source does not reproduce at its commit: {rel}")
    if declarations is None:
        return {}
    ruler = criterion_fingerprints(rev)
    return {"semantic": ruler["criteria"]["semantic"]}


def _apply_correction(results, corrections, metadata, base_manifest, corrected_manifest):
    from haenv.scoring.semantic.atom_correction import KINDS
    base = {key: (cell.get("judging_criteria") or {}).get("semantic") for key, cell in results.items()}
    semantic_report._apply_correction(results, corrections, metadata, base_manifest, corrected_manifest)
    if metadata["correction"]["kind"] in KINDS:
        for key, cell in corrections.items():
            a, b = base[key], (cell.get("judging_criteria") or {}).get("semantic")
            stamp = (a if a == b else hashlib.sha256(json.dumps(
                [a, b, metadata["correction"]["atom"]], separators=(",", ":")).encode()).hexdigest()[:16]) if a and b else None
            results[key]["judging_criteria"] = {"semantic": stamp} if stamp else {}


def read_chain(tip, batch):
    return _bind(semantic_report.read_chain, read_run=read_run)(tip, batch)


def read_composite(spec, batch):
    return _bind(semantic_report.read_composite, read_chain=read_chain,
                 read_run=read_run, _apply_correction=_apply_correction)(spec, batch)


def view_for_batch(rows, batch, out):
    return _bind(semantic_report.view_for_batch, read_run=read_run, read_composite=read_composite,
                 overlay_rows=overlay_rows, _apply_correction=_apply_correction)(rows, batch, out)


# Capture the source identity when preparing, before sealing or dispatching.
def prepare_batches(*args, **kwargs):
    return _bind(semantic_pipeline.prepare_batches, code_state=code_state)(*args, **kwargs)


def default_after_run(*args, **kwargs):
    return _bind(semantic_pipeline.default_after_run, prepare_batches=prepare_batches)(*args, **kwargs)


def prepare_correction(*args, **kwargs):
    from haenv.scoring.semantic.corrections import prepare_correction as prepare
    # The frozen function locally imports code_state. Stamp only its unsealed output.
    result = prepare(*args, **kwargs)
    return _stamp_prepared(result, kwargs.get("out", args[1] if len(args) > 1 else None), "semantic")


def prepare_atom_correction(*args, **kwargs):
    from haenv.scoring.semantic.atom_correction import prepare_atom_correction as prepare
    result = prepare(*args, **kwargs)
    return _stamp_prepared(result, kwargs.get("out", args[1] if len(args) > 1 else None), "semantic")


def _stamp_prepared(manifest, out, name):
    if manifest.get("sealed"):
        raise ValueError("Cannot restamp a sealed judgment manifest")
    ruler = current_fingerprints()
    manifest["code"]["judging_criteria"] = {name: ruler["criteria"][name]}
    manifest["code"]["judging_total_sha16"] = ruler["total_sha16"]
    semantic_seal._write_json(Path(out) / "manifest.json", manifest)
    return manifest


def management_prepare(*args, **kwargs):
    result = management_pipeline.prepare(*args, **kwargs)
    if "repeat" in result:
        return result  # Preserve the saved criterion identity of the repeated prompts.
    import yaml
    ruler = (criterion_fingerprints(overrides={"registry/management_rubric.yaml":
             yaml.safe_dump(result["policy"], allow_unicode=True, sort_keys=False).encode()})
             if kwargs.get("policy_path") else current_fingerprints())
    result["code"]["judging_criteria"] = {"management_rubric": ruler["criteria"]["management_rubric"]}
    result["code"]["judging_total_sha16"] = ruler["total_sha16"]
    management_pipeline.atomic_json(Path(kwargs.get("out", args[1] if len(args) > 1 else None)) / "manifest.json", result)
    return result


def management_run(out, **kwargs):
    if (kwargs.get("cfg") or {}).get("judge_routes") and not kwargs.get("offline", True):
        from haenv.scoring.management.route_run import run
        result = run(out, **kwargs)
    else:
        # The sealed scorer stays byte-identical. Compare its criterion identity in
        # this outer provenance layer, so transport-only changes do not invalidate
        # already prepared runs, including runs using the legacy single route.
        from haenv.scoring.management.route_run import validate as validate_management
        execute = _bind(management_pipeline._run, validate=validate_management)
        result = _bind(management_pipeline.run, _run=execute)(out, **kwargs)
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    _stamp_management_results(Path(out), manifest["code"].get("judging_criteria", {}),
                              manifest["code"].get("judging_total_sha16"))
    return result


def management_saved_prepare(previous_runs, out, **kwargs):
    return _stamp_prepared(management_pipeline.prepare_saved(previous_runs, out, **kwargs), out, "management_rubric")


def _stamp_management_results(out, stamps, total, *, preserve=False):
    path = out / "rubric_results.jsonl"
    rows = management_pipeline.read_jsonl(path)
    for row in rows:
        if preserve:
            row["source_judging_criteria"] = row.get("judging_criteria") or row.get("code", {}).get("judging_criteria")
        row["judging_criteria"] = dict(stamps)
        row["judging_total_sha16"] = total
        row["Q"] = row["summary"].get("quality_score")
    management_pipeline.write_jsonl(path, rows)


def management_precision_prepare(previous, out, *, run_id):
    from haenv.scoring.management.precision_rejudge import prepare_precision
    return _stamp_prepared(prepare_precision(previous, out, run_id=run_id), out, "management_rubric")


def management_precision_merge(out, destination=None):
    from haenv.scoring.management.precision_rejudge import merge_precision
    result = merge_precision(out, destination)
    code = json.loads((Path(out) / "manifest.json").read_text())["code"]
    _stamp_management_results(Path(result["out"]), code.get("judging_criteria", {}),
                              code.get("judging_total_sha16"))
    return result


def management_precision_run(out, **kwargs):
    from haenv.scoring.management.precision_rejudge import run_precision
    result = run_precision(out, **kwargs)
    code = json.loads((Path(out) / "manifest.json").read_text())["code"]
    for directory in (Path(out), Path(result["merged"]["out"])):
        _stamp_management_results(directory, code.get("judging_criteria", {}),
                                  code.get("judging_total_sha16"))
    return result


def management_rescore(out, destination, *, policy_path=None):
    import yaml
    policy = management_pipeline.load_policy(policy_path)
    ruler = criterion_fingerprints(overrides={"registry/management_rubric.yaml":
        yaml.safe_dump(policy, allow_unicode=True, sort_keys=False).encode()})
    result = management_pipeline.rescore(out, destination, policy_path=policy_path)
    _stamp_management_results(Path(destination), {"management_rubric": ruler["criteria"]["management_rubric"]},
                              ruler["total_sha16"], preserve=True)
    return result


def write_report(*args, **kwargs):
    from haenv import report
    render = _bind(report.render, judging_provenance=judging_provenance,
                   rank_ddx=rank_ddx, rank_models=rank_models)
    return _bind(report.write_report, render=render)(*args, **kwargs)


def rank_ddx(*args, **kwargs):
    return _bind(analytics.rank_ddx, judging_provenance=judging_provenance)(*args, **kwargs)


def rank_models(*args, **kwargs):
    return _bind(analytics.rank_models, judging_provenance=judging_provenance)(*args, **kwargs)
