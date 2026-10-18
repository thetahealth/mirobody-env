"""M2 background adapter: which known conditions a case carries besides its
primary disease, drawn from `registry/background_m2.yaml`. The generator (`tools/generation/m2_gen_job.py`) calls
`assign_background` once per pool; the build, the emission gates and the scene read the result as any other
comorbidity (`raw.comorbidities`), so nothing here writes a surface field.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import haenv.world.generation.rng as rng
from haenv.resources.regpath import load_registry


def registry() -> dict:
    return load_registry("background_m2.yaml") or {}


def rates() -> dict[str, float]:
    """condition -> per-case rate, the one number of `background_comorbidity.yaml:eligible`."""
    from haenv.world.generation.demographics import comorbid_pool
    pool = dict(comorbid_pool())
    return {f["condition"]: float(pool[f["condition"]]) for f in registry().get("factors") or ()}


def blocked_by_spectrum(factor: dict, spec_id: str) -> bool:
    """True when a drug of the factor moves a finding of the hidden line's spectrum."""
    if not spec_id:
        return False
    from haenv.m2.core import spectrum_ids
    moved = {fid for d in factor.get("drugs") or () for fid in (registry().get("drug_lab_effects") or {}).get(d, {}).get("findings") or ()}
    return bool(moved & spectrum_ids(spec_id))


def _draw(case_id: str, factor_id: str) -> float:
    return rng.unit(case_id, "m2-bg", factor_id)


def assign_background(rows: list[dict], by_id: dict) -> dict[str, list[str]]:
    """case_id -> conditions to attach (registry order, at most `max_background_comorbidities`).

    Each factor is a coin keyed on the case id and the factor id only. On a positive-arm case a factor whose
    drug can move a finding of the hidden line's spectrum is dropped; a negative-arm case, which has no hidden
    line, takes the factor at the share the positive arm of its frame realises, so the arm does not read
    through the base."""
    reg = registry()
    cap = int(reg.get("max_background_comorbidities") or 2)
    factors = list(reg.get("factors") or ())
    rate = rates()
    non_bridge = [r for r in rows if r["fw"] != "F0"]
    drop: dict[tuple, list[int]] = {}
    hit: dict[tuple, list[str]] = {}
    for r in non_bridge:
        if r["frame"].endswith("-neg"):
            continue
        cid, raw = r["case"], by_id[r["case"]]["raw"]
        spec = by_id[cid]["latent"].get("ddx_spec_id", "")
        for f in factors:
            if f["condition"] == raw.get("disease") or _draw(cid, f["id"]) >= rate[f["condition"]]:
                continue
            key = (r["fw"], f["id"])
            d = drop.setdefault(key, [0, 0])
            d[0] += 1
            if blocked_by_spectrum(f, spec):
                d[1] += 1
            else:
                hit.setdefault(cid, []).append(f["condition"])
    out: dict[str, list[str]] = {c: v[:cap] for c, v in hit.items()}
    for r in non_bridge:
        if not r["frame"].endswith("-neg"):
            continue
        cid, raw = r["case"], by_id[r["case"]]["raw"]
        for f in factors:
            n, b = drop.get((r["fw"], f["id"]), [0, 0])
            keep = 1.0 - (b / n if n else 0.0)
            if (f["condition"] != raw.get("disease") and _draw(cid, f["id"]) < rate[f["condition"]]
                    and rng.unit(cid, "m2-bg-keep", f["id"]) < keep):
                out.setdefault(cid, []).append(f["condition"])
    return {c: v[:cap] for c, v in out.items()}


def drug_allowed(condition: str, raw: dict) -> bool:
    """A condition whose drug is on the pregnancy-capable avoid list is not attached to such a woman."""
    import haenv.world.generation.demographics as D
    from haenv.world.clinical.med_course import CONCURRENT
    return not any(D.avoid_for(d, str(raw.get("sex") or ""), str(raw.get("age_range") or ""))
                   for d, _ in CONCURRENT.get(condition, ()))


def lookalike_record_format(entries: list[dict], cs) -> list[dict]:
    """Lookalike ledger entries of an F1-F3 case carry the fields of every other patient-reported entry
    (`reliability_status: reported`): the events renderer writes them with a `reliability` key of their own,
    which names the entry's role on its own. F0, v1.0.1 cases and a pack built with the layer off (no
    `m2.background` marker) keep the renderer's bytes."""
    if not ((getattr(cs, "latent", None) or {}).get("m2") or {}).get("background"):
        return entries
    out = []
    for e in entries:
        e = dict(e)
        if "reliability" in e:
            e.pop("reliability")
            e["reliability_status"] = "reported"
        out.append(e)
    return out


def sync_known_conditions(raw) -> None:
    """Name the background comorbidities in the record (`user_profile.known_conditions`) of an F1-F3 case.
    The build names the primary disease and the explainer bases only (`build.EXPLAINER_BASES`, shared with the
    v1.0.1 cases, whose comorbidities stay unnamed); a case of this pack names the conditions
    `registry/background_m2.yaml` attached, after the ones the build wrote. Idempotent."""
    meta = ((getattr(raw, "latent_premise", None) or {}).get("meta") or {})
    if not ((meta.get("_external") or {}).get("m2") or {}).get("background"):
        return
    conds = [f["condition"] for f in registry().get("factors") or ()]
    have = (getattr(raw, "latent_premise", None) or {}).get("patient_basics", {}).get("comorbidities") or []
    up = dict(raw.user_profile or {})
    known = list(up.get("known_conditions") or [])
    up["known_conditions"] = known + [c for c in conds if c in have and c not in known]
    raw.user_profile = up


def laboratory_documents(raw) -> None:
    """Laboratory procedure evidence, retrievable on every M2 case, blind to gold.

    Move the LDL calculation exemption out of its value/method field. The same
    laboratory document is supplied for direct and calculated results.
    """
    import haenv.world.generation.rng as rng
    pc = dict(raw.prediction_context or {})
    T = int(pc["prediction_time_T"])
    day = T - 1 - int(rng.unit(str(raw.case_id), "m2-lab-procedure") * 90)
    pc["laboratory_method_record"] = [{"day": day, "source": "检验科血脂方法文件", "scope": "LDL-C/TC/TG",
                                        "valid_from": day,
                                        "text": "甘油三酯≥4.5 mmol/L 时 LDL-C 采用直接测定，Friedewald 计算不适用。"}]
    raw.prediction_context = pc
    from haenv.world.observations.events_text import relocate_event_history
    relocate_event_history(raw)
