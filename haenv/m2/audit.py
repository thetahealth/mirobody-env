"""Pack ①'s declarations for the shared audit (`haenv/shared_audit.py`).

* `emission()` -- the shared emission gate on F1-F3 cases (SA-1 + SA-8) with ①'s callbacks: no known
  base condition equals a hidden line (BASE-1), and the frame scene names no gold line (LEAK-SCAN).
* `audit()` -- the batch audit (`tools/pack_audit.py --pack m2 --job <job> --ref <v1.0.1 batch>`):
  items are the F1-F3 cases; SA-5 reads the symptom length / structure features (SURF-1) against the
  review label and the hard tier, SA-6 the same features with the lab-stream baselines (world-v2
  PACK); SA-3 the (frame, tier) cells the job planned; SA-7 adds "no explained-away symptom dropped".
  Pack gates P-1 EA-1, P-2 EA-2, P-3 MENU-1, P-4 P0-2a(b), P-5 MED-1, P-7 BRIDGE read every case
  (`tools/m2_item_audit.py`); P-8 PLAN checks the job's cells against the proportion table at N
  (`haenv/m2/pack.py`). P-6 (V-3, frequency prior) is read on model rows by
  `tools/m2_verdicts.py`. SA-4's case-blind stubs are the stub ladder's Z-3 (`tools/m2_stub_ladder.py`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import functools
import importlib.util
import json

#: known base condition -> the hidden lines it would duplicate (BASE-1)
BASE_EQUIV = {"T2D": {"JD-T2D"}, "hypertension": {"JD-HTN", "JD-EHTN"}, "dyslipidemia": {"JD-DYSLIP"},
              "MASLD": {"JD-MASLD", "JD-NAFLD"}, "obesity": set(),
              "hypothyroidism": {"JD-HYPO"}, "CAD": {"JD-CAD"}, "CKD": {"JD-CKD", "JD-CKM"}}
BASE_STREAMS = ("LDL", "triglycerides", "ALT", "AST", "HbA1c", "fasting_glucose")


@functools.lru_cache(maxsize=1)
def gates():
    """`tools/m2_item_audit.py` (①'s gate functions and its item reader)."""
    from ..anchor import ROOT
    spec = importlib.util.spec_from_file_location("_m2_item_audit", ROOT / "tools" / "m2_item_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                # noqa: S102
    return mod


# ------------------------------------------------------------------------------ SA-1 callbacks

def _get(raw, k):
    return (raw.get(k) if isinstance(raw, dict) else getattr(raw, k, None)) or {}


def _plan_of(raw):
    if not isinstance(_get(raw, "adjudication").get("m2_frame"), dict):
        return None
    from ..external_gold import SLOT
    p = ((_get(raw, "latent_premise").get("meta") or {}).get(SLOT) or {}).get("m2")
    return p if isinstance(p, dict) else {}


def base_hits(known, threads) -> list[str]:
    comps = {t.get("component") for t in threads}
    return [f"known {k} is hidden line {sorted(BASE_EQUIV[k] & comps)}" for k in known if BASE_EQUIV.get(k, set()) & comps]


def gold_names(ddx: dict) -> set[str]:
    from .core import threads_of
    names = {str(a).lower() for t in threads_of(ddx) for a in [t.get("name"), *(t.get("aliases") or ())]
             if a and len(str(a)) >= 2}
    if ddx.get("diagnosis"):
        names.add(str(ddx["diagnosis"]).lower())
    return names


def _extra(gold, raw) -> list[tuple[str, str]]:
    from .core import threads_of
    adj = _get(raw, "adjudication")
    ddx = adj.get("ddx") or {}
    known = list(_get(raw, "user_profile").get("known_conditions") or ())
    out = [("base", d) for d in base_hits(known, threads_of(ddx))]
    scene = json.dumps(_get(raw, "prediction_context").get("m2_frame") or {}, ensure_ascii=False).lower()
    out += [("leak", f"gold line {n!r} in the frame scene") for n in sorted(gold_names(ddx)) if n in scene]
    return out


def emission():
    from ..shared_audit import Emission
    return Emission(gate="m2", block="m2_frame", plan_of=_plan_of, recompute=lambda gold, raw: [], extra=_extra)


# ------------------------------------------------------------------------------ batch audit

def _context(batch, job_path, ref):
    import yaml
    if job_path is None:
        raise SystemExit("pack ① needs --job (the M2 job the batch was built from)")
    cases = gates().collect(batch, job_path)
    job = yaml.safe_load(open(job_path, encoding="utf-8"))
    return {"cases": cases, "by_id": {c["cid"]: c for c in cases}, "job": job, "batch": batch, "ref": ref}


def rows(items, ctx) -> list[dict]:
    G = gates()
    out = []
    for it in items:
        c = ctx["by_id"][it["case_id"]]
        out.append({**dict(zip(G.SURF_FEATURES, G.surface_features(c["ledger_entries"]))),
                    **{f"base_{s}": float(c["base"].get(s, -1.0)) for s in BASE_STREAMS},
                    "warranted": int(bool(c["warranted"])), "hard": int(c.get("tier") == "hard"),
                    "class": it["class"], "spec": it["spec"], "case_id": it["case_id"]})
    return out


def _cell(it, ctx):
    c = ctx["by_id"][it["case_id"]]
    return (c["frame"], c.get("tier"))


def planned_cells(items, ctx) -> dict:
    """(frame, tier) -> items the generator planned (`_provenance.plan_cells`)."""
    plan = (ctx["job"].get("_provenance") or {}).get("plan_cells") or {}
    return {(f, t): n for f, ts in plan.items() for t, n in ts.items()}


def design_cells(job) -> dict:
    """Design cell (`haenv/m2/pack.py:DESIGN`) -> cases of the job."""
    out = collections.Counter()
    prior = (job.get("_provenance") or {}).get("m2_prior") or {}
    for c in job["cases"]:
        p = prior.get(c["case_id"]) or {}
        arm = ((c.get("latent") or {}).get("m2") or {}).get("arm")
        tier = p.get("tier")
        if p.get("frame") == "F0":
            out[f"f0_{tier}"] += 1
        elif arm == "negative":
            out[f"neg_{tier}"] += 1
        else:
            out["hard" if tier == "hard" else f"pos_{tier}"] += 1
    return dict(out)


def plan_gate(n_items):
    """P-8 PLAN: the job's cells equal the proportion table at n_items (the bridge cells capped by the
    job's recorded bridge supply)."""
    from .pack import BRIDGE, quotas

    def run(pool, rows_, ctx):
        design = (ctx["job"].get("_provenance") or {}).get("design") or {}
        want, _ = quotas(n_items, {k: design.get(k, 0) for k in BRIDGE})
        got = design_cells(ctx["job"])
        bad = {k: (want[k], got.get(k, 0)) for k in want if want[k] != got.get(k, 0)}
        return {"id": "P-8", "pass": not bad and design == want, "want": want, "got": got,
                **({"mismatch": {k: {"want": w, "got": g} for k, (w, g) in bad.items()}} if bad else {})}
    return run


def dropped_explained_symptom(it, ctx=None) -> list[str]:
    ea = (it.get("_m2") or {}).get("explain_away") or {}
    return [sy["text"] for ln in ea.get("lines") or () for sy in ln["symptoms"] if sy.get("dropped")]


def _gate(name):
    def run(pool, rows_, ctx):
        G, cases = gates(), ctx["cases"]
        if name == "P-4":
            r = G.check_p02a(G.p02a_matrix(cases), calibrate=True, acute_mat=G.p02a_matrix(cases, acute_ids=True))
        elif name == "P-7":
            r = ({"pass": False, "reason": "needs --ref (the v1.0.1 deterministic workup batch)"} if ctx["ref"] is None
                 else G.bridge(ctx["batch"], ctx["ref"], {c["cid"]: c["frame"] for c in cases}))
        else:
            r = {"P-1": G.check_ea1, "P-2": G.check_ea2, "P-3": G.check_menu1, "P-5": G.check_med1}[name](cases)
        return {"id": name, **r}
    return run


def audit(n_items: int = 50, seed: str = ""):
    from ..shared_audit import PackAudit
    G = gates()

    def with_case(items, ctx):
        for it in items:
            it["_m2"] = ctx["by_id"][it["case_id"]]
        return items

    return PackAudit(
        name="m2", block="m2_frame", classes=("hard", "other"),
        class_of=lambda g: g.get("frame") or None,
        rows=lambda items, ctx: [dict(r, **{"class": "hard" if r["hard"] else "other"}) for r in rows(items, ctx)],
        univariate=G.SURF_FEATURES, targets=("warranted", "hard"),
        surface=G.SURF_FEATURES + tuple(f"base_{s}" for s in BASE_STREAMS), surface_targets=("warranted", "hard"),
        # the score reads `warranted`; the hard tier is a generation label (distractor level), so a
        # shortcut to it is reported as profile, not gated
        gate_targets=("warranted",),
        cc=None, cell_of=_cell, planned_cells=planned_cells, select=None,
        realism={"dropped_explained_symptom": dropped_explained_symptom},
        pack_gates={**{k: _gate(k) for k in ("P-1", "P-2", "P-3", "P-4", "P-5", "P-7")}, "P-8": plan_gate(n_items)},
        context=_context, prepare=with_case)
