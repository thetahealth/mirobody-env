"""m2_item_audit.py -- pack ①'s own generation-time gates and its item reader (spec section 7).
Zero model calls. Run through the shared audit: `tools/pack_audit.py --pack m2` (declaration
`haenv/m2/audit.py`); the shared checks (SA-1 ... SA-9) live in `haenv/shared_audit.py`.

Each gate is a pure function over plain data, so the tests can feed it a mutated input and require
it to turn red:

  P-1 EA-1   an explained-away line shows no specific sign; each clause has a visible explainer
  P-2 EA-2   every gold line keeps a clue on the record, not explainable by another line of the case
  P-3 MENU-1 every gold line's key signals are reachable on the case's menu
  P-4 P0-2a  the hidden line is not guessable from demographics, known conditions and complaint
             category beyond the same-fold frame list (grouped GBM; v3 PREREG A2, m2-fix-r2 C)
  P-5 MED-1  no medicine named on the record is indicated for, or a marker of, a hidden line
  P-7 BRIDGE F0 payload, gold and world byte-identical to the v1.0.1 deterministic build
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: The M2 job the tool reads (revision r2: pack ① is `inputs/m2-pack1.job.yaml`); override with HAENV_M2_JOB.
M2_JOB = pathlib.Path(__import__("os").environ.get("HAENV_M2_JOB") or (ROOT / "inputs" / "m2-core.job.yaml"))
for _p in (str(ROOT), str(ROOT / "plugins")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: interpreter with scikit-learn installed, for `tools/m2_c2st.py`: `HAENV_SKLEARN_PY`, else this
#: interpreter when it has scikit-learn, else `python3` on PATH
SKLEARN_PY = (os.environ.get("HAENV_SKLEARN_PY")
              or (sys.executable if __import__("importlib.util").util.find_spec("sklearn") else None)
              or __import__("shutil").which("python3") or sys.executable)


def sklearn_job(job: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([SKLEARN_PY, str(ROOT / "tools" / "m2_c2st.py")], input=json.dumps(job),
                       capture_output=True, text=True, cwd=str(ROOT / "tools"), env=env, check=True)
    return json.loads(r.stdout)


def check_menu1(cases: list[dict]) -> dict:
    """Every gold line has at least one key signal and every key signal is on the menu (key signals
    are drawn from menu-reachable findings, so the second half alone would be circular); every
    explain-away key signal is on the menu. Coverage of all confirmatory findings is reported."""
    bad, unknown = [], set()
    n_conf = n_conf_reach = 0
    for c in cases:
        for t in c["threads"]:
            if not t.get("key_signals"):
                bad.append((c["cid"], t["name"], "line has no reachable key signal"))
            for k in t.get("key_signals") or []:
                if k not in c["reachable"]:
                    bad.append((c["cid"], t["name"], k))
            for k in t.get("confirmatory_all") or []:
                n_conf += 1
                n_conf_reach += k in c["reachable"]
        for k in c["ea_keys"]:
            if k not in c["vocab"]:
                unknown.add(k)
            elif k not in c["reachable"]:
                bad.append((c["cid"], "explain_away", k))
    return {"violations": bad, "ea_names_not_in_findings_vocab": sorted(unknown),
            "confirmatory_findings": n_conf, "confirmatory_on_menu": n_conf_reach, "pass": not bad}


from haenv.m2.audit import BASE_EQUIV  # noqa: E402  (the base-condition table of the ① emission gate)


def _ea1_terms(line: dict, c: dict, comps: list[str]) -> list:
    """(finding, row, visible explainers) of the table rows for this line and case whose explainer
    the solver can see. Rows come from the table by the case's spec (the record's row list only
    when the case carries no spec id). Retired rows are skipped and a drug-dependent row counts
    only when its drug is in the solver's medication text (`core.ea_rows` / `core.drug_visible`,
    the same filter and check the generator uses; revision r2-E1/E2)."""
    from haenv.m2 import core
    comp, spec = line["component"], c.get("spec")
    rows = [r for r in core.ea_rows() if core.EA_HIDDEN.get(str(r.get("hidden"))) == comp
            and (spec in (r.get("specs") or ()) if spec else r["id"] in (line.get("rows") or ()))]
    out = []
    for r in rows:
        if not core.drug_visible(r, c.get("visible_med_text")):
            continue
        vis = [k for k in (r.get("explainer_known") or ()) if k in c["known_visible"]]
        vis += [x for x in core.EA_LINE.get(str(r.get("explainer_line") or ""), ())
                if x in comps and x != comp and c["line_symptoms_on_record"].get(x)]
        if vis:
            out += [(f, r, vis) for f in r.get("findings") or ()]
    return out


def check_ea1(cases: list[dict]) -> dict:
    """EA-1 (spec 7; revision R-34): on every case with an explained-away line, the line shows no
    specific sign on the solver's record. Judged clause by clause on the record itself (the ledger),
    not on the generator's record: (a) every symptom the generator kept for the line is on the
    record; (b) every clause (symptom text and context) of every record symptom that belongs to the
    line -- by the generator's record, or by template retrieval -- is a finding of a table row whose
    explainer the solver can see (a known condition in `user_profile.known_conditions`, or another
    gold line with a symptom of its own on the record), or a context qualifier; (c) no clause of any
    record symptom carries a specific sign of the line (`specific_signs`) unless such a row
    licenses that clause. One violation per (case, line, symptom), with the offending clauses.
    Hard-tier cases are the gate's scope (spec); all cases are reported."""
    from haenv.m2 import core
    signs = core.specific_signs()
    name = {v: k for k, v in core.EA_HIDDEN.items()}
    bad, n_sym, n_cases, hard = [], 0, 0, 0
    hard_cases_bad: set = set()
    for c in cases:
        ea = c.get("explain_away") or {}
        if not ea.get("lines"):
            continue
        n_cases += 1
        is_hard = c.get("tier") == "hard"
        hard += is_hard
        comps = list(c.get("lines") or ()) or list(c.get("line_symptoms_on_record") or {})
        entries = c.get("ledger_entries") or [(t, "") for t in sorted(c.get("ledger_symptoms") or ())]
        srcs = c.get("ledger_sources") or [t for t, _ in entries]     # the phrase before the generation rewrite
        for line in ea["lines"]:
            comp = line["component"]
            terms = _ea1_terms(line, c, comps)
            kept = {s["text"] for s in line["symptoms"] if not s.get("dropped")}
            hit: dict = {}
            for t in sorted(kept):
                if t not in c["ledger_symptoms"]:
                    hit.setdefault(t, []).append("symptom not on the record")
            clue = (line.get("clue") or {}).get("text")
            for (text, ctx), s0 in zip(entries, srcs):
                if s0 == clue:          # the line's own clue distinguishes it: it is meant to show a specific sign
                    continue
                # Clauses are judged on the phrase the patient's sentence says (`symptom_lay.yaml`
                # re-words it with no added detail); a specific sign is looked for in both.
                mine = s0 in kept or core.symptom_owner(s0, comps) == comp
                lic = core.license_atoms(s0, terms) + core.license_atoms(ctx or "", terms)
                if mine:
                    n_sym += 1
                    hit.setdefault(text, []).extend(f"unlicensed clause {a!r}" for a, k, _ in lic
                                                    if k == "unlicensed")
                said = [a for a, k, _ in lic if k != "licensed"] + ([text] if text != s0 else [])
                ok = [a for a, k, _ in lic if k == "licensed"]
                for sg in signs.get(name.get(comp, ""), ()):
                    g = core._norm(sg)
                    if g and not any(g in core._norm(a) for a in ok):
                        for a in said:
                            if g in core._norm(a):
                                hit.setdefault(text, []).append(f"specific sign {sg!r} in {a!r}")
            for text, why in hit.items():
                why = list(dict.fromkeys(why))
                if why:
                    bad.append((c["cid"], comp, text, "; ".join(why), "hard" if is_hard else "other"))
                    if is_hard:
                        hard_cases_bad.add(c["cid"])
    hard_bad = [b for b in bad if b[-1] == "hard"]
    return {"cases_with_line": n_cases, "hard_cases_with_line": hard, "symptoms": n_sym,
            "violations": bad, "violations_hard": len(hard_bad),
            "hard_cases_with_violation": len(hard_cases_bad),
            "hard_cases_clean": hard - len(hard_cases_bad), "pass": not hard_bad}


def check_ea2(cases: list[dict]) -> dict:
    """EA-2 (refusing): every gold line of an F1-F3 case that has a registered clue
    (`explain_away.yaml:clues`) shows one finding that separates it from its nearest look-alike: its clue, or
    (a line that is not explained away) a specific sign, in a symptom the solver reads. Judged on the phrase
    said (`ledger_sources`), as EA-1 does. The clue shown must not be explainable by any other line of the case,
    a known condition on the record or another gold line (`explain_away.yaml:clue_explained_by`)."""
    from haenv.m2 import core
    signs, clues, expl = core.specific_signs(), core.clues(), core.clue_explained_by()
    name = {v: k for k, v in core.EA_HIDDEN.items()}
    bad, n = [], 0
    for c in cases:
        srcs = c.get("ledger_sources") or [t for t, _ in c.get("ledger_entries") or ()]
        ea = c.get("explain_away") or {}
        explained = {ln["component"]: (ln.get("clue") or {}).get("text") for ln in ea.get("lines") or []}
        for comp in set(c.get("lines") or ()) | set(explained):
            opts = clues.get(name.get(comp))
            if not opts or c.get("frame") not in ("F1", "F2", "F3"):
                continue
            n += 1
            shown = [t for t in opts if t in srcs]
            present = (set(c.get("known_visible") or ()) | set(c.get("lines") or ())) - {comp}
            if any(expl.get(t, frozenset()) & present for t in shown):
                bad.append((c["cid"], comp, "clue explainable by another line"))
                continue
            own = any(sg in t for t in srcs for sg in signs.get(name[comp], ()))
            if not (shown or (own and comp not in explained)):
                bad.append((c["cid"], comp))
    return {"lines": n, "violations": bad, "pass": not bad}


def drug_indications() -> dict[str, set[str]]:
    """Drug -> the indications the repo's tables give it: `registry/drug_indications.yaml` (approved or
    off-label pairs) and `med_course.CONCURRENT`."""
    import yaml

    from haenv import med_course
    out: dict[str, set[str]] = {}
    pairs = (yaml.safe_load((ROOT / "registry" / "drug_indications.yaml").read_text(encoding="utf-8")) or {}).get("pairs") or {}
    for ind, drugs in pairs.items():
        for d, row in drugs.items():
            if (row or {}).get("status") in ("approved", "off_label"):
                out.setdefault(d, set()).add(ind)
    for ind, opts in med_course.CONCURRENT.items():
        for d, _ in opts:
            out.setdefault(d, set()).add(ind)
    return out


def check_med1(cases: list[dict]) -> dict:
    """MED-1 (refusing): on F1-F3, no medicine the solver can read on the record is indicated for, or a
    marker of, a hidden line of the case (an indication of the drug that is one of the case's gold lines)."""
    ind_of = drug_indications()
    bad = []
    for c in cases:
        if c["frame"] == "F0":
            continue
        text = str(c.get("visible_med_text") or "").lower()
        hidden = set(c.get("lines") or ()) | {t["component"] for t in c.get("threads") or ()}
        for d, inds in sorted(ind_of.items()):
            hit = sorted(i for i in inds if BASE_EQUIV.get(i, set()) & hidden)
            if d.lower() in text and hit:
                bad.append((c["cid"], d, hit))
    return {"violations": bad, "pass": not bad}


#: Complaint category (v3 PREREG A2): the visit type the record opens with, one per frame.
COMPLAINT_CATEGORY = ("F0", "F1", "F2", "F3")


def p02a_matrix(cases: list[dict], *, acute_ids: bool = False) -> dict:
    """Prior-shortcut features (spec 7 P0-2a, v3 PREREG A2): age band, sex, the known conditions,
    their count and the complaint category (visit type). `y` = merged class, `lines` = the case's
    hidden lines (`["none"]` on an independent case), `frames` = the complaint category per row (the
    (b) gate's frame-conditioned baseline). `acute_ids` (the power table's real-size arm) adds the F1
    acute-event id -- the positive arm's event is the hidden line's own presentation (R-22)."""
    ages = sorted({c["age"] for c in cases})
    known = ["obesity", "dyslipidemia", "MASLD", "hypertension", "T2D", "hypothyroidism", "CAD", "CKD"]
    acute = sorted({c.get("acute") for c in cases if c.get("acute")})
    X, y, g, lines, frames = [], [], [], [], []
    for c in cases:
        if c["insufficient"]:
            continue
        frames.append(c.get("frame", "F0"))
        row = ([ages.index(c["age"]), 1.0 if c["sex"] == "F" else 0.0]
               + [1.0 if k in c["known"] else 0.0 for k in known] + [float(len(c["known"]))]
               + [1.0 if c.get("frame", "F0") == f else 0.0 for f in COMPLAINT_CATEGORY])
        if acute_ids:
            row += [1.0 if c.get("acute") == a else 0.0 for a in acute]
        X.append(row)
        y.append(c["join"])
        g.append(c["spec"])
        lines.append(sorted(c.get("lines") or []) or ["none"])
    return {"X": X, "y": y, "groups": g, "lines": lines, "frames": frames}


#: P0-2a (b) gate (m2-fix-r2 PREREG C): on the cases with a gold line, the learner's `dx_listed`-style
#: top-(n+1) score may exceed the SAME-FOLD frame-conditioned frequency list by at most this. The
#: comparison base is the frame list because the complaint category alone already moves the
#: top-(n+1) reading (B1); a global constant (the r1 gate: top-1 vs the fold's majority, `none`)
#: let any leak covering under ~37 % of the cases through. Readings of the power table, the
#: F1 acute-event-id arm and the label-permuted null are in the audit's P0-2a.power.
P02A_TOL = 0.05
P02A_POWER_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
P02A_POWER_SEEDS = 8
#: The label-permuted null may turn red on at most this share of its seeds, else calibration fails.
P02A_NULL_MAX_RED = 1 / 8
#: A leak share "turns the gate red" when at least this share of its seeds is red.
P02A_TURNS_RED = 0.5
P02A_POOL = 16


def _p02a_job(mat: dict, **extra) -> dict:
    return {"task": "guess_lines_topk", "X": mat["X"], "lines": mat["lines"], "groups": mat["groups"],
            "frames": mat.get("frames"), "seed": 20261001, **extra}


def p02a_power(mat: dict, *, fracs=P02A_POWER_FRACS, seeds: int = P02A_POWER_SEEDS, tol: float = P02A_TOL,
               null_seeds: int | None = None, acute_mat: dict | None = None) -> dict:
    """Power of the (b) gate on this matrix (PREREG C3): inject a line-naming feature (the case's own
    lines, multi-hot) into a `frac` of the cases with a gold line, `seeds` times, report the red rate
    per frac and the turns-red point (smallest frac with red rate >= P02A_TURNS_RED, None when no
    frac reaches it); the label-permuted null (`null_seeds`, default `seeds`) must stay at a red rate
    <= P02A_NULL_MAX_RED, else `calibration_holds` is False. `acute_mat` (the matrix with the F1
    acute-event-id feature) is the real-size arm, read once."""
    from concurrent.futures import ThreadPoolExecutor
    null_seeds = seeds if null_seeds is None else null_seeds
    jobs = [(f"leak_{f}", _p02a_job(mat, leak={"frac": f, "seed": s})) for f in fracs for s in range(seeds)]
    jobs += [("perm_null", _p02a_job(mat, perm_lines=s)) for s in range(null_seeds)]
    if acute_mat is not None:
        jobs.append(("acute_event_id", _p02a_job(acute_mat)))
    with ThreadPoolExecutor(P02A_POOL) as ex:
        out = list(ex.map(lambda j: sklearn_job(j[1]), jobs))
    table = {}
    for (arm, _), r in zip(jobs, out):
        table.setdefault(arm, []).append(r["edge"])
    table = {arm: {"red_rate": round(sum(e > tol + 1e-12 for e in v) / len(v), 3), "edge_mean": round(sum(v) / len(v), 3),
                   "edge_min": round(min(v), 3), "edge_max": round(max(v), 3), "runs": len(v)} for arm, v in table.items()}
    turns = [f for f in fracs if table[f"leak_{f}"]["red_rate"] >= P02A_TURNS_RED]
    res = {"tolerance": tol, "seeds": seeds, "null_seeds": null_seeds, "table": table,
           "turns_red_point": min(turns) if turns else None, "turns_red_rule": f"red_rate >= {P02A_TURNS_RED}",
           "null_red_rate": table["perm_null"]["red_rate"] if null_seeds else None,
           "null_max_red": P02A_NULL_MAX_RED}
    res["calibration_holds"] = (res["null_red_rate"] is not None and res["null_red_rate"] <= P02A_NULL_MAX_RED + 1e-12)
    if not res["calibration_holds"] and null_seeds:
        res["calibration"] = "the gate's calibration fails: the label-permuted null turns red above 1/8"
    if acute_mat is not None:
        a = next(r for (arm, _), r in zip(jobs, out) if arm == "acute_event_id")
        res["acute_event_id"] = {**a, "red": a["edge"] > tol + 1e-12}
    return res


def check_p02a(mat: dict, *, calibrate: bool = False, tol: float = P02A_TOL, acute_mat: dict | None = None) -> dict:
    """The gate reads (b) only (revision R-32), now in the `dx_listed` reading (m2-fix-r2 PREREG C):
    `hidden_line` = the cases with a gold line, per case |gold lines in the learner's top-(n+1)| / n;
    the learner (better of GBM and a feature-key lookup) must not beat the SAME-FOLD frame list
    (training-fold lines of the case's frame by coverage) by more than `tol`. (a) the merged class and
    the r1 top-1 hidden-line guess are profile readings (`profile: true`), not gated. `calibrate` adds
    the power table (leak shares, label-permuted null, and the `acute_mat` arm when given)."""
    a = sklearn_job({"task": "guess", "X": mat["X"], "y": mat["y"], "groups": mat["groups"], "seed": 20261001})
    a["within_majority_plus_0.05"] = a["acc"] <= a["majority"] + 0.05
    a["profile"] = True
    res = {"merged_class": a, "acc": a["acc"], "majority": a["majority"]}
    if mat.get("lines"):
        b = sklearn_job(_p02a_job(mat))
        b["tolerance"] = tol
        b["gate"] = "learner_topk <= same_fold_frame_list_topk + tolerance (cases with a gold line, top-(n+1) / n)"
        b["pass"] = b["edge"] <= tol + 1e-12
        res["hidden_line"] = b
        t1 = sklearn_job({"task": "guess_lines", "X": mat["X"], "lines": mat["lines"], "groups": mat["groups"],
                          "seed": 20261001})
        t1["profile"] = True
        t1["reading"] = "r1 top-1 guess vs the same-fold majority constant (all cases, `none` included); not gated"
        res["profile_top1"] = t1
        if calibrate:
            res["power"] = p02a_power(mat, tol=tol, acute_mat=acute_mat)
    res["gate"] = "hidden_line"
    res["pass"] = bool((res.get("hidden_line") or {}).get("pass"))
    return res
#: SURF-1 (revision r3-E2): length / structure features of the visible symptoms on F1-F3.
SURF_FEATURES = ("chars_per_symptom", "n_symptoms", "clauses_per_symptom", "context_share", "total_chars")


def surface_features(entries) -> list[float]:
    from haenv.m2.core import symptom_atoms
    n = len(entries)
    chars = [len(t) for t, _ in entries]
    return [sum(chars) / n if n else 0.0, float(n),
            (sum(len(symptom_atoms(t)) for t, _ in entries) / n) if n else 0.0,
            (sum(1 for _, c in entries if c) / n) if n else 0.0, float(sum(chars))]


# ------------------------------------------------------------------ collection
def collect(batch: pathlib.Path, job_path: pathlib.Path = M2_JOB) -> list[dict]:
    import logging
    logging.disable(logging.WARNING)
    import haenv  # noqa: F401
    import yaml
    from haenv_kernel.build import build_instance
    from haenv.evaluate import gated_menu
    from haenv.gated import menu_findings
    from haenv.job import load_job
    from haenv.m2 import core
    from haenv.store import load_cases
    load_job(job_path, root=ROOT)
    built = load_cases(batch / "cases.jsonl")
    au = batch / "audit.jsonl"             # `symptom_source`: each ledger symptom's phrase before the generation rewrite
    srcmap = ({a["case_id"]: a.get("symptom_source") or {} for a in map(json.loads, au.read_text(encoding="utf-8").splitlines())}
              if au.exists() else {})
    job = yaml.safe_load(pathlib.Path(job_path).read_text(encoding="utf-8"))
    J = {c["case_id"]: c for c in job["cases"]}
    ea = yaml.safe_load((ROOT / "registry" / "explain_away.yaml").read_text(encoding="utf-8"))["rows"]
    fx = core._findings()
    out = []
    for cid, raw in sorted(built.items()):
        T = int(raw.prediction_context["prediction_time_T"])
        src = lambda e, m=srcmap.get(cid, {}): m.get(str(e.get("evidence_id")), str(e.get("symptom")))
        sp, vp = build_instance(raw, T)
        adj = vp.adjudication or {}
        ddx = adj.get("ddx") or {}
        menu = gated_menu(raw, T)
        reach = set()
        for m in menu:
            if m.get("is_test"):
                reach.update(core.m2_menu_findings(m["target"], fx) if core.is_m2_framed(raw)
                             else menu_findings(m["target"], fx))
        jr = J[cid]["raw"]
        from haenv.build import clinical_plan
        cp = clinical_plan(jr["disease"], jr["devices"], jr.get("comorbidities") or [], cid)
        out.append({
            "cid": cid, "insufficient": bool(ddx.get("insufficient")),
            "join": ddx.get("join_gold"), "spec": ddx.get("spec_id"),
            "warranted": bool(adj.get("clinician_action_warranted")),
            "threads": [{**t, "confirmatory_all": core.confirmatory_all(t["component"])}
                        for t in core.threads_of(ddx)], "reachable": reach, "vocab": set(fx),
            "ea_keys": sorted({k for r in ea if ddx.get("spec_id") in (r.get("specs") or ()) for k in r["key_signals"]}),
            "known": [jr["disease"]] + list(jr.get("comorbidities") or []),
            "age": jr.get("age_range"), "sex": jr.get("sex"),
            # the case's lab-stream baselines (SA-6 reads them beside the symptom surface: world-v2 PACK)
            "base": {s: v["base"] for s, v in cp.items() if isinstance(v, dict) and "base" in v},
            "frame": ((adj.get("m2_frame") or {}).get("frame")) or "F0",
            "lines": (sorted(core.components(str(ddx.get("spec_id") or "")))
                      if ddx.get("join_gold") in ("unified", "comorbidity") else []),
            "acute": (((J[cid]["latent"].get("m2") or {}).get("acute_event") or {}).get("id")),
            "explain_away": ((J[cid]["latent"].get("m2") or {}).get("explain_away")),
            "tier": ((J[cid]["latent"].get("m2") or {}).get("prior") or {}).get("tier"),
            "ledger_symptoms": {src(e) for e in (sp.evidence_ledger or [])
                                if e.get("source_type") == "patient_reported_symptom"},
            "ledger_sources": [src(e) for e in (sp.evidence_ledger or [])
                               if e.get("source_type") == "patient_reported_symptom"],
            "ledger_entries": [(str(e.get("symptom")), str(e.get("context") or "")) for e in (sp.evidence_ledger or [])
                               if e.get("source_type") == "patient_reported_symptom"],
            "known_visible": list((sp.user_profile or {}).get("known_conditions") or []),
            # What the solver can read a drug name in (EA-1, revision r2-E2): the frame scene and
            # the record's symptom texts and contexts.
            "visible_med_text": core.visible_medication_text(
                ((sp.prediction_context or {}).get("m2_frame") or {}).get("scene"),
                [(str(e.get("symptom")), str(e.get("context") or "")) for e in (sp.evidence_ledger or [])
                 if e.get("source_type") == "patient_reported_symptom"]),
            # A line's own symptoms: its templates (age-stripped too) and the texts the
            # explain-away record attributes to it (a rewritten symptom still belongs to its line).
            "line_symptoms_on_record": {comp: any(src(e) in
                                                  ({t for x in core.component_templates(comp)
                                                    for t in (x, core._AGE.sub("", x).lstrip(",, "))}
                                                   | {sy["text"] for ln in (((J[cid]["latent"].get("m2") or {})
                                                                            .get("explain_away") or {}).get("lines") or [])
                                                      if ln["component"] == comp for sy in ln["symptoms"]})
                                                  for e in (sp.evidence_ledger or []))
                                        for comp in core.components(str(ddx.get("spec_id") or ""))},
        })
    return out


def bridge(batch: pathlib.Path, ref: pathlib.Path, frames: dict) -> dict:
    def load(p):
        return {json.loads(x)["case_id"]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
    A, B = load(ref / "payloads.jsonl"), load(batch / "payloads.jsonl")
    CA, CB = load(ref / "cases.jsonl"), load(batch / "cases.jsonl")
    res = {}
    for cid, f in frames.items():
        if cid not in A:
            # a v3 variant the v1.0.1 pack does not carry (R-29): new question, nothing to compare;
            # an F0 case missing from the reference stays in `n` and fails the gate
            r = res.setdefault(f, {"n": 0, "payload_same": 0, "gold_same": 0, "world_same": 0})
            if f == "F0":
                r["n"] += 1
            else:
                r["not_in_v101"] = r.get("not_in_v101", 0) + 1
            continue
        sp_same = A[cid]["sp"] == B[cid]["sp"]
        # F0: the whole adjudication, nothing stripped (revision r1-2: F0 carries no M2 block).
        # F1-F3 report the v1.0.1 part only (their frame gold is new).
        adj_b = CB[cid]["case"]["adjudication"]
        if f != "F0":
            adj_b = {k: v for k, v in adj_b.items() if k not in ("revision", "m2_frame")}
        adj_same = CA[cid]["case"]["adjudication"] == adj_b
        world_same = CA[cid]["case"]["longitudinal_data"] == CB[cid]["case"]["longitudinal_data"]
        r = res.setdefault(f, {"n": 0, "payload_same": 0, "gold_same": 0, "world_same": 0})
        r["n"] += 1
        r["payload_same"] += sp_same
        r["gold_same"] += adj_same
        r["world_same"] += world_same
    f0 = res.get("F0", {})
    return {"by_frame": res, "pass": f0.get("n") == f0.get("payload_same") == f0.get("gold_same") == f0.get("world_same")}


