"""Judging-layer floor, geometry scope and alias checks.

Each check here is two-sided: one case that must be judged negative (a
deliberate defect) and one that must not be (so the check is not
constant-negative).

* no-information floor -- a constant stub earns no points on Tracks B/C/D;
  the composite is measured above the batch's best constant;
* `STUB_GEOMETRIES` on the headline board;
* `gate_verified` / `n_gate_unknown` in the board cell;
* noop reading by answer channel and the batch-counted polarity;
* disputed-gold consumer;
* alias supplements (12 clinical synonyms).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys
from itertools import combinations
from types import SimpleNamespace

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

import verifier as K                                          # noqa: E402  kernel
from haenv import analytics as A                              # noqa: E402


# ------------------------------------------------------------ helpers
def _vp(y: int, gold: list[str], warranted: bool = True):
    return SimpleNamespace(outcome_label="event_occurred" if y else "no_event",
                           gold_drivers=list(gold),
                           adjudication={"clinician_action_warranted": warranted})


def _out(risk: float, cat: str, drivers: list[str], cls: str = "A3", wntd: bool = True):
    return SimpleNamespace(forecast={"risk": risk, "risk_category": cat},
                           drivers=[{"driver": d, "causal_status": "association_only"}
                                    for d in drivers],
                           action={"selected_action_class": cls,
                                   "what_not_to_do": ["x"] if wntd else []})


CASES = [  # (case, y, gold drivers)
    ("C1", 1, ["poor_medication_adherence"]),
    ("C2", 1, ["biological_low_response"]),
    ("C3", 0, ["unknown_or_multifactorial"]),
    ("C4", 1, ["poor_medication_adherence"]),
    ("C5", 0, ["poor_medication_adherence"]),
]


def _row(solver: str, case: str, y: int, gold: list[str], out, geometry: str = "single") -> dict:
    vp = _vp(y, gold)
    cat = out.forecast["risk_category"]
    pos = cat in ("elevated", "high")
    return {"solver": solver, "case": case, "y": float(y), "gold_drivers": gold,
            "geometry": geometry, "gates": [], "overall": "SCORED", "abstained": False,
            "direction_ok": (pos == bool(y)), "brier": (out.forecast["risk"] - y) ** 2,
            "driver_hit": bool(out.drivers) and out.drivers[0]["driver"] in gold,
            "tracks": {"A": 1.0, "B": K._track_B(out, vp), "C": K._track_C(out, vp),
                       "D": K._track_D(out, vp), "E": None}}


def _batch(extra_geometry: str = "single") -> list[dict]:
    rows = []
    for case, y, gold in CASES:
        # constant: the batch's best constant category and driver, never reads the case
        rows.append(_row("const_ddx", case, y, gold,
                         _out(0.6, "high", ["poor_medication_adherence"])))
        # informed: answers every case correctly
        rows.append(_row("model_good", case, y, gold,
                         _out(0.9 if y else 0.1, "high" if y else "low", gold)))
        # flip_flop under single geometry = constant "low"
        rows.append(_row("flip_flop", case, y, gold,
                         _out(0.15, "low", ["unknown_or_multifactorial"], cls="A0"),
                         geometry=extra_geometry))
    return rows


# ------------------------------------------------------------ (a) floors
def test_b_and_c_floors_match_kernel_brute_force():
    """The analytic floors re-implement `_track_B` / `_track_C`; this pins them
    to the kernel functions by brute force over the same policy space."""
    ys = [y for _, y, _ in CASES]
    got_b = A._b_floor([float(y) for y in ys])[0]
    best_b = max(sum(K._track_B(_out(k / 100, cat, []), _vp(y, [])) for y in ys) / len(ys)
                 for k in range(101) for cat in ("high", "low"))
    assert abs(got_b - best_b) < 2e-3, (got_b, best_b)
    golds = [g for _, _, g in CASES]
    vocab = sorted({d for g in golds for d in g})
    best_c = max(sum(K._track_C(_out(0.5, "low", list(s)), _vp(0, g)) for g in golds) / len(golds)
                 for s in [(d,) for d in vocab] + list(combinations(vocab, 2)))
    assert abs(A._c_floor(golds)[0] - best_c) < 1e-9
    # Track D: kernel gives the constant referral policy the maximum on every case
    assert all(K._track_D(_out(0.5, "high", [], cls="A3"), _vp(y, [], w)) == 1.0
               for y in (0, 1) for w in (True, False))


def test_constant_stub_scores_zero_informed_solver_does_not():
    rank = {r["model"]: r for r in A.rank_models(_batch(), multiround=False,
                                                  task_type="early_warning")}
    c, g = rank["const_ddx"], rank["model_good"]
    # negative control: under the uncorrected composite the constant earns points
    assert c["score_uncorrected"] is not None and c["score_uncorrected"] > 0.3, c
    assert c["score"] == 0.0, (c["score"], c["core_skill"])
    # not constant-negative: a solver that reads the case keeps a high score
    assert g["score"] is not None and g["score"] > 0.8, (g["score"], g["core_skill"])
    assert c["core_kind"]["trackD"] == "deduct"


def test_skill_over_floor_three_regimes():
    assert A.skill_over_floor(0.75, 0.5) == ("skill", 0.5)
    assert A.skill_over_floor(0.4, 0.5) == ("skill", 0.0)          # below floor clips
    assert A.skill_over_floor(0.8, 1.0) == ("deduct", 0.8)         # no headroom -> multiplier
    assert A.skill_over_floor(0.3, None) == ("raw", 0.3)           # unmeasured -> raw, flagged


# ------------------------------------------------------------ geometry
def test_out_of_geometry_stub_is_not_ranked():
    ranking = A.rank_models(_batch("single"), multiround=False, task_type="early_warning")
    ff = next(r for r in ranking if r["model"] == "flip_flop")
    assert ff["geometry_out_of_scope"] and ff["geometry_off"] == ["single"]
    labels = dict(zip([r["model"] for r in ranking], A.board_rank_labels(ranking)))
    assert labels["flip_flop"] == "out-of-scope"
    assert ranking[-1]["model"] == "flip_flop"
    # the other side: under its declared geometry it is ranked normally
    ranking2 = A.rank_models(_batch("slices"), multiround=False, task_type="early_warning")
    ff2 = next(r for r in ranking2 if r["model"] == "flip_flop")
    assert not ff2["geometry_out_of_scope"]
    assert "out-of-scope" not in A.board_rank_labels(ranking2)


# ------------------------------------------------------------ (b) gate cell
def test_gate_cell_distinguishes_never_ran_from_zero_hits():
    judged = [dict(r) for r in _batch() if r["solver"] == "model_good"]
    never = [{k: v for k, v in r.items() if k not in ("gates", "overall")} for r in judged]
    r_ok = next(x for x in A.rank_models(judged, False, "early_warning"))
    r_nv = next(x for x in A.rank_models(never, False, "early_warning"))
    # negative control: the plain cell text for the never-judged rows is "0/5" -- same as clean
    assert f"{r_nv['gate_fail']}/{r_nv['n']}" == f"{r_ok['gate_fail']}/{r_ok['n']}" == "0/5"
    assert "gate not run" in A.gate_cell(r_nv) and A.gate_state_note([r_nv])
    assert A.gate_cell(r_ok) == "0/5" and A.gate_state_note([r_ok]) == []
    partial = judged[:3] + never[3:]
    r_pt = A.rank_models(partial, False, "early_warning")[0]
    assert "unverified 2" in A.gate_cell(r_pt) and A.gate_cell(r_pt).startswith("0/3")


# ------------------------------------------------------------ (c) noop reading
def _noop_rows(channel: str, declared_gap: bool = False) -> list[dict]:
    rows = []
    for i in range(3):
        rows.append({"noop_polarity": "gap", "noop_declared": declared_gap,
                     "noop_answer_channel": channel, "noop_ok": float(declared_gap)})
    rows.append({"noop_polarity": "covered", "noop_declared": False,
                 "noop_answer_channel": channel, "noop_ok": 1.0})
    return rows


def test_noop_reading_separates_global_flag_from_fabrication():
    assert A.noop_reading(_noop_rows("global_only"))["kind"] == "global_flag"
    assert A.noop_reading(_noop_rows("none"))["kind"] == "no_answer"
    # not constant-negative: writing into the blank window without a negation is fabrication
    assert A.noop_reading(_noop_rows("signal_quality"))["kind"] == "fabricated"
    assert A.noop_reading(_noop_rows("signal_quality", declared_gap=True))["kind"] == "checking"
    # rows without the channel field fall back to the flag, never to "fabricated"
    old = [{k: v for k, v in r.items() if k != "noop_answer_channel"} | {"noop_used_global_flag": 1.0}
           for r in _noop_rows("x")]
    assert A.noop_reading(old)["kind"] == "global_flag"


def test_polarity_prose_is_counted_not_hard_coded():
    src = (ROOT / "haenv" / "report.py").read_text(encoding="utf-8")
    assert "极性 3 不存在 : 2 存在" not in src and "极性 3 假 : 2 真" not in src
    # and the prose does not name the global flag as something the judge reads
    assert "判据只读结构化枚举(`data_quality.data_sufficiency` / `signal_quality`)" not in src


# ------------------------------------------------------------ (d) disputed gold
def test_disputed_gold_register_loads_and_matches_cases():
    from haenv.registry import load_disputed_gold
    ids = {e["id"]: e["status"] for e in load_disputed_gold()}
    assert ids.get("DG-001") == "open"
    ckm = SimpleNamespace(adjudication={"ddx": {"spec_id": "JD-CKM"}})
    other = SimpleNamespace(adjudication={"ddx": {"spec_id": "JD-PCOS"}})
    hit = A.disputed_gold_hits({"JD-16": ckm, "JD-01": other})
    assert hit["ok"] and [h["cases"] for h in hit["hits"] if h["id"] == "DG-001"] == [["JD-16"]]
    miss = A.disputed_gold_hits({"JD-01": other})
    assert miss["ok"] and all(not h["cases"] for h in miss["hits"])


# ------------------------------------------------------------ (d) alias supplements
AUDIT_12 = {
    "JD-B12": ["亚急性联合变性", "恶性贫血"],
    "JD-GRAVES": ["弥漫性毒性甲状腺肿", "thyrotoxicosis", "突眼性甲状腺肿"],
    "JD-PHEO": ["paraganglioma", "肾上腺髓质肿瘤", "儿茶酚胺分泌性肿瘤"],
    "JD-OSA": ["OSAHS", "sleep apnoea"],
    "JD-CUSH": ["高皮质醇血症", "肾上腺皮质功能亢进"],
}


def test_alias_supplements_cover_the_audited_twelve():
    import joint_scenarios as JS
    from haenv.events import alias_hit
    from haenv.overlay import apply_alias_supplements
    sup = apply_alias_supplements(JS.DDX_SPECS)
    assert sum(len(v) for v in AUDIT_12.values()) == 12
    for sid, terms in AUDIT_12.items():
        before = [JS.DDX_SPECS[sid]["diagnosis"]] + list(JS.DDX_SPECS[sid]["aliases"])
        leak = sup[sid]["leak_aliases"] + [sup[sid]["diagnosis"]]
        for t in terms:
            assert not alias_hit(t, before), f"{sid}:{t} already hit before -- not a miss"
            assert alias_hit(t, leak), f"{sid}:{t} still missed by the leak-side list"


def test_alias_supplements_do_not_hit_unrelated_words():
    import joint_scenarios as JS
    from haenv.events import alias_hit
    from haenv.overlay import apply_alias_supplements
    from haenv.overlay import rivals_for
    sup = apply_alias_supplements(JS.DDX_SPECS)
    for sid in AUDIT_12:
        added = [a for a in sup[sid]["aliases"] if a not in JS.DDX_SPECS[sid]["aliases"]]
        for r in rivals_for(sid, sup[sid]):
            blob = " ".join([r["name"]] + list(r.get("aliases") or ()))
            assert not alias_hit(blob, added), (sid, r["name"])
    # umbrella / cause words stay out of the answer key
    for sid, t in (("JD-GRAVES", "thyrotoxicosis"), ("JD-B12", "恶性贫血"),
                   ("JD-B12", "亚急性联合变性"), ("JD-PHEO", "paraganglioma"),
                   ("JD-OSA", "sleep apnoea"), ("JD-OSA", "sleep apnea"),
                   ("JD-PHEO", "肾上腺髓质肿瘤"), ("JD-CUSH", "肾上腺皮质功能亢进")):
        assert not alias_hit(t, sup[sid]["aliases"]), (sid, t)
    # word boundary: JD-SLE's `sle` still does not fire on the new OSA synonyms
    assert not alias_hit("sleep apnea", JS.DDX_SPECS["JD-SLE"]["aliases"])
