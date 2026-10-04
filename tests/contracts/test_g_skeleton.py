"""G. The weight skeleton's irregularities keep the gold and say nothing about it.

`build._weight_series` adds an uneven descent, plateau episodes, a per-case rebound
shape and transient ups and downs. These contracts check the four promises that make that safe:

  g1  the skeleton up to T is the same in every outcome arm, given the same pre-T
      geometry (start, nadir, T, descent end);
  g2  pre-T skeleton features have one distribution across outcome arms;
  g3  the label rule reads the declared outcome or the plain skeleton's verdict;
  g4  the daily skeleton step stays within `max_weekly_delta`;
  g5  the LLM path: exact at the model's points, pre-T the same in every arm, rule kept;
  g6  no guard fallback changes a pre-T value.

Each has a negative control that breaks the promise on purpose.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import copy
import math
import random
import statistics
import types

import pytest

from haenv import build as B
from haenv import build_weight as BW   # where the skeleton helpers are read
from _patch_bound import patch_bound  # noqa: E402
from haenv import gates
from haenv import rng
from haenv.latent_rules import (DESCENT_BUDGET_FRAC, MIN_CHANGE_FRAC, MIN_CHANGE_KG,
                                MIN_PERSIST_DAYS)

pytestmark = pytest.mark.world

N_CASES = 60
RULE = {"min_change_frac": MIN_CHANGE_FRAC, "min_change_kg": MIN_CHANGE_KG,
        "min_persist_days": MIN_PERSIST_DAYS}
BUDGET_DAY = 1.5 / 7.0 * DESCENT_BUDGET_FRAC


def _geometry(i: int) -> dict:
    """Pre-T geometry drawn from the case id only."""
    cid = f"SK-{i:03d}"
    u = lambda k: rng.unit(cid, "pop", k)                      # noqa: E731
    T = 56 + 7 * int(40 * u(0))
    return {"case_id": cid, "start": round(60 + 50 * u(1), 1),
            "drop": 0.0 if u(2) < 0.35 else round(1 + 11 * u(3), 1),
            "T": T, "end_day": max(365, T + 168)}


def _arms(g: dict) -> list[tuple[str, int, float]]:
    """`(outcome, rev_week, slope)` per arm; every regain reversal is at or after T,
    so all arms end the descent on the same day."""
    lo, hi = -(-g["T"] // 7), (g["end_day"] - 63) // 7
    u = lambda k: rng.unit(g["case_id"], "arm", k)             # noqa: E731
    return [("maintain", lo, 0.35),
            ("regain", lo + int((hi - lo) * u(0)), 0.1 + 0.4 * u(1)),
            ("regain", lo + int((hi - lo) * u(2)), 0.1 + 0.4 * u(3))]


def _render(g: dict, outcome: str, rev_week: int, slope: float):
    return B._weight_render(g["start"], g["start"] - g["drop"], g["T"], outcome, rev_week,
                            slope, case_id=g["case_id"], step=1, end_day=g["end_day"])


@pytest.fixture(scope="module")
def normal_render():
    """One unpatched render of every geometry/arm, reused by the positive checks only.

    Mutation tests keep calling `_render` live. Copies keep one consumer from changing
    another consumer's reference, without memoizing any patched production function.
    """
    rendered = {}
    for i in range(N_CASES):
        g = _geometry(i)
        for arm in _arms(g):
            rendered[(tuple(sorted(g.items())), *arm)] = _render(g, *arm)

    def read(g, *arm):
        return copy.deepcopy(rendered[(tuple(sorted(g.items())), *arm)])

    return read


def _pre_t_mismatches(render=_render) -> list[str]:
    bad = []
    for i in range(N_CASES):
        g = _geometry(i)
        pre = [render(g, *arm)[1]["base"][:g["T"] + 1] for arm in _arms(g)]
        if any(p != pre[0] for p in pre[1:]):
            bad.append(g["case_id"])
    return bad


def test_g1_pre_t_skeleton_is_the_same_in_every_arm(normal_render):
    """Same case id and pre-T geometry, different outcome / reversal week / slope:
    the skeleton up to T is identical. Turns red when anything after T, or the
    outcome itself, reaches a pre-T value."""
    assert _pre_t_mismatches(normal_render) == []


def test_g1_negative_control_outcome_bump(monkeypatch):
    """A skeleton that adds 0.3 kg on day 10 in regain cases only must be caught."""
    real = B._skeleton_base

    def leaky(*a, **k):
        base, meta = real(*a, **k)
        if a[3] is not None:                       # rev_day: regain only
            base = list(base)
            base[10] += 0.3
            meta = {**meta, "base": base}
        return base, meta

    monkeypatch.setattr(BW, "_skeleton_base", leaky)
    assert len(_pre_t_mismatches()) == N_CASES


def _features(g: dict, outcome: str, rev_week: int, slope: float, render=_render) -> dict:
    """Pre-T skeleton features: how far the skeleton leaves the plain one, and how."""
    _, meta = render(g, outcome, rev_week, slope)
    T = g["T"]
    plain = meta["plain"]
    base = meta["base"]
    dev = [base[d] - plain[d] for d in range(T + 1)]
    return {"max_dev": max(abs(x) for x in dev), "sd_dev": statistics.pstdev(dev),
            "n_episodes": sum(1 for e in meta["episodes"] if e["t1"] < T),
            "n_stall": sum(1 for lo, hi in meta["stalls"] if hi <= T)}


def _perm_p(a: list[float], b: list[float], n_perm: int = 2000) -> float:
    """Two-sided permutation p for a difference in means; a fixed seed keeps it
    deterministic."""
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    pool, na = a + b, len(a)
    r = random.Random(20260925)
    hits = 0
    for _ in range(n_perm):
        r.shuffle(pool)
        if abs(statistics.fmean(pool[:na]) - statistics.fmean(pool[na:])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def test_g2_pre_t_features_have_one_distribution_across_arms(normal_render):
    """Each case goes to one arm by its own hash; geometry never reads the arm.
    Pre-T features must not separate the arms (permutation p >= 0.01 each)."""
    rows = {"maintain": [], "regain": []}
    for i in range(N_CASES):
        g = _geometry(i)
        arm = _arms(g)[0 if rng.unit(g["case_id"], "which_arm") < 0.5 else 1]
        rows[arm[0]].append(_features(g, *arm, render=normal_render))
    assert rows["maintain"] and rows["regain"]
    for f in ("max_dev", "sd_dev", "n_episodes", "n_stall"):
        p = _perm_p([r[f] for r in rows["maintain"]], [r[f] for r in rows["regain"]])
        assert p >= 0.01, f"{f}: permutation p={p:.4f} separates the outcome arms"


def _declared(outcome: str) -> str:
    return "event_occurred" if outcome == "regain" else "event_not_occurred"


def _label_breaks(render=_render) -> list[str]:
    bad = []
    for i in range(N_CASES):
        g = _geometry(i)
        for arm in _arms(g)[:2]:
            pts, _ = render(g, *arm)
            got = gates.derive_outcome({"weight": pts}, RULE)[0]
            # A verdict other than the declared one is allowed only where the plain
            # skeleton reads it too.
            if got != _declared(arm[0]) and got != _plain_verdict(g, *arm, render=render):
                bad.append(f"{g['case_id']}:{arm[0]}")
    return bad


def _plain_verdict(g: dict, outcome: str, rev_week: int, slope: float, render=_render) -> str:
    plain = render(g, outcome, rev_week, slope)[1]["plain"]
    return gates.derive_outcome({"weight": [{"ts": d, "value": v} for d, v in enumerate(plain)]},
                                RULE)[0]


def test_g3_label_rule_reads_the_declared_outcome(normal_render):
    """Every rendered case reads its declared outcome under `label_rule`, or the same
    verdict as the plain skeleton."""
    assert _label_breaks(normal_render) == []


def test_g3_negative_control_long_episodes_break_the_rule(monkeypatch):
    """Two-sided: 120-day, 2 kg hills must break the rule somewhere once the guard is
    switched off (the check has an object); with the guard on they must not."""
    monkeypatch.setattr(BW, "_episode_plan", lambda ep, bd, *a: {
        "kind": "up", "height": 2.0, "rise": 60, "n": 120, "residual": 0.5, "tau": 200.0,
        "valley": 10})
    assert _label_breaks() == []
    # A guard that reads one constant verdict for every series accepts everything;
    # no-regain cases also lose the stop at the low point.
    monkeypatch.setattr(BW, "gates", types.SimpleNamespace(
        derive_outcome=lambda ld, rule: ("event_occurred", {})))
    monkeypatch.setattr(BW, "_no_regain_until", lambda plain, outcome: None)
    assert _label_breaks() != []


def _step_breaks(render=_render) -> list[str]:
    bad = []
    for i in range(N_CASES):
        g = _geometry(i)
        for arm in _arms(g):
            _, meta = render(g, *arm)
            base = meta["base"]
            plain = meta["plain"]
            if any(abs(base[d + 1] - base[d]) > max(BUDGET_DAY, abs(plain[d + 1] - plain[d])) + 1e-6
                   for d in range(len(base) - 1)):
                bad.append(f"{g['case_id']}:{arm[0]}")
    return bad


def _plain_args(g: dict, outcome: str, rev_week: int, slope: float) -> tuple:
    rev_day = rev_week * 7 if outcome == "regain" else None
    desc_end = max(7, min(g["T"], rev_day)) if rev_day is not None else g["T"]
    desc_end = max(desc_end, math.ceil(g["drop"] / BUDGET_DAY))
    k_want, k_reb_want = B._trajectory_shape(g["case_id"])
    k_desc = B._feasible_k(g["drop"] / max(1, desc_end), BUDGET_DAY, k_want)
    k_reb = B._feasible_k(abs(slope) / 7.0, BUDGET_DAY, k_reb_want)
    return (g["start"], g["start"] - g["drop"], desc_end, rev_day, slope, g["end_day"],
            k_desc, k_reb, BUDGET_DAY, B._skeleton_draws(g["case_id"]))


def test_g4_daily_step_stays_within_the_descent_budget(normal_render):
    """On every day the skeleton's step stays within `max(descent budget, the plain
    skeleton's step that day)`: drift, episodes, re-pacing and the regain check together."""
    assert _step_breaks(normal_render) == []


def test_g4_negative_control_unscaled_episodes(monkeypatch):
    """Episodes applied at full height, with no step caps and no guard step check, must
    break the limit."""
    monkeypatch.setattr(BW, "_step_caps", lambda ref, bd: [1e9] * (len(ref) - 1))
    monkeypatch.setattr(BW, "_steps_within", lambda *a, **k: True)
    monkeypatch.setattr(BW, "_episode_plan", lambda ep, bd, *a: {
        "kind": "up", "height": 1.5, "rise": 2, "n": 20, "residual": 0.2, "tau": 60.0,
        "valley": 10})
    assert _step_breaks() != []


# ------------------------------------------------------------------ LLM path: `_weight_overlay`
def _llm_case(i: int, outcome: str):
    """Weekly model anchors: a descent to T, then flat (no regain) or rising from T+28."""
    g = _geometry(i)
    T, end = g["T"], g["end_day"]
    u = lambda k: rng.unit(g["case_id"], "llm", k)            # noqa: E731
    anchors, v = [], g["start"]
    for d in range(0, end + 1, 7):
        if d <= T:
            v = g["start"] - g["drop"] * d / T + 0.3 * (u(d) - 0.5)
        elif outcome == "regain" and d > T + 28:
            v += 0.35
        anchors.append({"ts": d, "value": round(v, 2)})
    lin = B._resample_to_grid([{"ts": d, "value": 0.0} for d in range(end + 1)], anchors)
    return g, anchors, lin


def test_g5_llm_curve_is_exact_at_the_model_anchors():
    """The shape-preserving curve passes through every model point."""
    for i in range(N_CASES):
        _, anchors, _ = _llm_case(i, "maintain")
        src = [(q["ts"], q["value"]) for q in anchors]
        daily = B._pchip_daily(src, src[-1][0], BUDGET_DAY)
        assert all(abs(daily[x] - y) < 1e-9 for x, y in src)


def test_g5_llm_pre_t_is_the_same_in_every_arm():
    """Same pre-T anchors, different post-T course and outcome: identical readings up to T."""
    bad = []
    for i in range(N_CASES):
        pre = []
        for outcome in ("maintain", "regain"):
            g, anchors, lin = _llm_case(i, outcome)
            out, _ = B._weight_overlay(lin, anchors, g["case_id"], g["T"], outcome,
                                       g["start"] - g["drop"])
            pre.append([q["value"] for q in out if q["ts"] <= g["T"]])
        if pre[0] != pre[1]:
            bad.append(i)
    assert bad == []


def test_g5_negative_control_pre_t_choice_reads_the_whole_course(monkeypatch):
    """A guard whose pre-T check reads the whole course makes pre-T values follow the
    outcome somewhere; the arm comparison must see it."""
    real = B._label_guard

    def leaky(plain, T, outcome, nadir=None):
        ok = real(plain, T, outcome, nadir)
        return lambda series, upto=None: ok(series) if upto is not None else ok(series)

    monkeypatch.setattr(BW, "_label_guard", leaky)
    bad = []
    for i in range(N_CASES):
        pre = []
        for outcome in ("maintain", "regain"):
            g, anchors, lin = _llm_case(i, outcome)
            out, _ = B._weight_overlay(lin, anchors, g["case_id"], g["T"], outcome,
                                       g["start"] - g["drop"])
            pre.append([q["value"] for q in out if q["ts"] <= g["T"]])
        if pre[0] != pre[1]:
            bad.append(i)
    assert bad != []


def test_g5_llm_label_rule_reads_the_declared_outcome_or_the_linear_verdict():
    bad = []
    for i in range(N_CASES):
        for outcome in ("maintain", "regain"):
            g, anchors, lin = _llm_case(i, outcome)
            out, _ = B._weight_overlay(lin, anchors, g["case_id"], g["T"], outcome,
                                       g["start"] - g["drop"], rev_week=(g["T"] + 28) // 7,
                                       declared_gain=0.35 * (g["end_day"] - g["T"] - 28) / 7,
                                       declared_lost=g["drop"])
            got = gates.derive_outcome({"weight": out}, RULE)[0]
            ref = gates.derive_outcome({"weight": lin}, RULE)[0]
            if got not in (_declared(outcome), ref):
                bad.append(f"{i}:{outcome}")
    assert bad == []


# ------------------------------------------------------------------ fallbacks never reach pre-T
def _forced_failure(monkeypatch):
    """A label guard that rejects every whole-course series: every fallback fires."""
    real = B._label_guard

    def strict(plain, T, outcome, nadir=None):
        ok = real(plain, T, outcome, nadir)
        return lambda series, upto=None: ok(series, upto) if upto is not None else False

    monkeypatch.setattr(BW, "_label_guard", strict)


def _pre_t_of_both_paths(render=_render) -> dict:
    out = {}
    for i in range(N_CASES):
        g = _geometry(i)
        for arm in _arms(g)[:2]:
            out[(i, "det", arm[0])] = render(g, *arm)[1]["base"][:g["T"] + 1]
        for outcome in ("maintain", "regain"):
            g2, anchors, lin = _llm_case(i, outcome)
            pts, _ = B._weight_overlay(lin, anchors, g2["case_id"], g2["T"], outcome,
                                       g2["start"] - g2["drop"])
            out[(i, "llm", outcome)] = [q["value"] for q in pts if q["ts"] <= g2["T"]]
    return out


@pytest.fixture(scope="module")
def normal_pre_t(normal_render) -> dict:
    """`_pre_t_of_both_paths` without patches, computed once for both g6 tests."""
    return _pre_t_of_both_paths(normal_render)


def test_g6_fallbacks_never_change_a_pre_t_value(monkeypatch, normal_pre_t):
    """Pre-T values are a function of the case's draws and pre-T data alone: with every
    whole-course check failing (all fallbacks fire), they are the same as normally."""
    normal = normal_pre_t
    _forced_failure(monkeypatch)
    forced = _pre_t_of_both_paths()
    assert [k for k in normal if normal[k] != forced[k]] == []


def test_g6_negative_control_fallback_to_the_plain_course(monkeypatch, normal_pre_t):
    """A last fallback that restores the whole plain course, pre-T included, must be seen."""
    normal = normal_pre_t
    _forced_failure(monkeypatch)
    monkeypatch.setattr(BW, "_post_t_plain", lambda base, plain, T, bd: list(plain))
    forced = _pre_t_of_both_paths()
    assert [k for k in normal if k[1] == "det" and normal[k] != forced[k]] != []


def _t7_case(i: int, outcome: str, off: int = 0):
    """Weekly model points; T sits `off` days after a point (0: T itself is a point). A
    regain starts at the first point after T, so that point already depends on the
    outcome."""
    g = _geometry(i)
    T0 = g["T"] - g["T"] % 7
    end = g["end_day"]
    u = lambda k: rng.unit(g["case_id"], "t7", k)             # noqa: E731
    anchors = []
    for d in range(0, end + 1, 7):
        if d <= T0:
            v = g["start"] - g["drop"] * d / T0 + 0.3 * (u(d) - 0.5)
        else:
            v = anchors[-1]["value"] + (0.6 if outcome == "regain" else -0.05)
        anchors.append({"ts": d, "value": round(v, 2)})
    lin = B._resample_to_grid([{"ts": d, "value": 0.0} for d in range(end + 1)], anchors)
    return {**g, "T": T0 + off}, anchors, lin


def _t7_pre_t_mismatches(off: int = 0) -> list[int]:
    bad = []
    for i in range(N_CASES):
        pre = []
        for outcome in ("maintain", "regain"):
            g, anchors, lin = _t7_case(i, outcome, off)
            out, _ = B._weight_overlay(lin, anchors, g["case_id"], g["T"], outcome,
                                       g["start"] - g["drop"], rev_week=g["T"] // 7 + 1,
                                       declared_gain=2.0, declared_lost=g["drop"])
            pre.append([q["value"] for q in out if q["ts"] <= g["T"]])
        if pre[0] != pre[1]:
            bad.append(i)
    return bad


@pytest.mark.parametrize("off", [0, 3, 6])
def test_g5_model_points_after_t_do_not_shape_days_before_it(off):
    """The model points after T depend on the outcome; the readings up to T stay the
    same, whether T is a model point (`off` 0) or falls 3 or 6 days after one. Red when
    one curve is fitted across T, or when the stretch before T bridges linearly to the
    first point after it."""
    assert _t7_pre_t_mismatches(off) == []


def test_g5_negative_control_one_curve_across_t(monkeypatch):
    """A single shape-preserving curve through all model points lets the points after T
    bend the last days before it."""
    real_pchip, real_overlay = B._pchip_daily, B._weight_overlay
    seen = {}

    def across(src, end, cap):
        return real_pchip(seen["all"], end, cap)

    def overlay(lin_pts, model_pts, *a, **k):
        seen["all"] = sorted({int(q["ts"]): float(q["value"]) for q in model_pts}.items())
        return real_overlay(lin_pts, model_pts, *a, **k)

    monkeypatch.setattr(BW, "_pchip_daily", across)
    patch_bound(monkeypatch, "_weight_overlay", overlay, source="haenv.build")
    assert _t7_pre_t_mismatches() != []
