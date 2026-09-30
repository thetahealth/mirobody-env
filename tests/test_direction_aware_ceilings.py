"""The degenerate ceiling must take "best" **according to each dimension's
direction**, and the leaderboard's hard-gate column must use the same quantity
as the multiplier.

## One quantity, one way of computing it

**`degenerate()` does not always use `max`.** On a `lower` dimension (lower is
better), "the best degenerate strategy" is the **best** stub under that
direction; taking the worst would inflate the ceiling and reverse the sign of
`margin = br - bd`, so a genuine capability dimension would be judged
`not_a_capability_dim`. On a `non_monotone` dimension, "exceeding" does not
exist as a concept, so a margin computed with max is meaningless. The scoring
dimensions include both kinds (`tool_dup_rate` is `lower`, `tool_budget_used`
is `non_monotone`).

**`rank_ddx`'s hard-gate column** counts gated **units**, not rows whose
`overall` starts with `FAIL`: on sliced packs `overall` is always `SCORED` (a
per-slice gate hit does not propagate into `overall`), so a row-based count
would be 0 while `n_gated_units` on the same row is non-zero. `rank_models`
uses `len(gated_units)` as well.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from verifier_core.ceilings import degenerate                  # noqa: E402

#: Two degenerate stubs + one real solution. On the `lower` dimension the real
#: solution is **lower** (better).
_ROWS = [
    {"case": "C1", "solver": "stub_a", "d_low": 0.90, "d_hi": 0.10},
    {"case": "C1", "solver": "stub_b", "d_low": 0.20, "d_hi": 0.30},
    {"case": "C1", "solver": "real_m", "d_low": 0.05, "d_hi": 0.80},
]
_DEG = ("stub_a", "stub_b")


def _run(direction=None, noise=0.001):
    return degenerate(_ROWS, ("d_low", "d_hi"), noise,
                      getval=lambda r, d: r.get(d),
                      degenerate_solvers=_DEG, direction=direction)


def test_lower_dim_uses_min_not_max():
    """`lower` dimension: the ceiling takes the **lowest** stub (0.20), the margin
    sign flips, so the real solution's 0.05 wins."""
    out = _run({"d_low": "lower", "d_hi": "higher"})
    r = out["d_low"]
    assert r["degenerate_ceiling"] == 0.20, r
    assert r["best_real"] == 0.05, r
    assert r["margin"] == 0.15, r        # bd - br, not br - bd
    assert r["verdict"] == "ok", r
    assert r["direction"] == "lower"


def test_lower_dim_was_wrong_under_max():
    """Inverse case — the same data computed under the `higher` convention
    gets judged "cannot measure capability": a wrong direction flips the
    conclusion.
    """
    out = _run({"d_low": "higher"})
    assert out["d_low"]["verdict"] == "not_a_capability_dim", out["d_low"]
    assert out["d_low"]["degenerate_ceiling"] == 0.90, "按 max 取上界 ⇒ 虚高"


def test_non_monotone_is_named_not_folded():
    """`non_monotone` gets a named verdict — **not folded into ok, and not folded
    into not_a_capability_dim**.

    "This dimension has no direction" and "this dimension can't measure
    capability" are two different things; collapsing them together loses the
    reason.
    """
    out = _run({"d_low": "non_monotone", "d_hi": "higher"})
    assert out["d_low"]["verdict"] == "direction_non_monotone", out["d_low"]
    assert out["d_low"]["margin"] is None
    assert out["d_hi"]["verdict"] in ("ok", "unresolved", "not_a_capability_dim")


def test_default_is_higher_backward_compatible():
    """Inverse case: with no `direction` passed, behavior is byte-for-byte
    identical to before the change (otherwise this is an implicit, unannounced
    convention change)."""
    assert _run(None)["d_hi"] == _run({"d_hi": "higher"})["d_hi"]
    assert _run(None)["d_low"] == _run({"d_low": "higher"})["d_low"]


def test_production_profile_has_objects_for_this_fix():
    """This fix has a real target on disk — otherwise it's a case of "declared to
    exist, never triggered"."""
    from haenv.scoring import load_profile
    prof = load_profile(root=ROOT)
    dirs = {d: prof.direction(d) for d in prof.metrics}
    assert "lower" in dirs.values(), f"没有 lower 维 ⇒ 这条修复没有对象:{dirs}"
    assert "non_monotone" in dirs.values(), f"没有 non_monotone 维:{dirs}"


def test_rank_ddx_gate_column_uses_gated_units():
    """`rank_ddx`'s hard-gate column must come from `gated_units`, not from
    counting `overall`-prefix matches.

    Constructs a batch of **sliced-shape** rows: `overall` is `SCORED`
    everywhere (a per-slice gate hit does not propagate into overall), while
    per-slice `gates` is non-empty, so the old convention reads 0 and the new
    one reads non-zero.
    """
    from haenv.report import rank_ddx
    rows = [{"case": f"C{i}", "solver": "gemini-3.1-pro", "overall": "SCORED",
             "gates": ["missed_emergency_red_flag"] if i == 1 else [],
             "dx_hit": 1, "n_slices": 3,
             # `rank_ddx` only accepts rows whose `gold_kind` starts with `ddx:` —
             # it picks the judge by gold kind
             "gold_kind": "ddx:unified"}
            for i in (1, 2)]
    recs = [r for r in rank_ddx(rows) if r["model"] == "gemini-3.1-pro"]
    assert recs, "rank_ddx 没产出这一行"
    r = recs[0]
    assert r["gate_fail_rows"] == 0, "前提不成立:这批的 overall 本就该全是 SCORED"
    assert r["gate_fail"] == r["n_gated_units"], (
        f"硬门列 {r['gate_fail']} 与乘子用的 {r['n_gated_units']} 不是同一个量")
    assert r["gate_fail"] >= 1, f"逐片门命中了却没进硬门列:{r}"
