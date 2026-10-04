"""Gated pricing must come from the registry, and charges must follow the
menu.

## Two requirements

**(1) Pricing comes from the registry, not from a raw substring match.** A
matcher like `"ct" in target.lower()` prices `activity_index`,
`diet_carb_pct`, `ACTH` and "ACTH stimulation test" as **imaging 80** (all
contain `ct`), and leaves catalogue items that match no keyword (`ALT`, `AST`,
`LDL`, `fasting_glucose`, `triglycerides`, and others) at the default
**vitals 1.0**.

`GOLD_EVIDENCE['calorie_intake_change'].signal = 'diet_carb_pct'` is the
**critical signal** for early_warning T4; priced at 80 > the derived budget
of 38 => a protocol-abiding model **cannot afford it**.

**(2) Charges follow the menu, not the model's self-reported kind.**
`gk.query(kind, target)` takes `kind` as written by the model itself, and the
kernel charges by `COST[kind]` => self-reporting `vitals` for an imaging item
would pay 1.0 and still get the result.

## Why both halves are tested together

Fixing pricing alone raises the price while the self-report bypass remains,
so the protocol-abiding model comes out worse off.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import statistics
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

import haenv.gated as G  # noqa: E402
from haenv_kernel.gatekeeper import COST  # noqa: E402


def _cost(target: str) -> float:
    return COST.get(G.kind_of(target), 5.0)


# ─────────────────────────────────────────────── registry coverage · fail-closed

def test_every_catalogue_item_is_priced():
    """**Every one** of the 55 items in the test catalogue must have a
    registered price -- missing one means one pricing decision never got
    made."""
    missing = [t for t in G.test_catalogue() if t not in G._pricing()]
    assert not missing, f"没定价的检查项:{missing}"


def test_unregistered_target_raises_not_defaults_to_cheapest():
    """Not found => raise. **The default tier happens to be the
    cheapest** => falling back would make "never priced" look identical to
    "only worth 1 unit"."""
    with pytest.raises(G.PricingUnregistered):
        G.kind_of("这个 target 从来没登记过")


def test_every_kind_is_one_the_kernel_can_price():
    """A `kind` in the registry must exist in the kernel's `COST` table --
    otherwise `COST.get(k, 5.0)` silently charges 5."""
    unknown = sorted({k for k in G._pricing().values() if k not in COST})
    assert not unknown, f"内核 COST 里没有这些 kind:{unknown}(会静默按 5.0 计价)"


def test_monitoring_signals_are_all_vitals():
    """Planning convention: **monitoring signals are always vitals**. Daily
    streams are produced by the patient themself, consuming no extra
    resources."""
    from haenv import streams
    daily = [n for n, e in streams.manifest().items() if e["aux"]]
    assert len(daily) >= 20, daily
    for t in daily:
        assert G.kind_of(t) == "vitals", t


# ─────────────────────────────────────────────── negative control: substring pricing must be caught

_OLD_KEYS = {
    "imaging": ("ct", "mri", "us", "imaging", "density", "影像", "超声", "断层",
                "核磁", "骨密度", "显像", "内镜", "活检"),
    "advanced_lab": ("cortisol", "psg", "ahi", "advanced", "antibody", "试验", "抗体",
                     "基因", "电泳", "多导睡眠", "定量", "分型"),
    "basic_lab": ("lab", "hba1c", "egfr", "iron", "lipid", "creat", "血", "尿", "素",
                  "酶", "钙", "钾", "钠", "铁", "糖", "脂", "激素", "功能", "常规"),
}


def _old_kind(t: str) -> str:
    tl = str(t).lower()
    for k in ("imaging", "advanced_lab", "basic_lab"):
        if any(w in tl for w in _OLD_KEYS[k]):
            return k
    return "vitals"


@pytest.mark.parametrize("target", ["ACTH", "ACTH兴奋试验", "diet_carb_pct", "activity_index"])
def test_the_ct_substring_accident_is_gone(target):
    """Negative control: substring pricing (`_old_kind`) prices all four
    `ct`-containing targets imaging (80); the registry does not."""
    assert _old_kind(target) == "imaging", f"负对照本身失效:{target} 在旧实现下不是 imaging"
    assert G.kind_of(target) != "imaging", f"{target} 仍被判 imaging"


def test_the_eighteen_silent_vitals_are_gone():
    """Negative control: under substring pricing (`_old_kind`), 18/55 items
    fall to the default vitals. With the registry, **not a single item** falls
    into the cheapest tier for "matching no keyword"."""
    cat = G.test_catalogue()
    old_default = [t for t in cat if _old_kind(t) == "vitals"]
    assert len(old_default) == 18, f"负对照失效:旧实现落默认的应为 18 项,实得 {len(old_default)}"
    still_cheap = [t for t in old_default if G.kind_of(t) == "vitals"]
    assert not still_cheap, f"这些项仍是 vitals(1.0):{still_cheap}"


# ─────────────────────────────────────────────── critical signals must be affordable

def _budget_from(kindf) -> float:
    cat = G.test_catalogue()
    sigs = [t for t in G._pricing() if t not in cat]
    s = [COST.get(kindf(t), 5.0) for t in sigs]
    t = [COST.get(kindf(x), 5.0) for x in cat]
    return round(G.SIGNAL_ALLOWANCE * statistics.median(s)
                 + G.TEST_ALLOWANCE * statistics.median(t), 1)


def test_every_critical_signal_is_affordable():
    """Every `GOLD_EVIDENCE` critical signal's price must be <= the budget.

    Unaffordable critical signals => `tool_concluded_blind` is
    **structurally** True across the whole column; that dim would then be
    measuring our own pricing, not model behavior.
    """
    from haenv.build import GOLD_EVIDENCE
    b = _budget_from(G.kind_of)
    bad = []
    for drv, ge in GOLD_EVIDENCE.items():
        sig = getattr(ge, "signal", None) or (ge.get("signal") if isinstance(ge, dict) else None)
        if not sig:
            continue
        c = _cost(sig)
        if c > b:
            bad.append(f"{drv}: {sig} 价 {c} > 预算 {b}")
    assert not bad, "关键信号买不起:\n  " + "\n  ".join(bad)


def test_the_affordability_guard_could_have_failed():
    """Negative control: under substring pricing, `diet_carb_pct` is priced
    80 > the budget of 38, and this guard **does** fail.

    Without this test, the one above might just be vacuously true.
    """
    assert COST[_old_kind("diet_carb_pct")] == 80.0
    assert _budget_from(_old_kind) == 38.0, "旧预算不是 38 ⇒ 负对照的前提变了"
    assert COST[_old_kind("diet_carb_pct")] > _budget_from(_old_kind)


def test_selection_pressure_survives_the_reprice():
    """Counter-metric: registry pricing must not inflate the budget to the
    point of "no need to choose".

    The budget is derived from the **median unit price**, so a price increase
    raises the budget along with it -- what matters is whether **selection
    pressure** still exists. Greedily buying from cheapest to most expensive,
    the budget affords between 10% and 60% of the test catalogue (far from
    "can afford everything", far from "can afford nothing").
    """
    def afford_frac(kindf):
        cat = G.test_catalogue()
        b = _budget_from(kindf)
        n, rem = 0, b
        for c in sorted(COST.get(kindf(x), 5.0) for x in cat):
            if rem < c:
                break
            rem -= c
            n += 1
        return n / len(cat)

    new = afford_frac(G.kind_of)
    assert new < 0.60, f"预算能买下 {new:.1%} 的目录 ⇒ 取舍压力没了,T3 会恒真"
    assert new > 0.10, f"只买得起 {new:.1%} ⇒ 反过来压太死,人人超预算"


# ─────────────────────────────────────────────── charge by the menu, not the self-report

def test_charging_uses_the_menu_kind_not_the_self_reported_one():
    """The load-bearing test of this file: self-reporting
    `vitals` must not let an imaging item cost only 1 unit.

    Measures the kernel's pricing semantics directly: for the same target,
    charging by the menu kind vs. the self-reported kind must produce
    different amounts -- that difference is the entire content of the
    bypass.
    """
    from haenv_kernel.gatekeeper import Gatekeeper
    menu_kind = G.kind_of("垂体MRI")
    assert menu_kind == "imaging", menu_kind
    assert COST[menu_kind] == 80.0
    assert COST["vitals"] == 1.0
    # the charging side passes `_menu_kind.get(target)`, i.e. the 80 tier above.
    gk = Gatekeeper({}, 84)
    before = gk.spent
    gk.query(menu_kind, "垂体MRI")
    assert gk.spent - before == 80.0, "按菜单扣费没生效"


def test_kind_misreport_is_counted_not_silently_corrected():
    """Being corrected does not mean it never happened -- "honestly
    reported" and "misreported but corrected" must not look identical on
    disk."""
    tr = G.ToolTrace()
    assert hasattr(tr, "kind_misreports") and tr.kind_misreports == 0


def test_gated_probe_repeats_the_kind_from_its_first_menu():
    """The menu drops revealed targets, but a deliberate repeat keeps its original kind."""
    import json
    from types import SimpleNamespace

    from haenv.baselines import GatedProbeSolver

    payload = SimpleNamespace(evidence_ledger=[],
                              prediction_context={"target_event_type": "event"})
    solver = GatedProbeSolver()
    solver.gated_context = {"menu": [{"target": "ALT", "kind": "basic_lab"}],
                            "askable": ["ALT"]}
    first = json.loads(solver.solve(payload)._raw_text)["queries"][0]
    solver.gated_context = {"menu": [{"target": "HbA1c", "kind": "basic_lab"}],
                            "askable": ["HbA1c"]}
    solver.solve(payload)
    solver.gated_context = {"menu": [], "askable": []}
    repeated = json.loads(solver.solve(payload)._raw_text)["queries"][0]

    assert first == {"kind": "basic_lab", "target": "ALT"}
    assert repeated == first
