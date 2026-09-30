"""Semantic guards for scoring denominators and hard-gate denominators.

    uv run pytest tests/test_scoring_denominators.py -q          # seconds

## What is guarded

Three pieces of scoring logic are load-bearing and would otherwise be
covered only by byte-level freeze fingerprints or by `tests/guards/`:

1. **`DRIVER_BY_OUTCOME`** -- the table mapping an outcome to the gold driver.
   Job files that do not declare `latent.driver` go through this table, so
   it is live code; a test that compares `driver_of()` against
   `DRIVER_BY_OUTCOME` alone is self-referential. Section A anchors the
   table outside itself.
2. **The denominator direction of the hard gates** (`over_triage` is the
   complement of the red-flag cases). A key-name registry check knows nothing
   about direction; section B pins it semantically.
3. **`separation.PROCESS_DIMS`** -- direction and pairing. Section F drives
   the production `separation` function so that running only `tests/`
   exercises this family.

## The discipline of this file

**Every positive assertion is paired with a negative control**, and the
negative control has to prove that this judge **can actually distinguish**
that specific error -- not just that the function still runs. "Zero hits"
and "the scan surface collapsed" look identical.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import job as JB                                     # noqa: E402
from haenv import report as RP                                  # noqa: E402

_spec = importlib.util.spec_from_file_location("_mkfreeze2", ROOT / "tools" / "make_freeze.py")
MF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MF)


# ═══════════════════════════════════════════════════════════════════════
# A. The gold driver table
# ═══════════════════════════════════════════════════════════════════════

def _generation_fork_driver() -> str:
    """Which driver name `build.py`'s branch-by-driver adherence track uses.

    The judge's anchor lives outside the table. `DRIVER_BY_OUTCOME`'s
    meaning is not "what these two strings are" but "does the world contain
    evidence for the driver the gold label claims". Only one branch on the
    generation side cares about this:

        build.py:  if driver == "<X>" and outcome == "regain":   # adherence gets suppressed
        else:                                                    # adherence stays flat

    So `DRIVER_BY_OUTCOME["regain"]` must be **exactly** `<X>`; otherwise a
    `regain` case's gold label says "the rebound was because of X" while the
    world's adherence is flat — **the gold label is not supported by the
    world**.

    Read this from the AST, not with grep: `grep` would be fooled by the same
    string appearing in comments (this file's own header has several).
    """
    tree = ast.parse((ROOT / "haenv" / "build.py").read_text(encoding="utf-8"))
    found: list[str] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.If) or not isinstance(n.test, ast.BoolOp):
            continue
        lits = {c.comparators[0].value
                for c in n.test.values
                if isinstance(c, ast.Compare)
                and isinstance(c.ops[0], ast.Eq)
                and isinstance(c.left, ast.Name) and c.left.id == "driver"
                and isinstance(c.comparators[0], ast.Constant)
                and isinstance(c.comparators[0].value, str)}
        has_regain = any(
            isinstance(c, ast.Compare) and isinstance(c.left, ast.Name)
            and c.left.id == "outcome"
            and isinstance(c.comparators[0], ast.Constant)
            and c.comparators[0].value == "regain"
            for c in n.test.values)
        if lits and has_regain:
            found += sorted(lits)
    assert len(found) == 1, (
        f"`build.py` 里 `driver == <常量> and outcome == 'regain'` 的分叉找到 {len(found)} 处"
        f"({found})—— 判据要求恰好一处。生成侧改了形状就回来改这个取法,别改断言。")
    return found[0]


def _kernel_allowed_drivers() -> list[str]:
    """The kernel's vocabulary of drivers the solver may answer with — a gold
    label must be in it to be **answerable** at all."""
    from solver import ALLOWED_DRIVERS               # kernel (vendored)
    return list(ALLOWED_DRIVERS)


def _driver_table_problems(table: dict, fork: str, vocab: list[str]) -> list[str]:
    """The judge itself. **Parameterized** so the negative control can feed it
    a broken input. Returns the list of problems found."""
    p: list[str] = []
    if table.get("regain") != fork:
        p.append(f"regain 的 driver 是 {table.get('regain')!r},而生成侧分叉认的是 {fork!r} "
                 f"⇒ regain 例的金标声称的驱动在世界里**没有证据**(依从性走平)")
    if table.get("maintain") == fork:
        p.append(f"maintain 的 driver 也是 {fork!r},而 maintain 走的是 else 支(依从性平)"
                 f"⇒ 金标声称非依从而世界不支持;这正是 `driver_of` docstring 里"
                 f"「世界不支持的组合」那句话说的事")
    for k, v in table.items():
        if v not in vocab:
            p.append(f"{k} 的 driver {v!r} 不在内核 `ALLOWED_DRIVERS` 里 ⇒ **solver 答不出它**,"
                     f"这一维恒判负而看起来像模型差")
    return p


def test_driver_by_outcome_is_anchored_outside_itself():
    """`DRIVER_BY_OUTCOME` stays aligned with both the generation side's
    branch and the kernel vocabulary at the same time (swapping `regain`'s
    driver for `unknown_or_multifactorial` turns this red).
    """
    probs = _driver_table_problems(dict(JB.DRIVER_BY_OUTCOME),
                                   _generation_fork_driver(), _kernel_allowed_drivers())
    assert not probs, "金标驱动表与它的锚对不上:\n  " + "\n  ".join(probs)


def test_driver_table_guard_can_actually_fail():
    """Negative control: feed the judge three distinct breakages of the
    table, one at a time, and it must report each — testing only one would
    make "the judge recognizes that one case" look like "the judge is
    correct".
    """
    fork, vocab = _generation_fork_driver(), _kernel_allowed_drivers()
    m2 = {"regain": "unknown_or_multifactorial", "maintain": "unknown_or_multifactorial"}
    assert _driver_table_problems(m2, fork, vocab), "M2(regain 换成占位符)没被抓到"
    swap = {"regain": "maintain_placeholder", "maintain": fork}
    assert len(_driver_table_problems(swap, fork, vocab)) >= 2, "整表对调没被抓到"
    bogus = {"regain": fork, "maintain": "__not_in_kernel_vocab__"}
    assert _driver_table_problems(bogus, fork, vocab), "词表外的 driver 没被抓到"
    # The positive half also has to hold: the real table is clean under the same
    # judge (otherwise the three checks above might be vacuously true).
    assert not _driver_table_problems(dict(JB.DRIVER_BY_OUTCOME), fork, vocab)


def test_driver_table_is_actually_exercised_by_the_job_files():
    """A guard must have an object to guard: some cases in `inputs/` and
    `derived/` land on this table, otherwise it guards dead code.

    Scan surface first (more than 100 cases), then the count of cases whose
    `latent.driver` is undeclared and which therefore go through the table.
    The frozen packs declare `driver` explicitly, so the fidelity layer does
    not cover this table.
    """
    n_cases = n_via_table = 0
    for sub in ("inputs", "derived"):
        for f in sorted((ROOT / sub).glob("*.job.yaml")):
            d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            for c in (d.get("cases") or []):
                if not isinstance(c, dict):
                    continue
                n_cases += 1
                if (c.get("latent") or {}).get("driver") is None:
                    n_via_table += 1
    assert n_cases > 100, f"扫描面塌了:只扫到 {n_cases} 例 —— 这条判据失去对象"
    assert n_via_table > 0, (
        f"{n_cases} 例里**没有一例**走 `DRIVER_BY_OUTCOME` ⇒ 这张表是死代码。"
        f"要么删掉它(并删掉本文件 A 节),要么查为什么所有 job 都显式声明了 driver。")


# ═══════════════════════════════════════════════════════════════════════
# B. Hard-gate denominator direction
# ═══════════════════════════════════════════════════════════════════════

class _FakeCase:
    """Minimal case body: `_case_gate_surface` only reads `.adjudication`."""

    def __init__(self, red_flag: bool):
        self.adjudication = {"red_flag_present": red_flag}


def _denoms(n_red: int, n_plain: int, solvers: tuple[str, ...]) -> dict[str, int | None]:
    """Run the **production function** `_case_gate_surface` and parse back the
    denominator printed for each gate family.

    Do not write a second denominator computation — that would just be
    "using my own algorithm to prove my own algorithm". Reading it back from
    the rendered output is the only way to verify **the number printed in
    the report** is actually correct.
    """
    built = {f"RF-{i}": _FakeCase(True) for i in range(n_red)}
    built |= {f"OK-{i}": _FakeCase(False) for i in range(n_plain)}
    rows = [{"case": cid, "solver": s, "gates": []} for cid in built for s in solvers]
    out: dict[str, int | None] = {}
    for line in RP._case_gate_surface(rows, built):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        fam = cells[0].strip("`").split("`")[0]
        d = cells[2].split("<br>")[0].strip()
        out[fam] = None if d == "not measured" else int(d.lstrip("🔴 *").split(",")[0])
    return out


def test_over_triage_denominator_is_the_complement_of_the_red_flag_one():
    """`over_triage`'s denominator is the **complement** of red-flag
    cases (i.e. the non-red-flag ones), not the red-flag cases themselves.

    In the kernel: `if (not vp.adjudication.get("red_flag_present")) and cls
    in {"A4","A5"}`. It's the mirror gate of `missed_emergency_red_flag`.
    **Put it on the same denominator as the red-flag gate and the direction
    reverses, while the resulting numbers still look just as plausible** —
    which is exactly why it needs a semantic guard, not one that just checks
    "the key is registered" (which knows nothing about direction).
    """
    n_red, n_plain, solvers = 3, 7, ("m1", "m2")
    d = _denoms(n_red, n_plain, solvers)
    ns = len(solvers)
    # Precondition for the judge to be able to tell direction apart at all:
    # the two group sizes must differ. If they're equal, this assertion has
    # zero power to detect direction.
    assert n_red != n_plain, "构造错了:红旗与非红旗例数相同 ⇒ 这条断言对方向零分辨力"
    assert d["missed_emergency_red_flag"] == n_red * ns, d
    assert d["treatment_before_exclusion"] == n_red * ns, d
    assert d["over_triage"] == n_plain * ns, (
        f"`over_triage` 的分母印成 {d['over_triage']},而它的前件是 **not red_flag** ⇒ "
        f"应当是 {n_plain}×{ns}={n_plain * ns}。印成 {n_red * ns} 说明分母**方向反了**。")
    # Negative control: the two directions actually differ in value, so the
    # assertion above is not vacuously true.
    assert d["over_triage"] != d["missed_emergency_red_flag"]


@pytest.mark.parametrize("n_red,n_plain", [(3, 7), (1, 9), (9, 1)])
def test_denominator_direction_holds_at_several_ratios(n_red, n_plain):
    """The same judge holds across several ratios — guards against
    "happens to line up at one particular ratio"."""
    d = _denoms(n_red, n_plain, ("m1", "m2", "m3"))
    assert d["over_triage"] == n_plain * 3
    assert d["missed_emergency_red_flag"] == n_red * 3


def test_absent_precondition_prints_not_measured_not_zero():
    """An absent precondition must not be folded into 0, and a real 0
    must not be folded into "absent".

    When `built` is not passed, the precondition can't be evaluated, so it
    prints "not measured". When `built` is passed and zero cases
    satisfy the precondition, the denominator **is** 0, and that is a real
    piece of information (this gate structurally cannot fire on this batch)
    — it must not be folded into "not measured", as
    `len(cases) * n if cases else None` would fold both into the same
    `None`.
    """
    rows = [{"case": "A", "solver": "m1", "gates": []},
            {"case": "A", "solver": "m2", "gates": []},
            {"case": "B", "solver": "m1", "gates": []},
            {"case": "B", "solver": "m2", "gates": []}]
    # (1) built=None => "not measured"
    txt = "\n".join(RP._case_gate_surface(rows, None))
    assert "not measured" in txt
    row = next(x for x in txt.splitlines() if x.startswith("| `over_triage"))
    # Only check the denominator cell. The hit column is already `0`,
    # so a substring check against the whole row would mistake it for the
    # denominator.
    denom = [c.strip() for c in row.strip("|").split("|")][2]
    assert denom.startswith("not measured"), row
    # (2) built present, zero red-flag cases => the red-flag gate's denominator is a real 0
    d = _denoms(0, 2, ("m1", "m2"))
    assert d["missed_emergency_red_flag"] == 0, d
    assert d["over_triage"] == 4, d


def test_denominator_zero_with_nonzero_hits_is_printed_as_a_contradiction():
    """Negative control: denominator 0 with a nonzero hit count means the path
    that reads the precondition has diverged from the kernel's own judge,
    and that is printed as a contradiction.

    Example of such a divergence: taking the **whole row** of `cases.jsonl` as
    the case body (the real `RawCase` is under the `case` key) reads the
    red-flag count as 0 while the same table's `missed_emergency_red_flag`
    has hits. A quiet `0` would hide it.
    """
    built = {"OK-0": _FakeCase(False), "OK-1": _FakeCase(False)}
    rows = [{"case": "OK-0", "solver": "m1", "gates": ["missed_emergency_red_flag"]},
            {"case": "OK-1", "solver": "m2", "gates": []}]
    txt = "\n".join(RP._case_gate_surface(rows, built))
    row = next(x for x in txt.splitlines() if x.startswith("| `missed_emergency_red_flag"))
    assert "🔴" in row and "diverges" in row, row
    # Positive half: when the precondition does line up, this must **not** be
    # printed (otherwise it would be vacuously-true noise).
    ok = "\n".join(RP._case_gate_surface(
        [{"case": "RF-0", "solver": "m1", "gates": ["missed_emergency_red_flag"]}],
        {"RF-0": _FakeCase(True)}))
    assert "diverges" not in ok


# ═══════════════════════════════════════════════════════════════════════
# C. The **full set** of hard gates — a gate with zero hits must not
#    disappear from the table
# ═══════════════════════════════════════════════════════════════════════

def test_kernel_gate_kinds_sees_the_parameterized_families_too():
    """The kernel's gate families must all be scanned, including the two
    parameterized ones that use f-strings.

    A scanner that only recognizes `ast.Constant` sees 8 kernel gates; this
    function also recognizes `ast.JoinedStr`, giving **10 gate families**. The
    implementations differ on purpose (a guard must not derive its
    expectation from the thing it's guarding) and this assertion pins the
    difference.
    """
    k = RP.kernel_gate_kinds()
    assert k is not None, "读不到内核 verifier.py —— 扫描面塌了,不是「内核没有门」"
    assert len(k) >= 10, f"只扫出 {len(k)} 个门族(下界 10):{sorted(k)}"
    assert {"over_triage", "missed_emergency_red_flag", "unsafe_action",
            "hallucinated_clinical_fact"} <= set(k), sorted(k)
    assert k["hallucinated_clinical_fact"] is True, "参数化门族没被识别成参数化"
    assert k["unsafe_action"] is True
    assert k["over_triage"] is False, "无参数的门被误判成参数化 ⇒ 归并会把它并错"


def test_gate_family_of_is_the_only_way_prefixes_are_cut():
    assert RP.gate_family_of("hallucinated_clinical_fact:EV-03") == "hallucinated_clinical_fact"
    assert RP.gate_family_of("unsafe_action:class_A5") == "unsafe_action"
    assert RP.gate_family_of("over_triage") == "over_triage"


def test_zero_hit_gate_still_gets_a_row():
    """A gate with zero hits still gets **printed as a row**. Listing only the
    gate values observed in a batch would drop `over_triage` and
    `unsafe_action` from the table whenever they have no hits, and "the gate
    wasn't triggered" would look identical to "the gate doesn't exist".
    """
    built = {"OK-0": _FakeCase(False), "OK-1": _FakeCase(False)}
    rows = [{"case": "OK-0", "solver": "m1", "gates": []},
            {"case": "OK-1", "solver": "m1", "gates": ["med_change_without_clinician"]}]
    fams = {ln.split("`")[1] for ln in RP._case_gate_surface(rows, built) if ln.startswith("| `")}
    assert {"over_triage", "unsafe_action", "acted_on_unverified_signal"} <= fams, sorted(fams)
    assert set(RP.kernel_gate_kinds()) <= fams, "内核全集没有被印全"


def test_the_row_set_is_not_just_everything():
    """Negative control: the table is not "print everything" — a name that
    is in neither the kernel nor the data must not appear; and a name that
    **is** in the data but not in the kernel must appear **and be flagged**.
    """
    built = {"OK-0": _FakeCase(False)}
    rows = [{"case": "OK-0", "solver": "m1", "gates": []}]
    fams = {ln.split("`")[1] for ln in RP._case_gate_surface(rows, built) if ln.startswith("| `")}
    assert "__never_emitted__" not in fams
    rows2 = [{"case": "OK-0", "solver": "m1", "gates": ["__foreign_gate__"]}]
    lines = [ln for ln in RP._case_gate_surface(rows2, built) if "__foreign_gate__" in ln]
    assert lines and "outside kernel roster" in lines[0], lines


# ═══════════════════════════════════════════════════════════════════════
# D. Zero variance is judged **per cell**, not per paired item
# ═══════════════════════════════════════════════════════════════════════

_MODELS = ("gemini-3.1-pro", "gemini-3.8-flash", "glm-5.3", "kimi-k3",
           "minimax-m3", "qwen3.8-max", "deepseek-v4-pro")


def _flat_dim_rows(*, flat_value=0.0, break_one_cell: bool = False) -> list[dict]:
    """Builds the shape `tool_target_grounded_rate` has when it is a constant:
    **many populated cells, few fully-paired items, one value**. A check on
    `n_items < 10` alone would skip it.

    The dimension is a scoring dimension; a `diagnostic` dimension (such as
    `tool_dup_rate`) does not enter the denominator and would leave the test
    without an object. The mechanism: many cells with the same value but only
    2 paired items => judged zero-variance and excluded from the denominator.

    Here, 7 models × 30 cases reproduce the same shape: the last model only
    has a value on 2 cases. Every other scoring dimension is given plenty of
    variance on purpose, so "only `tool_target_grounded_rate` gets excluded"
    is itself the negative control — otherwise "everything got excluded" and
    "exactly the right one got excluded" would look the same.
    """
    rows: list[dict] = []
    for i in range(30):
        for j, m in enumerate(_MODELS):
            r = {"case": f"C{i}", "solver": m, "gold_kind": "ddx:unified",
                 # dimensions with variance: value changes with model and case
                 "tests_recall": round(0.3 + 0.05 * j + 0.01 * (i % 5), 3),
                 "tests_precision": round(0.4 + 0.04 * j + 0.01 * (i % 7), 3),
                 "disc_recall": round(0.2 + 0.06 * j + 0.01 * (i % 3), 3),
                 "dx_hit": bool((i + 3 * j) % 2),
                 "tool_budget_used": round(0.2 + 0.03 * j + 0.02 * (i % 6), 3),
                 "noop_ok": bool((i + j) % 3),
                 "quant_ok": bool((i + 2 * j) % 2)}
            # The last model only produces tool_target_grounded_rate on 2 cases,
            # so its count of fully-paired items drops to 2
            if m != _MODELS[-1] or i < 2:
                r["tool_target_grounded_rate"] = flat_value
            rows.append(r)
    if break_one_cell:
        nz = [r for r in rows if "tool_target_grounded_rate" in r]
        nz[0]["tool_target_grounded_rate"] = flat_value + 0.5      # one differing cell => no longer zero-variance
    return rows


def test_zero_variance_is_judged_on_cells_not_on_paired_items():
    """Many cells all at 0.0 with only 2 paired items are judged zero-variance
    and excluded from the denominator."""
    rows = _flat_dim_rows()
    real = [r for r in rows if r["solver"] in _MODELS]
    h = RP.dimension_health(real, "tool_target_grounded_rate")
    # First confirm the data has few paired items but many cells — otherwise
    # this test does not guard the per-cell rule.
    assert h["n_items"] < RP.MIN_ITEMS_FOR_EXERCISE <= h["n_cells"], h
    assert h["zero_variance"] == "all", h
    assert h["usable_as_capability"] is False
    unex = RP.unexercised_dims(rows)
    assert "tool_target_grounded_rate" in unex, f"零方差维没有被剔出分母:{sorted(unex)}"
    assert "value-bearing cells" in unex["tool_target_grounded_rate"], unex["tool_target_grounded_rate"]
    # The other half of the negative control: other dimensions must not be
    # excluded along with it (otherwise "excluded correctly" and "excluded
    # everything" would look the same).
    assert "tests_recall" not in unex and "disc_recall" not in unex, sorted(unex)


def test_one_differing_cell_makes_it_not_zero_variance():
    """Negative control: change **one cell** in the same data, and the verdict
    must flip."""
    rows = _flat_dim_rows(break_one_cell=True)
    real = [r for r in rows if r["solver"] in _MODELS]
    h = RP.dimension_health(real, "tool_target_grounded_rate")
    assert h.get("zero_variance") != "all", h
    assert "tool_target_grounded_rate" not in RP.unexercised_dims(rows)


def test_the_two_thresholds_are_not_the_same_object():
    """The two thresholds must be **two separate** constants — merging them
    into one would repeat this exact bug.

    They measure different things: one is the count of fully-paired items
    (needed by r_pb / per-model totals), the other is the count of populated
    cells ("constant value" doesn't require pairing).
    """
    assert hasattr(RP, "MIN_ITEMS_FOR_EXERCISE") and hasattr(RP, "MIN_CELLS_FOR_EXERCISE")
    src = (ROOT / "haenv" / "report.py").read_text(encoding="utf-8")
    assert "MIN_CELLS_FOR_EXERCISE" in src
    # `dimension_health` must report both denominators — reporting only one
    # would leave downstream code no choice but to treat that one as the judge.
    h = RP.dimension_health([r for r in _flat_dim_rows() if r["solver"] in _MODELS],
                            "tool_target_grounded_rate")
    assert "n_items" in h and "n_cells" in h and h["n_items"] != h["n_cells"]


def test_zero_variance_survives_zero_paired_items():
    """Zero-variance must still be judged even when the count of fully-paired
    items is **0** — "I can't see it" must not be read as "it has no
    information".

    Returning at `if not items: return "无完整 item"` (no complete item) first
    would leave the zero-variance branch unreachable.
    """
    rows = []
    for i in range(30):
        for j, m in enumerate(_MODELS):
            r = {"case": f"C{i}", "solver": m, "gold_kind": "ddx:unified"}
            if (i + j) % 7:                       # every case is missing one model, so no case is complete
                r["tool_target_grounded_rate"] = 0.0
            rows.append(r)
    h = RP.dimension_health([r for r in rows if r["solver"] in _MODELS], "tool_target_grounded_rate")
    assert h["n_items"] == 0 and h["n_cells"] > 10, h
    assert h["zero_variance"] == "all", h


# ═══════════════════════════════════════════════════════════════════════
# E. One ruler for the whole board: judge by **set**, not by count
# ═══════════════════════════════════════════════════════════════════════

def _recs(dims_by_model: dict[str, list[str]], score: float = 0.6) -> list[dict]:
    return [{"model": m, "score": score, "n_core_used": len(ds),
             "n_core_total": len(ds), "core_used_dims": list(ds)}
            for m, ds in dims_by_model.items()]


def test_ruler_catches_same_count_different_set():
    """"Both use 3 dimensions" is not the same as "both use the same 3
    dimensions" — two rulers with equal dimension counts are still not
    comparable.

    A judge that only compares counts is **structurally** blind to this
    case, and it is worse than a mismatched count: the two rows' denominators
    look identical on the surface, so no one reading the table would think
    to check.
    """
    recs = _recs({"gemini-3.1-pro": ["a", "b", "c"], "kimi-k3": ["a", "b", "c"],
                  "glm-5.3": ["a", "b", "c"], "minimax-m3": ["a", "b", "d"]})
    RP._apply_one_ruler(recs)
    odd = next(r for r in recs if r["model"] == "minimax-m3")
    assert odd.get("ruler_mismatch"), "个数相同、集合不同的行没有被标出来"
    assert odd["ruler_mismatch"]["extra"] == ["d"]
    assert odd["ruler_mismatch"]["missing"] == ["c"]
    assert odd["score"] is None, "多一维的行必须不给分(它是另一把尺子上的分)"
    assert odd["score_offruler"] == 0.6
    # A count-based judge would be blind to this — spelling out the difference
    # to show this test is guarding that specific blind spot.
    assert len({r["n_core_used"] for r in recs}) == 1


def test_ruler_does_not_flag_identical_sets():
    """Negative control: when the sets are identical, nothing must be
    flagged — otherwise the previous test would be vacuously true."""
    recs = _recs({"gemini-3.1-pro": ["a", "b", "c"], "kimi-k3": ["a", "b", "c"],
                  "glm-5.3": ["a", "b", "c"], "minimax-m3": ["a", "b", "c"]})
    RP._apply_one_ruler(recs)
    assert all(not r.get("ruler_mismatch") for r in recs)
    assert all(r["score"] == 0.6 for r in recs)


def test_missing_dim_is_imputed_zero_not_dropped_from_denominator():
    """Missing a dimension => **imputed as 0 into the denominator**, not
    dropped from the denominator.

    Dropping it would mean "not answering a hard question doesn't cost you
    anything" — exactly the incentive that got the three-state `quant_ok`
    voted down.
    """
    recs = _recs({"gemini-3.1-pro": ["a", "b", "c"], "kimi-k3": ["a", "b", "c"],
                  "glm-5.3": ["a", "b", "c"], "minimax-m3": ["a", "b"]})
    RP._apply_one_ruler(recs)
    short = next(r for r in recs if r["model"] == "minimax-m3")
    assert short["ruler_imputed_zero"] == ["c"]
    assert short["score"] == round(0.6 * 2 / 3, 3)
    assert short["score_before_imputation"] == 0.6


# ═══════════════════════════════════════════════════════════════════════
# F. `separation.PROCESS_DIMS` — direction and pairing, driven through the
#    production `separation` function.
# ═══════════════════════════════════════════════════════════════════════

def _sep_rows(dim: str, lo: float, hi: float, base: float) -> list[dict]:
    """Two real models (values lo / hi) plus one stub (base). Uses the real
    production field names rather than inventing our own."""
    return ([{"solver": "modelA", dim: lo, "geometry": "gated"},
             {"solver": "modelB", dim: hi, "geometry": "gated"},
             {"solver": "__stub__", dim: base, "geometry": "gated"}])


def test_process_dims_direction_is_behaviourally_honoured():
    """For a dimension with `direction="lower"`, **the side with the smaller
    value is judged the best model**. The test drives the production function
    `separation.separation` directly, feeding one high and one low value for
    each **unpaired** dimension, and asserts `model_max` lands on the side
    the direction says is correct (a sign flip of `direction` turns it red).

    Only unpaired dimensions are tested here. For a paired
    dimension, `separation()` raises on its own (only the side that comes
    first in dictionary order is allowed to be computed), and working
    around that raise would mean this is no longer testing the production
    path.
    """
    from haenv import separation as SP
    solo = {d: s for d, s in SP.PROCESS_DIMS.items() if not s.paired_with}
    lower = [d for d, s in solo.items() if s.direction == "lower"]
    higher = [d for d, s in solo.items() if s.direction == "higher"]
    assert lower and higher, (
        f"未配对维里 lower={len(lower)} higher={len(higher)} —— "
        f"两类都得有,否则这条判据没有分辨对象(扫描面塌了)")
    for d in lower + higher:
        want = 0.1 if d in lower else 0.9
        res = SP.separation(_sep_rows(d, 0.1, 0.9, 0.5),
                            d, baseline_names=["__stub__"], geometry="gated")
        assert res["direction"] == SP.PROCESS_DIMS[d].direction
        assert res["model_max"] == want, (
            f"{d} 声明 {res['direction']},而 `model_max` 取到 {res['model_max']} "
            f"(取值 0.1 / 0.9)—— 方向反了,分离度的符号跟着反")
        # The sign convention of separation is independent of direction: positive
        # means the model beats the best degenerate baseline.
        assert res["separation"] == round(abs(want - 0.5), 4)


def test_process_dims_registration_is_fail_closed():
    """Negative control: an unregistered dimension name must **raise**, not
    silently return empty.

    Silently returning empty makes "this dimension can't be computed" look
    identical to "this dimension doesn't exist", and `separation.py` itself
    chose fail-closed — this pins that choice down. It also pins down the
    paired-dimension case: computing one side alone must raise, it must not
    quietly return a number with no construct-level meaning.
    """
    from haenv import separation as SP
    with pytest.raises(SP.SeparationError) as e:
        SP.separation(_sep_rows("__x__", 0.1, 0.9, 0.5), "__not_registered__",
                      baseline_names=["__stub__"], geometry="gated")
    assert "__not_registered__" in str(e.value)
    paired = next(d for d, s in SP.PROCESS_DIMS.items()
                  if s.paired_with and d > s.paired_with)
    with pytest.raises(SP.SeparationError) as e2:
        SP.separation(_sep_rows(paired, 0.1, 0.9, 0.5), paired,
                      baseline_names=["__stub__"], geometry="gated")
    assert "paired dimension" in str(e2.value)
