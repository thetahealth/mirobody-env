"""The diagnosis atom `dx_listed` (on the differential and not ruled out), and the two
matcher fixes it rests on: pending-exclusion phrasing is not a negation, and a
finding-level word does not by itself name a comorbidity thread.

Every rule is checked from both sides.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import overlay, quantities                        # noqa: E402
from haenv.judges import SINGLE, run_judges                  # noqa: E402
from haenv.scoring import load_profile                       # noqa: E402
from haenv.tracks import alias_hit_asserted, thread_aliases  # noqa: E402


class FakeOut:
    def __init__(self, raw=None):
        self._raw = raw or {}
        self.action = {"specific_action": ""}
        self.drivers = []
        self.forecast = {}


class FakeVP:
    def __init__(self, ddx):
        self.adjudication = {"ddx": ddx}
        self.gold_drivers = ["poor_medication_adherence"]
        self.outcome_label = "event_occurred"


def _diff(*names, ruled=()):
    return FakeOut({"differential": [
        {"rank": i, "diagnosis": n, "ruled_out_by": ("EV-1" if n in ruled else None)}
        for i, n in enumerate(names, 1)]})


PHEO = ["嗜铬细胞瘤", "pheochromocytoma", "pheo"]
UNI_PHEO = {"diagnosis": "嗜铬细胞瘤", "aliases": PHEO, "join_gold": "unified"}
# Thread aliases exactly as the composed spec HD-COM-04 carries them (kernel aliases).
HYPO = ["甲减", "甲状腺功能减退", "hypothyroid", "桥本"]
PHPT = ["甲状旁腺", "甲旁亢", "高钙", "hyperparathyroid", "pth"]
COM_PHPT = {"diagnosis": "原发性甲状腺功能减退(桥本) + 原发性甲状旁腺功能亢进(高钙血症)",
            "aliases": HYPO + PHPT, "join_gold": "comorbidity",
            "threads": [{"name": "甲状腺线", "aliases": HYPO}, {"name": "甲状旁腺线", "aliases": PHPT}]}


# ---------------------------------------------------------------- pending exclusion
PENDING = [
    "嗜铬细胞瘤（待排除）",
    "嗜铬细胞瘤待排除",
    "需排除嗜铬细胞瘤/副神经节瘤等继发性阵发性高血压",
    "需进一步分型并排除嗜铬细胞瘤",
    "有待排除嗜铬细胞瘤",
    "建议排除嗜铬细胞瘤",
    "阵发性继发性高血压: 嗜铬细胞瘤可能 (需排除肾血管性高血压等)",
    "pheochromocytoma, needs to be ruled out",
    "we need to rule out pheochromocytoma",
]
NEGATED = [
    "已排除嗜铬细胞瘤",
    "嗜铬细胞瘤已排除",
    "不支持嗜铬细胞瘤",
    "无需排除嗜铬细胞瘤",
    "排除嗜铬细胞瘤后考虑原发性高血压",
    "已排除嗜铬细胞瘤，需排除肾血管性高血压",
    "pheochromocytoma ruled out",
    "no need to rule out pheochromocytoma",
    "无需进一步排除嗜铬细胞瘤",
    "已进一步排除嗜铬细胞瘤",
    "we do not need to rule out pheochromocytoma",
]


@pytest.mark.parametrize("text", PENDING)
def test_pending_exclusion_is_not_a_negation(text):
    assert alias_hit_asserted(text, PHEO), text


@pytest.mark.parametrize("text", NEGATED)
def test_exclusion_is_still_a_negation(text):
    assert not alias_hit_asserted(text, PHEO), text


def test_pending_candidate_counts_as_listed_on_a_unified_case():
    r = run_judges(SINGLE, _diff("原发性高血压", "嗜铬细胞瘤（待排除）"), FakeVP(UNI_PHEO))
    assert r["dx_hit"] is True and r["dx_listed"] == 1.0, r


def test_pending_candidate_with_ruled_out_by_is_still_excluded():
    r = run_judges(SINGLE, _diff("原发性高血压", "嗜铬细胞瘤（待排除）", ruled=("嗜铬细胞瘤（待排除）",)),
                   FakeVP(UNI_PHEO))
    assert r["dx_hit"] is False and r["dx_listed"] == 0.0, r


# ---------------------------------------------------------------- finding-level thread aliases
def _finding_table() -> dict:
    return yaml.safe_load((ROOT / "registry/threads.yaml").read_text(encoding="utf-8"))["finding_aliases"]


def test_finding_table_entries_are_explained_and_well_formed():
    tab = _finding_table()
    assert tab, "registry/threads.yaml: finding_aliases is empty"
    for w, e in tab.items():
        assert str((e or {}).get("why") or "").strip(), f"{w}: missing why"
        for f in (e or {}).get("diagnosis_forms") or ():
            assert w.lower() in str(f).lower(), f"{w}: diagnosis form {f!r} does not contain the word"


def test_every_finding_word_is_used_by_some_registered_thread():
    """A table entry no thread uses is dead weight (and hides a typo)."""
    used = set()
    for sid, spec in overlay.condition_registry().items():
        for t in overlay.threads_for(sid, spec) or ():
            used |= {str(a).lower() for a in t["aliases"]}
    dead = sorted(w for w in _finding_table() if w.lower() not in used)
    assert not dead, f"finding_aliases entries no thread carries: {dead}"


def test_thread_aliases_drops_finding_words():
    assert thread_aliases(PHPT) == ["甲状旁腺", "甲旁亢", "hyperparathyroid"]
    assert thread_aliases(HYPO) == HYPO


@pytest.mark.parametrize("text", [
    "结节病伴高钙血症/肾结石/可能肾上腺受累",
    "系统性红斑狼疮(SLE)伴高钙血症及早期肾损伤",
    "恶性肿瘤伴肾上腺转移及高钙血症（如肿瘤分泌PTHrP或骨转移）",
    "高钙血症待复核，可能是需要单独评估的代谢异常",
])
def test_a_finding_alone_does_not_carry_the_parathyroid_thread(text):
    r = run_judges(SINGLE, _diff("原发性甲状腺功能减退(桥本)", text), FakeVP(COM_PHPT))
    assert r["dx_threads_matched"] == 1 and r["dx_all_threads"] is False and r["dx_listed"] == 0.5, r


@pytest.mark.parametrize("text", [
    "原发性甲状旁腺功能亢进症",
    "甲旁亢(高钙血症)",
    "Primary hyperparathyroidism",
    "疑似原发性甲状旁腺功能亢进(高钙血症+肾结石/镜下血尿,待PTH确认)",
])
def test_the_parathyroid_diagnosis_still_carries_its_thread(text):
    r = run_judges(SINGLE, _diff("原发性甲状腺功能减退(桥本)", text), FakeVP(COM_PHPT))
    assert r["dx_all_threads"] is True and r["dx_listed"] == 1.0, r


def _registry_thread(sid: str, name: str) -> list[str]:
    spec = overlay.condition_registry().get(sid)
    for t in overlay.threads_for(sid, spec) or ():
        if t["name"] == name:
            return list(t["aliases"])
    raise AssertionError(f"{sid} has no thread {name}")


@pytest.mark.parametrize("sid,thread,text,want", [
    # growth hormone: the hormone is a lab analyte, a GH-secreting tumour or GH excess is the disease
    ("HD-COM-10", "垂体生长激素线", "垂体生长激素腺瘤", True),
    ("HD-COM-10", "垂体生长激素线", "生长激素分泌型垂体瘤", True),
    ("HD-COM-10", "垂体生长激素线", "疑似生长激素过多", True),
    ("HD-COM-10", "垂体生长激素线", "成人生长激素缺乏症", False),
    ("HD-COM-10", "垂体生长激素线", "生长激素水平待查", False),
    # insulin: resistance is the diagnosis, the hormone/tumour is not
    ("HD-COM-19", "糖代谢线", "Metabolic syndrome with severe insulin resistance", True),
    ("HD-COM-19", "糖代谢线", "insulin-resistant state", True),
    ("HD-COM-19", "糖代谢线", "insulinoma", False),
    # GAD antibodies are the marker, not the diagnosis
    ("HD-COM-20", "胰岛自身免疫线", "GAD抗体阳性", False),
    ("HD-COM-20", "胰岛自身免疫线", "成人隐匿性自身免疫糖尿病(LADA)", True),
    # renal: chronic renal insufficiency is CKD; damage/proteinuria/declining function are findings
    ("JD-CKM", "肾脏线", "慢性肾功能不全", True),
    ("JD-CKM", "肾脏线", "慢性肾病3期", True),
    ("JD-CKM", "肾脏线", "肾功能不全", False),
    ("JD-CKM", "肾脏线", "高血压病1级伴早期肾损害", False),
    ("JD-CKM", "肾脏线", "早期靶器官损害（肾脏受累/容量负荷）", False),
    ("JD-CKM", "肾脏线", "肾性蛋白尿", False),
    ("JD-CKM", "肾脏线", "肾功能下降", False),
    # glucose: hyperglycaemia and poor control are findings, diabetes is the disease
    ("JD-CKM", "糖代谢线", "2型糖尿病失控", True),
    ("JD-CKM", "糖代谢线", "高血糖待查", False),
    ("JD-CKM", "糖代谢线", "糖代谢异常", False),
])
def test_registered_thread_after_filtering(sid, thread, text, want):
    assert alias_hit_asserted(text, thread_aliases(_registry_thread(sid, thread))) is want, text


# ---------------------------------------------------------------- the atom itself
def test_dx_listed_on_unified_cases():
    assert run_judges(SINGLE, _diff("嗜铬细胞瘤", "原发性醛固酮增多症"), FakeVP(UNI_PHEO))["dx_listed"] == 1.0
    assert run_judges(SINGLE, _diff("原发性醛固酮增多症", "库欣综合征"), FakeVP(UNI_PHEO))["dx_listed"] == 0.0
    r = run_judges(SINGLE, _diff("嗜铬细胞瘤", "原发性醛固酮增多症", ruled=("嗜铬细胞瘤",)), FakeVP(UNI_PHEO))
    assert r["dx_listed"] == 0.0 and r["dx_ruled_out_gold"] is True


def test_dx_listed_on_comorbidity_cases_is_thread_coverage():
    vp = FakeVP(COM_PHPT)
    assert run_judges(SINGLE, _diff("甲减", "原发性甲旁亢"), vp)["dx_listed"] == 1.0
    one = run_judges(SINGLE, _diff("甲减", "焦虑状态"), vp)
    # the any-component reading `dx_hit` is what inflated the composite; dx_listed does not
    assert one["dx_hit"] is True and one["dx_listed"] == 0.5
    assert run_judges(SINGLE, _diff("焦虑状态", "缺铁性贫血"), vp)["dx_listed"] == 0.0
    assert run_judges(SINGLE, _diff("甲减", "原发性甲旁亢", ruled=("原发性甲旁亢",)), vp)["dx_listed"] == 0.5


def test_independent_rows_carry_no_dx_listed():
    ind = {"diagnosis": "无统一病理", "aliases": ["独立", "无统一"], "join_gold": "independent"}
    r = run_judges(SINGLE, FakeOut({"differential": [{"rank": 1, "diagnosis": "过敏性鼻炎"}],
                                    "join_type": "independent"}), FakeVP(ind))
    assert "dx_listed" not in r and "dx_hit" not in r


# ---------------------------------------------------------------- long-list control
_FILLER = ["原发性醛固酮增多症", "库欣综合征", "肢端肥大症", "系统性红斑狼疮", "结节病", "乳糜泻",
           "多囊卵巢综合征", "阻塞性睡眠呼吸暂停", "维生素B12缺乏", "遗传性血色病"]


def test_gold_buried_in_a_long_live_list_is_not_listed():
    """Listing everything is not a differential: only the first `1 + DX_LIST_CAP_MARGIN`
    live candidates count on a unified case. `dx_hit` has no cap -- that is why it is not
    the scored atom."""
    from haenv.judges.differential import DX_LIST_CAP_MARGIN
    cap = 1 + DX_LIST_CAP_MARGIN
    deep = run_judges(SINGLE, _diff(*_FILLER[:cap], "嗜铬细胞瘤"), FakeVP(UNI_PHEO))
    assert deep["dx_hit"] is True and deep["dx_listed"] == 0.0, deep
    edge = run_judges(SINGLE, _diff(*_FILLER[:cap - 1], "嗜铬细胞瘤"), FakeVP(UNI_PHEO))
    assert edge["dx_listed"] == 1.0, edge
    # excluded candidates do not use up the cap: a thorough differential is not a long list
    ruled = run_judges(SINGLE, _diff(*_FILLER[:cap], "嗜铬细胞瘤", ruled=tuple(_FILLER[:cap])), FakeVP(UNI_PHEO))
    assert ruled["dx_listed"] == 1.0, ruled


def _registry_cases():
    reg = overlay.condition_registry()
    for sid, spec in reg.items():
        if spec.get("join_gold") not in ("unified", "comorbidity"):
            continue
        ddx = {"diagnosis": spec["diagnosis"], "aliases": list(spec.get("aliases") or []),
               "join_gold": spec["join_gold"]}
        if spec["join_gold"] == "comorbidity":
            ddx["threads"] = [dict(t) for t in overlay.threads_for(sid, spec)]
        yield sid, spec, ddx


def _long_list_means(make_list) -> tuple[float, float, int]:
    listed, hit, n = 0.0, 0, 0
    for sid, spec, ddx in _registry_cases():
        r = run_judges(SINGLE, _diff(*make_list(sid, spec)), FakeVP(ddx))
        listed += r["dx_listed"]
        hit += bool(r["dx_hit"])
        n += 1
    return listed / n, hit / n, n


#: A long-list strategy must stay below this ceiling; genuine differentials score far above it.
LONG_LIST_CEILING = 0.3


def test_cap_boundary_on_a_comorbidity_case():
    """Two threads: the cap is 2 + DX_LIST_CAP_MARGIN live candidates. A thread first listed one
    place past it counts as not listed; at the cap it counts. Ruled-out candidates are skipped."""
    from haenv.judges.differential import DX_LIST_CAP_MARGIN
    cap = 2 + DX_LIST_CAP_MARGIN
    vp = FakeVP(COM_PHPT)
    past = run_judges(SINGLE, _diff("甲减", *_FILLER[:cap - 1], "原发性甲旁亢"), vp)
    assert past["dx_listed_positions"] == [1, cap + 1] and past["dx_listed"] == 0.5, past
    assert past["dx_all_threads"] is True        # the uncapped reading still sees both threads
    at = run_judges(SINGLE, _diff("甲减", *_FILLER[:cap - 2], "原发性甲旁亢"), vp)
    assert at["dx_listed_positions"] == [1, cap] and at["dx_listed"] == 1.0, at
    skip = run_judges(SINGLE, _diff("甲减", *_FILLER[:cap - 1], "原发性甲旁亢", ruled=tuple(_FILLER[:cap - 1])), vp)
    assert skip["dx_listed"] == 1.0, skip


def test_catalog_dump_scores_low_on_dx_listed():
    """Gold-blind negative control: every unified diagnosis in the condition registry, in a
    fixed order, on every registered case. `dx_hit` (no cap) is fooled by it."""
    catalog = [str(s.get("diagnosis") or "") for s in overlay.condition_registry().values()
               if s.get("join_gold") == "unified"]
    listed, hit, n = _long_list_means(lambda sid, spec: catalog)
    assert n >= 30
    assert hit > 0.9, f"control lost its bite: dx_hit on the catalog dump is only {hit:.2f}"
    assert listed < LONG_LIST_CEILING, f"catalog dump scores {listed:.3f} on dx_listed"


def test_all_rivals_then_gold_scores_low_on_dx_listed():
    """Every registered look-alike of the whole registry, then the gold last."""
    reg = overlay.condition_registry()
    rivals: list[str] = []
    for sid, spec in reg.items():
        for r in overlay.rivals_for(sid, spec) or ():
            if r["name"] not in rivals:
                rivals.append(r["name"])
    from haenv.judges.differential import DX_LIST_CAP_MARGIN
    assert len(rivals) > 2 + DX_LIST_CAP_MARGIN

    def make(sid, spec):
        tail = [str(t["aliases"][0]) for t in (overlay.threads_for(sid, spec) or ())] \
            if spec.get("join_gold") == "comorbidity" else [str(spec["diagnosis"])]
        return rivals + tail

    listed, hit, _n = _long_list_means(make)
    assert hit > 0.9, f"control lost its bite: dx_hit is only {hit:.2f}"
    assert listed < LONG_LIST_CEILING, f"rivals-then-gold scores {listed:.3f} on dx_listed"


def test_every_registered_thread_keeps_a_diagnosis_alias():
    """Filtering must never leave a thread that nothing can name."""
    empty = [(sid, t["name"]) for sid, spec, ddx in _registry_cases()
             for t in (ddx.get("threads") or []) if not thread_aliases(t["aliases"])]
    assert not empty, empty



# ---------------------------------------------------------------- scoring profile and registry
def test_scoring_profile_scores_dx_listed_not_dx_hit():
    p = load_profile()
    assert "dx_listed" in p.scored_dims and "dx_hit" not in p.scored_dims
    assert p.metrics["dx_hit"]["role"] == "diagnostic"
    assert p.metrics["dx_listed"]["metric_type"] == "score01"


def test_dx_listed_resolves_on_rows_written_before_it_existed():
    """Legacy rows lack `dx_listed`; the registry derives it (no cap: those rows do not
    record live positions)."""
    assert quantities.resolve({"dx_kind": "unified", "dx_hit": True}, "dx_listed") == 1.0
    assert quantities.resolve({"dx_kind": "unified", "dx_hit": False}, "dx_listed") == 0.0
    assert quantities.resolve({"dx_kind": "comorbidity", "dx_hit": True, "dx_coverage": 0.5},
                              "dx_listed") == 0.5
    assert quantities.resolve({"held_independent": True}, "dx_listed") is None
    # a recorded value wins over the derivation
    assert quantities.resolve({"dx_kind": "comorbidity", "dx_coverage": 1.0, "dx_listed": 0.5},
                              "dx_listed") == 0.5
