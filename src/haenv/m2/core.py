"""M2 generation side: gold threads with their key signals, the round-2 push block, frame gold.

Everything here is deterministic: choices are keyed on `rng.unit(case_id, ...)`, readings come from
the same renderers the gated menu uses (`gated.synth_on_demand` for declared findings and the case's
own lab streams, `findings_render._value` for background and distractor values), and no generation
LLM is called.

Definitions this module fixes (spec sections 2.2 and 2.3; the choices the spec leaves open are
listed in the M2 implementation report as post-registration revisions):

* gold line ("thread"): each declared `ddx.threads` entry; a unified case has one line named by
  `ddx.aliases`; an independent or insufficient-tier case has none.
* key signals of a line: the confirmatory, abnormal findings (`condition_findings.yaml`,
  role `confirmatory`, direction high/low/positive) of the line's component condition that some
  menu item resolves to, most marked first, at most `KEY_PER_THREAD`.
* discriminating signals: the rival discriminator findings (`rivals.yaml`) of the case's spec,
  the set `tracks.key_signals_for` reads, minus the key signals.
* push block: `PUSH_N` readings filled key -> discriminating -> distractor -> filler and truncated
  at `PUSH_N`; shown in a hash order so position carries nothing.

The question templates (frame asks, revision prompt) live at the end of this module rather than in
`framings.py`: `framings.py` is in the GENERATION stamp, and editing it invalidates the world
contracts' fixtures (tests/contracts I0) for every pack. Here they are covered, for M2 jobs only,
by the external-gold manifest: the gold blocks are this module's functions, so its fingerprint
enters the M2 batch's world stamp. Segment assignment of `haenv/m2/` is left to the next re-anchor.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.resources.regpath import registry_path
from haenv.resources.regpath import registry_cached
from haenv.m2.workup import pre_review_rules
from haenv.m2.workup import workup_tests
from haenv.m2.workup import workup_alternatives

import copy
import functools
import hashlib
import math
import re

#: Round 2 (push block, purchase-weighted `dx_listed_r2_w`, `revision_*`) is an optional profile,
#: off by default (M2 is a single round). With it off the `revision`
#: gold block is not registered, no case is pushed, and nothing of round 2 enters the composite.
#: `HAENV_M2_ROUND2=1` turns the profile on (it moves the M2 world stamp: the block is registered).
ROUND2 = __import__("os").environ.get("HAENV_M2_ROUND2", "") == "1"

PUSH_N = 5
KEY_PER_THREAD = 2
FRAMES = ("F0", "F1", "F2", "F3")
FRAME_NAMES = {"F0": "weight_regain_bridge", "F1": "acute_triage",
               "F2": "chronic_med_adjustment", "F3": "followup_interpretation"}

_ABNORMAL = ("high", "low", "positive")
_MAG_ORDER = {"marked": 0, "moderate": 1, "mild": 2, None: 3}
#: Specimen notes shared by every push item, so a note's presence or length does not mark a role.
_NOTES_PLAIN = ("空腹静脉血", "晨起空腹采血", "门诊常规采血", "上午空腹静脉采血", "复诊当日空腹采血",
                "门诊复查采血", "空腹", "常规采血", "外院送检,结果已核对", "门诊抽血,标本合格",
                "晨8点空腹肘静脉采血", "复查")


def _u(*path) -> float:
    import haenv.world.generation.rng as rng
    return rng.unit("m2", *path)


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ registries
@functools.lru_cache(maxsize=1)
def _findings() -> dict:
    from haenv.resources.registry import load_findings
    return load_findings()


@functools.lru_cache(maxsize=1)
def _profiles() -> dict:
    from haenv.resources.registry import condition_findings_for_case
    return condition_findings_for_case(_findings())


# ------------------------------------------------------------------ M2 menu and world hooks
# MENU-1 (revisions R-19, r1-1). Everything M2 adds to the gated menu and to the readings a purchase
# returns lives here and applies only to cases carrying `prediction_context.m2_frame` (F1-F3).
# `haenv/interaction/gated.py`, `registry/gated_pricing.yaml` and `haenv/evaluate.py` stay as in v1.0.1, so the
# v1.0.1 packs and the F0 bridge cases are judged with the integration judging code unchanged
# (same `judging_sha16`); the hooks are installed by `haenv/m2/hooks.py`.

#: Discriminator test items the M2 frames add to the menu (kind = price tier). Beta-hydroxybutyrate
#: is the insulinoma line's confirmatory finding (routine chemistry, CLFS CPT 82010 = basic_lab).
LIPID_PANEL = "血脂四项(TC/TG/HDL-C/LDL-C)"
M2_DISCRIMINATORS: dict[str, str] = {"血β-羟丁酸": "basic_lab",
                                     "血胰岛素": "basic_lab",
                                     # the pulmonary-embolism rival's discriminators (rivals.yaml HD-UNI-16):
                                     # D-dimer CPT 85379 (routine chemistry); CTPA CPT 71275 (imaging)
                                     "D-二聚体": "basic_lab", "CT肺动脉造影(CTPA)": "imaging",
                                     # the lipid panel (CPT 80061, routine chemistry): the only item that
                                     # reads today's LDL-C; TC, TG and HDL-C come with it
                                     LIPID_PANEL: "basic_lab"}
#: Menu items that resolve to findings on M2 frames only. The first two are existing catalogue
#: names (total testosterone CPT 84403 + SHBG 84270; serum creatinine 82565), basic_lab.
M2_RESOLVES_TO: dict[str, tuple[str, ...]] = {
    "性激素六项/睾酮·SHBG(FAI)": ("Testosterone", "SHBG"),
    "eGFR/血肌酐": ("Cr",),
    "血β-羟丁酸": ("BetaOHB",),
    "血胰岛素": ("Insulin",),
    # Gold tests of the registered conditions that the integration catalogue prices but cannot read, and
    # composites that read only part of what their name says (an item returns each of its parts).
    "肾上腺CT/AVS": ("AdrenalImaging",),
    "确诊试验(盐负荷等)": ("SalineSuppression",),
    "内因子抗体": ("IntrinsicFactorAb",),
    "尿酮体": ("UrineKetone",),
    "MIBG(必要时)": ("MIBGScan",),
    "腹部CT/MRI": ("AbdominalImaging",),
    "垂体MRI": ("PituitaryMRI",),
    "OGTT-生长激素抑制试验": ("GHSuppression",),
    "肾脏超声/尿钙": ("KidneyUSCalcium",),
    "甲状腺摄碘率/超声": ("ThyroidUptakeUS",),
    "HFE基因检测": ("HFEGene",),
    "肝铁定量MRI/活检": ("LiverIronMRI",),
    "ACTH兴奋试验": ("ACTHStim",),
    "ACTH+垂体/肾上腺影像": ("ACTH", "PitAdrenalImaging"),
    "无需系统性排查;针对各自症状对症/观察即可": ("SymptomReview",),
    "尿蛋白/肌酐比": ("UPCR",),
    "血常规MCV": ("Hb", "MCV"),
    "AST/ALT/血小板并计算FIB-4": ("AST", "ALT", "PLT", "FIB4"),
    "血清tTG-IgA + 总IgA(除外IgA缺乏)": ("tTG_IgA", "TotalIgA"),
    "抗肌内膜抗体EMA": ("EMA",),
    "十二指肠活检(须在未停麸质前)": ("DuodenalBiopsy",),
    "甲襞毛细血管镜": ("Capillaroscopy",),
    "食管测压或钡餐": ("EsophagealStudy",),
    "胸部高分辨率CT": ("ChestHRCT",),
    "血清ACE·可溶性IL-2受体": ("SarcoidMarkers",),
    "支气管镜下活检(必要时)": ("BronchoBiopsy",),
    "胰岛素抗体": ("InsulinAb",),
    "定位检查:胰腺增强CT或超声内镜(**在生化确证之后**)": ("InsulinomaLocalization",),
    "骶髂关节MRI(脂肪抑制序列)": ("SIJMRI",),
    "骨盆正位X线": ("PelvisXray",),
    "胃镜+结肠镜": ("GIEndoscopy",),
    "肾脏超声": ("RenalUltrasound",),
    "CRP·ESR": ("CRP", "ESR"),
    "HbA1c·空腹血糖(先确认糖尿病本身)": ("FBG", "HbA1c"),
    "胰腺增强CT或MRI/MRCP": ("PancreaticMassImaging",),
    "血钙·24h尿钙·1,25-(OH)2D": ("Ca", "UrineCa24h", "Calcitriol"),
    "D-二聚体": ("DDimer",),
    "CT肺动脉造影(CTPA)": ("CTPA",),
    LIPID_PANEL: ("TG", "HDL", "LDL", "TC"),
}
#: Gold test names of the registered conditions that the integration catalogue does not carry, with the
#: price tier (resource intensity, `registry/gated_pricing.yaml:basis`). An item is on an F1-F3 menu
#: only while some registered condition names it as a gold test.
M2_GOLD_ITEMS: dict[str, str] = {
    "血清tTG-IgA + 总IgA(除外IgA缺乏)": "advanced_lab",
    "抗肌内膜抗体EMA": "advanced_lab",
    "十二指肠活检(须在未停麸质前)": "imaging",
    "甲襞毛细血管镜": "imaging",
    "食管测压或钡餐": "imaging",
    "胸部高分辨率CT": "imaging",
    "血清ACE·可溶性IL-2受体": "advanced_lab",
    "支气管镜下活检(必要时)": "imaging",
    "胰岛素抗体": "advanced_lab",
    "定位检查:胰腺增强CT或超声内镜(**在生化确证之后**)": "imaging",
    "骶髂关节MRI(脂肪抑制序列)": "imaging",
    "骨盆正位X线": "imaging",
    "胃镜+结肠镜": "imaging",
    "肾脏超声": "imaging",
    "CRP·ESR": "basic_lab",
    "HbA1c·空腹血糖(先确认糖尿病本身)": "basic_lab",
    "胰腺增强CT或MRI/MRCP": "imaging",
    "血钙·24h尿钙·1,25-(OH)2D": "advanced_lab",
}


def is_m2_framed(obj) -> bool:
    """F1-F3 case, payload or instance: `prediction_context.m2_frame` is set (absent on F0)."""
    return isinstance(((getattr(obj, "prediction_context", None) or {}).get("m2_frame")), dict)


def is_m2_case(obj) -> bool:
    """A case of the M2 pack: F1-F3 (`is_m2_framed`) or an F0 bridge case (latent `m2_bridge`). The M2
    menu, purchase readings and queried world apply to both; frame asks and frame atoms stay F1-F3."""
    if is_m2_framed(obj):
        return True
    from haenv.contracts.external_gold import SLOT
    lp = getattr(obj, "latent_premise", None) or {}
    meta = (lp.get("meta") if isinstance(lp, dict) else getattr(lp, "meta", None)) or {}
    return bool((meta.get(SLOT) or {}).get("m2_bridge"))


@functools.lru_cache(maxsize=1)
def registry_gold_tests() -> frozenset[str]:
    """Every gold test of every registered condition (`ddx.tests`)."""
    from haenv.world.overlay import condition_registry
    return frozenset(str(t) for sp in condition_registry(include_draft=True).values()
                     for t in sp.get("tests") or [] if str(t).strip())


def m2_extra_items() -> dict[str, str]:
    """name -> kind of the items an F1-F3 menu adds to the integration menu."""
    gold = registry_gold_tests()
    return {**M2_DISCRIMINATORS, **{t: k for t, k in M2_GOLD_ITEMS.items() if t in gold}}


def m2_catalogue() -> list[str]:
    """Test items on an M2-frame menu: the v1.0.1 catalogue and discriminators, then M2's own."""
    from haenv.interaction.gated import discriminator_tests
    from haenv.interaction.gated import test_catalogue
    base = test_catalogue() + discriminator_tests()
    return base + [t for t in m2_extra_items() if t not in base]


def m2_menu_findings(target: str, findings: dict | None = None) -> tuple[str, ...]:
    """`gated.menu_findings` with the M2 `resolves_to` rows first."""
    from haenv.interaction.gated import menu_findings
    t = str(target or "").strip()
    return M2_RESOLVES_TO.get(t) or menu_findings(t, findings if findings is not None else _findings())


@functools.lru_cache(maxsize=1)
def menu_target_of() -> dict[str, str]:
    """finding id -> the first M2-frame menu test item (catalogue order) that resolves to it."""
    out: dict[str, str] = {}
    for t in m2_catalogue():
        for fid in m2_menu_findings(t, _findings()):
            out.setdefault(fid, t)
    return out


def gold_tests_of(raw) -> list[str]:
    """The case's gold tests, read the way the production judge reads them (registry by `spec_id`, the
    case copy as fallback)."""
    from haenv.world.overlay import condition_registry
    from haenv.world.overlay import spec_id_of
    ddx = ((getattr(raw, "adjudication", None) or {}).get("ddx")) or {}
    if (getattr(raw, "adjudication", None) or {}).get("m2_workup") is not None:
        return [] if ddx.get("insufficient") else workup_tests(
            str(ddx["spec_id"]), raw.adjudication["m2_workup"].get("optional") or ())
    sid, _ = spec_id_of(raw, allow_case_id_lookup=False)
    reg = (condition_registry(include_draft=True).get(str(sid)) or {}).get("tests")
    return [str(t) for t in (reg or ddx.get("tests") or []) if str(t).strip()]


@registry_cached("gated_pricing.yaml", "m2_pre_review.yaml", "conditions_unified.yaml")
def workup_menu() -> list[dict]:
    """The test-price catalogue, independent of a patient's hidden diagnoses."""
    from haenv.interaction.gated import kind_of
    from haenv_kernel.gatekeeper import COST
    extra = m2_extra_items()
    return [{"target": t, "is_test": True, "kind": extra[t] if t in extra else kind_of(t),
             "cost": COST[extra[t] if t in extra else kind_of(t)]} for t in m2_catalogue()]


def workup_cost(tests, menu) -> float:
    prices = {i["target"]: float(i["cost"]) for i in menu if i.get("is_test")}
    missing = set(tests) - set(prices)
    if missing:
        raise ValueError(f"required tests absent from menu: {sorted(missing)}")
    return math.fsum(prices[t] for t in set(tests))


def workup_budget(tests, menu) -> float:
    policy = pre_review_rules()["budget"]
    cost = workup_cost(tests, menu)
    margin = max(float(policy["minimum_margin"]), cost * float(policy["fractional_margin"]))
    return max(float(policy["floor"]), math.ceil((cost + margin) / policy["quantum"]) * policy["quantum"])


def workup_gold(meta: dict) -> dict | None:
    from haenv.contracts.external_gold import SLOT
    ext = meta.get(SLOT) or {}
    if not (ext.get("m2") or ext.get("m2_bridge")):
        return None
    ddx = meta.get("ddx") or {}
    tests = [] if ddx.get("insufficient") else workup_tests(str(ddx["spec_id"]))
    menu = workup_menu()
    alternatives = {t: workup_alternatives(t) for t in tests if workup_alternatives(t)}
    return {"required": tests, "alternatives": alternatives,
            "minimum_cost": workup_cost(tests, menu), "budget": workup_budget(tests, menu),
            "source": "registry/m2_pre_review.yaml", "version": pre_review_rules()["version"]}


def credited_menu_items(test: str, items) -> list[str]:
    """Menu items the production judge credits for a gold test: the item is the test, or its name holds
    a form of one of the test's segments (`judges.differential._gold_test_segs` / `_seg_forms`)."""
    from haenv.judges.differential import _gold_test_segs
    from haenv.judges.differential import _seg_forms
    forms = [f for sg in (_gold_test_segs(test) or []) for f in _seg_forms(sg)]
    return [m for m in items if m == test or any(f in str(m).lower() for f in forms)]


def gold_menu_items(test: str, items) -> list[str]:
    """Menu items that stand for a gold test: credited by the judge (`credited_menu_items`) and named
    for it (the item is the test, or one name contains the other). A segment form alone ("mri") credits
    an item about another organ; it does not make the test purchasable."""
    t = str(test)
    return [m for m in credited_menu_items(t, items) if m == t or t in str(m) or str(m) in t]


def unbuyable_gold_tests(raw, T: int | None = None) -> list[tuple[str, str]]:
    """(gold test, reason) for each gold test of an M2 case that no menu item makes purchasable:
    `not_on_menu` when no catalogue item stands for it (`gold_menu_items`), `unavailable` when every
    such item's purchase returns nothing for this case (sex filter and composite rules included)."""
    pc = getattr(raw, "prediction_context", None) or {}
    T = int(T if T is not None else pc.get("prediction_time_T") or 0)
    menu = m2_catalogue()
    out = []
    for t in gold_tests_of(raw):
        items = gold_menu_items(t, menu)
        if not items:
            out.append((t, "not_on_menu"))
            continue
        for m in items:
            try:
                if m2_synth_menu_target(None, raw, m, T):
                    break
            except Exception:                                    # noqa: BLE001  a purchase that raises reads nothing
                continue
        else:
            out.append((t, "unavailable"))
    return out


def for_sex(raw, ids: tuple[str, ...]) -> tuple[str, ...]:
    """Drop ids whose finding is `sex_specific` to the other sex (total testosterone and SHBG carry
    the female range only, so a value drawn for a man would read against the wrong band)."""
    sex = str(((getattr(raw, "user_profile", None) or {}).get("sex")) or "").upper()[:1]
    if not sex:
        return ids
    fd = _findings()
    return tuple(i for i in ids
                 if not (fd.get(i) or {}).get("sex_specific")
                 or str((fd.get(i) or {}).get("sex_specific")).upper()[:1] == sex)


@functools.lru_cache(maxsize=1)
def base_condition_findings() -> dict[str, list[dict]]:
    """`registry/base_condition_findings.yaml`: base condition -> its declared findings."""
    from haenv.interaction.gated import _cached_yaml
    from haenv.interaction.gated import _dr
    doc = _cached_yaml(_dr() / "registry" / "base_condition_findings.yaml") or {}
    return {str(k): list((v or {}).get("findings") or []) for k, v in (doc.get("bases") or {}).items()}


def base_condition_decl(raw, fid: str) -> dict | None:
    """The declaration a known base condition makes for `fid`, or None. A case-and-finding keyed
    draw against `penetrance` decides whether this case shows it (else the background reading)."""
    from haenv.interaction.gated import _u as _gu
    pb = ((getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {})
    names = [str(pb.get("disease") or "")] + [str(c) for c in (pb.get("comorbidities") or [])]
    table = base_condition_findings()
    cid = str(getattr(raw, "case_id", "CASE"))
    for n in names:
        for d in table.get(n, ()):
            if d.get("id") == fid and _gu(f"{cid}|base|{n}|{fid}") < float(d.get("penetrance", 1.0)):
                return {"id": fid, "direction": d.get("direction", "high"),
                        "magnitude": d.get("magnitude", "mild"), "base_condition": n}
    return None


class BudgetedMenu(list):
    """Public purchase allowance travels with both runtime and rebuilt menus."""
    def __init__(self, items, budget):
        super().__init__(items)
        self.budget = budget


def case_budget(raw) -> float | None:
    plan = (getattr(raw, "adjudication", None) or {}).get("m2_workup")
    if plan is None or not is_m2_case(raw):
        return None
    value = plan.get("budget")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("M2 workup requires a valid registered budget")
    return float(value)


def m2_menu_for(orig, sp, withheld: dict, case_id: str = "", m2_case: bool = False,
                budget: float | None = None) -> list[dict]:
    """`gated.menu_for` plus `m2_extra_items()` on M2 cases (`m2_case`: the raw case is an M2 case; the
    payload of an F0 bridge case carries no frame), shuffled by the same key."""
    items = orig(sp, withheld, case_id)
    if not (is_m2_framed(sp) or m2_case):
        return items
    from haenv_kernel.gatekeeper import COST                     # kernel
    have = {str(i["target"]) for i in items}
    for t, k in m2_extra_items().items():
        if t not in have:
            items.append({"target": t, "kind": k, "real": None, "is_test": True, "cost": COST.get(k, 5.0)})
    items.sort(key=lambda x: hashlib.sha256(f"{case_id}|{x['target']}".encode()).hexdigest())
    return BudgetedMenu(items, budget) if budget is not None else items


def _on_demand(raw, target, T: int):
    """The M2 purchase of one finding (`m2_synth_on_demand` over the original `gated.synth_on_demand`),
    whether or not the judge group's hooks are installed in this process (the generator's emission
    gates run without them)."""
    import haenv.interaction.gated as G
    return m2_synth_on_demand(G._base_synth_on_demand, raw, target, T)


def m2_synth_menu_target(orig, raw, target: str, T: int):
    """`gated.synth_menu_target` with the M2 `resolves_to` rows and the sex filter on M2 cases."""
    if not is_m2_case(raw):
        return orig(raw, target, T)
    import haenv.interaction.gated as G
    t = str(target or "").strip()
    if t in G.DECOY_SIGNALS:
        return None
    if t == "动态心电图":
        return None  # No ambulatory acquisition is modelled; a 10-second ECG is not Holter.
    ids = M2_RESOLVES_TO.get(t) or G._resolves_to().get(t)
    if not ids:
        pts = _on_demand(raw, target, T)
    else:
        ids = for_sex(raw, ids)
        if not ids:
            return None
        pts = []
        for fid in ids:
            p = DERIVED[fid](raw, pts, T) if fid in DERIVED else _on_demand(raw, fid, T)
            if not p:
                return None
            pts.extend(p)
    if t == "CT肺动脉造影(CTPA)":
        return pts  # this targeted angiographic item reports pulmonary arteries
    if t == "肝脏弹性成像(VCTE/必要时MRE)":
        return elastography_scope(pts)
    pts = with_region_findings(raw, pts, T)
    if t != "肾上腺CT/AVS":
        for p in pts or []:
            if p.get("finding_id") == "AdrenalImaging":
                p["name"] = pre_review_rules()["imaging"]["adrenal_structural_name"]
    return pts


def elastography_scope(pts):
    """Project the registered assessment onto the purchased VCTE/MRE acquisition.

    Background/etiology validation still happens in the original renderer; its
    serology, interview and other MRI acquisitions are not purchased here.
    """
    for p in pts or []:
        if p.get("finding_id") != "FibrosisAssessment":
            continue
        d = p.get("details")
        if not isinstance(d, dict) or not all(isinstance(d.get(k), dict) for k in ("vcte", "mre")):
            # A generic qualitative finding is not a modelled elastography
            # acquisition. Report it unavailable rather than invent readings.
            return None
        v, m = d["vcte"], d["mre"]
        p["name"] = "肝脏弹性成像(VCTE/必要时MRE)"
        p["details"] = {k: d[k] for k in ("vcte", "mre", "specialist_assessment", "histology",
                                          "histological_stage", "sources", "synthetic", "clinical_review") if k in d}
        p["value"] = (f'空腹{v["fasting_hours"]}小时,VCTE采用{v["probe"]}探头,'
                      f'有效测量{v["valid_measurements"]}/{v["attempted_measurements"]}次,'
                      f'肝硬度中位数{v["lsm_kpa"]} kPa,IQR/中位数{v["iqr_over_median"]:.2f};'
                      f'MRE肝硬度{m["lsm_kpa"]} kPa,波传播充分且无明显伪影。'
                      '无创读数需结合临床复评,不能单凭本检查确定病因或组织学分期。')
    return pts


def _fib4(raw, pts: list[dict], T: int) -> list[dict] | None:
    """FIB-4 = age x AST / (platelets x sqrt(ALT)), from the AST, ALT and platelet readings of the same
    purchase and the case's age band midpoint; no draw of its own."""
    from haenv.world.observations.findings_render import _band
    by = {p.get("finding_id"): p.get("value") for p in pts}
    val = fib4_value(_age_mid(raw), by.get("AST"), by.get("ALT"), by.get("PLT"))
    if val is None:
        return None
    fspec = _findings()["FIB4"]
    blo, bhi = _band(fspec)
    return [{"ts": T, "value": val, "unit": fspec.get("unit", ""), "ref_low": blo, "ref_high": bhi,
             "flag": "H" if val > bhi else "", "finding_id": "FIB4", "name": fspec.get("name_cn", "FIB4")}]


def _lipid_tc(raw, pts: list[dict], T: int) -> list[dict] | None:
    """Total cholesterol of a lipid panel from the same draw's LDL-C, HDL-C and TG (mmol/L):
    TC = LDL + HDL + TG/2.2 - rho x LDL, rho the case's real Friedewald residual
    (`findings_render.tc_friedewald_residual`, fitted on TG < 4.5), TC >= LDL + HDL. At TG >= 4.5 the
    residual is outside its fitted domain and is not applied, and the LDL-C reading is marked as
    measured directly (Friedewald does not apply there). LDL-C is always the measured reading."""
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _round
    from haenv.world.observations.findings_render import _step
    from haenv.world.observations.findings_render import tc_friedewald_residual
    from haenv.world.clinical.relations import FRIEDEWALD_TG_DIVISOR
    from haenv.world.clinical.relations import FRIEDEWALD_TG_MAX_MMOL
    by = {p.get("finding_id"): p.get("value") for p in pts}
    try:
        ldl, hdl, tg = (float(by[k]) for k in ("LDL", "HDL", "TG"))
    except (KeyError, TypeError, ValueError):
        return None
    cid = str(getattr(raw, "case_id", "CASE"))
    rho = tc_friedewald_residual(cid, ldl, tg) if tg < FRIEDEWALD_TG_MAX_MMOL else 0.0
    tc = _round(ldl + hdl + tg / FRIEDEWALD_TG_DIVISOR - rho * ldl, "TC")
    while tc < ldl + hdl:
        tc = _round(tc + _step("TC", tc), "TC")
    if tg >= FRIEDEWALD_TG_MAX_MMOL:
        for p in pts:
            if p.get("finding_id") == "LDL":
                p["method"] = "直接测定"
    fspec = _sex_spec(raw, "TC", _findings()["TC"]) or _findings()["TC"]
    lo, hi = _band(fspec)
    import haenv.interaction.gated as G
    flag = "H" if tc > hi else ("L" if tc < lo else "NORMAL")
    name, sym = G._report_face(cid, "TC", flag)
    return [{"ts": T, "value": tc, "unit": fspec.get("unit", ""), "ref_low": lo, "ref_high": hi,
             "flag": sym, "finding_id": "TC", "name": name or fspec.get("name_cn", "TC")}]


def tc_point(raw, T: int) -> list[dict] | None:
    """The case's total cholesterol on day T: derived from its own LDL-C, HDL-C and TG of that day."""
    pts = []
    for f in ("TG", "HDL", "LDL"):
        p = _on_demand(raw, f, T)
        if not p:
            return None
        pts.extend(p)
    return _lipid_tc(raw, pts, T)


#: Items read from other items of the same purchase (no draw of their own).
DERIVED = {"FIB4": lambda raw, pts, T: _fib4(raw, pts, T), "TC": lambda raw, pts, T: _lipid_tc(raw, pts, T)}

#: Imaging and the anatomy it shows. An imaging purchase reports every positive imaging finding the case's
#: condition declares about a region the purchased study shows (an abdominal CT of a pancreatic cancer shows
#: the pancreatic mass; it does not read only "adrenal: negative"). `IMAGING_SHOWS`: study -> regions in its
#: field; `IMAGING_ABOUT`: finding -> the region its result is about. Functional or targeted studies (MIBG,
#: uptake scan, gastric emptying, quantitative liver iron, adrenal vein sampling's lateralisation) are not
#: listed, and neither is the sacroiliac joint: MRI shows sacroiliitis that a radiograph may not.
IMAGING_SHOWS: dict[str, tuple[str, ...]] = {
    "PancreaticMassImaging": ("pancreas",), "InsulinomaLocalization": ("pancreas",),
    "PancreaticImaging": ("pancreas",), "AbdominalImaging": ("adrenal", "pancreas", "kidney"),
    "AdrenalImaging": ("adrenal",), "PitAdrenalImaging": ("pituitary", "adrenal"),
    "PituitaryMRI": ("pituitary",),
    "RenalUltrasound": ("kidney",), "KidneyUSCalcium": ("kidney",),
    "ChestHRCT": ("chest",), "ChestFocalOpacity": ("chest",), "CTPA": ("chest",),
}
IMAGING_ABOUT: dict[str, tuple[str, ...]] = {
    "PancreaticMassImaging": ("pancreas",), "InsulinomaLocalization": ("pancreas",),
    "PancreaticImaging": ("pancreas",), "AbdominalImaging": ("adrenal",),
    "AdrenalImaging": ("adrenal",), "PitAdrenalImaging": ("pituitary", "adrenal"),
    "PituitaryMRI": ("pituitary",),
    "RenalUltrasound": ("kidney",), "KidneyUSCalcium": ("kidney",),
    "ChestHRCT": ("chest",), "ChestFocalOpacity": ("chest",), "CTPA": ("chest",),
}


def declared_positive_imaging(raw) -> list[str]:
    """Imaging findings the case's condition declares positive (none on an insufficient case)."""
    from haenv.world.overlay import spec_id_of
    from haenv.resources.registry import condition_findings_for_case
    if ((getattr(raw, "adjudication", None) or {}).get("ddx") or {}).get("insufficient"):
        return []
    spec_id, _ = spec_id_of(raw)
    cf = condition_findings_for_case(_findings()).get(spec_id, {}).get("findings", []) if spec_id else []
    return [str(it["id"]) for it in cf if it.get("direction") == "positive" and it.get("id") in IMAGING_ABOUT]


def with_region_findings(raw, pts, T: int):
    """`pts` plus the case's positive imaging findings in the regions the purchased study shows."""
    if not pts:
        return pts
    have = {p.get("finding_id") for p in pts if isinstance(p, dict)}
    regions = {r for f in have for r in IMAGING_SHOWS.get(str(f), ())}
    if not regions:
        return pts
    out = list(pts)
    for fid in declared_positive_imaging(raw):
        if fid not in have and regions & set(IMAGING_ABOUT[fid]):
            out.extend(_on_demand(raw, fid, T) or ())
    return out


def _base_m2_synth_on_demand(orig, raw, target: str, T: int):
    """`gated.synth_on_demand`; on M2 frames a known base condition shapes items the hidden spec
    does not declare (`registry/base_condition_findings.yaml`). Same rendering as the original."""
    if not is_m2_case(raw) or not target or not isinstance(target, str):
        return orig(raw, target, T)
    import haenv.interaction.gated as G
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _value
    from haenv.world.overlay import spec_id_of
    from haenv.resources.registry import condition_findings_for_case
    fx = _findings()
    fid = G._resolve_target(target.strip(), fx)
    if not fid or not G.has_reading_model(fx[fid]):
        return orig(raw, target, T)
    if fid == "FibrosisAssessment":
        spec_id, _ = spec_id_of(raw)
        if spec_id == "HD-UNI-13":
            from haenv.world.clinical.clinical_evidence import ClinicalEvidenceUnavailable
            from haenv.world.clinical.clinical_evidence import validate_fibrosis_background
            profile = condition_findings_for_case(fx)[spec_id]
            pb = (getattr(raw, "latent_premise", None) or {}).get("patient_basics")
            if not isinstance(pb, dict) or not isinstance(pb.get("comorbidities"), list):
                raise ClinicalEvidenceUnavailable("FibrosisAssessment: complete world background required")
            validate_fibrosis_background(profile, [pb.get("disease"), *pb["comorbidities"]])
            # These are consistency checks against an explicit no-acute-hepatitis
            # design; low readings do not themselves establish that exclusion.
            for enzyme in ("ALT", "AST"):
                points = _on_demand(raw, enzyme, T)
                if not points or any(not isinstance(p.get("value"), (int, float))
                                     or not p.get("ref_high") or p["value"] >= 5*p["ref_high"] for p in points):
                    raise ClinicalEvidenceUnavailable("FibrosisAssessment: transaminase quality context unavailable/conflicting")
            return orig(raw, target, T)
    if fid == "FIB4":
        return fib4_point(raw, T)
    if fid == "TC":
        return tc_point(raw, T)
    fspec = fx[fid]
    sx = _sex_spec(raw, fid, fspec)
    on_stream = G._named_on_stream(raw, fid, fspec, T)
    if on_stream is not None:
        return _reflag(raw, _same_day(raw, fid, T, on_stream), sx)
    spec_id, _ = spec_id_of(raw)
    cf = condition_findings_for_case(fx).get(spec_id, {}).get("findings", []) if spec_id else []
    decl = next((it for it in cf if it.get("id") == fid), None)
    declared_dir = (decl or {}).get("direction")
    if ((getattr(raw, "adjudication", None) or {}).get("ddx") or {}).get("insufficient"):
        decl = None
    bd = base_condition_decl(raw, fid) if decl is None else None
    if fid in KIDNEY_PERSISTENCE and bd is not None:
        bd = bd if any(_reads_abnormal(raw, f, T) for f in KIDNEY_PERSISTENCE[fid]) else None
    if decl is None:
        said = claimed_direction(raw, fid)
        if said:
            bd = {"id": fid, "direction": said, "magnitude": "mild", "claimed": True}
    if decl is None and bd is None and fid in BACKGROUND_DRAW:
        return _background_point(raw, fid, sx or fspec, T)
    if sx is not None and not fspec.get("qualitative") and declared_dir not in ("positive", "negative"):
        from haenv.world.clinical.clinical_evidence import _SUPPORTED
        if fid not in _SUPPORTED.get(str(spec_id or ""), ()):
            d = decl or bd
            return _numeric_point(raw, fid, sx, T, d.get("direction", "high") if d else "normal",
                                  d.get("magnitude", "moderate") if d else None)
    if decl is not None:
        return orig(raw, target, T)
    if bd is None:
        return orig(raw, target, T)
    case_id = getattr(raw, "case_id", "CASE")
    if bool(fspec.get("qualitative")) or declared_dir in ("positive", "negative"):
        pos = bd.get("direction") != "negative"
        return [{"ts": T, "value": "阳性 / 异常提示病理性改变" if pos else "阴性 / 未见明显异常",
                 "flag": "POSITIVE" if pos else "NEGATIVE", "finding_id": fid,
                 "name": fspec.get("name_cn", fid)}]
    val = _value(fspec, bd.get("direction", "high"), bd.get("magnitude", "moderate"), case_id, fid, 0)
    lo, hi = _band(fspec)
    flag = ("H" if val > hi else ("L" if val < lo else "NORMAL")) if isinstance(val, (int, float)) else "NORMAL"
    name, sym = G._report_face(str(case_id), fid, flag)
    return [{"ts": T, "value": val, "unit": fspec.get("unit", ""), "ref_low": lo, "ref_high": hi,
             "flag": sym, "finding_id": fid, "name": name or fspec.get("name_cn", fid)}]


#: Findings whose reading, on a case with the base condition, is the persistence of another finding's
#: abnormality: chronic kidney disease on the record is positive for "abnormal for at least three months"
#: only when the case's own kidney readings are abnormal (KDIGO 2024 Table 1: eGFR < 60 or albuminuria,
#: present >= 3 months; the record's CKD diagnosis supplies the duration).
KIDNEY_PERSISTENCE = {"CKDPersistence": ("Cr", "UACR")}


def _reads_abnormal(raw, fid: str, T: int) -> bool:
    """The case's purchase of `fid` on day T reads outside its printed band."""
    import haenv.interaction.gated as G
    for p in _on_demand(raw, fid, T) or ():
        v, lo, hi = p.get("value"), p.get("ref_low"), p.get("ref_high")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and (
                (hi is not None and v > hi) or (lo is not None and v < lo)):
            return True
    return False


#: Lab results the patient's own words state on the visible record: finding id -> (direction, pattern,
#: negation). A purchase of that finding reads the stated direction when neither the hidden spec nor a
#: known base condition declares it, so the record and the report say one thing; `lab_claim_conflicts`
#: is the emission gate for every claim (declared ones included).
LAB_CLAIMS: dict[str, tuple[str, str, str]] = {
    "UrineKetone": ("positive", r"酮体|尿酮", r"阴性|没有酮|无酮"),
    # urine ketones found: the serum ketone of the same day is raised too (a dipstick positive needs
    # beta-hydroxybutyrate above the 0.3 mmol/L band; not checked against a source)
    "BetaOHB": ("high", r"酮体|尿酮", r"阴性|没有酮|无酮"),
    "K": ("low", r"血钾.{0,4}(低|只有|偏低)", r"正常"),
    "FBG": ("high", r"空腹血糖.{0,3}(\d|偏高|高)", r"正常"),
}
#: A claim about a panel (the patient names the panel, not the item): any of these reading in the
#: stated direction makes the claim true (gate only; no single item is forced).
LAB_PANEL_CLAIMS: dict[str, tuple[tuple[str, ...], str, str, str]] = {
    "血脂": (("LDL", "TG", "TC"), "high", r"血脂.{0,2}(偏高|高)", r"正常"),
}


def visible_claim_text(raw) -> list[str]:
    """Patient words the solver sees by day T: symptom and context of the ledger, and the frame scene."""
    pc = getattr(raw, "prediction_context", None) or {}
    T = int(pc.get("prediction_time_T") or 10 ** 9)
    out = [str((pc.get("m2_frame") or {}).get("scene") or "")]
    for e in getattr(raw, "evidence_ledger", None) or ():
        if isinstance(e, dict) and int(e.get("source_timestamp", 0) or 0) <= T:
            out += [str(e.get("symptom") or ""), str(e.get("context") or "")]
    return [x for x in out if x]


def _claims_in(texts, pattern: str, negation: str) -> bool:
    import re
    return any(re.search(pattern, t) and not re.search(negation, t) for t in texts)


def claimed_direction(raw, fid: str) -> str | None:
    """The direction the visible record states for `fid` (`LAB_CLAIMS`), or None."""
    c = LAB_CLAIMS.get(fid)
    return c[0] if c and _claims_in(visible_claim_text(raw), c[1], c[2]) else None


def _reads_direction(pts, direction: str) -> bool:
    for p in pts or ():
        v, lo, hi = p.get("value"), p.get("ref_low"), p.get("ref_high")
        if direction == "positive" and str(p.get("flag")) == "POSITIVE":
            return True
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if direction == "high" and hi is not None and v > hi:
                return True
            if direction == "low" and lo is not None and v < lo:
                return True
    return False


def lab_claim_conflicts(raw, T: int | None = None) -> list[str]:
    """Lab results the visible record states that a purchase on day T contradicts (emission gate)."""
    import haenv.interaction.gated as G
    pc = getattr(raw, "prediction_context", None) or {}
    T = int(T if T is not None else pc.get("prediction_time_T") or 0)
    texts = visible_claim_text(raw)
    out = []
    for fid, (d, pat, neg) in LAB_CLAIMS.items():
        if _claims_in(texts, pat, neg) and not _reads_direction(_on_demand(raw, fid, T), d):
            out.append(f"{fid} {d}")
    for name, (fids, d, pat, neg) in LAB_PANEL_CLAIMS.items():
        if _claims_in(texts, pat, neg) and not any(_reads_direction(_on_demand(raw, f, T), d) for f in fids):
            out.append(f"{name} {d}")
    return out


def implausible_readings(raw, T: int | None = None) -> list[str]:
    """Declared low findings whose day-T purchase reads below the finding's plausible floor
    (`findings_render.plausible_floor`: the lowest level a patient population with that deficit shows),
    e.g. a vitamin B12 of 10 pg/mL. Emission gate: such a case is not emitted."""
    from haenv.world.observations.findings_render import plausible_floor
    from haenv.world.overlay import spec_id_of
    from haenv.resources.registry import condition_findings_for_case
    pc = getattr(raw, "prediction_context", None) or {}
    T = int(T if T is not None else pc.get("prediction_time_T") or 0)
    spec_id, _ = spec_id_of(raw)
    decl = list(condition_findings_for_case(_findings()).get(spec_id, {}).get("findings", []) if spec_id else [])
    pb = ((getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {})
    for n in [str(pb.get("disease") or "")] + [str(c) for c in (pb.get("comorbidities") or [])]:
        decl += base_condition_findings().get(n, [])
    out = []
    for fid in sorted({str(d.get("id")) for d in decl if d.get("direction") == "low"}):
        fspec = _findings().get(fid)
        if not fspec or fspec.get("qualitative"):
            continue
        floor = plausible_floor(_sex_spec(raw, fid, fspec) or fspec)
        if floor is None:
            continue
        for p in _on_demand(raw, fid, T) or ():
            v = p.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v < floor:
                out.append(f"{fid} {v} < {floor}")
    return out


#: Background (not declared) readings drawn from the general-population table rather than the band:
#: AST and ALT from the NHANES quantiles with their benign copula (`panel_background.yaml`,
#: `panel_copula.yaml`, ALT-AST normal-score correlation 0.70), so FIB-4 computed from them follows the
#: population; platelets normal over the 95 % reference band. Checked against Sugiyama et al., BMC
#: Gastroenterol 2022, doi:10.1186/s12876-022-02290-1, Table 1 (non-drinkers without fatty liver).
BACKGROUND_DRAW = ("AST", "ALT", "PLT")


@functools.lru_cache(maxsize=1)
def _population_quantiles() -> dict:
    from haenv.resources.regpath import load_registry
    doc = load_registry("panel_background.yaml") or {}
    return {fid: {sx: tuple(float(x) for x in e[sx]["q"]) for sx in ("F", "M", "all") if sx in e}
            for fid, e in (doc.get("background") or {}).items()}


def _background_point(raw, fid: str, spec: dict, T: int) -> list[dict]:
    import haenv.interaction.gated as G
    import haenv.world.generation.rng as rng
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _copula_index
    from haenv.world.observations.findings_render import _copula_u
    from haenv.world.observations.findings_render import _ppf
    from haenv.world.observations.findings_render import _round
    case_id = str(getattr(raw, "case_id", "CASE"))
    sex = str(((getattr(raw, "user_profile", None) or {}).get("sex")) or "").upper()[:1] or "F"
    lo, hi = _band(spec)
    q = (_population_quantiles().get(fid) or {})
    q = q.get(sex) or q.get("all")
    if q and fid in _copula_index():
        x = _copula_u(case_id, fid, 0, sex) * (len(q) - 1)
        i = min(int(x), len(q) - 2)
        val = q[i] + (x - i) * (q[i + 1] - q[i])
    else:
        val = (lo + hi) / 2 + (hi - lo) / (2 * 1.959964) * _ppf(rng.unit(case_id, "finding", fid, "0"))
    val = _round(max(val, 0.0), fid)
    flag = "H" if val > hi else ("L" if val < lo else "NORMAL")
    name, sym = G._report_face(case_id, fid, flag)
    return [{"ts": T, "value": val, "unit": spec.get("unit", ""), "ref_low": lo, "ref_high": hi,
             "flag": sym, "finding_id": fid, "name": name or spec.get("name_cn", fid)}]


def _age_mid(raw) -> float | None:
    age = (getattr(raw, "user_profile", None) or {}).get("age_range")
    try:
        lo, hi = (int(x) for x in str(age).split("-"))
    except (TypeError, ValueError):
        return None
    return (lo + hi) / 2


def fib4_value(age: float | None, ast, alt, plt) -> float | None:
    """FIB-4 = age x AST / (platelets x sqrt(ALT)) (Sterling et al., Hepatology 2006)."""
    try:
        return round(float(age) * float(ast) / (float(plt) * float(alt) ** 0.5), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def fib4_point(raw, T: int) -> list[dict] | None:
    """The case's FIB-4 on day T: computed from its own AST, ALT and platelet readings of that day."""
    import haenv.interaction.gated as G
    pts = []
    for f in ("AST", "ALT", "PLT"):
        p = _on_demand(raw, f, T)
        if not p:
            return None
        pts.extend(p)
    return _fib4(raw, pts, T)


def fib4_world(raw, longitudinal: dict, T: int) -> dict:
    """The queried world's FIB-4 stream: each point is FIB-4 of the same day's AST and ALT draws and the
    case's platelet count (a point without both draws leaves), so a stream point and a purchase read one
    formula on one set of inputs. No FIB4 stream => unchanged."""
    src = (longitudinal or {}).get("FIB4")
    if not src:
        return longitudinal
    import haenv.interaction.gated as G
    plt = next((p.get("value") for p in _on_demand(raw, "PLT", T) or ()), None)
    by = {st: {int(p["ts"]): p["value"] for p in longitudinal.get(st) or () if isinstance(p, dict)}
          for st in ("AST", "ALT")}
    age = _age_mid(raw)
    out = []
    for p in src:
        if not isinstance(p, dict):
            continue
        ts = int(p.get("ts", -1))
        v = fib4_value(age, by["AST"].get(ts), by["ALT"].get(ts), plt)
        if v is not None:
            out.append({**p, "value": v})
    return {**longitudinal, "FIB4": out}


def m2_world(raw, longitudinal: dict, plan: dict | None, T: int) -> dict:
    """The world an M2 solver queries: the F3 follow-up (`f3_world`), then FIB-4 from its inputs."""
    return fib4_world(raw, f3_world(longitudinal, plan, T), T)


def _sex_spec(raw, fid: str, fspec: dict) -> dict | None:
    """The finding's sex-specific reference band (`findings_render.SEX_REF`, the band the record
    renderer uses) for the case's sex, or None when the finding has none."""
    from haenv.world.observations.findings_render import SEX_REF
    from haenv.world.observations.findings_render import with_sex_ref
    sex = str(((getattr(raw, "user_profile", None) or {}).get("sex")) or "").upper()[:1]
    if fid not in SEX_REF or sex not in SEX_REF[fid]:
        return None
    return with_sex_ref({fid: fspec}, sex)[fid]


def _numeric_point(raw, fid: str, spec: dict, T: int, direction: str, magnitude) -> list[dict]:
    import haenv.interaction.gated as G
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _value
    case_id = getattr(raw, "case_id", "CASE")
    val = _value(spec, direction, magnitude, case_id, fid, 0)
    lo, hi = _band(spec)
    flag = ("H" if val > hi else ("L" if val < lo else "NORMAL")) if isinstance(val, (int, float)) else "NORMAL"
    name, sym = G._report_face(str(case_id), fid, flag)
    return [{"ts": T, "value": val, "unit": spec.get("unit", ""), "ref_low": lo, "ref_high": hi,
             "flag": sym, "finding_id": fid, "name": name or spec.get("name_cn", fid)}]


def _same_day(raw, fid: str, T: int, pts: list[dict]) -> list[dict]:
    """A named test on day T when the case's stream already has a draw on day T reads that draw
    (one sample, one value); otherwise the new draw `gated._named_on_stream` made."""
    from haenv.world.observations.findings_render import _TS_TO_PANEL
    stream = {v: k for k, v in _TS_TO_PANEL.items()}.get(fid)
    same = [q for q in ((getattr(raw, "longitudinal_data", None) or {}).get(stream) or [])
            if isinstance(q, dict) and int(q.get("ts", -1)) == int(T)
            and isinstance(q.get("value"), (int, float)) and not isinstance(q.get("value"), bool)]
    if not same or not pts:
        return pts
    return [{**pts[0], "value": same[-1]["value"]}] + list(pts[1:])


def _reflag(raw, pts: list[dict], spec: dict | None) -> list[dict]:
    """Re-read the flag of a stream-backed point against `spec`'s band (the sex band when given)."""
    import haenv.interaction.gated as G
    from haenv.world.observations.findings_render import _band
    out = []
    for p in pts or ():
        v = p.get("value")
        if spec is not None:
            lo, hi = _band(spec)
            p = {**p, "ref_low": lo, "ref_high": hi}
        lo, hi = p.get("ref_low"), p.get("ref_high")
        if isinstance(v, (int, float)):
            flag = "H" if (hi is not None and v > hi) else ("L" if (lo is not None and v < lo) else "NORMAL")
            name, sym = G._report_face(str(getattr(raw, "case_id", "CASE")), p.get("finding_id"), flag)
            p = {**p, "flag": sym, "name": name or p.get("name")}
        out.append(p)
    return out


def components(spec_id: str) -> list[str]:
    from haenv.world.overlay import HAENV_COMORBID_PAIRS
    cfg = HAENV_COMORBID_PAIRS.get(spec_id)
    return list(cfg["pair"]) if cfg else [spec_id]


def _key_findings_of(component: str) -> list[str]:
    prof = (_profiles().get(component) or {}).get("findings") or ()
    reach = menu_target_of()
    # A declared differential-exclusion check can be normal or negative.
    # Role, not abnormality, determines whether it is a diagnostic key check.
    cand = [p for p in prof if p.get("role") == "confirmatory" and p.get("id") in reach]
    cand.sort(key=lambda p: (_MAG_ORDER.get(p.get("magnitude"), 3), str(p["id"])))
    return [str(p["id"]) for p in cand[:KEY_PER_THREAD]]


#: Key signals per declared line of a single-spec comorbidity (revision R-18). `components()` gives
#: such a spec one component for both lines, so the confirmatory rule hands every line the same
#: signals -- for JD-CKM the kidney line's creatinine would stand as the glucose line's key signal,
#: and the glucose line has no confirmatory finding (HbA1c/FBG are declared `screening`). The
#: glucose line takes the diagnostic glycaemic pair, the kidney line creatinine and UACR (the
#: explain-away table's E18 key signals).
THREAD_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("JD-CKM", "糖代谢线"): ("HbA1c", "FBG"),
    ("JD-CKM", "肾脏线"): ("Cr", "UACR"),
}


def confirmatory_all(component: str) -> list[str]:
    """Every declared confirmation/exclusion check, reachable or not (audit MENU-1).

    A normal thyroid result can exclude a competing cause; it does not confirm a
    statin causal mechanism. Routine normal screening tests do not enter this set.
    """
    prof = (_profiles().get(component) or {}).get("findings") or ()
    return [str(p["id"]) for p in prof if p.get("role") == "confirmatory"]


def threads_of(ddx: dict | None) -> list[dict]:
    """Gold lines with aliases and key signals; [] for independent and insufficient-tier cases."""
    ddx = ddx or {}
    if ddx.get("insufficient") or ddx.get("join_gold") == "independent":
        return []
    sid = str(ddx.get("spec_id") or "")
    comps = components(sid)
    declared = [t for t in (ddx.get("threads") or []) if t.get("aliases")]
    if ddx.get("join_gold") == "comorbidity" and declared:
        out = []
        for i, t in enumerate(declared):
            comp = comps[i] if i < len(comps) else sid
            name = t.get("name") or f"line{i + 1}"
            over = THREAD_KEYS.get((sid, name))
            keys = ([f for f in over if f in menu_target_of()][:KEY_PER_THREAD] if over
                    else _key_findings_of(comp))
            out.append({"name": name, "aliases": list(t["aliases"]),
                        "component": comp, "key_signals": keys})
        from haenv.m2.history import replacements
        return [replacements(ddx, t) for t in out]
    from haenv.m2.history import replacements
    return [replacements(ddx, {"name": ddx.get("diagnosis") or sid, "aliases": list(ddx.get("aliases") or []),
             "component": sid, "key_signals": _key_findings_of(sid)})]


def discriminating_of(ddx: dict | None, exclude=()) -> list[str]:
    from haenv.world.overlay import rivals_for
    ddx = ddx or {}
    out: list[str] = []
    for r in (rivals_for(str(ddx.get("spec_id") or ""), ddx) or ()):
        f = str(((r or {}).get("discriminator_finding") or {}).get("finding") or "").strip()
        if f and f not in out and f not in exclude and f in menu_target_of():
            out.append(f)
    return out


def rival_names(ddx: dict | None) -> list[dict]:
    from haenv.world.overlay import rivals_for
    ddx = ddx or {}
    return [{"name": r["name"], "aliases": list(r.get("aliases") or [])}
            for r in (rivals_for(str(ddx.get("spec_id") or ""), ddx) or ())]


def spectrum_ids(spec_id: str) -> set[str]:
    from haenv.resources.registry import condition_findings_for_case
    prof = (condition_findings_for_case(_findings()).get(spec_id) or {}).get("findings") or ()
    return {str(p["id"]) for p in prof}


# ------------------------------------------------------------------ distractor
#: Registered benign changes that read abnormal (spec 2.2 item 3). `points_to` is the diagnosis a
#: reader who takes the reading at face value would add; `note` is the specimen note the reading
#: is printed with (the clinical cue a careful reader uses).
DISTRACTORS: tuple[dict, ...] = (
    {"kind": "artifact_hemolysis", "fid": "K", "direction": "high",
     "note": ("标本轻度溶血", "标本溶血(+)", "检验科备注:标本有轻度溶血"), "source": "registry/artifact_rates.yaml (溶血 -> 血钾假性升高)",
     "points_to": {"name": "高钾血症", "aliases": ["高钾", "高钾血症", "hyperkalemia"]}},
    {"kind": "artifact_nonfasting", "fid": "TG", "direction": "high",
     "note": ("餐后约2小时采血", "非空腹", "患者自述采血前吃过早饭"), "source": "registry/artifact_rates.yaml (非空腹 -> TG 升高)",
     "points_to": {"name": "高甘油三酯血症", "aliases": ["高甘油三酯", "高甘油三酯血症", "hypertriglyceridemia"]}},
    {"kind": "artifact_hemolysis_ldh", "fid": "LDH", "direction": "high",
     "note": ("标本轻度溶血", "溶血标本", "检验科备注:标本溶血,建议复查"), "source": "hemolysis releases red-cell LDH (pre-analytical artifact)",
     "points_to": {"name": "溶血性贫血", "aliases": ["溶血性贫血", "hemolytic anemia"]}},
    {"kind": "transient_inflammation", "fid": "CRP", "direction": "high",
     "note": ("近一周上呼吸道感染", "感冒后第5天", "上周发热咽痛,已自行好转"), "source": "transient acute-phase rise after a viral URTI",
     "points_to": {"name": "感染/炎症性疾病", "aliases": ["感染", "炎症", "infection"]}},
    # Specialty-test false positives pointing at a rival diagnosis (spec 2.2 item 3, lookalike type).
    {"kind": "preanalytic_metanephrine", "fid": "Metanephrine", "direction": "high",
     "note": ("采血前饮咖啡、坐位采血", "坐位采血", "采血前一小时喝过两杯咖啡"), "source": "caffeine / seated sampling raise plasma free metanephrines",
     "points_to": {"name": "嗜铬细胞瘤", "aliases": ["嗜铬", "嗜铬细胞瘤", "pheochromocytoma"]}},
    {"kind": "preanalytic_saliva_cortisol", "fid": "Cortisol_saliva_midnight", "direction": "high",
     "note": ("采样前两小时内吸烟", "采样前吸烟", "采样当晚刷牙时牙龈出血"), "source": "smoking / gum bleeding before late-night salivary sampling",
     "points_to": {"name": "库欣综合征", "aliases": ["库欣", "cushing"]}},
    {"kind": "recovery_tsh", "fid": "TSH", "direction": "high",
     "note": ("上月急性病后恢复期", "肺炎痊愈后两周", "近期住院后恢复期复查"), "source": "transient TSH rise in the recovery phase of non-thyroidal illness",
     "points_to": {"name": "甲状腺功能减退", "aliases": ["甲减", "甲状腺功能减退", "hypothyroid"]}},
    {"kind": "vitd_secondary_pth", "fid": "PTH", "direction": "high",
     "note": ("冬季日照少、维生素D摄入不足", "25-OH维生素D偏低", "长期室内工作、少晒太阳"), "source": "secondary PTH rise with vitamin D insufficiency",
     "points_to": {"name": "原发性甲状旁腺功能亢进", "aliases": ["甲旁亢", "甲状旁腺功能亢进", "hyperparathyroidism"]}},
    # Low-direction and qualitative benign changes, so direction and the qualitative form carry no
    # role information either (key signals include low and qualitative findings).
    {"kind": "pseudohyponatremia", "fid": "Na", "direction": "low",
     "note": ("标本明显脂血(高甘油三酯)", "脂血标本", "检验科备注:标本乳糜样"), "source": "pseudohyponatremia in a lipaemic sample (indirect ISE)",
     "points_to": {"name": "抗利尿激素分泌失调综合征", "aliases": ["低钠血症", "SIADH", "抗利尿激素分泌失调"]}},
    {"kind": "dilution_hb", "fid": "Hb", "direction": "low",
     "note": ("输液同侧手臂采血", "输液中采血", "急诊补液后即刻采血"), "source": "dilution when drawn above an infusion line",
     "points_to": {"name": "贫血", "aliases": ["贫血", "anemia"]}},
    {"kind": "low_titer_ana", "fid": "ANA", "direction": "positive", "qualitative": True,
     "note": ("低滴度1:80,健康人亦常见", "1:80", "滴度1:80,核型斑点型"), "source": "low-titre ANA in healthy adults",
     "points_to": {"name": "系统性红斑狼疮", "aliases": ["狼疮", "红斑狼疮", "SLE", "lupus"]}},
    {"kind": "nonfasting_insulin", "fid": "Insulin", "direction": "high",
     "note": ("采血前已进食早餐", "非空腹", "采血前喝了含糖饮料"), "source": "post-prandial insulin read as fasting",
     "points_to": {"name": "胰岛素抵抗", "aliases": ["胰岛素抵抗", "insulin resistance"]}},
)
#: The distractor's magnitude is drawn per case from the same tiers the key signals are rendered
#: at, so its out-of-band distance carries no role information (RV-6).
DISTRACTOR_MARKED_P = 0.55
#: Cumulative magnitude mix. Key signals are 69 % marked, but their out-of-band distance has a low
#: tail (low-direction findings with a wide band: B12, renin) that "marked" distractors lack; a
#: mild share reproduces it (RV-6 readings in the M2 implementation report).
DISTRACTOR_MAG_CUM = (("marked", DISTRACTOR_MARKED_P), ("moderate", 0.80), ("mild", 1.01))


#: Direction mix of the key signals in the pack (high 108, low 49, positive 8 of 165): the
#: distractor's direction class is drawn with these weights, so direction carries no role (RV-6).
DISTRACTOR_DIRECTION_P = (("high", 0.65), ("low", 0.30), ("positive", 0.05))


#: F3 RCV distractor (spec 3.2 F3 "RCV-based distractor generator"; 2.2 item 3 "within-RCV repeat
#: fluctuation"): what a reader taking a within-RCV repeat at face value would conclude.
RCV_POINTS_TO = {
    "HbA1c": {"name": "血糖控制恶化", "aliases": ["血糖控制恶化", "血糖失控", "糖尿病控制不佳"]},
    "fasting_glucose": {"name": "血糖控制恶化", "aliases": ["血糖控制恶化", "血糖失控", "糖尿病控制不佳"]},
    "LDL": {"name": "血脂异常加重", "aliases": ["血脂异常加重", "高胆固醇血症", "高脂血症"]},
    "triglycerides": {"name": "高甘油三酯血症", "aliases": ["高甘油三酯", "高甘油三酯血症"]},
    "ALT": {"name": "肝损伤", "aliases": ["肝损伤", "转氨酶升高", "肝炎"]},
    "AST": {"name": "肝损伤", "aliases": ["肝损伤", "转氨酶升高", "肝炎"]},
}
#: Share of the 95 % RCV the repeat moves by, drawn uniformly per case in this range: inside the
#: RCV by construction (a change a careful reader does not act on).
RCV_SHARE = (0.45, 0.9)
#: F3 push-block distractor from the RCV generator (R-23). See R-27 for why it is off.
F3_RCV_PUSH = False
#: The largest share of the RCV the repeat may move to reach a reference bound (still inside it).
RCV_CROSS_MAX = 0.95
_STREAM_FID = {"fasting_glucose": "FBG", "triglycerides": "TG", "ALT": "ALT", "AST": "AST",
               "HbA1c": "HbA1c", "LDL": "LDL"}


def rcv_distractor_for(case_id: str, ddx: dict, meta: dict, taken: set[str]) -> dict | None:
    """F3: a repeat of one of the case's live lab streams on day T whose change from the last draw
    stays inside the 95 % RCV (`F3_STREAMS`, EFLM total CV). The followed indicator is avoided when
    another live stream exists, so the round-1 RCV question keeps its own world readings."""
    from haenv.scoring.tracks import alias_hit_asserted
    gold_names = [str(ddx.get("diagnosis") or "")] + [str(a) for a in ddx.get("aliases") or []]
    spec = spectrum_ids(str(ddx.get("spec_id") or ""))
    live = [st for st in F3_STREAMS if st in (meta.get("clinical") or {})
            and _STREAM_FID.get(st) not in taken and _STREAM_FID.get(st) not in spec]
    followed = f3_stream(meta)
    pref = [st for st in live if st != followed] or live
    ok = [st for st in pref if not any(alias_hit_asserted(a, gold_names)
                                        for a in [RCV_POINTS_TO[st]["name"]] + RCV_POINTS_TO[st]["aliases"])]
    if not ok:
        return None
    st = sorted(ok, key=lambda x: _h(case_id, "rcv-dis", x))[0]
    share = RCV_SHARE[0] + (RCV_SHARE[1] - RCV_SHARE[0]) * _u(case_id, "rcv-share")
    return {"kind": "rcv_fluctuation", "fid": _STREAM_FID[st], "stream": st, "rcv_share": round(share, 4),
            "note": ("复查,与上次同一实验室", "门诊复查", "随访复查"),
            "source": "within-RCV repeat (EFLM total CV, physio_streams.yaml clinical_measurement.cv)",
            "points_to": RCV_POINTS_TO[st], "direction": "toward_bound"}


def distractor_for(case_id: str, ddx: dict, taken: set[str]) -> dict | None:
    """A registered benign change outside the case's spectrum, its lines' key signals and the other
    push items, never pointing at a gold line. The direction class is drawn per case from
    `DISTRACTOR_DIRECTION_P`, the type within the class by a case-keyed rotation; a class with no
    eligible type falls through to the next."""
    from haenv.scoring.tracks import alias_hit_asserted
    spec = spectrum_ids(str(ddx.get("spec_id") or ""))
    gold_names = [str(ddx.get("diagnosis") or "")] + [str(a) for a in ddx.get("aliases") or []]
    fx = _findings()

    def eligible(d):
        if d["fid"] in spec or d["fid"] in taken or d["fid"] not in fx:
            return False
        return not any(alias_hit_asserted(a, gold_names)
                       for a in [d["points_to"]["name"]] + d["points_to"]["aliases"])

    u, acc, first = _u(case_id, "distractor-dir"), 0.0, DISTRACTOR_DIRECTION_P[-1][0]
    for cls, w in DISTRACTOR_DIRECTION_P:
        acc += w
        if u < acc:
            first = cls
            break
    order = [first] + [c for c, _ in DISTRACTOR_DIRECTION_P if c != first]
    for cls in order:
        pool = [d for d in DISTRACTORS if d["direction"] == cls and eligible(d)]
        if not pool:
            continue
        d = pool[int(_u(case_id, "distractor", cls) * len(pool))]
        out = dict(d)
        um = _u(case_id, "distractor-mag")
        out["magnitude"] = next(m for m, c in DISTRACTOR_MAG_CUM if um < c)
        return out
    return None


# ------------------------------------------------------------------ filler
_LAB_STREAM_FID = {"fasting_glucose": "FBG", "triglycerides": "TG", "ALT": "ALT", "AST": "AST",
                   "HbA1c": "HbA1c", "LDL": "LDL"}
#: Catalogue findings read from the population background (`findings_render._value`, NHANES
#: quantile tables where adopted), the spec's filler source (2.2 item 4).
_FILLER_POOL = ("Hb", "Cr", "Na", "Ca", "TSH", "HDL", "K", "CRP", "LDH", "Ferritin", "ESR")


def filler_pool(case_id: str, ddx: dict, live_streams, taken: set[str]) -> list[dict]:
    """Non-gold background readings in a case-keyed order. A finding the case already carries as a
    live lab stream is skipped: a population draw would contradict the case's own series."""
    spec = spectrum_ids(str(ddx.get("spec_id") or ""))
    on_stream = {_LAB_STREAM_FID[s] for s in (live_streams or ()) if s in _LAB_STREAM_FID}
    fx = _findings()
    out: list[dict] = []
    for fid in sorted(_FILLER_POOL, key=lambda f: _h(case_id, "filler", f)):
        if fid in spec or fid in taken or fid in on_stream or fid not in fx:
            continue
        if isinstance((fx[fid] or {}).get("ref"), dict):
            out.append({"fid": fid, "source": "background"})
    return out


# ------------------------------------------------------------------ push plan (gold block)
def push_plan(meta: dict) -> dict | None:
    """`adjudication.revision`: the push block's fill plan, or None (no round 2) for the
    insufficient tier and for non-diagnosis cases. Values are rendered at solve time from the
    world (`render_push`), so the plan holds finding ids only."""
    ddx = meta.get("ddx") or {}
    if not ddx or ddx.get("insufficient"):
        return None
    # F0 bridge cases answer round 1 only, so they compare with v1.0.1 on round-1 dims (r1-2).
    if frame_of_meta(meta) == "F0":
        return None
    cid = str(meta.get("case_id"))
    th = threads_of(ddx)
    slots: list[dict] = []
    taken: set[str] = set()
    for i, t in enumerate(th):
        for f in t["key_signals"]:
            if f not in taken:
                slots.append({"role": "key", "fid": f, "thread": i})
                taken.add(f)
    for f in discriminating_of(ddx, exclude=taken):
        slots.append({"role": "discrim", "fid": f})
        taken.add(f)
    dis = (rcv_distractor_for(cid, ddx, meta, taken) if (F3_RCV_PUSH and frame_of_meta(meta) == "F3") else None) \
        or distractor_for(cid, ddx, taken)
    if dis is not None:
        slots.append({"role": "distractor", "fid": dis["fid"]})
        taken.add(dis["fid"])
    for f in filler_pool(cid, ddx, (meta.get("clinical") or {}).keys(), taken):
        slots.append({"role": "filler", **f})
    fill = slots[:PUSH_N]
    ids = [f"PB-{k + 1}" for k in range(len(fill))]
    order = sorted(range(len(fill)), key=lambda k: _h(cid, "push-order", fill[k]["fid"]))
    items = []
    for disp, k in enumerate(order):
        items.append({**fill[k], "item_id": ids[disp], "fill_rank": k})
    return {"n": len(items), "items": items, "threads": th,
            "distractor": (dis if any(s["role"] == "distractor" for s in fill) else None),
            "distractor_truncated": dis is not None and not any(s["role"] == "distractor" for s in fill),
            "fill_roles": [s["role"] for s in fill]}


def render_push(raw, T: int, plan: dict) -> list[dict]:
    """Solver-visible push items in display order (`item_id`, name, value, unit, reference, flag,
    note). Key, discriminating and stream filler readings use `gated.synth_on_demand`, the menu's
    own renderer; background filler and the distractor use `findings_render._value`."""
    from haenv.world.observations.findings_render import _band
    from haenv.world.observations.findings_render import _value
    from haenv.m2.hooks import install
    install()                     # the base-condition readings come through the hooked renderer
    from haenv.interaction.gated import _report_face
    from haenv.interaction.gated import synth_on_demand
    fx = _findings()
    cid = str(getattr(raw, "case_id", ""))
    sex = (getattr(raw, "user_profile", None) or {}).get("sex")
    out: list[dict] = []
    dis = plan.get("distractor") or {}
    for it in sorted(plan.get("items") or [], key=lambda x: x["item_id"]):
        fid = it["fid"]
        spec = fx.get(fid) or {}
        if it["role"] == "distractor" and dis.get("kind") == "rcv_fluctuation":
            pt = rcv_reading(raw, T, dis)
            note = _pick_note(dis.get("note"), cid)
        elif it["role"] == "distractor" and dis.get("qualitative"):
            pt = {"value": "阳性", "flag": "POSITIVE", "name": spec.get("name_cn", fid)}
            note = _pick_note(dis.get("note"), cid)
        elif it["role"] == "distractor":
            v = _value(spec, dis.get("direction", "high"), dis.get("magnitude", "moderate"),
                       cid, fid, 901)
            lo, hi = _band(spec)
            flag = "H" if v > hi else ("L" if v < lo else "NORMAL")
            name, sym = _report_face(cid, fid, flag)
            pt = {"value": v, "unit": spec.get("unit", ""), "ref_low": lo, "ref_high": hi,
                  "flag": sym, "name": name or spec.get("name_cn", fid)}
            note = _pick_note(dis.get("note"), cid)
        elif it["role"] == "filler" and it.get("source") == "background" and not _base_declared(raw, fid):
            v = _value({**spec, "bg_sex": sex}, "normal", None, cid, fid, 902, declared=False, spill=True)
            lo, hi = _band(spec)
            flag = "H" if v > hi else ("L" if v < lo else "NORMAL")
            name, sym = _report_face(cid, fid, flag)
            pt = {"value": v, "unit": spec.get("unit", ""), "ref_low": lo, "ref_high": hi,
                  "flag": sym, "name": name or spec.get("name_cn", fid)}
            note = None
        else:
            pts = synth_on_demand(raw, fid, int(T)) or []
            p = dict(pts[0]) if pts else {}
            if p.get("flag") in ("POSITIVE", "NEGATIVE"):
                p["value"] = "阳性" if p["flag"] == "POSITIVE" else "阴性"   # same short form for every qualitative item
            pt = {k: p.get(k) for k in ("value", "unit", "ref_low", "ref_high", "flag", "name")
                  if k in p}
            note = None
        if note is None:
            note = _NOTES_PLAIN[int(_u(cid, "note", fid) * len(_NOTES_PLAIN))]
        out.append({"item_id": it["item_id"], "ts": int(T), **pt, "note": note})
    return out


def rcv_reading(raw, T: int, dis: dict) -> dict:
    """The F3 distractor's value: the stream's last draw at or before T moved by `rcv_share` of the
    95 % RCV, toward the nearer reference bound (so it reads flagged when the move reaches it), at
    the stream's print precision. Never outside the RCV by construction."""
    from haenv.interaction.gated import _obs_cfg
    from haenv.interaction.gated import _report_face
    from haenv.world.clinical.indicators import of
    st = dis["stream"]
    pts = sorted((p for p in (getattr(raw, "longitudinal_data", None) or {}).get(st) or []
                  if isinstance(p, dict) and isinstance(p.get("value"), (int, float))
                  and int(p.get("ts", 10 ** 9)) <= int(T)), key=lambda p: int(p["ts"]))
    ind = of(st)
    rr = ind.get("reference_range") or {}
    lo, hi = rr.get("low"), rr.get("high")
    v0 = float(pts[-1]["value"]) if pts else float(hi or 1.0)
    cv = float((_obs_cfg()[1] or {}).get(st) or 0.0)
    rcv = 1.96 * math.sqrt(2) * cv
    up = hi is not None and (lo is None or abs(math.log(max(hi, 1e-9) / v0)) <= abs(math.log(max(lo, 1e-9) / v0)) if lo else True)
    share = float(dis.get("rcv_share", 0.7))
    # Surface salience (spec 2.2 item 3: out of band, flagged): when a within-RCV move can reach the
    # nearer bound (needs <= RCV_CROSS_MAX of the RCV), move just past it instead of the drawn share.
    bound = hi if up else lo
    if bound and rcv > 0 and v0 > 0:
        need = abs(math.log(float(bound) / v0)) / rcv
        if (bound > v0) == up and need <= RCV_CROSS_MAX:
            share = max(share, min(RCV_CROSS_MAX, need + 0.05))
    v = v0 * math.exp((1 if up else -1) * share * rcv)
    v = round(v, int(ind.get("ndigits", 2)))
    flag = "H" if (hi is not None and v > hi) else ("L" if (lo is not None and v < lo) else "NORMAL")
    name, sym = _report_face(str(getattr(raw, "case_id", "")), dis["fid"], flag)
    # Same printed fields as every other push item: the previous draw is in the case's own series,
    # an extra field here would mark the distractor's role (RV-6).
    return {"value": v, "unit": ind.get("unit", ""), "ref_low": lo, "ref_high": hi, "flag": sym,
            "name": name or (_findings().get(dis["fid"]) or {}).get("name_cn", st)}


def _base_declared(raw, fid: str) -> bool:
    """A known base condition declares `fid` (`registry/base_condition_findings.yaml`): the filler
    then reads the world (`synth_on_demand`, what a purchase returns), not the population draw."""
    pb = ((getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {})
    names = [str(pb.get("disease") or "")] + [str(c) for c in (pb.get("comorbidities") or [])]
    table = base_condition_findings()
    return any(d.get("id") == fid for n in names for d in table.get(n, ()))


def _pick_note(note, cid: str) -> str:
    """One wording variant of a distractor note, keyed on the case (template variants until the
    generation-LLM wording pass)."""
    if isinstance(note, (list, tuple)):
        return note[int(_u(cid, "dis-note") * len(note))]
    return str(note or "")


def roles_by_id(plan: dict) -> dict[str, str]:
    return {it["item_id"]: it["role"] for it in (plan or {}).get("items") or []}


def distractor_id(plan: dict) -> str | None:
    for it in (plan or {}).get("items") or []:
        if it["role"] == "distractor":
            return it["item_id"]
    return None


# ------------------------------------------------------------------ frames (gold + question side)
def m2_latent(meta: dict) -> dict:
    from haenv.contracts.external_gold import SLOT
    return dict((meta.get(SLOT) or {}).get("m2") or {})


def frame_of_meta(meta: dict) -> str:
    return str(m2_latent(meta).get("frame") or "F0")


#: F2 gold rule (spec 3.2, provisional, profile only), read in this order (`F2_RULE`):
#: 1. safety floor: today's fasting glucose below `F2_FLOOR` on a glucose-lowering drug => downtitrate;
#: 2. hidden lines affect the named drug through registered targets/adverse effects;
#:    an unrelated line leaves the own direction unchanged regardless of urgency;
#: 3. no hidden condition: the primary disease's own control (`f2_direction`).
MED_DIRECTIONS = ("investigate_first", "uptitrate", "keep", "downtitrate")
#: F2 negative arm (no hidden condition): the direction follows the primary disease's own control on
#: the record (`f2_direction`). Control target per primary disease: HbA1c < 7.0 % (ADA Standards of
#: Care 2025, 6.5a), LDL-C < 2.6 mmol/L (ESC/EAS 2019 moderate-risk goal), ALT <= 40 U/L with the printed sex band
#: below it (`F2_SEX_BAND`; MASLD has no ALT titration target). Provisional,
#: review: pending. A disease with no live marker has no gold (None, not scored).
F2_TARGETS = {"T2D": ("HbA1c", 7.0), "dyslipidemia": ("LDL", 2.6), "MASLD": ("ALT", 40.0)}
#: Hypertension's own direction (positive arm only): mean home systolic pressure over the 28 days up to
#: T against 130 mmHg (2025 AHA/ACC < 130/80, the value `control_targets.yaml:home_sbp` carries; NICE
#: NG136 1.4.22 reads the HBPM average against < 135/85 under 80 years). Not a negative-arm disease.
F2_BP_TARGET, F2_BP_WINDOW = 130.0, 28
#: Safety floor: fasting glucose below 3.9 mmol/L is level 1 hypoglycaemia (ADA Standards of Care 2025,
#: section 6: "a measurable glucose concentration <70 mg/dL (<3.9 mmol/L) and >=54 mg/dL"). On a
#: glucose-lowering drug (`drug_effects.yaml`, an HbA1c effect) reducing it is accepted; investigating is
#: accepted too unless the regimen holds a drug that causes hypoglycaemia (`F2_HYPO_DRUGS`).
#: No HbA1c floor: NICE NG28 asks to maintain an HbA1c below target when there is no hypoglycaemia.
F2_FLOOR = ("fasting_glucose", 3.9)
F2_NOT_URGENT = ("🟡", "🟢")
#: Hidden lines whose own management intensifies the primary treatment: with them the primary disease's
#: own uptitrate is accepted next to investigate_first. CKD with hypertension: KDIGO 2021 BP in CKD,
#: Rec 3.1.1 "a target systolic blood pressure (SBP) of <120 mm Hg, when tolerated", Practice Point 3.2.2
#: RASi "using the highest approved dose that is tolerated".
F2_INTENSIFY = {"HD-UNI-10": ("hypertension",)}
#: Hidden lines that present with vomiting or falling food and fluid intake (registered symptoms of each):
#: volume depletion, so uptitrating the primary treatment is not accepted. NICE NG148 1.2.14: "Consider
#: temporarily stopping ACE inhibitors and ARBs ... with diarrhoea, vomiting or sepsis"; KDIGO 2024 CKD,
#: Practice Point 4.3.2 commentary: sick day rules in "acute, dehydrating illness" stop sulfonylureas,
#: ACEi, diuretics, metformin, ARBs, NSAIDs and SGLT2i.
F2_INTAKE_LOSS = frozenset({"HD-UNI-09", "HD-UNI-11", "HD-UNI-12", "HD-UNI-15", "HD-UNI-20"})
#: The sick-day list of the same KDIGO 2024 commentary ("temporarily stop the following medications:
#: sulfonylureas, ACEi, diuretics/direct renin inhibitors, metformin, ARBs, NSAIDs, and SGLT2i"): with a
#: hidden line in `F2_INTAKE_LOSS`, holding such a drug (downtitrate) is accepted.
F2_SICK_DAY_NAMES = ("metformin", "hydrochlorothiazide", "chlorthalidone", "chlortalidone", "indapamide",
                     "furosemide", "torasemide", "bumetanide", "spironolactone", "eplerenone", "aliskiren",
                     "glimepiride", "gliclazide", "glipizide", "glibenclamide", "glyburide", "gliquidone")
F2_SICK_DAY_SUFFIXES = ("pril", "sartan", "gliflozin")


def sick_day_drug(drug: str) -> bool:
    """Is `drug` on the sick-day list (`F2_SICK_DAY_NAMES`, ACE inhibitors, ARBs, SGLT2 inhibitors)?"""
    d = str(drug or "").lower()
    return bool(d) and (d in F2_SICK_DAY_NAMES or d.endswith(F2_SICK_DAY_SUFFIXES))
#: Primary diseases whose control line has a lower bound at the printed sex-specific upper reference limit:
#: MASLD reads ALT against the band the report prints (female 0-25, male 0-33 U/L, `findings_render.SEX_REF`)
#: and against 40 U/L; between the two both keep and uptitrate are accepted. AASLD 2023 practice guidance:
#: "a true normal alanine aminotransferase (ALT) ranges from 29 to 33 U/L in men and from 19 to 25 U/L in
#: women"; ALT improvement is a surrogate (statement 29), not a titration target.
F2_SEX_BAND = frozenset({"MASLD"})
#: MASLD with FIB-4 above 2.67 is at high risk of advanced fibrosis: specialist referral comes first, so
#: investigate_first is the gold and the own direction stays accepted. AASLD 2023, Figure 2: "Direct referral
#: to gastroenterology/hepatology should be considered ... when FIB4 > 2.67"; EASL-EASD-EASO 2024, Fig. 2:
#: > 2.67 => hepatology referral. The 1.3-2.67 band (elastography, or a year of lifestyle intervention in
#: EASL) is not moved.
F2_FIB4_HIGH = ("MASLD", 2.67)


def f2_negative_eligible(disease: str) -> bool:
    """The F2 negative arm has a gold direction only for a primary disease with a control target
    (`F2_TARGETS`); other primary diseases are not given that arm."""
    return str(disease or "") in F2_TARGETS


F2_RULE = ("Every reading is today's as the solver can read it: the record's draw on day T, else a menu test "
           "that returns the marker bought on day T, else the last draw before T. "
           "1. fasting glucose < 3.9 mmol/L on a glucose-lowering drug: with a hidden named condition => "
           "investigate_first, downtitrate accepted too when it is 🟡/🟢; without one => downtitrate, investigate_first "
           "accepted too unless the regimen holds insulin or a secretagogue (the hypoglycaemia is not the drug's); "
           "an unrelated hidden line follows that same own safety rule, independent of its urgency; "
           "2. hidden named condition that does not affect the named drug's registered targets or safety => own direction, regardless of that condition's urgency; "
           "otherwise investigate_first; when it is 🟡/🟢 the primary disease's own direction is "
           "accepted too if the hidden line does not act on its control, and its own uptitrate if the hidden line "
           "indicates intensifying the treatment; never uptitrate when the hidden line cuts intake (vomiting), and "
           "then holding a sick-day drug (metformin, ACEi, ARB, diuretic, SGLT2i, sulfonylurea) is accepted; "
           "3. own direction: the primary marker within target => keep; above target => uptitrate once the "
           "marker's response time (HbA1c 90 days, LDL-C 60, ALT 90) has passed since the last dose change; "
           "inside it keep, uptitrate accepted too (a titration schedule in progress). MASLD: ALT within the printed sex band => keep, between "
           "that band and 40 U/L => keep or uptitrate; FIB-4 > 2.67 => investigate_first, the ALT direction "
           "accepted too (provisional, review pending)")
URGENCY_TIERS = ("🔴", "🟠", "🟡", "🟢")
#: Lab streams with an EFLM total per-draw CV (`physio_streams.yaml` clinical_measurement.cv).
F3_STREAMS = ("HbA1c", "fasting_glucose", "LDL", "triglycerides", "ALT", "AST")
#: Shortest gap (days) between the two draws an F3 follow-up compares. HbA1c: about three months
#: (ADA Standards of Care 2026, 6.2, "every 3 months"; red-cell turnover). Lipids and liver enzymes:
#: the lower end of the 8 +/- 4 week recheck after a treatment change (ESC/EAS 2019 dyslipidaemia,
#: monitoring section; not re-checked against the original text). Fasting glucose: the same 4 weeks
#: (our choice). review: pending.
F3_MIN_GAP_DAYS = {"HbA1c": 90, "fasting_glucose": 28, "LDL": 28, "triglycerides": 28, "ALT": 28, "AST": 28}
#: Routine laboratory analytical CV where the world's per-draw CV takes the EFLM desirable
#: 0.5 x CVI floor: HbA1c intralaboratory CV < 2 % (NGSP units; Sacks et al., NACB/ADA laboratory
#: guidelines, Clin Chem 2011 / Diabetes Care 2023; not re-checked against the original text).
RCV_LAB_CVA = {"HbA1c": 0.020}
#: EFLM within-subject CV behind the world's per-draw CV (`physio_streams.yaml` comment).
RCV_CVI = {"HbA1c": 0.016}


def rcv_cv(stream: str) -> float:
    """Total CV the RCV reads: sqrt(CVI^2 + CVA^2) with a routine laboratory CVA where one is
    registered (`RCV_LAB_CVA`), else the world's per-draw CV (`physio_streams.yaml`)."""
    if stream in RCV_LAB_CVA:
        return round(math.sqrt(RCV_CVI[stream] ** 2 + RCV_LAB_CVA[stream] ** 2), 4)
    from haenv.interaction.gated import _obs_cfg
    return float((_obs_cfg()[1] or {}).get(stream) or 0.0)


def _draws(longitudinal: dict, st: str, upto: int) -> list[dict]:
    return sorted((p for p in (longitudinal or {}).get(st) or []
                   if isinstance(p, dict) and isinstance(p.get("value"), (int, float))
                   and not isinstance(p.get("value"), bool) and int(p.get("ts", 10 ** 9)) <= int(upto)),
                  key=lambda p: int(p["ts"]))


#: F3 follow-up draw (revision R-25): today's repeat moves the last draw at or before T by k x the
#: 95 % RCV -- beyond it on the "exceeds" arm (the hidden line's own marker, in its declared
#: direction), inside it on the "within" arm.
F3_EXCEED_K = (1.4, 2.2)
F3_WITHIN_K = (0.15, 0.75)
#: Lab stream -> finding id (condition spectra are written in finding ids).
F3_STREAM_FID = {"HbA1c": "HbA1c", "fasting_glucose": "FBG", "LDL": "LDL", "triglycerides": "TG",
                 "ALT": "ALT", "AST": "AST"}


#: Markers the primary (record-named) disease moves when it truly worsens: the exceeds arm's second
#: choice when no hidden line declares a live marker (the change is then the named disease's).
PRIMARY_MARKERS = {"T2D": ("HbA1c", "fasting_glucose"), "dyslipidemia": ("LDL", "triglycerides"),
                   "MASLD": ("ALT", "AST"), "obesity": ("HbA1c", "triglycerides")}


def f3_plan_for(case_id: str, spec_id: str, arm: str, live: list[str], exceed: bool,
                primary: str = "") -> dict | None:
    """The F3 plan written by the job generator: which live stream is followed and on which arm.
    The exceeds arm follows a stream whose finding the hidden line declares abnormal (so a line
    explains the change); the within arm any live stream, sign drawn per case. On the negative arm
    (`arm == "negative"`, no hidden line) the exceeds arm is a real improvement of the named disease's
    own marker (down beyond the RCV), so "exceeds" does not imply "a hidden line is there"."""
    live = [st for st in F3_STREAMS if st in live]
    if not live:
        return None
    if exceed and arm == "negative":
        decl = [(st, -1) for st in PRIMARY_MARKERS.get(primary, ()) if st in live] or [(st, -1) for st in live]
        st, sign = sorted(decl, key=lambda x: _h(case_id, "f3-ex", x[0]))[0]
        lo, hi = F3_EXCEED_K
    elif exceed:
        decl = exceed_streams(spec_id, live) or [(st, 1) for st in PRIMARY_MARKERS.get(primary, ()) if st in live]
        if not decl:
            return None
        st, sign = sorted(decl, key=lambda x: _h(case_id, "f3-ex", x[0]))[0]
        lo, hi = F3_EXCEED_K
    else:
        st = sorted(live, key=lambda x: _h(case_id, "f3-in", x))[0]
        sign = 1 if _u(case_id, "f3-sign") < 0.5 else -1
        lo, hi = F3_WITHIN_K
    k = round(lo + (hi - lo) * _u(case_id, "f3-k"), 4)
    return {"stream": st, "arm": "exceeds" if exceed else "within", "sign": sign, "k": k}


def exceed_streams(spec_id: str, live: list[str]) -> list[tuple[str, int]]:
    """(stream, sign) for live streams whose finding a hidden component declares high/low."""
    out = []
    for comp in components(spec_id):
        for f in (_profiles().get(comp) or {}).get("findings") or ():
            for st in live:
                if F3_STREAM_FID.get(st) == f.get("id") and f.get("direction") in ("high", "low"):
                    out.append((st, 1 if f["direction"] == "high" else -1))
    return list(dict.fromkeys(out))


def f3_followup(longitudinal: dict, plan: dict | None, T: int) -> dict | None:
    """Today's follow-up result on the followed stream (solver-visible on F3, and the F3 gold's
    second reading): the last draw at least `F3_MIN_GAP_DAYS` before T moved by sign x k x RCV,
    at the stream's print precision. None without a plan or such a draw."""
    if not plan:
        return None
    from haenv.world.clinical.indicators import of
    st = plan["stream"]
    pts = _draws(longitudinal, st, int(T) - int(F3_MIN_GAP_DAYS.get(st, 0)))
    cv = rcv_cv(st)
    if not pts or cv <= 0 or float(pts[-1]["value"]) <= 0:
        return None
    rcv = 1.96 * math.sqrt(2) * cv
    prev = pts[-1]
    ind = of(st)
    v = round(float(prev["value"]) * math.exp(int(plan["sign"]) * float(plan["k"]) * rcv), int(ind.get("ndigits", 2)))
    return {"indicator": st, "unit": ind.get("unit", ""),
            "previous": {"day": int(prev["ts"]), "value": prev["value"]}, "today": {"day": int(T), "value": v}}


def f3_world(longitudinal: dict, plan: dict | None, T: int) -> dict:
    """The world the F3 solver queries: the followed stream carries today's follow-up as its draw on
    day T and no draw between the compared earlier draw and T, so the follow-up shown, the stream
    and a named test on day T read one value. Other streams unchanged; no plan => unchanged."""
    fu = f3_followup(longitudinal, plan, T)
    if fu is None:
        return longitudinal
    st, d0 = fu["indicator"], int(fu["previous"]["day"])
    src = list((longitudinal or {}).get(st) or [])
    gone = {int(p.get("ts", -1)) for p in src if isinstance(p, dict) and d0 < int(p.get("ts", -1)) <= int(T)}
    prev = next(p for p in src if isinstance(p, dict) and int(p.get("ts", -1)) == d0)
    out = dict(longitudinal)
    for mate in panel_mates(st):                      # one visit draws the whole panel: a removed draw
        if mate != st and out.get(mate):              # takes its panel mates of that day with it
            out[mate] = [p for p in out[mate] if not (isinstance(p, dict) and int(p.get("ts", -1)) in gone)]
    out[st] = [p for p in src if not (isinstance(p, dict) and int(p.get("ts", -1)) in gone)] + [
        {**prev, "ts": int(T), "value": fu["today"]["value"]}]
    return out


def panel_mates(stream: str) -> tuple[str, ...]:
    """Streams drawn on one visit with `stream` (`physio_streams.yaml:clinical_observation.groups`)."""
    from haenv.resources.regpath import load_registry
    groups = ((load_registry("physio_streams.yaml") or {}).get("clinical_observation") or {}).get("groups") or {}
    return next((tuple(g) for g in groups.values() if stream in (g or ())), (stream,))


def f3_truth(longitudinal: dict, frame_gold_block: dict, T: int) -> dict | None:
    """F3 gold: on a planned case the follow-up pair (last draw, today's result); otherwise the last
    two draws at or before T (`rcv_truth`)."""
    g = frame_gold_block or {}
    fu = f3_followup(longitudinal, g.get("f3_plan"), T)
    if fu is None:
        return None if g.get("f3_plan") else rcv_truth(longitudinal, g.get("followed_stream"), T)
    cv = rcv_cv(fu["indicator"])
    rcv = 1.96 * math.sqrt(2) * cv
    a, b = float(fu["previous"]["value"]), float(fu["today"]["value"])
    if a <= 0 or b <= 0:
        return None
    d = math.log(b / a)
    return {"stream": fu["indicator"], "prev": a, "last": b, "log_change": round(d, 4),
            "rcv_log": round(rcv, 4), "exceeds_rcv": abs(d) > rcv, "planned_arm": g["f3_plan"].get("arm")}


@functools.lru_cache(maxsize=None)
def menu_items_for(fid: str) -> tuple[str, ...]:
    """Test items of the M2-frame menu (`m2_catalogue`) that stand for finding `fid`."""
    fx = _findings()
    return tuple(t for t in m2_catalogue() if fid in m2_menu_findings(t, fx))


def bought_today(raw, fid: str, T: int) -> float | None:
    """The value of `fid` a test item on the M2-frame menu returns when bought on day T (the solver's
    purchase path, `m2_synth_menu_target`), or None when no menu test returns it that day."""
    import haenv.interaction.gated as G
    orig = getattr(G.synth_menu_target, "__wrapped__", G.synth_menu_target)
    for t in menu_items_for(fid):
        try:
            got = m2_synth_menu_target(orig, raw, t, int(T)) or []
        except Exception:                                    # noqa: BLE001  a purchase that raises reads nothing
            continue
        for p in got:
            v = p.get("value")
            if (p.get("finding_id") == fid and int(p.get("ts", -1)) == int(T)
                    and isinstance(v, (int, float)) and not isinstance(v, bool)):
                return float(v)
    return None


def visible_today(raw, stream: str, T: int) -> tuple[float, int] | None:
    """(value, day) of a lab stream as the solver can read it on day T: the record's draw on day T;
    else the value a menu test bought on day T returns (`bought_today`); else the last draw before T.
    FIB-4 reads the queried world's stream (`fib4_world`). None when none of the three exists."""
    ld = getattr(raw, "longitudinal_data", None) or {}
    if stream == "FIB4":
        ld = fib4_world(raw, ld, T)
    pts = _draws(ld, stream, T)
    if pts and int(pts[-1]["ts"]) == int(T):
        return float(pts[-1]["value"]), int(T)
    v = bought_today(raw, F3_STREAM_FID.get(stream, stream), T)
    if v is not None:
        return v, int(T)
    return (float(pts[-1]["value"]), int(pts[-1]["ts"])) if pts else None


def f2_today(raw, stream: str, T: int) -> float | None:
    """Today's reading of a lab stream as the solver can read it (`visible_today`); None when the
    stream has no draw at or before T."""
    if not _draws(getattr(raw, "longitudinal_data", None) or {}, stream, T):
        return None
    got = visible_today(raw, stream, T)
    return got[0] if got else None


#: Critical values that set the urgency by themselves, read as the solver can read them on day T
#: (`visible_today`). Serum potassium < 2.5 mmol/L: "Severe and life-threatening hypokalemia is defined
#: when potassium levels are <2.5 mEq/L" and is treated with intravenous potassium "with close follow-up,
#: continuous ECG monitoring" (Kardalas E et al., Endocr Connect 2018;7(4):R135-R146, PMC5881435).
CRITICAL_LOW = {"K": 2.5}


def critical_reading(raw, T: int) -> dict | None:
    """The first critical value the case shows on day T (`CRITICAL_LOW`): 🔴 with a red flag, else None."""
    for fid, lim in CRITICAL_LOW.items():
        got = visible_today(raw, fid, T)
        if got and got[0] < lim:
            return {"finding": fid, "today": got[0], "day": got[1], "limit": lim, "urgency": "🔴"}
    return None


#: Glucose-lowering drugs that cause hypoglycaemia by themselves. ADA Standards of Care 2025, section 6:
#: "Individuals at risk for hypoglycemia are those treated with insulin, sulfonylureas, or meglitinides;
#: clinically significant hypoglycemia is rare among individuals taking other diabetes medication classes"
#: (Rec 6.6 deintensifies those three); section 9, Table 9.2: hypoglycaemia risk "No" for metformin,
#: GLP-1 RAs and dual GIP/GLP-1 RA.
F2_HYPO_DRUGS = ("insulin", "glimepiride", "gliclazide", "glipizide", "glibenclamide", "glyburide",
                 "gliquidone", "repaglinide", "nateglinide")


def hypo_regimen(drugs) -> bool:
    """Does the regimen hold a drug that causes hypoglycaemia by itself (`F2_HYPO_DRUGS`)?"""
    return any(str(d or "").lower().startswith(h) for d in drugs or () for h in F2_HYPO_DRUGS)


def regimen_of(raw, drug: str) -> list[str]:
    """The primary drug and the medicines the world gives for the known conditions (`events.shown_drugs`,
    the list the scene prints)."""
    pb = (getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {}
    out = [str(drug or "")]
    try:
        from haenv.world.observations.events import shown_drugs
        out += shown_drugs(str(getattr(raw, "case_id", "")), pb.get("disease"), pb.get("comorbidities") or [],
                           str(drug or ""), str(pb.get("sex") or ""), str(pb.get("age_range") or ""))
    except Exception:                                        # noqa: BLE001  a bare record lists the primary drug only
        pass
    return [d for d in dict.fromkeys(out) if d]


def glucose_lowering(drug: str) -> bool:
    """A drug with a registered HbA1c effect (`registry/drug_effects.yaml`)."""
    import yaml
    d = ((yaml.safe_load(registry_path("drug_effects.yaml").read_text(encoding="utf-8")) or {})
         .get("drugs") or {}).get(str(drug or "")) or {}
    return isinstance(d.get("total_hba1c_pp"), (int, float)) and d["total_hba1c_pp"] < 0


def f2_hidden_linked(spec_id: str, disease: str, drug: str) -> bool:
    """Does a hidden component act on this drug's registered targets or adverse effects?

    The disease argument remains for callers; the named drug determines the
    relationship. An unregistered drug fails explicitly rather than guessing.
    """
    comps = components(spec_id)
    rules = pre_review_rules()
    if any(str(drug or "") in (rules["drug_adverse_effects"].get(c) or {}).get("drugs", ()) for c in [spec_id, *comps]):
        return True
    ctl = rules["drug_targets"].get(str(drug or ""))
    if not ctl:
        raise ValueError(f"F2 drug has no registered target relationship: {drug!r}")
    if "systolic_bp" in ctl and any(c in rules["secondary_hypertension"]["components"] for c in comps):
        return True
    return any(f.get("id") in ctl and f.get("direction") in ("high", "low")
               for c in comps for f in (_profiles().get(c) or {}).get("findings") or ())


#: Time a marker needs to show a dose change; above target and inside it the change is not judged yet
#: (keep, with uptitrate accepted: the world titrates every few weeks and the next scheduled step is an
#: uptitration), past it the target decides (uptitrate). HbA1c: "A1C reflects average glycemia over approximately
#: 2-3 months", tested "typically every 3 months" when not at goal (ADA Standards of Care 2025, section 6).
#: LDL-C: "Measure liver transaminase and full lipid profile at 2 to 3 months after starting or changing
#: lipid-lowering treatment" (NICE NG238 1.11.1), the lower bound. ALT (MASLD): no source gives a response
#: time; the HbA1c window is used (provisional, review pending).
F2_RESPONSE_DAYS = {"HbA1c": 90, "LDL": 60, "ALT": 90}


def last_dose_change(longitudinal: dict, T: int) -> int | None:
    """Day of the last change of the primary drug's dose at or before T (`dose_timeline`; its first point
    is the start of the treatment). None without a dose timeline."""
    pts = _draws(longitudinal, "dose_timeline", T)
    if not pts:
        return None
    day = int(pts[0]["ts"])
    for a, b in zip(pts, pts[1:]):
        if float(b["value"]) != float(a["value"]):
            day = int(b["ts"])
    return day


def f2_direction(raw, disease: str, T: int, drug: str = "", spec_id: str | None = None) -> dict | None:
    """F2 reading of a case (`F2_RULE`): the safety floor, the primary disease's own direction and,
    with a hidden line (`spec_id`), how it bears on the treatment. None when there is neither a floor
    breach nor an own direction."""
    out: dict = {}
    if glucose_lowering(drug):
        got = visible_today(raw, F2_FLOOR[0], T) if _draws(getattr(raw, "longitudinal_data", None) or {},
                                                          F2_FLOOR[0], T) else None
        if got is not None and got[0] < F2_FLOOR[1]:
            out["floor"] = {"marker": F2_FLOOR[0], "today": got[0], "day": got[1], "limit": F2_FLOOR[1],
                            "hypo_regimen": hypo_regimen(regimen_of(raw, drug))}
    ld = getattr(raw, "longitudinal_data", None) or {}
    dis = str(disease or "")
    if dis in F2_TARGETS:
        st, lim = F2_TARGETS[dis]
        pts = _draws(ld, st, T)
        got = visible_today(raw, st, T) if pts else None
        if got is not None:
            last, day = got
            changed = last_dose_change(ld, T)
            settling = changed is not None and int(T) - changed < F2_RESPONSE_DAYS[st]
            out.update(direction="keep" if (last <= lim or settling) else "uptitrate", marker=st, target=lim,
                       first=float(pts[0]["value"]), last=last, last_day=int(day), dose_change_day=changed,
                       response_days=F2_RESPONSE_DAYS[st], within_response_time=bool(settling))
            if settling and last > lim:
                out["also"] = ["uptitrate"]
            if dis in F2_SEX_BAND:
                sx = _sex_spec(raw, st, _findings().get(F3_STREAM_FID.get(st, st)) or {})
                if sx is not None:
                    from haenv.world.observations.findings_render import _band
                    uln = float(_band(sx)[1])
                    out["sex_band_high"] = uln
                    if uln < last <= lim:
                        out["also"] = ["uptitrate"]
            if dis == F2_FIB4_HIGH[0]:
                fib = visible_today(raw, "FIB4", T)
                if fib is not None and fib[0] > F2_FIB4_HIGH[1]:
                    out["fib4"] = {"today": fib[0], "day": fib[1], "limit": F2_FIB4_HIGH[1]}
                    out["also"] = [d for d in dict.fromkeys([out["direction"], *(out.get("also") or [])])]
                    out["direction"] = "investigate_first"
    elif dis == "hypertension":
        sbp = [float(p["value"]) for p in _draws(ld, "systolic_bp", T) if int(p["ts"]) > int(T) - F2_BP_WINDOW]
        if sbp:
            m = round(math.fsum(sbp) / len(sbp), 1)
            out.update(direction="keep" if m < F2_BP_TARGET else "uptitrate", marker="systolic_bp",
                       target=F2_BP_TARGET, last=m, last_day=int(T), n_readings=len(sbp))
    if out.get("floor"):
        if out.get("direction"):
            out["own_direction"] = out["direction"]
        out["direction"] = "downtitrate"
    if spec_id:
        comps = [spec_id, *components(spec_id)]
        out["hidden_linked"] = f2_hidden_linked(spec_id, dis, drug)
        out["hidden_intensifies"] = any(dis in F2_INTENSIFY.get(c, ()) for c in comps)
        out["hidden_intake_loss"] = any(c in F2_INTAKE_LOSS for c in comps)
        if out["hidden_intake_loss"] and sick_day_drug(drug):
            out["sick_day_hold"] = True
    if out.get("direction") or out.get("floor") or out.get("sick_day_hold"):
        if drug:
            out["dose_state"] = f2_dose_state(raw, dis, drug, T)
        return out
    return None


def f2_dose_state(raw, disease: str, drug: str, T: int) -> dict:
    """Feasibility of changes to the named drug, from its recorded dose and registered ladder.

    Zero means held/not started, never authorization to restart. The ladder's top
    is a benchmark ceiling, not a personalized treatment recommendation.
    Amlodipine 10 mg/day agrees with NORVASC US prescribing information 2.1.
    """
    from haenv.resources.regpath import load_registry
    from haenv.world.clinical.med_course import dose_at
    pair = ((load_registry("drug_indications.yaml").get("pairs") or {}).get(disease) or {}).get(drug) or {}
    ladder = pair.get("dose_steps")
    pts = (getattr(raw, "longitudinal_data", None) or {}).get("dose_timeline")
    if not ladder or not pts:
        return {"drug": drug, "status": "unavailable", "current_dose": None, "maximum_dose": None}
    dose, maximum = dose_at(pts, T), max(float(x) for x in ladder)
    if not math.isfinite(dose) or dose < 0 or dose > maximum:
        raise ValueError(f"F2 {drug}: recorded dose outside registered range")
    wm = (getattr(raw, "latent_premise", None) or {}).get("world_medication")
    if wm is not None:
        primary = wm.get("primary") or {}
        if primary.get("drug") != drug or dose_at(primary.get("dose_timeline"), T) != dose:
            raise ValueError("F2 named dose and world primary medication disagree")
    state = {"drug": drug, "status": "active" if dose > 0 else "held_or_not_started",
             "current_dose": dose, "maximum_dose": maximum}
    reduction = f2_adverse_reduction(raw, drug, T, ladder)
    if reduction:
        state["adverse_reduction"] = reduction
    return state


def f2_adverse_reduction(raw, drug: str, T: int, ladder) -> dict | None:
    """A dated, same-drug dose reduction explicitly attributed to an adverse effect.

    Reads the same medication record the solver sees. The record has no CK or
    rechallenge proof: it authorizes reassessment, not a confirmed intolerance
    label. A higher dose already resumed at T is not blocked by an older lower
    dose episode; events after T cannot affect the answer.
    """
    from haenv.world.clinical.med_course import record_rows
    from haenv.world.clinical.med_course import dose_at
    from haenv.world.clinical.med_course import DISPLAY
    policy = pre_review_rules().get("adverse_reduction", {}).get(drug)
    if not policy:
        return None
    wm = (getattr(raw, "latent_premise", None) or {}).get("world_medication") or {}
    primary = wm.get("primary") or {}
    if primary.get("drug") != drug:
        return None
    pts = primary.get("dose_timeline") or []
    current = dose_at(pts, T)
    if current <= 0 or not pts:
        return None
    rows = record_rows(wm, min(int(p["ts"]) for p in pts), T, primary=True)
    failed = []
    for row in rows:
        match = re.fullmatch(r"剂量 ([\d.]+) → ([\d.]+)", row.get("event", ""))
        if row.get("drug") == DISPLAY[drug][0] and row.get("reason") == policy["reason"] and match:
            before, after = map(float, match.groups())
            if before > after > 0 and before > current:
                failed.append((before, row))
    if not failed:
        return None
    dose, row = min(failed, key=lambda x: x[0])
    next_dose = min((float(d) for d in ladder if float(d) > current), default=None)
    return {"day": row["day"], "dose": dose, "reason": row["reason"],
            "status": "reassessment_required", "blocks_next_step": next_dose is not None and next_dose >= dose,
            "protocol_id": policy["protocol_id"]}


def f2_accept(hidden: bool, urgency: str | None, f2: dict | None) -> list[str]:
    """Clinical directions intersected with the named drug's feasible dose changes."""
    accepted = _f2_clinical_accept(hidden, urgency, f2)
    state = (f2 or {}).get("dose_state")
    if state is None:  # Pure clinical-rule callers; built F2 readings always carry a dose state.
        return accepted
    if state["status"] != "active":
        return ["investigate_first"]
    if state["current_dose"] >= state["maximum_dose"]:
        accepted = [d for d in accepted if d != "uptitrate"]
    if (state.get("adverse_reduction") or {}).get("blocks_next_step") and "uptitrate" in accepted:
        accepted = list(dict.fromkeys("keep" if d == "uptitrate" else d for d in accepted))
        if "investigate_first" not in accepted:
            accepted.append("investigate_first")
    # Above target at maximum requires reassessment; it does not redefine the
    # control target or claim that continuing the current drug achieves control.
    return accepted or ["investigate_first"]


def _f2_clinical_accept(hidden: bool, urgency: str | None, f2: dict | None) -> list[str]:
    """Accepted F2 directions, gold first (`F2_RULE`)."""
    f2 = f2 or {}
    quiet = urgency in F2_NOT_URGENT
    own = [d for d in dict.fromkeys([f2.get("direction"), *(f2.get("also") or [])]) if d]
    fl = f2.get("floor")
    if fl:
        if hidden and (f2.get("hidden_linked") is not False or f2.get("hidden_intake_loss")):
            return ["investigate_first"] + (["downtitrate"] if quiet else [])
        return ["downtitrate"] + ([] if fl.get("hypo_regimen") else ["investigate_first"])
    if hidden and f2.get("hidden_linked") is False and own and not f2.get("hidden_intake_loss"):
        return own
    if not hidden:
        return own
    acc = ["investigate_first"]
    if quiet and own:
        add = own if f2.get("hidden_linked") is False else (
            [d for d in own if d == "uptitrate"] if f2.get("hidden_intensifies") else [])
        if f2.get("hidden_intake_loss"):
            add = [d for d in add if d != "uptitrate"]
        acc += [d for d in add if d not in acc]
    if f2.get("sick_day_hold") and "downtitrate" not in acc:
        acc.append("downtitrate")
    return acc


def f3_stream(meta: dict) -> str | None:
    plan = m2_latent(meta).get("f3")
    if plan:
        return plan.get("stream")
    live = [s for s in (meta.get("clinical") or {}) if s in F3_STREAMS]
    if not live:
        return None
    return sorted(live, key=lambda s: _h(meta.get("case_id"), "f3", s))[0]


def frame_gold(meta: dict) -> dict | None:
    """`adjudication.m2_frame`: the frame-specific gold (F1-F3), None on F0."""
    lat = m2_latent(meta)
    fr = lat.get("frame")
    if fr not in ("F1", "F2", "F3"):
        return None
    ddx = meta.get("ddx") or {}
    hidden = ddx.get("join_gold") in ("unified", "comorbidity")
    g = {"frame": fr, "arm": lat.get("arm"), "hidden_named_condition": bool(hidden),
         "prior": lat.get("prior")}
    if fr == "F1":
        g["urgency"] = ddx.get("urgency")
        g["red_flag"] = bool(meta.get("red_flag", False))
    elif fr == "F2":
        f2 = lat.get("f2") or {}
        acc = f2_accept(hidden, ddx.get("urgency"), f2)
        g["med_direction"] = acc[0] if acc else None
        g["med_accept"] = acc
        g["med_rule"] = F2_RULE
        if f2:
            g["f2_reading"] = f2
    else:
        g["followed_stream"] = f3_stream(meta)
        g["f3_plan"] = lat.get("f3")
        g["rcv_rule"] = ("|ln(v_today / v_prev)| > 1.96 * sqrt(2) * cv_total (core.rcv_cv); "
                         "v_prev = last draw at least F3_MIN_GAP_DAYS before T, v_today = the follow-up "
                         "result (f3_followup)")
    return g


def frame_probe(meta: dict) -> dict | None:
    """`prediction_context.m2_frame`: frame name and the scene text (template); None on F0."""
    lat = m2_latent(meta)
    fr = lat.get("frame")
    if fr not in ("F1", "F2", "F3"):
        return None
    out = {"frame": FRAME_NAMES[fr], "scene": lat.get("scene") or ""}
    if fr == "F3":
        out["followed_indicator"] = f3_stream(meta)
    return out


def rcv_truth(longitudinal: dict, stream: str | None, T: int) -> dict | None:
    """F3 gold from the world: the last two draws of `stream` at or before T against the 95% RCV."""
    if not stream:
        return None
    from haenv.interaction.gated import _obs_cfg
    pts = sorted((p for p in (longitudinal or {}).get(stream) or []
                  if isinstance(p, dict) and isinstance(p.get("value"), (int, float))
                  and int(p.get("ts", 10 ** 9)) <= int(T)), key=lambda p: int(p["ts"]))
    if len(pts) < 2 or float(pts[-2]["value"]) <= 0 or float(pts[-1]["value"]) <= 0:
        return None
    cv = rcv_cv(stream)
    if cv <= 0:
        return None
    rcv = 1.96 * math.sqrt(2) * cv
    d = math.log(float(pts[-1]["value"]) / float(pts[-2]["value"]))
    return {"stream": stream, "prev": pts[-2]["value"], "last": pts[-1]["value"],
            "log_change": round(d, 4), "rcv_log": round(rcv, 4), "exceeds_rcv": abs(d) > rcv}


# ------------------------------------------------------------------ F1 acute event pool
@functools.lru_cache(maxsize=1)
def _acute_events() -> dict:
    import pathlib

    import yaml
    p = registry_path("acute_events.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def acute_event_for(case_id: str, spec_id: str, arm: str) -> dict | None:
    """The F1 day-T presentation (`registry/acute_events.yaml`): the hidden line's own presentation
    on the positive arm (first component with an entry), a self-limited complaint drawn per case
    on the negative arm. None when the positive arm's components have no entry."""
    doc = _acute_events()
    if arm == "negative":
        pool = list(doc.get("negative") or [])
        return copy.deepcopy(pool[int(_u(case_id, "acute") * len(pool))]) if pool else None
    for comp in components(spec_id):
        e = (doc.get("positive") or {}).get(comp)
        if e:
            return copy.deepcopy({k: v for k, v in e.items() if k != "source"})
    return None


def _vitals_text(ev: dict) -> str:
    v = (ev or {}).get("vitals")
    return f",{v['name']} {v['value']} {v['unit']}" if v else ""


# ------------------------------------------------------------------ scene templates (job generator)
def sync_current_medications(raw) -> None:
    """Refresh the visible table AND the legacy scene after the world is observed.

    Replay clients must call this after re-rendering ``scene_text``; that function
    still constructs a plan-time scene and has no realized dose course. No gold
    or latent plan text is rewritten here. Safe to call repeatedly after edits.
    """
    import re
    from haenv.world.clinical.med_course import active_medicines
    from haenv.world.clinical.med_course import display
    wm = (getattr(raw, "latent_premise", None) or {}).get("world_medication")
    if wm is None:
        raise ValueError("M2 medication display requires the world's medication record")
    pc = dict(raw.prediction_context or {})
    shown = [display(drug, dose) for drug, dose, _ in active_medicines(wm, int(pc["prediction_time_T"]))]
    pc["current_medications"] = shown
    if "history_lo" in wm:
        # the dated record behind the list over the world's record (`med_course.record_rows`)
        from haenv.world.clinical.med_course import record_rows
        pc["medication_record"] = record_rows(wm, int(wm["history_lo"]), int(pc["prediction_time_T"]), primary=True)
    if pc.get("m2_frame"):
        fr = dict(pc["m2_frame"])
        sentence = "患者目前在用 " + "、".join(shown) if shown else "患者目前未使用药物"
        fr["scene"], n = re.subn(r"(?:患者)?目前(?:在用 [^。]*|未使用药物)", sentence, str(fr.get("scene") or ""))
        if n != 1:
            raise ValueError("M2 scene requires exactly one current-medication sentence")
        pc["m2_frame"] = fr
    raw.prediction_context = pc
    from haenv.m2.background import laboratory_documents
    laboratory_documents(raw)


def co_medications(case_id: str, raw: dict) -> list[str]:
    """The medicines the world gives for the case's known conditions (`raw.disease`, `raw.comorbidities`):
    the primary drug and `med_course.concurrent_medicines`, the draws the build makes. A hidden line has none."""
    from haenv.world.observations.events import shown_drugs
    return shown_drugs(case_id, raw.get("disease"), raw.get("comorbidities") or [], str(raw.get("drug") or ""),
                       str(raw.get("sex") or ""), str(raw.get("age_range") or ""))


def scene_text(frame: str, raw: dict, latent: dict, case_id: str) -> str:
    """The frame scene, written from the case's own facts by template. F1 presents the latest true
    symptom before T as the reason for today's visit; F2 the treated base condition; F3 a follow-up
    draw; every frame names the medicines the world gives for the known conditions (`co_medications`), and
    F2 names the primary drug as the one its medication direction asks about. No fact is added that the case does not already contain."""
    T = int(latent.get("index_time_T") or 0)
    disease = raw.get("disease") or ""
    meds = "患者目前在用 " + "、".join(co_medications(case_id, raw))
    names = {"T2D": "2型糖尿病", "hypertension": "高血压", "dyslipidemia": "血脂异常",
             "MASLD": "代谢相关脂肪性肝病", "obesity": "肥胖"}
    if frame == "F1":
        ev = (latent.get("m2") or {}).get("acute_event") or latent.get("acute_event")
        if ev:
            what = f"「{ev['text']}」{_vitals_text(ev)}"
        else:
            sy = [s for s in (raw.get("symptoms") or []) if isinstance(s, dict) and int(s.get("day", 10 ** 9)) <= T]
            from haenv.world.observations.events import scene_lay_sentence
            ph = str(sy[-1]["text"]) if sy else "不适"
            what = f"「{scene_lay_sentence(case_id, ph, raw.get('sex')) or ph}」"
        return (f"今天(第 {T} 天)患者因{what}来诊。{meds}。请判断分诊紧急度(🔴 立即急诊 / 🟠 数日内 / "
                f"🟡 数周内 / 🟢 常规随访)、首选检查、鉴别诊断,以及是否需要立即请临床医生复核。")
    if frame == "F2":
        ask = ("请从完整用药记录中识别本次需要评估处置的药物,给出该药的调药方向"
               "(先查因暂不调 / 加量 / 维持 / 减量),")
        return (f"今天(第 {T} 天)是{names.get(disease, disease)}的慢病复诊,{meds}。{ask}"
                f"并判断是否有其他疾病在扰乱基础病的控制。")
    if frame == "F3":
        return (f"今天(第 {T} 天)一次随访复查到来,结果见 `followup_result`(同一实验室,附上次结果)。{meds}。"
                f"请就 `followed_indicator` 指出的指标,判断这两次结果的变化是否超出生物学与分析变异(RCV),"
                f"是哪条线解释了这一变化,以及下一步怎么做。")
    return ""


# ================================================================ M2 single track (spec 2026-10-01)
# Frame asks and the round-2 revision prompt (kept here, not in `framings.py`: see the module
# docstring). Appended after the question by `evaluate._m2_suffix`; a case without `prediction_context.m2_frame` and a round without a
# revision context get nothing, so every v1.0.1 question renders byte-identically.
# Wording is template-written (awaiting the generation-LLM wording pass); the fields asked for
# are the contract the M2 judges read.

#: Frame-specific answer fields, keyed by `prediction_context.m2_frame.frame`.
M2_FRAME_ASKS: dict[str, str] = {
    "acute_triage": (
        "\n\n除上面要求的字段外,另加一个字段:"
        '\n  "urgency": "🔴|🟠|🟡|🟢"   (🔴 立即急诊 · 🟠 数日内 · 🟡 数周内 · 🟢 常规随访)'),
    "chronic_med_adjustment": (
        "\n\n除上面要求的字段外,另加一个字段:"
        '\n  "treatment_plan": {"target_drug": "药物通用名或登记英文 ID", "direction": "investigate_first|uptitrate|keep|downtitrate", "reason": "..."}'
        "\n  从全部用药及记录中自行识别本次评估处置的一种药物;维持也须识别该药。"
        "\n  (investigate_first = 先查因、暂不调药;给出供医生复核的方向,不自主改药。已暂停的药应核对暂停记录,不据此自行恢复。)"),
    "followup_interpretation": (
        "\n\n查阅本次复查及历次报告,判断变化是否超出生物学与分析变异(RCV)。除上面要求的字段外,另加一个字段:"
        '\n  "followup": {"indicator": "...", "exceeds_rcv": true|false, "explained_by": "...", "next_step": "..."}'),
}

#: Round 2. `{items}` is the five pushed readings (JSON lines), `{round1}` the round-1 answer.
M2_REVISION_PROMPT = (
    "\n\n【回合 ②:新到的结果】查询菜单已关闭。下面是这位患者今天新到的 5 条结果"
    "(固定推送,与你在回合 ① 是否购买无关;条目顺序不代表重要性):\n{items}\n"
    "\n你在回合 ① 的完整答案是:\n{round1}\n"
    "\n请判断要不要改判。只输出一个 JSON 对象(无多余文字),包含:"
    '\n  "revision": {{"decision": "keep|change", "changed_fields": ["differential", ...], '
    '"because": ["PB-..."]}}'
    "\n  以及改后的**完整**答案(字段与回合 ① 相同;保持不变也要完整给出)。"
    "\n  `because` 只填促使你改判(或确认不改)的推送条目编号。")

#: The answer-contract fields the M2 judges read (recorded with the frame on the question side).
M2_ANSWER_CONTRACT: dict = {
    "version": "m2-answer-v2",
    "frame_fields": {"acute_triage": ["urgency"], "chronic_med_adjustment": ["treatment_plan.target_drug", "treatment_plan.direction"],
                     "followup_interpretation": ["followup.exceeds_rcv"]},
    "revision_fields": ["revision.decision", "revision.changed_fields", "revision.because"],
}


# ================================================================ explain-away rendering (spec 4.2)
#: `registry/explain_away.yaml` names -> component spec ids.
EA_HIDDEN = {"B12缺乏": "JD-B12", "LADA": "JD-LADA", "血色病": "JD-HEMO", "甲旁亢": "JD-PHPT",
             "甲减": "JD-HYPO", "OSA": "JD-OSA", "Addison": "JD-ADDISON", "嗜铬": "JD-PHEO",
             "库欣": "JD-CUSH", "肢端肥大": "JD-ACRO", "PCOS": "JD-PCOS", "CKD进展": "JD-CKM",
             "原醛": "JD-ALDO"}
EA_LINE = {"原醛": ("JD-ALDO",), "库欣": ("JD-CUSH",), "嗜铬": ("JD-PHEO",), "SLE": ("JD-SLE",),
           "甲减": ("JD-HYPO",), "原醛/嗜铬": ("JD-ALDO", "JD-PHEO"), "甲亢": ("JD-GRAVES",),
           "LADA": ("JD-LADA",)}
#: Age written into a symptom template ("38岁难治高血压"): removed on F1-F3 (schedule D8, the
#: template's age contradicts the sampled age range). F0 keeps the v1.0.1 text (bridge).
_AGE = __import__("re").compile(r"\d{2}\s*岁")


@functools.lru_cache(maxsize=1)
def _ea_rows() -> tuple:
    import pathlib

    import yaml
    p = registry_path("explain_away.yaml")
    return tuple((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("rows") or ())


def ea_rows() -> tuple:
    """The rows generation, K-EA and the audit EA-1 read: every table row except `status: retired`
    (revision r2-E1; a retired row stays in the table with its `retired_why`)."""
    return tuple(r for r in _ea_rows() if str(r.get("status") or "") != "retired")


def drug_visible(row: dict, med_text: str | None) -> bool:
    """A row with `requires_visible_drug` licenses only when one of those drug names occurs in the
    case's visible medication text (revision r2-E2; case-insensitive substring). Rows without the
    field are not drug-dependent. No text = no drug visible (fail closed)."""
    need = row.get("requires_visible_drug") or ()
    if not need:
        return True
    t = str(med_text or "").lower()
    return any(str(d).lower() in t for d in need if str(d))


def visible_medication_text(scene: str | None, symptoms=()) -> str:
    """The text in which a solver can read a drug name: the frame scene (F2 names the primary drug,
    「患者目前在用 {drug}」) and the record's symptom texts and contexts. `user_profile` carries no
    medication field and concurrent medicines are not named on the record."""
    parts = [str(scene or "")]
    for s in symptoms or ():
        if isinstance(s, dict):
            parts += [str(s.get("text") or ""), str(s.get("context") or "")]
        elif isinstance(s, (list, tuple)):
            parts += [str(x or "") for x in s]
        else:
            parts.append(str(s or ""))
    return "\n".join(p for p in parts if p)


@functools.lru_cache(maxsize=None)
def component_templates(comp: str) -> tuple[str, ...]:
    """Symptom template texts of a component spec (comorbid specs take their symptoms from these)."""
    import haenv_kernel.joint_scenarios as _JS
    from haenv.world.overlay import haenv_comorbid_specs
    from haenv.world.overlay import haenv_independent_specs
    specs = {**_JS.DDX_SPECS, **haenv_comorbid_specs(_JS.DDX_SPECS), **haenv_independent_specs()}
    sy = (specs.get(comp) or {}).get("symptoms") or ()
    return tuple(str(s[1] if isinstance(s, (list, tuple)) else (s or {}).get("text", "")) for s in sy)


_PUNCT = __import__("re").compile(r"[\s,,、。;;:：()()+\-/·]")


def _bigrams(x: str) -> set[str]:
    x = _PUNCT.sub("", str(x))
    return {x[i:i + 2] for i in range(len(x) - 1)}


def _matches(text: str, terms) -> bool:
    """A symptom shows a table finding when at least half of the finding's character bigrams occur
    in it (punctuation ignored): 「第2-3掌指关节痛」 ~ 「第2、3掌指关节晨僵隐痛」."""
    tb = _bigrams(text)
    for x in terms:
        fb = _bigrams(x)
        if fb and len(fb & tb) / len(fb) >= 0.5:
            return True
    return False


def visible_explainers(row: dict, comps: list[str], known: list[str], hidden: str,
                       on_record: set[str] | None = None, med_text: str | None = None) -> list[str]:
    """The explainers of `row` this case shows: a condition named in `known` (the caller passes the
    conditions the record names), or another gold line with at least one of its own symptoms in
    `on_record` (component ids; None = do not check). A row whose mechanism needs a drug
    (`requires_visible_drug`) shows none unless that drug is in `med_text` (`drug_visible`)."""
    if not drug_visible(row, med_text):
        return []
    out = [k for k in (row.get("explainer_known") or ()) if k in known]
    for c in EA_LINE.get(str(row.get("explainer_line") or ""), ()):
        if c in comps and c != hidden and (on_record is None or c in on_record):
            out.append(c)
    return out


def _templates_any(comp: str) -> set[str]:
    return {t for x in component_templates(comp) for t in (x, _AGE.sub("", x).lstrip(",, "))}


_NUM = __import__("re").compile(r"[≈~约]?\d+(?:\.\d+)?(?:\s*(?:h|天|次|kg|cm|岁|mmHg))?\+?\s*(?:天)?")
_SPLIT = __import__("re").compile(r"[、,,;;+＋()()。:：]|(?<![A-Za-z])/|/(?![A-Za-z])|却|伴")
#: Context phrases that report what was tried or ruled out, not a finding (an atom matching one is
#: neither licensed nor a violation): 「严格低盐」「三联降压仍≈146/96」「去角质无效」「无怀孕可能」.
_QUALIFIER = __import__("re").compile(r"无效|无改善|部分缓解|几乎不降|仍高|^严格|三联降压|活动后加重|"
                                      r"^无怀孕可能$|^无家族史$")


def _norm(x: str) -> str:
    return __import__("re").sub(r"[\s\-·≈~]", "", _NUM.sub("", str(x)))


def symptom_atoms(text: str) -> list[str]:
    """The clauses of a symptom text (split on 、 , + / parentheses 却 伴; numbers stay in the surface
    and are ignored for comparison). Atoms shorter than two characters after normalisation drop."""
    return [a.strip() for a in _SPLIT.split(str(text or "")) if len(_norm(a)) >= 2]


def is_qualifier(atom: str) -> bool:
    return bool(_QUALIFIER.search(_norm(atom)))


def atom_licensed(atom: str, finding: str) -> bool:
    """A clause shows a table finding when one finding clause and the clause contain each other
    (normalised, at least two characters): 「睡够8h仍极度乏力」 ~ 「乏力」. Every clause is judged on its
    own: 「乏力、舌炎/口角炎」 licenses 乏力 and leaves 舌炎, 口角炎 unlicensed."""
    a = _norm(atom)
    for f in symptom_atoms(finding):
        f = _norm(f)
        if len(f) >= 2 and len(a) >= 2 and (f in a or a in f):
            return True
    return False


@functools.lru_cache(maxsize=1)
def specific_signs() -> dict:
    """`registry/explain_away.yaml:specific_signs`: hidden condition -> its specific signs (an
    explained-away line may not show one on the record unless a row of its own covers it)."""
    import pathlib

    import yaml
    p = registry_path("explain_away.yaml")
    return dict((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("specific_signs") or {})


@functools.lru_cache(maxsize=1)
def clues() -> dict:
    """`registry/explain_away.yaml:clues`: hidden condition -> its clue phrases, each one finding (a case
    shows one of them, drawn by seed)."""
    import pathlib

    import yaml
    p = registry_path("explain_away.yaml")
    return {k: tuple(v) for k, v in ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("clues") or {}).items()}


@functools.lru_cache(maxsize=1)
def clue_explained_by() -> dict:
    """`registry/explain_away.yaml:clue_explained_by`: clue phrase -> the known conditions and gold lines of
    a case that could explain it."""
    import pathlib

    import yaml
    p = registry_path("explain_away.yaml")
    return {k: frozenset(v) for k, v in ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("clue_explained_by") or {}).items()}


@functools.lru_cache(maxsize=1)
def clue_contradicted_by() -> dict:
    """`registry/explain_away.yaml:clue_contradicted_by`: clue phrase -> the known conditions and treatment
    goals of a case that the clue would contradict (a patient in a weight-loss programme does not say he
    has always been thin)."""
    import yaml
    p = registry_path("explain_away.yaml")
    return {k: frozenset(v) for k, v in ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("clue_contradicted_by") or {}).items()}


def _clue_opts(comp: str, present: frozenset) -> list[str]:
    return [t for t in clues().get({v: k for k, v in EA_HIDDEN.items()}.get(comp)) or ()
            if not (clue_explained_by().get(t, frozenset()) | clue_contradicted_by().get(t, frozenset())) & present]


def _clue(case_id: str, comp: str, out: list, mine: list, rec: dict, lines_clue: dict | None = None,
          present: frozenset = frozenset()) -> bool:
    """The line's last symptom becomes one of its registered clues that nothing else on the record
    (`present`: the known conditions and the other gold lines) explains, drawn by case and line; the clue is
    recorded on `lines_clue` (an explained-away line) or in `rec["clues"]` (any other gold line)."""
    import haenv.world.generation.rng as rng
    opts = _clue_opts(comp, present)
    if not opts:
        return False
    text, i = rng.pick(opts, "m2-clue", case_id, comp), mine[-1]
    c = {"day": out[i].get("day"), "text": text, "source": out[i]["text"]}
    out[i]["text"], out[i]["context"] = text, ""
    if lines_clue is None:
        rec.setdefault("clues", []).append({"component": comp, **c})
    else:
        lines_clue["clue"] = c
    return True


def symptom_owner(text: str, comps) -> str | None:
    """Retrieval only (never a verdict): the component whose templates share the most character
    bigrams with `text` (share >= 0.5 of the text's bigrams, unique best), else None. Variant texts
    differ from the templates (「颈后对称发黑增厚、去角质无效」), so exact template lookup misses them."""
    tb = _bigrams(_AGE.sub("", str(text)))
    if not tb:
        return None
    best: list = []
    for c in comps:
        sc = max((len(tb & _bigrams(_AGE.sub("", t))) / len(tb) for t in component_templates(c)), default=0.0)
        best.append((sc, c))
    best.sort(reverse=True)
    if best and best[0][0] >= 0.5 and (len(best) < 2 or best[0][0] > best[1][0]):
        return best[0][1]
    return None


def license_atoms(text: str, terms: list) -> list[tuple[str, str, object]]:
    """Per clause of `text`: (atom, 'licensed' | 'qualifier' | 'unlicensed', the (finding, row,
    explainers) that licenses it or None). `terms` are (finding, row, explainers) of rows whose
    explainer is visible."""
    out = []
    for a in symptom_atoms(text):
        hit = next(((f, r, ex) for f, r, ex in terms if atom_licensed(a, f)), None)
        out.append((a, "licensed" if hit else ("qualifier" if is_qualifier(a) else "unlicensed"), hit))
    return out


#: Clauses an explained-away line may lose, the same for every condition. An explained line rewrites
#: exactly one of its symptoms (the one that loses the fewest clauses; a specific sign first among
#: equals) and loses at most this many clauses; a line whose cheapest rewrite would lose more, or
#: nothing, stays unexplained. 3 is the smallest cap under which every table condition with a
#: patient-worded covered form has an explained arm (B12, hypothyroidism: one symptom = 3 clauses).
EA_MAX_REMOVED = 3


def explain_away_render(spec_id: str, known: list[str], symptoms: list[dict],
                        T: int | None = None, med_text: str | None = None,
                        case_id: str = "", explain: bool = True,
                        max_removed: int = EA_MAX_REMOVED, goals=("sustained_weight_loss",)) -> tuple[list[dict], dict]:
    # `known`: the conditions the record names (`user_profile.known_conditions`: the primary
    # disease and the M2 explainer bases); other comorbidities are not named and explain nothing.
    # `goals`: the case's treatment goals (`patient_basics.goals`, the world's default); a clue they
    # contradict is not shown (`clue_contradicted_by`).
    # `med_text`: the case's visible medication text (`visible_medication_text`); a row with
    # `requires_visible_drug` covers nothing without it. Retired rows are skipped (`ea_rows`).
    """(symptoms, record). With `explain` on, each hidden component that the table lets a visible
    explainer cover loses at most `max_removed` clauses, all in one rewritten symptom (`_explain_line`);
    no symptom is dropped. A line is recorded as explained away only when at least one clause was
    removed; a covered line with nothing to remove, and every line with `explain` off, renders as an
    unexplained line. Every gold line with two or more symptoms whose clue is not explained by the
    record shows one registered clue on its last symptom (explained lines always, unexplained lines
    when none of their symptoms carries a specific sign). Age tokens are dropped from every symptom.
    The record lists, per symptom of an explained-away line, the row, the visible explainer and the
    clauses removed (read by the audit EA-1 and the K-EA profile; judging never reads it)."""
    comps = components(spec_id)
    out = [dict(s) for s in symptoms or []]
    for s in out:
        s["text"] = _AGE.sub("", str(s.get("text") or "")).lstrip(",, ")
    rec: dict = {"lines": []}
    if len(comps) < 2 and not any(r for r in ea_rows() if spec_id in (r.get("specs") or ())):
        return out, rec
    owner = [symptom_owner(str(s0.get("text")), comps) for s0 in symptoms or []]
    on_record = {owner[i] for i, s0 in enumerate(symptoms or [])
                 if owner[i] and (T is None or int(s0.get("day", 0)) <= int(T))}
    names = {v: k for k, v in EA_HIDDEN.items()}
    for comp in comps:
        rows = [r for r in ea_rows() if EA_HIDDEN.get(str(r.get("hidden"))) == comp
                and spec_id in (r.get("specs") or ())]
        cov = [(r, visible_explainers(r, comps, known, comp, on_record, med_text)) for r in rows] if explain else []
        cov = [(r, ex) for r, ex in cov if ex]
        mine = [i for i in range(len(out)) if owner[i] == comp]
        present = frozenset(known) | frozenset(goals or ()) | (frozenset(comps) - {comp})
        clue_last = len(mine) >= 2 and bool(_clue_opts(comp, present))
        line = _explain_line(comp, cov, out, mine, max_removed, clue_last) if cov else None
        clue_at = mine
        if line is None and cov and clue_last:
            # the only symptom a visible explainer can cover is the line's last one (e.g. 乏力、爬楼腿软 of
            # primary aldosteronism under known hypothyroidism, row E29): that one is rewritten and the clue
            # takes the latest other symptom of the line, so the clue never erases the covered finding (W7)
            line = _explain_line(comp, cov, out, mine, max_removed, False, among=mine[-1:])
            if line is not None:
                clue_at = mine[:-1]
        if line is None:
            # a gold line that is not explained away shows a finding that separates it from its nearest
            # look-alike when none of its own symptoms carries a specific sign
            name = names.get(comp)
            if name and len(mine) >= 2 and not any(sg in out[i]["text"] for i in mine for sg in specific_signs().get(name, ())):
                _clue(case_id, comp, out, mine, rec, present=present)
            continue
        if len(mine) >= 2 and _clue(case_id, comp, out, clue_at, rec, line, present):    # the line's last symptom is its clue
            line["symptoms"] = [x for x in line["symptoms"] if x["index"] != clue_at[-1]]
        for x in line["symptoms"]:
            x.pop("index", None)
        rec["lines"].append(line)
    return out, rec


def _explain_line(comp: str, cov: list, out: list, mine: list, budget: int, clue_last: bool,
                  among: list | None = None) -> dict | None:
    """Rewrite one symptom of a covered line to its covered form (mutates `out`): the symptom keeps its
    licensed clauses only, or, with none, becomes the next covered finding not yet on the record; an
    unlicensed context goes with it. A rewrite is a candidate only when the result has patient sentences
    (`registry/symptom_lay.yaml`) and removes at most `budget` clauses; the candidate that removes the
    fewest clauses is taken (one removing a specific sign of the line first, then record order).
    The candidates are the line's symptoms, minus its last when that one is kept for the clue (`clue_last`),
    or exactly `among` when given.
    Returns the line's record, or None (with `out` untouched) when there is no candidate."""
    from haenv.world.observations.events import symptom_lay
    lay = symptom_lay()
    terms = [(f, r, ex) for r, ex in cov for f in r.get("findings") or ()]
    signs = [_norm(sg) for sg in specific_signs().get({v: k for k, v in EA_HIDDEN.items()}.get(comp, ""), ())]
    used = {f for i in range(len(out)) for f, _, _ in terms
            if any(atom_licensed(a, f) for a in symptom_atoms(out[i]["text"]))}
    cand = []
    for i in (among if among is not None else (mine[:-1] if clue_last else mine)):
        lic = license_atoms(out[i]["text"], terms)
        ok = [(a, h) for a, k, h in lic if k == "licensed"]
        bad = [a for a, k, _ in lic if k == "unlicensed"]
        ctx_bad = [a for a, k, _ in license_atoms(str(out[i].get("context") or ""), terms) if k == "unlicensed"]
        if not bad and not ctx_bad:
            continue
        if not bad:
            new, hit = out[i]["text"], ok[0][1] if ok else None
        elif ok:
            new, hit = "、".join(a for a, _ in ok), ok[0][1]
        else:
            fresh = [t for t in terms if t[0] in lay]
            if not fresh:
                continue
            hit = ([t for t in fresh if t[0] not in used] or fresh)[0]
            new, ctx_bad = hit[0], [a for a in symptom_atoms(str(out[i].get("context") or "")) if not is_qualifier(a)]
        if new not in lay or len(bad) + len(ctx_bad) > budget:
            continue
        sp = 0 if any(g and g in _norm(a) for g in signs for a in bad + ctx_bad) else 1
        cand.append((sp, i, new, hit, bad, ctx_bad))
    if not cand:
        return None
    sp, i, new, hit, bad, ctx_bad = min(cand, key=lambda c: (len(c[4]) + len(c[5]), c[0], c[1]))
    take = {i: (new, hit, bad, ctx_bad)}
    line = {"component": comp, "rows": [r["id"] for r, _ in cov], "symptoms": []}
    for i in mine:
        src, rewritten, hit, removed = out[i]["text"], False, None, []
        if i in take:
            new, hit, bad, ctx_bad = take[i]
            rewritten, removed = new != src, bad + ctx_bad
            out[i]["text"] = new
            if ctx_bad:
                out[i]["context"] = ""
            used.update(f for f, _, _ in terms if atom_licensed(new, f))
        if hit is None:
            hit = next((h for _, k, h in license_atoms(out[i]["text"], terms) if k == "licensed"), None)
        line["symptoms"].append({"index": i, "day": out[i].get("day"), "text": out[i]["text"],
                                 "row": hit[1]["id"] if hit else None, "explainer": hit[2] if hit else [],
                                 "rewritten": rewritten, "source": src, "removed": removed})
    return line


def m2_synth_on_demand(orig, raw, target: str, T: int):
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import call_hook
    return call_hook(Stage.M2_FINDING, _base_m2_synth_on_demand, orig, raw, target, T)
