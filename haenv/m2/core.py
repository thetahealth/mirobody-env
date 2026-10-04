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

from ..regpath import registry_path

import copy
import functools
import hashlib
import math

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
#: Every template-written text carries this source tag (the generation LLM only touches wording).
TEXT_SOURCE = "template (awaiting generation-LLM pass)"

_ABNORMAL = ("high", "low", "positive")
_MAG_ORDER = {"marked": 0, "moderate": 1, "mild": 2, None: 3}
#: Specimen notes shared by every push item, so a note's presence or length does not mark a role.
_NOTES_PLAIN = ("空腹静脉血", "晨起空腹采血", "门诊常规采血", "上午空腹静脉采血", "复诊当日空腹采血",
                "门诊复查采血", "空腹", "常规采血", "外院送检,结果已核对", "门诊抽血,标本合格",
                "晨8点空腹肘静脉采血", "复查")


def _u(*path) -> float:
    from .. import rng
    return rng.unit("m2", *path)


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ registries
@functools.lru_cache(maxsize=1)
def _findings() -> dict:
    from ..registry import load_findings
    return load_findings()


@functools.lru_cache(maxsize=1)
def _profiles() -> dict:
    from ..registry import condition_findings_for_case
    return condition_findings_for_case(_findings())


# ------------------------------------------------------------------ M2 menu and world hooks
# MENU-1 (revisions R-19, r1-1). Everything M2 adds to the gated menu and to the readings a purchase
# returns lives here and applies only to cases carrying `prediction_context.m2_frame` (F1-F3).
# `haenv/gated.py`, `registry/gated_pricing.yaml` and `haenv/evaluate.py` stay as in v1.0.1, so the
# v1.0.1 packs and the F0 bridge cases are judged with the integration judging code unchanged
# (same `judging_sha16`); the hooks are installed by `haenv/m2/hooks.py`.

#: Discriminator test items the M2 frames add to the menu (kind = price tier). Beta-hydroxybutyrate
#: is the insulinoma line's confirmatory finding (routine chemistry, CLFS CPT 82010 = basic_lab).
M2_DISCRIMINATORS: dict[str, str] = {"血β-羟丁酸": "basic_lab"}
#: Menu items that resolve to findings on M2 frames only. The first two are existing catalogue
#: names (total testosterone CPT 84403 + SHBG 84270; serum creatinine 82565), basic_lab.
M2_RESOLVES_TO: dict[str, tuple[str, ...]] = {
    "性激素六项/睾酮·SHBG(FAI)": ("Testosterone", "SHBG"),
    "eGFR/血肌酐": ("Cr",),
    "血β-羟丁酸": ("BetaOHB",),
}


def is_m2_framed(obj) -> bool:
    """F1-F3 case, payload or instance: `prediction_context.m2_frame` is set (absent on F0)."""
    return isinstance(((getattr(obj, "prediction_context", None) or {}).get("m2_frame")), dict)


def m2_catalogue() -> list[str]:
    """Test items on an M2-frame menu: the v1.0.1 catalogue and discriminators, then M2's own."""
    from ..gated import discriminator_tests, test_catalogue
    base = test_catalogue() + discriminator_tests()
    return base + [t for t in M2_DISCRIMINATORS if t not in base]


def m2_menu_findings(target: str, findings: dict | None = None) -> tuple[str, ...]:
    """`gated.menu_findings` with the M2 `resolves_to` rows first."""
    from ..gated import menu_findings
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
    from ..gated import _cached_yaml, _dr
    doc = _cached_yaml(_dr() / "registry" / "base_condition_findings.yaml") or {}
    return {str(k): list((v or {}).get("findings") or []) for k, v in (doc.get("bases") or {}).items()}


def base_condition_decl(raw, fid: str) -> dict | None:
    """The declaration a known base condition makes for `fid`, or None. A case-and-finding keyed
    draw against `penetrance` decides whether this case shows it (else the background reading)."""
    from ..gated import _u as _gu
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


def m2_menu_for(orig, sp, withheld: dict, case_id: str = "") -> list[dict]:
    """`gated.menu_for` plus `M2_DISCRIMINATORS` on M2 frames, shuffled by the same key."""
    items = orig(sp, withheld, case_id)
    if not is_m2_framed(sp):
        return items
    from haenv_kernel.gatekeeper import COST                     # kernel
    have = {str(i["target"]) for i in items}
    for t, k in M2_DISCRIMINATORS.items():
        if t not in have:
            items.append({"target": t, "kind": k, "real": None, "is_test": True, "cost": COST.get(k, 5.0)})
    items.sort(key=lambda x: hashlib.sha256(f"{case_id}|{x['target']}".encode()).hexdigest())
    return items


def m2_synth_menu_target(orig, raw, target: str, T: int):
    """`gated.synth_menu_target` with the M2 `resolves_to` rows and the sex filter on M2 frames."""
    if not is_m2_framed(raw):
        return orig(raw, target, T)
    from .. import gated as G
    t = str(target or "").strip()
    if t in G.DECOY_SIGNALS:
        return None
    ids = M2_RESOLVES_TO.get(t) or G._resolves_to().get(t)
    if not ids:
        return G.synth_on_demand(raw, target, T)
    ids = for_sex(raw, ids)
    if not ids:
        return None
    pts: list[dict] = []
    for fid in ids:
        p = G.synth_on_demand(raw, fid, T)
        if not p:
            return None
        pts.extend(p)
    return pts


def m2_synth_on_demand(orig, raw, target: str, T: int):
    """`gated.synth_on_demand`; on M2 frames a known base condition shapes items the hidden spec
    does not declare (`registry/base_condition_findings.yaml`). Same rendering as the original."""
    if not is_m2_framed(raw) or not target or not isinstance(target, str):
        return orig(raw, target, T)
    from .. import gated as G
    from ..findings_render import _band, _value
    from ..overlay import spec_id_of
    from ..registry import condition_findings_for_case
    fx = _findings()
    fid = G._resolve_target(target.strip(), fx)
    if not fid or not G.has_reading_model(fx[fid]):
        return orig(raw, target, T)
    fspec = fx[fid]
    if G._named_on_stream(raw, fid, fspec, T) is not None:
        return orig(raw, target, T)
    spec_id, _ = spec_id_of(raw)
    cf = condition_findings_for_case(fx).get(spec_id, {}).get("findings", []) if spec_id else []
    decl = next((it for it in cf if it.get("id") == fid), None)
    declared_dir = (decl or {}).get("direction")
    if ((getattr(raw, "adjudication", None) or {}).get("ddx") or {}).get("insufficient"):
        decl = None
    if decl is not None:
        return orig(raw, target, T)
    bd = base_condition_decl(raw, fid)
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


def components(spec_id: str) -> list[str]:
    from ..overlay import HAENV_COMORBID_PAIRS
    cfg = HAENV_COMORBID_PAIRS.get(spec_id)
    return list(cfg["pair"]) if cfg else [spec_id]


def _key_findings_of(component: str) -> list[str]:
    prof = (_profiles().get(component) or {}).get("findings") or ()
    reach = menu_target_of()
    cand = [p for p in prof if p.get("role") == "confirmatory"
            and str(p.get("direction")) in _ABNORMAL and p.get("id") in reach]
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
    """Every confirmatory abnormal finding of a component, reachable or not (audit MENU-1)."""
    prof = (_profiles().get(component) or {}).get("findings") or ()
    return [str(p["id"]) for p in prof if p.get("role") == "confirmatory"
            and str(p.get("direction")) in _ABNORMAL]


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
        return out
    return [{"name": ddx.get("diagnosis") or sid, "aliases": list(ddx.get("aliases") or []),
             "component": sid, "key_signals": _key_findings_of(sid)}]


def discriminating_of(ddx: dict | None, exclude=()) -> list[str]:
    from ..overlay import rivals_for
    ddx = ddx or {}
    out: list[str] = []
    for r in (rivals_for(str(ddx.get("spec_id") or ""), ddx) or ()):
        f = str(((r or {}).get("discriminator_finding") or {}).get("finding") or "").strip()
        if f and f not in out and f not in exclude and f in menu_target_of():
            out.append(f)
    return out


def rival_names(ddx: dict | None) -> list[dict]:
    from ..overlay import rivals_for
    ddx = ddx or {}
    return [{"name": r["name"], "aliases": list(r.get("aliases") or [])}
            for r in (rivals_for(str(ddx.get("spec_id") or ""), ddx) or ())]


def spectrum_ids(spec_id: str) -> set[str]:
    from ..registry import condition_findings_for_case
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
    from ..tracks import alias_hit_asserted
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
    from ..tracks import alias_hit_asserted
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
    from ..findings_render import _band, _value
    from .hooks import install
    install()                     # the base-condition readings come through the hooked renderer
    from ..gated import _report_face, synth_on_demand
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
    from ..gated import _obs_cfg, _report_face
    from ..indicators import of
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
    from ..external_gold import SLOT
    return dict((meta.get(SLOT) or {}).get("m2") or {})


def frame_of_meta(meta: dict) -> str:
    return str(m2_latent(meta).get("frame") or "F0")


#: F2 gold rule (spec 3.2, provisional, profile only): a hidden named condition is the cause of the
#: worse control => investigate first, do not adjust; otherwise adherence/progression => up-titrate.
MED_DIRECTIONS = ("investigate_first", "uptitrate", "keep", "downtitrate")
URGENCY_TIERS = ("🔴", "🟠", "🟡", "🟢")
#: Lab streams with an EFLM total per-draw CV (`physio_streams.yaml` clinical_measurement.cv).
F3_STREAMS = ("HbA1c", "fasting_glucose", "LDL", "triglycerides", "ALT", "AST")


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
    explains the change); the within arm any live stream, sign drawn per case."""
    live = [st for st in F3_STREAMS if st in live]
    if not live:
        return None
    if exceed:
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
    second reading): the last draw at or before T moved by sign x k x RCV, at the stream's print
    precision. None without a plan or a draw at or before T."""
    if not plan:
        return None
    from ..gated import _obs_cfg
    from ..indicators import of
    st = plan["stream"]
    pts = sorted((p for p in (longitudinal or {}).get(st) or []
                  if isinstance(p, dict) and isinstance(p.get("value"), (int, float))
                  and not isinstance(p.get("value"), bool) and int(p.get("ts", 10 ** 9)) <= int(T)),
                 key=lambda p: int(p["ts"]))
    cv = float((_obs_cfg()[1] or {}).get(st) or 0.0)
    if not pts or cv <= 0 or float(pts[-1]["value"]) <= 0:
        return None
    rcv = 1.96 * math.sqrt(2) * cv
    prev = pts[-1]
    ind = of(st)
    v = round(float(prev["value"]) * math.exp(int(plan["sign"]) * float(plan["k"]) * rcv), int(ind.get("ndigits", 2)))
    return {"indicator": st, "unit": ind.get("unit", ""),
            "previous": {"day": int(prev["ts"]), "value": prev["value"]}, "today": {"day": int(T), "value": v}}


def f3_truth(longitudinal: dict, frame_gold_block: dict, T: int) -> dict | None:
    """F3 gold: on a planned case the follow-up pair (last draw, today's result); otherwise the last
    two draws at or before T (`rcv_truth`)."""
    g = frame_gold_block or {}
    fu = f3_followup(longitudinal, g.get("f3_plan"), T)
    if fu is None:
        return None if g.get("f3_plan") else rcv_truth(longitudinal, g.get("followed_stream"), T)
    from ..gated import _obs_cfg
    cv = float((_obs_cfg()[1] or {}).get(fu["indicator"]) or 0.0)
    rcv = 1.96 * math.sqrt(2) * cv
    a, b = float(fu["previous"]["value"]), float(fu["today"]["value"])
    if a <= 0 or b <= 0:
        return None
    d = math.log(b / a)
    return {"stream": fu["indicator"], "prev": a, "last": b, "log_change": round(d, 4),
            "rcv_log": round(rcv, 4), "exceeds_rcv": abs(d) > rcv, "planned_arm": g["f3_plan"].get("arm")}


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
        g["med_direction"] = "investigate_first" if hidden else "uptitrate"
        g["med_rule"] = "hidden named condition => investigate_first; else uptitrate (provisional, profile only)"
    else:
        g["followed_stream"] = f3_stream(meta)
        g["f3_plan"] = lat.get("f3")
        g["rcv_rule"] = ("|ln(v_today / v_prev)| > 1.96 * sqrt(2) * cv_total (physio_streams.yaml cv); "
                         "v_prev = last draw at or before T, v_today = the follow-up result (f3_followup)")
    return g


def frame_probe(meta: dict) -> dict | None:
    """`prediction_context.m2_frame`: frame name and the scene text (template); None on F0."""
    lat = m2_latent(meta)
    fr = lat.get("frame")
    if fr not in ("F1", "F2", "F3"):
        return None
    out = {"frame": FRAME_NAMES[fr], "scene": lat.get("scene") or "", "text_source": TEXT_SOURCE}
    if fr == "F3":
        out["followed_indicator"] = f3_stream(meta)
    return out


def rcv_truth(longitudinal: dict, stream: str | None, T: int) -> dict | None:
    """F3 gold from the world: the last two draws of `stream` at or before T against the 95% RCV."""
    if not stream:
        return None
    from ..gated import _obs_cfg
    pts = sorted((p for p in (longitudinal or {}).get(stream) or []
                  if isinstance(p, dict) and isinstance(p.get("value"), (int, float))
                  and int(p.get("ts", 10 ** 9)) <= int(T)), key=lambda p: int(p["ts"]))
    if len(pts) < 2 or float(pts[-2]["value"]) <= 0 or float(pts[-1]["value"]) <= 0:
        return None
    cv = float((_obs_cfg()[1] or {}).get(stream) or 0.0)
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
def co_medications(case_id: str, raw: dict) -> list[str]:
    """The medicines the world gives for the case's known conditions (`raw.disease`, `raw.comorbidities`):
    the primary drug and `med_course.concurrent_medicines`, the draws the build makes. A hidden line has none."""
    from ..events import shown_drugs
    return shown_drugs(case_id, raw.get("disease"), raw.get("comorbidities") or [], str(raw.get("drug") or ""))


def scene_text(frame: str, raw: dict, latent: dict, case_id: str) -> str:
    """The frame scene, written from the case's own facts by template. F1 presents the latest true
    symptom before T as the reason for today's visit; F2 the treated base condition; F3 a follow-up
    draw; every frame names the medicines the world gives for the known conditions (`co_medications`).
    No fact is added that the case does not already contain."""
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
            what = f"「{sy[-1]['text'] if sy else '不适'}」明显加重"
        return (f"今天(第 {T} 天)患者因{what}来诊。{meds}。请判断分诊紧急度(🔴 立即急诊 / 🟠 数日内 / "
                f"🟡 数周内 / 🟢 常规随访)、首选检查、鉴别诊断,以及是否需要立即请临床医生复核。")
    if frame == "F2":
        return (f"今天(第 {T} 天)是{names.get(disease, disease)}的慢病复诊,{meds}。"
                f"请评估基础病控制情况,给出调药方向(先查因暂不调 / 加量 / 维持 / 减量),"
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
        "\n\n【本题场景:急性来诊】见 `prediction_context.m2_frame.scene`。除上面要求的字段外,另加一个字段:"
        '\n  "urgency": "🔴|🟠|🟡|🟢"   (🔴 立即急诊 · 🟠 数日内 · 🟡 数周内 · 🟢 常规随访)'),
    "chronic_med_adjustment": (
        "\n\n【本题场景:慢病调药】见 `prediction_context.m2_frame.scene`。除上面要求的字段外,另加一个字段:"
        '\n  "treatment_plan": {"direction": "investigate_first|uptitrate|keep|downtitrate", "reason": "..."}'
        "\n  (investigate_first = 先查因、暂不调药;只给方向,不给具体药名剂量 —— 不自主改药的硬约束仍然有效)"),
    "followup_interpretation": (
        "\n\n【本题场景:随访解读】见 `prediction_context.m2_frame.scene` 与 `followed_indicator`;今天的复查结果附在本段末。除上面要求的字段外,另加一个字段:"
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
    "version": "m2-answer-v1",
    "frame_fields": {"acute_triage": ["urgency"], "chronic_med_adjustment": ["treatment_plan.direction"],
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
    from ..overlay import haenv_comorbid_specs, haenv_independent_specs
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


def _clue(case_id: str, comp: str, out: list, mine: list, rec: dict, lines_clue: dict | None = None,
          present: frozenset = frozenset()) -> bool:
    """The line's last symptom becomes one of its registered clues that nothing else on the record
    (`present`: the known conditions and the other gold lines) explains, drawn by case and line; the clue is
    recorded on `lines_clue` (an explained-away line) or in `rec["clues"]` (any other gold line)."""
    from .. import rng
    opts = [t for t in clues().get({v: k for k, v in EA_HIDDEN.items()}.get(comp)) or ()
            if not clue_explained_by().get(t, frozenset()) & present]
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


def explain_away_render(spec_id: str, known: list[str], symptoms: list[dict],
                        T: int | None = None, med_text: str | None = None,
                        case_id: str = "") -> tuple[list[dict], dict]:
    # `known`: the conditions the record names (`user_profile.known_conditions`: the primary
    # disease and the M2 explainer bases); other comorbidities are not named and explain nothing.
    # `med_text`: the case's visible medication text (`visible_medication_text`); a row with
    # `requires_visible_drug` covers nothing without it. Retired rows are skipped (`ea_rows`).
    """(symptoms, record). For each hidden component that the table lets a visible explainer cover,
    every clause of each of its symptoms is checked on its own (`license_atoms`): a symptom whose
    clauses are all licensed by a covered finding (or are context qualifiers) is kept; one with
    some licensed clauses renders those clauses only (its context is dropped unless licensed too);
    one with none is rewritten to the next covered finding not yet on the record (or, when all are
    on the record, the first one); no symptom is dropped (generation rule of spec 4.2: the
    explained-away line shows no specific sign). The patient's wording of every symptom is chosen at
    entry (`events.enter_symptoms`, `registry/symptom_lay.yaml`). Age tokens are dropped from every symptom. The record lists, per symptom of an
    explained-away line, the row, the visible explainer and the clauses removed (read by the audit
    EA-1 and the K-EA profile; judging never reads it)."""
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
    for comp in comps:
        rows = [r for r in ea_rows() if EA_HIDDEN.get(str(r.get("hidden"))) == comp
                and spec_id in (r.get("specs") or ())]
        cov = [(r, visible_explainers(r, comps, known, comp, on_record, med_text)) for r in rows]
        cov = [(r, ex) for r, ex in cov if ex]
        mine = [i for i in range(len(out)) if owner[i] == comp]
        if not cov:
            # a gold line that is not explained away shows a finding that separates it from its nearest
            # look-alike when none of its own symptoms carries a specific sign
            name = {v: k for k, v in EA_HIDDEN.items()}.get(comp)
            if name and len(mine) >= 2 and not any(sg in out[i]["text"] for i in mine for sg in specific_signs().get(name, ())):
                _clue(case_id, comp, out, mine, rec, present=frozenset(known) | (frozenset(comps) - {comp}))
            continue
        terms = [(f, r, ex) for r, ex in cov for f in r.get("findings") or ()]
        line = {"component": comp, "rows": [r["id"] for r, _ in cov], "symptoms": []}
        used = {f for i in range(len(out)) for f, _, _ in terms
                if any(atom_licensed(a, f) for a in symptom_atoms(out[i]["text"]))}
        for i in mine:
            s = out[i]
            lic = license_atoms(s["text"], terms)
            ok = [(a, h) for a, k, h in lic if k == "licensed"]
            bad = [a for a, k, _ in lic if k == "unlicensed"]
            ctx = str(s.get("context") or "")
            ctx_bad = [a for a, k, _ in license_atoms(ctx, terms) if k == "unlicensed"]
            rewritten = False
            src = s["text"]
            hit = ok[0][1] if ok else None
            if ok and bad:
                s["text"] = "、".join(a for a, _ in ok)
                rewritten = True
            elif not ok:
                # Never drop a symptom (with every covered finding already on the record, one is shown again).
                hit = ([(f, r, ex) for f, r, ex in terms if f not in used] or terms)[0]
                s["text"] = hit[0]
                rewritten = True
            if ctx and (ctx_bad or (rewritten and not ok)):
                s["context"] = ""
            if hit:
                used.update(f for f, _, _ in terms if any(atom_licensed(a, f) for a in symptom_atoms(s["text"])))
            line["symptoms"].append({"day": s.get("day"), "text": s["text"],
                                     "row": hit[1]["id"] if hit else None,
                                     "explainer": hit[2] if hit else [], "rewritten": rewritten,
                                     "source": src, "removed": bad + ctx_bad})
        if len(line["symptoms"]) >= 2 and _clue(case_id, comp, out, mine, rec, line,
                                                  frozenset(known) | (frozenset(comps) - {comp})):    # the line's last symptom is its clue
            line["symptoms"].pop()
        if line["symptoms"]:
            rec["lines"].append(line)
    return out, rec
