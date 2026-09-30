"""Two matcher rules on the diagnosis atom `dx_listed`:

1. judging-side synonyms: a diagnosis the gold names in one wording and the model in another
   ("原发性肾上腺皮质功能不全" for "肾上腺皮质功能减退", "Pancreatic adenocarcinoma" for
   "胰腺外分泌肿瘤") counts; each synonym comes with the exclusions that keep it off the
   registered rivals ("继发性肾上腺皮质功能不全");
2. negation scope ends at a clause boundary: in "不，需排除 X" the "不" answers something
   else, and X is still on the differential; "不需排除 X" stays a negation.

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

from haenv import overlay                                    # noqa: E402
from haenv.judges import SINGLE, run_judges                  # noqa: E402
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


def _unified(sid: str) -> dict:
    spec = overlay.condition_registry()[sid]
    return {"diagnosis": spec["diagnosis"], "aliases": list(spec["aliases"]), "join_gold": "unified"}


def _comorbid(sid: str) -> dict:
    spec = overlay.condition_registry()[sid]
    return {"diagnosis": spec["diagnosis"], "aliases": list(spec["aliases"]), "join_gold": "comorbidity",
            "threads": [dict(t) for t in overlay.threads_for(sid, spec)]}


def _thread(sid: str, name: str) -> list[str]:
    spec = overlay.condition_registry()[sid]
    for t in overlay.threads_for(sid, spec) or ():
        if t["name"] == name:
            return list(t["aliases"])
    raise AssertionError(f"{sid} has no thread {name}")


def _listed(sid: str, text: str) -> float:
    return run_judges(SINGLE, _diff(text, "焦虑状态"), FakeVP(_unified(sid)))["dx_listed"]


# ---------------------------------------------------------------- negation scope
PHEO = ["嗜铬细胞瘤", "pheochromocytoma", "pheo"]


@pytest.mark.parametrize("text", [
    "不，需排除嗜铬细胞瘤",
    "不,需排除嗜铬细胞瘤",
    "不；需要排除嗜铬细胞瘤",
    "需排除嗜铬细胞瘤",
    "no; pheochromocytoma should be ruled out",
    "No, pheochromocytoma needs to be ruled out",
])
def test_negator_in_an_earlier_clause_does_not_reach_a_pending_exclusion(text):
    assert alias_hit_asserted(text, PHEO), text


@pytest.mark.parametrize("text", [
    "不需排除嗜铬细胞瘤",
    "无需排除嗜铬细胞瘤",
    "排除了嗜铬细胞瘤",
    "已排除嗜铬细胞瘤",
    "嗜铬细胞瘤已排除",
    "不，已排除嗜铬细胞瘤",
    "no need to rule out pheochromocytoma",
    "pheochromocytoma ruled out",
])
def test_negation_within_the_clause_still_negates(text):
    assert not alias_hit_asserted(text, PHEO), text


def test_pending_after_a_bare_no_counts_as_listed():
    vp = FakeVP({"diagnosis": "嗜铬细胞瘤", "aliases": PHEO, "join_gold": "unified"})
    r = run_judges(SINGLE, _diff("原发性高血压", "不，需排除嗜铬细胞瘤"), vp)
    assert r["dx_listed"] == 1.0 and r["dx_hit"] is True, r
    r = run_judges(SINGLE, _diff("原发性高血压", "不需排除嗜铬细胞瘤"), vp)
    assert r["dx_listed"] == 0.0 and r["dx_hit"] is False, r


# ---------------------------------------------------------------- synonyms, unified cases
@pytest.mark.parametrize("sid,text", [
    ("JD-ADDISON", "原发性肾上腺皮质功能不全（待确诊）"),
    ("JD-ADDISON", "疑似原发性肾上腺功能不全，尚待紧急排查"),
    ("JD-ADDISON", "Primary adrenal insufficiency"),
    ("HD-UNI-02", "Systemic sclerosis (likely limited cutaneous)"),
    ("JD-OSA", "Obstructive sleep apnea / sleep-disordered breathing"),
    ("HD-UNI-06", "Pancreatic adenocarcinoma"),
    ("HD-UNI-06", "Pancreatic ductal adenocarcinoma (body/tail)"),
    ("HD-UNI-06", "Pancreatic malignancy (Pancreatic cancer)"),
    ("HD-UNI-06", "胰腺恶性肿瘤（胰腺癌可能）"),
    ("HD-UNI-06", "胰腺导管腺癌（或胰腺神经内分泌肿瘤）导致内、外分泌功能不全"),
    ("JD-LADA", "糖尿病(空腹胰岛素极低提示自身免疫性糖尿病可能,需与T2DM鉴别)"),
    ("JD-LADA", "Autoimmune diabetes of adults"),
])
def test_synonym_names_the_gold(sid, text):
    assert _listed(sid, text) == 1.0, (sid, text)


@pytest.mark.parametrize("sid,text", [
    # the registered rival of Addison, in the synonym wordings
    ("JD-ADDISON", "继发性肾上腺皮质功能不全"),
    ("JD-ADDISON", "中枢性肾上腺功能不全"),
    ("JD-ADDISON", "Secondary adrenal insufficiency"),
    # neither the neuroendocrine tumour nor chronic pancreatitis is a pancreatic exocrine cancer
    ("HD-UNI-06", "胰腺神经内分泌肿瘤"),
    ("HD-UNI-06", "慢性胰腺炎所致糖尿病"),
    # negation still applies to a synonym
    ("HD-UNI-06", "已排除胰腺癌"),
    ("JD-ADDISON", "不支持肾上腺皮质功能不全"),
])
def test_synonym_does_not_reach_rivals_or_negations(sid, text):
    assert _listed(sid, text) == 0.0, (sid, text)


# ---------------------------------------------------------------- synonyms, threads
@pytest.mark.parametrize("sid,thread,text,want", [
    ("JD-CKM", "肾脏线", "非糖尿病病因的慢性肾脏病伴容量过负荷", True),
    ("JD-CKM", "肾脏线", "Chronic kidney disease, stage 3", True),
    # diabetic nephropathy is carried by the glucose thread (`vocab_alias_excludes.yaml: 肾病`)
    ("JD-CKM", "肾脏线", "2型糖尿病伴糖尿病肾病", False),
    ("JD-CKM", "肾脏线", "Diabetic kidney disease", False),
    # autoimmune diabetes is the LADA thread, not the type 2 glucose thread
    ("JD-CKM", "糖代谢线", "自身免疫性糖尿病可能", False),
    ("JD-CKM", "糖代谢线", "2型糖尿病失控", True),
    ("HD-COM-62", "胰岛自身免疫线", "自身免疫性糖尿病可能,需与T2DM鉴别", True),
    # "1型" is type 1 diabetes, not multiple endocrine neoplasia type 1
    ("HD-COM-20", "胰岛自身免疫线", "多发性内分泌腺瘤病1型 (MEN1)", False),
    ("HD-COM-20", "胰岛自身免疫线", "新发1型糖尿病", True),
    ("HD-COM-10", "肾上腺皮质线", "疑似原发性肾上腺功能不全", True),
    ("HD-COM-10", "肾上腺皮质线", "继发性肾上腺皮质功能不全", False),
    ("HD-COM-54", "呼吸线", "Obstructive sleep apnea with nocturnal hypoxemia", True),
    # The Chinese `糖尿病肾病` names the glucose thread through `糖尿病`; the English form does the same
    ("JD-CKM", "糖代谢线", "Diabetic Nephropathy (Diabetic Kidney Disease) with fluid overload", True),
    ("JD-CKM", "糖代谢线", "2型糖尿病伴糖尿病肾病", True),
    # "类狼疮表现" is lupus-like features, not lupus
    ("HD-COM-62", "自身免疫线", "系统性自身免疫/风湿性疾病（如未分化结缔组织病早期或类狼疮表现）", False),
    ("HD-COM-62", "自身免疫线", "系统性红斑狼疮伴狼疮肾炎", True),
])
def test_synonyms_on_registered_threads(sid, thread, text, want):
    assert alias_hit_asserted(text, thread_aliases(_thread(sid, thread))) is want, (sid, thread, text)


def test_sleep_apnea_synonym_does_not_name_the_lupus_thread():
    assert not alias_hit_asserted("Obstructive sleep apnea", thread_aliases(_thread("HD-COM-01", "自身免疫线")))


# ---------------------------------------------------------------- the table itself
def _table() -> dict:
    return yaml.safe_load((ROOT / "registry/threads.yaml").read_text(encoding="utf-8"))["judging_synonyms"]


def test_synonym_entries_are_explained_and_well_formed():
    tab = _table()
    assert tab
    for anchor, e in tab.items():
        assert str((e or {}).get("why") or "").strip(), f"{anchor}: missing why"
        assert set(e) <= {"why", "names"}, f"{anchor}: unregistered fields {sorted(set(e) - {'why', 'names'})}"
        assert e.get("names"), f"{anchor}: no names"


def test_every_synonym_anchor_is_a_registered_alias():
    """An anchor no spec or thread carries is dead weight (and hides a typo)."""
    used = set()
    for sid, spec in overlay.condition_registry().items():
        used |= {str(a).lower() for a in spec.get("aliases") or ()}
        for t in overlay.threads_for(sid, spec) or ():
            used |= {str(a).lower() for a in t["aliases"]}
    dead = sorted(a for a in _table() if a.lower() not in used)
    assert not dead, f"judging_synonyms anchors no spec carries: {dead}"


def test_no_synonym_names_a_registered_rival():
    """A synonym that matched a rival of its own spec would score the look-alike as the gold."""
    tab = {a.lower(): e["names"] for a, e in _table().items()}
    bad = []
    for sid, spec in overlay.condition_registry().items():
        carried = {str(a).lower() for a in spec.get("aliases") or ()}
        for t in overlay.threads_for(sid, spec) or ():
            carried |= {str(a).lower() for a in t["aliases"]}
        syn = [s for a in carried for s in tab.get(a, ())]
        if not syn:
            continue
        for r in overlay.rivals_for(sid, spec) or ():
            if alias_hit_asserted(r["name"], syn):
                bad.append((sid, r["name"]))
    assert not bad, bad


def test_synonyms_stay_on_the_judging_side():
    """The kernel aliases (generation side, also the leak-scan words) do not carry them."""
    syn = {s.lower() for e in _table().values() for s in e["names"]}
    for sid, spec in overlay.condition_registry().items():
        leaked = syn & {str(a).lower() for a in spec.get("aliases") or ()}
        assert not leaked, (sid, leaked)
