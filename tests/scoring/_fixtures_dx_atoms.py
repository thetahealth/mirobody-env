"""The diagnosis atom `dx_listed` (on the differential and not ruled out), and the two
matcher fixes it rests on: pending-exclusion phrasing is not a negation, and a
finding-level word does not by itself name a comorbidity thread.

Every rule is checked from both sides.

SYNTHETIC data, evaluation only, not medical advice.
"""

from __future__ import annotations
from _paths import ROOT as _TEST_ROOT

import pathlib

import sys

import pytest

import yaml

ROOT = _TEST_ROOT

sys.path.insert(0, str(ROOT))

_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

import haenv.world.overlay as overlay

import haenv.scoring.quantities as quantities                        # noqa: E402

from haenv.judges import SINGLE, run_judges                  # noqa: E402

from haenv.scoring.legacy import load_profile                       # noqa: E402

from haenv.scoring.tracks import alias_hit_asserted

from haenv.scoring.tracks import thread_aliases  # noqa: E402

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

HYPO = ["甲减", "甲状腺功能减退", "hypothyroid", "桥本"]

PHPT = ["甲状旁腺", "甲旁亢", "高钙", "hyperparathyroid", "pth"]

COM_PHPT = {"diagnosis": "原发性甲状腺功能减退(桥本) + 原发性甲状旁腺功能亢进(高钙血症)",
            "aliases": HYPO + PHPT, "join_gold": "comorbidity",
            "threads": [{"name": "甲状腺线", "aliases": HYPO}, {"name": "甲状旁腺线", "aliases": PHPT}]}

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

def _finding_table() -> dict:
    return yaml.safe_load((ROOT / "registry/threads.yaml").read_text(encoding="utf-8"))["finding_aliases"]

def _registry_thread(sid: str, name: str) -> list[str]:
    spec = overlay.condition_registry().get(sid)
    for t in overlay.threads_for(sid, spec) or ():
        if t["name"] == name:
            return list(t["aliases"])
    raise AssertionError(f"{sid} has no thread {name}")

_FILLER = ["原发性醛固酮增多症", "库欣综合征", "肢端肥大症", "系统性红斑狼疮", "结节病", "乳糜泻",
           "多囊卵巢综合征", "阻塞性睡眠呼吸暂停", "维生素B12缺乏", "遗传性血色病"]

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

LONG_LIST_CEILING = 0.3


__all__ = [name for name in globals() if not name.startswith("__")]
