"""Render a trajectory as clinical-record text (the inverse of `extract.py`).

`render_record` reads only the solver payload `sp`, so a rendered record cannot contain
hidden variables or gold, and "render then re-extract" measures whether the trajectory
itself supports the patient's facts. `leak_words()` rejects records whose wording states
the answer.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("haenv.record_render")

#: Words whose presence in the prose is an automatic fail: conclusion words (the answer
#: itself) and gold field names (which can only come from `vp`). Neutral indicator names are
#: not listed, because records are supposed to mention indicators.
LEAK_WORDS: tuple[str, ...] = (
    # conclusion words (= the answer to the question)
    "体重回升", "体重反弹", "减重失败", "依从性差", "依从性不佳", "停药",
    "应答不佳", "低应答", "无应答", "热量摄入增加", "热量缺口",
    # gold field names (can only have come from vp)
    "outcome_label", "gold_drivers", "adjudication", "reversal_point",
    "latent_premise", "label_rule", "future_data", "primary_driver",
)

_PROMPT = """你是一名内分泌科医生,正在为一位随访患者书写门诊病历。

下面是这位患者在随访期内的**可获得资料**(设备与化验读数、就诊记录片段)。
请据此写一份中文门诊病历,包含:主诉与现病史、用药与剂量调整经过、
客观检查(把关键读数写进去,含日期)、以及**当前情况的客观描述**。

🔴 **三条硬要求**:
1. **只写资料里有的**。资料里没有的数字、日期、症状一律不许出现。
2. **不写结论、不写预测、不写评估意见** —— 不许出现「回升」「反弹」「依从性差」
   「应答不佳」「热量摄入增加」这类判断性措辞。你只负责**记录**,不负责下结论。
3. 不许出现任何英文字段名。

资料:
```json
{payload}
```

直接输出病历正文,不要任何前后缀说明。"""


def leak_words(text: str) -> list[str]:
    """Leak words matched in the prose; empty means clean. Case-insensitive."""
    low = text.lower()
    return [w for w in LEAK_WORDS if (w.lower() in low)]


def _trim(sp: dict, max_points: int = 60) -> dict:
    """Shrink the payload to fit one prompt by even sampling, so the full time span survives."""
    out = {k: v for k, v in sp.items() if k != "longitudinal_data"}
    ld = {}
    for sig, pts in (sp.get("longitudinal_data") or {}).items():
        if not isinstance(pts, list):
            continue
        if len(pts) <= max_points:
            ld[sig] = pts
        else:
            step = len(pts) / max_points
            ld[sig] = [pts[int(i * step)] for i in range(max_points)]
    out["longitudinal_data"] = ld
    return out


def render_record(sp: dict, *, model: str = "gemini-3.8-flash",
                  cfg: dict | None = None, offline: bool = False) -> tuple[str, str]:
    """`sp` -> (record text, provenance tag).

    Takes only `sp` so the truth cannot be passed in. The provenance tag uses
    `extract._complete`'s states (`replay:` / `cache:` / `llm-extract:`).
    """
    from . import extract as _ex
    if not cfg:
        from haenv import data_root

        from .yamlcache import load_yaml as _cached
        cfg = _cached(data_root() / "config.yaml") or {}
    prompt = _PROMPT.format(payload=json.dumps(_trim(sp), ensure_ascii=False))
    # Separate cache: rendered records are prose, while the generation cache expects JSON
    # (`poisoned_cache_scan` relies on that).
    text, origin = _ex._complete(prompt, model, cfg, offline=offline,
                                 cache_subdir="_record_cache")
    hits = leak_words(text)
    if hits:
        log.warning("[record_render] %s rendered record hit leak word(s) %s -- unusable",
                    sp.get("case_id"), hits)
    return text, origin


#: Patient facts expected to come back out of a rendered record; the round-trip recovery rate
#: is computed over these fields only. Generation knobs and inferred gold are excluded.
ROUNDTRIP_FIELDS: tuple[str, ...] = (
    "age_range", "sex", "drug", "dose_steps", "devices",
    "start_weight", "nadir_weight",
)


def compare_facts(want: dict, got: dict) -> dict:
    """Field-by-field comparison; returns `{field: bool}` plus `n_ok / n_total`.

    Numbers match within 1 kg, lists as sets, everything else as whitespace-stripped strings.
    """
    res: dict[str, bool] = {}
    for f in ROUNDTRIP_FIELDS:
        a, b = want.get(f), got.get(f)
        if a is None:
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            res[f] = abs(float(a) - float(b)) <= 1.0
        elif isinstance(a, (list, tuple)):
            res[f] = set(map(str, a)) == set(map(str, b or []))
        else:
            res[f] = re.sub(r"\s+", "", str(a)) == re.sub(r"\s+", "", str(b or ""))
    n = len(res)
    return {"per_field": res, "n_ok": sum(res.values()), "n_total": n,
            "rate": (sum(res.values()) / n) if n else None}
