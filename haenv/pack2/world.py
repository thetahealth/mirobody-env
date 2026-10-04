"""Pack 2 world glue.

The three world mechanisms are separate modules -- `vitals.py` (T-time vital-sign snapshot,
point-of-care labs, 72 h logs, the visible block), `events.py` (structured acute-event pool and
wording, `registry/pack2_acute_events.yaml`) and `grading.py` (NEWS2 and the grading table G2,
`registry/pack2_triage_thresholds.yaml`, provisional K3). This module joins them into the two
registered sides (`gold_block` -> `adjudication.triage`, `probe_block` ->
`prediction_context.triage`), re-arms the verifier fields (`finalize_adjudication`) and declares
the per-case checks of the shared emission gate (`emission`).

Everything is a pure function of the case meta (case id, T, the `pack2` latent plan, the hidden
spec), so gold, question, gate and the composer (`tools/pack2_gen_job.py`) compute the same
snapshot. The gold is G2 on the values the item shows, never the plan; the gate refuses a case
whose recomputed G2 differs from the plan.

Stamps: all of these files are the PACK2_WORLD segment (`tools/make_freeze.py`). Its fingerprint
is written into every pack-2 gold (`world_sha16`) and into the registered probe
(`plugin.register`), whose hash enters the external-gold manifest and hence `world_sha`, so a
change to any pack-2 world file moves the pack-2 `world_sha` and nothing else.

Imports stay light (yaml + stdlib): the composer and the audit run it under the sklearn python.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re

# The three world mechanisms live in their own modules; their names are re-exported here so
# callers (gate, audit, composer, tests) keep one import.
from ._util import ROOT, pick as _pick, registry_table, u01  # noqa: F401
from .events import (EVENTS, FOCALS, WORDINGS, chief_complaint, distractor_results,  # noqa: F401
                     eligible_events, events_table)
from .grading import (DISPOSITIONS, TABLE, class_edges, g2, margin_units, news2,  # noqa: F401
                      rank_of, thresholds, urgency_of)
from .vitals import FOCAL_FIELDS, snapshot, values_from_visible, visible_block  # noqa: F401

LATENT_KEY = "pack2"


BLOCK = "triage"


#: Words that would name a disposition in the item text (L2-7). The question template states the
#: four options once for every item; this scan covers the per-item text only.
DISPOSITION_WORDS = ("ed_now", "within_24h", "routine_followup", "watchful_waiting",
                     "急诊", "急救", "立即就医", "马上就医", "马上去医院", "24小时", "24 小时",
                     "常规随访", "继续观察", "无需就医", "不用就医", "危急值", "红旗")


#: Copied from `haenv/verify.py:RED_FLAG_WORDS` (that module is not import-light); a test pins
#: the two tuples equal.
RED_FLAG_WORDS = ("胸痛", "胸闷压迫", "呼吸困难", "喘不上气", "晕厥", "意识丧失", "抽搐",
                  "偏瘫", "半身无力", "言语不清", "黑便", "呕血", "咯血", "血尿", "剧烈头痛",
                  "视力骤降", "视物模糊加重", "高热不退", "持续呕吐", "腹痛剧烈", "自杀")


def tables_sha16() -> str:
    blob = json.dumps({"t": thresholds(), "e": events_table()}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@functools.lru_cache(maxsize=4)
def _segment_fp_cached(files: tuple, key: tuple) -> str:
    from haenv.anchor import _fingerprint_of
    return _fingerprint_of(files)


def segment(name: str) -> tuple[str, ...]:
    """A segment of `tools/make_freeze.py` (`PACK2_WORLD` / `PACK2_JUDGING`)."""
    from haenv.pack_skeleton import segment as seg
    return seg(name)


def world_fingerprint() -> str:
    """Semantic fingerprint of the PACK2_WORLD segment (same rule as `world_sha`)."""
    files = segment("PACK2_WORLD")
    from haenv.anchor import _fingerprint_path
    paths = [(f, _fingerprint_path(f)) for f in files]
    key = tuple((f, p.stat().st_mtime_ns, p.stat().st_size) for f, p in paths)
    return _segment_fp_cached(files, key)


def _hidden(meta: dict) -> int:
    d = meta.get("ddx") or {}
    return int(d.get("join_gold") in ("unified", "comorbidity"))


def plan_of(meta: dict) -> dict | None:
    ext = (meta or {}).get("_external") or {}
    p = ext.get(LATENT_KEY)
    return p if isinstance(p, dict) and p.get("disposition") else None


def gold_block(meta: dict) -> dict | None:
    """`adjudication.triage`: G2 on the shown values plus the plan and its provenance."""
    plan = plan_of(meta)
    if plan is None:
        return None
    cid, T = str(meta.get("case_id")), int(meta.get("prediction_time_T"))
    snap = snapshot(cid, T, plan)
    vis = visible_block(cid, T, plan, snap)
    vals = values_from_visible(vis)
    s, single3 = news2(vals)
    disp = g2(vals)
    lows = sum(1 for x in vals["glucose_log"] if x < thresholds()["rules"]["glucose_low"])
    ddx = meta.get("ddx") or {}
    return {"pack": "pack2", "status": "provisional", "disposition": disp,
            "planned": plan["disposition"], "focal": plan["focal"],
            "focal_fields": list(FOCAL_FIELDS[plan["focal"]]),
            "values": {k: vals[k] for k in ("rr", "spo2", "sbp", "hr", "temp", "glucose", "K")},
            "news2": s, "single_param_3": single3, "glucose_lows_72h": lows,
            "glucose_nadir_72h": min(vals["glucose_log"]) if vals["glucose_log"] else None,
            "hidden": _hidden(meta), "event": plan["event"], "cause": plan.get("cause"),
            "explains_away": bool(plan.get("explains_away")), "wording": plan["wording"],
            "distractors": int(plan.get("distractors", 0)), "prior_tier": plan.get("prior_tier"),
            "spec": {"urgency": ddx.get("urgency"), "red_flag": bool(meta.get("red_flag", False)),
                     "clinician_warranted": meta.get("clinician_warranted")},
            "tables_sha16": tables_sha16(), "world_sha16": world_fingerprint()}


def probe_block(meta: dict) -> dict | None:
    plan = plan_of(meta)
    if plan is None:
        return None
    return visible_block(str(meta.get("case_id")), int(meta.get("prediction_time_T")), plan)


def finalize_adjudication(adj: dict) -> None:
    """Re-arm the verifier fields from the world gold (in place): `red_flag_present`,
    `clinician_action_warranted` and `ddx.urgency` follow G2. Called by the build hook."""
    t = adj.get(BLOCK)
    if not isinstance(t, dict) or t.get("pack") != "pack2":
        return
    disp = t["disposition"]
    adj["red_flag_present"] = disp == "ed_now"
    adj["clinician_action_warranted"] = disp != "watchful_waiting"
    if isinstance(adj.get("ddx"), dict):
        adj["ddx"] = {**adj["ddx"], "urgency": urgency_of(disp)}


_DIGIT = re.compile(r"[0-9０-９]")


def leak_hits(block: dict) -> list[str]:
    """L2-7 on the per-item visible text."""
    out = []
    blob = json.dumps(block, ensure_ascii=False)
    for w in DISPOSITION_WORDS:
        if w in blob:
            out.append(f"disposition word {w!r} in the visible block")
    cc = str(block.get("chief_complaint") or "")
    if _DIGIT.search(cc):
        out.append(f"digit in the chief complaint: {cc!r}")
    for w in RED_FLAG_WORDS:
        if w in cc:
            out.append(f"red-flag word {w!r} in the chief complaint")
    return out


def margin_hits(gold: dict) -> list[str]:
    tb = thresholds()
    m = tb["margins"]
    v = gold["values"]
    focal = gold["focal"]
    fq = {"K": {"K"}, "SBP": {"sbp"}, "glucose": {"glucose"},
          "NEWS2": set(tb["focal_values"]["NEWS2"].get(gold["planned"], {}))}[focal]
    out = []
    for q in ("rr", "spo2", "sbp", "hr", "temp", "glucose", "K"):
        need = m["focal_min_units"] if q in fq else m["background_min_units"]
        d = margin_units(q, v[q], tb)
        if d < need:
            out.append(f"{q}={v[q]} is {d} print units from a class edge (< {need})")
    if focal == "glucose" and gold["planned"] in ("within_24h", "routine_followup"):
        lo_edge = tb["rules"]["glucose_low"] - tb["margins"]["units"]["glucose"] / 2
        nad = gold.get("glucose_nadir_72h")
        if nad is not None and (lo_edge - nad) / tb["margins"]["units"]["glucose"] - 0.5 < m["focal_min_units"]:
            out.append(f"glucose nadir {nad} too close to {tb['rules']['glucose_low']}")
    return out


def _plan_of_case(raw) -> dict | None:
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    return plan_of((lp or {}).get("meta") or {})


def _recompute(gold: dict, raw) -> list[str]:
    """G2 on the values the item shows equals the gold and the plan; the verifier fields follow G2."""
    adj = (raw.get("adjudication") if isinstance(raw, dict) else getattr(raw, "adjudication", None)) or {}
    pc = (raw.get("prediction_context") if isinstance(raw, dict) else getattr(raw, "prediction_context", None)) or {}
    vis = pc.get(BLOCK)
    if not isinstance(vis, dict):
        return []
    vals = values_from_visible(vis)
    disp = g2(vals)
    out = []
    if disp != gold["disposition"] or disp != gold["planned"]:
        out.append(f"G2 on the shown values = {disp}, gold = {gold['disposition']}, plan = {gold['planned']}")
    out += [f"shown {k}={vals[k]} differs from gold value {gold['values'][k]}"
            for k in ("rr", "spo2", "sbp", "hr", "temp", "glucose", "K") if vals[k] != gold["values"][k]]
    if adj.get("red_flag_present") != (gold["disposition"] == "ed_now"):
        out.append(f"red_flag_present={adj.get('red_flag_present')} but G2={gold['disposition']}")
    if adj.get("clinician_action_warranted") != (gold["disposition"] != "watchful_waiting"):
        out.append(f"clinician_action_warranted={adj.get('clinician_action_warranted')} but G2={gold['disposition']}")
    if isinstance(adj.get("ddx"), dict) and adj["ddx"].get("urgency") != urgency_of(gold["disposition"]):
        out.append(f"ddx.urgency={adj['ddx'].get('urgency')} but G2={gold['disposition']}")
    return out


def emission():
    """Pack 2's declaration for the shared emission gate (`haenv/shared_audit.py`, SA-1 + SA-8):
    G2 recompute and the re-armed verifier fields, the L2-4 margins, the L2-7 words of the visible
    block, the block in the solver payload."""
    from haenv.shared_audit import Emission
    return Emission(gate="pack2", block=BLOCK, plan_of=_plan_of_case, recompute=_recompute,
                    class_of=lambda g: g.get("disposition"), planned_class=lambda p: p.get("disposition"),
                    leak=leak_hits, extra=lambda gold, raw: [("margin", d) for d in margin_hits(gold)])
