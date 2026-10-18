"""Reusable archival immune/genetic findings change the evidence acquisition location.

No diagnosis name is printed. Current activity still needs the remaining current
checks. Archived source identifiers are credited only after their rows return.
"""
from __future__ import annotations
from copy import deepcopy
from haenv.world.history.history_dependency import HistoryDependency
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
BLOCK = "m2_frame"


def prepare(raw, cid, plan):
    return None


def finalize(raw, cid, plan):
    from haenv.m2.core import threads_of
    from haenv.m2.core import _on_demand
    framed = BLOCK in raw.adjudication
    g = raw.adjudication.get(BLOCK) or raw.adjudication
    ddx = raw.adjudication.get("ddx") or {}
    T = int(raw.prediction_context["prediction_time_T"])
    reusable = set(table()["packs"]["m2"]["reusable_findings"])
    keys = sorted({k for t in threads_of(ddx) for k in t["key_signals"]} & reusable)
    active = framed and bool(keys) and uniform("history-role", cid) < rate("m2", plan)
    fid = keys[0] if keys else sorted(reusable)[int(uniform("history-item", cid) * len(reusable))]
    phr = table()["phrasing"]
    src = phr["encounter_sources"][int(uniform("history-source", cid) * len(phr["encounter_sources"]))]
    from haenv.resources.registry import load_findings
    spec = load_findings()[fid]
    result = _on_demand(raw, fid, T)
    typing_positive = fid == "HLAB27" and any(r.get("flag") == "POSITIVE" for r in result or ())
    day = old_day("m2", cid, T)
    if result:
        result = [{**x, "ts": day} for x in result]
    if not active:
        result = [{"ts": day, "value": "阴性", "flag": "NEGATIVE", "finding_id": fid, "name": spec.get("name_cn", fid)}]
        if typing_positive:
            # A genotype does not seroconvert. An unusable earlier typing
            # report can be inert without contradicting a current positive.
            result[0].update(value="标本质控不足，无法提供可解释的分型", flag="UNINTERPRETABLE")
    # A historical result is retained as a full laboratory report, with the
    # original finding vocabulary, not an inferred diagnosis or a role label.
    row = {"day": old_day("m2", cid, T), "source": src, "finding": fid,
           "name": spec.get("name_cn", fid), "observation": result,
           "result": "既往阳性，报告可核对" if active else "当时初筛可疑，复查阴性；原报告记录复查结果。",
           "followup": "当时未进一步评估，当前症状需另行评估。" if active else "后续未有同类持续异常。"}
    if not active and typing_positive:
        row.update(result="当时初筛可疑，分型未获可解释结果；原报告记录标本质量问题。",
                   followup="当时未完成有效分型，当前需另行评估。")
    if active and not result:
        active = False
        row.update(observation=[], result="当时结果无法解释，后复查阴性。", followup="未有持续异常。")
    raw.prediction_context["archived_laboratory_record"] = [row]
    raw.prediction_context["record_policy"] = table()["packs"]["m2"]["protocol"]
    if active:
        ddx["historical_key_signals"] = {fid: "archive:" + fid}
    g["history_dependency"] = {"active": active, "recent_lower": T - 365,
                               "records": [{"section": "archived_laboratory_record", "day": row["day"]}],
                               "source": table()["packs"]["m2"]["source"]}


def replacements(ddx, thread):
    swap = ddx.get("historical_key_signals") or {}
    return {**thread, "key_signals": [swap.get(k, k) for k in thread["key_signals"]]}


def answer(raw):
    from haenv.m2.core import threads_of
    ddx = deepcopy(raw["adjudication"].get("ddx") or {})
    rows = raw["prediction_context"].get("archived_laboratory_record") or ()
    ddx["historical_key_signals"] = {k[8:]: k for k in archive_keys(raw["prediction_context"])}
    return sorted({k for t in threads_of(ddx) for k in t["key_signals"]})


def twin(raw, day=None):
    out = {**raw, "prediction_context": deepcopy(raw["prediction_context"])}
    out["prediction_context"]["archived_laboratory_record"] = [r for r in out["prediction_context"].get("archived_laboratory_record", ()) if day is not None and int(r["day"]) != day]
    return out


def credit(row, raw, solver):
    ddx = raw.adjudication.get("ddx") or {}
    if not ddx.get("historical_key_signals"):
        return
    from haenv.interaction.query.profile import returned_rows
    from haenv.m2.core import threads_of
    trace = (getattr(solver, "_last_query", None) or ({},))[0]
    seen = {"archive:" + r["finding"] for _, r in returned_rows(trace)
            if r.get("_from") == "archived_laboratory_record" and r.get("observation") and r.get("finding")
            and r.get("result") == "既往阳性，报告可核对"}
    targets = list(row.get("tool_targets") or ()) + sorted(seen)
    row["m2_historical_key_seen"] = sorted(seen)
    from haenv.scoring.tracks import key_coverage
    keys = sorted({k for t in threads_of(ddx) for k in t["key_signals"]})
    row.update(key_coverage(targets, keys, committed=True))


def plan_of(raw):
    from haenv.contracts.external_gold import SLOT
    slot = (((raw.latent_premise or {}).get("meta") or {}).get(SLOT) or {})
    if slot.get("m2"):
        return slot["m2"]
    return {} if slot.get("m2_bridge") else None


def ready(raw):
    from haenv.m2.core import is_m2_case
    return is_m2_case(raw)


POLICY = HistoryDependency(BLOCK, plan_of, prepare, finalize, answer, twin, ready)


def archive_keys(pc):
    return {"archive:" + r["finding"] for r in pc.get("archived_laboratory_record") or ()
            if r.get("observation") and r.get("finding") and r.get("result") == "既往阳性，报告可核对"}
