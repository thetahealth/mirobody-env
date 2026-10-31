"""Reusable archival immune/genetic findings change the evidence acquisition location.

No diagnosis name is printed. Current activity still needs the remaining current
checks. Archived source identifiers are credited only after their rows return.
"""
from __future__ import annotations

from haenv.i18n.text import L, is_text
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
        result = [{"ts": day, "value": L("m2.obs.negative"), "flag": "NEGATIVE", "finding_id": fid, "name": spec.get("name_cn", fid)}]
        if typing_positive:
            # A genotype does not seroconvert. An unusable earlier typing
            # report can be inert without contradicting a current positive.
            result[0].update(value=L("m2.obs.typing_unusable"), flag="UNINTERPRETABLE")
    # A historical result is retained as a full laboratory report, with the
    # original finding vocabulary, not an inferred diagnosis or a role label.
    row = {"day": old_day("m2", cid, T), "source": src, "finding": fid,
           "name": spec.get("name_cn", fid), "observation": result,
           "result": L("m2.hist.result.prior_positive" if active else "m2.hist.result.screen_then_negative"),
           "followup": L("m2.hist.followup.not_evaluated" if active else "m2.hist.followup.no_persisting")}
    if not active and typing_positive:
        row.update(result=L("m2.hist.result.typing_failed"),
                   followup=L("m2.hist.followup.typing_incomplete"))
    if active and not result:
        active = False
        row.update(observation=[], result=L("m2.hist.result.uninterpretable_then_negative"), followup=L("m2.hist.followup.none"))
    raw.prediction_context["archived_laboratory_record"] = [row]
    raw.prediction_context["record_policy"] = table()["packs"]["m2"]["protocol"]
    if active:
        ddx["historical_key_signals"] = {fid: "archive:" + fid}
    if "m2_evidence" in raw.adjudication:
        raw.adjudication["m2_evidence"]["threads"] = threads_of(ddx)
    g["history_dependency"] = {"active": active, "recent_lower": T - 365,
                               "records": [{"section": "archived_laboratory_record", "day": row["day"]}],
                               "source": table()["packs"]["m2"]["source"]}


def replacements(ddx, thread):
    from haenv.m2.evidence import with_paths
    return with_paths(ddx, thread)


def answer(raw):
    from haenv.m2.core import threads_of
    ddx = deepcopy(raw["adjudication"].get("ddx") or {})
    rows = raw["prediction_context"].get("archived_laboratory_record") or ()
    ddx["historical_key_signals"] = {k[8:]: k for k in archive_keys(raw["prediction_context"], ddx)}
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
            and ddx.get("historical_key_signals", {}).get(r["finding"]) == "archive:" + r["finding"]}
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


def archive_keys(pc, ddx=None):
    """Historical acquisition ids, from gold or an external reader's report identity."""
    def positive(row):
        if ddx is not None:
            return ddx.get("historical_key_signals", {}).get(row["finding"]) == "archive:" + row["finding"]
        # Public readers may load a persisted legacy record without private gold.
        # Msg uses its source id; is_text also recognizes a persisted translation.
        return is_text(row.get("result"), "m2.hist.result.prior_positive")
    return {"archive:" + r["finding"] for r in pc.get("archived_laboratory_record") or ()
            if r.get("observation") and r.get("finding") and positive(r)}
