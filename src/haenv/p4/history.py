"""Remote comparison reports are gold evidence, independent of background quantity."""
from __future__ import annotations
from copy import deepcopy
from haenv.world.history.history_dependency import HistoryDependency
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
from haenv.p4.gate import plan_of
BLOCK = "followup_p4"


def prepare(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.generation.build_clinical import _LAB_CTX
    if not enabled(plan):
        return
    ctx = _LAB_CTX.get(str(cid))
    spec = ((ctx or {}).get("clinical") or {}).get("HbA1c") or {}
    if not spec.get("cohort") or spec["cohort"] == "T2D":
        return
    # Draw patient-level cadence from the same real source's screening stratum,
    # before observation and independently of the P4 source or gold.
    from haenv.p4.realize import clinical_sources
    screening = clinical_sources()["screening_draw_policy"]["value"]
    ctx["additional_draw_policy"] = {**(ctx.get("additional_draw_policy") or {}),
                                     **{sig: {"cooldown_days": screening_gap(cid, cfg),
                                              "exception_p": cfg["exception_p"]}
                                        for sig, cfg in screening.items()}}


def screening_gap(cid, cfg):
    u = uniform("screening-cadence", cid)
    probabilities, gaps = cfg["quantile_probabilities"], cfg["gap_quantiles_days"]
    for p, q, a, b in zip(probabilities, probabilities[1:], gaps, gaps[1:]):
        if p <= u <= q:
            return round(a + (b - a) * (u - p) / (q - p))
    raise ValueError("uncovered screening-cadence quantile")


def supporting_report(g, r0, day, cid):
    """Draw a genuine neutral report whose comparison has an admissible gold.

    Use the same clipped total-variation model as realization, with rejection
    of ambiguous pairs. This conditions the item, rather than relaxing G4 for
    a deletion probe. The sampler is shared by active and inactive histories.
    """
    from haenv.world.labworld import analyte
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import z
    from haenv.world.labworld.observe import make_record
    from haenv.p4.gold import g4
    from haenv.p4.gold import pair_rcv
    cfg = analyte(g["analyte"])
    today = next(r for r in g["records"] if int(r["day"]) == int(g["t1"]))
    for attempt in range(int(table()["packs"]["p4"]["support_attempts"]["value"])):
        e = cv_total(g["analyte"]) * z("history-support", cid, day, attempt)
        rec = make_record(day, r0["time"], r0["truth"], e, [],
                          int(cfg["ndigits"]), tuple(cfg["range"]))
        cls = g4(rec, today, pair_rcv(g["analyte"], rec, today), tuple(g["band"]), source_policy=g.get("source_policy"))["class"]
        if cls is not None:
            return rec
    raise ValueError("no unambiguous historical supporting comparison")


def finalize(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan):
        return finalize_r2(raw, cid, plan)
    from haenv.p4.render import render_block
    from haenv.p4.gold import acting
    from haenv.p4.gold import gold_block_consistent
    from haenv.world.labworld.observe import make_record
    from haenv.world.labworld import analyte
    g = raw.adjudication[BLOCK]
    if "class" not in g:
        return
    T, prev = int(g["t1"]), int(g["t0"])
    rows = {int(x["day"]): x for x in g["records"]}
    r0 = rows[prev]
    active = (g["analyte"] == "creatinine" and bool(acting(r0)) and
              g["class"] in ("preanalytical", "method_difference") and uniform("history-role", cid) < rate("p4", plan))
    day = old_day("p4", cid, T)
    cfg = analyte(g["analyte"])
    # A neutral earlier report provides an independently observed comparison if
    # the decisive report is absent in a twin. It is not a fabricated result.
    support_day = max([day - 60] + [int(x["day"]) for x in g["records"] if int(x["day"]) < day])
    old = supporting_report(g, r0, support_day, cid)
    if active:
        shift = day - prev
        for key in ("events", "decoy_events"):
            for event in (g.get("acei") or {}).get(key) or ():
                if int(event["day"]) <= prev:
                    event["day"] += shift
        r0["day"] = day
        g["t0"] = day
        # These visits belong to the gold history. The background prefix may
        # only precede the known last report, never insert a new last report.
        g["records"] = [x for x in g["records"] if int(x["day"]) <= day or int(x["day"]) == T]
    additions = [old]
    if not active:
        additions.append(supporting_report(g, r0, day, cid))
    dates = {int(x["day"]) for x in additions}
    g["records"] = sorted([x for x in g["records"] if int(x["day"]) not in dates] + additions, key=lambda x: int(x["day"]))
    g["comparison_basis"] = table()["packs"]["p4"]["comparison_basis"]
    from haenv.p4.hooks import world_record
    world_record(raw, g)
    b = render_block(g, raw, cid)
    b["comparison_basis"] = g["comparison_basis"]
    raw.prediction_context[BLOCK] = b
    g["history_dependency"] = {"active": active, "recent_lower": T - 182,
                               "records": [{"section": "report", "day": day}],
                               "source": table()["packs"]["p4"]["source"]}
    bad = gold_block_consistent(g)
    if bad:
        raise ValueError("remote report changed gold: " + "; ".join(bad))
    if active:
        alternate = answer(twin(vars(raw), day))
        if alternate is None or alternate == g["class"]:
            raise ValueError("remote source has no different, derivable deletion gold")


def answer(raw):
    from haenv.p4.gold import g4
    from haenv.p4.gold import pair_rcv
    g = raw["adjudication"][BLOCK]
    T = int(g["t1"])
    past = sorted((x for x in g["records"] if int(x["day"]) < T), key=lambda x: int(x["day"]))
    today = next(x for x in g["records"] if int(x["day"]) == T)
    if (g.get("history_dependency") or {}).get("deletion") == "indeterminate" and not any(int(x["day"]) == int(g["t0"]) for x in past):
        return None
    if not past:
        return None
    prev = past[-1]
    rcv = pair_rcv(g["analyte"], prev, today)
    return g4(prev, today, rcv, tuple(g["band"]), source_policy=g.get("source_policy"))["class"]


def twin(raw, day=None):
    out = {**raw, "adjudication": {**raw["adjudication"], BLOCK: deepcopy(raw["adjudication"][BLOCK])}}
    g = out["adjudication"][BLOCK]
    lo = g["history_dependency"]["recent_lower"]
    g["records"] = [x for x in g["records"] if (int(x["day"]) != day if day is not None else int(x["day"]) >= lo)]
    return out


POLICY = HistoryDependency(BLOCK, plan_of, prepare, finalize, answer, twin)


def report_day(analyte, cid, today, *, occupied=()):
    """Empirical remote-report gap, shared by acting and inactive source roles."""
    from haenv.world.validation.difficulty_r2 import config
    cfg = config("p4")["analyte_gap_ge28"][analyte]
    gaps, probabilities = cfg["remote_gap_days"], cfg["quantile_probabilities"]
    for attempt in range(int(table()["packs"]["p4"]["support_attempts"]["value"])):
        u = uniform("history-date" if not attempt else f"history-date-{attempt}", cid)
        for a, b, da, db in zip(probabilities, probabilities[1:], gaps, gaps[1:]):
            if a <= u <= b:
                day = int(today) - round(da + (db - da) * (u - a) / (b - a))
                if day not in occupied:
                    return day
                break
        else:
            raise ValueError("uncovered report-gap quantile")
    raise ValueError("no unoccupied report day in the registered source distribution")


def finalize_r2(raw, cid, plan):
    if plan.get("version") == 2:
        from haenv.p4.diversity import finalize as diverse
        return diverse(raw, cid, plan)
    from haenv.world.validation.difficulty_r2 import register
    from haenv.p4.gold import acting
    from haenv.p4.gold import gold_block_consistent
    from haenv.p4.render import render_block
    from haenv.p4.hooks import world_record
    from haenv.p4.realize import BG_ACEI_MIN_BEFORE
    from haenv.p4.realize import clinical_sources
    g = raw.adjudication[BLOCK]
    if "class" not in g:
        return
    today, previous = int(g["t1"]), int(g["t0"])
    r0 = next(r for r in g["records"] if int(r["day"]) == previous)
    baseline = g["class"] == "true_change"
    cfg = clinical_sources()["comparison_calendar"]
    # Choose the remote calendar before inspecting the source or its side.
    # Previously a source on r0 alone selected a remote date, making direction
    # plus interval identify meals and next steps. Every source now uses the
    # same enriched comparison share and empirical conditional gap law.
    remote = uniform("history-comparison-role", cid) < cfg["remote_share"]
    active = remote and uniform("history-role", cid) < rate("p4", plan)
    day = report_day(g["analyte"], cid, today, occupied={int(r["day"]) for r in g["records"]})
    if active:
        shift = day - previous
        for key in ("events", "decoy_events"):
            for event in (g.get("acei") or {}).get(key) or ():
                if int(event["day"]) <= previous:
                    event["day"] += shift
        # Translate the entire earlier calendar; retain every measured result.
        for record in g["records"]:
            if int(record["day"]) <= previous:
                record["day"] += shift
        g["t0"] = day
    else:
        g["records"].append(supporting_report(g, r0, day, cid))
    # An older source-bearing report precedes the sampled comparison in both
    # roles. On a stable case the last same-method report rules out that older
    # sampling/method episode; deleting it changes the comparison's source.
    if not baseline:
        older = [r for r in g["records"] if int(r["day"]) < day]
        lower = max([day - 60] + [int(r["day"]) + 1 for r in older])
        # A stable comparison must follow the settling period of any earlier
        # treatment episode; its result cannot be assigned a pre-treatment truth.
        settled = [int(e["day"]) + BG_ACEI_MIN_BEFORE for e in (g.get("acei") or {}).get("events") or ()
                   if int(e["day"]) < day]
        lower = max([lower] + settled)
        support_day = day - 1
        if lower > support_day:
            raise ValueError("no settled historical comparison interval")
        support = alternate_report(g, r0, support_day, cid, min_day=lower)
        g["records"].append(support)
    g["records"].sort(key=lambda r: int(r["day"]))
    g["comparison_basis"] = table()["packs"]["p4"]["comparison_basis"]
    world_record(raw, g)
    raw.prediction_context[BLOCK] = render_block(g, raw, cid)
    register(g, "report", day, active, recent_lower=today - 182,
             deletion="indeterminate" if baseline and active else "different")
    g["specimen_procedure_history"] = [{"section": "specimen_procedure", "day": r["day"],
                                       "acts": False} for r in raw.prediction_context[BLOCK]["specimen_procedure"]]
    bad = gold_block_consistent(g)
    if bad:
        raise ValueError("remote comparison changed gold: " + "; ".join(bad))
    if active:
        alternate = answer(twin(vars(raw), day))
        if alternate == g["class"] or (alternate is None and not baseline):
            raise ValueError("remote report is not decisive under the registered deletion policy")


def alternate_report(g, r0, day, cid, *, min_day=None):
    """A genuine earlier meal/method episode; shared sampler for both roles."""
    from haenv.world.labworld import analyte
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import u01
    from haenv.world.labworld import z
    from haenv.world.labworld.observe import make_record
    from haenv.world.labworld.perturb import factor
    from haenv.p4.gold import g4
    from haenv.p4.gold import pair_rcv
    from haenv.p4.gold import method_key
    from haenv.p4.realize import _clock
    from haenv.p4.realize import _perturbation
    from haenv.p4.render import meat_slot
    cfg = analyte(g["analyte"])
    today = next(r for r in g["records"] if int(r["day"]) == int(g["t1"]))
    upper = int(day)
    lower = upper if min_day is None else int(min_day)
    available = [d for d in range(lower, upper + 1)
                 if not any(int(r["day"]) == d for r in g["records"])]
    if not available:
        raise ValueError("no available historical report day")
    for attempt in range(int(table()["packs"]["p4"]["support_attempts"]["value"])):
        day = available[int(u01("history-alternate-date", cid, attempt) * len(available))]
        fac = "meat_meal" if u01("history-alternate-source", cid, attempt) < 0.5 else "jaffe_enzymatic_switch"
        if fac == "jaffe_enzymatic_switch" and method_key(today) == "jaffe":
            # Identical Jaffe reports do not reveal a method switch. Do not
            # manufacture a hidden offset that the clinical record cannot show.
            continue
        params = {}
        clock = _clock(cid, day)
        if fac == "meat_meal":
            slot = meat_slot(clock[:5], cid, day)
            if slot is None:
                continue
            params = {"meal": slot[0], "meal_time": slot[1]}
        lo, hi = factor(fac)["effect"]
        bias = float(lo) + (float(hi) - float(lo)) * u01("history-alternate-bias", cid, attempt)
        p = _perturbation(fac, g["analyte"], "cause", bias_log=bias, params=params)
        e = cv_total(g["analyte"], "jaffe" if fac.startswith("jaffe") else None) * z("history-alternate-error", cid, attempt)
        if abs(e) > 3 * cv_total(g["analyte"], "jaffe" if fac.startswith("jaffe") else None):
            continue
        rec = make_record(day, clock, r0["truth"], e, [p], int(cfg["ndigits"]), tuple(cfg["range"]))
        cls = g4(rec, today, pair_rcv(g["analyte"], rec, today), tuple(g["band"]), source_policy=g.get("source_policy"))["class"]
        if cls is not None and cls != g["class"]:
            return rec
    # A neutral earlier draw can also identify that the newer source-bearing
    # report was the exception. Do not change any existing measured report.
    return supporting_report(g, r0, day, cid)
