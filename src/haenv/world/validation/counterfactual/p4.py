"""P4 clinical-source and next-step counterfactuals, with production rendering."""
import contextlib
import copy
import json
import math
from functools import lru_cache
from types import SimpleNamespace

from haenv.i18n import current
from haenv.i18n.text import render as i18n_render
from haenv.p4.gold import (g4, next_step, pair_rcv, true_change_pct, acting,
                           gold_block_consistent, ingested_interference, ACEI_RISE_MAX_PCT)
from haenv.p4.score import accepted_classes
from haenv.p4.render import render_block
from haenv.p4.features import visible_class, _blk
from haenv.p4.diversity import table, verification_consistent, realize
from haenv.p4.realize import Unrealizable
from haenv.p4.diversity_reader import answer, applies, significance_violations
import haenv.p4.render as _R
import haenv.p4.diversity as DV
from haenv.world.labworld import analyte, cv_total
from haenv.world.labworld.observe import make_record
from haenv.world.generation.twins import same_day_medication_swap

from .batch import Operator, InvalidTwin, load_batch
from .trace import RuleTrace, recording


@contextlib.contextmanager
def _meds(rows):
    """`diversity.render` imports `medication_rows` from `haenv.p4.render` at call time; pin it to the
    base-world rows the build saw (the builder prunes the
    dose stream and `register_medicines` adds the P4 courses to world_medication after rendering)."""
    orig = _R.medication_rows
    _R.medication_rows = lambda item, raw, T: copy.deepcopy(rows)
    try:
        yield
    finally:
        _R.medication_rows = orig

def jdump(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True)

def pre_registration(raw):
    """The raw as `diversity.render` saw it: without the P4 courses `register_medicines` adds afterwards
    (else `existing_medicines` skips the P4 keys whose drug it registered: thyroxine, biotin, ...)."""
    r = copy.copy(raw)
    lp = copy.deepcopy(raw.latent_premise or {})
    wm = lp.get("world_medication") or {}
    if wm.get("concurrent"):
        wm["concurrent"] = [m for m in wm["concurrent"] if m.get("source") != "P4 dated medication record"]
    r.latent_premise = lp
    return r

def base_med_rows(item, raw, cid, lang, stored_block):
    """Stored medication rows minus the rows `diversity.render` itself adds (rendered with no base rows)."""
    from haenv.world.protocols.delivery import library_block
    with _meds([]):
        own = i18n_render(render_block(item, pre_registration(raw), cid), lang)["medication_record"]
    want = [jdump(r) for r in own]
    base = []
    for r in stored_block["medication_record"]:
        d = jdump(r)
        if d in want:
            want.remove(d)
        else:
            base.append(r)
    if want:
        raise ValueError(f"{len(want)} P4-own medication rows not found in the stored block")
    return base

def prod_render(item, raw, cid, lang, base_rows):
    """render_block + protocol delivery (`library_block`) + batch-language render, with pinned base rows.
    Base rows are already language-rendered plain strings; P4 rows are Msg values rendered here."""
    from haenv.world.protocols.delivery import library_block
    with _meds([]):
        b = render_block(item, pre_registration(raw), cid)
    b = i18n_render(library_block(b, "p4"), lang)
    # diversity.render returns medication rows = base rows + own rows, sorted only where the base helper sorts;
    # re-insert the base rows in the production position (the base helper output comes first).
    b["medication_record"] = copy.deepcopy(base_rows) + b["medication_record"]
    return b

def reprint(rec, an):
    a = analyte(an)
    new = make_record(rec["day"], rec["time"], rec["truth"], rec["e"], rec["perturbations"], int(a["ndigits"]), tuple(a["range"]))
    rec.update(new)

def base_truth(g):
    """round(base, 4) as every unaffected record carries it."""
    return round(float(g["base"]), 4)

def truth_at(g, day, response):
    """The realizer's truth law (`diversity.realize`) for a true-change item, with a given response."""
    cfg = table()
    fac, sign, t0 = g["cause"], int(g["sign"]), int(g["t0"])
    p = g["cause_params"]
    intrinsic, event = int(p["intrinsic"]), int(p["event_day"])
    base = float(g["base"])
    if fac in cfg["medication_world"]:
        prior = t0 - cfg["medication_world_design"]["prior_established_days"]
        taking = day >= event if sign == intrinsic else prior <= day < event
        return base * math.exp(intrinsic * response) if taking else base
    return base * math.exp(sign * response) if day > event else base

def retruth(g, response):
    for rec in list(g["records"]) + list(g.get("verification_records") or []):
        rec["truth"] = truth_at(g, int(rec["day"]), response)
        reprint(rec, g["analyte"])
    g["cause_params"]["response"] = response

def side_rec(g):
    p = g["cause_params"]
    day = int(g["t1"]) if p.get("side") else int(g["t0"])
    return next(r for r in g["records"] if int(r["day"]) == day)

def comp(g):
    R = {int(r["day"]): r for r in g["records"]}
    return R[int(g["t0"])], R[int(g["t1"])]

def op_remove_source(g):
    """Acting premise: delete the item's source fact (pre/method: the cause perturbation on its draw; true
    change: the in-interval exposure/event, truths back to baseline). cause -> None (render shows only the
    inactive look-alike rows that every item carries)."""
    if g["class"] == "analytic_biological_noise" or not g.get("cause"):
        return None
    h = copy.deepcopy(g)
    if g["class"] == "true_change":
        retruth(h, 0.0)
    else:
        rec = side_rec(h)
        n0 = len(rec["perturbations"])
        rec["perturbations"] = [p for p in rec["perturbations"] if not (p.get("role") == "cause" and p.get("factor") == g["cause"])]
        if len(rec["perturbations"]) != n0 - 1:
            raise ValueError("cause perturbation not found on its side draw")
        reprint(rec, g["analyte"])
    h["cause"], h["factor"] = None, None
    if g["analyte"] == "creatinine" and g["cause"] == "acei_start_or_stop":
        h["acei"] = {**h["acei"], "events": []}
    return h

def _scale_to(g, target_abs_dtrue):
    r0, r1 = comp(g)
    t = math.log(float(r1["truth"]) / float(r0["truth"]))
    resp = float(g["cause_params"]["response"])
    if t == 0 or resp == 0:
        return None
    h = copy.deepcopy(g)
    retruth(h, resp * target_abs_dtrue / abs(t))
    return h

def op_dtrue_small(g):
    """|Δtrue| -> RCV/8 (below the legacy gray band RCV/4)."""
    if g["class"] != "true_change":
        return None
    return _scale_to(g, float(g["rcv"]) / 8)

def op_dtrue_large(g):
    """|Δtrue| -> 0.75 RCV (above the legacy RCV/2) when it is below; else 1.5x."""
    if g["class"] != "true_change":
        return None
    r0, r1 = comp(g)
    t = abs(math.log(float(r1["truth"]) / float(r0["truth"])))
    target = 0.75 * float(g["rcv"]) if t < 0.75 * float(g["rcv"]) else 1.5 * t
    return _scale_to(g, target)

def op_side_flip(g):
    """Side of the source: the same perturbation on the other compared draw (sign of the change flips; the
    realizer emits both sides, side = 1 iff sign == intrinsic)."""
    if g["class"] not in ("preanalytical", "method_difference"):
        return None
    h = copy.deepcopy(g)
    old = side_rec(h)
    p = next(p for p in old["perturbations"] if p.get("role") == "cause" and p.get("factor") == g["cause"])
    old["perturbations"].remove(p)
    reprint(old, g["analyte"])
    h["cause_params"]["side"] = 0 if g["cause_params"]["side"] else 1
    h["sign"] = -int(g["sign"])
    new = side_rec(h)
    new["perturbations"].append(p)
    if g["cause"] == "jaffe_enzymatic_switch":
        # the Jaffe draw carries the Jaffe total CV; the realizer draws e = cv * z, so rescale e
        an = g["analyte"]
        new["e"] = round(new["e"] * cv_total(an, "jaffe") / cv_total(an), 6)
        old["e"] = round(old["e"] * cv_total(an) / cv_total(an, "jaffe"), 6)
        reprint(old, an)
        h["cv_total"] = list(reversed(h["cv_total"]))
    if g["analyte"] == "tsh" and g["cause"] == "sampling_time":
        for r in (old, new):
            r["time"] = "15:00:00" if r is new else "08:00:00"
    reprint(new, g["analyte"])
    return h

def op_method_equal(g):
    """Method key equal on both draws: for Jaffe items the only renderable form is 'neither draw Jaffe'
    (= remove_source); 'both Jaffe' has no renderer path (diversity.render marks only the side draw)."""
    if g.get("cause") != "jaffe_enzymatic_switch":
        return None
    return op_remove_source(g)

def op_exposure_toggle(g):
    """Ingested interference: ongoing <-> ended (end_day T+1, clean follow-up), the realizer's two states."""
    if not ingested_interference(g.get("factor")) or "ingested_exposure" not in g:
        return None
    h = copy.deepcopy(g)
    e = h["ingested_exposure"]
    e["end_day"] = int(h["t1"]) + 1 if e["end_day"] is None else None
    e["clean_followup"] = True
    return h

def op_acei_rise(g):
    """ACEI start: move the printed rise across 30% (to 15% or 45%)."""
    if g.get("cause") != "acei_start_or_stop":
        return None
    r0, r1 = comp(g)
    obs = float(r1["printed"]) / float(r0["printed"]) - 1
    if obs <= 0:
        return None
    threshold = ACEI_RISE_MAX_PCT / 100
    target_obs = 1.5 * threshold if obs <= threshold else 0.5 * threshold
    # Δtrue moves by the same log amount as the wanted printed change (e fixed)
    t = math.log(float(r1["truth"]) / float(r0["truth"]))
    want = t + math.log1p(target_obs) - math.log1p(obs)
    if want <= 0:
        return None
    return _scale_to(g, want)

def op_drop_episode(g):
    """Remove the archived episode perturbation (a draw before the compared pair)."""
    if (g.get("history_dependency") or {}).get("active"):
        return None
    t0 = int(g["t0"])
    for r in g["records"]:
        if int(r["day"]) < t0 and any(p.get("acts") for p in r["perturbations"]):
            h = copy.deepcopy(g)
            rr = next(x for x in h["records"] if x["day"] == r["day"])
            rr["perturbations"] = []
            reprint(rr, g["analyte"])
            return h
    return None

def op_perturb_old_draw(g):
    """Move e of the earliest archived draw by +1 total CV (not a compared draw)."""
    rs = sorted(g["records"], key=lambda r: int(r["day"]))
    if len(rs) < 3:
        return None
    h = copy.deepcopy(g)
    r = min(h["records"], key=lambda r: int(r["day"]))
    r["e"] = round(r["e"] + cv_total(g["analyte"]), 6)
    reprint(r, g["analyte"])
    return h

def op_drop_inactive_candidates(g):
    """Remove every non-acting look-alike candidate the render prints for the analyte (rule rows of keys
    other than the cause): implemented by rendering with those keys removed from the analyte's lists."""
    return copy.deepcopy(g)

def op_reword_method_notes(g):
    """Reword every method notice with the next registered wording (`question_update_14.wording`)."""
    if not any(k in table()["analytes"][g["analyte"]]["method"] for k in ("platform_calibration", "reagent_lot", "jaffe_enzymatic_switch")):
        return None
    return copy.deepcopy(g)

def op_swap_same_day_meds(g):
    return copy.deepcopy(g)

@contextlib.contextmanager
def patched_table(g, mode):
    orig_table, orig_h01 = DV.table, DV.h01
    if mode == "drop_inactive":
        t = copy.deepcopy(orig_table())
        a = t["analytes"][g["analyte"]]
        for k in ("pre", "method", "true_causes"):
            a[k] = [x for x in a[k] if x == g.get("cause")]
        DV.table = lambda: t
    elif mode == "reword":
        def h01(tag, *args):
            v = orig_h01(tag, *args)
            if tag == "p4-method-wording":
                n = len(orig_table()["question_update_14"]["wording"][args[1]])
                v = ((int(v * n) + 1) % n + 0.5) / n
            return v
        DV.h01 = h01
    try:
        yield
    finally:
        DV.table, DV.h01 = orig_table, orig_h01


def recompute(g, class_rule=g4, next_rule=next_step):
    """Use the production class/factor reconstruction before next-step scoring."""
    r0, r1 = comp(g)
    rcv = pair_rcv(g["analyte"], r0, r1)
    o = class_rule(r0, r1, rcv, tuple(g["band"]), source_policy=g.get("source_policy"))
    if o["class"] is None:
        return None, None
    h = {**g, "class": o["class"],
         "factor": o["factor"] if o["factor"] else g.get("factor") if o["class"] == "true_change" else o["factor"]}
    S = tuple(sorted((c, n) for c in accepted_classes(h) for n in next_rule(h)[1]))
    return S, o


@lru_cache(maxsize=1)
def traces():
    return RuleTrace(g4, label="p4.g4"), RuleTrace(next_step, label="p4.next_step")


def decide(g, override=None):
    """One recording across class and next-step rules; overrides share a map."""
    a, b = traces()
    with recording(override) as events:
        S, _ = recompute(g, a.instrumented, b.instrumented)
    return S, events


class _Composite:
    sources = (g4, next_step)
    run = staticmethod(decide)

    @staticmethod
    def instrumented_equals(g):
        return decide(g)[0] == recompute(g)[0]


def source_target(g):
    return "p4.true_component" if g["class"] == "true_change" else "p4.source_named"


def side_target(g):
    return "p4.source_today" if acting(comp(g)[1]) else "p4.source_previous"


class Adapter:
    pack = "p4"
    items = staticmethod(load_batch)
    rule = staticmethod(_Composite)
    decide = staticmethod(decide)
    decision = staticmethod(lambda result: result)

    @staticmethod
    def block(raw):
        return raw if isinstance(raw, dict) else raw.adjudication["followup_p4"]

    def rule_args(self, item):
        return (self.block(item.raw),)

    def scored(self, raw):
        S, _ = recompute(self.block(raw))
        return list(S) if S is not None else None

    @staticmethod
    def visible_answer(sp):
        b = _blk(sp)
        cls = visible_class(sp)[0]
        try:
            nxt = answer(b)["next_step"] if applies(b) else None
        except ValueError:
            nxt = "reader_refused"
        return cls, nxt

    @staticmethod
    def visible_blocks(sp):
        return sp["prediction_context"]["followup_p4"]

    def prepare(self, item):
        self.item = item
        self.base_rows = base_med_rows(self.block(item.raw), item.raw, item.case_id, current(), self.visible_blocks(item.sp))
        self.op = None

    def set_operator(self, name):
        self.op = name

    def baseline_payload(self, item):
        return self._payload(item, self.block(item.raw))

    def _payload(self, item, g, mode=None):
        with patched_table(g, mode):
            block = prod_render(g, item.raw, item.case_id, current(), self.base_rows)
        if self.op == "neg_swap_same_day_meds":
            block["medication_record"] = same_day_medication_swap(block["medication_record"])[0]
        sp = copy.deepcopy(item.sp)
        sp["prediction_context"]["followup_p4"] = block
        return sp

    def twin_payload(self, item, twin):
        S, o = recompute(twin)
        if S is None:
            raise ValueError("class rule refuses the twin")
        # These are the dependent fields finish() regenerates before rendering.
        g = copy.deepcopy(twin)
        r0, r1 = comp(g)
        g.update({"class": o["class"], "factor": g.get("cause"),
                  "rcv": round(pair_rcv(g["analyte"], r0, r1), 6), "g4": o,
                  "true_change_pct": true_change_pct(r0, r1),
                  "obs_pct": round(100 * math.expm1(o["dlog_obs"]), 3)})
        g["next_step"] = next_step(g)[0]
        mode = {"neg_drop_inactive_candidates": "drop_inactive", "neg_reword_method_notes": "reword"}.get(self.op)
        payload = self._payload(item, g, mode)
        self.finished = g
        return payload

    def scored_metadata(self, item, twin, hidden_outcome):
        return dict(hidden_outcome=hidden_outcome, refusals=[], history_refusal=False)

    def twin_metadata(self, item, twin, payload):
        g, refusals = self.finished, []
        r0, r1 = comp(g)
        o, rcv = g["g4"], g["rcv"]
        if o["class"] == "analytic_biological_noise" and abs(o["dlog_obs"]) >= rcv:
            refusals.append("noise_outside_rcv")
        if o["dlog_obs"] * int(g["sign"]) <= 0 and o["class"] != "analytic_biological_noise":
            refusals.append("observed_sign")
        if gold_block_consistent(g):
            refusals.append("gold_block_consistent")
        if verification_consistent(g):
            refusals.append("verification_consistent")
        try:
            if significance_violations(self.visible_blocks(payload), o["class"], g.get("cause")):
                refusals.append("significance")
        except Exception:
            refusals.append("significance_error")
        v = self.visible_answer(payload)
        if v[0] != o["class"]:
            refusals.append("visible_class")
        elif v[1] != g["next_step"]:
            refusals.append("visible_next_step")
        history_refusal = False
        if (g.get("history_dependency") or {}).get("active") and o["class"] != "true_change":
            prior = [r for r in g["records"] if int(r["day"]) < int(g["t0"])]
            if prior:
                pr = max(prior, key=lambda r: int(r["day"]))
                alt = g4(pr, r1, pair_rcv(g["analyte"], pr, r1), (1.3, 2.0), source_policy="clinical")["class"]
                history_refusal = alt is None or alt == o["class"]
        from .receipt import _outcome
        result = dict(hidden_outcome=_outcome(self.scored(twin), self.scored(item.raw)),
                      refusals=refusals, history_refusal=history_refusal)
        if refusals:
            result["outcome_R"] = "invalid"
        return result

    def metadata(self, item, result):
        g = self.block(item.raw)
        stored = tuple(sorted((c, n) for c in accepted_classes(g) for n in next_step(g)[1]))
        return dict(stored_gold_equal=result == stored, cause=g.get("cause"), analyte=g["analyte"],
                    class0=g["class"])

    def _plan(self, g, name, cls=None, cause=None):
        plan = g.get("plan")
        if not plan:
            return None
        if name == "plan_add_source" and (g["class"] != "analytic_biological_noise"
                or cause not in table()["analytes"][g["analyte"]][{"preanalytical": "pre", "method_difference": "method", "true_change": "true_causes"}[cls]]):
            return None
        if name != "plan_add_source" and g["class"] == "analytic_biological_noise":
            return None
        if name == "plan_swap_mechanism":
            group = {"preanalytical": "pre", "method_difference": "method", "true_change": "true_causes"}[g["class"]]
            if cause == g.get("cause") or cause not in table()["analytes"][g["analyte"]][group]:
                return None
        if name == "plan_remove_source":
            plans = [{**plan, "class": "analytic_biological_noise", "cause": None, "sign": s}
                     for s in (int(plan["sign"]), -int(plan["sign"]))]
        elif name == "plan_sign_flip":
            plans = [{**plan, "sign": -int(plan["sign"])}]
        else:
            plans = [{**plan, "cause": cause, **({"class": cls} if cls else {})}]
        last = None
        for pl in plans:
            try:
                with _meds([]):
                    return realize(pre_registration(self.item.raw), self.item.case_id, pl, None)
            except Unrealizable as exc:
                last = exc
        raise InvalidTwin(str(last))

    def operators(self):
        ops = {}
        def add(name, target, fn, tier="R"):
            ops[name] = Operator(name, target, tier, fn)
        add("remove_source", source_target, op_remove_source)
        add("plan_remove_source", source_target, lambda g: self._plan(g, "plan_remove_source"))
        for cls, group in (("preanalytical", "pre"), ("method_difference", "method"), ("true_change", "true_causes")):
            causes = sorted({cause for a in table()["analytes"].values() for cause in a[group]})
            for cause in causes:
                name = f"plan_add_source:{cls}:{cause}"
                add(name, "p4.source_named", lambda g, c=cls, k=cause: self._plan(g, "plan_add_source", c, k))
        for name, target, fn in (("method_equal", "p4.source_named", op_method_equal),
                                 ("acei_rise", "p4.acei_rise_le30", op_acei_rise),
                                 ("side_flip", side_target, op_side_flip),
                                 ("exposure_toggle", "p4.exposure_ended", op_exposure_toggle)):
            add(name, target, fn)
        add("dtrue_small", "p4.true_component", op_dtrue_small, "R-info")
        add("dtrue_large", "p4.true_component", op_dtrue_large, "R-info")
        add("plan_sign_flip", side_target, lambda g: self._plan(g, "plan_sign_flip"), "R-info")
        for cause in sorted(table()["rules"]):
            add("plan_swap_mechanism:" + cause, "p4.source_named",
                lambda g, k=cause: self._plan(g, "plan_swap_mechanism", cause=k), "R-info")
        for name, fn in (("neg_drop_inactive_candidates", op_drop_inactive_candidates),
                         ("neg_reword_method_notes", op_reword_method_notes),
                         ("neg_perturb_old_draw", op_perturb_old_draw), ("neg_drop_episode", op_drop_episode)):
            add(name, None, fn)
        def swap(g):
            return copy.deepcopy(g) if same_day_medication_swap(self.visible_blocks(self.item.sp)["medication_record"]) else None
        add("neg_swap_same_day_meds", None, swap)
        return ops
