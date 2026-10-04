"""Where pack 3 attaches to the production path, without editing a frozen-segment file.

Pack 3 wraps two module attributes when its group loads (`plugin.register`); each wrapper hands a
case without a pack-3 plan straight to the original (pack 4 wraps the same two; the wrappers chain
and each passes the other's cases through):

* `build.observe_labs` -- after the world's own lab measurement layer, realize the planned
  medication item (`realize.realize`), write the visible clinic record to
  `prediction_context.meds_p3` and the gold with its records to `adjudication.meds_p3`;
* `evaluate.render_for` -- the pack-3 question template without the diagnosis packs' run-time
  suffixes.

The generation-model premise drops the plan through the shared skeleton (`plugin.pack()`), and every
patient sentence is entered as a patient says it by the production path (`events.enter_symptoms`,
`registry/symptom_lay.yaml`).

The judging code is segment `tools/make_freeze.py:P3_JUDGING` (stamped on pack-3 rows as
`p3_judging_sha16`); the world code is `P3_WORLD` (stamped on the gold block as `p3_world_sha16`
and carried into `world_sha` through the registered probe). Neither enters `judging_sha16` or the
generation fingerprint.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()
BLOCK = "meds_p3"


def p3_judging_fingerprint() -> str:
    from ..pack_skeleton import segment_fingerprint
    return segment_fingerprint("P3_JUDGING")


def p3_world_fingerprint() -> str:
    from ..pack_skeleton import segment_fingerprint
    return segment_fingerprint("P3_WORLD")


def is_p3_payload(payload) -> bool:
    pc = getattr(payload, "prediction_context", None) or {}
    return isinstance(pc, dict) and isinstance(pc.get(BLOCK), dict)


def observe_meds(raw, case_id: str, truth: dict, audit: dict) -> tuple[dict, dict]:
    """The pack-3 part of the measurement layer for one case (no-op without a plan)."""
    from .gate import plan_of
    from .realize import Unrealizable, realize
    from .render import render_block
    plan = plan_of(raw)
    if plan is None:
        return truth, audit
    adj = raw.adjudication if isinstance(raw.adjudication, dict) else {}
    whitelist_ledger(raw, plan["drug"])
    try:
        item = realize(raw, str(case_id), plan)
    except Unrealizable as e:
        adj[BLOCK] = {"plan": dict(plan), "unrealizable": str(e)}
        raw.adjudication = adj
        return truth, {**(audit or {}), "p3": {"unrealizable": str(e)[:500]}}
    pc = dict(raw.prediction_context or {})
    pc[BLOCK] = render_block(item)
    raw.prediction_context = pc
    # the managed condition is a known condition of the patient (A5: the visible profile agrees with the block)
    from ..labworld.meds import drug
    up = dict(raw.user_profile or {})
    kc = list(up.get("known_conditions") or [])
    for k in [drug(item["drug"])["known_condition"]] + [c["code"] for c in [item.get("comorbidity")] if c]:
        if k not in kc:         # A24: and the comorbidity on the record
            kc = kc + [k]
            up["known_conditions"] = kc
            raw.user_profile = up
    from ..labworld.meds import TARGETS, MEDS
    adj[BLOCK] = {**item, "plan": dict(plan), "p3_world_sha16": p3_world_fingerprint(),
                  "tables": [MEDS, TARGETS]}
    raw.adjudication = adj
    rec = {"drug": item["drug"], "axis": item["axis"], "class": item["class"], "subtype": item["subtype"],
           "control_truth": [[int(r["day"]), r["truth"]] for r in item["visits"]],
           "layout_attempt": item["layout_attempt"], "level_attempt": item["level_attempt"]}
    return truth, {**(audit or {}), "p3": rec}


def whitelist_ledger(raw, drug_name: str) -> None:
    """A16: the complaints on a pack-3 case's record are drawn from those registered as irrelevant
    to the managed drug: a complaint that matches the drug's `background_deny` words is redrawn, on
    the same day, from the registered complaints that do not. The case's own condition line is
    whitelisted at eligibility (`plan.dx_allowed`)."""
    from ..events import symptom_lay_every
    from ..labworld.meds import meds
    from .plan import background_ok, h01
    reg, every = meds()["benign_complaints"], symptom_lay_every()
    sex = str((getattr(raw, "user_profile", None) or {}).get("sex") or "")
    allowed = sorted(t for t in reg if background_ok(drug_name, t, *every.get(t, ()), sex=sex))
    used = {e.get("symptom") for e in raw.evidence_ledger or ()}
    for e in raw.evidence_ledger or ():
        if (e.get("source_type") == "patient_reported_symptom"
                and not background_ok(drug_name, e.get("symptom"), e.get("context"),
                                       *every.get(e.get("symptom"), ()), sex=sex)):
            pick = min((t for t in allowed if t not in used), key=lambda t: h01("p3bg", e.get("evidence_id"), t))
            used.add(pick)
            e["symptom"] = pick
            e.pop("context", None)


def install() -> None:
    """Idempotent. Wraps the module attributes the production path looks up at call time."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from .. import build as B
        from .. import evaluate as EV
        from .. import qside as QS
        from ..run_state import run_ctx_of
        o_observe, o_render = B.observe_labs, QS.render_for

        @functools.wraps(o_observe)
        def observe_labs(raw, case_id):
            truth, audit = o_observe(raw, case_id)
            return observe_meds(raw, case_id, truth, audit)

        @functools.wraps(o_render)
        def render_for(solver, payload):
            ctx = run_ctx_of(solver)
            if not is_p3_payload(payload) or not ctx.probes:
                return o_render(solver, payload)
            pid = EV.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
            return EV.render_with_probe(payload, pid, ctx.probes, allow_retired=ctx.allow_retired), pid

        B.observe_labs = observe_labs
        QS.render_for = render_for
        _INSTALLED.append(True)
