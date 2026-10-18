"""The `haenv.m2` judge group (entry point declared by `plugins/haenv_m2-0.1.0.dist-info`).

`judges()` registers, through the shared skeleton (`haenv/app/packs.py`):

* gold block `m2_frame` (latent key `m2`, class `knob`): `adjudication.m2_frame` is the frame gold
  (F1 urgency, F2 medication direction, F3 RCV rule), `prediction_context.m2_frame` the frame scene;
  both are absent on F0; the bridge keeps its diagnostic gold while using the current M2 environment.
* gold block `revision` (no latent key): the round-2 push-block plan for every diagnosis case
  outside the insufficient tier.
* emission gate `m2`: the shared gate (`haenv/world/validation/shared_audit.py`) with ①'s callbacks (`audit.emission`).
* judge `m2_frame` (deterministic, mounted on single-shot and gated): the frame atoms read from the
  round-1 answer.

`register_kind_rule` is deliberately not used: the four frames keep their ddx gold kinds, so every
v1.0.1 judge still attaches; a frame kind ahead of `join_gold` would detach them.
"""
from __future__ import annotations

NAME_FRAME = "m2_frame"
NAME_REVISION = "revision"
JUDGE = "m2_frame"
KEYS = ("m2_frame", "m2_urgency_said", "m2_urgency_gold", "m2_urgency_ok",
        "m2_med_said", "m2_med_gold", "m2_med_ok", "m2_med_accept_n",
        "m2_drug_gold", "m2_drug_said", "m2_drug_ok", "m2_med_direction_ok",
        "m2_rcv_said", "m2_rcv_gold", "m2_rcv_ok", "m2_rcv_emitted")


def _drug_of(said) -> str | None:
    """One exact registered ID or Chinese generic name, never a list of candidates."""
    from haenv.world.clinical.med_course import DISPLAY
    if not isinstance(said, str):
        return None
    value = said.strip().casefold()
    for key, (name, _unit, _frequency) in DISPLAY.items():
        if value in (key.casefold(), name.casefold()):
            return key
    return None


def _urgency_of(said) -> str | None:
    s = str(said or "").strip()
    for t in ("🔴", "🟠", "🟡", "🟢"):
        if t in s:
            return t
    words = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "green": "🟢",
             "immediate": "🔴", "emergency": "🔴", "days": "🟠", "weeks": "🟡", "routine": "🟢"}
    low = s.lower()
    for w, t in words.items():
        if w in low:
            return t
    return None


def _med_of(said) -> str | None:
    s = str(said or "").strip().lower()
    table = (("investigate", "investigate_first"), ("查因", "investigate_first"), ("暂不调", "investigate_first"),
             ("uptitrate", "uptitrate"), ("加量", "uptitrate"), ("keep", "keep"), ("维持", "keep"),
             ("downtitrate", "downtitrate"), ("减量", "downtitrate"))
    for k, v in table:
        if k in s:
            return v
    return None


def frame_atoms(raw_answer: dict, vp, longitudinal: dict | None = None) -> dict:
    """Frame atoms from a structured answer (round 1 or round 2) and the verifier payload. The F3
    truth needs the readings at or before T, which the verifier payload does not carry: pass the
    case's `longitudinal_data` (the round-2 hook does); without it the F3 truth stays None."""
    from haenv.m2.core import f3_truth
    adj = getattr(vp, "adjudication", None) or {}
    g = adj.get(NAME_FRAME)
    out = {k: None for k in KEYS}
    if not isinstance(g, dict):
        return out
    a = raw_answer if isinstance(raw_answer, dict) else {}
    out["m2_frame"] = g.get("frame")
    if g["frame"] == "F1":
        said = _urgency_of(a.get("urgency"))
        out.update(m2_urgency_said=said, m2_urgency_gold=g.get("urgency"),
                   m2_urgency_ok=(None if g.get("urgency") is None else int(said == g.get("urgency"))))
    elif g["frame"] == "F2":
        tp = a.get("treatment_plan") if isinstance(a.get("treatment_plan"), dict) else {}
        said = _med_of(tp.get("direction"))
        drug_said = _drug_of(tp.get("target_drug"))
        drug_ok = int(drug_said is not None and drug_said == g.get("target_drug"))
        acc = g.get("med_accept") or ([g["med_direction"]] if g.get("med_direction") else [])
        out.update(m2_med_said=said, m2_med_gold=g.get("med_direction"), m2_med_accept_n=len(acc) or None,
                   m2_drug_gold=g.get("target_drug"), m2_drug_said=drug_said, m2_drug_ok=drug_ok,
                   m2_med_direction_ok=None if g.get("med_direction") is None else int(said in acc),
                   m2_med_ok=None if g.get("med_direction") is None else int(drug_ok and said in acc))
    else:
        T = int(getattr(vp, "T", 0) or 0)
        truth = f3_truth(longitudinal, g, T) if longitudinal is not None else None
        fu = a.get("followup") if isinstance(a.get("followup"), dict) else {}
        said = fu.get("exceeds_rcv")
        said = said if isinstance(said, bool) else None
        out["m2_rcv_said"] = said
        out["m2_rcv_emitted"] = None if longitudinal is None else truth is not None
        if truth is not None:
            out.update(m2_rcv_said=said, m2_rcv_gold=truth["exceeds_rcv"],
                       m2_rcv_ok=int(said is not None and said == truth["exceeds_rcv"]))
    return out


def _judge(out, vp, ctx):
    return frame_atoms(getattr(out, "_raw", None) or {}, vp)


def _when(vp) -> bool:
    return isinstance((getattr(vp, "adjudication", None) or {}).get(NAME_FRAME), dict)


def pack():
    from haenv.app.packs import Pack
    from haenv.app.packs import PayloadView
    from haenv.m2 import core            # block functions live in core.py: its fingerprint stamps M2 worlds
    from haenv.m2.emr import frame_probe
    from haenv.m2.audit import emission
    blocks = [(NAME_FRAME, core.frame_gold, frame_probe, ("m2", "m2_bridge"), {"m2": "knob", "m2_bridge": "knob"},
               "M2 规格 §3.2：F1–F3 的框架金标(紧急度/调药方向/RCV 判定规则)与题面场景;"
               "F0 两侧都返回 None;F0 只带 m2_bridge,菜单与购买读数随 M2。")]
    blocks.append(("m2_workup", core.workup_gold, None, (), {},
                   "M2 initial diagnostic stage: sourced required tests, alternatives and feasible purchase budget."))
    if core.ROUND2:
        blocks.append((NAME_REVISION, core.push_plan, None, (), {},
                       "M2 规格 §2.2/§2.3：回合 ② 推送块的装填计划(固定 5 条:关键→鉴别→干扰→补位)与金标线、"
                       "关键信号;读数在作答时由世界层渲染。信息不足档不推送。"))
    from haenv.m2.history import POLICY
    from haenv.m2.hooks import gated_row
    from haenv.m2.workup import gold_derivations
    return Pack(
        name="m2", history_dependency=POLICY, gated_row=gated_row,
        gold_derivations=gold_derivations(), owns=_when, source="haenv.m2 (haenv/m2/core.py)",
        blocks=tuple(blocks), emission=emission(),
        judge=(JUDGE, _judge, ("ddx:unified", "ddx:comorbidity", "ddx:independent"), _when),
        mounts={"single": "out", "gated": "out"},
        why_not={"slices": "M2 只有工具查询式一条轨;切片重放形态已退役(规格 §1)",
                 "multi": "M2 的第二回合是固定推送后的改判,不是多轮追踪"},
        judging_segment="M2_JUDGING",
        # F1-F3 ask no weight forecast: the weight-regain scaffold leaves; every stream stays (the gated
        # menu sells the withheld ones). F0 has no frame block, but its bridge marker opts into the M2
        # menu, readings and current medication display without adding frame-specific gold.
        # F1-F3 take the age band of their conditions (`build.pack_age_of`): the generator writes it into
        # the job before attaching explainer bases (with their age floors, `m2_gen_job.BASE_MIN_AGE`)
        view=PayloadView(streams=None))


def register() -> list:
    """Idempotent registration through the shared skeleton (also used by tools that skip the job
    loader); returns the judge the first time."""
    from haenv.app.packs import register_pack
    from haenv.m2.hooks import install
    install()
    from haenv.m2.emr import install as install_emr
    install_emr()
    return register_pack(pack())


def judges():
    """Entry point of the `haenv.m2` group."""
    return register()
