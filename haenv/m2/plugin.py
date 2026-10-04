"""The `haenv.m2` judge group (entry point declared by `plugins/haenv_m2-0.1.0.dist-info`).

`judges()` registers, through the shared skeleton (`haenv/pack_skeleton.py`):

* gold block `m2_frame` (latent key `m2`, class `knob`): `adjudication.m2_frame` is the frame gold
  (F1 urgency, F2 medication direction, F3 RCV rule), `prediction_context.m2_frame` the frame scene;
  both are absent on F0, so the bridge cases stay byte-identical to v1.0.1.
* gold block `revision` (no latent key): the round-2 push-block plan for every diagnosis case
  outside the insufficient tier.
* emission gate `m2`: the shared gate (`haenv/shared_audit.py`) with ①'s callbacks (`audit.emission`).
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
        "m2_med_said", "m2_med_gold", "m2_med_ok",
        "m2_rcv_said", "m2_rcv_gold", "m2_rcv_ok", "m2_rcv_emitted")


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
    from .core import f3_truth
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
        out.update(m2_med_said=said, m2_med_gold=g.get("med_direction"),
                   m2_med_ok=int(said == g.get("med_direction")))
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
    from haenv.pack_skeleton import Pack
    from . import core            # block functions live in core.py: its fingerprint stamps M2 worlds
    from .audit import emission
    blocks = [(NAME_FRAME, core.frame_gold, core.frame_probe, ("m2",), {"m2": "knob"},
               "M2 规格 §3.2：F1–F3 的框架金标(紧急度/调药方向/RCV 判定规则)与题面场景;"
               "F0 两侧都返回 None,桥接题与 v1.0.1 逐字节相同。")]
    if core.ROUND2:
        blocks.append((NAME_REVISION, core.push_plan, None, (), {},
                       "M2 规格 §2.2/§2.3：回合 ② 推送块的装填计划(固定 5 条:关键→鉴别→干扰→补位)与金标线、"
                       "关键信号;读数在作答时由世界层渲染。信息不足档不推送。"))
    return Pack(
        name="m2", owns=_when, source="haenv.m2 (haenv/m2/core.py)", blocks=tuple(blocks), emission=emission(),
        judge=(JUDGE, _judge, ("ddx:unified", "ddx:comorbidity", "ddx:independent"), _when),
        mounts={"single": "out", "gated": "out"},
        why_not={"slices": "M2 只有工具查询式一条轨;切片重放形态已退役(规格 §1)",
                 "multi": "M2 的第二回合是固定推送后的改判,不是多轮追踪"},
        judging_segment="M2_JUDGING")


def register() -> list:
    """Idempotent registration through the shared skeleton (also used by tools that skip the job
    loader); returns the judge the first time."""
    from haenv.pack_skeleton import register_pack
    from .hooks import install
    install()
    return register_pack(pack())


def judges():
    """Entry point of the `haenv.m2` group."""
    return register()
