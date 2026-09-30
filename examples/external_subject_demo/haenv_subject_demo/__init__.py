"""Demo of an externally supplied observed subject: an out-of-repo package
registers a new view, `episode`, and a judge that consumes only that view,
without changing haenv.

`register_subject(name, build, source=..., why=...)`: `build(ctx)` receives
`{"subjects": existing views, "vp": ..., "geometry": ...}` and derives the new view
from existing ones; returning `None` skips the judges that consume it (logged as
`mount_subject_missing`), never substituting another subject.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

SUBJECT = "episode"
NAME = "xdemo_stage_monotonic"
KEYS = ("xdemo_n_stages", "xdemo_risk_monotonic", "xdemo_stage_note")


def _build_episode(ctx: dict):
    """Build a stage-structured episode from `traj` (multi-round) or `rows`
    (slices); `None` when neither exists, which is not the same as zero stages.
    """
    subs = (ctx or {}).get("subjects") or {}
    src = subs.get("traj") or subs.get("rows")
    if not isinstance(src, list) or not src:
        return None
    stages = []
    for i, r in enumerate(src):
        if not isinstance(r, dict):
            return None                      # unexpected row shape
        stages.append({"idx": i,
                       "risk": r.get("risk_cat") or r.get("risk"),
                       "day": r.get("day") or r.get("t")})
    return {"stages": stages, "n": len(stages)}


_RISK_ORDER = {"low": 0, "indeterminate": 1, "elevated": 2, "high": 3}


def _judge(episode, vp, ctx):
    """Share of stage transitions where the risk level does not decrease."""
    stages = (episode or {}).get("stages") or []
    ranks = [_RISK_ORDER.get(str(s.get("risk")), None) for s in stages]
    seen = [r for r in ranks if r is not None]
    if len(seen) < 2:
        return {"xdemo_n_stages": len(stages), "xdemo_risk_monotonic": None,
                "xdemo_stage_note": "可比阶段 <2 ⇒ 不可判(不是判 0)"}
    drops = sum(1 for a, b in zip(seen, seen[1:]) if b < a)
    return {"xdemo_n_stages": len(stages),
            "xdemo_risk_monotonic": round(1.0 - drops / max(1, len(seen) - 1), 4),
            "xdemo_stage_note": None}


def _ensure_registered():
    """Register the subject, the mount, and return the judge."""
    from haenv import mount_table as MT
    from haenv.judges import Judge, register_judge

    if SUBJECT not in MT.SUBJECTS:
        MT.register_subject(
            SUBJECT, _build_episode,
            source="本包 `_build_episode`,从 `traj` / `rows` 派生",
            why="阶段结构的 episode:判据要按『阶段序』读风险等级,"
                "而 `traj` 是逐轮 dict、`rows` 是逐片行,两者都没有阶段这个轴",
            row_fields=("idx", "risk", "day"))
    if NAME not in MT.MOUNT:
        MT.mount(NAME, {"multi": SUBJECT, "slices": SUBJECT},
                 why_not={"single": "单拍没有阶段序 —— 一格作答上「单调性」无对象",
                          "gated": "门控是一次作答 + 若干轮采买,阶段轴同样不存在"})
    # output keys carry the `xdemo_` prefix to avoid collisions
    return Judge(name=NAME, fn=_judge, kinds=("*",))


def judges():
    """Called by `job.yaml`'s `plugins:` load point. Mounts first, then returns the judges."""
    return [_ensure_registered()]
