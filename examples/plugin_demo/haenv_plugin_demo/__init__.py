"""Minimal end-to-end demo of an externally supplied haenv judge.

This package lives outside `haenv/`. Wiring a judge takes three steps:

    judge   ->  judges.register_judge(...)   <- returned by the `judges()` factory, called by the loader
    mount   ->  mount_table.mount(...)       <- `_ensure_mounted()`
    load    ->  job.yaml's `plugins:` field  <- `examples/plugin_demo/demo.job.yaml`

`load_judge_plugins` only registers the judge; the plugin must mount it
itself, or the judge is registered but never runs. The judge uses
`kinds=("*",)` rather than a new gold-kind rule: an appended rule is never
reached (the last built-in rule always matches), and inserting one with
`before=` would change existing questions' kinds.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

#: The judge's name; a collision with an existing judge is rejected by `register_judge`.
NAME = "pdemo_evidence_density"

#: Output keys, prefixed `pdemo_` so they cannot overwrite another judge's keys.
KEYS = ("pdemo_n_cited_ev", "pdemo_drivers_sourced", "pdemo_drivers_total",
        "pdemo_evidence_ok")


def judge_evidence_density(out, vp, ctx=None) -> dict:
    """Whether every driver cites its own evidence (citation density, not correctness).

    Reads only the answer (`drivers[*].evidence_for` / `cited_evidence`), no gold.

    . `pdemo_n_cited_ev`      -- distinct EV ids cited.
    . `pdemo_drivers_total`   -- drivers listed.
    . `pdemo_drivers_sourced` -- drivers with at least one evidence id.
    . `pdemo_evidence_ok`     -- 1 if every driver has evidence, else 0; `None` if no driver was listed.
    """
    drivers = list(getattr(out, "drivers", None) or [])
    cited = {str(e) for e in (getattr(out, "cited_evidence", None) or []) if e}
    sourced = sum(1 for d in drivers
                  if isinstance(d, dict) and (d.get("evidence_for") or []))
    return {
        "pdemo_n_cited_ev": len(cited),
        "pdemo_drivers_total": len(drivers),
        "pdemo_drivers_sourced": sourced,
        "pdemo_evidence_ok": (1 if sourced == len(drivers) else 0) if drivers else None,
    }


def _ensure_mounted() -> None:
    """Mount this judge on the `single` geometry (idempotent). `mount()` requires a
    reason for each unmounted geometry; `multi` is covered by the table's
    wildcard reason.
    """
    from haenv import mount_table as mt
    if NAME in mt.MOUNT:
        return
    mt.mount(NAME, {"single": mt.OUT}, why_not={
        "gated": "门控几何的 payload 先收走信号再按需查询,`cited_evidence` 里混着"
                 "工具轨取回的编号 ⇒ 「引用密度」在两种几何下不是同一个量,"
                 "并排读会把口径差读成能力差。演示只要一个几何,这里不挂。",
        "slices": "切片几何一次求诊拆成 N 片各自作答,逐片的驱动列表是局部的;"
                  "跨片合并需要一条独立的口径(合并前去重?按末片?),"
                  "没有想清楚之前不挂 —— 挂上会产出一个说不清分母的数。",
    })


def judges():
    """Entry-point factory: `() -> Iterable[Judge]`. Mounts first, then returns the
    judge; returns `[]` if it is already registered.
    """
    from haenv.judges import JUDGES, Judge
    _ensure_mounted()
    if any(j.name == NAME for j in JUDGES):
        return []
    return [Judge(NAME, ("*",), judge_evidence_density)]
