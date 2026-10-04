"""The `haenv.pack2` judge group (entry point declared by `plugins/haenv_pack2-0.1.0.dist-info`).

`judges()` registers, through the shared skeleton (`haenv/pack_skeleton.py`):

* gold block `triage` (latent key `pack2`, class `knob`): `adjudication.triage` = G2 on the shown
  values + plan + provenance; `prediction_context.triage` = today's visit as the solver sees it
  (both from `world.py`). Absent on every case without the `pack2` latent key.
* emission gate `pack2`: the shared gate (`haenv/shared_audit.py`) on `world.emission()` -- G2
  recompute, red-flag fields, L2-4 margins, L2-7 words; a hit refuses the case.
* question template `PACK2_TRIAGE_PROMPT` and probe `pack2.triage` (`framing.py`).
* judge `pack2_triage` (deterministic, mounted on single-shot): the atoms of `score.atoms`.

`register_kind_rule` is not used: pack-2 cases keep their ddx gold kinds.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

JUDGE = "pack2_triage"


def _judge(out, vp, ctx):
    from .score import atoms
    from .world import BLOCK
    return atoms(getattr(out, "_raw", None) or {}, (getattr(vp, "adjudication", None) or {}).get(BLOCK))


def _when(vp) -> bool:
    t = (getattr(vp, "adjudication", None) or {}).get("triage")
    return isinstance(t, dict) and t.get("pack") == "pack2"


def pack():
    from haenv.pack_skeleton import Pack
    from . import framing, world
    return Pack(
        name="pack2", owns=_when, source="haenv.pack2 (haenv/pack2/world.py)",
        blocks=((world.BLOCK, world.gold_block, world.probe_block, (world.LATENT_KEY,),
                 {world.LATENT_KEY: "knob"},
                 "四题包设计 §2.2:去向金标 G2 由题面显示的生命体征/即时检验/72 h 记录按分级阈值表算出;"
                 "题面侧为 T 时刻就诊快照。无 pack2 隐变量的例两侧都返回 None。"),),
        emission=world.emission(),
        framing=(framing.NAME, framing.PACK2_TRIAGE_PROMPT,
                 "② 急性分诊单回合题面:四个去向、无菜单(K2)、A4 在本包内为 24 小时内(K8)。"),
        # the probe carries the PACK2_WORLD fingerprint: its hash is in the external-gold manifest,
        # so `world_sha` covers every pack-2 world file, not only the block function's own file
        probe=lambda: {**framing.PROBE, "world_segment_sha16": world.world_fingerprint()},
        judge=(JUDGE, _judge, ("*",), _when), mounts={"single": "out"},
        why_not={"gated": "② 不要菜单(K2):单次请求,没有工具查询轨",
                 "slices": "② 是 T 时刻的单次分诊,没有切片重放", "multi": "② 是单回合,没有多轮追踪"},
        judging_segment="PACK2_JUDGING")


def register() -> list:
    """Idempotent registration through the shared skeleton; returns the judge the first time."""
    from haenv.pack_skeleton import register_pack
    from .hooks import install
    install()
    return register_pack(pack())


def judges():
    """Entry point of the `haenv.pack2` group."""
    return register()
