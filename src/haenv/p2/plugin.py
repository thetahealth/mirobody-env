"""The `haenv.p2` judge group: pack 2 rebuilt (entry point `plugins/haenv_p2-0.1.0.dist-info`).

Registers through the shared skeleton (`haenv/app/packs.py`):
* gold block `triage_p2` (latent key `p2`, class `knob`): the plan at generation; after the
  measurement layer (`hooks.observe_triage`) the rule reader's verdict on the shown block;
* emission gate `p2` (`gate.py`);
* question template `P2_TRIAGE_PROMPT` (prints the rules of `registry/pack2_triage_scenarios.yaml`
  once for every item) and probe `p2.triage`, carrying the P2_WORLD fingerprint and the table hash;
* judge `triage_p2` (deterministic, single shot): the atoms of `score.atoms`.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import J, L as _L, T, text_of as _rt

NAME = "triage_p2"
LATENT = "p2"
PROBE_ID = "p2.triage"
FRAMING = "P2_TRIAGE_PROMPT"

_HEAD = _L('runtime.p2.plugin.module.1')
_TAIL = _L('runtime.p2.plugin.module.2')


def prompt() -> str:
    from haenv.p2.rules import table
    rules = table()
    return T('runtime.p2.triage_prompt', rules=J('lines', [
        *(_L(r['printed']) for r in rules['rules']),
        _L(rules['acceptance']['single_level2']['printed'])]))


def _gold(meta):
    from haenv.contracts.external_gold import SLOT
    p = ((meta or {}).get(SLOT) or {}).get(LATENT)
    return {"plan": dict(p)} if isinstance(p, dict) else None


def probe_dict() -> dict:
    from haenv.world.generation.framings import framing_sha256
    from haenv.p2.hooks import p2_world_fingerprint
    from haenv.p2.rules import table_sha16
    return {"probe_id": PROBE_ID, "mode": "ddx", "difficulty_probe": "P0", "framing_ref": FRAMING,
            "framing_sha256": framing_sha256(FRAMING), "answer_space": "p2_triage", "hint_level": 0,
            "budget": {"max_rounds": 1}, "p2_world_sha16": p2_world_fingerprint(), "p2_table_sha16": table_sha16()}


def _judge(out, vp, ctx):
    from haenv.p2.hooks import p2_judging_fingerprint
    from haenv.p2.score import atoms
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return {**atoms(getattr(out, "_raw", None) or {}, g), "p2_judging_sha16": p2_judging_fingerprint()}


def _when(vp) -> bool:
    g = (getattr(vp, "adjudication", None) or {}).get(NAME)
    return isinstance(g, dict) and bool(g.get("set"))


KINDS = ("*",)


def pack():
    from haenv.app.packs import Pack
    from haenv.app.packs import PayloadView
    from haenv.p2.gate import emission
    from haenv.p2.history import POLICY
    return Pack(
        name="p2", history_dependency=POLICY, owns=_when, source="haenv.p2 (haenv/p2/plugin.py)",
        blocks=((NAME, _gold, lambda meta: None, (LATENT,), {LATENT: "knob"},
                 _rt('runtime.p2.plugin.pack.1')),),
        emission=emission(),
        framing=(FRAMING, prompt(), _rt('runtime.p2.plugin.pack.2')),
        probe=probe_dict, judge=(NAME, _judge, KINDS, _when), mounts={"single": "out"},
        why_not={"gated": _rt('runtime.p2.plugin.pack.3'), "slices": _rt('runtime.p2.plugin.pack.4'),
                 "multi": _rt('runtime.p2.plugin.pack.5')},
        prompt_gold_meta=("outcome", "driver"), judging_segment="P2_JUDGING",
        base_labels=(), answers_action=False,
        # the item is read off the triage block; world streams are background only and the
        # quantities the block shows are removed in its window (`hooks.OVERLAP_STREAMS`)
        view=PayloadView(streams=("HbA1c",)))


def register() -> list:
    from haenv.app.packs import register_pack
    from haenv.p2.hooks import install
    install()
    return register_pack(pack())


def judges():
    """Entry point of the `haenv.p2` group."""
    return register()
