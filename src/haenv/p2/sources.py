"""P2 visit and source metadata, after clinical gold and protocol delivery.

No clinical input, rule, acceptance set or system-score receipt changes here.
The legacy section IDs remain stable retrieval addresses; their catalog labels
describe sources. Visit metadata is supplied in the header, outside the catalog.
"""
from __future__ import annotations



def finalize(raw):
    block = (raw.prediction_context or {}).get("triage_p2") or {}
    if block.get("rules_delivery") != "library":
        return
    day = int(block["today_day"])
    for section in ("intake_checklist", "background", "home_check_today"):
        block[section]["day"] = day
    block["intake_checklist"]["patient_statement"] = block["message"]


def install():
    from haenv.p2.clinical_rules import install as install_rules
    from haenv.p2.context import install_invariants
    from haenv.p2.protocols import install as install_protocols
    install_rules()
    install_invariants()
    install_protocols()
    from haenv.app.pack_hooks import register_pack_extension
    register_pack_extension("p2:sources", register)

    # Pack-local replacement keeps all other protocol questions and their
    # world segments byte-identical. Both query and whole-record views use it.
    from haenv.world.protocols import delivery

    def source_question(next_call, pack, block):
        if pack != "p2" or block.get("rules_delivery") != "library":
            return next_call(pack, block)
        from haenv.p2.plugin import _TAIL
        from haenv.scoring.management.plan import INSTRUCTION
        tail = _TAIL.replace("R0..R14 中你用的那条规则编号", "所用规程的规则编号")
        tail = tail.replace("你依据的读数,如 home_check_today.spo2_pct=92", "你依据的患者记录")
        return "请根据患者记录与规程库给出就医去向。\n" + tail + INSTRUCTION

    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.PROTOCOL_QUESTION, "p2:sources", source_question)


def register(pack):
    if pack.name != "p2":
        return
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    def with_sources(next_call, raw, case_id):
        result = next_call(raw, case_id)
        finalize(raw)
        return result
    register_hook(Stage.OBSERVE, "p2:sources", with_sources)
