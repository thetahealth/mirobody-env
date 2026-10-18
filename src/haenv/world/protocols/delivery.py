"""Optional protocol delivery, after gold-layer histories and before emission."""
from __future__ import annotations

def install():
    from haenv.app.pack_hooks import register_pack_extension
    register_pack_extension("protocol_delivery", register)


def register(pack):
    if pack.name not in ("p2", "p3", "p4"):
        return
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    def with_delivery(next_call, raw, case_id):
        result = next_call(raw, case_id)
        finalize(raw, str(case_id), pack.name)
        return result
    register_hook(Stage.OBSERVE, "protocol_delivery:" + pack.name, with_delivery)
    install_prompt()


def rules_delivery() -> str:
    from haenv.app.config import load_cfg
    cfg = load_cfg()
    value = cfg["synth"]["rules_delivery"]
    if value is not None:
        return value
    history = cfg.get("world", {}).get("history") or cfg["synth"]["world_history"]
    return "printed" if history == "short" else "library"


def library_block(block: dict, pack: str) -> dict:
    if rules_delivery() == "printed":
        return block
    out = dict(block)
    out.pop("triage_protocol" if pack == "p2" else "clinic_protocol", None)
    if pack == "p3":
        out.pop("dose_ladder", None)
    day = int(out["today_day" if pack == "p2" else "today"])
    out.update(rules_delivery="library", protocol_library="本院规程库",
               today_date=f"病程第 {day} 天（日期相对于本记录第 0 天）",
               encounter_type={"p2": "线上家庭监测分诊", "p3": "慢病用药复诊", "p4": "随访复查解读"}[pack],
               instruction="按本院规程处理")
    return out


def finalize(raw, case_id: str, pack: str):
    if pack not in ("p2", "p4") or rules_delivery() != "library":
        return
    key = "triage_p2" if pack == "p2" else "followup_p4"
    if key not in (raw.prediction_context or {}):
        return
    raw.prediction_context[key] = library_block(raw.prediction_context[key], pack)
    if pack == "p2" and "home_check_today" in raw.prediction_context[key]:
        from haenv.world.protocols.scoring import attach
        attach(raw.prediction_context[key], case_id)


def _base_question(pack: str, block: dict) -> str | None:
    if block.get("rules_delivery") != "library":
        return None
    from haenv.scoring.management.plan import INSTRUCTION
    if pack == "p2":
        from haenv.p2.plugin import _TAIL
        return '你是健康助手分诊求解器。按本院规程处理。今天日期与就诊类型见题头。\n' + _TAIL + INSTRUCTION
    if pack == "p3":
        from haenv.p3.plugin import P3_MEDS_PROMPT
        # Keep the answer contract, without carrying the old clinic-rule header.
        return '你是慢病调药求解器。按本院规程处理。今天日期与就诊类型见题头。\n' + P3_MEDS_PROMPT[P3_MEDS_PROMPT.index("请对这个药"):] + INSTRUCTION
    if pack == "p4":
        from haenv.p4.plugin import P4_FOLLOWUP_PROMPT
        return '你是随访复查解读求解器。按本院规程处理。今天日期与就诊类型见题头。\n' + P4_FOLLOWUP_PROMPT[P4_FOLLOWUP_PROMPT.index("请判断"):] + INSTRUCTION
    return None


def render_for(next_call, solver, payload):
    import haenv.interaction.qside as qside
    pc = getattr(payload, "prediction_context", None) or {}
    for pack, key in (("p2", "triage_p2"), ("p3", "meds_p3"), ("p4", "followup_p4")):
        template = question(pack, pc.get(key) or {})
        if template is not None:
            return template.format(payload=payload.dumps()), qside.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
    return next_call(solver, payload)


def install_prompt():
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.PROMPT, "protocol_delivery", render_for)


def question(pack: str, block: dict) -> str | None:
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import call_hook
    return call_hook(Stage.PROTOCOL_QUESTION, _base_question, pack, block)
