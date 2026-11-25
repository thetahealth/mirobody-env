"""Where the rl pack attaches to the production path, without editing a frozen-segment file.

* OBSERVE (`rl:realize`): after the world's measurement / history layer, realize the planned item
  (`realize.realize`): `prediction_context.redline_rl` gets the record block the solver reads and
  `adjudication.redline_rl` the truth (rule id, arm, gold, facts, sweep, declared patch span, imminence,
  item-specific must-flag text). A case without an rl plan passes through untouched.
* CASE (`rl:language`, zh builds only): the case-level gates (emission gate, leak scan) run on the English
  block, exactly as in the English build, so both languages admit the same cases; after them the zh block
  (realized from the same draws, its gold, sweep and status equal to English, G1-G3 re-checked on its text)
  replaces the English one, and the zh truth fields replace the English ones (the must-flag text stays English).
* PROMPT (`rl:framing`): the pack's framing (`Pack.framing`, lexicon `runtime.rl.plugin.framing`) filled with
  today's date, the rendered record (`render.render_text`) and the user's question; the answer envelope is
  `{"reply": ...}` with no other field.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()
BLOCK = "redline_rl"


def is_rl_payload(payload) -> bool:
    pc = getattr(payload, "prediction_context", None) or {}
    return isinstance(pc, dict) and isinstance(pc.get(BLOCK), dict)


def observe(raw, case_id: str, truth, audit):
    from haenv.rl.gate import plan_of
    from haenv.rl.realize import Unrealizable, realize_langs
    plan = plan_of(raw)
    if plan is None:
        return truth, audit
    adj = raw.adjudication if isinstance(raw.adjudication, dict) else {}
    try:
        item, zh = realize_langs(raw, str(case_id), plan)
    except Unrealizable as e:
        adj[BLOCK] = {"plan": dict(plan), "rule_id": plan.get("rule_id"), "family": plan.get("family"),
                      "variant": plan.get("variant"), "unrealizable": str(e)}
        raw.adjudication = adj
        return truth, {**(audit or {}), "rl": {"unrealizable": str(e)}}
    pc = dict(raw.prediction_context or {})
    pc[BLOCK] = item.pop("block")
    raw.prediction_context = pc
    adj[BLOCK] = {**item, "plan": dict(plan)}
    if zh is not None:
        adj[BLOCK][ZH] = zh                             # swapped in after the case-level gates (`swap_language`)
    raw.adjudication = adj
    return truth, {**(audit or {}), "rl": {"rule_id": item["rule_id"], "variant": item["variant"],
                                            "gold": item["gold"], "sweep": item["sweep"]}}


ZH = "_zh_item"


def swap_language(raw) -> list[str]:
    """After the case-level gates: the zh block and truth replace the English ones. Returns G1-G3 findings of
    the zh item (a finding blocks the case; none is expected, the zh gold equals the English one)."""
    from haenv.rl.gate import g1, g2, g3
    adj = raw.adjudication if isinstance(raw.adjudication, dict) else {}
    g = adj.get(BLOCK) or {}
    zh = g.pop(ZH, None)
    if zh is None:
        return []
    block = zh.pop("block")
    bad = [f"zh:{k}:{v}" for k, v in g1(zh) + g2(zh, block) + g3(zh, block)]
    if bad:
        return bad
    pc = dict(raw.prediction_context or {})
    pc[BLOCK] = block
    raw.prediction_context = pc
    adj[BLOCK] = {**g, **zh}
    raw.adjudication = adj
    return []


def prompt_text(payload, template: str) -> str:
    from haenv.rl.render import render_text
    blk = payload.prediction_context[BLOCK]
    return template.format(today=blk["today"], record=render_text(blk), question=blk["question"])


def install() -> None:
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from haenv.runtime.state import run_ctx_of

        def observe_labs(next_call, raw, case_id):
            truth, audit = next_call(raw, case_id)
            return observe(raw, case_id, truth, audit)

        def render_for(next_call, solver, payload):
            import haenv.interaction.qside as QS
            from haenv.i18n.runtime import runtime_text
            ctx = run_ctx_of(solver)
            if not is_rl_payload(payload) or not ctx.probes:
                return next_call(solver, payload)
            pid = QS.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
            p = ctx.probes.get(pid)
            if not p:
                raise KeyError(f"unregistered probe_id: {pid}")
            tpl = runtime_text(QS._framings()[p["framing_ref"]])
            if payload.prediction_context[BLOCK].get("lang") == "zh":      # a zh record: the zh framing
                from haenv.i18n.text import templates
                tpl = templates("zh")["runtime.rl.plugin.framing"]
            return prompt_text(payload, tpl), pid

        def case_language(next_call, cs, *args, **kwargs):
            raw, audit = next_call(cs, *args, **kwargs)
            if raw is not None and isinstance(getattr(raw, "adjudication", None), dict):
                bad = swap_language(raw)
                if bad:
                    return None, {**(audit or {}), "rl": {"zh_refused": bad[:5]}}
            return raw, audit

        from haenv.app.pack_hooks import Stage, register_hook
        register_hook(Stage.OBSERVE, "rl:realize", observe_labs)
        register_hook(Stage.CASE, "rl:language", case_language)
        register_hook(Stage.PROMPT, "rl:framing", render_for)
        _INSTALLED.append(True)
