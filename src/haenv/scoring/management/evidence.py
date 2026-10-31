"""Answer-specific judge evidence, kept separate from registered expectations.

Only delivered user/tool content enters the acquisition view. Snapshot decision
facts (including unpurchased world observations and gold decisions) never do.
"""
from copy import deepcopy


def obtained_context(snapshot, *, trace=None, query_mode=False, solver_payload=None):
    if trace is not None:
        return {"status": "recorded", "mode": "query",
                "question": [m["content"] for m in trace.get("messages", []) if m.get("role") == "user"],
                "records": [{k: m[k] for k in ("name", "content") if k in m}
                            for m in trace.get("messages", [])
                            if m.get("role") == "tool" and isinstance(m.get("content"), dict)
                            and m["content"].get("found") is not False]}
    if query_mode:
        raise ValueError("Query-mode Q requires the saved acquisition trace")
    return {"status": "recorded", "mode": "full_payload",
            "records": deepcopy(solver_payload if solver_payload is not None else snapshot["visible_facts"])}


def case_facts(snapshot, context, obtained):
    events = []
    for i, p in enumerate(context["prohibitions"]):
        events.append({"id": f"safety.prohibition.{i + 1}", "criterion": p["text"],
                       "applicability": p["text"],
                       "actor": p.get("actor", "answer_endorsed_action"),
                       "time_window": p.get("time_window", "current_encounter"),
                       "required_evidence": {"registered_fact_ids": p["fact_ids"],
                                             "required_actions_or_alternatives": p["required_controls"]}})
    for i, rule in enumerate(context["guidelines"]["not_recommended"]):
        events.append({"id": f"safety.guideline.{i + 1}", "criterion": rule["text"],
                       "applicability": rule["text"], "actor": "answer_endorsed_action",
                       "time_window": "current_encounter", "required_evidence": {"source": rule["source"]}})
    return {"registered_decisive_history": deepcopy(context["history"]),
            "safety_events": events,
            "obtained_evidence": deepcopy(obtained) if obtained is not None else
                {"status": "not_supplied", "records": []}}


def archived_chinese(value):
    """Literal historical facts have no English asset; Msg-bearing worlds do."""
    from haenv.i18n.text import Msg, has_cjk
    if isinstance(value, Msg):
        return False
    if isinstance(value, str):
        return has_cjk(value)
    if isinstance(value, dict):
        return any(archived_chinese(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(archived_chinese(v) for v in value)
    return False
