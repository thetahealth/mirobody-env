"""Pack 2 structured acute-event pool: event rows (focal indicator, cause kind, eligibility,
benign-explanation flag), chief-complaint wording and the distractor results, read from
`registry/pack2_acute_events.yaml`. PACK2_WORLD segment.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from ._util import pick as _pick
from ._util import registry_table, u01

EVENTS = "pack2_acute_events.yaml"


FOCALS = ("K", "glucose", "SBP", "NEWS2")


WORDINGS = ("alarming", "calm")



def events_table() -> dict:
    return registry_table(EVENTS)


def eligible_events(focal: str, hidden_dx: str, conditions: set[str], ev: dict | None = None) -> list[dict]:
    ev = ev or events_table()
    out = []
    for row in ev["events"]:
        if row["focal"] != focal:
            continue
        req = row.get("requires") or {}
        if req.get("hidden_marker") and not any(m in (hidden_dx or "") for m in req["hidden_marker"]):
            continue
        if req.get("any_condition") and not (set(req["any_condition"]) & conditions):
            continue
        out.append(row)
    return out


def distractor_results(case_id: str, n: int, ev: dict | None = None) -> list[dict]:
    ev = ev or events_table()
    pool = sorted(ev["distractors"], key=lambda r: u01("dpool", case_id, r["name"]))[:n]
    return [{"test": r["label"], "name": r["name"],
             "value": _pick(r["range"][0], r["range"][1], u01("dval", case_id, r["name"]), int(r["dp"]))}
            for r in pool]


def chief_complaint(case_id: str, plan: dict, ev: dict | None = None) -> str:
    """Tone sentence (alarming | calm) + the focal finding's symptoms at the severity of the planned
    disposition + the event context."""
    ev = ev or events_table()
    w = ev["wording"][plan["focal"]]
    tone = w[plan["wording"]]
    sym = w["symptoms"][plan["disposition"]]
    head = tone[int(u01("tpl", case_id) * len(tone)) % len(tone)]
    body = sym[int(u01("sym", case_id) * len(sym)) % len(sym)]
    row = next(r for r in ev["events"] if r["id"] == plan["event"])
    return f"{head}。{body}。{row['context']}。"
