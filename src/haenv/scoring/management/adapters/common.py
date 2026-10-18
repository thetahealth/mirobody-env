"""Saved visible facts and control renderings; no criteria or new thresholds."""
from haenv.scoring.management.rubric import canonical
from haenv.scoring.management.rubric import pointer


class Facts:
    def __init__(self, snapshot, policy):
        self.snapshot, self.policy = snapshot, policy
        self.history, self.key, self.prohibitions, self.refs = [], [], [], []
        self.guidelines = {"not_recommended": [], "initial": []}
        for old in snapshot["decisive_old_records"]:
            rows = self.records(old["section"], lambda r: r == old["record"])
            if len(rows) != 1:
                raise ValueError("Decisive old record has no unique visible source")
            self.add_history(*rows[0])

    def records(self, section, predicate=lambda r: True):
        rows = self.snapshot["visible_facts"].get(section, [])
        if isinstance(rows, dict):
            rows = [rows]
        return [(f"/visible_facts/{section}/{i}", r) for i, r in enumerate(rows) if predicate(r)]

    def add_history(self, ref, value):
        if not any(row["fact_id"] == ref for row in self.history):
            self.history.append({"fact_id": ref, "record": value})
            self.refs.append(ref)

    def add_key(self, ref):
        self.key.append({"fact_id": ref, "record": pointer(self.snapshot, ref)})
        self.refs.append(ref)

    def prohibit(self, text, refs, *, violation, required=()):
        self.prohibitions.append({"text": text, "fact_ids": list(refs),
                                  "violation_control": violation, "required_controls": list(required)})
        self.refs.extend(refs)

    def finish(self, *, actions, selected, explanation, followup, initial=None,
               brief_explanation=None, accepted=None):
        from haenv.scoring.management.judge import _human_claim
        if not actions or selected not in actions:
            raise ValueError("Saved control has no registered selected action")
        history = ("旧记录显示：" + "；".join(_human_claim(canonical(h["record"])) for h in self.history)
                   + "；据此保留该旧记录对本次评估和处置的影响。") if self.history else None
        control = {"actions": actions, "selected": selected, "accepted": accepted or [selected], "history": history,
                   "explanation": _human_claim(explanation), "followup": followup, "initial": initial,
                   "brief_explanation": brief_explanation or _human_claim(explanation)}
        control["management_alternatives"] = [actions[selected].replace(test, alternative)
            for test, alternatives in self.snapshot["check_alternatives"].items()
            for alternative in alternatives if test in actions[selected]]
        return {"history": self.history, "key_facts": self.key,
                "prohibitions": self.prohibitions, "guidelines": self.guidelines,
                "source_fact_ids": list(dict.fromkeys(self.refs)), "control": control}


def applicable_guidelines(snapshot):
    """Only v4 not-recommended rules enter safety; monitoring intervals do not."""
    from haenv.scoring.management.precision import _matches
    from haenv.scoring.management.precision import load_indications
    data = load_indications()
    v, f = snapshot["visible_facts"], snapshot["decision_facts"]
    background = v.get("background", {})
    groups = [g for g in list(data["families"].values()) + list(data["chronic"].values())
              if _matches(g, f.get("tier", ""), set(background.get("known_conditions", [])),
                          canonical(background.get("medications", [])).casefold())]
    return {kind: [{"text": r["text"], "source": data["sources"][r["source"]],
                    "control": r.get("control")} for g in groups for r in g.get(kind, [])]
            for kind in ("not_recommended", "initial")}
