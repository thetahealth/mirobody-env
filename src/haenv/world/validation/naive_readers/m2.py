"""Diagnosis vocabulary stubs scored with the production list contract.

SYNTHETIC data, evaluation only, not medical advice.
"""
import json
import re
from types import SimpleNamespace

from haenv.m2.round2 import alias_sets_of
from haenv.m2.score import _listed
from .core import Reader

NAME = "m2"
CODEBOOK = {}
ALIASES = {}
FACTS = ()
FACT_NOTE = "Key signals and history dependency are declared, not knockout tested. No visible full reader exists."


def gold(case):
    return case["adjudication"]


def aliases(g):
    return alias_sets_of(SimpleNamespace(adjudication=g))


def spec_id(g):
    return g["ddx"]["spec_id"]


def configure(items):
    global CODEBOOK, ALIASES
    CODEBOOK, ALIASES = {}, {}
    for item in items:
        g, ddx = item.gold, item.gold["ddx"]
        key = spec_id(g)
        if ddx["join_gold"] == "independent":
            names = []
        else:
            names = [t.get("diagnosis") or (t.get("aliases") or [t["name"]])[0]
                     for t in ddx.get("threads") or []] or [ddx["diagnosis"]]
        if key in CODEBOOK:
            assert CODEBOOK[key] == names, f"nonconstant diagnosis codebook: {key}"
        CODEBOOK[key] = names
        ALIASES.setdefault(key, set()).update(a for group in aliases(g) for a in group if a)
    ALIASES = {k: sorted(v) for k, v in sorted(ALIASES.items())}
    CODEBOOK = dict(sorted(CODEBOOK.items()))
    return {"codebook": CODEBOOK, "aliases": ALIASES}


def ok(answer, g):
    if answer is None:
        return False
    if g["ddx"]["join_gold"] == "independent":
        return answer["join"] == "independent"
    raw = {"differential": [{"rank": i + 1, "diagnosis": name, "certainty": "probable", "ruled_out_by": None}
                            for i, name in enumerate(answer["names"])]}
    return _listed(raw, aliases(g))["dx_listed"] == 1.0


def gold_label(g):
    return "independent" if g["ddx"]["join_gold"] == "independent" else spec_id(g)


def constant_answers(g):
    return {"independent"} if g["ddx"]["join_gold"] == "independent" else set()


def answer_from_label(label):
    return {"join": label if label in {"unified", "comorbidity", "independent"} else "unified",
            "names": CODEBOOK.get(label, [])}


def group_key(sp):
    return None


def match_text(text):
    text = text.lower()
    hits = []
    for key, terms in ALIASES.items():
        positions = []
        for term in terms:
            pattern = re.escape(term.lower())
            if not re.search(r"[\u3400-\u9fff]", term):
                pattern = r"\b" + pattern + r"\b"
            match = re.search(pattern, text)
            if match is not None:
                positions.append(match.start())
        if positions:
            hits.append((-len(positions), min(positions), key))
    names = list(dict.fromkeys(n for _, _, key in sorted(hits)[:3] for n in CODEBOOK[key]))
    return {"join": "unified" if hits else "independent", "names": names}


def keyword_only(sp):
    return match_text(json.dumps(sp, ensure_ascii=False))


def summary_sentence_only(sp):
    return match_text(json.dumps(sp.get("prediction_context", {}).get("m2_frame", {}), ensure_ascii=False))


READERS = (
    Reader("keyword_only", "stub", keyword_only, "Pool-codebook aliases anywhere in the solver payload; no protocol."),
    Reader("summary_sentence_only", "stub", summary_sentence_only, "Alias matching only in the current encounter frame; no protocol."),
)
NOT_APPLICABLE = {
    "full": "Gold diagnosis is stored registry ddx; no visible full reader exists.",
    "today_only": "No production diagnosis decision procedure to ablate.",
    "latest_row_only": "No production latest-row diagnosis reader.",
    "trust_system_score": "No machine diagnosis score.",
    "checklist_only": "No production diagnosis checklist reader.",
    "latest_dispense_pillbox": "No dispensing and pill-count diagnosis reader.",
    "printed_value_only": "No production printed-value diagnosis reader.",
    "anomaly": "No production diagnosis anomaly reader.",
}


def features(sp):
    pc, latest = sp.get("prediction_context", {}), {}
    for section in ("outpatient_laboratory_record", "archived_laboratory_record"):
        for row in pc.get(section) or []:
            observations = row.get("observation") or [row]
            for observation in observations:
                day = observation.get("ts", row.get("day", row.get("draw_day", 0)))
                name = observation.get("name", row.get("name", row.get("test", row.get("analyte"))))
                value = observation.get("value", observation.get("result"))
                if name is not None and isinstance(value, (float, int)) and not isinstance(value, bool):
                    if name not in latest or day >= latest[name][0]:
                        latest[name] = day, value
    naive = {f"lab:{key}": value for key, (_, value) in latest.items()}
    frame = pc.get("m2_frame") or {}
    for key in ("frame", "encounter_type"):
        if frame.get(key) is not None:
            prefix = "encounter" if key == "encounter_type" else "frame"
            naive[f"{prefix}:{frame[key]}"] = 1
    noise = {f"n:{key}": len(value) for key, value in pc.items() if isinstance(value, list)}
    noise["payload_chars"] = len(json.dumps(sp, ensure_ascii=False))
    return naive, noise


def knockout(sp, full_answer):
    return None


def declared(g):
    signals = {s for t in (g.get("m2_evidence") or {}).get("threads", []) for s in t.get("key_signals", [])}
    return {"declared_key_signals": len(signals),
            "history_dependency_active": bool((g.get("history_dependency") or {}).get("active")),
            "note": "declared, not knockout tested"}


def blind_lists(pool, published):
    import numpy as np
    from sklearn.model_selection import KFold
    predictions, lists = {}, []
    if len(pool) < 5:
        return predictions, lists
    vocabulary = sorted({name for names in CODEBOOK.values() for name in names})
    for train, test in KFold(5, shuffle=True, random_state=0).split(np.arange(len(pool))):
        chosen = []
        for _ in range(min(3, len(vocabulary))):
            candidates = [name for name in vocabulary if name not in chosen]
            def count(name):
                return sum(ok({"join": "independent", "names": (chosen + [name])[:len(aliases(pool[i].gold)) + 1]},
                              pool[i].gold) for i in train)
            chosen.append(min(candidates, key=lambda name: (-count(name), name)))
        lists.append(chosen)
        for i in test:
            predictions[pool[i].case_id] = {"join": "independent", "names": chosen[:len(aliases(pool[i].gold)) + 1]}
    return predictions, lists
