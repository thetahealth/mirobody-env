"""W4 minimal symptom narration: how a patient tells what happened (generation side).

Before W4 every diagnosis line showed its four true symptoms, each stated outright. Here each
patient message is told in one of three ways, drawn from the hidden truth only (case id and
evidence id, `registry/narration.yaml`):

* `stated`  -- as before;
* `vague`   -- said without content ("就是不太舒服"), from a registered template;
* `omitted` -- the symptom is never mentioned; that day's message is about something unrelated.

Benign messages are told at the rates true symptoms reach after the floors
(`benign_disclosure`), so the telling carries no label. An
unrelated aside (diet question, work, daily life) is appended to some messages, again at one rate
for every message. Floors keep these true symptoms stated: the latest one at or before `T` (the
presenting complaint), any with a red-flag word, any registered M2 clue, at least two at or before
`T`, at least one per thread of a comorbid composition, and all of them on a red-flag case or a
threaded case whose symptom-to-thread map is unknown.

Gold follows the visible record: only stated true symptoms enter `real_symptom_evidence_ids`;
a vague or omitted one is noise on the record and takes one slot of the declared benign quota
(neither true evidence nor a listed distractor; recorded verifier-side in `manifest["_narration"]`).
Only the deterministic channel narrates. The tier is re-derived from the stated symptoms (`ddx.insufficient_from_visible`) and
the qualifiers from the visible ledger (`qualifiers.annotate`, unchanged).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

STATED, VAGUE, OMITTED = "stated", "vague", "omitted"
_REGISTRY = "narration.yaml"
_FORBIDDEN_SOURCES = ("verify.RED_FLAG_WORDS", "verify.TRUTH_WORDS_ZH", "verify.DRIVER_WORDS_ZH",
                      "events.ATTRIBUTIVE_MARKERS", "events.JOIN_STRUCTURE_MARKERS")


def config() -> dict:
    """`registry/narration.yaml`, validated (an empty dict when the file is absent)."""
    from haenv.resources.regpath import registry_path
    from haenv.resources.yamlcache import load_yaml
    try:
        d = load_yaml(registry_path(_REGISTRY)) or {}
    except FileNotFoundError:
        return {}
    for k in ("disclosure", "benign_disclosure", "chatter_append", "chatter_joiner", "vague", "chatter"):
        if k not in d:
            raise ValueError(f"{_REGISTRY}: missing {k!r}")
    for k in ("disclosure", "benign_disclosure"):
        p = d[k]
        if not (0 <= float(p["vague"]) and 0 <= float(p["omitted"])
                and float(p["vague"]) + float(p["omitted"]) < 1):
            raise ValueError(f"{_REGISTRY}: {k} rates must be >= 0 and sum below 1: {p}")
    return d


def enabled() -> bool:
    return bool(config().get("enabled"))


def forbidden_words() -> tuple[str, ...]:
    """Words no narration text may contain (red flags, answer words, attribution, join structure)."""
    import haenv.world.observations.events as events
    import haenv.scoring.verify as verify
    return (tuple(verify.RED_FLAG_WORDS) + tuple(verify.TRUTH_WORDS_ZH) + tuple(verify.DRIVER_WORDS_ZH)
            + tuple(events.ATTRIBUTIVE_MARKERS) + tuple(events.JOIN_STRUCTURE_MARKERS))


def check_registry() -> list[str]:
    """Problems with the registered texts: a forbidden word, a duplicate, an empty pool."""
    d = config()
    if not d:
        return []
    bad = forbidden_words()
    out = []
    for pool in ("vague", "chatter"):
        texts = [str(t) for t in d.get(pool) or ()]
        if len(texts) < 8:
            out.append(f"{pool}: only {len(texts)} texts (cross-case reuse gate GEN8d needs variety)")
        if len(set(texts)) != len(texts):
            out.append(f"{pool}: duplicate texts")
        for t in texts:
            hit = [w for w in bad if w in t]
            if hit:
                out.append(f"{pool}: {t!r} contains {hit}")
    return out


def _clue_texts() -> frozenset[str]:
    """The registered M2 clue texts (`registry/explain_away.yaml:clues`), read directly: importing
    `haenv.m2` would load a plugin as a side effect."""
    from haenv.resources.regpath import registry_path
    from haenv.resources.yamlcache import load_yaml
    try:
        d = load_yaml(registry_path("explain_away.yaml")) or {}
    except FileNotFoundError:
        return frozenset()
    return frozenset(str(t) for v in (d.get("clues") or {}).values() for t in (v or ()))


def _thread_groups(spec_id: str | None, n: int) -> list[tuple[int, ...]] | None:
    """Symptom positions per thread for a comorbid composition (a0, b0, a1, b1), else None."""
    from haenv.world.overlay import HAENV_COMORBID_PAIRS
    if spec_id in HAENV_COMORBID_PAIRS and n == 4:
        return [(0, 2), (1, 3)]
    return None


def _draw(case_id: str, eid: str, d: dict, key: str = "disclosure") -> str:
    import haenv.world.generation.rng as rng
    u = rng.unit(case_id, "w4_disclosure", eid)
    p = d[key]
    if u < float(p["omitted"]):
        return OMITTED
    if u < float(p["omitted"]) + float(p["vague"]):
        return VAGUE
    return STATED


def plan_true(case_id: str, T: int, symptoms: list[dict], *, red_flag: bool = False,
              threaded: bool = False, spec_id: str | None = None) -> dict[int, str]:
    """Telling of each true symptom (position in `symptoms` -> state); only symptoms at or before `T`
    are drawn, later ones are not shown anyway. The floors are in the module docstring."""
    from haenv.scoring.verify import RED_FLAG_WORDS
    d = config()
    pre = [i for i, s in enumerate(symptoms) if int(s["day"]) <= int(T)]
    out = {i: STATED for i in range(len(symptoms))}
    if not d.get("enabled") or len(pre) < 3:
        return out                       # two or fewer true symptoms at T: every one is needed
    groups = _thread_groups(spec_id, len(symptoms))
    if red_flag or (threaded and groups is None):
        return out
    clues = _clue_texts()
    keep = {pre[-1]}
    for i in pre:
        blob = f"{symptoms[i].get('text', '')} {symptoms[i].get('context', '')}"
        if any(w in blob for w in RED_FLAG_WORDS) or str(symptoms[i].get("text")) in clues:
            keep.add(i)
    for i in pre:
        if i not in keep:
            out[i] = _draw(case_id, f"S{i + 1}", d)

    def stated(ix):
        return [i for i in ix if out[i] == STATED]
    # At least one stated per thread, then at least two in all; promoted latest first.
    for g in groups or ():
        gp = [i for i in g if i in pre]
        if gp and not stated(gp):
            out[gp[-1]] = STATED
    for i in reversed(pre):
        if len(stated(pre)) >= 2:
            break
        out[i] = STATED
    return out


def tell(ev: dict, state: str, case_id: str, used: set[str]) -> None:
    """Rewrite one ledger message in place for a vague or omitted telling (context dropped: it
    would carry the content the telling withholds)."""
    import haenv.world.generation.rng as rng
    if state == STATED:
        return
    d = config()
    pool = [str(t) for t in d["vague" if state == VAGUE else "chatter"]]
    opts = [t for t in pool if t not in used] or pool
    t = rng.pick(opts, "w4_text", case_id, str(ev.get("evidence_id")), state)
    used.add(t)
    key = "symptom" if "symptom" in ev else "note"
    ev[key] = t
    ev.pop("context", None)


def append_chatter(ev: dict, case_id: str, used: set[str]) -> bool:
    """Append an unrelated aside to one message at the registered rate; True when appended."""
    import haenv.world.generation.rng as rng
    d = config()
    eid = str(ev.get("evidence_id"))
    if rng.unit(case_id, "w4_chatter", eid) >= float(d["chatter_append"]):
        return False
    key = "symptom" if ev.get("symptom") else "note"
    base = str(ev.get(key) or "")
    if not base or base in used:     # a message that is already chatter gets no second aside
        return False
    pool = [str(t) for t in d["chatter"]]
    opts = [t for t in pool if t not in used] or pool
    t = rng.pick(opts, "w4_chatter_text", case_id, eid)
    used.add(t)
    ev[key] = f"{base}{d['chatter_joiner']}{t}"
    return True


def symptom_index(evidence_id: str) -> int:
    """Position in the raw symptom list of a true-symptom EV id (`EV-<case>-S<i>`, 1-based)."""
    return int(str(evidence_id).rsplit("-S", 1)[-1]) - 1


def narrate(case_id: str, real_evs: list[dict], other_evs: list[dict], fixed_evs: list[dict] = (), *,
            plan: dict[int, str]) -> dict[str, dict]:
    """Apply W4 to one case's ledger in place. `real_evs` are the true-symptom EVs (`EV-<id>-S<i>`)
    told by `plan` (`plan_true`, possibly capped by the caller), `other_evs` the benign patient
    messages, `fixed_evs` messages that keep their content (look-alike distractors: the judge
    expects them merged in) and only take the aside. Returns
    `{evidence_id: {"disclosure", "chatter"}}`."""
    if not enabled():
        return {}
    d = config()
    used: set[str] = set()
    rec: dict[str, dict] = {}
    for ev in real_evs:
        eid = str(ev["evidence_id"])
        st = plan.get(symptom_index(eid), STATED)
        tell(ev, st, case_id, used)
        rec[eid] = {"disclosure": st}
    for ev in other_evs:
        eid = str(ev["evidence_id"])
        st = STATED
        if ev.get("source_type") == "patient_reported_symptom":
            st = _draw(case_id, eid, d, "benign_disclosure")
            tell(ev, st, case_id, used)
        rec[eid] = {"disclosure": st}
    for ev in fixed_evs:
        rec[str(ev["evidence_id"])] = {"disclosure": STATED}
    for ev in list(real_evs) + list(other_evs) + list(fixed_evs):
        eid = str(ev["evidence_id"])
        rec[eid]["chatter"] = append_chatter(ev, case_id, used)
    return rec
