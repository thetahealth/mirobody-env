"""The rl declaration for the shared batch audit (`shared_audit.PackAudit`, `tools/validation/pack_audit.py`).

An rl pack is a pair of batches built on the same worlds (same case ids): the redline arm and its control arm
(`make_pack --paired-controls`, then `--paired-with ... --paired-source control`). Each arm's job records the
other arm's pool batch in `_provenance.rl_partner_batch`; the audit loads it, so

* SA-5 / SA-6 read both arms (the label is the gold; a surface feature that tells the arms apart is a cue), and
* the pair gates run: G4 one-patch pair (the two blocks differ only inside the declared patch span, by
  `twins._diff` on a row-keyed form of the block) and G5 paired imminence (the same planned entry and plan
  message in both arms).
* MC (every item, both arms) medication consistency: the record, Medications, Dispensing and the problem
  list agree (`med_consistency`).

The published subset is this arm's items only. Without a partner the pair gates report `not_applicable`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import json
import pathlib

import haenv.world.validation.pack_size as PS

BLOCK = "redline_rl"
CLASSES = ("answer_and_raise", "answer_only")
PARTNER = "#partner"
#: surface features that are the same text in both arms by construction (SA-5 / SA-6)
SURFACE = ("age", "sex", "n_visits", "n_messages", "n_meds", "n_problems", "q_chars", "n_dispensing", "fam", "T")


def class_of(g):
    return g.get("gold") if g.get("gold") in CLASSES else None


def context(batch, job_path, ref) -> dict:
    ctx = {"partner": None}
    if job_path:
        from haenv.resources.yamlcache import load_yaml
        prov = (load_yaml(pathlib.Path(job_path), wrap_text=False) or {}).get("_provenance") or {}
        if prov.get("rl_partner_batch"):
            ctx["partner"] = str(prov["rl_partner_batch"])
        if prov.get("rl_family_mix"):
            ctx["family_mix"] = dict(prov["rl_family_mix"])
    return ctx


def prepare(items, ctx) -> list[dict]:
    from haenv.world.validation.shared_audit import load_items
    for it in items:
        it["pair_id"] = it["case_id"]
        it["arm"] = it["gold"].get("variant")
    if ctx.get("partner"):
        other = load_items(pathlib.Path(ctx["partner"]), BLOCK, class_of)
        for it in other:
            it["pair_id"] = it["case_id"]
            it["arm"] = it["gold"].get("variant")
            it["case_id"] = it["case_id"] + PARTNER
            it["partner"] = True
        items = items + other
    ctx["own_arm"] = next((it["arm"] for it in items if not it.get("partner")), None)
    ctx["partner_ids"] = {it["pair_id"] for it in items if it.get("partner")}
    return items


def _own(items):
    return [it for it in items if not it.get("partner")]


def family_quota(n: int, families, mix: dict | None = None) -> dict:
    """The published count per present family: the job's recorded mix (`_provenance.rl_family_mix`) when set,
    else the registry's `family_mix`."""
    from haenv.rl.rules import table
    mix = {f: w for f, w in (mix or table().get("family_mix") or {}).items() if f in set(families)}
    return PS.apportion(n, mix) if mix else {}


def planned(items, n_items: int, ctx) -> dict:
    """{(family, rule_id): n}: the families present split by the registry mix, each split evenly over its rules,
    a rule's shortfall moving to the other rules of its family (available supply: this arm's items that also
    exist in the partner arm, when one is loaded)."""
    own = _own(items)
    if ctx.get("partner_ids"):
        own = [it for it in own if it["pair_id"] in ctx["partner_ids"]]
    have = collections.Counter((it["gold"]["family"], it["gold"]["rule_id"]) for it in own)
    fams = sorted({f for f, _ in have})
    out = {}
    for fam, k in family_quota(n_items, fams, ctx.get("family_mix")).items():
        rules = sorted(r for f, r in have if f == fam)
        left, need = k, {}
        open_ = list(rules)
        while left and open_:
            share = PS.apportion(left, dict.fromkeys(open_, 1))
            moved = 0
            for r in list(open_):
                take = min(share[r], have[(fam, r)] - need.get(r, 0))
                need[r] = need.get(r, 0) + take
                moved += take
                if need[r] >= have[(fam, r)]:
                    open_.remove(r)
            left -= moved
            if not moved:
                break
        out.update({(fam, r): v for r, v in need.items() if v})
    return out


def select(items, n_items: int, seed: str, ctx) -> list[dict]:
    """The published items, nested (`PS.nest_chain`): the pack of each smaller nested size is planned by the
    same mix and comes first, and a larger pack only extends it (each cell in its seeded order)."""
    from haenv.rl.plan import h01
    own = _own(items)
    if ctx.get("partner_ids"):
        own = [it for it in own if it["pair_id"] in ctx["partner_ids"]]
    tag = PS.seeded("rlpick", seed)
    out, have = [], collections.Counter()
    for n in PS.nest_chain(n_items):
        want = planned(items, n, ctx)
        for cell, k in sorted(want.items()):
            if have[cell] > k:
                raise SystemExit(f"[pack] rl non-monotone nested quota: {cell} {have[cell]} > {k} at N={n}")
            pool = sorted((it for it in own if (it["gold"]["family"], it["gold"]["rule_id"]) == cell),
                          key=lambda it: h01(tag, it["case_id"]))
            out += pool[have[cell]:k]
            have[cell] = k
        if any(have[c] and c not in want for c in have):
            raise SystemExit(f"[pack] rl non-monotone nested quota: a cell left the plan at N={n}")
    return out


def rows(items, ctx=None) -> list[dict]:
    out = []
    for it in items:
        b = ((it.get("sp") or {}).get("prediction_context") or {}).get(BLOCK) or {}
        pat = b.get("patient", "")
        age = int(pat.split(", ")[1].split()[0]) if ", " in pat else 0
        if b.get("lang") == "zh" and "，" in pat:              # "女，63 岁；..."
            age = int(pat.split("，")[1].split()[0])
        out.append({"age": age, "sex": int(pat.startswith(("female", "女"))), "n_visits": len(b.get("visits") or ()),
                    "n_messages": len(b.get("messages") or ()), "n_meds": len(b.get("medications") or ()),
                    "n_problems": len(b.get("problem_list") or ()), "q_chars": len(b.get("question") or ""),
                    "n_dispensing": len(b.get("dispensing") or ()), "fam": int(it["gold"].get("family") == "E"),
                    "T": int(((it.get("sp") or {}).get("prediction_context") or {}).get("prediction_time_T") or 0),
                    "class": it["class"], "spec": it["spec"], "case_id": it["case_id"]})
    return out


# ------------------------------------------------------------------------------------- pair gates

def canonical(block: dict) -> dict:
    b = dict(block or {})
    b["results"] = {f"{r['date']}|{r['analyte']}": r for r in block.get("results") or ()}
    b["home_monitoring"] = {h["month"][:7]: h for h in block.get("home_monitoring") or ()}
    return b


#: block sections a carrier (families A, B, G) may change between arms, by exact path
CARRIER_SECTIONS = ("allergies", "past_history", "visits", "dispensing", "glucose_log")


def _allowed(path: str, patch: dict) -> bool:
    if any(path == p or path.startswith(p + "/") for p in patch.get("paths") or ()):
        return True
    parts = path.strip("/").split("/")
    if parts[0] == "results" and len(parts) >= 2:
        date, _, analyte = parts[1].partition("|")
        return analyte in (patch.get("analytes") or ()) and (not patch.get("from_date") or date >= patch["from_date"])
    if parts[0] == "home_monitoring" and len(parts) >= 3 and patch.get("hm_field"):
        return parts[2] == patch["hm_field"] and parts[1] >= patch.get("hm_from_month", "9999")
    return False


def pair_diff(a: dict, b: dict, patch: dict) -> list[str]:
    """Paths where two arms' blocks differ outside the declared patch span."""
    from haenv.world.generation.twins import _diff
    return [p for p in _diff(canonical(a), canonical(b)) if not _allowed(p, patch)]


def _patch_ok(g: dict) -> list[str]:
    """A declared span may only name the rule's own trigger / companion / test rows."""
    from haenv.rl.rules import display, rule
    r = rule(g["rule_id"])
    t = r["trigger"]
    comp = display().get("companions") or {}
    tests = display().get("tests") or {}
    own = {t.get("display")} | {comp[c["analyte"]]["display"] for c in t.get("companions") or ()}
    own |= {tests[a]["display"] for a in t.get("analytes") or () if a in tests}
    if t.get("test") == "creatinine_crcl":
        own |= {tests["creatinine"]["display"], tests["crcl"]["display"]}
    from haenv.rl.realize import PANEL_TESTS, panel_specs
    if t.get("test") in PANEL_TESTS:                    # the required result is part of the chemistry panel draw
        own |= {sp["display"] for sp in panel_specs(r, "chem")}
    if t.get("kind") == "committed_check_missing":       # F: the committed check's rows after the start
        fu = r.get("followup") or {}
        own |= set(t.get("displays") or ())
        own |= {sp["display"] for p in fu.get("panels") or () for sp in panel_specs(r, p)}
        own |= {fu["level"]["display"]} if fu.get("level") else set()
    from haenv.rl import zh as Z
    own |= {z for o in list(own) if o for z in Z.both(o)[1:]}          # a zh record names them in zh
    bad = [a for a in (g.get("patch") or {}).get("analytes") or () if a not in own]
    bad += [p for p in (g.get("patch") or {}).get("paths") or () if p.strip("/").split("/")[0] not in CARRIER_SECTIONS]
    return bad


def _pairs(pool):
    by = collections.defaultdict(dict)
    for it in pool:
        by[it["pair_id"]][it["arm"]] = it
    return {k: v for k, v in by.items() if "redline" in v and "control" in v}


def _block(it):
    return ((it.get("sp") or {}).get("prediction_context") or {}).get(BLOCK) or {}


def g4(pool, rows_, ctx) -> dict:
    pairs = _pairs(pool)
    if not ctx.get("partner"):
        return {"id": "G4", "status": "not_applicable: no partner arm loaded", "pass": True}
    bad = {}
    for pid, arms in sorted(pairs.items()):
        red, ctl = arms["redline"], arms["control"]
        why = []
        if red["gold"]["patch"] != ctl["gold"]["patch"]:
            why.append("declared patch spans differ between arms")
        why += [f"undeclared patch analyte {a}" for a in _patch_ok(red["gold"])]
        why += pair_diff(_block(red), _block(ctl), red["gold"]["patch"])
        if why:
            bad[pid] = why[:5]
    return {"id": "G4", "n_pairs": len(pairs), "violations": len(bad), "examples": dict(list(bad.items())[:5]),
            "pass": not bad and bool(pairs)}


def g5(pool, rows_, ctx) -> dict:
    pairs = _pairs(pool)
    if not ctx.get("partner"):
        return {"id": "G5", "status": "not_applicable: no partner arm loaded", "pass": True}
    bad = {}
    for pid, arms in sorted(pairs.items()):
        a, b = _block(arms["redline"]), _block(arms["control"])
        why = []
        if not a.get("planned") or a.get("planned") != b.get("planned"):
            why.append("planned entries differ or are missing")
        msg = arms["redline"]["gold"]["imminence"]["message"]
        if msg not in (a.get("messages") or ()) or msg not in (b.get("messages") or ()):
            why.append("plan message missing from an arm")
        if arms["redline"]["gold"]["imminence"] != arms["control"]["gold"]["imminence"]:
            why.append("imminence differs between arms")
        if why:
            bad[pid] = why
    return {"id": "G5", "n_pairs": len(pairs), "violations": len(bad), "examples": dict(list(bad.items())[:5]),
            "pass": not bad and bool(pairs)}


_ENDED = ("stopped", "held", "停用", "暂停")


def _line_name(line: str) -> str:
    import re
    return re.split(r"\s\d|\s\(|,|（|，", line.lower(), maxsplit=1)[0].strip()


def _event_dose(event: str) -> str | None:
    """The dose a record event leaves the patient on: the new dose of a change, else the first number."""
    import re
    m = re.search(r"→\s*([\d.]+)", event)
    if m:
        return m.group(1)
    m = re.search(r"\d+(?:\.\d+)?", event)
    return m.group(0) if m else None


def med_consistency(block: dict, g: dict) -> list[str]:
    """Medication-section consistency of one record: a medicine whose last record event is not a stop or hold
    is in Medications at that dose, in Dispensing, and has its indication in the problem list; a listed
    medicine is not stopped or held by the record; every listed medicine is dispensed."""
    import re
    meds = [m.lower() for m in block.get("medications") or ()]
    disp = [re.sub(r"^\d{4}-\d{2}-\d{2}:\s*", "", x).lower() for x in block.get("dispensing") or ()]
    problems = [p.lower() for p in block.get("problem_list") or ()]
    ind = {str(m["name"]).lower(): m.get("indication") for m in g.get("med_index") or ()}
    last = {}
    for r in sorted(block.get("medication_record") or (), key=lambda r: r["date"]):
        last[str(r["drug"]).lower()] = str(r["event"])
    out = []
    word = lambda n, text: re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", text)
    for drug, ev in last.items():
        line = next((m for m in meds if m.startswith(drug)), None)
        if ev.lower().startswith(_ENDED):
            if line:
                out.append(f"{drug}: listed in Medications but the record's last event is {ev!r}")
            continue
        if line is None:
            out.append(f"{drug}: current by the record ({ev!r}) but not in Medications")
            continue
        dose = _event_dose(ev)
        if dose and not re.search(r"(?<![\d.])" + re.escape(dose) + r"(?![\d.])", line):
            out.append(f"{drug}: record leaves dose {dose}, Medications says {line!r}")
        if drug in ind and ind[drug] and ind[drug].lower() not in problems:
            out.append(f"{drug}: indication {ind[drug]!r} not in the problem list")
    for line in meds:
        name = _line_name(line)
        if not any(x.startswith(name) for x in disp):
            out.append(f"{name}: in Medications but never dispensed")
    for name, i in ind.items():
        if i and i.lower() not in problems:
            out.append(f"{name}: indication {i!r} not in the problem list")
        if not any(word(name, m) for m in meds):
            out.append(f"{name}: an item medicine missing from Medications")
    return sorted(set(out))


def med_gate(pool, rows_, ctx) -> dict:
    bad = {}
    for it in pool:
        why = med_consistency(_block(it), it["gold"])
        if why:
            bad[it["case_id"]] = why[:4]
    return {"id": "MC", "n": len(pool), "violations": len(bad), "examples": dict(list(bad.items())[:5]),
            "pass": not bad}


def bv_gate(pool, rows_, ctx) -> dict:
    """BV: no result moves from the patient's own recent mean beyond biological variation unless the rule
    designs it (`variation.py`)."""
    from haenv.rl import variation as V
    from haenv.rl.rules import rule
    bad = {}
    for it in pool:
        why = V.violations(_block(it), rule(it["gold"]["rule_id"]), it["gold"]["variant"])
        if why:
            bad[it["case_id"]] = why[:3]
    return {"id": "BV", "n": len(pool), "violations": len(bad), "examples": dict(list(bad.items())[:5]),
            "pass": not bad}


def _med_names(block: dict, g: dict) -> dict[str, list[str]]:
    """Medicine -> the names it is mentioned by: the item's drug by its ingredient, classes and aliases; every
    other listed medicine by the words of its list line before the first number."""
    import re
    terms = [t.lower() for t in g.get("target_names") or ()]
    out = {"__target__": terms}
    for line in block.get("medications") or ():
        low = line.lower()
        if any(re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", low) for t in terms):
            continue
        name = re.split(r"\s\d|\s\(", low, maxsplit=1)[0].strip()
        if name:
            out[name] = [name]
    return out


def salience_of(block: dict, g: dict) -> dict:
    """Rank of the item's drug among the record's medicines by (a) mentions in the rendered record and (b) the
    position of its last mention (rank 1 = most mentions / latest last mention; ties share the better rank)."""
    import re
    from haenv.rl.render import render_text
    text = render_text(block).lower()
    count, last = {}, {}
    for med, names in _med_names(block, g).items():
        spans = sorted((m.start(), m.end()) for t in names
                       for m in re.finditer(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", text))
        hits, end = [], -1                              # one mention per place ("insulin glargine" ~ "glargine")
        for a, b in spans:
            if a >= end:
                hits.append(a)
            end = max(end, b)
        count[med], last[med] = len(hits), max(hits) if hits else -1
    rank = lambda d: 1 + sum(1 for k, v in d.items() if k != "__target__" and v > d["__target__"])
    return {"n_meds": len(count), "rank_mentions": rank(count), "rank_last": rank(last),
            "mentions": count["__target__"], "max_other_mentions": max([v for k, v in count.items() if k != "__target__"] or [0])}


def salience(items) -> dict:
    """Report only (no gate): how often the item's drug is the most-mentioned / last-mentioned medicine, against
    the base rate 1 / number of medicines."""
    rows = [salience_of(_block(it), it["gold"]) for it in items]
    if not rows:
        return {"id": "salience", "report_only": True, "pass": True, "n": 0}
    n = len(rows)
    return {"id": "salience", "report_only": True, "pass": True, "n": n,
            "share_rank1_mentions": round(sum(r["rank_mentions"] == 1 for r in rows) / n, 3),
            "share_rank1_last_mention": round(sum(r["rank_last"] == 1 for r in rows) / n, 3),
            "base_rate_mean_1_over_n_meds": round(sum(1 / r["n_meds"] for r in rows) / n, 3),
            "mean_n_meds": round(sum(r["n_meds"] for r in rows) / n, 2),
            "by_family": {f: round(sum(r["rank_mentions"] == 1 for r, it in zip(rows, items) if it["gold"]["family"] == f)
                                   / max(1, sum(1 for it in items if it["gold"]["family"] == f)), 3)
                          for f in sorted({it["gold"]["family"] for it in items})}}


def audit(n_items: int = 30, seed: str = ""):
    from haenv.world.validation.shared_audit import PackAudit
    state: dict = {}

    def _prepare(items, ctx):
        out = prepare(items, ctx)
        state.update(ctx)
        state["_pool"] = out
        return out
    return PackAudit(
        name="rl", block=BLOCK, classes=CLASSES, class_of=class_of, rows=rows,
        univariate=SURFACE, surface=SURFACE, cc=None,
        cell_of=lambda it, ctx: (it["gold"]["family"], it["gold"]["rule_id"]),
        planned_cells=lambda items, ctx: planned(items_all(state), n_items, state),
        select=lambda items: select(items, n_items, seed, state),
        context=context, prepare=_prepare,
        pack_gates={"G4": g4, "G5": g5, "MC": med_gate, "BV": bv_gate, "salience": lambda pool, rows_, ctx: salience(_own(pool))})


def items_all(state):
    return state.get("_pool", [])
