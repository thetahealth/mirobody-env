"""Pack 4 allocation: which case carries which follow-up item (design doc 4.2-4.3, PREREG 4).

Every decision here is a function of `sha256(tag, case_id)` and of the case's eligibility, never
of a case feature that the solver sees:

* class: inside each stratum of the hidden-disease flag H, cases sorted by `h01("pack4", id)` are
  dealt the four classes round-robin from a stratum offset (`alloc_balanced`, the probe's rule);
* analyte: inside each class, dealt alternately (fasting glucose / creatinine), falling back to
  the other analyte when the dealt one is not eligible for the case;
* sign: alternated inside each (analyte, class);
* band position, decoys and the preferred non-true factor: per-case hashes.

Eligibility (PREREG 3): the followed analyte must not be a marker of the case's hidden line, and
the world must not already carry a lab stream of the same name.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json

CLASSES = ("true_change", "analytic_biological_noise", "preanalytical", "method_difference")
#: Followed analytes (PREREG R10: creatinine only; glucose true changes exist only during GLP-1
#: titration in this world, so on glucose the course stage named the class).
ANALYTES = ("creatinine",)
DECOY_Q = (0.85, 0.40)            # P(>= 1 decoy), P(2nd decoy | 1st)
PLAN_VERSION = 1
#: Fasting glucose is followed only on cases whose primary drug is a GLP-1-class agent: its
#: visible true-change cause is that dose line (another glucose-acting line, metformin, has no
#: entry in the factor vocabulary; a non-glucose line leaves only a rare weight change).
GLP1_CLASS = ("semaglutide", "liraglutide", "dulaglutide", "tirzepatide")


def h01(*parts) -> float:
    s = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(s[:12], 16) / float(16 ** 12)


def hidden(latent: dict) -> int:
    return int(str((latent or {}).get("ddx_join_gold")) in ("unified", "comorbidity"))


def marker_text(case: dict, spec: dict | None) -> str:
    """Everything that names the case's hidden line: its symptoms, the ddx gold fields and the
    condition registry entry. Only H = 1 cases have a hidden line."""
    lat = case.get("latent") or {}
    bits = {"symptoms": (case.get("raw") or {}).get("symptoms"),
            "ddx": {k: v for k, v in lat.items() if k.startswith("ddx_")},
            "spec": spec or {}}
    return json.dumps(bits, ensure_ascii=False, default=str)


def acei_indication(disease, comorbidities) -> str | None:
    """The condition (code) a patient with this disease and these comorbidities takes an ACEI for,
    in the registry's order; None when the patient has no ACEI indication."""
    from ..labworld import tables
    have = {str(disease or "")} | {str(c) for c in (comorbidities or ())}
    ind = tables()["effects"]["acei_on_creatinine"]["indications"]
    return next((c for c in ind if c in have), None)


def today_fasting(case: dict) -> bool:
    """Whether the case's draw on its prediction day T falls in the fasting hours (world clock)."""
    from ..build import _draw_clock
    from .render import is_fasting
    T = int((case.get("latent") or {}).get("index_time_T", 84))
    return is_fasting(_draw_clock(str(case["case_id"]), T, 0)[:5])


def eligible_analytes(case: dict, spec: dict | None, world_streams: set[str]) -> list[str]:
    from ..labworld import analyte
    out = []
    raw = case.get("raw") or {}
    txt = marker_text(case, spec) if hidden(case.get("latent") or {}) else ""
    for a in ANALYTES:
        toks = analyte(a).get("marker_tokens") or ()
        if txt and any(t.lower() in txt.lower() for t in toks):
            continue
        if a in world_streams:
            continue
        if a == "fasting_glucose" and str(raw.get("drug")) not in GLP1_CLASS:
            continue
        # creatinine items carry an ACEI record, and an ACEI only goes to a patient with an indication
        if a == "creatinine" and acei_indication(raw.get("disease"), raw.get("comorbidities")) is None:
            continue
        out.append(a)
    return out


def signs_of(analyte: str) -> list[int]:
    """Signs the analyte is followed on (`registry/labworld.yaml`; both when not restricted)."""
    from ..labworld import tables
    if analyte == "fasting_glucose":
        return [int(x) for x in tables()["effects"]["glp1_on_fasting_glucose"].get("signs") or (1, -1)]
    return [1, -1]


def alloc_balanced(ids, H, classes=CLASSES, tag="pack4", block=None) -> dict[str, str]:
    """Inside each H stratum, cases in hash order are dealt the classes round-robin. With `block`
    (`{case_id: value}`, R11: the patient's sex), cases are ordered by block first, so the deal
    runs through each block in turn and the block value cannot predict the class."""
    lab = {}
    for h in (0, 1):
        sub = sorted([i for i, hh in zip(ids, H) if hh == h],
                     key=lambda c: ((block or {}).get(c, ""), h01(tag, c)))
        off = int(h01(tag, "offset", h) * len(classes))
        for k, c in enumerate(sub):
            lab[c] = classes[(k + off) % len(classes)]
    return lab


#: Decoy rates per followed analyte (PREREG revision R6, the part R10 keeps). The design's (0.85, 0.40) assumed half
#: the items on each analyte; here a third of the pack follows glucose, so the creatinine-acting
#: factors (each the cause of about a sixth of the pack) would appear as decoys far less often
#: than as causes and would name their class without the analyte. Each factor should appear as a
#: decoy about as often as it appears as a cause: on a glucose item each of the three
#: creatinine-acting factors independently with 0.5 (at most two); on a creatinine item the
#: glucose-acting pool of five with (0.45, 0.15).
DECOY_INDEPENDENT = {"fasting_glucose": (0.5, 2)}
DECOY_Q_BY_ANALYTE = {"creatinine": (0.45, 0.15)}


def decoys(analyte: str, cid: str, q=None) -> list[str]:
    from ..labworld.perturb import decoys_for
    pool = decoys_for(analyte)
    out: list[str] = []
    if q is None and analyte in DECOY_INDEPENDENT:
        p, cap = DECOY_INDEPENDENT[analyte]
        out = [f for f in pool if h01("p4di", cid, f) < p]
        out.sort(key=lambda f: h01("p4dio", cid, f))
        return out[:cap]
    q = q or DECOY_Q_BY_ANALYTE.get(analyte, DECOY_Q)
    if h01("p4d1", cid) < q[0]:
        out.append(pool[int(h01("p4d1f", cid) * len(pool))])
        if h01("p4d2", cid) < q[1]:
            rest = [f for f in pool if f not in out]
            out.append(rest[int(h01("p4d2f", cid) * len(rest))])
    return out


def factor_pref(analyte: str, cls: str, cid: str) -> str | None:
    """The non-true factor this item uses (true change picks its cause from the world)."""
    from ..labworld.perturb import factors_of
    kind = {"preanalytical": "preanalytical", "method_difference": "method"}.get(cls)
    if kind is None:
        return None
    opts = factors_of(analyte, kind)
    return opts[int(h01("p4fac", cid) * len(opts))]


#: R15: classes a CKD patient can carry (a Jaffe switch cannot carry a change beyond the RCV on a
#: CKD creatinine, so there is no method-difference item at that level)
COMORB_CLASSES = ("true_change", "analytic_biological_noise", "preanalytical")


def comorbidity_of(ids, Hs: dict, cls: dict) -> dict[str, str]:
    """R15: inside each (H, class) that can carry it, every `1/share`-th case in
    `h01("p4comorb", id)` order has CKD, so the same share of every such class carries it."""
    from ..labworld import tables
    reg = (tables().get("comorbidities") or {}).get("CKD")
    if not reg:
        return {}
    step = int(round(1.0 / float(reg["share"])))
    out = {}
    for h in (0, 1):
        for c in COMORB_CLASSES:
            sub = sorted([i for i in ids if Hs[i] == h and cls[i] == c], key=lambda x: h01("p4comorb", x))
            out.update({i: "CKD" for k, i in enumerate(sub) if k % step == 0})
    return out


def make_plans(cases: list[dict], specs: dict[str, dict], world_streams: dict[str, set[str]],
               feasible: dict[str, list[list]] | None = None) -> dict[str, dict]:
    """`{case_id: plan}` for every case that can carry an item (cases with no eligible analyte
    are left out and counted by the caller).

    `feasible` (`{case_id: [[analyte, sign], ...]}`, from a deterministic build of the first-pass
    plan, `tools/p4_gen_job.py`) makes the analyte / sign dealing inside each class world-aware:
    inside each (hash-dealt) class, glucose goes first to the cases that can carry nothing else,
    then to the others that can, up to the smallest glucose supply over the classes plus one;
    every other case takes the creatinine sign with the fewest cases so far. The class itself
    never looks at it."""
    ok = {str(c["case_id"]): eligible_analytes(c, specs.get(str((c.get("latent") or {}).get("ddx_spec_id"))),
                                               world_streams.get(str(c["case_id"]), set()))
          for c in cases}
    ids = [str(c["case_id"]) for c in cases if ok[str(c["case_id"])]]
    by_id = {str(c["case_id"]): c for c in cases}
    Hs = {str(c["case_id"]): hidden(c.get("latent") or {}) for c in cases}
    sex = {str(c["case_id"]): str((c.get("raw") or {}).get("sex") or "") for c in cases}
    cls = alloc_balanced(ids, [Hs[i] for i in ids], block=sex)
    an_of, sign_of = {}, {}
    fg_target = None
    if feasible is not None:
        # glucose items per class: the smallest glucose supply over the four classes, plus one
        # (the pack takes the same number from every class, so the analyte does not tell the class)
        sup = {c: sum(1 for i in ids if cls[i] == c and any(o[0] == "fasting_glucose" for o in feasible.get(i) or ()))
               for c in CLASSES}
        fg_target = min(sup.values()) + 1
    for c in CLASSES:
        sub = sorted([i for i in ids if cls[i] == c], key=lambda x: h01("p4an", x))
        if feasible is not None:
            cnt: dict[tuple, int] = {}
            n_fg = 0
            only_fg = [i for i in sub if (feasible.get(i) or ()) and all(o[0] == "fasting_glucose" for o in feasible[i])]
            some_fg = [i for i in sub if i not in only_fg and any(o[0] == "fasting_glucose" for o in feasible.get(i) or ())]
            rest = [i for i in sub if i not in only_fg and i not in some_fg]
            for cid in only_fg + some_fg + rest:
                opts = [tuple(o) for o in feasible.get(cid) or ()]
                if not opts:
                    opts = [(ok[cid][0], signs_of(ok[cid][0])[0])]
                fg = [o for o in opts if o[0] == "fasting_glucose"]
                cr = [o for o in opts if o[0] != "fasting_glucose"]
                if fg and (n_fg < fg_target or not cr):
                    pick = fg[0]
                    n_fg += 1
                else:
                    cr.sort(key=lambda o: (cnt.get(o, 0), h01("p4opt", cid, *o)))
                    pick = cr[0] if cr else opts[0]
                an_of[cid], sign_of[cid] = pick[0], int(pick[1])
                cnt[pick] = cnt.get(pick, 0) + 1
            continue
        n = {a: 0 for a in ANALYTES}
        for k, cid in enumerate(sub):
            pref = ANALYTES[k % len(ANALYTES)]
            cand = [a for a in ANALYTES if a in ok[cid]]
            cand.sort(key=lambda a: (n[a], a != pref))
            an_of[cid] = cand[0]
            n[cand[0]] += 1
        for a in ANALYTES:
            sub2 = sorted([i for i in sub if an_of[i] == a], key=lambda x: h01("p4sign", x))
            signs = signs_of(a)
            off = int(h01("p4sign", "offset", c, a) * len(signs))
            for k, cid in enumerate(sub2):
                sign_of[cid] = signs[(k + off) % len(signs)]
                if c == "preanalytical" and a == "creatinine":
                    # a meal sits only before a non-fasting draw: on today's when today's draw is
                    # non-fasting (the value rises), else on the previous one (it falls)
                    sign_of[cid] = 1 if not today_fasting(by_id[cid]) else -1
    cm = comorbidity_of(ids, Hs, cls)
    return {cid: {"version": PLAN_VERSION, "class": cls[cid], "analyte": an_of[cid],
                  **({"comorbidity": cm[cid]} if cm.get(cid) else {}),
                  "sign": sign_of[cid], "u_ratio": round(h01("p4ratio", cid), 6),
                  "decoys": {a: decoys(a, cid) for a in ok[cid]},
                  "factor": {a: factor_pref(a, cls[cid], cid) for a in ok[cid]},
                  "analytes_ok": list(ok[cid]), "H": Hs[cid]}
            for cid in ids}
