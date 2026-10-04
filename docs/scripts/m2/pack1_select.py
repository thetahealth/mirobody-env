"""pack1_select.py -- the selection rule of pack 1 (differential diagnosis): which pool cases a pack of
N items takes. Imported by `tools/m2_gen_job.py`. It reads the prior tier, the arm, the spec id, the
sex and whether the hidden line carries an explain-away rendering (gold fields; never a score):

  hard      F1-F3 positive-arm hard, apportioned over F1/F2/F3 by their hard counts (largest
            remainder); within a frame explained-away cases first, then sha256(case_id + seed) order,
            skipping a spec id the frame already holds
  mid/easy  F1-F3 positive, F0 bridge non-insufficient and negative arm, each in sha256 order
  Negative arm: apportioned over F1/F2/F3 by the frames' positive counts (equal review share per
  frame); easy over the frames that have easy negatives; within the frame a female quota =
  round(quota x the frame's positive female share); spec ids not repeated.
  Balance gate: BAL-1 every frame both labels and >= 2 tiers; BAL-2 frame-only Youden J <= 0.05.

The cell sizes come from `haenv/m2/pack.py:quotas`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
from collections import Counter

BAL2_MAX = 0.05
FRAMES = ("F1", "F2", "F3")


def key(cid: str, seed: str = "") -> str:
    from haenv import pack_size as PS
    return hashlib.sha256(PS.seeded(cid, seed).encode("utf-8")).hexdigest()


def apportion(n: int, counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of n over counts (ties to the larger count, then name)."""
    tot = sum(counts.values())
    if tot == 0 or n == 0:
        return {f: 0 for f in counts}
    exact = {f: n * c / tot for f, c in counts.items()}
    out = {f: int(exact[f]) for f in counts}
    rest = n - sum(out.values())
    for f in sorted(counts, key=lambda f: (-(exact[f] - out[f]), -counts[f], f))[:rest]:
        out[f] += 1
    return out


def rows_of(job: dict) -> list[dict]:
    from haenv.m2.core import components
    prior = job["_provenance"]["m2_prior"]
    out = []
    for c in job["cases"]:
        cid = c["case_id"]
        lat = c.get("latent") or {}
        m2 = lat.get("m2") or {}
        frame = prior[cid]["frame"]
        tier = (m2.get("prior") or {}).get("tier") if frame != "F0" else prior[cid]["tier"]
        out.append({"case": cid, "frame": frame, "tier": tier,
                    "arm": m2.get("arm") if frame != "F0" else "bridge",
                    "insufficient": bool(lat.get("ddx_insufficient")),
                    "ea": bool((m2.get("explain_away") or {}).get("lines")),
                    "spec": lat.get("ddx_spec_id"), "sex": (c.get("raw") or {}).get("sex"),
                    "lines": (sorted(components(str(lat.get("ddx_spec_id") or "")))
                              if lat.get("ddx_join_gold") in ("unified", "comorbidity") else [])})
    return out


def _dedup_take(pool, n, held_specs, seed="", first=lambda r: 0):
    """`first`, then sha256 order, skipping spec ids already held; refilled from the skipped."""
    order = sorted(pool, key=lambda r: (first(r), key(r["case"], seed)))
    got, skipped, held = [], [], set(held_specs)
    for r in order:
        if len(got) == n:
            break
        (skipped if r["spec"] in held else got).append(r)
        held.add(r["spec"])
    return got + skipped[:n - len(got)]


def cells(rows: list[dict]) -> dict:
    """design cell -> eligible pool rows."""
    pos = [r for r in rows if r["arm"] == "positive"]
    neg = [r for r in rows if r["arm"] == "negative"]
    f0 = [r for r in rows if r["frame"] == "F0" and not r["insufficient"]]
    return {"hard": [r for r in pos if r["tier"] == "hard"],
            "pos_mid": [r for r in pos if r["tier"] == "mid"], "pos_easy": [r for r in pos if r["tier"] == "easy"],
            "f0_mid": [r for r in f0 if r["tier"] == "mid"], "f0_easy": [r for r in f0 if r["tier"] == "easy"],
            "neg_mid": [r for r in neg if r["tier"] == "mid"], "neg_easy": [r for r in neg if r["tier"] == "easy"]}


def select(rows: list[dict], q: dict, seed: str = "") -> list[dict]:
    from haenv import pack_size as PS
    pick: list[dict] = []
    C = cells(rows)
    PS.short(q, {k: len(v) for k, v in C.items()}, "m2 pack cells")

    def add(rs, why):
        pick.extend({**r, "why": why} for r in rs)
        return rs

    def sha(rs):
        return sorted(rs, key=lambda r: key(r["case"], seed))

    hard, neg = C["hard"], C["neg_mid"] + C["neg_easy"]
    qh = apportion(q["hard"], {f: sum(1 for r in hard if r["frame"] == f) for f in FRAMES})
    for f in FRAMES:
        add(_dedup_take([r for r in hard if r["frame"] == f], qh[f], set(), seed, first=lambda r: not r["ea"]),
            f"hard: {f} quota {qh[f]}, explained-away first, sha256, spec ids not repeated")
    add(sha(C["pos_mid"])[:q["pos_mid"]], "mid: F1-F3 positive")
    add(sha(C["pos_easy"])[:q["pos_easy"]], "easy: F1-F3 positive")
    add(sha(C["f0_mid"])[:q["f0_mid"]], "mid: F0 bridge non-insufficient")
    add(sha(C["f0_easy"])[:q["f0_easy"]], "easy: F0 bridge non-insufficient")
    n_neg = q["neg_mid"] + q["neg_easy"]
    npos = {f: sum(1 for r in pick if r["frame"] == f) for f in FRAMES}
    qn = apportion(n_neg, npos)
    has_easy = {f: qn[f] for f in FRAMES if any(r["frame"] == f and r["tier"] == "easy" for r in neg)}
    qe = {**{f: 0 for f in FRAMES}, **apportion(q["neg_easy"], has_easy)}
    for f in FRAMES:
        fem = [r["sex"] == "F" for r in pick if r["frame"] == f]
        n_f = round(qn[f] * sum(fem) / len(fem)) if fem else 0
        held: set = set()
        for tier, n in (("easy", qe[f]), ("mid", qn[f] - qe[f])):
            pool = [r for r in neg if r["frame"] == f and r["tier"] == tier]
            PS.short({f"neg {f} {tier}": n}, {f"neg {f} {tier}": len(pool)}, "m2 pack negative arm")
            have_f = sum(1 for r in pick if r["frame"] == f and r["arm"] == "negative" and r["sex"] == "F")
            want_f = max(0, min(n, n_f - have_f))
            got = _dedup_take([r for r in pool if r["sex"] == "F"], want_f, held, seed)
            held |= {r["spec"] for r in got}
            got += _dedup_take([r for r in pool if r["sex"] != "F"], n - len(got), held, seed)
            held |= {r["spec"] for r in got}
            if len(got) < n:                          # availability: fill in sha256 order
                rest = [r for r in pool if r not in got]
                got += _dedup_take(rest, n - len(got), held, seed)
            add(got, f"{tier}: negative arm {f} quota {n} (frame {qn[f]}, female {n_f}) by sha256")
    return pick


def youden_best(groups: dict) -> float:
    """Best in-sample Youden J of a rule 'answer warranted on this subset of groups':
    groups = {g: (n_pos, n_neg)}; J = max over subsets of TPR - FPR (= sum of the positive gaps)."""
    P = sum(p for p, _ in groups.values())
    N = sum(n for _, n in groups.values())
    if not P or not N:
        return 0.0
    return sum(max(0.0, p / P - n / N) for p, n in groups.values())


def balance(pick: list[dict], rows: list[dict]) -> dict:
    import math
    fc = [r for r in pick if r["frame"] in FRAMES]
    lab = {f: (sum(1 for r in fc if r["frame"] == f and r["arm"] == "positive"),
               sum(1 for r in fc if r["frame"] == f and r["arm"] == "negative")) for f in FRAMES}
    tiers = {f: dict(sorted(Counter(r["tier"] for r in fc if r["frame"] == f).items())) for f in FRAMES}
    fs = {}
    for r in fc:
        p, n = fs.get((r["frame"], r["sex"]), (0, 0))
        fs[(r["frame"], r["sex"])] = (p + (r["arm"] == "positive"), n + (r["arm"] == "negative"))
    tv = sorted({r["tier"] for r in fc})
    tab = [[sum(1 for r in fc if r["frame"] == f and r["tier"] == t) for t in tv] for f in FRAMES]
    tot = sum(map(sum, tab))
    chi = sum((tab[i][j] - sum(tab[i]) * sum(row[j] for row in tab) / tot) ** 2
              / (sum(tab[i]) * sum(row[j] for row in tab) / tot)
              for i in range(len(FRAMES)) for j in range(len(tv)) if sum(tab[i]) and sum(row[j] for row in tab))
    cv = math.sqrt(chi / (tot * (min(len(FRAMES), len(tv)) - 1)))
    # E_line: same-frame leave-one-out list vs pack leave-one-out list, top-(n+1), on positives
    posr = [r for r in pick if r["lines"]]

    def hit(r, pool):
        cnt = Counter(x for o in pool if o is not r for x in o["lines"])
        top = [x for x, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))][:len(r["lines"]) + 1]
        return len(set(top) & set(r["lines"])) / len(r["lines"])
    e = {f: [hit(r, [o for o in posr if o["frame"] == f]) - hit(r, posr) for r in posr if r["frame"] == f]
         for f in FRAMES}
    allv = [x for v in e.values() for x in v]
    bal1 = all(p and n for p, n in lab.values()) and all(len(t) >= 2 for t in tiers.values())
    j = youden_best(lab)
    return {"labels_by_frame": {f: {"positive": p, "negative": n} for f, (p, n) in lab.items()},
            "tiers_by_frame": tiers, "sex_by_frame_label": {f"{k[0]}:{k[1]}": {"positive": v[0], "negative": v[1]}
                                                            for k, v in sorted(fs.items(), key=str)},
            "BAL-1": {"pass": bal1}, "BAL-2": {"youden_frame": round(j, 4), "max": BAL2_MAX, "pass": j <= BAL2_MAX},
            "youden_frame_x_sex": round(youden_best(fs), 4), "cramers_v_frame_x_tier": round(cv, 3),
            "E_line": {"mean": round(sum(allv) / len(allv), 3),
                       "by_frame": {f: (round(sum(v) / len(v), 3) if v else None) for f, v in e.items()}},
            "specs_repeated_within_frame": {f: sorted(k for k, c in Counter(r["spec"] for r in fc if r["frame"] == f).items() if c > 1)
                                            for f in FRAMES},
            "pass": bal1 and j <= BAL2_MAX}
