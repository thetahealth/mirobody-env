"""Generation-time audit shared by every plugin pack (SA-1 ... SA-9).

A pack declares what only it knows (its gold block, how gold is re-derived from the records, its
gold words, its composition cells, the content-free variables a solver could read off the payload,
its realism rules); the checks themselves are written once, here.

  SA-1 `emission_gate`     per case at build: unrealizable, class != plan, gold re-derived from the
                           records != gold, pack callbacks (margins, ...), block missing, gold words
                           or gold numbers in what the solver sees  -> refused
  SA-2 `prompt_gold_hits`  no pack's plan or gold key in a generation-model premise
  SA-3 `composition`       emitted cells equal the planned cells (an assertion, not a test)
  SA-4 `stub_floor`        constant stubs score cc 0, a uniform random stub averages 0 (3 SE), an
                           optional prior stub stays at chance (cc <= 0.05)
  SA-5 `univariate`        one content-free variable at a time, binned lookup stub  -> red if the
                           largest leave-one-out chance-corrected score reaches SHORTCUT_CC and the
                           NULL_Q quantile of that largest score under label shuffles inside
                           `strata`; the shuffled null is drawn only when the largest score reaches the
                           line. The permutation p (strata, Holm) is a profile reading, computed only
                           when `profile` is set
  SA-6 `surface_classifier` one GBM over the union of the content-free features, 6 group orders x 5
                           seeds, mean balanced accuracy -> red if its chance-corrected score reaches
                           SHORTCUT_CC and their NULL_Q quantile (permutations are drawn only when the
                           score reaches the line); the p is a profile reading, computed only with `profile`;
                           a feature declared decisive evidence leaves the union only if the gold
                           reads the record it counts (`decisive_evidence`), else SA-6 is red

A little shortcut is normal (real cases have them): a shortcut blocks a batch only if a stub that
reads no clinical evidence reaches cc = (BA - 1/K) / (1 - 1/K) >= SHORTCUT_CC; below that line it is
a report line. The line gates only targets the score reads (`PackAudit.gate_targets`); a generation
label such as a difficulty tier is reported as profile.
  SA-7 `realism`           registered predicates counted per item; every count must be 0
  SA-8 `sex_hits`          no patient sentence carries a word of the other sex (`events.sex_words`)
  SA-9 `repro` / marker    two builds byte-identical; a batch is scored only with its audit marker

Zero model calls. SA-5 and SA-6 need numpy / scikit-learn (the sklearn interpreter); the per-case
checks do not. SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Callable

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

SEED = 20261002
N_PERM_UNIVARIATE = 2000
N_PERM_SURFACE = 400
#: SA-5 noise floor: label shuffles of the largest per-variable score (enough for a stable 0.99 quantile)
N_NULL_UNIVARIATE = 1000
#: quantile of the shuffled null a shortcut must also reach (SA-5, SA-6): a pipeline run at any N and
#: seed must not fail valid packs by chance; a real generator shortcut sits far above the shuffled max
NULL_Q = 0.99
ALPHA = 0.05
#: a stub that reads no clinical evidence blocks a batch at this chance-corrected score
SHORTCUT_CC = 0.20


# ------------------------------------------------------------------------------ SA-1 + SA-8

@dataclass(frozen=True)
class Emission:
    """What SA-1 needs from a pack. Every callback sees the assembled case `raw` (gold included)."""
    gate: str                                       # gate name, prefix of every hit kind
    block: str                                      # adjudication / prediction_context key
    plan_of: Callable                               # raw -> plan dict | None (None: not this pack)
    recompute: Callable                             # (gold, raw) -> [mismatch]: gold re-derived from records
    class_of: Callable | None = None                # gold -> class
    planned_class: Callable | None = None           # plan -> class
    words: tuple[str, ...] = ()                     # gold words never shown (whole-token for ASCII)
    numbers: Callable | None = None                 # gold -> [number string never shown in the block]
    leak: Callable | None = None                    # visible block -> [detail] (a pack's own word rules)
    extra: Callable | None = None                   # (gold, raw) -> [(kind, detail)] pack callbacks
    visible: Callable | None = None                 # (gold, solver payload) -> [(kind, detail)] pack callbacks
    unrealizable: Callable | None = None            # gold -> reason | None


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def word_hits(text: str, words) -> list[str]:
    """Gold words in `text`: ASCII words as whole tokens, the others as substrings."""
    text = text or ""
    asc = [w for w in words if w.isascii()]
    out = []
    if asc:
        rx = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, asc)) + r")(?![A-Za-z0-9_])")
        out += [m.group(1) for m in rx.finditer(text)]
    out += [w for w in words if not w.isascii() and w in text]
    return out


def number_hits(block_text: str, numbers) -> list[str]:
    return [s for s in numbers
            if re.search(r"(?<![0-9.])" + re.escape(s) + r"(?![0-9])", block_text or "")]


def sex_hits(case) -> list[str]:
    """SA-8: patient-reported sentences carrying a word of the other sex (GEN19's anatomical words
    plus the grooming words of `events.sex_words`)."""
    from .events import sex_words
    sex = (_get(case, "user_profile") or {}).get("sex")
    words = sex_words(sex)
    out = []
    for e in _get(case, "evidence_ledger") or ():
        if e.get("source_type") == "patient_reported_symptom":
            got = [w for w in words if w in str(e.get("symptom") or "")]
            if got:
                out.append(f"sex={sex} {e.get('evidence_id')}: {got}")
    return out


def emission_hits(E: Emission, raw, sp) -> list[tuple[str, str]]:
    """SA-1 + SA-8 on one case as (kind, detail); [] for a case without the pack's plan."""
    plan = E.plan_of(raw)
    if plan is None:
        return []
    gold = (_get(raw, "adjudication") or {}).get(E.block) or {}
    why = E.unrealizable(gold) if E.unrealizable else None
    if why:
        return [("unrealizable", str(why))]
    out = [("gold_mismatch", d) for d in E.recompute(gold, raw)]
    if E.class_of and E.planned_class and E.class_of(gold) != E.planned_class(plan):
        out.append(("class_not_planned", f"{E.class_of(gold)} != planned {E.planned_class(plan)}"))
    out += list(E.extra(gold, raw) if E.extra else ())
    out += list(E.visible(gold, sp) if E.visible else ())
    vis = (_get(sp, "prediction_context") or {}).get(E.block)
    if not isinstance(vis, dict):
        out.append(("block_missing", f"solver payload has no {E.block} block"))
    elif vis != (_get(raw, "prediction_context") or {}).get(E.block):
        out.append(("block_missing", f"solver payload {E.block} block differs from the built one"))
    text = sp.dumps() if hasattr(sp, "dumps") else json.dumps(sp, ensure_ascii=False, default=str)
    btxt = json.dumps(vis, ensure_ascii=False)
    leaks = [f"word:{w}" for w in word_hits(text, E.words)]
    leaks += [f"num:{n}" for n in number_hits(btxt, E.numbers(gold) if E.numbers else ())]
    leaks += list(E.leak(vis) if (E.leak and isinstance(vis, dict)) else ())
    if leaks:
        out.append(("leak", ", ".join(sorted(set(leaks)))))
    out += [("sex", d) for d in sex_hits(raw)]
    return out


def emission_gate(E: Emission) -> Callable:
    """The registered emission gate of one pack (`external_gold.register_gate(E.gate, ...)`)."""
    def gate(raw, cs, sp, T) -> list[dict]:
        return [{"kind": f"{E.gate}_{k}", "severity": "gate", "detail": str(d)[:300]}
                for k, d in emission_hits(E, raw, sp)]
    gate.__qualname__ = f"emission_gate[{E.gate}]"
    # the external-gold manifest fingerprints a gate by its module's file: point it at the pack's
    # declaration, so editing a pack's callbacks moves that pack's `world_sha` (this module is in
    # each pack's world segment, `tools/make_freeze.py`)
    gate.__module__ = E.plan_of.__module__
    return gate


# ------------------------------------------------------------------------------ SA-2

def prompt_gold_hits(premise: dict, latent_keys, gold_meta) -> list[str]:
    """Plan or gold keys in a generation-model premise: a pack's latent key anywhere under `meta`,
    or one of the gold `meta` keys of the premise."""
    meta = (premise or {}).get("meta") or {}
    hits = [f"meta.{k}" for k in gold_meta if k in meta]
    blob = json.dumps(premise, ensure_ascii=False, default=str)
    hits += [f"latent:{k}" for k in latent_keys if f'"{k}"' in blob]
    return hits


# ------------------------------------------------------------------------------ SA-3

def composition(cells, expected: dict) -> dict:
    """Emitted cells equal the planned cells, exactly."""
    got = collections.Counter(cells)
    keys = sorted(set(got) | set(expected), key=str)
    bad = {str(k): [got.get(k, 0), expected.get(k, 0)] for k in keys if got.get(k, 0) != expected.get(k, 0)}
    return {"id": "SA-3", "cells": {str(k): got.get(k, 0) for k in keys}, "mismatch": bad, "pass": not bad}


# ------------------------------------------------------------------------------ SA-4

def stub_floor(gold: list, classes, cc: Callable, prior: list | None = None, n_random: int = 1000,
               seed: int = SEED) -> dict:
    """Constant stubs: cc exactly 0. Uniform random stub: mean cc within 3 SE of 0. Prior stub (a
    class list built without the item's content, e.g. from training-pack class frequencies or the
    registry's urgency): cc <= 0.05."""
    import numpy as np
    const = {str(c): cc(gold, [c] * len(gold)) for c in classes}
    rng = np.random.default_rng(seed)
    rs = [cc(gold, list(rng.choice(list(classes), size=len(gold)))) for _ in range(n_random)]
    mean, sd = float(np.mean(rs)), float(np.std(rs))
    out = {"id": "SA-4", "constant": {k: round(v, 6) for k, v in const.items()},
           "random_mean": round(mean, 4), "random_se": round(sd / n_random ** 0.5, 5)}
    ok = all(abs(v) < 1e-9 for v in const.values()) and abs(mean) <= 3 * sd / n_random ** 0.5
    if prior is not None:
        out["prior_cc"] = round(cc(gold, prior), 4)
        ok &= out["prior_cc"] <= 0.05
    out["pass"] = bool(ok)
    return out


# ------------------------------------------------------------------------------ SA-5

def chance_corrected(ba: float, k: int) -> float:
    """(BA - 1/K) / (1 - 1/K): 0 at chance, 1 at a perfect reader."""
    return (ba - 1.0 / k) / (1.0 - 1.0 / k)


def balanced_acc(y, p) -> float:
    cl = sorted(set(y), key=str)
    return sum(sum(1 for a, b in zip(y, p) if a == c and b == c) / sum(1 for a in y if a == c) for c in cl) / len(cl)


def _bins(vals, k: int = 5) -> list:
    """A variable as categories: itself with <= k levels, else its k quantile bins."""
    import numpy as np
    lv = sorted(set(vals))
    if len(lv) <= k:
        return list(vals)
    cuts = list(np.quantile(vals, [i / k for i in range(1, k)]))
    return [int(np.searchsorted(cuts, v, side="right")) for v in vals]


def _lookup_ba(cats, y) -> float:
    by = collections.defaultdict(collections.Counter)
    for c, g in zip(cats, y):
        by[c][g] += 1
    m = {c: cnt.most_common(1)[0][0] for c, cnt in by.items()}
    return balanced_acc(list(y), [m[c] for c in cats])


def _lookup_ba_loo(cats, y) -> float:
    """The lookup stub scored the way a reader meets an item it has not seen: each item is answered
    by the table built from the other items (its bin's most frequent class, else the overall one)."""
    by = collections.defaultdict(collections.Counter)
    for c, g in zip(cats, y):
        by[c][g] += 1
    tot = collections.Counter(y)

    def top(cnt):
        live = [(n, g) for g, n in cnt.items() if n > 0]
        return min(live, key=lambda x: (-x[0], str(x[1])))[1] if live else None
    pred = []
    for c, g in zip(cats, y):
        own = by[c].copy()
        own[g] -= 1
        rest = tot.copy()
        rest[g] -= 1
        pred.append(top(own) or top(rest))
    return balanced_acc(list(y), pred)


def _permute(y, strata, rng):
    import numpy as np
    if strata is None:
        return rng.permutation(y)
    yy = y.copy()
    for sv in sorted(set(strata.tolist()), key=str):
        ix = np.where(strata == sv)[0]
        yy[ix] = rng.permutation(y[ix])
    return yy


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def univariate(rows: list[dict], variables, targets=("class",), strata: str | None = None,
               n_perm: int = 0, seed: int = SEED, gate_targets=None,
               n_null: int = N_NULL_UNIVARIATE) -> dict:
    """SA-5: for every (variable, target) a stub that knows only that variable (each level or
    quantile bin answers its most frequent class), scored leave-one-out (`_lookup_ba_loo`: a table
    scored on the items it was built from overfits small batches). Red when the largest
    chance-corrected score over the gated variables reaches both SHORTCUT_CC and the NULL_Q quantile
    of the same largest score under `n_null` label shuffles inside `strata` (many variables on a
    small batch reach the line on noise alone). The permutation p inside `strata` (Holm over all)
    is a profile reading: `n_perm` > 0 computes it, 0 leaves it null. The shuffled null runs only when the
    largest score reaches the line (below it the batch passes either way). A row whose value is None is outside that variable (e.g. a vital where it
    is the focal one)."""
    import numpy as np
    res = {}
    for t in targets:
        for v in variables:
            sub = [r for r in rows if r[v] is not None]          # None: the variable does not apply
            y = np.asarray([r[t] for r in sub])
            st = np.asarray([r[strata] for r in sub]) if strata else None
            cats = _bins([r[v] for r in sub])
            if len(set(cats)) < 2 or len(set(y.tolist())) < 2:
                res[(v, t)] = (None, 1.0, None)
                continue
            obs = _lookup_ba(cats, list(y))
            k = len(set(y.tolist()))
            rng = np.random.default_rng(seed)
            ge = sum(_lookup_ba(cats, list(_permute(y, st, rng))) >= obs - 1e-12 for _ in range(n_perm))
            res[(v, t)] = (obs, (ge + 1) / (n_perm + 1), chance_corrected(_lookup_ba_loo(cats, list(y)), k))
    keys = list(res)
    # n_perm = 0: the permutation profile is not computed and p is null (the gate reads cc and the shuffled null)
    adj = holm([res[k][1] for k in keys]) if n_perm else [None] * len(keys)
    by = {f"{v}" if len(targets) == 1 else f"{v}->{t}":
          {"ba": None if res[(v, t)][0] is None else round(res[(v, t)][0], 4),
           "cc": None if res[(v, t)][2] is None else round(res[(v, t)][2], 4),
           "p": round(res[(v, t)][1], 4) if n_perm else None,
           "p_holm": None if a is None else round(a, 4)} for (v, t), a in zip(keys, adj)}
    gt = set(targets if gate_targets is None else gate_targets)
    tgt = {(f"{v}" if len(targets) == 1 else f"{v}->{t}"): t for (v, t) in keys}
    scored = [k for k in by if by[k]["cc"] is not None and tgt[k] in gt]
    worst = max(scored, key=lambda k: by[k]["cc"]) if scored else None
    max_cc = by[worst]["cc"] if worst else 0.0
    floor = (_null_max_q(rows, variables, [t for t in targets if t in gt], strata, n_null, seed)
             if scored and max_cc >= SHORTCUT_CC else None)
    return {"id": "SA-5", "n": len(rows), "n_perm": n_perm, "strata": strata, "by_variable": by,
            "worst": worst, "max_cc": max_cc, "line": SHORTCUT_CC, "n_null": n_null,
            "null_quantile": NULL_Q, "null_floor_max_cc": None if floor is None else round(floor, 4),
            "p_values": "computed" if n_perm else "not computed", "gate_targets": sorted(gt),
            "profile": {k: by[k]["cc"] for k in by if tgt[k] not in gt and by[k]["cc"] is not None},
            "report": [f"{k} cc {by[k]['cc']} p_holm {by[k]['p_holm']} (below the line, reported)"
                       for k in scored if by[k]["p_holm"] is not None and by[k]["p_holm"] < ALPHA
                       and by[k]["cc"] < SHORTCUT_CC],
            "pass": max_cc < SHORTCUT_CC or (floor is not None and max_cc < floor)}


def _null_max_q(rows, variables, targets, strata, n_null, seed) -> float:
    """NULL_Q quantile of the largest leave-one-out score over (variables x targets) when each
    target's labels are shuffled across rows inside `strata` (one shuffle per row set, so the
    variables keep their joint structure). 0.0 with n_null = 0 (the line alone gates)."""
    import numpy as np
    if not n_null:
        return 0.0
    st = np.asarray([r[strata] for r in rows]) if strata else None
    cols = []
    for t in targets:
        y = np.asarray([r[t] for r in rows])
        for v in variables:
            ix = [i for i, r in enumerate(rows) if r[v] is not None]
            cats = _bins([rows[i][v] for i in ix])
            if len(set(cats)) >= 2 and len(set(y[ix].tolist())) >= 2:
                cols.append((t, ix, cats, len(set(y[ix].tolist()))))
    if not cols:
        return 0.0
    rng = np.random.default_rng(seed + 1)
    ys = {t: np.asarray([r[t] for r in rows]) for t in targets}
    mx = []
    for _ in range(n_null):
        perm = {t: _permute(y, st, rng) for t, y in ys.items()}
        mx.append(max(chance_corrected(_lookup_ba_loo(cats, list(perm[t][ix])), k) for t, ix, cats, k in cols))
    return float(np.quantile(mx, NULL_Q))


# ------------------------------------------------------------------------------ SA-6

def surface_classifier(rows: list[dict], features, label: str = "class", groups: str = "spec",
                       strata: str | None = None, n_perm: int = N_PERM_SURFACE, jobs: int = 32,
                       seed: int = SEED, n_orders: int = 6, n_seeds: int = 5, profile: bool = False) -> dict:
    """SA-6: GBM (60 trees, depth 2) over the union of the declared content-free features;
    GroupKFold(5) under `n_orders` group orders x `n_seeds` seeds; the mean balanced accuracy against
    the same mean on `n_perm` label permutations inside `strata`. Red when the chance-corrected
    score of the mean reaches SHORTCUT_CC and the NULL_Q quantile of the permuted means. The permutations
    are drawn only when the score reaches the line (below it the batch passes either way) or `profile`
    asks for the p; the p and the 0.95 quantile are profile readings and stay null without `profile`."""
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X = np.asarray([[float(r[k]) for k in features] for r in rows], float)
    y = np.asarray([r[label] for r in rows])
    g = [r[groups] for r in rows]
    combos = []
    for o in range(n_orders):
        ug = sorted(set(g), key=str)
        np.random.default_rng(o).shuffle(ug)
        idx = {k: i for i, k in enumerate(ug)}
        splits = list(GroupKFold(5).split(X, y, np.asarray([idx[k] for k in g])))
        combos += [(sd, splits) for sd in range(n_seeds)]

    def mean_ba(yy):
        v = []
        for sd, splits in combos:
            pred = np.empty(len(yy), dtype=object)
            for tr, te in splits:
                pred[te] = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=sd).fit(
                    X[tr], yy[tr]).predict(X[te])
            v.append(balanced_acc(yy.tolist(), pred.tolist()))
        return float(np.mean(v)), v

    obs, per = mean_ba(y)
    cc = chance_corrected(obs, len(set(y.tolist())))
    rng = np.random.default_rng(seed)
    st = np.asarray([r[strata] for r in rows]) if strata else None
    perms = [_permute(y, st, rng) for _ in range(n_perm)] if profile or cc >= SHORTCUT_CC else []
    null = [m for m, _ in Parallel(n_jobs=jobs)(delayed(mean_ba)(pp) for pp in perms)] if perms else []
    floor = float(np.quantile(null, NULL_Q)) if null else None
    q95 = float(np.quantile(null, 0.95)) if null and profile else None
    p = (1 + sum(m >= obs for m in null)) / (n_perm + 1) if null and profile else None
    return {"id": "SA-6", "n": len(rows), "features": list(features), "strata": strata, "n_perm": n_perm,
            "mean_ba": round(obs, 4), "cc": round(cc, 4), "line": SHORTCUT_CC,
            "p_values": "computed" if profile and n_perm else "not computed",
            "perm_q95_mean_ba": None if q95 is None else round(q95, 4), "p": None if p is None else round(p, 4),
            "combos_over_q95": None if q95 is None else sum(x > q95 for x in per),
            "report": ([f"cc {cc:.4f} p {p:.4f} (below the line, reported)"]
                       if p is not None and p <= ALPHA and cc < SHORTCUT_CC else []),
            # where the permutation null was drawn, the line blocks only above its NULL_Q quantile (as SA-5)
            "perm_floor_mean_ba": None if floor is None else round(floor, 4),
            "pass": bool(cc < SHORTCUT_CC or (floor is not None and obs < floor))}


def decisive_evidence(items: list[dict], decisive: dict, regold: Callable | None) -> dict:
    """A feature the gold function reads is decisive evidence, not a surface cue: a pack may declare
    it (feature -> a perturbation of the gold block's record that the feature counts), and SA-6 then
    leaves it out of the union. The declaration stands only if the gold reads that record: the
    perturbation must change the gold's reading of its records (`regold`) on at least one item."""
    import copy
    out = {}
    for f, perturb in decisive.items():
        n = 0
        for it in items:
            g = it["gold"]
            try:
                n += regold(perturb(copy.deepcopy(g))) != regold(g) if regold else 0
            except Exception:                                    # noqa: BLE001  reading a broken record = reading it
                n += 1
        out[f] = n
    bad = sorted(f for f, n in out.items() if n == 0)
    return {"changed": out, "not_read_by_gold": bad, "pass": not bad}


# ------------------------------------------------------------------------------ SA-7

def said_sentences() -> set[str]:
    """Every sentence a patient may say: a benign-pool sentence, a hand-written lay sentence or a
    registered lookalike complaint."""
    from . import events as E
    from .registry import load_lookalikes
    E._sync_pools()
    return ({x["text"] for x in E.BENIGN_EVENTS} | {t for v in E.symptom_lay_every().values() for t in v}
            | {str(x["text"]) for v in load_lookalikes().values() for x in v})


def _patient_sentences(case) -> list[str]:
    return [str(e.get("symptom") or "") for e in _get(case, "evidence_ledger") or ()
            if e.get("source_type") == "patient_reported_symptom"]


def not_said_by_patient(said: set[str]) -> Callable:
    """Shared predicate: a patient-reported sentence that is not a registered patient sentence."""
    def pred(it) -> list[str]:
        return [t for t in _patient_sentences(it["case"]) if t not in said]
    return pred


def side_effect_complaint(it) -> list[str]:
    """Shared predicate: a benign complaint that is a known side effect of a drug the solver sees."""
    from . import events as E
    E._sync_pools()
    lay = E.symptom_lay_every()
    topic = {s: x["topic"] for x in E.BENIGN_EVENTS for s in [x["text"], *lay.get(x["text"], ())]}
    text = json.dumps(it["sp"], ensure_ascii=False, default=str).lower()
    drugs = [d for d in (E._cached_yaml(E._rp("drug_side_effects.yaml")) or {}).get("drugs") or ()
             if d.lower() in text]
    hit = E.side_effect_topics(drugs)
    return [t for t in _patient_sentences(it["case"]) if topic.get(t) in hit]


def realism(items: list[dict], predicates: dict[str, Callable]) -> dict:
    """SA-7: each predicate maps an item to its violations; every count must be 0."""
    counts, examples = {}, {}
    for name, fn in predicates.items():
        bad = [(it["case_id"], v) for it in items for v in fn(it)]
        counts[name] = len(bad)
        examples[name] = bad[:3]
    return {"id": "SA-7", "counts": counts, "examples": {k: v for k, v in examples.items() if v},
            "pass": not any(counts.values())}


def sex_consistency(items: list[dict]) -> dict:
    """SA-8 on an emitted batch (the build gate already refuses; this is the batch's own count)."""
    bad = [(it["case_id"], d) for it in items for d in sex_hits(it["case"])]
    return {"id": "SA-8", "n_hits": len(bad), "examples": bad[:3], "pass": not bad}


# ------------------------------------------------------------------------------ SA-9

def marker_name(pack: str) -> str:
    """The file `tools/pack_audit.py` writes into a batch whose audit passed."""
    return f"{pack.upper()}_AUDIT_OK"


def repro(b1: pathlib.Path, b2: pathlib.Path) -> dict:
    """Two builds of the same job are byte-identical (cases and payloads)."""
    def shas(p):
        return {f: hashlib.sha256((pathlib.Path(p) / f).read_bytes()).hexdigest()
                for f in ("cases.jsonl", "payloads.jsonl") if (pathlib.Path(p) / f).is_file()}
    a, b = shas(b1), shas(b2)
    return {"id": "SA-9", "files": sorted(a), "pass": bool(a) and a == b}


# ------------------------------------------------------------------------------ batch items

def load_items(batch: pathlib.Path, block: str, class_of: Callable) -> list[dict]:
    """Emitted items of a batch: the case (gold included), exactly what the solver saw, the gold
    block and its class, the condition spec as the group of the cross-validation."""
    cases, pays = {}, {}
    for line in open(pathlib.Path(batch) / "cases.jsonl", encoding="utf-8"):
        r = json.loads(line)
        cases[r["case_id"]] = r["case"]
    pp = pathlib.Path(batch) / "payloads.jsonl"
    if pp.is_file():
        for line in open(pp, encoding="utf-8"):
            r = json.loads(line)
            pays[r["case_id"]] = r["sp"]
    out = []
    for cid, c in sorted(cases.items()):
        adj = c.get("adjudication") or {}
        g = adj.get(block)
        if not isinstance(g, dict) or class_of(g) is None:
            continue
        out.append({"case_id": cid, "case": c, "sp": pays.get(cid), "gold": g, "class": class_of(g),
                    "spec": str((adj.get("ddx") or {}).get("spec_id") or cid)})
    return out


@dataclass
class PackAudit:
    """A pack's declaration for the batch audit (`tools/pack_audit.py`). Callbacks receive `ctx`, what
    `context(batch, job_path, ref)` returned ({} without one)."""
    name: str
    block: str
    classes: tuple
    class_of: Callable                                # gold -> class (None: not an item)
    rows: Callable                                    # (items, ctx) -> [row dict] (one per item)
    univariate: tuple[str, ...]                       # SA-5 variables (row keys)
    surface: tuple[str, ...]                          # SA-6 features (row keys)
    cc: Callable | None                               # chance-corrected agreement (the pack's scorer); None: no SA-4
    cell_of: Callable                                 # (item, ctx) -> composition cell (SA-3)
    planned_cells: Callable                           # (items, ctx) -> {cell: n} (SA-3)
    targets: tuple[str, ...] = ("class",)             # SA-5 targets (row keys)
    surface_targets: tuple[str, ...] = ("class",)     # SA-6 labels (row keys); Bonferroni over them
    strata: str | None = None                         # SA-5/SA-6 permutation strata (row key)
    select: Callable | None = None                    # items -> the published subset (else all)
    prior: Callable | None = None                     # (items, ctx) -> prior-stub classes (SA-4)
    realism: dict[str, Callable] = field(default_factory=dict)   # SA-7 pack predicates: item -> [violation]
    pack_gates: dict[str, Callable] = field(default_factory=dict)  # P-x: (pool, pool rows, ctx) -> dict
    decisive: dict[str, Callable] = field(default_factory=dict)  # SA-6 feature -> perturbation of the gold record it counts
    regold: Callable | None = None                    # gold block -> the gold's reading of its records (`decisive`)
    gate_targets: tuple[str, ...] | None = None       # SA-5/SA-6 targets the score reads (None: all); the rest are profile
    context: Callable | None = None                   # (batch, job_path, ref) -> ctx
    prepare: Callable | None = None                   # (items, ctx) -> items (attach what the pack reads)


def run_audit(A: PackAudit, batch: pathlib.Path, job_path=None, ref=None, *,
              profile: bool = False, jobs: int = 32) -> dict:
    """SA-3 ... SA-8 and the pack gates on one emitted batch. SA-3/SA-4/SA-7/SA-8 read the published
    subset; SA-5/SA-6 and the pack gates read every emitted item (the pool: three times the n of a
    50-item pack, same generator). The gates read the observed labels (chance-corrected score at the
    line, against the shuffled null); `profile` also computes the permutation p readings of SA-5 and SA-6."""
    ctx = A.context(batch, job_path, ref) if A.context else {}
    pool = load_items(batch, A.block, A.class_of)
    if A.prepare:
        pool = A.prepare(pool, ctx)
    pack = A.select(pool) if A.select else pool
    rows_pool = A.rows(pool, ctx)
    G = {}
    G["SA-3"] = composition([A.cell_of(it, ctx) for it in pack], A.planned_cells(pack, ctx))
    if A.cc is not None:
        G["SA-4"] = stub_floor([it["class"] for it in pack], A.classes, A.cc, A.prior(pack, ctx) if A.prior else None)
    G["SA-5"] = univariate(rows_pool, A.univariate, A.targets, A.strata, N_PERM_UNIVARIATE if profile else 0,
                      gate_targets=A.gate_targets)
    surface = tuple(f for f in A.surface if f not in A.decisive)
    per = {t: surface_classifier(rows_pool, surface, label=t, strata=A.strata, profile=profile, jobs=jobs)
           for t in A.surface_targets}
    G["SA-6"] = dict(next(iter(per.values()))) if len(per) == 1 else {"id": "SA-6", "by_target": per}
    gt = set(A.surface_targets if A.gate_targets is None else A.gate_targets)
    for t, r in per.items():
        r["profile"] = t not in gt
    G["SA-6"]["pass"] = all(r["pass"] for t, r in per.items() if t in gt)
    if A.decisive:
        G["SA-6"]["decisive"] = decisive_evidence(pool, A.decisive, A.regold)
        G["SA-6"]["pass"] &= G["SA-6"]["decisive"]["pass"]
    said = said_sentences()
    G["SA-7"] = realism(pack, {"not_said_by_patient": not_said_by_patient(said),
                               "side_effect_complaint": side_effect_complaint, **A.realism})
    G["SA-8"] = sex_consistency(pack)
    for name, fn in A.pack_gates.items():
        G[name] = fn(pool, rows_pool, ctx)
    return {"pack": A.name, "batch": str(batch), "profile": profile, "n_emitted": len(pool), "n_pack": len(pack),
            "pack_ids": sorted(it["case_id"] for it in pack), "gates": G,
            "pass": all(v.get("pass") for v in G.values())}
