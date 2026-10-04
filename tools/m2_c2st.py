"""m2_c2st.py -- the sklearn half of `tools/m2_item_audit.py` (the haenv venv has no sklearn).

Reads a JSON job from stdin, writes a JSON result to stdout. Tasks:
  {"task": "guess", "X": [[...]], "y": [labels], "groups": [...], "seed": ...}
      grouped 5-fold GBM accuracy against the majority-class accuracy (P0-2a a, merged class).
  {"task": "guess_lines", "X": [[...]], "lines": [[line, ...]], "groups": [...], "seed": ...}
      the hidden line (P0-2a b): top-1 guess right when it is one of the case's lines, scored for the
      GBM and a lookup against the same folds' majority constant; optional "leak"/"perm_lines" arms
      for the power calibration (see guess_lines). Kept as the top-1 profile reading only.
  {"task": "guess_lines_topk", "X", "lines", "groups", "frames": [frame per row], "seed", ...}
      the P0-2a (b) gate (m2-fix-r2 PREREG C): `dx_listed`-style top-(n+1) score on the cases with
      a gold line, learner vs the same-fold frame-conditioned frequency list (see guess_lines_topk).

Run with an interpreter that has scikit-learn (`HAENV_SKLEARN_PY`), from a directory without a stray `inspect.py`.
"""
import json
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold


def guess(job):
    X = np.asarray(job["X"], float)
    labels = sorted(set(job["y"]))
    y = np.asarray([labels.index(v) for v in job["y"]], int)
    g = np.asarray(job["groups"])
    seed = int(job.get("seed", 20261001))
    pred = np.zeros(len(y), int)
    k = min(5, len(set(g)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        m = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    acc = float((pred == y).mean())
    maj = float(np.bincount(y).max() / len(y))
    return {"acc": acc, "majority": maj, "n": int(len(y)), "classes": labels}


def _inject_leak(X, L, labels, leak):
    """Power calibration (P0-2a b): append a multi-hot column block naming the case's own lines to a
    random `frac` of the cases (case-level draw, seed-fixed); the other cases get zeros."""
    rng = np.random.default_rng(int(leak.get("seed", 0)))
    n = len(L)
    hit = np.zeros(n, bool)
    hit[rng.permutation(n)[:int(round(float(leak["frac"]) * n))]] = True
    block = np.zeros((n, len(labels)))
    for i in np.flatnonzero(hit):
        for x in L[i]:
            block[i, labels.index(x)] = 1.0
    return np.hstack([X, block])


def guess_lines(job):
    """Hidden-line guess (P0-2a b), scored per fold against the SAME fold's baselines. Per fold, on
    GroupKFold(5) over the spec groups: (1) majority constant = the line most common in the training
    fold (case coverage); (2) lookup = most common training-fold line among cases with the same
    feature key (age band and sex dropped from the key; global most common when the key is unseen);
    (3) GBM trained on one row per (case, line). A case counts right when the guess is one of its
    lines. `learner` = the better of lookup and GBM (a weak GBM cannot let the gate off);
    `edge` = learner - same-fold majority constant. `job["leak"]` = {"frac", "seed"} injects the
    line-naming feature (see _inject_leak); `job["perm_lines"]` = seed shuffles the line sets (null)."""
    X = np.asarray(job["X"], float)
    L = [list(x) for x in job["lines"]]
    g = np.asarray(job["groups"])
    labels = sorted({x for ls in L for x in ls})
    seed = int(job.get("seed", 20261001))
    if job.get("perm_lines") is not None:    # null arm: line sets shuffled across cases
        L = [L[i] for i in np.random.default_rng(int(job["perm_lines"])).permutation(len(L))]
    if job.get("leak"):
        X = _inject_leak(X, L, labels, job["leak"])
    n = len(L)
    gbm = np.zeros(n, bool)
    maj = np.zeros(n, bool)
    look = np.zeros(n, bool)
    k = min(5, len(set(g)))
    for tr, te in GroupKFold(n_splits=k).split(X, np.zeros(n), g):
        cov = Counter(x for i in tr for x in L[i])
        top = max(cov, key=lambda x: (cov[x], x))
        tab, glob = {}, Counter()
        for i in tr:
            key = tuple(X[i, 2:].tolist())
            for x in L[i]:
                tab.setdefault(key, Counter())[x] += 1
                glob[x] += 1
        Xt = np.asarray([X[i] for i in tr for _ in L[i]])
        yt = np.asarray([labels.index(x) for i in tr for x in L[i]])
        if len(set(yt)) < 2:
            pred = np.full(len(te), yt[0])
        else:
            m = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=seed)
            m.fit(Xt, yt)
            pred = m.predict(X[te])
        for i, p in zip(te, pred):
            gbm[i] = labels[int(p)] in L[i]
            maj[i] = top in L[i]
            c = tab.get(tuple(X[i, 2:].tolist())) or glob
            look[i] = max(c, key=lambda x: (c[x], x)) in L[i]
    cover = {lab: sum(lab in ls for ls in L) / n for lab in labels}
    gtop = max(cover, key=lambda x: (cover[x], x))
    learner = max(float(gbm.mean()), float(look.mean()))
    return {"acc": float(gbm.mean()), "gbm_acc": float(gbm.mean()), "lookup_acc": float(look.mean()),
            "learner_acc": learner, "same_fold_majority": float(maj.mean()),
            "edge": learner - float(maj.mean()),
            "majority": float(cover[gtop]), "majority_line": gtop, "n": n, "n_lines": len(labels)}


def _ranked(cnt, fill=()):
    """Lines of `cnt` by count (desc, then name), then the lines of `fill` not yet listed, in order."""
    out = sorted(cnt, key=lambda x: (-cnt[x], x))
    seen = set(out)
    return out + [x for x in fill if x not in seen]


def _topk_score(ranked, gold):
    """`dx_listed` reading: |gold lines in the first n+1 of the ranking| / n, n = number of gold lines."""
    n = len(gold)
    return len(set(ranked[:n + 1]) & set(gold)) / n


def guess_lines_topk(job):
    """Hidden-line gate (P0-2a b, m2-fix-r2 PREREG C1-C2), scored like `dx_listed`. Only cases with
    at least one gold line (rows whose line set is empty or `["none"]` are dropped). Per case score =
    |gold lines in the learner's top-(n+1)| / n. Per GroupKFold fold (spec groups):
      frame list  = the training-fold lines of the case's frame ranked by case coverage (lines the frame
                    never shows follow in whole-fold coverage order; a frame with no training case uses
                    the whole-fold list) -- the comparison base;
      global list = the whole-fold list, frame ignored (reported only);
      GBM         = trained on one row per (case, line), ranked by predict_proba, ties and unseen lines
                    filled from the frame list;
      lookup      = training-fold line counts among cases with the same feature key (age band and sex
                    dropped from the key), filled from the frame list.
    `learner` = the better of GBM and lookup; `edge` = learner - frame list. `job["frames"]` = one frame
    per row (default F0). `leak` / `perm_lines` as in guess_lines, applied to the kept cases."""
    keep = [i for i, ls in enumerate(job["lines"]) if ls and list(ls) != ["none"]]
    frames_all = job.get("frames") or ["F0"] * len(job["lines"])
    if len(keep) < 2:
        return {"n": len(keep), "no_object": "fewer than 2 cases with a gold line", "edge": 0.0}
    X = np.asarray([job["X"][i] for i in keep], float)
    L = [list(job["lines"][i]) for i in keep]
    g = np.asarray([job["groups"][i] for i in keep])
    F = [frames_all[i] for i in keep]
    seed = int(job.get("seed", 20261001))
    if job.get("perm_lines") is not None:    # null arm: line sets shuffled across the kept cases
        L = [L[i] for i in np.random.default_rng(int(job["perm_lines"])).permutation(len(L))]
    labels = sorted({x for ls in L for x in ls})
    if job.get("leak"):
        X = _inject_leak(X, L, labels, job["leak"])
    n = len(L)
    s_frame, s_glob, s_gbm, s_look = (np.zeros(n) for _ in range(4))
    k = min(5, len(set(g)))
    for tr, te in GroupKFold(n_splits=k).split(X, np.zeros(n), g):
        glob = Counter(x for i in tr for x in L[i])
        glist = _ranked(glob)
        byf, tab = {}, {}
        for i in tr:
            fc = byf.setdefault(F[i], Counter())
            tc = tab.setdefault(tuple(X[i, 2:].tolist()), Counter())
            for x in L[i]:
                fc[x] += 1
                tc[x] += 1
        flist = {f: _ranked(c, glist) for f, c in byf.items()}
        Xt = np.asarray([X[i] for i in tr for _ in L[i]])
        yt = np.asarray([x for i in tr for x in L[i]])
        m = None
        if len(set(yt)) >= 2:
            m = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=seed)
            m.fit(Xt, yt)
            P = m.predict_proba(X[te])
        for j, i in enumerate(te):
            fl = flist.get(F[i], glist)
            if m is None:
                gr = _ranked(Counter({yt[0]: 1}), fl)
            else:
                pr = dict(zip(m.classes_, P[j]))
                gr = sorted(pr, key=lambda x: (-pr[x], fl.index(x) if x in fl else len(fl), x))
                gr += [x for x in fl if x not in pr]
            lk = tab.get(tuple(X[i, 2:].tolist()))
            s_frame[i] = _topk_score(fl, L[i])
            s_glob[i] = _topk_score(glist, L[i])
            s_gbm[i] = _topk_score(gr, L[i])
            s_look[i] = _topk_score(_ranked(lk, fl) if lk else fl, L[i])
    learner = max(float(s_gbm.mean()), float(s_look.mean()))
    return {"gbm": float(s_gbm.mean()), "lookup": float(s_look.mean()), "learner": learner,
            "frame_list": float(s_frame.mean()), "global_list": float(s_glob.mean()),
            "edge": learner - float(s_frame.mean()), "n": n, "n_lines": len(labels),
            "n_dropped_none": len(job["lines"]) - n, "frames": dict(Counter(F))}


if __name__ == "__main__":
    job = json.load(sys.stdin)
    out = (guess_lines(job) if job["task"] == "guess_lines"
           else guess_lines_topk(job) if job["task"] == "guess_lines_topk"
           else guess(job))
    json.dump(out, sys.stdout)
