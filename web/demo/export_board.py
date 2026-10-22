#!/usr/bin/env python
"""Write `board.json`, the board the demo page shows in its third act, from a board readout.

The public 1.3.1 export needs only released inputs:

    python web/demo/export_board.py --score docs/boards/1.3.1/k3.json \\
        --pk1 docs/boards/1.3.1/pk1.json --sample-board web/demo/board-1.3.0.json

Add `--check` to compare without writing. Act 2 retains only the historical snapshot's
sample tool traces, explicitly labelled with their version; its board readings are not reused.

For 1.3.0 use `--score <board_score.json> --figures <figures_data.json> --work <run-root>`
(repeat --work for separate pack runs). Scores, overall intervals, ranks, profiles and adjacent
paired differences are copied from the score readout; chart values are copied from the approved
figure data. Stored answer rows provide only decision counts and round disagreement.
Per-pack intervals and per-round S are absent in this readout and are not reconstructed.

The historical readout is the JSON `tools/evaluation/board_readout.py` writes: per pack, the chance-corrected score of
every model with its 95% case x round bootstrap interval, per-round scores, answer and format
counts and the top-six separability, and the equal-weight overall board with each model's
bootstrap rank interval. This script copies those readings as they are; it computes no score. It
adds what the readout does not carry, read from the board batches' `eval.jsonl`: how each model's
answers line up with the ground truth for the classification packs (answer x gold counts over every
item and round), the review decision of pack 1, and the round-to-round disagreement of each model.

    python web/demo/export_board.py --readout <readout.json> --results <results root> \\
        --batch m2=m2-pack1/<stamp> --batch pack2=pack2-triage/<stamp> --batch p3=p3-meds/<stamp> \\
        --batch p4=p4-followup/<stamp>

A pack run in two parts (the first items in one batch, the rest in a byte copy of it) is named
`--batch pack=<stampA>+<stampB>` with `--items <stampA>=<job.yaml> --items <stampB>=<job.yaml>`: each part
contributes only the items its job file lists.
    python web/demo/build.py

`board.json` is generated; do not hand-edit it.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "board.json"

#: pack key -> (job id, the answer and gold fields of a classification pack, its classes)
PACKS = {
    "m2": ("m2-pack1", None),
    "pack2": ("pack2-triage", ("pack2_disposition_said", "pack2_disposition_gold",
                               ["ed_now", "within_24h", "routine_followup", "watchful_waiting"])),
    "p3": ("p3-meds", ("p3_class_said", "p3_class_gold",
                       ["uptitrate", "downtitrate", "maintain", "switch", "check_adherence_or_adverse_effect"])),
    "p4": ("p4-followup", ("p4_class_said", "p4_class_gold",
                           ["true_change", "analytic_biological_noise", "preanalytical", "method_difference"])),
}
ROUNDS = (1, 2, 3)


sys.path.insert(0, str(ROOT))
from tools._bootstrap import bootstrap
bootstrap(__file__)
from tools.evaluation.rank_rule import unanswered_reason  # noqa: E402


class Missing(Exception):
    pass


def _parts(spec: str) -> list[str]:
    return spec.split("+")


def _round_parts(results: pathlib.Path, spec: str, i: int, items: dict, what: str):
    """One round of a pack: [(batch dir, case ids, eval rows)] per part, each limited to its items."""
    out = []
    for ref in _parts(spec):
        d = results / "joint_dx" / f"{ref}-pk{i}"
        try:
            cases = [json.loads(x)["case_id"] for x in (d / "cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            rows = [json.loads(x) for x in (d / "eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        except OSError as e:
            raise Missing(f"{what}: {e}")
        keep = items.get(ref)
        if keep is not None:
            cases = [c for c in cases if c in keep]
            rows = [r for r in rows if r["case"] in keep]
        out.append((d, cases, rows))
    return out


def _sha(p: pathlib.Path) -> str:
    """Short hash of the readout's values: the `inputs` paths are machine-specific and left out."""
    d = json.loads(p.read_text(encoding="utf-8"))
    d.pop("inputs", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _num(v, what: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
        raise Missing(f"{what}: not a number ({v!r})")
    return float(v)


def _board(rows: list[dict], what: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        ci = r.get("ci95") or []
        if len(ci) != 2:
            raise Missing(f"{what}/{r.get('model')}: no 95% interval")
        out.append({"model": r["model"], "rank": i + 1, "score": _num(r["score"], what),
                    "ci95": [_num(ci[0], what), _num(ci[1], what)], "se": _num(r["se"], what)})
    if [x["score"] for x in out] != sorted((x["score"] for x in out), reverse=True):
        raise Missing(f"{what}: the readout board is not ordered by score")
    return out


def _top6(t: dict, what: str) -> dict:
    pairs = [{"a": p["a"], "b": p["b"], "diff": _num(p["diff"], what), "se": _num(p["se"], what),
              "separable": bool(p["separable"])} for p in t["pairs"]]
    if len(pairs) != 15 or sum(p["separable"] for p in pairs) != t["n_separable"]:
        raise Missing(f"{what}: top-six pairs do not add up to n_separable")
    return {"top6": list(t.get("top6") or t["top"]), "pairs": pairs, "n_separable": t["n_separable"],
            "adjacent_separable": t["adjacent_separable"]}


def _pairs_all(t: dict, models: list[str], what: str) -> dict:
    """Every pair on the overall board, paired difference against 2 paired SE."""
    pairs = [{"a": p["a"], "b": p["b"], "diff": _num(p["diff"], what), "se": _num(p["se"], what),
              "separable": bool(p["separable"])} for p in t["pairs"]]
    n = len(models)
    if list(t["models"]) != models or len(pairs) != n * (n - 1) // 2:
        raise Missing(f"{what}: the pairs do not cover the board's {n} models in board order")
    if sum(p["separable"] for p in pairs) != t["n_separable"]:
        raise Missing(f"{what}: separable pairs do not add up to n_separable")
    if any(p["separable"] != (abs(p["diff"]) > 2 * p["se"]) for p in pairs):
        raise Missing(f"{what}: a pair's flag disagrees with |diff| > 2 SE")
    return {"models": models, "pairs": pairs, "n_separable": t["n_separable"]}


def _m2_review(results: pathlib.Path, ref: str, models: list[str], want: dict, items: dict) -> dict:
    """Pack 1 as it is scored, per model, from the board batches' rows (the last row per case and
    model of each round, with the semantic judge's view where the batch has one): the review
    decision (truth warranted or not x flagged or not) over the rows the review score reads, the
    share of gold lines listed and the tests' F1 and recall. The production scorer's composite on
    the same rows must equal the readout's pack-1 score."""
    sys.path.insert(0, str(ROOT))
    from haenv.m2 import score as S
    from haenv.scoring.semantic.report import view_for_batch
    from haenv.scoring.semantic.rubric import load_policy
    rows = []
    for i in ROUNDS:
        for d, _cases, got in _round_parts(results, ref, i, items, "decisions/m2"):
            last = list({(r["case"], r["solver"]): r for r in got if r.get("solver") in models}.values())
            sd = d / "semantic" / load_policy()["version"]
            rows += view_for_batch(last, d, sd) if (sd / "manifest.json").is_file() else last
    board = S.board(rows)
    out = {}
    for m in models:
        rs = [r for r in rows if r["solver"] == m]
        got = board[m]["composite_raw"]
        if got is None or abs(got - want[m]) > 1e-9:
            raise Missing(f"decisions/m2/{m}: scorer gives {got}, the readout {want[m]}: not the same rows")
        sub = [r for r in rs if r.get("m2_frame") in ("F1", "F2", "F3")] or rs
        cells = collections.Counter()
        for r in sub:
            w = r.get("review_warranted")
            if S._aborted(r) and w is not None:
                cells[(bool(w), "none")] += 1
            elif isinstance(r.get("review_flag_ok"), (int, float)):
                said = bool(w) if r["review_flag_ok"] else not bool(w)
                cells[(bool(w), "flagged" if said else "not_flagged")] += 1
        rec = [0.0 if S._aborted(r) else float(r["tests_recall"]) for r in rs
               if not S._is_insufficient(r) and (S._aborted(r) or isinstance(r.get("tests_recall"), (int, float)))]
        out[m] = {"review": [[w, s_, n] for (w, s_), n in sorted(cells.items())],
                  "dx_listed": board[m]["dx_listed"], "tests_f1": board[m]["tests_recall_f1"],
                  "tests_recall": statistics.fmean(rec) if rec else None, "review_cc": board[m]["review_utility_cc"]}
    n_items = len({r["case"] for r in rows})
    return {"kind": "review", "n_items": n_items, "n_rounds": len(ROUNDS), "models": out}


def _pack(p: dict, key: str) -> dict:
    models = list(p["ranked"])
    per_round = {m: [_num(p["per_round"][str(i)][m], f"{key}/per_round") for i in ROUNDS] for m in models}
    return {"status": "final", "n_cases": p["n_cases"], "n_rounds": len(ROUNDS),
            "board": _board(p["board_cc"], key), "per_round": per_round,
            "answered": {m: [p["answer"][m]["answered"], p["answer"][m]["of"]] for m in models},
            "format_fail": {m: int(p["format_fail"][m]) for m in models},
            "unranked": list(p.get("unranked") or []), "top6": _top6(p["top6"], key)}


def _stability(results: pathlib.Path, ref: str, key: str, models: list[str], items: dict) -> dict:
    """Share of a model's items whose three rounds disagree on the pack's decision fields (items
    answered in all three rounds). It describes the model and enters no score."""
    said = PACKS[key][1][0] if PACKS[key][1] else None
    by = {}
    for i in ROUNDS:
        for _d, _cases, rows in _round_parts(results, ref, i, items, f"stability/{key}"):
            for r in rows:
                by[(r["case"], r["solver"], i)] = r
    cases = sorted({c for c, _m, _i in by})
    out = {}
    for m in models:
        diff = n = 0
        for c in cases:
            rs = [by.get((c, m, i)) for i in ROUNDS]
            if any(r is None or unanswered_reason(r) is not None for r in rs):
                continue
            if said:
                sig = [r.get(said) for r in rs]
            else:
                sig = [(r.get("dx_rank"), r.get("m2_urgency_said"), r.get("m2_med_said"),
                        r.get("m2_rcv_said"), r.get("review_declared")) for r in rs]
            n += 1
            diff += len({json.dumps(x, ensure_ascii=False, sort_keys=True, default=str) for x in sig}) > 1
        out[m] = round(diff / n, 4) if n else 0.0
    return out


def _decisions(results: pathlib.Path, ref: str, key: str, models: list[str], items: dict) -> dict:
    """Answer x gold counts per model over every item and round: the last row per (case, model)
    of each round's batch, as the board reads them. A missing answer is counted as `none`."""
    said_f, gold_f, classes = PACKS[key][1]
    gold, cells, n_items = {}, {m: collections.Counter() for m in models}, None
    for i in ROUNDS:
        cases, rows = [], []
        for _d, c, r in _round_parts(results, ref, i, items, f"decisions/{key}"):
            cases += c
            rows += r
        answered = {r["case"] for r in rows}      # the batch can hold more cases than the board ran
        cases = [c for c in cases if c in answered]
        n_items = n_items or len(cases)
        last = {(r["case"], r["solver"]): r for r in rows if r.get("solver") in models}
        for (c, _m), r in last.items():
            if r.get(gold_f) is not None:
                gold[c] = r[gold_f]
        for c in cases:
            if c not in gold:
                raise Missing(f"decisions/{key}: no gold class for {c}")
            for m in models:
                r = last.get((c, m))
                s = r.get(said_f) if r else None
                cells[m][(gold[c], s if s in classes else "none")] += 1
    counts = collections.Counter(gold.values())
    return {"classes": classes, "n_items": n_items, "n_rounds": len(ROUNDS),
            "gold_counts": {k: counts.get(k, 0) for k in classes},
            "models": {m: [[g, s, n] for (g, s), n in sorted(cells[m].items())] for m in models}}


def _overall(ov: dict, models_ranked: list[str]) -> dict:
    """The equal-weight overall board with each model's bootstrap rank interval. Models whose rank
    intervals overlap are not separable: the page shows the interval, not a definite order."""
    eq = ov["equal"]
    board = _board(eq["board"], "overall")
    ranks = {r["model"]: r for r in ov["rank_table"]}
    for row in board:
        lo, hi = ranks[row["model"]]["rank_ci95_equal"]
        row["rank_ci95"] = [int(lo), int(hi)]
    models = [r["model"] for r in board]
    out = {"status": "final", "packs": list(PACKS), "rule": "mean of the four chance-corrected pack scores, "
           "equal weight; independent per-pack bootstraps combined draw by draw",
           "board": board, "top6": _top6(eq["top6"], "overall")}
    out["pairs"] = _pairs_all({"models": models, "pairs": eq["top6"]["all_pairs"],
                               "n_separable": sum(bool(p["separable"]) for p in eq["top6"]["all_pairs"])},
                              models, "overall/pairs")
    return out


def build(readout: pathlib.Path, results: pathlib.Path | None, batches: dict[str, str],
          items: dict | None = None) -> dict:
    R = json.loads(readout.read_text(encoding="utf-8"))
    items = items or {}
    packs = {}
    for key, (job, _cls) in PACKS.items():
        if key not in R["packs"]:
            raise Missing(f"the readout has no pack `{key}`")
        packs[key] = {"job": job, **_pack(R["packs"][key], key)}
    overall = _overall(R["overall"], list(R["overall"]["models"]))
    models = [r["model"] for r in overall["board"]]
    decisions, seed = {}, None
    if results is not None:
        # the pack seed: every board batch must carry the same hash (the seed itself never ships)
        seeds = {}
        for key, spec in batches.items():
            for ref in _parts(spec):
                try:
                    meta = json.loads((results / "joint_dx" / f"{ref}-pk1" / "batch.json").read_text(encoding="utf-8"))
                except OSError as e:
                    raise Missing(f"seed/{key}: {e}")
                seeds[f"{key}/{ref}"] = ((meta.get("pack") or {}).get("seed_sha256"))
        if len(set(seeds.values())) != 1 or None in seeds.values():
            raise Missing(f"the board batches do not share one pack seed: {seeds}")
        seed = next(iter(seeds.values()))
        for key in PACKS:
            if key not in batches:
                raise Missing(f"no --batch for `{key}`")
            packs[key]["stability"] = _stability(results, batches[key], key, list(R["packs"][key]["ranked"]), items)
            if PACKS[key][1] is None:
                want = {r["model"]: r["score"] for r in R["packs"][key]["board_raw"]}
                decisions[key] = _m2_review(results, batches[key], models, want, items)
                continue
            decisions[key] = _decisions(results, batches[key], key, models, items)
    return {"_generated": "do not hand-edit; rebuild with `python web/demo/export_board.py` (see its docstring)",
            "source": {"boot": R["boot"], "mode": R["mode"], "seeds": R["seeds"], "pack_seed_sha256": seed},
            # Where the readings come from; batch stamps live only here.
            "provenance": {"readout_sha16": _sha(readout), "batches": dict(sorted(batches.items()))},
            "models": models, "packs": packs, "overall": overall, "decisions": decisions}


# The preregistered board carries scores, overall CIs and adjacent differences only.
# Exporting it must never invoke a scorer or manufacture per-pack intervals.
SCORE_PACKS = {"m2": "m2", "pack2": "p2", "p3": "p3", "p4": "p4"}
TRAJECTORY_CASE = "PT-91fbb6a5120fd049a7d7"


def build_released(score: pathlib.Path, pk1: pathlib.Path, sample_board: pathlib.Path) -> dict:
    """Export the public 1.3.1 readouts without inventing unpublished descriptors."""
    R = json.loads(score.read_text(encoding="utf-8"))
    first = json.loads(pk1.read_text(encoding="utf-8"))
    if R["aggregation"] != "k3" or first["aggregation"] != "pk1":
        raise Missing("released board needs k3 and pk1 readouts")
    ranked = sorted((m for m, r in R["overall"].items() if r["score"] is not None),
                    key=lambda m: R["overall"][m]["rank"])
    first_order = sorted((m for m, r in first["overall"].items() if r["score"] is not None),
                         key=lambda m: first["overall"][m]["rank"])
    if not ranked or set(ranked) != set(first_order) or set(R["packs"]) != set(SCORE_PACKS.values()):
        raise Missing("released readouts disagree on model or pack coverage")
    overall, packs = [], {}
    for m in ranked:
        r = R["overall"][m]
        lo, hi = r["ci"]
        rlo, rhi = r["rank_ci"]
        if not (0 <= lo <= r["score"] <= hi <= 1 and 1 <= rlo <= r["rank"] <= rhi <= len(ranked)):
            raise Missing(f"overall/{m}: invalid score or rank interval")
        overall.append({"model": m, "score": r["score"], "ci95": r["ci"], "rank": r["rank"],
                        "rank_ci95": [int(rlo), int(rhi)],
                        "pass3": math.fsum(R["profile"][p][m]["pass_k"] for p in SCORE_PACKS.values()) / 4})
    if [r["rank"] for r in overall] != list(range(1, len(ranked) + 1)):
        raise Missing("overall: incomplete rank order")
    for key, pack in SCORE_PACKS.items():
        prof = R["profile"][pack]
        counts = {prof[m]["n_pk1"] for m in ranked}
        if len(counts) != 1 or any(prof[m]["k"] != 3 for m in ranked):
            raise Missing(f"profiles/{pack}: inconsistent item or answer counts")
        rows = [{"model": m, "score": R["overall"][m]["packs"][pack]} for m in ranked]
        if any(not 0 <= r["score"] <= 1 for r in rows):
            raise Missing(f"packs/{pack}: invalid score")
        packs[key] = {"status": "final", "n_cases": counts.pop(), "n_rounds": 3,
                      "board": rows, "profile": prof}
    pairs = R["adjacent_pairs"]
    if ([p["pair"] for p in pairs] != [a + " - " + b for a, b in zip(ranked, ranked[1:])]
            or any(p["separable"] != (p["ci"][0] > 0 or p["ci"][1] < 0) for p in pairs)):
        raise Missing("overall: adjacent pairs disagree with the board order or intervals")
    draws = {R["overall"][m]["rank_ci_draws"] for m in ranked}
    if len(draws) != 1:
        raise Missing("overall: inconsistent bootstrap draw counts")
    sample = json.loads(sample_board.read_text(encoding="utf-8"))
    if sample["version"] != "1.3.0":
        raise Missing("sample traces require the published 1.3.0 snapshot")
    return {"_generated": "do not hand-edit; rebuild with `python web/demo/export_board.py "
            "--score docs/boards/1.3.1/k3.json --pk1 docs/boards/1.3.1/pk1.json "
            "--sample-board web/demo/board-1.3.0.json`",
            "version": "1.3.1", "source": {"boot": draws.pop(), "answer": "mean_of_three",
                                          "aggregation": "k3", "seed": "public"},
            "provenance": {"score_sha16": _sha(score), "pk1_sha16": _sha(pk1),
                           "sample_board_sha16": _sha(sample_board)},
            "models": ranked, "packs": packs,
            "overall": {"status": "final", "packs": list(SCORE_PACKS), "board": overall,
                        "adjacent_pairs": pairs},
            "first_answer": {"aggregation": "pk1", "models": first_order},
            "unranked": {m: r for m, r in R["overall"].items() if r["score"] is None},
            "profile": R["profile"], "decisions": {},
            "trajectory": {**sample["trajectory"], "version": sample["version"]}}


def _profile_rows(work: list[pathlib.Path], pack: str) -> dict:
    """Last stored answer per cell, including recorded replacements of transport aborts.

    These rows supply decision counts and disagreement only, never board scores.
    """
    cells = {}
    for root in work:
        for f in sorted(root.glob(f"*/{pack}/results/joint_dx/*/*-pk[123]/eval.jsonl")):
            k = int(f.parent.name.rsplit("-pk", 1)[1])
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    cells[(r["solver"], r["case"], k)] = r
            rep = f.parent.with_name(f.parent.name.rsplit("-pk", 1)[0] + f"-rep{k}") / "eval.jsonl"
            if rep.is_file():
                for ln in rep.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln); key = (r["solver"], r["case"], k)
                    old = cells.get(key)
                    if old and str(old.get("overall", "")).startswith("ABORT(no_response)") and not str(r.get("overall", "")).startswith("ABORT"):
                        cells[key] = r
    if not cells:
        raise Missing(f"profiles/{pack}: no stored rows")
    return cells


def _descriptors(cells: dict, pack: str, models: list[str], profiles: dict) -> tuple[dict, dict]:
    """Descriptive counts from the answers; no score, CI, rank or rubric calculation."""
    key = "pack2" if pack == "p2" else pack
    said_field = {"p2": "p2_said", "p3": "p3_class_said", "p4": "p4_class_said"}.get(pack)
    gold_field = {"p2": "p2_floor", "p3": "p3_class_gold", "p4": "p4_class_gold"}.get(pack)
    cases = sorted({c for m, c, k in cells if m in models})
    if not cases:
        raise Missing(f"profiles/{pack}: no items")
    # Transport/parse aborts can lack gold fields. Join the same item's stored
    # code truth across rows; contradictory or absent truth must stop export.
    truth = {}
    if gold_field:
        for (_, c, _), r in cells.items():
            gold = r.get(gold_field)
            if gold is not None:
                if c in truth and truth[c] != gold:
                    raise Missing(f"decisions/{pack}/{c}: conflicting truth")
                truth[c] = gold
        if set(cases) - truth.keys():
            raise Missing(f"decisions/{pack}: missing item truth")
    stability, counts = {}, {}
    for m in models:
        missing = [(c, k) for c in cases for k in ROUNDS if (m, c, k) not in cells]
        if missing:
            raise Missing(f"profiles/{pack}/{m}: missing {len(missing)} cells")
        if len(cases) != profiles[m]["n_pk1"] or profiles[m]["k"] != len(ROUNDS):
            raise Missing(f"profiles/{pack}/{m}: counts disagree with the score readout")
        signatures = []
        counter = collections.Counter()
        for c in cases:
            sig = []
            for k in ROUNDS:
                r = cells[(m, c, k)]
                answered = not str(r.get("overall", "")).startswith("ABORT")
                if said_field:
                    said = r.get(said_field) if answered else None
                    if answered and r.get(gold_field) is None:
                        raise Missing(f"decisions/{pack}/{c}: missing truth")
                    counter[(truth[c], said if said in PACKS[key][1][2] else "none")] += 1
                    sig.append(said)
                else:
                    w = r.get("review_warranted")
                    if w is not None:
                        flag = r.get("review_declared") if answered else None
                        counter[(bool(w), "none" if flag is None else "flagged" if flag else "not_flagged")] += 1
                    sig.append([r.get(f) for f in ("dx_rank", "m2_urgency_said", "m2_med_said", "m2_rcv_said", "review_declared")] if answered else None)
            signatures.append(len({json.dumps(x, sort_keys=True) for x in sig}) > 1)
        stability[m] = sum(signatures) / len(signatures)
        counts[m] = [[g, a, n] for (g, a), n in sorted(counter.items())]
    if said_field:
        return stability, {"classes": PACKS[key][1][2], "n_items": len(cases), "n_rounds": len(ROUNDS), "models": counts}
    return stability, {"kind": "review", "n_items": len(cases), "n_rounds": len(ROUNDS),
                       "models": {m: {"review": counts[m]} for m in models}}


def build_score(score: pathlib.Path, figures: pathlib.Path, work: list[pathlib.Path]) -> dict:
    """Copy the preregistered readings and approved figure values, verbatim."""
    R = json.loads(score.read_text(encoding="utf-8"))
    F = json.loads(figures.read_text(encoding="utf-8"))
    if F["tag"] is not None or R != F["board_score"]:
        raise Missing("figure values are draft or disagree with the score readout")
    ranked = sorted((m for m, r in R["overall"].items() if r["score"] is not None), key=lambda m: R["overall"][m]["rank"])
    if not ranked:
        raise Missing("no ranked models")
    overall = []
    for m in ranked:
        r = R["overall"][m]
        if set(r["packs"]) != set(SCORE_PACKS.values()) or len(r["ci"]) != 2 or len(r["rank_ci"]) != 2:
            raise Missing(f"overall/{m}: incomplete reading")
        prof = [R["profile"][pk][m] for pk in SCORE_PACKS.values()]
        if any(x["k"] != 3 for x in prof):
            raise Missing(f"overall/{m}: pass^3 needs three answers per item")
        overall.append({"model": m, "score": r["score"], "ci95": r["ci"], "rank": r["rank"],
                        "rank_ci95": [int(x) for x in r["rank_ci"]],
                        # reliability profile, not a ranking input: equal-weight mean of the packs' pass^3
                        "pass3": sum(x["pass_k"] for x in prof) / len(prof)})
    packs, decisions = {}, {}
    for key, pack in SCORE_PACKS.items():
        prof = R["profile"][pack]
        stability, decision = _descriptors(_profile_rows(work, pack), pack, ranked, prof)
        decisions[key] = decision
        packs[key] = {"status": "final", "n_cases": decision["n_items"], "n_rounds": decision["n_rounds"],
                      "board": [{"model": m, "score": R["overall"][m]["packs"][pack]} for m in ranked],
                      "profile": prof, "stability": stability}
    for kind in ("dq", "cost"):
        if set(F["drawn"][kind]) != set(ranked):
            raise Missing(f"figures/{kind}: model coverage differs")
    traj = F["trajectory"]
    for m, t in traj.items():
        if any(n < 1 or n > len(t["calls"]) for n in t["first_seen"].values()):
            raise Missing(f"trajectory/{m}: evidence outside the call sequence")
    draws = {R["overall"][m]["rank_ci_draws"] for m in ranked}
    if len(draws) != 1:
        raise Missing("overall: inconsistent bootstrap draw counts")
    return {"_generated": "do not hand-edit; rebuild with `python web/demo/export_board.py --score ... --figures ... --work ...`",
            "version": "1.3.0", "source": {"boot": next(iter(draws)), "answer": "mean_of_three" if R["aggregation"] == "k3" else "first",
                                               "aggregation": R["aggregation"], "seed": "public"},
            "provenance": {"score_sha16": _sha(score), "figures_sha16": _sha(figures)},
            "models": ranked, "packs": packs,
            "overall": {"status": "final", "packs": list(SCORE_PACKS), "board": overall,
                        "adjacent_pairs": R["adjacent_pairs"]},
            "unranked": {m: r for m, r in R["overall"].items() if r["score"] is None},
            "profile": R["profile"], "decisions": decisions,
            "history": F["history"], "drawn": F["drawn"],
            "spend": {m: r for m, r in F["spend"].items() if m != "_source"}, "cells": F["cells"],
            "trajectory": {"pack": "p4", "case": TRAJECTORY_CASE, "answer": 1, "models": traj}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--readout", type=pathlib.Path, help="historical chance-corrected board")
    source.add_argument("--score", type=pathlib.Path, help="preregistered board_score.json; copied, never recalculated")
    ap.add_argument("--figures", type=pathlib.Path, help="approved figures_data.json")
    ap.add_argument("--pk1", type=pathlib.Path, help="released first-answer readout (public 1.3.1 mode)")
    ap.add_argument("--sample-board", type=pathlib.Path, help="published 1.3.0 snapshot for Act 2's sample traces")
    ap.add_argument("--work", type=pathlib.Path, action="append", default=[], help="board run root, repeatable; descriptive answer counts only")
    ap.add_argument("--results", type=pathlib.Path, help="results root holding the board batches")
    ap.add_argument("--batch", action="append", default=[], metavar="PACK=JOB/STAMP",
                    help="board batch of a classification pack (rounds are <STAMP>-pk1..3)")
    ap.add_argument("--items", action="append", default=[], metavar="STAMP=JOB.yaml",
                    help="a part of a pack run in two parts contributes only the items this job file lists")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="compare with the generated JSON without writing it")
    a = ap.parse_args()
    try:
        batches = dict(b.split("=", 1) for b in a.batch)
        import yaml
        items = {k: {c["case_id"] for c in yaml.safe_load(pathlib.Path(v).read_text(encoding="utf-8"))["cases"]}
                 for k, v in (x.split("=", 1) for x in a.items)}
        if a.score:
            if a.pk1 and a.sample_board and not a.figures and not a.work:
                out = build_released(a.score, a.pk1, a.sample_board)
            elif not a.figures or not a.work:
                raise Missing("--score needs --figures and at least one --work")
            else:
                out = build_score(a.score, a.figures, a.work)
        else:
            out = build(a.readout, a.results, batches, items)
    except (Missing, KeyError, ValueError) as e:
        print(f"export stopped: {e}")
        return 2
    rendered = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    if a.check:
        if not a.out.is_file() or a.out.read_text(encoding="utf-8") != rendered:
            print(f"stale board: {a.out.name}")
            return 1
        print(f"ok: {a.out.name} matches its released inputs ({len(out['models'])} models)")
        return 0
    a.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {a.out.name}: packs {sorted(out['packs'])} · decisions {sorted(out['decisions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
