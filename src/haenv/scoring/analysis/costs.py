"""Cost-efficiency and budget-ceiling calculations."""
from __future__ import annotations



def cost_efficiency_analysis(rows: list[dict], ranking: list[dict]) -> list[dict]:
    """Per-model token cost, latency and Pareto frontier against the composite score.

    `rows` are eval rows; `ranking` are composite-score rows. Each model is flagged
    for whether it is on the frontier (maximize score, minimize tokens/cost).
    """
    scores = {str(r.get("model") or r.get("solver") or ""): r.get("score") for r in ranking}
    by_solver: dict[str, list[dict]] = {}
    for r in rows:
        s = str(r.get("solver") or r.get("model") or "")
        if s:
            by_solver.setdefault(s, []).append(r)

    stats = []
    for s, rs in by_solver.items():
        score = scores.get(s)
        in_toks = []
        out_toks = []
        lats = []
        for r in rs:
            b = r.get("billed") or {}
            u = r.get("usage") or {}
            in_t = int(b.get("in_total") or u.get("prompt_tokens") or 0)
            out_t = int(b.get("out") or u.get("completion_tokens") or 0)
            if in_t > 0 or out_t > 0:
                in_toks.append(in_t)
                out_toks.append(out_t)
            lat = r.get("latency_s")
            if lat is not None and isinstance(lat, (int, float)):
                lats.append(float(lat))

        mean_in = (sum(in_toks) / len(in_toks)) if in_toks else 0.0
        mean_out = (sum(out_toks) / len(out_toks)) if out_toks else 0.0
        mean_tot = mean_in + mean_out
        mean_lat = (sum(lats) / len(lats)) if lats else None

        # cost per cell at a standardized $3/M input, $15/M output
        est_cost = (mean_in * 3.0 + mean_out * 15.0) / 1_000_000.0 if mean_tot > 0 else 0.0
        eff = ((float(score) / (mean_tot / 1000.0)) * 100.0) if (score is not None and mean_tot > 0) else None

        stats.append({
            "model": s,
            "score": score,
            "n_grids": len(rs),
            "n_usage": len(in_toks),
            "mean_in": round(mean_in, 1),
            "mean_out": round(mean_out, 1),
            "mean_total": round(mean_tot, 1),
            "mean_latency": round(mean_lat, 2) if mean_lat is not None else None,
            "cost_est": round(est_cost, 4),
            "efficiency": round(eff, 3) if eff is not None else None,
            "is_pareto": False,
        })

    valid = [st for st in stats if st["score"] is not None and st["mean_total"] > 0]
    for st in valid:
        dominated = False
        for other in valid:
            if other["model"] == st["model"]:
                continue
            if (float(other["score"]) >= float(st["score"]) and
                    other["mean_total"] <= st["mean_total"] and
                    (float(other["score"]) > float(st["score"]) or other["mean_total"] < st["mean_total"])):
                dominated = True
                break
        st["is_pareto"] = not dominated

    return sorted(stats, key=lambda x: (x["score"] is None, -(float(x["score"]) if x["score"] is not None else -1.0), x["mean_total"]))


def budget_ceiling_by_solver(rows: list[dict]) -> dict:
    """Per-model count of budget-limited cells, from `row["ceiling"]` (`ceil_by_finish` / `ceil_by_ratio`).

    Reasoning and answer share `max_tokens`, so truncation must be counted apart
    from capability failures. Only `measured` rows count; `absent:no_budget` goes to
    `n_no_budget` and rows without the key to `n_unmeasured`. A zero denominator
    gives `None`.
    """
    out: dict[str, dict] = {}
    for r in rows or []:
        m = str(r.get("solver") or "")
        if not m:
            continue
        s = out.setdefault(m, {"n_cells": 0, "n_measured": 0, "n_no_budget": 0,
                               "n_unmeasured": 0, "n_by_finish": 0, "n_by_ratio": 0,
                               "budgets": set()})
        s["n_cells"] += 1
        c = r.get("ceiling")
        if not isinstance(c, dict):
            s["n_unmeasured"] += 1
            continue
        if c.get("status") != "measured":
            s["n_no_budget"] += 1
            continue
        s["n_measured"] += 1
        b = c.get("budget")
        for x in (b if isinstance(b, list) else [b]):
            if x:
                s["budgets"].add(int(x))
        if c.get("by_finish"):
            s["n_by_finish"] += 1
        if c.get("by_ratio"):
            s["n_by_ratio"] += 1
    for s in out.values():
        s["budgets"] = sorted(s["budgets"])
        # Denominator is `n_measured`, not all cells.
        s["ceiling_rate"] = (round((s["n_by_finish"] + s["n_by_ratio"]
                                    - min(s["n_by_finish"], s["n_by_ratio"]))
                                   / s["n_measured"], 3) if s["n_measured"] else None)
    return out
