"""Per-item discrimination between model and baseline panels."""
from __future__ import annotations

from haenv.scoring.analysis.precision import _fsum_mean
from haenv.scoring.analysis.tracks import BLIND_BASELINES
from haenv.scoring.analysis.tracks import HEURISTIC_BASELINES


def item_discrimination(rows: list[dict], key: str = "dx_hit",
                        baseline_solvers: tuple = BLIND_BASELINES + HEURISTIC_BASELINES,
                        ) -> list[dict]:
    """Per-question discrimination: does this question measure capability.

    - `answer_rate`: correct rate among non-baseline solvers (outside [0.3, 0.7]
      carries little information);
    - `baseline_rate`: correct rate across all baselines;
    - `discrimination`: the difference;
    - `disc_vs_blind` (vs. answer-blind; decides `low_information`) and
      `disc_vs_heuristic` (vs. domain heuristics; headroom only, never
      disqualifies).

    `low_information` questions stay listed in the report with their reason.
    """
    by_case: dict[str, dict] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        c = by_case.setdefault(r.get("case"), {"case": r.get("case"), "model": [],
                                              "base": [], "blind": [], "heur": []})
        s, hit = r.get("solver"), (1.0 if v else 0.0)
        if s in baseline_solvers:
            c["base"].append(hit)
            if s in BLIND_BASELINES:
                c["blind"].append(hit)
            elif s in HEURISTIC_BASELINES:
                c["heur"].append(hit)
        else:
            c["model"].append(hit)
    out = []

    def _rate(xs: list[float]) -> float | None:
        return _fsum_mean(xs)

    for c in by_case.values():
        m, b = c["model"], c["base"]
        ar, br = _rate(m), _rate(b)
        blind_r, heur_r = _rate(c["blind"]), _rate(c["heur"])
        disc = round(ar - br, 3) if (ar is not None and br is not None) else None
        d_blind = round(ar - blind_r, 3) if (ar is not None and blind_r is not None) else None
        d_heur = round(ar - heur_r, 3) if (ar is not None and heur_r is not None) else None
        # `no_model_spread` (all models agree) and `low_information` (no gain over
        # baseline) are separate: all models right and baseline wrong is still
        # informative. `no_model_spread` needs at least two models.
        if len(m) < 2:
            no_spread = None
        else:
            spread = not (all(x == 1.0 for x in m) or all(x == 0.0 for x in m))
            no_spread = not spread
        # `low_information` uses `d_blind` (falls back to `disc` when no answer-blind stub
        # ran); beating a heuristic is never required.
        _d = d_blind if d_blind is not None else disc
        low = (no_spread is not False) and (_d is None or _d <= 0)
        out.append({"case": c["case"], "n_model": len(m), "n_base": len(b),
                    "answer_rate": ar, "baseline_rate": br, "discrimination": disc,
                    "n_blind": len(c["blind"]), "n_heuristic": len(c["heur"]),
                    "blind_rate": blind_r, "heuristic_rate": heur_r,
                    "disc_vs_blind": d_blind, "disc_vs_heuristic": d_heur,
                    "low_information_basis": ("blind" if d_blind is not None else "mixed"),
                    "no_model_spread": no_spread, "low_information": low,
                    "why": ("ceiling: every model correct and no better than the baseline" if low and m and m[0] == 1.0 else
                            "floor: every model wrong" if low else
                            "no difference between models, but better than the baseline (still informative)" if no_spread else
                            "single model: cannot rank models against each other (needs >= 2 models)" if no_spread is None else "")})
    return sorted(out, key=lambda x: (x["low_information"], x["case"]))
