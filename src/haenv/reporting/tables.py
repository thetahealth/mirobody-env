"""Tables for the evaluation report."""
from __future__ import annotations

from haenv.reporting.formatting import (
    _fmt,
)


def render_budget_ceiling(rel_ceiling: dict, order: list[str]) -> list[str]:
    """The budget-limited subsection within §1r. `rel_ceiling` is the
    output of `budget_ceiling_by_solver`."""
    ms = [m for m in order if m in rel_ceiling] or sorted(rel_ceiling)
    real = [m for m in ms if rel_ceiling[m]["n_measured"]]
    if not real:
        _tot = sum(rel_ceiling[m]["n_cells"] for m in ms)
        return [f"> **Budget-limited: not measured.** Not one of the {_tot} cells in this batch carries a budget"
                f"(offline stubs / baselines have no `max_tokens`) -- **do not read this as \"never hit the ceiling\"**.\n", ""]
    L = ["> **Budget-limited** (the `ceiling` block): "
         "reasoning tokens and answer tokens share the same `max_tokens`, "
         "so a silently-truncated cell looks in the artifact exactly like \"this one just answered poorly\" -- the same shape as \"not capable\". "
         "**This column separates the two**, and likewise does not count toward the composite score.\n",
         "| Model | Cells with a measured budget | `finish=length` | Ratio hit the ceiling | **Budget-limited share** | Budget | Not measured |",
         "|---|---|---|---|---|---|---|"]
    for m in ms:
        s = rel_ceiling[m]
        rate = "not measured" if s["ceiling_rate"] is None else f"**{s['ceiling_rate'] * 100:.1f}%**"
        L.append(f"| `{m}` | {s['n_measured']} | {s['n_by_finish']} | {s['n_by_ratio']} | "
                 f"{rate} | {s['budgets'] or '—'} | "
                 f"{s['n_no_budget'] + s['n_unmeasured']} |")
    L.append("")
    return L


def render_reliability_table(rel: dict, ranking: list[dict]) -> list[str]:
    """The §1r reliability table: printed next to the board, not part of the
    composite score (reasons are printed below the table).
    """
    from haenv.runtime.ledger import retry_classes as _classes
    L = ["### 1r. Reliability -- **how this cell got its answer** (not counted toward the composite score, printed next to the board)\n"]
    if rel["source"] == "unavailable":
        L.append("> **Not measured**: this batch has neither `responses.jsonl` nor any `retry_*` fields in `eval.jsonl`. "
                 "**Absence is not folded into 0** -- \"no retries\" and \"retries not recorded\" are different states.\n")
        return L
    L.append(f"> Source: `{rel['source']}` · attribution goes through `tools/attribute_failures.classify`"
             f" -- the scoring side and the diagnostic side share **the same classification table**.\n")
    for w in rel["mismatch"]:
        L.append(f"> {w}\n")
    classes = list(_classes())
    #: Share of failed retries not chargeable to the model, computed from this batch.
    _n_all0 = sum(int(s["by_class"].get(c, 0)) for s in rel["models"].values() for c in classes)
    _nm0 = sum(int(s["by_class"].get(c, 0)) for s in rel["models"].values()
               for c in ("infra", "auth", "budget"))
    #: The third reason printed depends on whether this batch had any failed retries.
    _reason3 = (f"> ③ in this batch, **{_nm0 / _n_all0 * 100:.1f}%** of failed retries are attributed to "
                f"`infra`/`auth`/`budget`; that share reflects operating conditions, not model behaviour.\n"
                if _n_all0 else
                "> ③ **0 failed retries in this batch** ⇒ this column has no variance here; "
                "a zero-variance quantity carrying weight would give false precision.\n")
    order = {str(r.get("model")): i for i, r in enumerate(ranking)}
    models = sorted(rel["models"], key=lambda m: (order.get(m, 10**6), m))
    tot = {c: 0 for c in classes}
    for s in rel["models"].values():
        for c in classes:
            tot[c] += int(s["by_class"].get(c, 0))
    n_all = sum(tot.values())
    if n_all:
        _nm = tot.get("infra", 0) + tot.get("auth", 0) + tot.get("budget", 0)
        L.append(f"> This batch had **{n_all}** failed retries, of which **{_nm} ({_nm / n_all * 100:.1f}%)** are attributed to "
                 f"`infra`+`auth`+`budget` and are not charged to the model "
                 f"(provider outage, key pool, or the configured `max_tokens`).\n")
    L.append("| Model | Cells | Cells with a retry | **Rescue rate** | Retry count | "
             + " | ".join(f"`{c}`" for c in classes) + " | Not chargeable to the model |")
    L.append("|---|---|---|---|---|" + "---|" * (len(classes) + 1))
    for m in models:
        s = rel["models"][m]
        nc = s["n_cells"]
        bc = s["by_class"]
        nm = sum(int(bc.get(c, 0)) for c in ("infra", "auth", "budget"))
        n = sum(int(bc.get(c, 0)) for c in classes)
        _rr = f"**{s['n_rescued'] / nc * 100:.2f}%**" if nc else "not measured"
        L.append(f"| `{m}` | {nc or 'not measured'} | {s['n_cells_retry']} | {_rr} | {s['n_retry']} | "
                 + " | ".join(str(bc.get(c, 0)) for c in classes)
                 + f" | {nm}/{n if n else '—'} |")
    L.append("")
    L.extend(render_budget_ceiling(rel.get("ceiling") or {}, models))
    L.append("> **\"Rescue rate\" = cells that had a failed retry and still ended up with a verdict (`SCORED` / `FAIL(gate)`), divided by that model's cell count.**\n")
    L.append("> **It is not part of the composite score**, for three reasons:\n"
             "> ① every scored dimension traces to a criterion and a gold standard, while the rescue rate reflects operating conditions "
             "(provider stability, `max_tokens`), not the model's answer to the item;\n"
             "> ② it **rises monotonically with the retry limit**: raising `retries` from 2 to 5 can raise it on the same responses "
             "with no change in model behaviour;\n"
             + _reason3
             + "> ⇒ read it as **corroborating evidence next to the board**: two close composite scores with rescue rates an order of magnitude apart "
               "reflect different ways of getting an answer.\n")
    return L


def render_cost_efficiency_table(eff_list: list[dict]) -> list[str]:
    valid = [e for e in eff_list if e.get("n_usage", 0) > 0]
    if not valid:
        return []
    L = ["### 1i. Cost- and token-controlled efficiency analysis (Cost-Controlled & Pareto Frontier)\n",
         "> Compute/token-normalized efficiency analysis: evaluates model quality under equal or controlled cost, identifying the Pareto frontier (non-dominated solutions).\n",
         "| Model | Score | Prompt tokens/cell | Output tokens/cell | Total tokens/cell | Mean latency(s) | Est. cost/cell($) | Efficiency (score/k-token) | Pareto frontier |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in eff_list:
        if e.get("n_usage", 0) == 0:
            continue
        _pareto_tag = "⭐ **Pareto optimal**" if e.get("is_pareto") else "Dominated"
        _score_str = _fmt(e.get("score"))
        _lat_str = f"{e['mean_latency']:.2f}s" if e.get("mean_latency") is not None else "—"
        _eff_str = f"{e['efficiency']:.3f}" if e.get("efficiency") is not None else "—"
        _cost_str = f"${e['cost_est']:.4f}" if e.get("cost_est") else "—"
        L.append(f"| `{e['model']}` | {_score_str} | {e['mean_in']:,} | {e['mean_out']:,} | {e['mean_total']:,} | "
                 f"{_lat_str} | {_cost_str} | {_eff_str} | {_pareto_tag} |")
    L.append("")
    return L
