"""gen_judge_inventory.py -- generate `docs/design/judge-inventory.md` from code.

Sources: `judges.JUDGES` (mounted judges), `registry/scoring.yaml` (role, validity,
pending decisions), `process.UNIMPLEMENTED` (gold fields no judge implements) and
`registry/disputed_gold.yaml` (gold that may itself be wrong). Only language-neutral
facts are copied; explanations stay in their source files. The output has no date, so
`--check` can compare it byte for byte.

Usage:  uv run python tools/gen_judge_inventory.py           # rewrite the file
        uv run python tools/gen_judge_inventory.py --check   # exit 1 if it is stale

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
_KP = (yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8")) or {}).get("kernel_path")
if _KP:
    sys.path.insert(0, str((_ROOT / _KP).resolve()))

# Judge -> the metric keys it produces, used to look up each judge's role in the scoring
# profile. Explicit, never guessed from names.
JUDGE_METRICS: dict[str, tuple[str, ...]] = {
    "forecast": ("brier", "direction_ok", "abstained"),
    "driver": ("driver_hit",),
    "alternative": ("a1",),
    "dx_unified": ("dx_hit", "dx_hit_top1", "dx_rank"),
    "dx_comorbidity": ("dx_hit", "dx_threads_matched"),
    "dx_independent": ("held_independent",),
    "join_type": ("join_hit", "join_said"),
    "dx_rival": ("rival_considered", "rival_ruled_out", "rival_top1_live", "excl_no_evidence"),
    "disc_tool": ("disc_recall", "disc_covered", "disc_n_tests_proposed"),
    "join_selfcheck": ("join_self_contradiction", "top1_claims_unified"),
    "join_cover": ("join_top1_covers_all", "join_top1_cover_frac", "join_scope_ok"),
    "workup": ("tests_recall", "tests_precision", "urgency_ok", "specialty_ok"),
    "slices": ("slice_converged",),
    "slice_revision": ("rev_stability", "rev_responsiveness"),
    "slices_workup": ("wk_urgency_ok_at", "wk_n_slices_urgency_ok"),
    # Keys from each judge's return statements; `*_note` keys are left out.
    "abstention": ("abst_ok", "abst_over", "abst_abstained"),
    "commit_timing": ("ct_ok", "ct_declared", "ct_forced_dx"),
    "slices_abstention": ("abst_ok", "abst_over", "abst_under", "abst_two_sided"),
    "review_flag": ("review_flag_ok", "review_declared", "review_warranted"),
    "slices_gates": ("slice_gate_n_judged", "slice_gate_unknown"),
    "slices_review": ("review_flag_ok", "review_declared", "review_warranted"),
    "self_contradictory_exclusion": ("sce_self_contradictory", "sce_contradiction_rate",
                                     "sce_unrevealed_rate"),
    "slices_self_contradictory_exclusion": ("sce_self_contradictory", "sce_contradiction_rate",
                                            "sce_unrevealed_rate"),
    "what_not_to_do": ("wnd_n_items", "wnd_n_distinct", "wnd_dup_rate"),
    "slices_wnd": ("wnd_n_items", "wnd_n_distinct", "wnd_dup_rate"),
    "multiround_revision": ("mr_grounded_rate", "mr_n_revisions", "mr_neutral_stability"),
    "premise_repair": ("rep_verdict", "rep_latency_rounds"),
}

OUT = _ROOT / "docs/design/judge-inventory.md"
REBUILD = "uv run python tools/gen_judge_inventory.py"


def _marker(text: str) -> str:
    """The leading status mark of a registry note (green / amber / red circle), or a dash."""
    t = str(text or "").lstrip()
    return next((m for m in ("🟢", "🟡", "🔴") if t.startswith(m)), "—")


def render() -> str:
    """The inventory as text; deterministic (no timestamp, no environment)."""
    import haenv.judges as J
    from haenv.process import UNIMPLEMENTED
    from haenv.scoring import load_profile

    prof = load_profile()
    L: list[str] = []
    P = L.append

    P("<!-- generated, do not edit. Rebuild: `" + REBUILD + "` -->")
    P("# Judge inventory and open items (**generated from code — do not hand-edit**)")
    P("")
    P("Source of truth: `judges.JUDGES` · `registry/scoring.yaml` · `process.UNIMPLEMENTED` · "
      "`registry/disputed_gold.yaml`.")
    P("This file is generated from code, not hand-written, so it can't go stale silently. Rebuild")
    P(f"with `{REBUILD}`; `--check` exits 1 when this file differs from what the code says.")
    P("")

    P("## 1. Judges mounted on the table")
    P("")
    P(f"**{len(J.JUDGES)}** in total. `role` is taken from the scoring profile; blank means that "
      "judge's output is not")
    P("registered as a scoring dimension. A judge the generator's explicit metric table does not "
      "list shows")
    P("`(not in table)`: the table is never guessed from names.")
    P("")
    P("| Judge | Metrics produced | role | Validity |")
    P("|---|---|---|---|")
    for j in J.JUDGES:
        ms = JUDGE_METRICS.get(j.name)
        roles, vals = [], []
        for m in ms or ():
            for pname, pm in prof.metrics.items():
                if pname == m or pm.get("source_metric") == m:
                    roles.append(pm["role"])
                    st, ag = prof.validity_state(pname)
                    vals.append(f"{st}" + (f" {ag}" if ag is not None else ""))
        role = "/".join(sorted(set(roles))) or "—"
        val = "/".join(sorted(set(vals))) or "—"
        shown = ", ".join(ms) if ms else "(not in table)"
        P(f"| `{j.name}` | {shown} | {role} | {val} |")
    P("")

    P("## 2. Role distribution in the scoring profile")
    P("")
    for role in ("dim", "anchor", "diagnostic"):
        names = prof.by_role(role)
        P(f"* **{role}** ({len(names)}): {', '.join(f'`{n}`' for n in names) or '—'}")
    P("")
    unmet = prof.unmet_validity()
    P(f"{len(unmet)} of the {len(prof.scored_dims)} scoring dimensions do not "
      f"satisfy the validity precondition (threshold {prof.threshold}):")
    P("")
    P("| Dimension | Validity status | Agreement rate | Pending decision |")
    P("|---|---|---|---|")
    for u in unmet:
        pend = "recorded in `registry/scoring.yaml`" if str(u["pending"] or "").strip() else "—"
        P(f"| `{u['metric']}` | {u['state']} | "
          f"{u['agreement'] if u['agreement'] is not None else '—'} | {pend} |")
    P("")

    P("## 3. Gold standard is written, judge not implemented")
    P("")
    P("The status mark is copied from the source; what is implemented, what is missing and why is")
    P("written next to each key in `haenv/process.py` (`UNIMPLEMENTED`).")
    P("")
    for k, v in UNIMPLEMENTED.items():
        P(f"* {_marker(v)} **`{k}`**")
    P("")

    P("## 3b. Gold standards under dispute — **the case itself may be wrong**")
    P("")
    P("This repo has machinery for \"the judge might be wrong\"; this section is the same treatment")
    P("for \"the case might be wrong\". That error is the hardest to notice: the judge and the model")
    P("are both right, the case is wrong, and the report prints it as \"the model's capability is")
    P("poor.\" Affected cases are still scored and are marked in the report.")
    P("")
    try:
        from haenv.registry import _load_yaml, load_disputed_gold
        dg = load_disputed_gold()
        applies = (_load_yaml("disputed_gold.yaml").get("applies_to") or {})
    except Exception as e:
        P(f"**The registry could not be read**: `{type(e).__name__}`. Nothing below is a reading.")
        dg, applies = (), {}
    if dg:
        P("| Id | Gold field | Status | Applies to specs | Applies to dimensions |")
        P("|---|---|---|---|---|")
        for d in dg:
            a = applies.get(d["id"]) or {}
            specs = ", ".join(f"`{x}`" for x in a.get("specs") or ()) or "not enumerated"
            dims = ", ".join(f"`{x}`" for x in a.get("dims") or ()) or "not enumerated"
            P(f"| **{d['id']}** | `{d['field']}` | `{d['status']}` | {specs} | {dims} |")
        P("")
        P("Scope, the question, the evidence, the counter-evidence, the impact and who can resolve "
          "each one")
        P("are recorded per entry in `registry/disputed_gold.yaml`. The report matches cases through")
        P("`applies_to`, so an entry whose specs are not enumerated marks no case.")
    else:
        P("No entry registered.")
    P("")

    P("*SYNTHETIC, for evaluation only, not medical advice.*")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate docs/design/judge-inventory.md from code.")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if the file on disk differs from the code")
    a = ap.parse_args(argv)
    text = render()
    if a.check:
        on_disk = OUT.read_text(encoding="utf-8") if OUT.is_file() else None
        if on_disk == text:
            print(f"[inventory] {OUT.relative_to(_ROOT)} is current")
            return 0
        print(f"[inventory] {OUT.relative_to(_ROOT)} differs from the code; rebuild: {REBUILD}",
              file=sys.stderr)
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n-> {OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
