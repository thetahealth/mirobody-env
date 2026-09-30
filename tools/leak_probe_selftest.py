"""leak_probe_selftest.py -- controls for the kernel's `leakage_probe`: each leak mechanism
must fire red (negative controls, `expect_leak`) on every payload field, and clean input must
pass (positive controls, `expect_clean`, `kind` prefixed with `!`). Without these, a gate that
never fires cannot be told from clean data.

Run: uv run python tools/leak_probe_selftest.py
SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from haenv import kernel_path as _kernel_path       # noqa: E402
# The kernel path comes from `haenv.kernel_path()`, the repo's single resolution point.
if (_kp := _kernel_path()):
    sys.path.insert(0, str(_kp))

RESULTS: list[tuple[str, str, bool, str]] = []

CANARY = "adjudication"          # use a real forbidden token as the probe, don't invent a new one


def expect_leak(label: str, kind: str, sp, T: int, want: str) -> None:
    """Negative control: must fire red with `want` in the reason (not just any violation)."""
    from build import leakage_probe                                   # kernel
    ok, viol = leakage_probe(sp, T)
    hit = (not ok) and any(want in v for v in viol)
    RESULTS.append((label, kind, hit, f"ok={ok} viol={viol[:3]}"))


def expect_clean(label: str, kind: str, sp, T: int) -> None:
    """Positive control: a normal input must not fire red (guards against
    false positives / a gate that's always red). `kind` is marked with a `!` prefix."""
    from build import leakage_probe                                   # kernel
    ok, viol = leakage_probe(sp, T)
    RESULTS.append((label, f"!{kind}", ok, "clean" if ok else f"false positive {viol[:3]}"))


def _real_payload():
    """A payload built by the production path, so the scan surface is tested on the real shape."""
    from haenv import job as J
    from haenv.build import build_case
    from haenv.ddx import ddx_case_specs
    from build import build_instance                                  # kernel

    for s in ddx_case_specs(only=["JD-PCOS"]):
        cs = J.CaseSpec(case_id=s["case_id"], raw=s["raw"], latent=s["latent"])
        raw, audit = build_case(cs)
        if raw is None:
            raise SystemExit(f"Fixture case failed to emit, so the negative control has nothing to test: {audit}")
        T = int(raw.prediction_context["prediction_time_T"])
        sp, _ = build_instance(raw, T)
        return sp, T
    raise SystemExit("Could not obtain the JD-PCOS fixture case")


def main() -> int:
    from build import FORBIDDEN_TOKENS, leakage_probe                 # kernel

    sp0, T = _real_payload()

    # ---------- B: doesn't fire when it shouldn't (pin the baseline first: every red after this must be one this file deliberately caused) ----------
    expect_clean("A real payload, unmodified, must be let through", "false_positive", sp0, T)

    sp = copy.deepcopy(sp0)
    sig = next(iter(sp.longitudinal_data))
    sp.longitudinal_data[sig].append({"ts": T, "value": 1.0})
    expect_clean("A boundary point at ts == T must not be judged a leak (≤T is a closed interval)", "boundary_ts", sp, T)

    sp = copy.deepcopy(sp0)
    if sp.evidence_ledger:
        sp.evidence_ledger[0]["source_timestamp"] = T
        expect_clean("A boundary EV with source_timestamp == T must not be judged a leak", "boundary_ev", sp, T)

    # ---------- A: fires when it should -- each of the three mechanisms fires red once ----------
    sp = copy.deepcopy(sp0)
    sp.longitudinal_data[sig].append({"ts": T + 1, "value": 1.0})
    expect_leak("① Temporal leak: a point at ts = T+1 must fire red", "future_timepoint",
                sp, T, "future_timepoint")

    sp = copy.deepcopy(sp0)
    if sp.evidence_ledger:
        sp.evidence_ledger[0]["source_timestamp"] = T + 1
        expect_leak("② EV temporal leak: source_timestamp = T+1 must fire red", "future_evidence",
                    sp, T, "future_evidence")

    for tok in FORBIDDEN_TOKENS:
        sp = copy.deepcopy(sp0)
        sp.user_profile = {**sp.user_profile, "_probe": f"值里带 {tok} 这个词"}
        expect_leak(f"③ Forbidden token `{tok}` must fire red", "forbidden_token",
                    sp, T, f"forbidden_token:{tok}")

    # ---------- C: exercised surface -- every payload field is fired red individually ----------
    def _plant(mut, label: str) -> None:
        sp = copy.deepcopy(sp0)
        mut(sp)
        expect_leak(f"Exercised surface · {label}", "surface", sp, T, f"forbidden_token:{CANARY}")

    def _p_case_id(s):
        s.case_id = f"{s.case_id}-{CANARY}"

    def _p_profile_nested(s):
        s.user_profile = {**s.user_profile, "_probe": {"深": {"层": CANARY}}}

    def _p_profile_list(s):
        s.user_profile = {**s.user_profile, "_probe": ["a", CANARY]}

    def _p_context(s):
        s.prediction_context = {**s.prediction_context, "_probe": CANARY}

    def _p_signal_key(s):
        s.longitudinal_data = {**s.longitudinal_data, f"hba1c_{CANARY}": [{"ts": 0, "value": 1}]}

    def _p_point_key(s):
        s.longitudinal_data = copy.deepcopy(s.longitudinal_data)
        s.longitudinal_data[sig][0][CANARY] = "x"

    def _p_point_value(s):
        s.longitudinal_data = copy.deepcopy(s.longitudinal_data)
        s.longitudinal_data[sig][0]["note"] = CANARY

    def _p_ev_value(s):
        s.evidence_ledger = copy.deepcopy(s.evidence_ledger)
        s.evidence_ledger[0]["content"] = f"结论来自 {CANARY}"

    def _p_ev_nested(s):
        s.evidence_ledger = copy.deepcopy(s.evidence_ledger)
        s.evidence_ledger[0]["_probe"] = {"里层": [CANARY]}

    for mut, label in (
        (_p_case_id,        "case_id"),
        (_p_context,        "prediction_context top level"),
        (_p_profile_nested, "user_profile, a two-level nested dict"),
        (_p_profile_list,   "an element of a list in user_profile"),
        (_p_signal_key,     "longitudinal_data's signal name (the key, not the value)"),
        (_p_point_key,      "longitudinal_data point's extra key name"),
        (_p_point_value,    "longitudinal_data point's extra key value"),
        (_p_ev_value,       "evidence_ledger entry's value"),
        (_p_ev_nested,      "a nested list inside an evidence_ledger entry"),
    ):
        if label.startswith("evidence") and not sp0.evidence_ledger:
            continue
        _plant(mut, label)

    # Case: the blob is compared after .lower(), and the wordlist is all lowercase => an uppercase spelling must fire red the same way
    sp = copy.deepcopy(sp0)
    sp.user_profile = {**sp.user_profile, "_probe": CANARY.upper()}
    expect_leak("Exercised surface · uppercase ADJUDICATION must fire red the same way", "surface",
                sp, T, f"forbidden_token:{CANARY}")

    # ---------- D: declared boundaries (expected to pass) ----------
    # Chinese answer words are the verify layer's job (`solver_text_no_answer_info`).
    sp = copy.deepcopy(sp0)
    sp.user_profile = {**sp.user_profile, "_probe": "结局:体重反弹,真驱动是依从性下降"}
    ok, _ = leakage_probe(sp, T)
    RESULTS.append(("Boundary declaration: Chinese-language answer words aren't the kernel gate's job "
                    "(covered by the verify layer's `solver_text_no_answer_info`)", "!scope_chinese", ok,
                    "The kernel gate letting this through is expected; coverage at this layer is guaranteed by verify_selftest"))

    # An EV without `source_timestamp` passes the probe (default 0), while `build_instance`
    # excludes it (default 10**9).
    sp = copy.deepcopy(sp0)
    sp.evidence_ledger = copy.deepcopy(sp.evidence_ledger)
    sp.evidence_ledger.append({"evidence_id": "EV-NOTS", "content": "无时间戳"})
    ok, _ = leakage_probe(sp, T)
    RESULTS.append(("Boundary declaration: an EV missing source_timestamp is let through here"
                    " (build_instance treats a missing key as +∞ and excludes it; the probe treats a missing key as 0 and lets it through -- the defaults are opposite)",
                    "!scope_missing_ts", ok, "The two defaults disagree; the build side is conservative ⇒ the direction is safe"))

    # ---------- Summary ----------
    print(f"|{'-' * 52}|{'-' * 24}|------|")
    n_miss = 0
    for label, kind, caught, detail in RESULTS:
        if not caught:
            n_miss += 1
        print(f"| {label:50s} | {kind:22s} | {'✅' if caught else '❌ missed'} |")
    print(f"\n{len(RESULTS)} checks, {n_miss} missed.")
    for label, kind, caught, detail in RESULTS:
        print(f"  - {label} → {kind}: {detail}")
    return 0 if n_miss == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
