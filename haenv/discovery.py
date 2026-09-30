"""discovery.py -- self-discovery judges for the gated geometry: did the model fetch the
evidence behind the gate itself, or conclude without it?

* D1 `sd_coverage` -- real gated signals self-fetched / real gated signals available.
  Descriptive only: flooding every signal also scores high, so report it next to the
  `gated_flood` baseline. Query quality is judged by `tool_target_grounded_rate` and
  `tool_budget_thrift`.
* D2 `sd_blind_sufficient` / `sd_unearned_insufficient` -- declares data sufficient, or
  insufficient, having queried nothing behind the gate (reported separately).
* D3 `sd_unqueried_refs` -- cites a signal that was neither handed free nor queried.

Nothing behind the gate means not applicable, never a full score. The key name
matches other benchmarks' `self_discovery` but the construct differs; do not compare them.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import re

#: `data_sufficiency` enum values; anything else is undeclared (no fuzzy matching).
SUFFICIENT = ("sufficient", "adequate")
INSUFFICIENT = ("insufficient_data", "insufficient", "limited")

_SIG_RE = re.compile(r"[a-z][a-z0-9_]{2,}_series|hba1c|weight|waist|glucose|bp_|potassium")


def sufficiency_of(raw) -> str | None:
    """Pull the sufficiency declaration out of the answer. Anything outside
    the enum is `None` (undeclared != declared sufficient)."""
    if not isinstance(raw, dict):
        return None
    dq = raw.get("data_quality")
    v = str((dq or {}).get("data_sufficiency") or "").strip().lower()
    if v in SUFFICIENT:
        return "sufficient"
    if v in INSUFFICIENT:
        return "insufficient"
    return None


def signal_refs(raw) -> set[str]:
    """Signal names the answer cites: the keys of `data_quality.signal_quality` (free text is
    not scanned)."""
    if not isinstance(raw, dict):
        return set()
    sq = ((raw.get("data_quality") or {}).get("signal_quality") or {})
    return {str(k).strip() for k in sq} if isinstance(sq, dict) else set()


def judge_self_discovery(raw, *, available: set[str], free: set[str],
                         queried: set[str] | None = None,
                         n_unique_targets: int | None = None,
                         n_grounded_signal: int | None = None) -> dict:
    """D1-D3. Without the queried set, coverage is bounded from `n_grounded_signal` /
    `n_unique_targets` and flagged `sd_coverage_is_bound`."""
    gated = set(available) - set(free)
    out: dict = {"sd_n_gated_available": len(gated), "sd_n_free": len(free & set(available))}
    if not gated:
        out.update({"sd_coverage": None, "sd_blind_sufficient": None,
                    "sd_unearned_insufficient": None, "sd_unqueried_refs": None,
                    "sd_note": "nothing behind the gate to query -> not applicable (not scored as full credit)"})
        return out

    if queried is not None:
        got = len(set(queried) & gated)
        out["sd_coverage_is_bound"] = False
    elif n_grounded_signal is not None:
        # Upper bound: grounded calls can repeat a signal or include the free `weight`.
        caps = [int(n_grounded_signal), len(gated)]
        if n_unique_targets is not None:
            caps.append(int(n_unique_targets))
        got = max(0, min(caps))
        out["sd_coverage_is_bound"] = True
    else:
        out.update({"sd_coverage": None, "sd_blind_sufficient": None,
                    "sd_unearned_insufficient": None, "sd_unqueried_refs": None,
                    "sd_note": "no call record -> not applicable"})
        return out

    out["sd_n_self_obtained"] = got
    out["sd_coverage"] = round(got / len(gated), 3)

    # ---- D2: report both sides separately ----
    suf = sufficiency_of(raw)
    out["sd_declared"] = suf
    # Keyed on `got == 0`: saying "insufficient" after querying is correct abstention.
    out["sd_blind_sufficient"] = bool(suf == "sufficient" and got == 0)
    out["sd_unearned_insufficient"] = bool(suf == "insufficient" and got == 0)

    # ---- D3: cited a signal that was neither handed free nor queried ----
    if queried is not None:
        seen = set(free) | set(queried)
        bad = sorted(r for r in signal_refs(raw) if r in available and r not in seen)
        out["sd_unqueried_refs"] = len(bad)
        out["sd_unqueried_ref_names"] = bad[:5] or None
    else:
        out["sd_unqueried_refs"] = None
        out["sd_unqueried_note"] = "missing per-turn target record -> not applicable"
    return out


#: Prompt block for the gap-naming probe (a separate probe id, so `ddx.direct` batches
#: stay comparable).
GAP_SCHEMA_HINT = """
若你判断数据不足,再给一个 "missing_data" 数组,列出**你认为缺的是哪几项**。
⚠️ 只能填**菜单里的 target 名**(逐字照抄),不要用自己的说法;菜单里没有的项不要填。
留空数组表示"数据已足够"。"""


def judge_gap_naming(raw, *, available: set[str], queried: set[str],
                     free: set[str] = frozenset()) -> dict:
    """`gap_naming_precision`: when the answer declares data missing, are the named items
    real menu targets that were not one query away? Needs the `missing_data` field requested
    by `GAP_SCHEMA_HINT`; an empty array (declared sufficient) is kept separate from a missing
    field."""
    if not isinstance(raw, dict) or "missing_data" not in raw:
        return {"gap_named_n": None, "gap_naming_precision": None,
                "gap_note": "the item did not require missing_data -> not applicable (needs a new probe)"}
    named = [str(x).strip() for x in (raw.get("missing_data") or []) if str(x).strip()]
    if not named:
        return {"gap_named_n": 0, "gap_naming_precision": None,
                "gap_note": "declared data sufficient (empty array) -> nothing to judge, kept separate from a missing field"}
    avail, q = set(available), set(queried)
    invalid = [x for x in named if x not in avail]
    # Behind the gate, never queried, and not handed free ⇒ one query away
    obtainable = [x for x in named if x in avail and x not in q and x not in set(free)]
    bad = len(set(invalid) | set(obtainable))
    return {"gap_named_n": len(named), "gap_named_invalid": len(invalid),
            "gap_named_obtainable": len(obtainable),
            "gap_naming_precision": round(1 - bad / len(named), 3),
            "gap_named_obtainable_names": sorted(obtainable)[:5] or None}


def visibility_split(sp, withheld: dict) -> tuple[set[str], set[str]]:
    """(signals this patient actually has, the subset handed free). Reuses
    gated's definition rather than writing a second one."""
    from .gated import ALWAYS_VISIBLE, available_targets
    avail = available_targets(sp, withheld)
    return avail, {k for k in ALWAYS_VISIBLE if k in avail}
