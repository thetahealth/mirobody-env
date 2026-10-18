"""No-op probe domain measurements."""
from __future__ import annotations



def noop_reading(rs: list[dict], gap: tuple = ("gap", "absent"),
                 cov: tuple = ("covered", "present")) -> dict:
    """One solver's row of the noop-probe table (`report` section 1f).

    Counts hits on blank windows and false alarms on covered ones, and which
    channel blank-window cells answered through (`noop_answer_channel`):
    * both high -> "always says insufficient";
    * hit rate < 0.2 -> by channel: `signal_quality` = fabricated, `global_only` =
      wrong channel, `none` = no answer;
    * hit >= 0.7 and false alarms <= 0.3 -> actually checking.
    Rows without `noop_answer_channel` fall back to `noop_used_global_flag`, then
    `unknown`.
    """
    def _chan(x: dict) -> str:
        c = x.get("noop_answer_channel")
        if c in ("signal_quality", "global_only", "none"):
            return c
        if x.get("noop_used_global_flag"):
            return "global_only"
        return "unknown"
    ab = [x for x in rs if x.get("noop_polarity") in gap]
    pr = [x for x in rs if x.get("noop_polarity") in cov]
    a1 = sum(1 for x in ab if x.get("noop_declared"))
    p1 = sum(1 for x in pr if x.get("noop_declared"))
    oks = [x["noop_ok"] for x in rs if isinstance(x.get("noop_ok"), (int, float))]
    hit = a1 / len(ab) if ab else None
    fa = p1 / len(pr) if pr else None
    ch = {k: sum(1 for x in ab if _chan(x) == k)
          for k in ("signal_quality", "global_only", "none", "unknown")}
    und = [x for x in ab if not x.get("noop_declared")]
    fab = sum(1 for x in und if _chan(x) == "signal_quality")
    glob = sum(1 for x in und if _chan(x) == "global_only")
    none = sum(1 for x in und if _chan(x) == "none")
    if hit is not None and fa is not None and hit >= 0.7 and fa > 0.5:
        rd, kind = "🔴 **Always says data is insufficient** (both sides high)", "always_insufficient"
    elif hit is not None and hit < 0.2:
        if und and fab * 2 > len(und):
            rd, kind = "🔴 **Answers about absent data anyway** -- fabricated premise", "fabricated"
        elif und and glob * 2 > len(und):
            rd, kind = ("🟠 **Used the global `data_sufficiency`, not the queried signal** -- "
                        "wrong channel, not read as fabrication"), "global_flag"
        elif und and none * 2 > len(und):
            rd, kind = ("⚪ **Did not answer the queried signal** (no such key in `signal_quality`) -- "
                        "unanswered, not read as fabrication"), "no_answer"
        else:
            rd, kind = "🔴 Rarely declares on the absent side -- mixed channels, inspect `noop_answer_channel` per cell", "mixed"
    elif hit is not None and fa is not None and hit >= 0.7 and fa <= 0.3:
        rd, kind = "Actually checking the environment", "checking"
    else:
        rd, kind = "—", None
    return {"n_gap": len(ab), "n_cov": len(pr), "a1": a1, "p1": p1, "hit": hit, "fa": fa,
            "ok": (sum(oks) / len(oks) if oks else None), "channels": ch,
            "reading": rd, "kind": kind}
