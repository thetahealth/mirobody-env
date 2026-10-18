"""Rendered probe readouts, including the original missing/coverage conventions."""
from __future__ import annotations

from haenv.scoring.analysis.dimensions import _rowdim
from haenv.scoring.analysis.probes import noop_reading


def append_probe_sections(L, rows) -> None:
    """Append the original probes report section in its original order."""
    _pm = [r for r in rows if r.get("prem_polarity")]
    if _pm:
        L.append("### 1b. Q-side false-premise probe -- **both rates must be read side by side**\n")
        # Polarity counts come from this batch; `PREMISE_RATIO` is only the declared ratio.
        from haenv.interaction.qside import PREMISE_RATIO as _PR
        _n_pf = sum(1 for r in _pm if r.get("prem_polarity") == "false")
        _n_pt = sum(1 for r in _pm if r.get("prem_polarity") != "false")
        L.append(f"> The user asks with a premise that is wrong (the world is unchanged, only the framing changes). "
                 f"Polarity in this batch: **false {_n_pf} cells : true {_n_pt} cells** (declared assignment "
                 f"ratio {_PR[0]}:{_PR[1]}). "
                 "**Raising a conflict on a true premise is a false alarm, not caution** -- reporting only the false-premise hit rate "
                 "would give a full score to the \"always dispute every number\" strategy.\n")
        L.append("| Model | False-premise hit rate | **True-premise false-alarm rate** | How to read it |")
        L.append("|---|---|---|---|")
        _by: dict = {}
        for r in _pm:
            _d = _by.setdefault(r.get("model") or r.get("solver"), [0, 0, 0, 0])
            _i = 0 if r["prem_polarity"] == "false" else 2
            _d[_i + (0 if r.get("prem_flagged") else 1)] += 1
        for _m, (_f1, _f0, _t1, _t0) in sorted(_by.items()):
            _hit = _f1 / max(_f1 + _f0, 1)
            _fa = _t1 / max(_t1 + _t0, 1)
            if _hit >= 0.7 and _fa > 0.5:
                _read = "**disputes every number** (both sides high)"
            elif _hit >= 0.7 and _fa <= 0.3:
                _read = "genuinely checking"
            elif _hit < 0.7 and _fa <= 0.3:
                _read = "under-reports"
            else:
                _read = "unstable on both sides"
            L.append(f"| `{_m}` | {_f1}/{_f1 + _f0} = {_hit:.2f} | "
                     f"{_t1}/{_t1 + _t0} = {_fa:.2f} | {_read} |")
        L.append("")
    # ---- multi-round: revision must be evidence-based / neutral-window
    # stability (three quantities kept separate, never merged) ----
    _mr = [r for r in rows if r.get("mr_n_transitions")]
    if _mr:
        L.append("### 1c. Multi-round: revisions are evidence-based / the neutral window doesn't drift\n")
        L.append("> `mr_cited_new_rate` is an **observational quantity**, not read as right-or-wrong: "
                 "a model not citing an evidence ID does not mean it didn't use that information.\n")
        L.append("| Model | mr_grounded_rate | EV-only count version | mr_neutral_stability | # revisions |")
        L.append("|---|---|---|---|---|")
        _b2: dict = {}
        for r in _mr:
            _b2.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_b2.items()):
            def _avg(k, _rs=_rs):
                v = [x[k] for x in _rs if isinstance(x.get(k), (int, float))]
                return f"{sum(v) / len(v):.3f}" if v else "—"
            L.append(f"| `{_m}` | {_avg('mr_grounded_rate')} | "
                     f"{_avg('mr_grounded_rate_ev_only')} | "
                     f"{_avg('mr_neutral_stability')} | "
                     f"{sum(int(x.get('mr_n_revisions') or 0) for x in _rs)} |")
        L.append("")
    # ---- error-correction track ----
    _rp = [r for r in rows if r.get("rep_verdict")]
    if _rp:
        L.append("### 1d. Error-correction track -- does it actually change its mind once refuted by evidence\n")
        L.append("> Judged only on false-premise cases (a true premise has no counter-evidence to speak of ⇒ recorded as not applicable, **not as 0**). "
                 "`blanket_doubt` = was already disputing it **before** the counter-evidence became visible ⇒ "
                 "the evidence had no effect, **this does not count as correction**.\n")
        L.append("| Model | Judgeable cells | repaired | blanket_doubt | unfounded doubt | never flagged "
                 "| median latency (rounds) | median round the counter-evidence became visible / total rounds | flagged after / flagged before |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        _b3: dict = {}
        for r in _rp:
            _b3.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_b3.items()):
            _c = {k: sum(1 for x in _rs if x.get("rep_verdict") == k)
                  for k in ("repaired", "blanket_doubt",
                            "doubt_without_evidence", "never_flagged")}
            _lat = sorted(x["rep_latency_rounds"] for x in _rs
                          if isinstance(x.get("rep_latency_rounds"), (int, float)))
            _md = _lat[len(_lat) // 2] if _lat else "—"
            _fv = sorted(x["rep_first_visible_round"] for x in _rs
                         if isinstance(x.get("rep_first_visible_round"), (int, float)))
            _nr = sorted(x["rep_n_rounds"] for x in _rs
                         if isinstance(x.get("rep_n_rounds"), (int, float)))
            _aft = sum(1 for x in _rs if x.get("rep_flagged_after"))
            _bef = sum(1 for x in _rs if x.get("rep_flagged_before"))
            L.append(f"| `{_m}` | {len(_rs)} | {_c['repaired']} | "
                     f"{_c['blanket_doubt']} | {_c['doubt_without_evidence']} | "
                     f"{_c['never_flagged']} | {_md} | "
                     f"{_fv[len(_fv) // 2] if _fv else '—'} / "
                     f"{_nr[len(_nr) // 2] if _nr else '—'} | {_aft} / {_bef} |")
        L.append("")
    # ---- tool track (T1-T4) ----
    # Grounding and redundancy are computed over cells that queried at all; zero-query
    # cells are a separate column so they cannot shrink the denominator.
    _tk = [r for r in rows if r.get("tool_budget") is not None]
    if _tk:
        L.append("### 1e. Tool track -- did it query what it should have, did it query wastefully\n")
        L.append("> **Grounding rate and query volume must be read side by side**: querying less does not mean being disciplined. "
                 "A solution that queries 3 times with 1 landing on an irrelevant signal, versus one that queries 8 times and hits every time -- "
                 "looking at either column alone will misread this.\n")
        L.append("| Solution | Cells | **Zero-query cells** | T1 grounding | T2 redundancy | Signals/cell | Checks/cell "
                 "| tests_recall | tests_prec | Spend | Budget used | How to read it |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        _by: dict = {}
        for r in _tk:
            _by.setdefault(r.get("model") or r.get("solver"), []).append(r)
        def _mean(xs):
            xs = [x for x in xs if isinstance(x, (int, float))]
            return (sum(xs) / len(xs)) if xs else None

        def _f(x, n=3):
            """`None` prints as `-`, never as 0: when a solution never
            queries at all across a batch, its grounding rate is not
            applicable, not 0.000. Printing 0 would misrepresent "never
            exercised" as "entirely wrong"."""
            return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"
        for _m, _rs in sorted(_by.items(), key=lambda kv: (-(_mean([_rowdim(x, "tool_target_grounded_rate")
                                                                     for x in kv[1]]) or 0), kv[0])):
            _use = _mean([x["tool_spent"] / x["tool_budget"] for x in _rs if x.get("tool_budget")])
            _cap = sum(1 for x in _rs
                       if (x.get("tool_spent") or 0) >= (x.get("tool_budget") or 1e9))
            _zero = sum(1 for x in _rs if not (x.get("tool_n_calls") or 0))
            # A mostly zero-query solution can post perfect precision from avoidance alone;
            # such rows are annotated.
            _ntest = _mean([x.get("tool_n_test_calls") for x in _rs]) or 0
            _tp = _mean([_rowdim(x, "tests_precision") for x in _rs])
            _tr = _mean([_rowdim(x, "tests_recall") for x in _rs])
            _zfrac = _zero / max(len(_rs), 1)
            if _zfrac >= 0.5:
                _read = "**most cells zero-query -- the grounding rate was bought by avoidance, cannot be read as precision**"
            elif _ntest < 0.5:
                _read = "**almost never touches a check item -- the two tests dimensions weren't exercised on this cell set**"
            elif isinstance(_tp, (int, float)) and _tp >= 0.99 and _ntest < 1.0:
                _read = "**one-sided perfect score + near-zero denominator, unreadable**"
            elif isinstance(_tr, (int, float)) and _tr >= 0.5 and isinstance(_tp, (int, float)) \
                    and _tp >= 0.7:
                _read = "both sides high (genuinely exercising this geometry)"
            else:
                _read = "—"
            L.append(f"| `{_m}` | {len(_rs)} | **{_zero}** "
                     f"| {_f(_mean([_rowdim(x, 'tool_target_grounded_rate') for x in _rs]))} "
                     f"| {_f(_mean([_rowdim(x, 'tool_dup_rate') for x in _rs]))} "
                     f"| {_f(_mean([x.get('tool_n_signal_calls') for x in _rs]), 1)} "
                     f"| {_f(_ntest, 1)} | {_f(_tr)} | {_f(_tp)} "
                     f"| {_f(_mean([x.get('tool_spent') for x in _rs]), 1)} "
                     f"| {(_use or 0) * 100:.1f}% | {_read} |")
        L.append("")
        _spread = {}
        for _k in ("tool_target_grounded_rate", "tool_dup_rate", "tool_n_calls", "tool_spent"):
            _v = [_mean([_rowdim(x, _k) for x in _rs]) for _rs in _by.values()]
            _v = [x for x in _v if x is not None]
            _spread[_k] = (max(_v) - min(_v)) if len(_v) > 1 else 0.0
        _dead = [k for k, v in _spread.items() if v <= 1e-9]
        L.append("> Discrimination self-check: " + " · ".join(f"`{k}` spread {v:.3f}" for k, v in _spread.items()))
        if _dead:
            L.append(f">\n> **{', '.join('`' + k + '`' for k in _dead)} has zero spread** -- "
                     f"this column **does not constitute a capability finding** in this batch. Check first whether the item even gave "
                     f"a chance to fail (e.g. a legitimate roster handed over for free would make grounding structurally impossible to fail).\n")
        else:
            L.append(">\n> ✅ All four columns have a range -- the items gave a chance to fail.\n")
        _t4 = [r for r in rows if r.get("tool_key_covered") is not None]
        if not _t4:
            L.append("> **T4 (concluding without querying the key signal) does not apply to this batch**: it requires a \"key signal\" to be declared, "
                     "and only the early-warning item type declares one. **Record it as not applicable, not as 0** -- scoring it 0 would misrepresent "
                     "\"this item doesn't test this\" as \"the model failed to do it\".\n")
    # ---- noop probe: hit rate and false-alarm rate side by side ----
    _np = [r for r in rows if r.get("noop_polarity")]
    if _np:
        L.append("### 1f. noop probe -- asked about a signal that doesn't exist in the environment, does it make one up\n")
        # Polarity counts come from this batch; `evaluate.NOOP_RATIO` is only the declared ratio.
        from haenv.interaction.qside import NOOP_RATIO as _NR
        _GAP_P = ("gap", "absent")
        _COV_P = ("covered", "present")
        _n_gap = sum(1 for r in _np if r.get("noop_polarity") in _GAP_P)
        _n_cov = sum(1 for r in _np if r.get("noop_polarity") in _COV_P)
        # The judge (`judges.trajectory.judge_noop_probe`) reads only
        # `signal_quality[target]`; the global `data_sufficiency` flag is
        # deliberately *not* a declaration about this window (reasoning in
        # that function).
        L.append(f"> Polarity in this batch: **absent {_n_gap} cells : present {_n_cov} cells** "
                 f"(declared assignment ratio {_NR[0]}:{_NR[1]}; after per-case assignment the batch count is "
                 "what holds). **Declaring insufficient data for a signal that is actually present is a false alarm**, "
                 "not caution. The criterion reads only the three-value enum (`present` / `no_data_in_window` / "
                 "`unreliable`) in `data_quality.signal_quality[<the asked signal>]`; an unanswered or off-enum value "
                 "scores 0; **the global `data_sufficiency` does not count as "
                 "a declaration about this window** (\"the whole case lacks information\" and \"this window has no "
                 "readings\" are two different things); cells that used the global flag without writing the signal "
                 "are listed separately in the \"Answer channel\" column (`noop_used_global_flag`). "
                 "**It does not scan free text** ⇒ an identity relation ⇒ this dimension needs no anchor.\n")
        L.append("| Solution | Absent · correctly declared | **Present · false alarm** | Consistency `noop_ok` "
                 "| Answer channel (absent side)<br><sub>that signal / global flag only / unanswered</sub> | How to read it |")
        L.append("|---|---|---|---|---|---|")
        _bn: dict = {}
        for r in _np:
            _bn.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_bn.items()):
            # Polarity values `gap` / `covered` as produced by `evaluate._noop_probe_for`;
            # the older `absent` / `present` are also accepted.
            _GAP = ("gap", "absent")        # this window has no data => the
                                             # correct answer is "insufficient"
            _COV = ("covered", "present")   # this window has data => declaring
                                             # insufficient is a false alarm
            # The per-solver reading (including the split by answer
            # channel) lives in `analytics.noop_reading` so it can be tested
            # without rendering a whole report.
            _nr = noop_reading(_rs, gap=_GAP, cov=_COV)
            _f = lambda x: (f"{x:.2f}" if isinstance(x, (int, float)) else "—")
            _ch = _nr["channels"]
            _chs = (f"{_ch['signal_quality']} / {_ch['global_only']} / {_ch['none']}"
                    + (f" · unknown {_ch['unknown']}" if _ch["unknown"] else ""))
            L.append(f"| `{_m}` | {_nr['a1']}/{_nr['n_gap']} = {_f(_nr['hit'])} "
                     f"| {_nr['p1']}/{_nr['n_cov']} = {_f(_nr['fa'])} "
                     f"| {_f(_nr['ok'])} | {_chs} | {_nr['reading']} |")
        L.append("")
    # ---- data-verification items, by kind (their difficulty differs) ----
    _qq = [r for r in rows if r.get("quant_kind")]
    if _qq:
        L.append("### 1g. Data-verification items -- did it read the data correctly (computable, needs no anchor)\n")
        L.append("> `trend` up/down/flat · `peak_day` which day was highest · `abnormal_days` how many days were outliers. "
                 "The gold value is **computed by code**, the answer is an **enum or integer**, and the criterion does an **exact match** (integers get no tolerance).\n"
                 ">\n"
                 "> **This dimension is a check on external validity**: a model that can't even get \"which day was weight highest\" right "
                 "should have its diagnostic readings discounted.\n")
        _kinds = sorted({r["quant_kind"] for r in _qq})
        L.append("| Solution | Answer rate | " + " | ".join(f"`{k}`" for k in _kinds) + " | Total |")
        L.append("|---|---|" + "---|" * (len(_kinds) + 1))
        _bq: dict = {}
        for r in _qq:
            _bq.setdefault(r.get("model") or r.get("solver"), []).append(r)
        # Sort keys are total orders so the report does not depend on row order.
        for _m, _rs in sorted(_bq.items(),
                              key=lambda kv: (-sum(x.get("quant_ok") or 0 for x in kv[1]), kv[0])):
            _ans = sum(1 for x in _rs if x.get("quant_answered"))
            cells = []
            for k in _kinds:
                sub = [x for x in _rs if x["quant_kind"] == k]
                v = [x["quant_ok"] for x in sub if isinstance(x.get("quant_ok"), (int, float))]
                cells.append(f"{sum(v):.0f}/{len(v)}" if v else "—")
            _all = [x["quant_ok"] for x in _rs if isinstance(x.get("quant_ok"), (int, float))]
            L.append(f"| `{_m}` | {_ans}/{len(_rs)} | " + " | ".join(cells)
                     + f" | **{(sum(_all) / len(_all) if _all else 0):.3f}** |")
        L.append("")
        if not any(x.get("quant_answered") for x in _qq):
            L.append("> **Not one cell in the batch answered** -- offline stubs don't emit `quant_answer`, "
                     "that is a property of the stub, not a model reading.\n")
    # ---- abstention calibration: both halves side by side ----
    _ab = [r for r in rows if r.get("abst_sufficient") is not None]
    if _ab:
        L.append("### 1h. Abstention calibration -- don't abstain when data is sufficient, don't force an answer when it isn't\n")
        L.append("> `abst_over` = **abstained despite sufficient information** (over-abstention); mirrors `noop_ok`'s "
                 "\"insufficient but didn't say so\". Abstention **only** recognizes `data_sufficiency`, "
                 "**action tiers never count** -- not even `A1` (improve data quality, semantically the closest tier to abstaining) "
                 "counts either.\n")
        L.append("| Solution | Sufficient cases · abstained (over) | Insufficient cases · abstained (correct) | Calibration `abst_ok` |")
        L.append("|---|---|---|---|")
        _bb: dict = {}
        for r in _ab:
            _bb.setdefault(r.get("model") or r.get("solver"), []).append(r)
        for _m, _rs in sorted(_bb.items()):
            _suf = [x for x in _rs if x.get("abst_sufficient")]
            _ins = [x for x in _rs if not x.get("abst_sufficient")]
            _so = sum(1 for x in _suf if x.get("abst_abstained"))
            _io = sum(1 for x in _ins if x.get("abst_abstained"))
            _ok = [x["abst_ok"] for x in _rs if isinstance(x.get("abst_ok"), (int, float))]
            L.append(f"| `{_m}` | {_so}/{len(_suf)} | {_io}/{len(_ins)} "
                     f"| {(sum(_ok) / len(_ok) if _ok else 0):.3f} |")
        L.append("")
