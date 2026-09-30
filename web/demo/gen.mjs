/* gen.mjs — the piece that **actually generates the patient** in the browser.
 *
 * The page's knobs feed this file, and it computes the patient's weight, adherence and lab
 * streams live, so every knob changes what is drawn.
 *
 * ## Staying in step with production
 *
 * 1. **Not a single constant is hardcoded.** Everything comes from `data.json`'s `kernel`
 *    section, pulled live from the production modules by `page_data.py:_kernel()`.
 *    The day the kernel gets recalibrated, re-running the export re-syncs it automatically;
 *    hardcoding it would leave the page silently showing a stale number.
 * 2. **The random layer isn't reimplemented, it's looked up.** `events._det_shock`
 *    (blake2b) and `build._weighed_on` (sha256) only depend on `(case_id, index)` —
 *    **never on any knob** ⇒ the production function itself computes and exports them.
 *    The browser side never implements the hash, so it has no way to get the hash wrong.
 * 3. **Deterministic math is checked against a golden vector.** This remaining layer really
 *    is two separate implementations. `data.json.golden` holds the point-by-point values
 *    the production function computes for a set of parameter combinations; `selfCheck()` recomputes them
 *    with this file and compares point-by-point, and **the largest deviation is printed on
 *    the page**. The moment of drift is visible without anyone having to go check for it.
 *
 * ## Correspondence table (use this as a checklist when changing production code)
 *
 * | This file | Production |
 * |---|---|
 * | `shape` | `haenv/build.py:_shape` |
 * | `feasibleK` | `haenv/build.py:_feasible_k` |
 * | `weightSeries` · `weightRender` | `haenv/build.py:_weight_series` · `_weight_render` (deterministic path) |
 * | `skeletonBase` | `_skeleton_base` (three-segment skeleton, descent pacing, rebound shape) |
 * | `pacePieces` · `paceAt` · `descentPieces` · `reboundPieces` · `pacedProgress` | `_pace_pieces` · `_pace_at` · `_descent_pieces` · `_rebound_pieces` · `_paced_progress` |
 * | `driftSeries` | `_drift_series` |
 * | `episodePlan` · `episodeProfile` · `smoothstep` · `largestOk` | `_episode_plan` · `_episode_profile` · `_smoothstep` · `_largest_ok` |
 * | `stepCaps` · `withinCaps` · `applyIrregularity` | `_step_caps` · `_within_caps` · `_apply_irregularity` |
 * | `ensureRegain` · `postTPlain` | `_ensure_regain` · `_post_t_plain` |
 * | `labelGuard` · `noRegainUntil` · `stepsWithin` · `reversalVisible` · `reversalGain` · `slopeKgWeek` | `_label_guard` · `_no_regain_until` · `_steps_within` · `_reversal_visible` · `_reversal_gain` · `_slope_kg_week` |
 * | `maxWeeklyDelta` | `_max_weekly_delta` |
 * | `deriveOutcome` · `labelSeries` | `haenv/gates.py:derive_outcome` · `label_series` (7-reading centred median, floor, `reading`) |
 * | `weightOverlay` · `pchipDaily` · `resampleToGrid` | `_weight_overlay` · `_pchip_daily` · `_resample_to_grid` (the LLM generator's path: PCHIP split at T, a held point at T when T is not a model point) |
 * | `pySum` | CPython's float `sum()` (compensated) |
 * | `adherencePts` | the `traj` segment inside `haenv/build.py:premise_spec` |
 * | `clinicalSeries` | `haenv/build.py:render_clinical` |
 * | `quantizeWithinSlope` | `haenv/build.py:_quantize_within_slope` |
 * | `effectAt` | `haenv/drug_effects.py:effect_at` + `direct_effect` + `total_effect` + `applies_to` |
 * | `kineticFraction` | `haenv/drug_effects.py:_kinetic_fraction` |
 * | `responseOf` | `haenv/drug_effects.py:response_for` (per-person draw looked up from `PER.response`) |
 *
 * The skeleton's uniforms (`build._skeleton_draws`) depend on `case_id` only and are looked
 * up from `PER.skel`; its constants (`SKEL_*`) and the label rule come from
 * `kernel.skel` and `kernel.label_rule`. The weight observation noise
 * (`haenv/physio/noise.py:weight_noise_shaped`) is not drawn: the page shows the true course.
 *
 * SYNTHETIC data, evaluation use only, not medical advice.
 */
(function () {
  "use strict";

  /* ── Python's `round`: **exact-decimal** fixed-point rounding, ties-to-even ─────────
     JS's `Math.round(v*100)/100` and `toFixed` both disagree with CPython on ties
     (the former also introduces an extra floating-point error). The golden vector is
     compared point-by-point, and a one-digit difference flips a check red ⇒ match the
     convention exactly here. `toFixed(nd+60)` is an **exact expansion** at this page's
     scale (weight ~10² · labs ~10⁰): a double's fractional part is at most 52 decimal
     digits. */
  function pyRound(v, nd) {
    if (!isFinite(v)) return v;
    const neg = v < 0, a = Math.abs(v);
    const s = a.toFixed(Math.min(100, nd + 60));
    const dot = s.indexOf(".");
    const keep = s.slice(0, dot) + s.slice(dot + 1, dot + 1 + nd);
    const rest = s.slice(dot + 1 + nd);
    let n = BigInt(keep || "0");
    if (rest.length) {
      const half = "5" + "0".repeat(rest.length - 1);
      if (rest > half || (rest === half && n % 2n === 1n)) n += 1n;
    }
    const out = Number(n) / Math.pow(10, nd);
    return neg ? -out : out;
  }

  /* ── Shape (build._shape): normalized exponential, `f(0)=0 · f(1)=1`, `k=0` degenerates
     to a straight line ── */
  function shape(x, k) {
    x = Math.min(1, Math.max(0, x));
    if (k <= 1e-9) return x;
    return (1 - Math.exp(-k * x)) / (1 - Math.exp(-k));
  }

  /* ── This case's **feasible** bend coefficient (build._feasible_k): keeps the exponential
     segment's peak slope within the physiological budget. A case whose loss is pinned to
     the cap solves out to k≈0 (it could only be a straight line anyway); a case with room
     to spare gets the full k. */
  function feasibleK(trendPerDay, budgetPerDay, kWant) {
    if (kWant <= 1e-9 || trendPerDay <= 1e-12) return kWant;
    const allow = budgetPerDay / trendPerDay;
    if (allow <= 1.0) return 0.0;
    if (kWant / (1 - Math.exp(-kWant)) <= allow) return kWant;
    let lo = 0, hi = kWant;
    for (let i = 0; i < 40; i++) {
      const mid = (lo + hi) / 2;
      const m = mid <= 1e-9 ? 1.0 : mid / (1 - Math.exp(-mid));
      if (m <= allow) lo = mid; else hi = mid;
    }
    return lo;
  }

  /** Sampling step: the `max(1, round(7/measure_per_week))` branch inside `premise_spec`. */
  function stepOf(P) {
    return Math.max(1, pyRound(7.0 / Math.max(0.1, P.measure_per_week || 7), 0));
  }

  /** Adherence line (build.premise_spec). Drops further when `driver=poor_medication_adherence`
   *  and the outcome is a regain. */
  function adherencePts(P) {
    const T = P.T, ce = P.course_end_day, low = P.adherence_low;
    const mid = T + Math.floor((ce - T) / 2);
    if (P.driver === "poor_medication_adherence" && P.outcome === "regain") {
      return [[42, 0.95], [56, 0.88], [T, 0.80],
              [mid, pyRound((0.80 + low) / 2, 2)], [ce, low]];
    }
    return [[42, 0.96], [T, 0.94], [mid, 0.93], [ce, Math.max(0.90, low)]];
  }

  /* ── CPython's `sum()` of floats (3.12+): Neumaier-compensated, so a plain left-to-right
     loop can differ in the last bits and flip a guard decision. */
  function pySum(a) {
    let s = 0.0, c = 0.0;
    for (const x of a) {
      const t = s + x;
      if (Math.abs(s) >= Math.abs(x)) c += (s - t) + x; else c += (x - t) + s;
      s = t;
    }
    return (c && isFinite(c)) ? s + c : s;
  }

  const clamp01 = x => Math.min(1, Math.max(0, x));
  const argmin = vals => { let k = 0; for (let i = 1; i < vals.length; i++) if (vals[i] < vals[k]) k = i; return k; };
  const byTs = (a, b) => a[0] - b[0] || a[1] - b[1];

  /* ── Weight skeleton (build.py, "weight skeleton" section) ──────────────────────────
     Every random number comes from `PER.skel` (`build._skeleton_draws(case_id)`, exported);
     the rest is the deterministic math below, function for function. Series are arrays of
     `[day, value]`. */

  function pacePieces(pieces) {                       // build._pace_pieces
    const td = pySum(pieces.map(p => p[0])), tp = pySum(pieces.map(p => p[1]));
    const out = []; let ad = 0.0;
    for (const [d, q] of pieces) { ad += d; out.push([ad / td, (q / tp) / (d / td)]); }
    return out;
  }

  function paceAt(pieces, x) {                        // build._pace_at
    for (const [endX, pace] of pieces) if (x < endX) return pace;
    return pieces[pieces.length - 1][1];
  }

  function descentPieces(dr, S) {                     // build._descent_pieces
    const M = S.descent_max_phases;
    const k = 3 + Math.trunc(dr[0] * (M - 2));
    const paces = [];
    for (let i = 0; i < k; i++) paces.push(0.35 + 1.3 * dr[1 + i]);
    for (let j = 0; j < Math.trunc(dr[1 + M] * 3); j++)
      paces[1 + Math.trunc(dr[2 + M + j] * (k - 2))] = S.stall_pace;
    return pacePieces(paces.map(q => [1.0, q]));
  }

  function reboundPieces(dr, S) {                     // build._rebound_pieces
    const pieces = [];
    for (let i = 0; i < 2 + Math.trunc(dr[1] * 2); i++) {
      const rise = 0.25 + 0.2 * dr[2 + i], share = 0.6 + 0.8 * dr[5 + i];
      pieces.push([rise, share * (1.0 - S.hold_pace)], [1.0 - rise, share * S.hold_pace]);
    }
    return pacePieces(pieces);
  }

  /** build._paced_progress: re-pace a 0..1 progress curve, water-filled under `cap`. */
  function pacedProgress(plain, pieces, cap) {
    const n = plain.length - 1;
    if (n <= 0) return plain.slice();
    let inc = [];
    for (let d = 0; d < n; d++) inc.push((plain[d + 1] - plain[d]) * paceAt(pieces, (d + 0.5) / n));
    let mx = -Infinity;
    for (let d = 0; d < n; d++) mx = Math.max(mx, plain[d + 1] - plain[d]);
    cap = Math.max(cap, mx);
    const fixed = new Array(n).fill(false);
    for (let it = 0; it < n + 1; it++) {
      const free = pySum(inc.filter((v, i) => !fixed[i]));
      const room = 1.0 - pySum(inc.filter((v, i) => fixed[i]));
      if (free <= 0.0) break;
      inc = inc.map((v, i) => fixed[i] ? v : v * room / free);
      const over = [];
      for (let i = 0; i < n; i++) if (!fixed[i] && inc[i] > cap) over.push(i);
      if (!over.length) break;
      for (const i of over) { inc[i] = cap; fixed[i] = true; }
    }
    const out = [0.0]; let acc = 0.0;
    for (let i = 0; i < inc.length - 1; i++) { acc += inc[i]; out.push(Math.min(1.0, acc)); }
    out.push(1.0);
    return out;
  }

  function smoothstep(x) { x = clamp01(x); return x * x * (3.0 - 2.0 * x); }

  function episodePlan(ep, budgetDay, mass, S) {      // build._episode_plan
    const kind = ep[0] < 0.75 ? "up" : "down";
    const [lo, hi] = S.gain_frac;
    const h = kind === "up" ? mass * (lo + (hi - lo) * ep[1]) : 0.3 + 0.5 * ep[1];
    let rise = Math.max(2 + Math.trunc(4 * ep[2]),
                        Math.ceil(1.5 * h / Math.max(1e-9, S.episode_pace_frac * budgetDay)));
    rise = Math.min(rise, Math.floor(S.episode_max_days / 3));
    const n = Math.min(S.episode_max_days, rise + Math.ceil(rise * (2.0 + 3.0 * ep[3])));
    return { kind: kind, height: h, rise: rise, n: n,
             residual: S.gain_residual[0] + (S.gain_residual[1] - S.gain_residual[0]) * ep[5],
             tau: S.gain_tau_days[0] + (S.gain_tau_days[1] - S.gain_tau_days[0]) * ep[6],
             valley: S.valley_min_days + Math.trunc(25 * ep[4]) };
  }

  function episodeProfile(plan, length) {             // build._episode_profile
    const { rise, n, residual: r, tau } = plan, out = [];
    for (let i = 0; i <= length; i++) {
      if (i <= rise) out.push(smoothstep(i / rise));
      else if (i <= n) out.push(1.0 - (1.0 - r) * smoothstep((i - rise) / (n - rise)));
      else out.push(r * Math.exp(-(i - n) / tau));
    }
    return out;
  }

  function largestOk(ok) {                            // build._largest_ok
    if (ok(1.0)) return 1.0;
    let lo = 0.0, hi = 1.0;
    for (let i = 0; i < 20; i++) { const mid = (lo + hi) / 2.0; if (ok(mid)) lo = mid; else hi = mid; }
    return lo;
  }

  function driftSeries(draws, endDay, budgetDay, S) { // build._drift_series
    const dr = draws.drift;
    const tau = S.drift_tau_days[0] + (S.drift_tau_days[1] - S.drift_tau_days[0]) * dr[0];
    const sd = S.drift_sd_kg[0] + (S.drift_sd_kg[1] - S.drift_sd_kg[0]) * dr[1];
    const a = Math.exp(-7.0 / tau);
    const stepCap = S.drift_slope_frac * budgetDay * 7.0 / 1.5;
    if (Math.floor(endDay / 7) + 1 > dr.length - 2)
      throw new Error("course of " + endDay + " days is longer than the drift draws cover");
    let x = 0.0; const knots = [0.0];
    for (let k = 1; k < Math.floor(endDay / 7) + 2; k++) {
      const u = dr[2 + (k - 1)];
      const shock = sd * Math.sqrt(1.0 - a * a) * Math.sqrt(3.0) * (2.0 * u - 1.0);
      x += Math.max(-stepCap, Math.min(stepCap, (a - 1.0) * x + shock));
      const bound = x >= 0.0 ? S.drift_bound_kg : S.drift_bound_down_kg;
      knots.push(bound * Math.tanh(x / bound));
    }
    const out = [];
    for (let d = 0; d <= endDay; d++) {
      const i = Math.floor(d / 7);
      out.push(knots[i] + (knots[i + 1] - knots[i]) * smoothstep((d % 7) / 7.0));
    }
    return out;
  }

  function stepCaps(ref, budgetDay) {                 // build._step_caps
    const out = [];
    for (let d = 0; d < ref.length - 1; d++) out.push(Math.max(budgetDay, Math.abs(ref[d + 1] - ref[d])) + 1e-9);
    return out;
  }

  function withinCaps(vals, caps, lo, hi) {           // build._within_caps
    lo = Math.max(0, lo); hi = Math.min(vals.length - 1, hi);
    for (let d = lo; d < hi; d++) if (!(Math.abs(vals[d + 1] - vals[d]) <= caps[d])) return false;
    return true;
  }

  /** build._apply_irregularity: drift, then episodes, into `base` in place. */
  function applyIrregularity(base, draws, budgetDay, floor, T, endDay, o, S) {
    const caps = stepCaps(base, budgetDay);
    const drift = driftSeries(draws, endDay, budgetDay, S);
    const dT = drift[Math.min(T, endDay)];
    const target = d => {
      if (d <= T) return o.pre_drift ? drift[d] : 0.0;
      const fade = smoothstep((d - T) / S.fade_days);
      if (o.pre_drift && o.post_drift) return drift[d];
      if (o.pre_drift) return dT * (1.0 - fade);
      if (o.post_drift) return drift[d] - dT * (1.0 - fade);
      return 0.0;
    };
    const ref = base.slice();
    let off = target(0);
    base[0] = ref[0] + off;
    for (let d = 1; d <= endDay; d++) {
      let step = ref[d] - ref[d - 1] + target(d) - off;
      step = Math.max(-caps[d - 1], Math.min(caps[d - 1], step));
      off = base[d - 1] + step - ref[d];
      base[d] = ref[d] + off;
    }
    const added = [];
    let t = 7 + Math.trunc(28 * draws.episodes[0][4]);
    for (const ep of draws.episodes) {
      if (t >= endDay) break;
      const plan = episodePlan(ep, budgetDay, base[0], S);
      const n = plan.n, t0 = t, t1 = Math.min(endDay, t + n);
      t = t + n + plan.valley;
      if ((t0 <= T && T <= t1 + 1) || (!o.post_eps && t0 > T) || (!o.pre_eps && t1 < T)
          || (o.post_t_until != null && t0 > T && t1 >= o.post_t_until)) continue;
      let h = plan.height;
      if (plan.kind === "down") {
        h = Math.min(h, 0.5 * (Math.min(...base.slice(t0, t1 + 1)) - floor));
        if (h < 0.05) continue;
      }
      const prof = episodeProfile(plan, endDay - t0);
      const sign = plan.kind === "down" ? -1.0 : 1.0;
      const withScale = s => {
        const v = base.slice();
        for (let i = 0; i < Math.min(endDay, t1 + 1) - t0 + 1; i++) v[t0 + i] += sign * s * h * prof[i];
        return v;
      };
      const s = largestOk(s => withinCaps(withScale(s), caps, t0 - 1, t1 + 1));
      if (s * h < 0.05) continue;
      for (let i = 0; i < endDay - t0 + 1; i++) base[t0 + i] += sign * s * h * prof[i];
      added.push({ kind: plan.kind, t0: t0, t1: t1, height: pyRound(s * h, 4),
                   residual: pyRound(plan.residual, 3) });
    }
    for (let d = 1; d <= endDay; d++) {
      const step = Math.max(-caps[d - 1], Math.min(caps[d - 1], base[d] - base[d - 1]));
      base[d] = base[d - 1] + step;
    }
    return { episodes: added, caps: caps };
  }

  /** build._ensure_regain: make a declared regain hold after `max(revDay, T)`. */
  function ensureRegain(base, revDay, T, endDay, budgetDay, declaredGain, declaredLost, caps, nadir, S, R) {
    const out = { state: "n/a", dip_kg: 0.0, lift_kg: 0.0, end_over_declared_kg: 0.0 };
    caps = caps || stepCaps(base, budgetDay);
    const before = base.slice();
    const done = state => {
      out.state = state;
      const s0_ = Math.min(Math.max(revDay, T) + 1, endDay);
      out.dip_kg = pyRound(Math.max(0.0, Math.min(...before.slice(0, s0_)) - Math.min(...base.slice(0, s0_))), 3);
      let lift = -Infinity;
      for (let i = 0; i < base.length; i++) lift = Math.max(lift, base[i] - before[i]);
      out.lift_kg = pyRound(Math.max(0.0, lift), 3);
      out.end_over_declared_kg = pyRound(base[endDay] - (base[0] - declaredLost + declaredGain), 3);
      return out;
    };
    if (declaredGain < Math.max(R.min_change_frac * Math.max(0.0, declaredLost), R.min_change_kg))
      return done("declared_below_rule");
    const s0 = Math.max(revDay, T) + 1;
    if (s0 >= endDay) return done("no_room");
    let tgt = base[0] - S.settle_kg;
    if (nadir != null) tgt = Math.max(tgt, nadir - S.anchor_margin_kg + 0.05);
    const room = s0 - 1 - T;
    if (room >= S.settle_days && Math.min(...base.slice(0, s0)) > tgt) {
      for (let width = S.settle_days; width <= room; width++) {
        const end = s0 - 1 - Math.min(28, room - width);
        const trial = base.slice();
        for (let i = 0; i <= width; i++) {
          const d = end - width + i;
          trial[d] -= Math.max(0.0, trial[d] - tgt) * Math.sin(Math.PI * i / width);
        }
        if (withinCaps(trial, caps, end - 1 - width, end + 1)) { for (let i = 0; i < base.length; i++) base[i] = trial[i]; break; }
      }
    }
    const low = Math.min(...base.slice(0, s0));
    const lost = Math.max(0.0, base[0] - low);
    const gap = Math.max(R.min_change_frac * lost, R.min_change_kg);
    const runAbove = (vals, level) => {
      let best = 0, run = null;
      for (let d = s0; d <= endDay; d++) {
        if (vals[d] >= level) { run = run === null ? d : run; best = Math.max(best, d - run); }
        else run = null;
      }
      return best;
    };
    for (const [marginKg, need] of [[S.regain_margin_kg, R.min_persist_days + S.regain_margin_days],
                                    [0.1, R.min_persist_days + 1]]) {
      const level = low + gap + marginKg;
      if (runAbove(base, level) >= need) return done("held");
      const rise = Math.max(0.0, level - base[s0 - 1]);
      let width = Math.max(14, Math.ceil(1.5 * rise / Math.max(1e-9, 0.5 * budgetDay)));
      if (endDay - (s0 + width) < need)
        width = Math.max(Math.ceil(1.5 * rise / Math.max(1e-9, budgetDay)), endDay - s0 - need);
      if (endDay - (s0 + width) < need) continue;
      const b0 = base[s0 - 1];
      for (let d = s0; d <= endDay; d++) base[d] = Math.max(base[d], b0 + rise * smoothstep((d - s0 + 1) / width));
      return done("lifted");
    }
    return done("no_room");
  }

  /** build._skeleton_base: the daily three-segment skeleton; level 4 is the plain one. */
  function skeletonBase(start, nadir, descEnd, revDay, slope, endDay, kDesc, kReb, budgetDay,
                        draws, level, T, prePace, S) {
    const span = revDay !== null ? Math.max(1, endDay - revDay) : 1;
    const endV = revDay !== null ? nadir + slope * span / 7.0 : nadir;
    const drop = nadir - start, dHi = Math.min(descEnd, endDay);
    const plain = [];
    for (let d = 0; d <= endDay; d++) {
      if (d <= descEnd) plain.push(start + drop * shape(d / Math.max(1, descEnd), kDesc));
      else if (revDay === null || d <= revDay) plain.push(nadir);
      else plain.push(nadir + (endV - nadir) * shape((d - revDay) / span, kReb));
    }
    const meta = { level: level, n_stall: 0, stalls: [], rebound: "gradual", episodes: [] };
    if (level >= 4) return [plain, meta];
    const base = plain.slice();
    if (prePace && Math.abs(drop) > 1e-9 && dHi >= 2) {
      const pieces = descentPieces(draws.descent, S);
      const cur = [];
      for (let d = 0; d <= dHi; d++) cur.push(shape(d / Math.max(1, descEnd), kDesc));
      const prog = pacedProgress(cur, pieces, budgetDay / Math.abs(drop));
      for (let d = 0; d <= dHi; d++) base[d] = start + drop * prog[d];
      const stalls = []; let loX = 0.0;
      for (const [endX] of pieces) {
        const lo = pyRound(loX * dHi, 0), hi = pyRound(endX * dHi, 0);
        loX = endX;
        if (hi > lo && (prog[hi] - prog[lo]) * 3.0 < (plain[hi] - plain[lo]) / drop) stalls.push([lo, hi]);
      }
      meta.n_stall = stalls.length; meta.stalls = stalls;
    }
    if (level === 0 && revDay !== null && T <= revDay && revDay < endDay) {
      const V = S.rebound_variants;
      const variant = V[Math.trunc(draws.rebound[0] * V.length)];
      meta.rebound = variant;
      const us = [];
      for (let d = revDay; d <= endDay; d++) us.push((d - revDay) / span);
      let frac;
      if (variant === "accelerating") frac = us.map(u => 0.5 * (1.0 - shape(1.0 - u, kReb)) + 0.5 * shape(u, kReb));
      else if (variant === "steps") frac = pacedProgress(us.map(u => shape(u, kReb)), reboundPieces(draws.rebound, S),
                                                         budgetDay / Math.max(1e-9, Math.abs(endV - nadir)));
      else frac = us.map(u => shape(u, kReb));
      for (let i = 0, d = revDay; d <= endDay; d++, i++) if (d > descEnd) base[d] = nadir + (endV - nadir) * frac[i];
    }
    return [base, meta];
  }

  function postTPlain(base, plain, T, budgetDay) {    // build._post_t_plain
    const out = base.slice();
    if (T >= base.length - 1) return out;
    const off = base[T] - plain[T];
    const width = Math.max(28, Math.ceil(3.0 * Math.abs(off) / Math.max(1e-9, budgetDay)));
    for (let d = T + 1; d < base.length; d++) out[d] = plain[d] + off * (1.0 - smoothstep((d - T) / width));
    return out;
  }

  function noRegainUntil(plain, outcome) {            // build._no_regain_until
    if (outcome === "regain" || !plain.length) return null;
    return plain[argmin(plain.map(q => q[1]))][0];
  }

  function stepsWithin(pts, maxWeekly, upto) {        // build._steps_within
    const v = pts.filter(q => upto == null || q[0] <= upto).map(q => [q[0], q[1]]).sort(byTs);
    for (let i = 1; i < v.length; i++)
      if (!(Math.abs(v[i][1] - v[i - 1][1]) <= maxWeekly * (v[i][0] - v[i - 1][0]) / 7.0 + 1e-9)) return false;
    return true;
  }

  function slopeKgWeek(pts, a, z) {                   // build._slope_kg_week
    const v = pts.filter(q => a <= q[0] && q[0] <= z);
    if (v.length < 5) return null;
    let sd = 0; for (const q of v) sd += q[0];
    const mx = sd / v.length, my = pySum(v.map(q => q[1])) / v.length;
    const sxx = pySum(v.map(q => (q[0] - mx) * (q[0] - mx)));
    return sxx ? 7.0 * pySum(v.map(q => (q[0] - mx) * (q[1] - my))) / sxx : null;
  }

  function reversalGain(pts, revDay) {                // build._reversal_gain
    const b = slopeKgWeek(pts, revDay - 28, revDay), a = slopeKgWeek(pts, revDay, revDay + 28);
    return b === null || a === null ? null : a - b;
  }

  function reversalVisible(pts, revDay, ref, S) {     // build._reversal_visible
    if (revDay === null) return true;
    const gain = reversalGain(pts, revDay);
    if (gain === null) return true;
    let need = S.reversal_slope_gain;
    const rg = ref ? reversalGain(ref, revDay) : null;
    if (rg !== null) need = Math.min(need, rg);
    return gain >= need - 1e-9;
  }

  /* ── The label rule (gates.derive_outcome / label_series) ──────────────────────────── */
  function median(w) {
    w = w.slice().sort((a, b) => a - b);
    const n = w.length;
    return n % 2 ? w[(n - 1) / 2] : 0.5 * (w[n / 2 - 1] + w[n / 2]);
  }

  /** gates.label_series: the series the rule reads, smoothed as `rule.smoothing` says. */
  function labelSeries(pts, rule) {
    pts = pts.map(q => [q[0], q[1]]).sort((a, b) => a[0] - b[0]);
    const sm = String((rule || {}).smoothing || "");
    if (sm === "rolling_median_7pt") {
      const v = pts.map(q => q[1]), n = v.length;
      return pts.map((q, i) => [q[0], median(v.slice(Math.max(0, i - 3), Math.min(n, i + 4)))]);
    }
    if (sm === "rolling_median_7d") {
      return pts.map(q => [q[0], median(pts.filter(p => Math.abs(p[0] - q[0]) <= 3).map(p => p[1]))]);
    }
    return pts;
  }

  /** gates.derive_outcome on one weight series: `[verdict, detail]`. A rise of
   *  `max(frac × lost, floor)` above the low point, held for `min_persist_days`; with less
   *  lost than the floor it reads as a gain above the low point (`reading`). */
  function deriveOutcome(pts, rule) {
    if (pts.length < 3) return [null, { reason: "weight has too few points (" + pts.length + ")" }];
    const frac = Number(rule.min_change_frac), persist = Number(rule.min_persist_days);
    const floorKg = rule.min_change_kg == null ? 0.0 : Number(rule.min_change_kg);
    const s = labelSeries(pts, rule);
    const start = s[0][1], iN = argmin(s.map(q => q[1])), nadir = s[iN][1];
    const lost = Math.max(0.0, start - nadir);
    const thresh = nadir + Math.max(frac * lost, floorKg);
    const hasLoss = lost > 0 || floorKg > 0;
    let best = 0, run = null;
    for (const q of s.slice(iN)) {
      if (q[1] >= thresh && hasLoss) { run = run === null ? q[0] : run; best = Math.max(best, q[0] - run); }
      else run = null;
    }
    const occurred = hasLoss && best >= persist;
    return [occurred ? "event_occurred" : "event_not_occurred",
            { start: pyRound(start, 2), nadir: pyRound(nadir, 2), lost: pyRound(lost, 2),
              threshold: pyRound(thresh, 2), sustained_days: best, need_days: persist,
              frac: frac, floor_kg: floorKg, smoothing: String(rule.smoothing || "") || "none",
              reading: floorKg > 0 && lost < floorKg ? "gain_from_low_point" : "rebound" }];
  }

  /** The rule the shipped packs carry (`latent_rules`), from the exported kernel. */
  function labelRule(K) {
    const R = K.label_rule;
    return { min_change_frac: R.min_change_frac, min_change_kg: R.min_change_kg,
             smoothing: R.smoothing, min_persist_days: R.min_persist_days };
  }

  /** build._label_guard: `ok(series, upto)` — the rendered series keeps the label rule as
   *  `plain` does, with the noise margins, and keeps the declared low point. */
  function labelGuard(plain, T, outcome, nadir, K) {
    const S = K.skel, R = K.label_rule, rule = labelRule(K);
    const read = (series, upto, shift) => {
      if (upto != null) series = series.filter(q => q[0] <= upto);
      const [got, det] = deriveOutcome(series, rule);
      if (got === null || det.threshold === undefined) return [got, 0];
      const vals = labelSeries(series, rule);
      const iMin = argmin(vals.map(q => q[1]));
      const line = Number(det.threshold) + shift;
      let run = null, best = 0;
      for (const q of vals.slice(iMin)) {
        if (q[1] >= line) { run = run === null ? q[0] : run; best = Math.max(best, q[0] - run); }
        else run = null;
      }
      return [got, best];
    };
    const anchorOk = (series, upto) => {
      if (nadir == null) return true;
      const vals = series.filter(q => upto == null || q[0] <= upto).map(q => q[1]);
      const pvals = plain.filter(q => upto == null || q[0] <= upto).map(q => q[1]);
      let line = nadir - S.anchor_margin_kg;
      if (pvals.length) line = Math.min(line, Math.min(...pvals));
      return !vals.length || Math.min(...vals) >= line - 1e-9;
    };
    return (series, upto) => {
      if (!anchorOk(series, upto)) return false;
      const [got, low] = read(series, upto, -S.run_margin_kg);
      const [pGot, pLow] = read(plain, upto, -S.run_margin_kg);
      if (upto != null) return got === pGot && low <= Math.max(S.max_run_days, pLow);
      const declared = outcome === "regain" ? "event_occurred" : "event_not_occurred";
      if (got !== pGot && got !== declared) return false;
      if (outcome !== "regain") return low <= Math.max(S.max_run_days, pLow);
      const high = read(series, null, S.regain_run_margin_kg)[1];
      const pHigh = read(plain, null, S.regain_run_margin_kg)[1];
      return high >= Math.min(R.min_persist_days, pHigh);
    };
  }

  function maxWeeklyDelta(disease, K) {               // build._max_weekly_delta
    const w = (K.domain[disease] || {}).weight || {};
    return w.max_weekly_delta == null ? 1.5 : Number(w.max_weekly_delta);
  }

  /** build._weight_render: the deterministic path. Pre-T choices (descent pacing, drift,
   *  episodes) read pre-T readings only; the guard levels after it change post-T days only. */
  function weightRender(start, nadir, T, outcome, revWeek, slope, caseId, step, disease, endDay,
                        nadirDay, K, PER) {
    const S = K.skel, R = K.label_rule;
    step = Math.max(1, Math.trunc(step));
    const revDay = outcome === "regain" ? revWeek * 7 : null;
    let descEnd = revDay === null ? (nadirDay ? Math.trunc(nadirDay) : T) : Math.max(7, Math.min(T, revDay));
    const mw = maxWeeklyDelta(disease, K), frac = K.descent_budget_frac;
    const need = Math.abs(nadir - start) / Math.max(1e-9, mw / 7.0 * frac);
    descEnd = Math.max(descEnd, Math.ceil(need));
    const trend = Math.max(Math.abs(nadir - start) / Math.max(1, descEnd), Math.abs(slope) / 7.0);
    const budget = Math.max(0.0, mw / 7.0 * 0.9 - trend);
    const amp = Math.min(K.amp_cap, budget * K.wobble_period / (2 * Math.PI));
    const budgetDay = mw / 7.0 * frac;
    const kDesc = feasibleK(Math.abs(nadir - start) / Math.max(1, descEnd), budgetDay, PER.descent_k);
    const kReb = feasibleK(Math.abs(slope) / 7.0, budgetDay, K.rebound_k);

    // AR(1) wobble as `render_stream`; the shock table starts at k = -1, so k reads shock[k+1].
    const shock = PER.shock, phi = Math.pow(K.ar_phi, Math.max(1, step));
    const sd = K.ar_sd_frac * amp, sigma = sd * Math.sqrt(Math.max(0, 1 - phi * phi));
    const bound = sigma * Math.sqrt(3.0);
    let dropped = 0;
    const emit = base => {
      const pts = []; let gap = 0, x = 0, wk = 0; dropped = 0;
      for (let d = 0; d <= endDay; d += step) {
        if (d === 0) { x = sd * shock[0]; wk = 0; } else { wk += 1; x = phi * x + bound * shock[wk + 1]; }
        const keep = step !== K.weight_step || d === 0 || gap >= K.weight_max_gap || PER.weighed[d] === 1;
        if (!keep) { gap += 1; dropped += 1; continue; }
        gap = 0;
        pts.push([d, pyRound(base[d] + x, 2)]);
      }
      return pts;
    };
    const draws = PER.skel;
    const args = [start, nadir, descEnd, revDay, slope, endDay, kDesc, kReb, budgetDay, draws];
    const plain = skeletonBase(...args, 4, T, true, S)[0];
    const plainPts = emit(plain);
    const ruleOk = labelGuard(plainPts, T, outcome, nadir, K);
    const ok = (series, upto) => ruleOk(series, upto) && stepsWithin(series, mw, upto)
      && (upto != null || reversalVisible(series, revDay, plainPts, S));
    const until = noRegainUntil(plainPts, outcome);
    const compose = (level, pre, postDrift, postEps) => {
      let [base, meta] = skeletonBase(...args, Math.min(level, 1), T, pre.pace, S);
      if (level >= 4) base = postTPlain(base, plain, T, budgetDay);
      const irr = applyIrregularity(base, draws, budgetDay, nadir, T, endDay,
        { pre_drift: pre.drift, post_drift: postDrift, pre_eps: pre.eps, post_eps: postEps,
          post_t_until: until }, S);
      meta.episodes = irr.episodes;
      const rg = revDay !== null
        ? ensureRegain(base, revDay, T, endDay, budgetDay, slope * Math.max(1, endDay - revDay) / 7.0,
                       start - nadir, irr.caps, nadir, S, R)
        : { state: "n/a" };
      meta.regain = rg.state; meta.regain_detail = rg;
      meta.level = level; meta.pre = Object.assign({}, pre);
      return [emit(base), meta];
    };
    let pre = { pace: false, drift: false, eps: false };
    for (const key of ["pace", "drift", "eps"]) {
      const trial = Object.assign({}, pre, { [key]: true });
      if (ok(compose(1, trial, trial.drift, trial.eps)[0], T)) pre = trial;
    }
    let pts, meta;
    for (const [level, pd, pe] of [[0, true, true], [1, true, true], [2, true, false],
                                   [3, false, false], [4, false, false]]) {
      [pts, meta] = compose(level, pre, pd, pe);
      if (ok(pts)) break;
    }
    meta.guard_ok = ok(pts);
    pts.meta = Object.assign(meta, { descEnd: descEnd, revDay: revDay, amp: amp, kDesc: kDesc,
                                     kReb: kReb, step: step, dropped: dropped, trend: trend });
    return pts;
  }

  /** The page's weight trajectory: `build._weight_series` for the knobs in `P`. */
  function weightSeries(P, K, PER) {
    return weightRender(P.start, P.nadir, P.T, P.outcome, P.reversal_week, P.regain_slope,
                        P.case_id, stepOf(P), P.disease, P.course_end_day, null, K, PER);
  }

  /* ── The LLM generator's path (build._weight_overlay) ──────────────────────────────── */

  /** build._resample_to_grid: model points linearly onto the grid's days, 3 decimals;
   *  outside the model's range the nearest end value. */
  function resampleToGrid(grid, model) {
    const src = model.map(q => [Math.trunc(q[0]), Number(q[1])]).sort((a, b) => a[0] - b[0]);
    if (src.length < 2 || !grid.length) return grid;
    const xs = src.map(q => q[0]), ys = src.map(q => q[1]);
    const at = t => {
      if (t <= xs[0]) return ys[0];
      if (t >= xs[xs.length - 1]) return ys[ys.length - 1];
      let i = 0; while (xs[i] < t) i++;
      if (xs[i] === t) return ys[i];
      const x0 = xs[i - 1], x1 = xs[i], y0 = ys[i - 1], y1 = ys[i];
      return y0 + (y1 - y0) * (t - x0) / Math.max(1, x1 - x0);
    };
    return grid.map(g => [Math.trunc(g[0]), pyRound(at(Math.trunc(g[0])), 3)]);
  }

  /** build._pchip_daily: Fritsch–Carlson through `src` over days 0..end; a stretch that
   *  would step faster than `max(its chord slope, cap)` stays linear. */
  function pchipDaily(src, end, cap) {
    const xs = src.map(q => q[0]), ys = src.map(q => q[1]), n = xs.length;
    const h = [], dl = [];
    for (let k = 0; k < n - 1; k++) { h.push(xs[k + 1] - xs[k]); dl.push((ys[k + 1] - ys[k]) / h[k]); }
    const der = [dl[0]].concat(new Array(Math.max(0, n - 2)).fill(0.0), [dl[dl.length - 1]]);
    for (let k = 1; k < n - 1; k++) {
      if (dl[k - 1] * dl[k] > 0.0) {
        const w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1];
        der[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k]);
      }
    }
    const out = [];
    for (let d = 0; d <= end; d++) out.push(d <= xs[0] ? ys[0] : d >= xs[n - 1] ? ys[n - 1] : null);
    for (let k = 0; k < n - 1; k++) {
      const x0 = xs[k], x1 = xs[k + 1], y0 = ys[k], y1 = ys[k + 1];
      let seg = [];
      for (let d = x0; d <= x1; d++) {
        const t = (d - x0) / h[k];
        seg.push((2 * Math.pow(t, 3) - 3 * Math.pow(t, 2) + 1) * y0
                 + (Math.pow(t, 3) - 2 * Math.pow(t, 2) + t) * h[k] * der[k]
                 + (-2 * Math.pow(t, 3) + 3 * Math.pow(t, 2)) * y1
                 + (Math.pow(t, 3) - Math.pow(t, 2)) * h[k] * der[k + 1]);
      }
      const limit = Math.max(Math.abs(dl[k]), cap) + 1e-9;
      let bad = false;
      for (let i = 0; i < seg.length - 1; i++) if (Math.abs(seg[i + 1] - seg[i]) > limit) { bad = true; break; }
      if (bad) { seg = []; for (let d = x0; d <= x1; d++) seg.push(y0 + dl[k] * (d - x0)); }
      for (let i = 0, d = x0; d <= x1; d++, i++) if (0 <= d && d <= end) out[d] = seg[i];
    }
    return out;
  }

  /** build._weight_overlay: a shape-preserving curve through the model's points, split at
   *  T (when T is not a model point, a point holding the last pre-T value is added there),
   *  then drift, episodes and the regain check, with the same guard. `linPts` is the grid
   *  resampled linearly (`resampleToGrid`); `model` the model's `[day, value]` points. */
  function weightOverlay(linPts, model, caseId, T, outcome, nadir, disease, revWeek,
                         declaredGain, declaredLost, K, PER) {
    const S = K.skel, R = K.label_rule;
    const src = new Map();
    for (const q of model || []) { const x = Math.trunc(q[0]); if (!src.has(x)) src.set(x, Number(q[1])); }
    let srcL = [...src.entries()].sort((a, b) => a[0] - b[0]);
    if (srcL.length < 2 || linPts.length < 3) { const o = linPts.slice(); o.meta = { level: 5, episodes: [], guard_ok: true }; return o; }
    const beforeT = srcL.filter(p => p[0] < T);
    if (!src.has(T) && beforeT.length && srcL[srcL.length - 1][0] > T) {
      src.set(T, beforeT[beforeT.length - 1][1]);
      srcL = [...src.entries()].sort((a, b) => a[0] - b[0]);
      linPts = resampleToGrid(linPts, srcL);
    }
    const budgetDay = maxWeeklyDelta(disease, K) / 7.0 * K.descent_budget_frac;
    const grid = linPts.map(q => [Math.trunc(q[0]), Number(q[1])]).sort(byTs);
    const end = grid[grid.length - 1][0];
    const lin = []; let j = 0;
    for (let d = 0; d <= end; d++) {
      while (j + 1 < grid.length && grid[j + 1][0] <= d) j++;
      const [x0, y0] = grid[j], [x1, y1] = grid[Math.min(j + 1, grid.length - 1)];
      lin.push(d <= x0 || x1 === x0 ? y0 : y0 + (y1 - y0) * (d - x0) / (x1 - x0));
    }
    const pch = lin.slice();
    for (const piece of [srcL.filter(p => p[0] <= T), srcL.filter(p => p[0] >= T)]) {
      if (piece.length >= 2) {
        const daily = pchipDaily(piece, end, budgetDay);
        const lo = piece[0][0], hi = piece[piece.length - 1][0];
        for (let d = Math.max(0, lo); d <= Math.min(end, hi); d++) pch[d] = daily[d];
      }
    }
    const splitP = srcL.find(p => p[0] >= T);
    const split = splitP ? splitP[0] : end;
    const draws = PER.skel;
    const until = noRegainUntil(linPts, outcome);
    const revDay = outcome === "regain" && revWeek != null ? Math.trunc(revWeek) * 7 : null;
    const compose = (pre, post) => {
      const base = [];
      for (let d = 0; d <= end; d++) base.push(((d <= split ? pre.pchip : post.pchip) ? pch : lin)[d]);
      const info = applyIrregularity(base, draws, budgetDay, nadir, T, end,
        { pre_drift: pre.drift, post_drift: post.drift, pre_eps: pre.eps, post_eps: post.eps,
          post_t_until: until }, S);
      const rg = revDay !== null
        ? ensureRegain(base, revDay, T, end, budgetDay, declaredGain, declaredLost, info.caps, nadir, S, R)
        : { state: "n/a" };
      return [linPts.map(q => [q[0], pyRound(base[Math.trunc(q[0])], 3)]),
              { episodes: info.episodes, regain: rg.state, regain_detail: rg }];
    };
    const ruleOk = labelGuard(linPts, T, outcome, nadir, K);
    const mw = maxWeeklyDelta(disease, K);
    const ok = (series, upto) => ruleOk(series, upto) && stepsWithin(series, mw, upto)
      && (upto != null || reversalVisible(series, revDay, linPts, S));
    const off = { pchip: false, drift: false, eps: false };
    let pre = Object.assign({}, off);
    for (const key of ["pchip", "drift", "eps"]) {
      const trial = Object.assign({}, pre, { [key]: true });
      if (ok(compose(trial, trial)[0], T)) pre = trial;
    }
    let out, info, level;
    for (const [lv, post] of [[0, { pchip: true, drift: true, eps: true }],
                              [2, { pchip: true, drift: true, eps: false }],
                              [3, { pchip: true, drift: false, eps: false }], [4, off]]) {
      level = lv; [out, info] = compose(pre, post);
      if (ok(out)) break;
    }
    out.meta = Object.assign({ level: level, pre: pre, guard_ok: ok(out) }, info);
    return out;
  }

  /* ── Drug effect (drug_effects) ───────────────────────────────────────────────── */
  function resolveDrug(drug, K) {
    const d = String(drug || "");
    for (const pre of Object.keys(K.drugs || {})) if (d.startsWith(pre)) return pre;
    return null;
  }

  /** `(total effect on the signal's registry field, trial weight change)`, or `null`
   *  (drug_effects.total_effect). A dose ladder uses the highest rung not above the dose. */
  function totalEffect(spec, doseMg, field) {
    let tot = spec[field], dw = spec.trial_weight_kg;
    const ladder = spec.dose_ladder || {};
    const keys = Object.keys(ladder);
    if (keys.length && doseMg != null) {
      const nums = keys.map(Number).sort((a, b) => a - b);
      const ok = nums.filter(k => k <= Number(doseMg) + 1e-9);
      const pick = ok.length ? ok[ok.length - 1] : nums[0];
      const kk = keys.find(k => Number(k) === pick) || keys[0];
      tot = ladder[kk][field]; dw = ladder[kk].trial_weight_kg;
    }
    if (tot == null || dw == null) return null;     // can't isolate a direct term => no effect
    return [tot, dw];
  }

  /** Fraction of the full effect reached on `day` (drug_effects._kinetic_fraction): two
   *  first-order stages, drive -> fasting glucose -> HbA1c. */
  function kineticFraction(sig, day, drive, K) {
    const kg = 1 - Math.pow(0.5, 1 / K.half_time_days.glucose);
    const ka = 1 - Math.pow(0.5, 1 / K.half_time_days.HbA1c);
    let g = 0, a = 0;
    for (let d = 1; d <= Math.trunc(day); d++) {
      g += kg * (drive(d) - g);
      a += ka * (g - a);
    }
    return sig === "fasting_glucose" ? g : a;
  }

  /** `adherence` is a function `day -> [0, 1]` or a constant, as in production. */
  function effectAt(drug, sig, day, adherence, perKg, atten, doseMg, cohort, K, response) {
    // `weight` carries the gold label; production's `FORBIDDEN` throws here too.
    if ((K.drug_forbidden || []).indexOf(sig) >= 0)
      throw new Error(sig + " carries the gold label, drug effects may not be applied to it");
    const field = (K.effect_fields || {})[sig];
    if (!field) return 0;
    if ((K.drug_cohorts || []).indexOf(String(cohort)) < 0) return 0;  // cohort not in the trial population
    const pre = resolveDrug(drug, K); if (pre === null) return 0;
    const got = totalEffect(K.drugs[pre], doseMg, field); if (!got) return 0;
    const direct = got[0] - Number(perKg) * Number(atten) * got[1];
    const adh = typeof adherence === "function" ? adherence : () => Number(adherence);
    const drive = d => d <= K.onset_days ? 0 : Math.max(0, Math.min(1, Number(adh(d))));
    return direct * kineticFraction(sig, day, drive, K) * (response == null ? 1 : Number(response));
  }

  /** This person's drug-response multiplier (drug_effects.response_for). The per-person
   *  draw depends only on `case_id`; production maps it for every drug and for the
   *  non-response band, and the page looks the value up. */
  function responseOf(P, K, PER) {
    const lo = K.response_bounds.low, hi = K.response_bounds.responder;
    const low = P.driver === K.low_response_driver;
    if (P.drug_response != null) {
      const v = Number(P.drug_response);
      const ok = low ? (lo[0] <= v && v <= lo[1]) : (hi[0] <= v && v <= hi[1]);
      if (!ok) throw new Error("drug_response=" + v + " conflicts with driver=" + P.driver);
      return v;
    }
    const R = PER.response || {};
    if (low) return R.low;
    const pre = resolveDrug(P.drug, K);
    return pre === null ? 1.0 : R.by_drug[pre];
  }

  /** Round to `ndigits` while keeping the weekly-slope limit
   *  (build._quantize_within_slope). */
  function quantizeWithinSlope(prev, v, day, mwd, nd) {
    const step = nd ? 1 / Math.pow(10, nd) : 1.0;
    let q = pyRound(v, nd || 0);
    if (!prev) return q;
    const gap = Math.max(1e-9, (day - prev[0]) / 7);
    const n = Math.floor(mwd * gap / step + 1e-9);
    const p = prev[1];
    q = Math.min(Math.max(q, p - n * step), p + n * step);
    return pyRound(q, nd || 0);
  }

  /* ── Clinical signal (build.render_clinical): attenuation + lag + drug x adherence x
     response + clamping + slope-limited rounding ── */
  function clinicalSeries(sig, spec, weightPts, endDay, P, K, PER) {
    if (!weightPts.length) return [];
    const w0 = weightPts[0][1], adh = adherencePts(P);
    const adhAt = day => { let v = null; for (const q of adh) if (q[0] <= day) v = q[1];
                           return v === null ? 1.0 : v; };
    const wAt = day => { const d = Math.max(0, day - K.clinical_lag_days);
                         let prev = weightPts[0];
                         for (const p of weightPts) if (p[0] <= d) prev = p;
                         return prev[1]; };
    const lo = spec.range[0], hi = spec.range[1], mwd = spec.max_weekly_delta;
    const cv = (K.clinical_cv || {})[sig] || 0;
    const meas = ((PER.meas || {})[sig]) || [];
    const resp = responseOf(P, K, PER);
    const valueAt = (day, k) => {
      let v = spec.base + spec.per_kg * K.clinical_atten * (wAt(day) - w0);
      v += effectAt(P.drug, sig, day, adhAt, spec.per_kg, K.clinical_atten,
                    P.dose_mg, spec.cohort, K, resp);
      if (cv) v *= 1 + cv * Math.sqrt(3) * (meas[k] || 0);       // indexed by draw, not by day
      return Math.min(Math.max(v, lo), hi);
    };
    const pts = [];
    for (let k = 0, day = 0; day <= endDay; day += spec.step, k++) {
      pts.push([day, quantizeWithinSlope(pts[pts.length - 1], valueAt(day, k), day, mwd,
                                         spec.ndigits || 0)]);
    }
    // A final sample on `endDay`, which a coarse step would otherwise miss. The draw index
    // is the count before appending, as `_det_shock(..., len(pts))` in production.
    if (pts.length && pts[pts.length - 1][0] < endDay) {
      pts.push([endDay, quantizeWithinSlope(pts[pts.length - 1], valueAt(endDay, pts.length),
                                            endDay, mwd, spec.ndigits || 0)]);
    }
    return pts;
  }

  /** Which lab streams this patient has: (disease domain ∩ what the devices can produce) −
   *  weight. Same definition as `clinical_plan`. */
  function signalsFor(P, K) {
    const dom = K.domain[P.disease] || {};
    const producible = new Set();
    (P.devices || []).forEach(dev => ((K.by_device || {})[dev] || []).forEach(s => producible.add(s)));
    return Object.keys(dom).filter(s => s !== "weight" && producible.has(s)
                                        && (K.clinical_spec || {})[s]).sort();
  }

  /** This signal's full spec. **Every field except `base` comes from an exported kernel
   *  constant** — nothing here is guessed. In production `base` is sampled per cohort
   *  (`_indicators.sample_case`); on the page it's turned into a knob instead. */
  function specFor(sig, P, K, base) {
    const dom = (K.domain[P.disease] || {})[sig] || {};
    const cs = (K.clinical_spec || {})[sig] || {};
    return { base: base, per_kg: cs.per_kg, step: cs.step, ndigits: cs.ndigits,
             cohort: (cs.cohort || {})[P.disease], unit: dom.unit, range: dom.range,
             max_weekly_delta: dom.max_weekly_delta };
  }

  /* ── Gold label: the thing the model has to get right on this case ──────────────────── */
  function gold(P) {
    const rebound = P.outcome === "regain";
    return {
      outcome_label: rebound ? "event_occurred" : "no_event",
      reversal_points: rebound ? [{ week: P.reversal_week, day: P.reversal_week * 7 }] : [],
      answerable_at_T: rebound ? P.reversal_week * 7 > P.T : null,
    };
  }

  /* ── Carry-forward: copies some points verbatim from the previous value ─────────────
       This is **post-processing** (`post_inject.carried_forward`), it never touches the
       world layer's ground truth. Which points get chosen uses this patient's own
       `weighed` table as the deterministic sequence, rather than drawing new randomness. */
  function carryForward(pts, rate, PER) {
    const out = pts.map(p => [p[0], p[1], 0]);
    if (!(rate > 0)) return out;
    let acc = 0;
    for (let i = 1; i < out.length; i++) {
      acc += rate;
      if (acc >= 1 && PER.weighed[out[i][0] % PER.weighed.length] === 1) {
        acc -= 1; out[i][1] = out[i - 1][1]; out[i][2] = 1;
      }
    }
    return out;
  }

  /* ── Parameters ⇄ job.yaml ──────────────────────────────────────────────────── */
  //: `lang` only changes the one `#` comment line — every other line is the literal
  //  job.yaml file format (field names, enum values) and is not UI prose, so it never
  //  changes with the page's language toggle. Defaults to "en" to match the page default.
  function toJobYaml(P, lang) {
    const L = [];
    L.push(lang === "zh"
      ? "# 由 demo 页导出。跑法:haenv build <这个文件> --gen deterministic"
      : "# Exported from the demo page. Run: haenv build <this file> --gen deterministic");
    L.push("job_id: my-first-case");
    L.push("task_type: joint_dx");
    L.push("sample_cases: 1");
    L.push("models: []");
    L.push("cases:");
    L.push("- case_id: " + P.case_id);
    L.push("  raw:");
    L.push("    disease: " + P.disease);
    L.push("    sex: " + P.sex);
    L.push("    age_range: " + P.age_range);
    L.push("    devices:");
    (P.devices || []).forEach(d => L.push("    - " + d));
    L.push("    drug: " + P.drug);
    L.push("    dose_steps: [" + (P.dose_steps || [P.dose_mg]).join(", ") + "]");
    L.push("    start_weight: " + P.start);
    L.push("    nadir_weight: " + P.nadir);
    L.push("    sampling_days: " + stepOf(P));
    L.push("  latent:");
    L.push("    index_time_T: " + P.T);
    L.push("    course_end_day: " + P.course_end_day);
    L.push("    outcome: " + P.outcome);
    L.push("    driver: " + P.driver);
    L.push("    reversal_week: " + P.reversal_week);
    L.push("    regain_slope: " + P.regain_slope);
    L.push("    adherence_low: " + P.adherence_low);
    L.push("    event_density:");
    L.push("      measure_per_week: " + P.measure_per_week);
    return L.join("\n") + "\n";
  }

  //: The fields `parseJobYaml` recognizes. **A whitelist, not a wildcard** — unrecognized
  //  keys are reported to the user as-is, never guessed. The page lists both "recognized N"
  //  and "didn't recognize these".
  const YAML_FIELDS = {
    disease: "s", sex: "s", age_range: "s", drug: "s",
    start_weight: "n", nadir_weight: "n", sampling_days: "n",
    index_time_T: "n", course_end_day: "n", outcome: "s", driver: "s",
    reversal_week: "n", regain_slope: "n", adherence_low: "n",
    measure_per_week: "n", case_id: "s",
  };

  function parseJobYaml(text) {
    const got = {}, unknown = [];
    String(text || "").split(/\r?\n/).forEach(raw => {
      const line = raw.replace(/#.*$/, "");
      const m = line.match(/^\s*-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$/);
      if (!m) return;
      const k = m[1], v = m[2].trim().replace(/^["']|["']$/g, "");
      if (!(k in YAML_FIELDS)) { if (v !== "" && unknown.indexOf(k) < 0) unknown.push(k); return; }
      if (v === "") return;
      got[k] = YAML_FIELDS[k] === "n" ? Number(v) : v;
    });
    return { fields: got, unknown: unknown };
  }

  /* ── Chart-note text → parameters (keyword extraction) ───────────────────────────────
     This is keyword extraction with visible rules; in production a model extracts
     structured facts. The patterns match Chinese medical vocabulary because the sample chart
     note (`SAMPLE_EMR`) is benchmark content and stays in Chinese in both page languages. */
  const LEX = [
    { k: "disease", v: "T2D", re: /2\s*型糖尿病|Ⅱ型糖尿病|T2DM|type\s*2\s*diabetes/i },
    { k: "disease", v: "obesity", re: /肥胖|超重|obesity/i },
    { k: "sex", v: "F", re: /女性?[,，。\s]|女[,，]\s*\d+\s*岁|female/i },
    { k: "sex", v: "M", re: /男性?[,，。\s]|男[,，]\s*\d+\s*岁|male/i },
    { k: "drug", v: "semaglutide", re: /司美格鲁肽|semaglutide|诺和/i },
    { k: "drug", v: "tirzepatide", re: /替尔泊肽|tirzepatide/i },
    { k: "drug", v: "dulaglutide", re: /度拉糖肽|dulaglutide/i },
    { k: "drug", v: "liraglutide", re: /利拉鲁肽|liraglutide/i },
    { k: "drug", v: "metformin", re: /二甲双胍|metformin/i },
    { k: "outcome", v: "regain", re: /体重(again|反弹|回升|复增)|再次增重|regain/i },
    { k: "outcome", v: "maintain", re: /体重(维持|保持|平稳)|未见反弹|maintain/i },
    { k: "driver", v: "poor_medication_adherence", re: /漏服|自行停药|依从性?差|未按时|停用/ },
  ];

  function fromCaseText(text) {
    const t = String(text || ""), got = {}, hits = [];
    LEX.forEach(e => { if (!(e.k in got) && e.re.test(t)) { got[e.k] = e.v; hits.push([e.k, e.v]); } });
    const age = t.match(/(\d{2})\s*岁/);
    if (age) {
      const a = Number(age[1]), lo = Math.floor(a / 5) * 5;
      got.age_range = lo + "-" + (lo + 4); hits.push(["age_range", got.age_range]);
    }
    // Weight: grab every "xx.x kg / 公斤", the largest becomes the starting weight, the
    // smallest becomes the lightest weight
    const ws = [...t.matchAll(/(\d{2,3}(?:\.\d)?)\s*(?:kg|公斤|千克)/gi)].map(m => Number(m[1]));
    if (ws.length >= 2) {
      got.start = Math.max(...ws); got.nadir = Math.min(...ws);
      hits.push(["start_weight", got.start], ["nadir_weight", got.nadir]);
    }
    const hb = t.match(/(?:HbA1c|糖化(?:血红蛋白)?)[^\d]{0,6}(\d{1,2}(?:\.\d)?)\s*%?/i);
    // The hit key `hba1c_base` matches the `k-hb` knob, so the page can translate it.
    if (hb) { got.hba1c = Number(hb[1]); hits.push(["hba1c_base", got.hba1c]); }
    const wk = t.match(/第\s*(\d{1,2})\s*周[^。；;]{0,8}(?:反弹|回升|复增|增重)/);
    if (wk) { got.reversal_week = Number(wk[1]); hits.push(["reversal_week", got.reversal_week]); }
    const fu = t.match(/随访\s*(\d{1,3})\s*(?:天|日)/);
    if (fu) { got.T = Number(fu[1]); hits.push(["index_time_T", got.T]); }
    return { fields: got, hits: hits };
  }

  /* ── Self-check: recompute the golden vector with this file, compare point-by-point ───── */
  function selfCheck(D) {
    const G = D.golden, K = D.kernel, PS = D.personas;
    if (!G || !K || !PS) return { ok: false, reason: "data.json has no golden / kernel / personas section" };
    let worstW = 0, worstC = 0, n = 0, bad = [];
    G.cases.forEach(g => {
      const P = Object.assign({}, g.params, { devices: g.params.devices });
      const PER = PS[P.case_id];
      if (!PER) { bad.push(g.label + ": no random layer for this patient"); return; }
      const w = weightSeries(P, K, PER);
      if (w.length !== g.weight.length) {
        bad.push(g.label + ": " + w.length + " points ≠ " + g.weight.length); return;
      }
      for (let i = 0; i < w.length; i++) {
        if (w[i][0] !== g.weight[i][0]) { bad.push(g.label + ": day mismatch at point " + i); break; }
        worstW = Math.max(worstW, Math.abs(w[i][1] - g.weight[i][1])); n++;
      }
      bad.push(...sameSkeleton(g.label, w.meta, g.skeleton));
      if (g.rule_readout) {
        const [verdict, det] = deriveOutcome(w, labelRule(K));
        const got = Object.assign({ verdict: verdict }, ...Object.keys(g.rule_readout)
          .filter(k => k !== "verdict").map(k => ({ [k]: det[k] })));
        if (JSON.stringify(got) !== JSON.stringify(g.rule_readout))
          bad.push(g.label + ": rule readout " + JSON.stringify(got) + " ≠ " + JSON.stringify(g.rule_readout));
      }
      Object.keys(g.clinical || {}).forEach(sig => {
        const c = g.clinical[sig];
        const got = clinicalSeries(sig, c.spec, w, P.course_end_day, P, K, PER);
        if (got.length !== c.pts.length) { bad.push(g.label + "/" + sig + ": point count mismatch"); return; }
        for (let i = 0; i < got.length; i++) {
          worstC = Math.max(worstC, Math.abs(got[i][1] - c.pts[i][1])); n++;
        }
      });
    });
    (G.overlay || []).forEach(o => {
      const PER = PS[o.args.case_id];
      if (!PER) { bad.push(o.label + ": no random layer for this patient"); return; }
      const lin = resampleToGrid(o.grid.map(d => [d, 0.0]), o.model);
      const a = o.args;
      const w = weightOverlay(lin, o.model, a.case_id, a.T, a.outcome, a.nadir, a.disease,
                              a.rev_week, a.declared_gain, a.declared_lost, K, PER);
      if (w.length !== o.weight.length) { bad.push(o.label + ": point count mismatch"); return; }
      for (let i = 0; i < w.length; i++) {
        if (w[i][0] !== o.weight[i][0]) { bad.push(o.label + ": day mismatch at point " + i); break; }
        worstW = Math.max(worstW, Math.abs(w[i][1] - o.weight[i][1])); n++;
      }
      bad.push(...sameSkeleton(o.label, w.meta, o.skeleton));
    });
    const tol = G.tol == null ? 0.005 : G.tol;
    return { ok: !bad.length && worstW <= tol && worstC <= tol,
             worstWeight: worstW, worstClinical: worstC, nPoints: n,
             nCases: G.cases.length + (G.overlay || []).length, tol: tol, bad: bad };
  }

  /** The guard's decisions (level, pass, pre-T choices, regain check) must match exactly:
   *  a close curve from a different decision is still a different patient. */
  function sameSkeleton(label, meta, want) {
    if (!want) return [];
    const got = { level: meta.level, guard_ok: meta.guard_ok, pre: meta.pre, regain: meta.regain };
    const norm = x => JSON.stringify(x, Object.keys(x).sort()) + JSON.stringify(Object.keys(x.pre || {}).sort().map(k => [k, x.pre[k]]));
    return norm(got) === norm(want) ? [] : [label + ": guard " + JSON.stringify(got) + " ≠ " + JSON.stringify(want)];
  }

  globalThis.HaenvGen = {
    pyRound, shape, feasibleK, stepOf, adherencePts, weightSeries, weightRender, weightOverlay,
    resampleToGrid, deriveOutcome, labelSeries, labelRule,
    clinicalSeries, effectAt, responseOf, kineticFraction, quantizeWithinSlope, signalsFor, specFor, gold, carryForward,
    toJobYaml, parseJobYaml, fromCaseText, selfCheck, YAML_FIELDS, LEX,
  };
})();
