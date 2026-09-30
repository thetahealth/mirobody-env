/* Runs the built page once in jsdom and checks its behaviour. Not part of pytest.
 *
 * The main check drags every knob to its other end and diffs the DOM of the live sections
 * before and after: a knob that changes nothing is reported by name. The other checks cover
 * load errors, the three entry points, the "Generate" gate, the toggles, navigation, reader
 * prose, empty cells, colour slots and i18n keys.
 *
 * Layout (overlaps, overflow, phone width) needs a real browser: see `layout_check.py`.
 *
 *     npm i -D jsdom          # or bun add -d jsdom; do not commit node_modules
 *     node web/demo/render_check.mjs
 *
 * Exit code: 0 all green · 1 some check failed · 2 missing prerequisite.
 *
 * SYNTHETIC data, evaluation use only, not medical advice.
 */
import { JSDOM } from "jsdom";
import fs from "fs";

const html = fs.readFileSync(process.env.HAENV_PAGE || new URL("./index.html", import.meta.url), "utf-8");
const errs = [];
const dom = new JSDOM(html, {
  runScripts: "dangerously", pretendToBeVisual: true,
  // A new visitor still starts light when the host device prefers dark.
  beforeParse(window){ window.matchMedia = q => ({ matches: q.includes("prefers-color-scheme:dark") }); },
});
dom.window.addEventListener("error", e => errs.push(String(e.message)));
await new Promise(r => setTimeout(r, 500));
const W = dom.window, doc = W.document;
const $ = s => doc.querySelector(s);
const $$ = s => [...doc.querySelectorAll(s)];
const fire = (n, t) => n.dispatchEvent(new W.Event(t, { bubbles: true }));
const click = n => n.dispatchEvent(new W.MouseEvent("click", { bubbles: true }));

const fail = [], out = {};
const check = (name, ok, extra) => { out[name] = extra === undefined ? ok : extra;
  if (!ok) fail.push(name); };

/* ① Zero errors on load */
check("no_js_errors", errs.length === 0, errs);

/* ② One long page: every section is in the DOM with content, and none is display:none */
const SECS = ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"];
const missing = SECS.filter(id => { const n = doc.getElementById(id);
  return !n || n.textContent.trim().length < 40; });
check("all_sections_rendered", missing.length === 0, missing);
const hidden = SECS.filter(id => {
  const st = (doc.getElementById(id) || {}).getAttribute
    ? doc.getElementById(id).getAttribute("style") || "" : "";
  return /display\s*:\s*none/.test(st);
});
check("no_section_is_hidden", hidden.length === 0, hidden);

/* ③ Self-check: the page's own generator matches haenv/build.py point-by-point */
const sc = W.HaenvGen.selfCheck(JSON.parse($("#haenv-data").textContent));
check("generator_matches_backend", sc.ok, { worstWeight: sc.worstWeight,
  worstClinical: sc.worstClinical, nPoints: sc.nPoints, bad: sc.bad });

/* ④ All three entry points change the settings. Success renders inside `.note.ok` and
   "nothing recognized" inside `.note.warn`; the check reads those classes, so it works in
   either language. */
click($("#how button[data-k='emr']"));
click($("#btn-sample")); click($("#btn-read"));
const emrOut = $("#emr-out");
check("entry_emr_extracts", !!emrOut && !!emrOut.querySelector(".note.ok"),
  ((emrOut && emrOut.textContent) || "").slice(0, 80));
click($("#how button[data-k='yaml']"));
click($("#btn-cur")); click($("#btn-parse"));
const yOut = $("#y-out");
check("entry_yaml_parses", !!yOut && !!yOut.querySelector(".note.ok"),
  ((yOut && yOut.textContent) || "").slice(0, 60));
click($("#how button[data-k='knobs']"));

/* ⑤ Before "Generate this patient" the disease-course sections show a `.gate` placeholder;
   after the click the section is repainted and the placeholder is gone. */
const gated = !!doc.querySelector("#s2 .gate");
click($("#btn-born"));
await new Promise(r => setTimeout(r, 60));
check("born_button_reveals", gated && !doc.querySelector("#s2 .gate"));

/* ⑥ Every knob changes the output */
const snap = () => SECS.slice(2, 6).map(id => doc.getElementById(id).innerHTML).join("|");
const knobs = $$("#entry input[type=range], #entry select")
  .map(n => n.id).filter(Boolean);
const dead = [];
for (const id of knobs){
  const n = doc.getElementById(id);
  if (!n || n.disabled) continue;
  const s0 = snap(), old = n.value;
  if (n.tagName === "SELECT"){
    const alt = [...n.options].find(o => o.value !== n.value);
    if (!alt) continue;
    n.value = alt.value;
  } else {
    const min = +n.min, max = +n.max, cur = +n.value;
    n.value = String(cur - min > max - cur ? min : max);
  }
  fire(n, "input"); fire(n, "change");
  if (snap() === s0) dead.push(id);
  n.value = old; fire(n, "input"); fire(n, "change");
}
check("every_knob_changes_output", dead.length === 0, { n: knobs.length, dead: dead });

/* ⑦ The two segmented-button toggles (not <input>, checked separately). Each one is tested
   from the "will regain" state, the state in which it affects the curves. */
for (const [id, label, pre] of [["b-mt", "maintained", "b-re"], ["b-dr2", "unclear", "b-re"]]){
  click(doc.getElementById(pre));                 // reset to "will regain"
  const s0 = snap(); click(doc.getElementById(id));
  check("seg_" + id + "_changes_output", snap() !== s0, label);
}
click($("#b-re")); click($("#b-dr1"));

/* ⑦b The step-2 real curves share one scale: every mini draws the same y labels and the
   same day ticks. Negative control: two auto-scaled plots of different ranges must differ. */
{
  const scale = h => JSON.stringify([$$("#" + h.id + " .axlab.y").map(n => n.textContent),
    $$("#" + h.id + " .axlab.x").map(n => n.textContent)]);
  const minis = $$("#s2 [id^='mini'].plot");
  const sc = minis.map(scale);
  check("real_curves_share_one_scale", minis.length > 1 && new Set(sc).size === 1
    && JSON.parse(sc[0])[0].length > 0, { n: minis.length, scales: [...new Set(sc)] });
  const a = doc.createElement("div"), b = doc.createElement("div");
  a.id = "neg-a"; b.id = "neg-b"; doc.body.append(a, b);
  W.plot(a, { series: [{ pts: [[0, 0], [100, -2]] }], yfmt: v => v.toFixed(0) });
  W.plot(b, { series: [{ pts: [[0, 0], [100, -12]] }], yfmt: v => v.toFixed(0) });
  check("real_curves_scale_check_negative_control", scale(a) !== scale(b), [scale(a), scale(b)]);
  a.remove(); b.remove();
}

/* ⑧ The carry-forward-rate knob (lives in section 3, not #entry) */
{
  const n = $("#k-rate");
  if (!n) { check("cf_rate_knob_exists", false); }
  else { const s0 = doc.getElementById("s3").innerHTML;
    n.value = n.value === n.max ? n.min : n.max; fire(n, "input");
    check("cf_rate_changes_output", doc.getElementById("s3").innerHTML !== s0); }
}

/* ⑧b Step 4's "today" slider moves only the cut. The course drawn behind (grey) stays
   point-for-point the same, the visible line (black) is its prefix, and step 1's "today"
   is untouched. Regenerating the course here would reshape the visible part as it is dragged. */
await new Promise(r => setTimeout(r, 60));        // step 4 draws on the next frame
{
  const n = $("#k-T2"), kT = $("#k-T");
  const paths = () => $$("#p-cut path").map(p => p.getAttribute("d"));
  if (!n || !paths().length) check("cut_slider_exists", false);
  else {
    const [g0] = paths(), t0 = kT.value, v0 = n.value, bad = [];
    for (const v of [n.min, Math.round((+n.min + +n.max) / 2), n.max]){
      n.value = String(v); fire(n, "input");
      const [g, k] = paths();
      if (g !== g0) bad.push(v + ": course redrawn");
      if (!k || !g0.startsWith(k)) bad.push(v + ": visible line is not a prefix of the course");
    }
    if (kT.value !== t0) bad.push("step 1 today changed " + t0 + " -> " + kT.value);
    n.value = v0; fire(n, "input");
    check("cut_slider_keeps_the_course", bad.length === 0, bad);
  }
}

/* ⑧c A tap on a heatmap cell shows what hover shows (touch screens have no hover). */
{
  const cell = $("#s9 .hcell"), pick = $("#s9 .heatpick");
  if (cell) click(cell);
  check("heat_tap_shows_the_cell", !!cell && !!pick && pick.textContent === cell.getAttribute("title"),
        pick ? pick.textContent : "no .heatpick");
}

/* ⑨ The theme toggle really switches, and isn't decorative */
const th0 = doc.documentElement.getAttribute("data-theme");
const bg0 = W.getComputedStyle(doc.documentElement).getPropertyValue("--background").trim();
check("default_light_on_dark_system", th0 === "light" && bg0.toUpperCase() === "#FAFAF8",
  { theme: th0, background: bg0 });
click($("#btn-theme"));
const th1 = doc.documentElement.getAttribute("data-theme");
const bg1 = W.getComputedStyle(doc.documentElement).getPropertyValue("--background").trim();
check("theme_toggles", th1 === "dark" && bg1 !== bg0 && bg1.toUpperCase() === "#0D0D0E",
  { theme: th1, background: bg1 });
check("preview_fallback_removed_when_scripts_run", !$(".preview-hint"));

/* ⑩ No <text> inside any SVG (labels would overlap on scaling) · zero external requests */
check("no_text_in_svg", $$("svg text").length === 0, $$("svg text").length);
const ext = $$("script[src],link[href],img[src],iframe").map(e => e.src || e.href)
  .filter(u => /^https?:/.test(u));
check("no_external_assets", ext.length === 0, ext);

/* ⑪ Navigation is anchors (one long page), not tab buttons. Both navs are checked: the top
   rail (narrow screens) and the side TOC (wide screens). */
const rail = $$("#rail a"), toc = $$("#toc a");
check("nav_is_anchors", rail.length === SECS.length
  && rail.every(a => (a.getAttribute("href") || "").startsWith("#")), rail.length);
check("toc_covers_every_section",
  toc.length === SECS.length
  && SECS.every(id => toc.some(a => a.getAttribute("href") === "#" + id)),
  { n: toc.length, want: SECS.length });

/* ⑫ Both navs highlight the same section together. */
{
  const id = SECS[3];
  W.eval && null;                       // don't inject a script, just call the page's own function
  const fn = W.markNav;
  if (typeof fn !== "function") check("both_navs_highlight_together", false, "markNav is missing");
  else {
    fn(id);
    const a = $(`#rail a[href="#${id}"]`), b = $(`#toc a[href="#${id}"]`);
    check("both_navs_highlight_together",
      !!a && !!b && a.getAttribute("aria-current") === "true"
      && b.getAttribute("aria-current") === "true",
      { rail: a && a.getAttribute("aria-current"), toc: b && b.getAttribute("aria-current") });
  }
}

/* ⑬ No internal jargon in the reader-facing prose, measured on the rendered text with the
   "Technical details" folds removed. Model names, batch stamps and the field-name tag on each
   item card are provenance the reader is meant to look up, so they are allowed. */
{
  const clone = doc.body.cloneNode(true);
  // Also removed: <script> (the inlined data), and <pre>/<textarea>, whose content is a
  // verbatim quote (the generated config file, a model's raw answer), not page prose.
  [...clone.querySelectorAll("details, script, style, pre, textarea")].forEach(d => d.remove());
  const seen = clone.textContent || "";
  const JARGON = ["task_type", "prediction_time_T", "build_instance", "_weight_series",
    "premise_spec", "source_metric", "evidence_ledger", "injected_manifest", "world_sha",
    "batch.json", "responses.jsonl", "eval.jsonl", "trace.jsonl", "BASELINE_NAMES",
    "leakage_probe", "reversal_points", "render_clinical", "scoring.yaml", "label_rule",
    "job.yaml", "joint_dx", "early_warning", "tracking_review", "longitudinal_data"];
  const leaked = JARGON.filter(j => seen.includes(j));
  check("no_jargon_in_rendered_body", leaked.length === 0, leaked);
}

/* ⑭ No rendered table cell is blank. A cell with nothing to show says so. */
{
  const blanks = [];
  [...doc.querySelectorAll("table")].forEach((t, i) => {
    const sec = t.closest("section");
    [...t.querySelectorAll("tr")].forEach((r, j) => {
      [...r.children].forEach((c, k) => {
        if (!c.textContent.trim()) blanks.push(`${sec ? sec.id : "?"}/table${i}/row${j}/cell${k}`);
      });
    });
  });
  check("no_blank_table_cells", blanks.length === 0, blanks.slice(0, 8));
}

/* ⑮ No unrendered markdown (`**`, backticks) in the prose. Page strings are inserted as HTML,
   so `**x**` would print literal asterisks. <pre>/<textarea> hold verbatim quotes and are
   skipped. */
{
  const clone = doc.body.cloneNode(true);
  [...clone.querySelectorAll("script, style, pre, textarea")].forEach(d => d.remove());
  const t = clone.textContent || "";
  const stars = [...t.matchAll(/\*\*[^*\n]{0,30}/g)].map(m => m[0]);
  const ticks = [...t.matchAll(/`[^`\n]{1,30}`/g)].map(m => m[0]);
  check("no_raw_markdown_in_body", stars.length === 0 && ticks.length === 0,
    stars.slice(0, 4).concat(ticks.slice(0, 4)));
}

/* ⑯ Colours and published columns follow names, never the withheld composite. */
{
  const M = W.MSLOT || {}, N = W.MSLOTS || 8;
  const names = Object.keys(M).sort();
  const wrongFor = slots => names.filter((n, i) =>
    slots[n] !== (i < N ? "var(--m" + (i + 1) + ")" : "var(--c4)"));
  const wrong = wrongFor(M);
  check("colour_follows_name", names.length > 0 && wrong.length === 0,
    { n: names.length, wrong: wrong.slice(0, 3) });
  const tampered = { ...M, [names[0]]: "var(--c4)" };
  check("colour_check_has_negative_control", wrongFor(tampered).length > 0);

  const data = JSON.parse($("#haenv-data").textContent), B = data.board || {};
  const modelNames = (B.models || []).map(r => r.solver);
  const hidden = new Set((B.withheld || []).map(r => r.metric));
  const columns = (B.dims || []).flatMap(d => d.split("+"));
  const forbidden = ["rank", "macro", "macro_all_dims", "intervals", "gate_mult", "gate_fail"];
  check("eligible_subset_has_no_composite_or_provisional_scores",
    modelNames.length > 1 && JSON.stringify(modelNames) === JSON.stringify([...modelNames].sort())
      && forbidden.every(k => !(k in B))
      && (B.models || []).every(r => Object.keys(r).sort().join() === "dims,solver")
      && columns.every(d => !hidden.has(d)),
    { modelNames, columns, hidden: [...hidden] });
  check("single_answer_and_stats_hide_unvalidated_values",
    Object.values((data.answers || {}).models || {}).every(a =>
      Object.keys(a.verdict || {}).every(d => (B.dims || []).includes(d)) && !("gate_fail" in a))
      && Object.keys(data.dim_stats || {}).every(d => !hidden.has(d)));
  const cards = $$("#s8 .dimcard");
  const hiddenCards = cards.filter(c => [...c.querySelectorAll(".tag.mono")]
    .some(t => hidden.has(t.textContent.trim())));
  check("pending_validity_is_explicit_in_step8",
    hiddenCards.length === hidden.size && hiddenCards.every(c =>
      /provisionally|初步/.test(c.querySelector('[data-field="stats"] + dd')?.textContent || "")));
  const cr = B.clinical_review || {}, a = cr.answers || {}, t = cr.trace || {};
  const s9 = $("#s9")?.textContent || "";
  check("step9_states_scope_and_clinical_review",
    /not final|并非最终结果/i.test(s9)
      && /k=1/.test(s9) && s9.includes(`${a.n_pending}/${a.n_cases}`)
      && s9.includes(`${t.n_pending}/${t.n_cases}`)
      && s9.includes(String(cr.conditions_pending)));
}

/* Per-dimension orders use the same public values; other tracks and budget stay separate. */
{
  const data = JSON.parse($("#haenv-data").textContent), before = JSON.stringify(data.board.models);
  const order = W.eval("observedOrder");
  const notes = [];
  for (const [track, board] of Object.entries(data.preliminary.tracks)){
    const sel = $("#score-track"); sel.value = track; fire(sel, "change");
    for (const dim of ["__overall__", ...board.dims]){
      const choose = $("#score-dimension"); choose.value = dim; fire(choose, "change");
      const shown = $$("#dimension-order .score-line");
      const value = r => dim === "__overall__" ? r.score : r.dims[dim];
      const count = r => dim === "__overall__" ? r.n_cases : board.samples[r.solver][dim];
      const expected = [...board.models].sort((a,b) => value(b) - value(a) || a.solver.localeCompare(b.solver));
      notes.push({track, dim, ok: shown.length === expected.length && shown.every((node,i) =>
        node.dataset.solver === expected[i].solver && Number(node.dataset.value) === value(expected[i])
        && node.querySelector(".score-name small").textContent.includes(String(count(expected[i])))) });
    }
  }
  check("both_preliminary_tracks_sort_existing_total_and_dimensions", notes.every(n => n.ok), notes);
  const provisionalOrder = W.eval("preliminaryOrder");
  let unlabeled = false;
  try { provisionalOrder({...data.preliminary.tracks.answers, not_final:false}, "__overall__"); }
  catch { unlabeled = true; }
  check("preliminary_totals_require_disclaimer_metadata", unlabeled
    && /not final|并非最终/i.test($("#preliminary-notice")?.textContent || "")
    && $("#s9").textContent.includes("dx_listed") && $("#s9").textContent.includes("noop_ok"));
  const sample = {dims:["d"], models:[{solver:"a",dims:{d:.8}},{solver:"b",dims:{d:.8}},
    {solver:"c",dims:{d:0}},{solver:"missing",dims:{d:null}}]};
  const tied = order(sample, "d");
  check("dimension_ties_zero_and_missing_are_distinct", tied.map(r => r.rank || null).join() === "1,1,3,"
    && tied[0].tied && tied[1].tied && tied[2].value === 0 && tied[3].value === null);
  let rejected = false;
  try { order(data.board, "dx_hit"); } catch { rejected = true; }
  check("eligible_order_rejects_pending_metrics_and_source_stays_unmutated", rejected && JSON.stringify(data.board.models) === before
    && JSON.stringify(W.eval("D.board.models")) === before);
  const usage = $$("#budget-usage .score-line");
  check("budget_is_descriptive_and_alphabetical", usage.length === data.tool_board.models.length
    && usage.every((node,i) => !node.dataset.rank && node.dataset.solver === data.tool_board.models[i].solver));
  const coverage = $$("#score-coverage tbody tr");
  check("coverage_explains_every_registered_item", coverage.length === data.scoring.length
    && new Set(coverage.map(row => row.dataset.metric)).size === data.scoring.length
    && coverage.every(row => [...row.cells].every(cell => cell.textContent.trim())));
  const computed = $$("#s8 .dimcard").filter(c => /quant_ok|review_macro/.test(c.querySelector(".tag.mono")?.textContent || ""));
  check("computable_metrics_are_not_pending_human_anchors", computed.length === 2
    && computed.every(c => /not required|不要求/.test(c.querySelector('[data-field="agree"] + dd').textContent)
      && !c.querySelector('[data-field="revisit"]')));
  const back = $("#score-track"); back.value = "answers"; fire(back, "change");
}

/* Board exclusion: the default is the main rule (nothing ticked, ranks shown). Ticking a failure
   reason recomputes every model on the remaining cells in the page; each precomputed
   combination must agree with production `rank_ddx` within 0.001, or the page must say so. */
{
  const data = JSON.parse($("#haenv-data").textContent);
  const gaps = {};
  const notes = [];
  const boxes = () => $$("#score-exclude input[type=checkbox]");
  check("no_answered_view_control_left", !$("#score-view") && !$("#answered-hint")
    && !("answered_view" in data.preliminary.tracks.answers));
  for (const [track, board] of Object.entries(data.preliminary.tracks)){
    const sel = $("#score-track"); sel.value = track; fire(sel, "change");
    const dimSel = $("#score-dimension"); dimSel.value = "__overall__"; fire(dimSel, "change");
    const X = board.exclusion, present = X.categories.map(c => c.key);
    notes.push({track, main: !!$("#zero-note") && !$("#exclusion-hint") && boxes().length === present.length
      && boxes().every(b => !b.checked)
      && $$("#dimension-order .score-line").every(n => n.dataset.rank)
      && $$("#dimension-order .score-line").every(n => Number(n.dataset.value) === board.models
        .find(m => m.solver === n.dataset.solver).score)});
    notes.push({track, labels: boxes().every((b,i) => b.parentNode.textContent.includes(
      W.eval("LANG") === "zh" ? X.categories[i].zh : X.categories[i].en))});
    for (const combo of X.backend){
      const off = new Set(combo.exclude);
      boxes().forEach(b => { if (b.checked !== off.has(b.dataset.category)){ b.checked = off.has(b.dataset.category); fire(b, "change"); } });
      const shown = Object.fromEntries($$("#dimension-order .score-line").map(n =>
        [n.dataset.solver, n.dataset.value === "" ? null : Number(n.dataset.value)]));
      let gap = 0;
      for (const m of X.models){
        const a = shown[m], p = combo.scores[m];
        gap = Math.max(gap, a == null && p == null ? 0 : a == null || p == null ? 1 : Math.abs(a - p));
      }
      const key = track + ":" + (combo.exclude.join("+") || "none");
      gaps[key] = Number(gap.toFixed(6));
      const diverges = !!$("#exclusion-divergence");
      const excluded = combo.exclude.length > 0;
      notes.push({key, ok: (gap <= 0.001 && !diverges) || (gap > 0.001 && diverges),
        marked: excluded ? !!$("#exclusion-hint") && !!$("#exclusion-counts")
          && !$$("#dimension-order .score-line").some(n => n.dataset.rank)
          && /剔除后各模型的病例集可能不同|case sets may differ/.test($("#exclusion-hint").textContent)
          && X.models.every(m => $("#exclusion-counts").textContent.includes(W.eval("mlabel")(m) + " "
            + X.categories.filter(c => off.has(c.key)).reduce((a,c) => a + c.cells[m], 0))) : !$("#exclusion-hint")});
    }
    boxes().forEach(b => { if (b.checked){ b.checked = false; fire(b, "change"); } });
  }
  check("board_default_is_main_rule_with_ranks_and_bilingual_reason_labels",
    notes.filter(n => "main" in n || "labels" in n).every(n => n.main !== false && n.labels !== false), notes.filter(n => "main" in n || "labels" in n));
  check("exclusion_frontend_equals_backend_within_0.001_or_page_labels_divergence",
    notes.filter(n => "key" in n).every(n => n.ok), { max_abs_gap_by_combination: gaps });
  check("exclusion_view_shows_hint_counts_and_no_ranks", notes.filter(n => "key" in n).every(n => n.marked));
  const back = $("#score-track"); back.value = "answers"; fire(back, "change");
}

/* The criterion catalog separates capability, execution method and scoring role. */
{
  const data = JSON.parse($("#haenv-data").textContent);
  const cards = $$("#s8 .dimcard"), groups = $$("#s8 details[data-judge-group]");
  const keys = data.scoring.map(m => m.key).sort();
  const actual = cards.map(c => c.dataset.metric).sort();
  const grouped = groups.flatMap(g => [...g.querySelectorAll(".dimcard")].map(c => c.dataset.metric)).sort();
  const complete = entries => entries.length === keys.length && new Set(entries).size === keys.length
    && entries.every((key, i) => key === keys[i]);
  check("criterion_catalog_covers_registry_once", complete(actual) && complete(grouped), { actual, grouped });
  check("criterion_catalog_coverage_has_negative_controls",
    !complete(actual.slice(1)) && !complete([...actual.slice(1), actual[1]].sort()));
  check("criterion_catalog_has_six_capability_groups", groups.length === 6
    && new Set(groups.map(g => g.dataset.judgeGroup)).size === 6
    && groups.every(g => g.querySelector("summary")?.textContent.trim() && g.querySelector(".dimcard")));
  const details = cards.map(c => c.querySelector("details.criterion-details"));
  check("criterion_details_start_collapsed", details.length === keys.length
    && details.every(d => d && !d.open && d.querySelector("summary")));
  if (details[0]){
    details[0].querySelector("summary").click(); const expanded = details[0].open;
    details[0].querySelector("summary").click();
    check("criterion_detail_opens_and_closes", expanded && !details[0].open);
  }
  const method = W.eval("typeof judgeMethod === 'function' ? judgeMethod : null");
  const proxy = data.scoring.filter(m => m.metric_class === "judgment" && m.key !== "rubric_binary");
  check("proxy_judgment_does_not_imply_llm", !!method && proxy.length > 0 && proxy.every(m =>
    method(m) === "code" && cards.find(c => c.dataset.metric === m.key)?.dataset.method === "code"));
  check("llm_and_reference_methods_are_distinct", !!method
    && method(data.scoring.find(m => m.key === "rubric_binary")) === "llm"
    && method(data.scoring.find(m => m.key === "scope_anchor_unified")) === "reference"
    && cards.every(c => ["code", "llm", "reference"].includes(c.dataset.method)));
  check("execution_method_explanation_and_optional_plugin_are_present",
    /LLM/.test($("#judge-methods")?.textContent || "")
    && /judgment/.test($("#judge-methods")?.textContent || "")
    && !!$("#llm-judge-note") && !$("#llm-judge-note")?.closest(".dimcard")
    && /llmj_rival_discriminator/.test($("#llm-judge-note")?.textContent || ""));
  const rubric = cards.find(c => c.dataset.metric === "rubric_binary");
  check("rubric_does_not_imply_automatic_score_promotion", !!rubric
    && !rubric.querySelector('[data-field="revisit"]')
    && /report-only|辅助观察/.test(rubric.querySelector('[data-field="counts"] + dd')?.textContent || ""));
  const execution = W.eval("judgeRunSummary");
  const saved = W.eval("D.judge_execution");
  W.eval('D.judge_execution = {answers:{rubric_rows:0,rubric_observed:true},trace:{rubric_rows:0,rubric_observed:false}}');
  const unscored = execution("rubric");
  W.eval('D.judge_execution = {answers:{rubric_rows:1,rubric_observed:true}}');
  const measured = execution("rubric");
  W.DUMMY_RESTORE_EXECUTION = saved;
  W.eval('D.judge_execution = globalThis.DUMMY_RESTORE_EXECUTION');
  delete W.DUMMY_RESTORE_EXECUTION;
  check("judge_execution_distinguishes_missing_status_and_actual_score", unscored !== measured
    && /no scored|没有可用/.test(unscored) && /1/.test(measured));
  const search = $("#judge-search"), methodSelect = $("#judge-method"), roleSelect = $("#judge-role");
  const reset = $("#judge-reset"), count = $("#judge-count"), empty = $("#judge-empty");
  const controls = !!search && !!methodSelect && !!roleSelect && !!reset && !!count && !!empty;
  check("criterion_catalog_filter_controls_exist", controls);
  if (controls){
    const visible = () => cards.filter(c => !c.hidden && !c.closest("details[data-judge-group]")?.hidden);
    const activeGroupsOpen = () => groups.filter(g => !g.hidden).every(g => g.open);
    const setSelect = (control, value) => { control.value = value; fire(control, "change"); };
    setSelect(methodSelect, "llm");
    check("llm_filter_finds_only_registered_rubric", visible().map(c => c.dataset.metric).join() === "rubric_binary"
      && activeGroupsOpen() && !/disabled|已禁用/i.test(visible()[0]?.textContent || "")
      && count.textContent.includes("1"));
    setSelect(roleSelect, "dim");
    check("method_and_role_filters_intersect", visible().length === 0 && !empty.hidden);
    click(reset);
    setSelect(methodSelect, "reference");
    check("reference_filter_keeps_normalization_separate", visible().map(c => c.dataset.metric).join() === "scope_anchor_unified"
      && activeGroupsOpen());
    click(reset);
    setSelect(roleSelect, "diagnostic");
    const diagnostics = data.scoring.filter(m => m.role === "diagnostic").map(m => m.key).sort();
    check("role_filter_uses_registered_role", JSON.stringify(visible().map(c => c.dataset.metric).sort()) === JSON.stringify(diagnostics));
    click(reset);
    search.value = "quant_ok"; fire(search, "input");
    check("criterion_search_matches_metric_key", visible().map(c => c.dataset.metric).join() === "quant_ok" && activeGroupsOpen());
    search.value = "__no_such_criterion__"; fire(search, "input");
    check("criterion_search_has_explicit_empty_state", visible().length === 0 && !empty.hidden
      && groups.every(g => g.hidden) && $$("#s8 .dimcard").length === keys.length);
    click(reset);
    check("criterion_reset_restores_full_collapsed_catalog", visible().length === keys.length
      && search.value === "" && methodSelect.value === "all" && roleSelect.value === "all"
      && empty.hidden && details.every(d => !d.open) && count.textContent.includes(String(keys.length)));
  }
}

/* ⑰ Every i18n key the page asks for exists in both languages, and no rendered `<dd>` is
   empty in either language. A missing key makes `T()` return undefined, which renders as an
   empty cell rather than an error. */
{
  const src = $$("script").map(s => s.textContent).join("\n");
  // Literal keys (`T("key"` followed by `,` or `)`), plus the nav labels built as "nav_" + id.
  const used = [...new Set([...src.matchAll(/\bT\(\s*"([A-Za-z0-9_]+)"\s*[,)]/g)].map(m => m[1])
    .concat(SECS.map(id => "nav_" + id)))];
  const UIt = W.eval("typeof UI === 'object' ? UI : null");
  const missing = !UIt ? ["UI table not found"] : [].concat(...["en", "zh"].map(l =>
    used.filter(k => !(k in (UIt[l] || {}))).map(k => l + ":" + k)));
  check("every_i18n_key_exists_in_both_languages", missing.length === 0, missing.slice(0, 12));

  const emptyDd = lang => $$("dd").filter(d => !d.textContent.trim())
    .map(d => lang + ":" + ((d.closest("section") || {}).id || "?") + ":"
      + ((d.previousElementSibling || {}).getAttribute
          ? d.previousElementSibling.getAttribute("data-field") : "?"));
  const empties = [];
  const heroCopy = [];
  for (const lang of ["en", "zh"]){
    const b = $(`#lang-toggle button[data-lang="${lang}"]`);
    if (b) click(b);
    await new Promise(r => setTimeout(r, 30));
    empties.push(...emptyDd(lang));
    const coverageFold = $("#score-coverage");
    const summary = coverageFold?.querySelector("summary");
    const closed = coverageFold?.tagName === "DETAILS" && !coverageFold.open;
    if (summary) summary.click();
    const expanded = coverageFold?.open === true && coverageFold.querySelectorAll("tbody tr").length === W.eval("D.scoring.length");
    if (summary) summary.click();
    check("coverage_fold_opens_and_closes_" + lang, closed && expanded && !coverageFold.open);
    const subtitle = lang === "en" ? "Generation and Evaluation in One Pipeline" : "生成与评测，一条流程贯通";
    heroCopy.push({ lang, ok: $("#s0 h1")?.textContent === "Health Agent Environment"
      && $("#s0 .hero-subtitle")?.textContent === subtitle
      && doc.title === "Health Agent Environment — " + subtitle });
  }
  const bEn = $('#lang-toggle button[data-lang="en"]'); if (bEn) click(bEn);
  check("no_empty_dd_in_either_language", empties.length === 0, empties.slice(0, 8));
  check("pipeline_heading_matches_browser_title_in_both_languages", heroCopy.every(x => x.ok), heroCopy);
}

console.log(JSON.stringify(out, null, 1));
if (fail.length){
  console.error("FAIL: " + fail.length + " check(s) failed: " + fail.join(" · "));
  process.exit(1);
}
console.log("ok: all green (" + Object.keys(out).length + " checks)");
