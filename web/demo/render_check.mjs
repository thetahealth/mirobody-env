/* Runs the built page once in jsdom and checks its behaviour. Not part of pytest.
 *
 * The main check moves every control of the first act to another value and diffs the act's DOM
 * before and after: a control that changes nothing is reported by name. The other checks cover
 * load errors, the three acts, the generator self-check, the two chart layers, the pack switch,
 * the board numbers against `board.json`, the model switch, theme and language, and i18n keys.
 *
 * Layout (overlaps, overflow, phone width, tap targets) needs a real browser: see `layout_check.py`.
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
function load(url){
  const errs = [];
  const dom = new JSDOM(html, { url: url || "file:///index.html", runScripts: "dangerously", pretendToBeVisual: true,
    // A new visitor still starts light when the host device prefers dark.
    beforeParse(window){ window.matchMedia = q => ({ matches: q.includes("prefers-color-scheme:dark") }); } });
  dom.window.addEventListener("error", e => errs.push(String(e.message)));
  return { dom, errs };
}
const { dom, errs } = load();
await new Promise(r => setTimeout(r, 400));
const W = dom.window, doc = W.document;
const $ = s => doc.querySelector(s);
const $$ = s => [...doc.querySelectorAll(s)];
const fire = (n, t) => n.dispatchEvent(new W.Event(t, { bubbles: true }));
const click = n => n.dispatchEvent(new W.MouseEvent("click", { bubbles: true }));
const fail = [], out = {};
const check = (name, ok, extra) => { out[name] = extra === undefined ? ok : extra; if (!ok) fail.push(name); };
const data = JSON.parse($("#haenv-data").textContent), board = JSON.parse($("#haenv-board").textContent);

check("no_js_errors", errs.length === 0, errs);
const SECS = ["s0", "act1", "act2", "act3"];
const missing = SECS.filter(id => { const n = doc.getElementById(id); return !n || n.textContent.trim().length < 40; });
check("three_acts_rendered", missing.length === 0, missing);
check("preview_fallback_removed_when_scripts_run", !$(".preview-hint"));

const sc = W.HaenvGen.selfCheck(data);
check("generator_matches_backend", sc.ok && sc.nObserved > 0, { worstWeight: sc.worstWeight, worstObserved: sc.worstObserved,
  nObserved: sc.nObserved, nPoints: sc.nPoints, bad: sc.bad });

/* Act 1: the patient is drawn on load, with no gate */
check("patient_drawn_on_load", $$("#c1 svg circle").length > 50, $$("#c1 svg circle").length);
const snap1 = () => $("#act1 .chartcard").innerHTML;
const dead = [];
const knobs = $$("#knobs select, #knobs input[type=range]");
knobs.forEach(k => {
  const before = snap1(), v0 = k.value;
  if (k.tagName === "SELECT") k.value = k.options[(k.selectedIndex + 1) % k.options.length].value;
  else { const mn = +k.min, mx = +k.max; k.value = String(Math.abs(+v0 - mn) < Math.abs(+v0 - mx) ? mx : mn); }
  fire(k, "input");
  if (snap1() === before) dead.push(k.id);
  k.value = v0; fire(k, "input");
});
check("every_knob_changes_output", knobs.length >= 10 && dead.length === 0, { n: knobs.length, dead });
{
  const b0 = snap1(); click($('#k-outcome button[data-o="maintain"]'));
  const changed = snap1() !== b0; const off = $("#k-week").disabled;
  click($('#k-outcome button[data-o="regain"]'));
  check("outcome_switch_changes_output", changed && off && !$("#k-week").disabled);
}
/* The two layers share one chart */
{
  const copied = () => $$('#c1 circle[data-copied="1"]').length;
  const want = W.eval("W1").dirty.filter(p => p[2]).length;
  const n0 = copied();
  const noise = $("#L-noise"); noise.checked = false; fire(noise, "change");
  const n1 = copied(), truthOnly = $$("#c1 svg circle").length;
  noise.checked = true; fire(noise, "change");
  check("noise_layer_marks_every_copied_reading", n0 === want && want > 0 && n1 === 0, { n0, want, n1, truthOnly });
  const cut = $("#L-cut"), T0 = W.eval("P").T;
  const after = () => W.eval("W1").dirty.filter(p => p[0] > T0).length;
  const c0 = $$("#c1 svg circle").length;
  cut.checked = true; fire(cut, "change");
  const c1 = $$("#c1 svg circle").length, hatched = !!$('#c1 rect[fill="url(#h1)"]');
  cut.checked = false; fire(cut, "change");
  check("cut_layer_withholds_after_today", hatched && c1 < c0 && after() > 0, { c0, c1, hatched });
}
check("no_text_in_svg", $$("svg text").length === 0, $$("svg text").length);

/* Act 2: four packs, each switch repaints the record, the chart and the gold */
{
  const tabs = $$("#tabs .tab");
  const seen = new Set(), bad = [];
  tabs.forEach(t => { click($('#tabs .tab[data-p="' + t.dataset.p + '"]'));
    const sel = $('#tabs .tab[aria-selected="true"]');
    const sig = $("#pack").innerHTML;
    if (!sel || sel.dataset.p !== t.dataset.p || seen.has(sig) || !$$("#c2 svg circle").length || !$("#pack .goldrow").textContent.trim()) bad.push(t.dataset.p);
    seen.add(sig); });
  click($('#tabs .tab[data-p="m2"]'));
  check("four_pack_tabs_each_draw_their_item", tabs.length === 4 && bad.length === 0, { n: tabs.length, bad });
  // the pack-1 chart colours the diary entries the build attributes, and only those
  const att = data.packs.m2.viz.attribution, real = data.packs.m2.viz.diary.filter(e => e.kind === "symptom" && att[e.day]);
  const lineDots = $$("#c2 svg circle").filter(c => /accent|c2/.test(c.getAttribute("fill") + c.getAttribute("stroke")) && +c.getAttribute("r") === 6);
  check("pack1_marks_each_attributed_entry", real.length > 0 && lineDots.length === real.length, { real: real.length, drawn: lineDots.length });
}

/* Act 3: the numbers on the page are the board's */
{
  const ov = board.overall.board;
  const rows = $$("#act3 .fig .axlab.ann, #act3 .fig .axlab.annb").map(n => n.textContent).filter(t => /^\d+\s/.test(t));
  check("overall_rows_follow_the_board_order", rows.length === ov.length && rows.every((t, i) => t.endsWith(ov[i].model)), rows.slice(0, 3));
  const bad = [];
  $$("#act3 td.cell[data-k]").forEach(td => { const m = td.parentElement.dataset.m, r = board.packs[td.dataset.k].board.find(q => q.model === m);
    if (td.textContent !== r.score.toFixed(2)) bad.push(m + "/" + td.dataset.k); });
  const nCells = $$("#act3 td.cell[data-k]").length, nFinal = Object.values(board.packs).filter(p => p.status === "final").length;
  check("heat_cells_are_the_board_scores", nCells === nFinal * ov.length && bad.length === 0, { nCells, bad });
  const pend = Object.entries(board.packs).filter(([, p]) => p.status !== "final").map(([k]) => k);
  const leaked = pend.filter(k => $$("#act3 td.cell[data-k='" + k + "']").length > 0);
  check("a_pending_pack_shows_no_numbers", leaked.length === 0 && (!pend.length || $$("#act3 td.upd").length === ov.length), { pend, leaked });
  check("interim_overall_is_labelled", board.overall.status !== "interim" || !!$("#act3 .tagi"));
  const fills = $$("#act3 .fill"), badFill = fills.filter(f => { const m = f.closest("tr").dataset.m, k = f.closest("td").dataset.k;
    return Math.abs(parseFloat(f.style.width) - board.packs[k].stability[m] * 100) > 0.06; });
  check("disagreement_bars_are_the_board_shares", fills.length > 0 && badFill.length === 0, { n: fills.length, bad: badFill.length });
  const p0 = $("#mpanel").innerHTML, sel = $("#mp-model"), other = sel.options[3].value;
  sel.value = other; fire(sel, "change");
  check("model_switch_repaints_the_panel", $("#mpanel").innerHTML !== p0 && $("#act3 tr.sel").dataset.m === other);
  const cf = board.decisions.pack2, cells = cf.models[other], tot = cells.reduce((a, c) => a + c[2], 0);
  check("decision_counts_cover_every_item_and_round", tot === cf.n_items * cf.n_rounds && $$("#cf-pack2 td[data-g]").length === cf.classes.length * (cf.classes.length + 1),
    { tot, want: cf.n_items * cf.n_rounds });
}

/* Theme and chrome */
const th0 = doc.documentElement.getAttribute("data-theme");
const bg0 = W.getComputedStyle(doc.documentElement).getPropertyValue("--background").trim();
check("default_light_on_dark_system", th0 === "light" && bg0.toUpperCase() === "#F9F9F7", { th0, bg0 });
click($("#btn-theme"));
const th1 = doc.documentElement.getAttribute("data-theme");
check("theme_toggles", th1 === "dark");
click($("#btn-theme"));
const ext = $$("script[src], link[href]").map(n => n.getAttribute("src") || n.getAttribute("href")).filter(u => /^(https?:)?\/\//.test(u));
check("no_external_assets", ext.length === 0, ext);
check("nav_is_anchors", $$("#nav a").map(a => a.getAttribute("href")).join() === "#act1,#act2,#act3");

/* i18n: every key in both tables; the English page has no Chinese text; the Chinese page renders */
{
  const UI = W.eval("UI"), TECH = W.eval("TECH");
  const miss = [];
  [UI, TECH].forEach(tb => { const en = Object.keys(tb.en), zh = Object.keys(tb.zh);
    miss.push(...en.filter(k => !zh.includes(k)).map(k => "zh:" + k), ...zh.filter(k => !en.includes(k)).map(k => "en:" + k)); });
  check("every_i18n_key_exists_in_both_languages", miss.length === 0, miss.slice(0, 12));
  const cjk = /[㐀-鿿＀-￯　-〿]/;
  const walk = (root, acc) => { const tw = doc.createTreeWalker(root, W.NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n; n = tw.nextNode()) { const p = n.parentElement;
      if (!p || p.closest("script, style, #lang-toggle, .preview-hint")) continue;
      if (cjk.test(n.textContent)) acc.push(n.textContent.trim().slice(0, 30)); } return acc; };
  const shown = [];
  ["m2", "pack2", "p3", "p4"].forEach(k => { click($('#tabs .tab[data-p="' + k + '"]')); walk(doc.body, shown); });
  click($('#tabs .tab[data-p="m2"]'));
  check("english_page_shows_no_chinese_text", shown.length === 0, [...new Set(shown)].slice(0, 8));
  const z = load("file:///index.html?lang=zh");
  await new Promise(r => setTimeout(r, 400));
  const zd = z.dom.window.document;
  check("chinese_page_renders", z.errs.length === 0 && zd.documentElement.lang === "zh-CN" && cjk.test(zd.querySelector("#act2").textContent), z.errs);
}

/* A height-only resize keeps the page */
{
  const before = $("#act3").innerHTML;
  fire(W, "resize");
  await new Promise(r => setTimeout(r, 250));
  check("height_only_resize_keeps_the_page", $("#act3").innerHTML === before);
}

console.log(JSON.stringify(out, null, 1));
if (fail.length){ console.error("FAILED: " + fail.join(", ")); process.exit(1); }
console.log(`ok: ${Object.keys(out).length} checks`);
