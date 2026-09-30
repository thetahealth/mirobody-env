/* Run `gen.mjs`'s golden-vector comparison headlessly. Run it after changing production
 * trajectory code.
 *
 * The page runs the same `selfCheck()` on load and prints any deviation for the reader; this
 * copy needs no browser, and its exit code is the answer:
 *
 *     node web/demo/gen_check.mjs        # 0 = matches point for point · 1 = drifted · 2 = missing input
 *
 * It checks the browser generator against the golden vectors in `data.json`.
 * `check_page_data.py` checks that those golden vectors are what production code computes now.
 * `.github/workflows/pages.yml` runs both before publishing.
 *
 * SYNTHETIC data, evaluation use only, not medical advice.
 */
import fs from "fs";
await import("./gen.mjs");
const G = globalThis.HaenvGen;
const p = new URL("./data.json", import.meta.url);
if (!fs.existsSync(p)) { console.error("data.json is missing; run export_demo_data.py first"); process.exit(2); }
const D = JSON.parse(fs.readFileSync(p, "utf-8"));
const r = G.selfCheck(D);
console.log(JSON.stringify(r, null, 1));
if (!r.ok) {
  console.error("The browser generator and haenv/build.py have drifted apart. "
    + "The correspondence table is at the top of gen.mjs; find which segment's definition "
    + "changed rather than widening the tolerance.");
  process.exit(1);
}
console.log(`ok: ${r.nCases} parameter set(s) · ${r.nPoints} point(s) matched point for point`
  + ` (worst weight deviation ${r.worstWeight} · lab ${r.worstClinical}, tolerance ${r.tol})`);
