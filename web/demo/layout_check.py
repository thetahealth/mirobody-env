#!/usr/bin/env python3
"""Real-browser layout check for `web/demo/index.html`.

Opens the built page in headless Chromium at 360, 390 and 430 px (phones), 768 px (tablet), 900,
1440 and 1600 px, in English and Chinese, light and dark (28 configurations), and fails when the
page is visibly broken:

| kind | fails when |
|---|---|
| `hscroll` | the document is wider than the viewport (the page scrolls sideways) |
| `content-alignment` | the header, sections and footer do not share their content edges |
| `overflow` | an element sticks out past the viewport edge (clipped, so it cannot be scrolled to) |
| `control-outside` | a button / link / input / summary is not fully inside the viewport, or is cut off by its own scroll box |
| `control-overlap` | two controls intersect |
| `tap-target` | at 900 px or narrower, a control is less than 40 px tall (a checkbox is measured by its label) |
| `label-overlap` | a chart text label intersects another piece of text (another label, a legend, body copy) |
| `label-on-data` | a chart text label is drawn on top of a plotted line or point |
| `truncated` | text is cut with an ellipsis and has no `title` carrying the full text |
| `wrapped-number` | a number cell or tag breaks over two lines |
| `empty-dd` | a `<dd>` has no content |
| `empty-card` | a card has no content, or starts with a tall blank band |
| `console` | the console shows an error, or the page throws |

Every configuration is checked as first rendered, with every `<details>` opened, with both chart
layers of the first act switched, on each of the four packs of the second act, and with another
model selected in the third act.

    python web/demo/layout_check.py                  # 28 configurations, exit 1 on any failure
    python web/demo/layout_check.py --shots DIR      # also save a full-page PNG per configuration
    python web/demo/layout_check.py --selftest       # negative controls: each defect kind is injected
                                                     # into a copy of the page and must be reported

Needs `playwright` (Python) and a Chromium. `HAENV_CHROME` points at a browser binary;
otherwise Playwright's own bundle is used, then `chromium` / `google-chrome` on PATH.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PAGE = HERE / "index.html"

WIDTHS = (360, 390, 430, 768, 900, 1440, 1600)
LANGS = ("en", "zh")
SCHEMES = ("light", "dark")

AUDIT_JS = r"""
() => {
  const vw = document.documentElement.clientWidth;
  const fails = [];
  const add = (kind, detail) => fails.push({ kind, detail });
  const R = e => e.getBoundingClientRect();
  const shown = e => {
    // content inside a closed <details> still reports a box in Chromium; checkVisibility knows better
    if (!e.checkVisibility({ contentVisibilityAuto: true, opacityProperty: true, visibilityProperty: true })) return false;
    const r = R(e);
    return r.width >= 1 && r.height >= 1;
  };
  const name = e => {
    const t = (e.getAttribute("aria-label") || e.textContent || e.getAttribute("title") || "").trim();
    return e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + ' "' + t.replace(/\s+/g, " ").slice(0, 32) + '"';
  };
  const where = e => { const s = e.closest("section"); return s ? s.id || "" : (e.closest("header") ? "header" : ""); };
  const inter = (a, b, m) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > m
                          && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > m;
  // Boxes that are meant to scroll sideways. Anything inside them may extend past the viewport.
  const SCROLLERS = ".tablewrap, .scroll";
  const scroller = e => {
    for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
      const ox = getComputedStyle(p).overflowX;
      if (ox === "auto" || ox === "scroll" || ox === "hidden" || ox === "clip") return p;
    }
    return null;
  };

  // hscroll
  const sw = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
  if (sw > vw + 1) add("hscroll", "document is " + sw + " px wide in a " + vw + " px viewport");

  // Compare content edges, not box widths: the header/footer include page padding.
  const contentEdges = e => {
    const r = R(e), c = getComputedStyle(e);
    return [r.left + parseFloat(c.paddingLeft) + parseFloat(c.borderLeftWidth),
            r.right - parseFloat(c.paddingRight) - parseFloat(c.borderRightWidth)];
  };
  const header = document.querySelector("header .wrap");
  if (header) {
    const expected = contentEdges(header);
    document.querySelectorAll("main section > .wrap, footer.wrap").forEach(e => {
      const actual = contentEdges(e);
      if (actual.some((x, i) => Math.abs(x - expected[i]) > 1))
        add("content-alignment", name(e) + " edges " + actual.map(Math.round)
            + " differ from header " + expected.map(Math.round));
    });
  }

  // overflow: outermost element past the viewport edge that is not inside a clipping box
  const all = [...document.body.querySelectorAll("*")];
  const over = new Set();
  all.forEach(e => {
    if (!shown(e)) return;
    const r = R(e);
    if (r.right <= vw + 1 && r.left >= -1) return;
    if (scroller(e)) return;
    if (getComputedStyle(e).position === "fixed") return;
    if ([...over].some(o => o.contains(e))) return;
    over.add(e);
    add("overflow", name(e) + " in " + where(e) + " spans " + Math.round(r.left) + ".." + Math.round(r.right) + " of " + vw);
  });

  // controls
  const ctrls = [...document.querySelectorAll("button, a[href], input, select, textarea, summary, [role=button]")]
    .filter(shown).map(e => {
      let r = R(e); const sc = scroller(e);
      let clipped = false;
      if (sc) {
        const c = R(sc);
        if (!sc.matches(SCROLLERS) && (r.left < c.left - 1 || r.right > c.right + 1)) clipped = true;
        r = { left: Math.max(r.left, c.left), right: Math.min(r.right, c.right),
              top: Math.max(r.top, c.top), bottom: Math.min(r.bottom, c.bottom) };
      }
      return { e, r, clipped };
    }).filter(c => c.r.right - c.r.left >= 1 && c.r.bottom - c.r.top >= 1);
  ctrls.forEach(c => {
    if (c.clipped) add("control-outside", name(c.e) + " in " + where(c.e) + " is cut off by its container");
    else if (c.r.left < -1 || c.r.right > vw + 1)
      add("control-outside", name(c.e) + " in " + where(c.e) + " spans " + Math.round(c.r.left) + ".." + Math.round(c.r.right) + " of " + vw);
  });
  for (let i = 0; i < ctrls.length; i++) for (let j = i + 1; j < ctrls.length; j++) {
    const a = ctrls[i], b = ctrls[j];
    if (a.e.contains(b.e) || b.e.contains(a.e)) continue;
    if (inter(a.r, b.r, 2)) add("control-overlap", name(a.e) + " / " + name(b.e) + " in " + where(a.e));
  }
  // tap-target: on a touch-sized screen every control is at least 40 px tall
  if (vw <= 900) ctrls.forEach(c => {
    const box = c.e.matches("input[type=checkbox]") && c.e.closest("label") ? R(c.e.closest("label")) : R(c.e);
    if (box.height < 39.5) add("tap-target", name(c.e) + " in " + where(c.e) + " is " + Math.round(box.height) + " px tall");
  });

  // text rectangles, one entry per line box of every visible text node
  const CHART = ".fig, .legend";
  const texts = [];
  const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = tw.nextNode(); n; n = tw.nextNode()) {
    if (!n.textContent.trim()) continue;
    const p = n.parentElement;
    if (!p || p.closest("script, style, svg") || !shown(p)) continue;
    const rg = document.createRange(); rg.selectNodeContents(n);
    const sc = scroller(p), clip = sc ? R(sc) : null;
    [...rg.getClientRects()].forEach(r => {
      if (r.width < 1 || r.height < 1) return;
      if (clip && (r.right < clip.left || r.left > clip.right || r.bottom < clip.top || r.top > clip.bottom)) return;
      texts.push({ n, p, r, chart: !!p.closest(CHART), s: n.textContent.trim().slice(0, 24) });
    });
  }
  // label-overlap: a chart label against any other text node
  const seen = new Set();
  texts.forEach(a => {
    if (!a.chart) return;
    texts.forEach(b => {
      if (a.n === b.n) return;
      const key = a.chart && b.chart ? [a.s, b.s].sort().join("|") + Math.round(a.r.top) : a.s + "|" + b.s;
      if (seen.has(key)) return;
      // inset: glyph boxes of adjacent lines touch by a pixel or two, that is not an overlap
      const ia = { left: a.r.left, right: a.r.right, top: a.r.top + 2, bottom: a.r.bottom - 2 };
      const ib = { left: b.r.left, right: b.r.right, top: b.r.top + 2, bottom: b.r.bottom - 2 };
      if (!inter(ia, ib, 1)) return;
      seen.add(key);
      add("label-overlap", '"' + a.s + '" / "' + b.s + '" in ' + where(a.p) + " at " + Math.round(a.r.left) + "," + Math.round(a.r.top + scrollY) + " / " + Math.round(b.r.left) + "," + Math.round(b.r.top + scrollY));
    });
  });
  // label-on-data: a chart label drawn over a plotted path or point
  const marks = [];
  document.querySelectorAll(".fig svg").forEach(svg => {
    if (!shown(svg)) return;
    const pt = svg.createSVGPoint();
    svg.querySelectorAll("path").forEach(p => {
      const m = p.getScreenCTM(); if (!m) return;
      const L = p.getTotalLength(), k = Math.max(40, Math.ceil(L / 3));
      for (let i = 0; i <= k; i++) {
        const q = p.getPointAtLength(L * i / k); pt.x = q.x; pt.y = q.y;
        const s = pt.matrixTransform(m); marks.push([s.x, s.y, svg]);
      }
    });
    svg.querySelectorAll("circle").forEach(c => {
      const m = c.getScreenCTM(); if (!m) return;
      pt.x = +c.getAttribute("cx"); pt.y = +c.getAttribute("cy");
      const s = pt.matrixTransform(m); marks.push([s.x, s.y, svg]);
    });
  });
  const flagged = new Set();
  texts.forEach(a => {
    if (!a.chart || flagged.has(a.n)) return;
    const r = a.r;
    const hit = marks.find(([x, y]) => x > r.left + 1 && x < r.right - 1 && y > r.top + 2 && y < r.bottom - 2);
    if (hit) { flagged.add(a.n); add("label-on-data", '"' + a.s + '" in ' + where(a.p)); }
  });

  // truncated
  all.forEach(e => {
    if (!shown(e)) return;
    const cs = getComputedStyle(e);
    if (cs.textOverflow !== "ellipsis" || e.scrollWidth <= e.clientWidth + 1) return;
    if (e.getAttribute("title")) return;
    add("truncated", name(e) + " in " + where(e));
  });

  // wrapped-number
  document.querySelectorAll("td.cell, .knob b, .gold b").forEach(e => {
    if (!shown(e) || !e.textContent.trim()) return;
    const lh = parseFloat(getComputedStyle(e).lineHeight) || 20;
    const rg = document.createRange(); rg.selectNodeContents(e);
    const tops = new Set([...rg.getClientRects()].filter(r => r.width > 0).map(r => Math.round(r.top)));
    if (tops.size > 1 && R(e).height > lh * 1.5) add("wrapped-number", name(e) + " in " + where(e));
  });

  // empty-dd
  document.querySelectorAll("dd").forEach(d => {
    if (d.textContent.trim() || d.querySelector("svg, img, .dot")) return;
    const dt = d.previousElementSibling;
    add("empty-dd", "after <dt> " + (dt ? '"' + dt.textContent.trim().slice(0, 24) + '"' : "?") + " in " + where(d));
  });

  // empty-card: nothing inside, or the first visible content starts far below the top edge
  document.querySelectorAll(".card").forEach(c => {
    if (!shown(c)) return;
    if (!c.textContent.trim() && !c.querySelector("svg, input, select, textarea, img")) {
      add("empty-card", name(c) + " in " + where(c)); return;
    }
    const top = R(c).top;
    const kids = [...c.querySelectorAll("*")].filter(k => shown(k) && (k.children.length === 0 || k.matches("svg, summary")));
    if (!kids.length) return;
    const first = Math.min(...kids.map(k => R(k).top));
    if (first - top > 72) add("empty-card", "blank band of " + Math.round(first - top) + " px at the top of a card in " + where(c));
  });
  return fails;
}
"""


def chrome_exe(p) -> str | None:
    """The browser binary to launch, or None for Playwright's own bundle."""
    env = os.environ.get("HAENV_CHROME")
    if env:
        return env
    if pathlib.Path(p.chromium.executable_path).exists():
        return None
    for cand in ("chromium", "chromium-browser", "google-chrome-stable", "google-chrome"):
        exe = shutil.which(cand)
        # Ubuntu ships a `chromium-browser` stub that only prints "install the snap".
        if exe and subprocess.run([exe, "--version"], capture_output=True).returncode == 0:
            return exe
    return None


def launch(p):
    return p.chromium.launch(executable_path=chrome_exe(p),
                             args=["--no-sandbox", "--disable-dev-shm-usage"])


def audit(browser, page_path: pathlib.Path, width: int, lang: str, scheme: str,
          shot: pathlib.Path | None = None, inject: str | None = None) -> list[dict]:
    ctx = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
    pg = ctx.new_page()
    errs: list[str] = []
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.on("pageerror", lambda e: errs.append(f"uncaught: {e}"))
    pg.goto(page_path.as_uri() + f"?lang={lang}")
    pg.wait_for_load_state("load")
    pg.evaluate("() => document.fonts.ready")
    # The visitor starts light even when their OS is dark; test dark explicitly
    # instead of silently taking four light screenshots under color_scheme=dark.
    initial = pg.evaluate("""() => ({theme: document.documentElement.getAttribute('data-theme'),
        background: getComputedStyle(document.body).backgroundColor})""")
    theme_fails = []
    if initial != {"theme": "light", "background": "rgb(249, 249, 247)"}:
        theme_fails.append({"kind": "theme", "detail": f"initial {scheme}: {initial}"})
    if scheme == "dark":
        pg.evaluate("() => document.documentElement.setAttribute('data-theme', 'dark')")
        actual = pg.evaluate("""() => ({theme: document.documentElement.getAttribute('data-theme'),
            background: getComputedStyle(document.body).backgroundColor})""")
        if actual != {"theme": "dark", "background": "rgb(13, 13, 13)"}:
            theme_fails.append({"kind": "theme", "detail": f"manual dark: {actual}"})
    pg.wait_for_timeout(500)
    pg.evaluate("() => window.scrollTo(0, 0)")
    pg.wait_for_timeout(150)
    if inject:
        pg.evaluate(inject)
        pg.wait_for_timeout(100)
    fails = theme_fails + pg.evaluate(AUDIT_JS)
    if shot:
        pg.screenshot(path=str(shot), full_page=True)

    def again(js: str | None = None, wait: int = 200) -> None:
        if js:
            pg.evaluate(js)
        pg.wait_for_timeout(wait)
        fails.extend(f for f in pg.evaluate(AUDIT_JS) if f not in fails)

    again("() => document.querySelectorAll('details').forEach(d => d.open = true)")
    # the other layer state of the first act: truth only, then withheld after today
    again("""() => { const n = document.querySelector('#L-noise'), c = document.querySelector('#L-cut');
        n.checked = false; n.dispatchEvent(new Event('change', {bubbles: true}));
        c.checked = true; c.dispatchEvent(new Event('change', {bubbles: true})); }""")
    for key in ("pack2", "p3", "p4", "m2"):
        again(f"() => document.querySelector('#tabs .tab[data-p=\"{key}\"]').click()")
    again("""() => { const s = document.querySelector('#mp-model'); s.value = s.options[s.options.length - 1].value;
        s.dispatchEvent(new Event('change', {bubbles: true}));
        document.querySelectorAll('details').forEach(d => d.open = true); }""")
    fails += [{"kind": "console", "detail": e[:200]} for e in errs]
    ctx.close()
    return fails


def run_matrix(page_path: pathlib.Path, shots: pathlib.Path | None, widths=WIDTHS, langs=LANGS, schemes=SCHEMES) -> int:
    from playwright.sync_api import sync_playwright
    total = 0
    with sync_playwright() as p:
        b = launch(p)
        for w in widths:
            for lang in langs:
                for scheme in schemes:
                    tag = f"{lang}-{w}-{scheme}"
                    shot = shots / f"{tag}.png" if shots else None
                    fails = audit(b, page_path, w, lang, scheme, shot)
                    total += len(fails)
                    print(f"{'FAIL' if fails else 'ok  '} {tag:16s} {len(fails)}")
                    for f in fails:
                        print(f"     {f['kind']:16s} {f['detail']}")
        b.close()
    print(f"\n{total} layout failure(s) across {len(widths) * len(langs) * len(schemes)} configurations")
    return 1 if total else 0


#: One injected defect per failure kind. Each runs on its own fresh copy of the page and must
#: raise its own kind; `console` is injected by the copy itself (a script tag), the rest by
#: DOM edits after the page has rendered.
DEFECTS = {
    "content-alignment": """() => { document.querySelector('#act2 > .wrap').style.marginLeft = '24px'; }""",
    "label-overlap": """() => { const a = document.querySelector('#c1 .axlab');
        const b = a.cloneNode(true); b.textContent = 'overlapping label'; a.parentElement.appendChild(b);
        b.style.left = a.style.left; b.style.top = a.style.top; }""",
    "label-on-data": """() => { const svg = document.querySelector('#c1 svg'); const path = svg.querySelector('path');
        const L = path.getTotalLength(), q = path.getPointAtLength(L / 2), m = path.getScreenCTM();
        const pt = svg.createSVGPoint(); pt.x = q.x; pt.y = q.y; const s = pt.matrixTransform(m);
        const host = svg.closest('.fig'), h = host.getBoundingClientRect();
        const n = document.createElement('span'); n.className = 'axlab'; n.textContent = 'on the line';
        n.style.left = (s.x - h.left) + 'px'; n.style.top = (s.y - h.top - 7) + 'px'; host.appendChild(n); }""",
    "empty-dd": """() => { const d = document.querySelector('dd'); d.innerHTML = ''; }""",
    "overflow": """() => { const d = document.createElement('div'); d.style.width = '3000px'; d.style.height = '4px';
        document.querySelector('#act1 .wrap').appendChild(d); }""",
    "hscroll": """() => { const s = document.createElement('style');
        s.textContent = 'html,body{overflow-x:visible!important}'; document.head.appendChild(s);
        const d = document.createElement('div'); d.style.width = '3000px'; d.style.height = '4px';
        document.querySelector('#act1 .wrap').appendChild(d); }""",
    "control-outside": """() => { const b = document.createElement('button'); b.textContent = 'off screen';
        b.style.position = 'relative'; b.style.left = (innerWidth + 50) + 'px'; b.style.minHeight = '44px';
        document.querySelector('#act1 .wrap').prepend(b); }""",
    "control-overlap": """() => { const t = document.querySelector('#btn-theme'); const b = document.createElement('button');
        b.textContent = 'X'; b.style.position = 'absolute'; const r = t.getBoundingClientRect();
        b.style.left = (r.left + scrollX) + 'px'; b.style.top = (r.top + scrollY) + 'px'; b.style.zIndex = 99;
        b.style.minHeight = '44px'; document.body.appendChild(b); }""",
    "tap-target": """() => { const s = document.querySelector('#k-sex'); s.style.setProperty('min-height', '0', 'important');
        s.style.setProperty('height', '22px', 'important'); }""",
    "truncated": """() => { const s = document.createElement('span'); s.textContent = 'a label that is much too long to fit';
        Object.assign(s.style, {display: 'block', width: '60px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'});
        document.querySelector('#act1 .wrap').appendChild(s); }""",
    "wrapped-number": """() => { const t = document.querySelector('.knob b'); t.textContent = '25 sets of something';
        t.style.whiteSpace = 'normal'; t.style.display = 'inline-block'; t.style.width = '40px'; }""",
    "empty-card": """() => { const p = document.createElement('div'); p.className = 'card';
        p.style.height = '80px'; document.querySelector('#act1 .wrap').appendChild(p); }""",
    "console": None,
}


def selftest(page_path: pathlib.Path) -> int:
    from playwright.sync_api import sync_playwright
    html = page_path.read_text(encoding="utf-8")
    bad = 0
    with tempfile.TemporaryDirectory(prefix="layout-selftest-") as td:
        clean = pathlib.Path(td) / "clean.html"
        clean.write_text(html, encoding="utf-8")
        with sync_playwright() as p:
            b = launch(p)
            base = audit(b, clean, 390, "en", "light")
            base_n = {}
            for f in base:
                base_n[f["kind"]] = base_n.get(f["kind"], 0) + 1
            for kind, js in DEFECTS.items():
                copy = pathlib.Path(td) / f"{kind}.html"
                src = html
                if kind == "console":
                    src = html.replace("</body>", "") + "\n<script>console.error('injected error')</script>\n"
                copy.write_text(src, encoding="utf-8")
                fails = audit(b, copy, 390, "en", "light", inject=js)
                want = kind.split("@")[0]
                n = sum(f["kind"] == want for f in fails)
                fired = n > base_n.get(want, 0)
                bad += not fired
                print(f"{'fires' if fired else 'MISSED'}  {kind:24s} clean {base_n.get(want, 0)} -> injected {n}")
            b.close()
    print(f"\n{len(DEFECTS) - bad}/{len(DEFECTS)} injected defects reported")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--page", type=pathlib.Path, default=PAGE)
    ap.add_argument("--shots", type=pathlib.Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--widths", type=lambda v: tuple(int(x) for x in v.split(",")), default=WIDTHS,
                    help="comma-separated widths (default: the full matrix)")
    a = ap.parse_args()
    if not a.page.is_file():
        print(f"{a.page} is missing; run `python web/demo/build.py` first", file=sys.stderr)
        return 2
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("playwright is not installed: `pip install playwright && playwright install chromium`",
              file=sys.stderr)
        return 2
    if a.selftest:
        return selftest(a.page.resolve())
    if a.shots:
        a.shots.mkdir(parents=True, exist_ok=True)
    return run_matrix(a.page.resolve(), a.shots, widths=a.widths)


if __name__ == "__main__":
    raise SystemExit(main())
