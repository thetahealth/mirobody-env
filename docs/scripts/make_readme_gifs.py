#!/usr/bin/env python3
"""The four README animations. Generated files, do not hand-edit; re-run this script.

    python docs/scripts/make_readme_gifs.py                       # all four
    python docs/scripts/make_readme_gifs.py --only today fan      # some of them
    python docs/scripts/make_readme_gifs.py --record ~/mirobody-env   # re-record the quick start first

Needs Playwright (Python), Pillow and a Chromium; `HAENV_CHROME` points at the browser
binary, otherwise Playwright's own bundle is used. No model calls, no network.

Output in `docs/figures/`, each GIF with a PNG still of the same size (the static fallback):

* `readme_today`      demo step 4: dragging "today" across the patient built in step 1;
* `readme_fan`        demo steps 1 and 2: two course knobs move all 32 parallel futures;
* `readme_answers`    demo step 6: selecting models in the answer matrix rings the
                      evidence each one cited on the recorded case's timeline;
* `readme_quickstart` the four quick-start commands and what they print.

The three demo animations drive the shipped page `web/demo/index.html` in a headless
browser at its default settings and screenshot it; no page content or data is edited.
Besides the pointer, two of them are composed from two regions of the same page:
`readme_fan` stacks the two step-1 knob rows, on a drawn card in the page's panel colours,
above the step-2 fan; `readme_answers` stacks the step-6 timeline above the answer table and
is scaled to 960 px. Their still is the first frame. The terminal animation replays
`docs/figures/readme_quickstart.transcript.json`: every line the four commands printed, in
order, recorded under a pseudo-terminal by `--record` in a fresh clone (no `.venv`,
`results/`, `reports/` or `cases/` yet, so the first `uv run` prints its environment set-up).
Three machine-specific parts are redacted when recording: the clone's path is shown as
`~/mirobody-env`, the interpreter uv picks as `<python>` and the batch stamp the run creates as
`<batch>`. On display lines are clipped at the
terminal's right edge, playback is faster than the recording, and colour follows each line's
prefix (log level, `[gates]` or `[haenv]`). Its still is the last frame, with all four runs done.

SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = ROOT / "web" / "demo" / "index.html"
OUT = ROOT / "docs" / "figures"
TRANSCRIPT = OUT / "readme_quickstart.transcript.json"

#: The commands the README's quick start prints, in its order and spelling.
QUICKSTART = (
    "uv run haenv build  inputs/example-ew.job.yaml --gen deterministic --fresh",
    "uv run haenv verify inputs/example-ew.job.yaml --gen deterministic",
    "uv run haenv run    inputs/example-ew.job.yaml --offline",
    "uv run haenv report inputs/example-ew.job.yaml --offline",
)
CLONE = "~/mirobody-env"

FRAME_MS = 100          # 10 frames per second
MAX_BYTES = 1_000_000   # every GIF stays under this
WIDTH = (880, 1000)     # every GIF's width lies in this range

# ------------------------------------------------------------------ GIF assembly


def _pointer(img, x: float, y: float):
    """Draw an arrow pointer with its tip at (x, y)."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    pts = [(0, 0), (0, 17), (4.5, 13), (7.5, 20), (10, 19), (7, 12), (12.5, 12)]
    poly = [(x + px, y + py) for px, py in pts]
    d.polygon(poly, fill=(24, 24, 24), outline=(255, 255, 255))
    d.line(poly + [poly[0]], fill=(255, 255, 255), width=1)
    return img


def _palette(frames: list, colors: int, force: list[tuple] = ()):
    """One palette for the whole animation: the page's own colour tokens, then the most
    frequent exact colours (flat fills and text), then a median cut of the rest. Small
    accents (a legend swatch, the "today" handle) keep their hue that way."""
    from PIL import Image
    w, h = frames[0].size
    sample = frames[:: max(1, len(frames) // 12)]
    sheet = Image.new("RGB", (w, h * len(sample)), "white")
    for i, f in enumerate(sample):
        sheet.paste(f, (0, h * i))
    exact = sorted(sheet.getcolors(maxcolors=1 << 24), reverse=True)
    keep = list(dict.fromkeys(list(force) + [c for _, c in exact[: colors // 2]]))[:colors - 8]
    cut = sheet.quantize(colors=colors - len(keep), method=Image.Quantize.MEDIANCUT,
                         dither=Image.Dither.NONE).getpalette()[: 3 * (colors - len(keep))]
    rest = [tuple(cut[i:i + 3]) for i in range(0, len(cut), 3)]
    pal = list(dict.fromkeys(keep + rest))[:256]
    flat = [v for c in pal for v in c]
    img = Image.new("P", (1, 1))
    img.putpalette(flat + flat[-3:] * (256 - len(pal)))
    return img


def save_gif(frames: list, durations: list[int], path: pathlib.Path, colors: int,
             still: int = 0, force: list[tuple] = ()) -> int:
    """Write a looping GIF with one shared palette, and its still as a PNG.

    Pillow writes each frame as the rectangle that differs from the previous one, so a
    frame where only the pointer moves costs a few hundred bytes.
    """
    from PIL import Image
    pal = _palette(frames, colors, force)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0,
              optimize=False, disposal=1)
    q[still].save(path.with_suffix(".png"), optimize=True)   # same palette as the GIF
    return path.stat().st_size


# ------------------------------------------------------------------ the demo page


def _chrome(p):
    exe = os.environ.get("HAENV_CHROME")
    return p.chromium.launch(executable_path=exe or None,
                             args=["--no-sandbox", "--disable-dev-shm-usage"])


def _open(browser, width: int, height: int = 1000, dpr: int = 1):
    pg = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=dpr)
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.goto(PAGE.as_uri() + "?lang=en")
    pg.wait_for_load_state("load")
    pg.evaluate("() => document.documentElement.setAttribute('data-theme', 'light')")
    pg.locator("#btn-born").click()
    _fan_done(pg)
    # In-flow header (a sticky one would sit over the shots); no transitions or smooth
    # scrolling, so every screenshot shows a settled state.
    pg.add_style_tag(content="header{position:static!important}"
                             "*{transition:none!important;animation:none!important;caret-color:transparent}"
                             "html,body{scroll-behavior:auto!important}")
    pg.wait_for_timeout(300)
    pg._haenv_errors = errors
    return pg


def _hex(c: str) -> tuple:
    c = c.strip().lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _page_colors(pg) -> list[tuple]:
    """The light theme's colour tokens (`--accent`, the model swatches, ...), as RGB."""
    vals = pg.evaluate(
        "() => { const css = [...document.querySelectorAll('style')].map(s => s.textContent).join('\\n');"
        " const names = [...new Set([...css.matchAll(/(--[a-z0-9-]+)\\s*:/gi)].map(m => m[1]))];"
        " const cs = getComputedStyle(document.documentElement);"
        " return names.map(n => cs.getPropertyValue(n).trim()).filter(v => /^#[0-9a-f]{6}$/i.test(v)); }")
    return [_hex(v) for v in dict.fromkeys(vals)]


def _fan_done(pg):
    pg.wait_for_function(
        "() => !document.querySelector('#fan') || document.querySelector('#fan').dataset.done === '1'",
        timeout=60000)


def _box(pg, sel: str) -> dict:
    b = pg.locator(sel).first.bounding_box()
    assert b, f"{sel} is not on the page"
    return b


def _shot(pg, clip: dict):
    """Screenshot a viewport rectangle given in CSS px (the image is CSS px x DPR)."""
    from PIL import Image
    png = pg.screenshot(clip={k: round(clip[k]) for k in ("x", "y", "width", "height")},
                        animations="disabled")
    return Image.open(io.BytesIO(png)).convert("RGB")


def _scroll_top_to(pg, sel: str, top: int = 16):
    pg.evaluate("([s, t]) => { const r = document.querySelector(s).getBoundingClientRect();"
                " window.scrollBy({top: r.top - t, behavior: 'instant'}); }", [sel, top])
    pg.wait_for_timeout(200)


def _assert_clean(pg):
    errs = getattr(pg, "_haenv_errors", [])
    assert not errs, f"the demo page raised errors while being recorded: {errs[:3]}"


def _lerp(a, b, n):
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(1, n + 1)]


def gif_today(browser) -> tuple[list, list, int, list]:
    """Step 4: drag "today" from the generated day to late follow-up, and back."""
    pg = _open(browser, 1050)
    panel = "#c-body > .panel"
    _scroll_top_to(pg, panel)
    clip = _box(pg, panel)
    plot = _box(pg, "#tl-you .tl-plot")
    x1 = int(pg.evaluate("() => +document.getElementById('k-T2').max"))
    t0 = int(pg.evaluate("() => +document.getElementById('k-T2').value"))
    day_x = lambda d: plot["x"] + plot["width"] * d / x1

    def handle():
        h = _box(pg, "#tl-you .tl-t")
        return h["x"] + h["width"] / 2, h["y"] + h["height"] / 2

    frames, dur = [], []

    def grab(xy, ms):
        frames.append(_pointer(_shot(pg, clip), xy[0] - clip["x"], xy[1] - clip["y"]))
        dur.append(ms)

    h = handle()
    rest = (h[0] + 28, h[1] + 150)                # resting over the lanes: crosshair shows
    pg.mouse.move(*rest)
    grab(rest, 1500)
    for xy in _lerp(rest, h, 3):
        pg.mouse.move(*xy)
        grab(xy, FRAME_MS)
    pg.mouse.down()
    for d in [t0 + (330 - t0) * i / 18 for i in range(1, 19)]:     # the fog recedes
        pg.mouse.move(day_x(d), h[1])
        pg.wait_for_timeout(30)
        grab((day_x(d), h[1]), FRAME_MS)
    dur[-1] = 1000
    for d in [330 - (330 - t0) * i / 8 for i in range(1, 9)]:      # and comes back
        pg.mouse.move(day_x(d), h[1])
        pg.wait_for_timeout(30)
        grab((day_x(d), h[1]), FRAME_MS)
    pg.mouse.up()
    h = handle()
    for xy in _lerp(h, rest, 3):
        pg.mouse.move(*xy)
        grab(xy, FRAME_MS)
    _assert_clean(pg)
    force = _page_colors(pg)
    pg.close()
    return frames, dur, 0, force


def gif_fan(browser) -> tuple[list, list, int, list]:
    """Steps 1 and 2: move the regain week and the regain speed; all 32 futures follow."""
    from PIL import Image
    pg = _open(browser, 1050, height=3200)
    # the knobs (step 1) and the fan (step 2) have to fit in one tall viewport
    _scroll_top_to(pg, "#k-week", 60)
    knobs = pg.evaluate(
        "() => { const a = document.getElementById('k-week').closest('.knob').getBoundingClientRect(),"
        " b = document.getElementById('k-slope').closest('.knob').getBoundingClientRect();"
        " return {x: Math.min(a.left, b.left) - 2, y: a.top - 6, width: Math.max(a.right, b.right)"
        " - Math.min(a.left, b.left) + 4, height: b.bottom - a.top + 12}; }")
    fan = _box(pg, "#fan")
    lists = pg.evaluate(
        "() => { const n = [...document.querySelectorAll('#fan *')].find(e => e.children.length === 0"
        " && /same in every future/i.test(e.textContent)); return n ? n.getBoundingClientRect().top : null; }")
    fclip = dict(fan)
    if lists:
        fclip["height"] = lists - fan["y"] - 4
    assert fclip["y"] + fclip["height"] < 3200, "the fan is below the viewport"
    W, top_pad, gap = round(fclip["width"]), 12, 14
    bg = _hex(pg.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--background')"))
    kx = round(knobs["x"] - fclip["x"])
    # the regain has to start after "today"; an earlier week is a different question
    week_min = int(pg.evaluate("() => Math.ceil(+document.getElementById('k-T').value / 7)")) + 1

    def slider(sel):
        return pg.evaluate("(s) => { const r = document.getElementById(s), b = r.getBoundingClientRect();"
                           " const f = (r.value - r.min) / (r.max - r.min);"
                           " return [b.left + 10 + (b.width - 20) * f + 3, b.top + b.height / 2 + 3]; }", sel)

    frames, dur = [], []

    card = _hex(pg.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--card')"))
    line = _hex(pg.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--border')"))

    def grab(xy, ms):
        from PIL import ImageDraw
        k, f = _shot(pg, knobs), _shot(pg, fclip)
        im = Image.new("RGB", (W, top_pad + k.height + gap + f.height), bg)
        # the knobs sit on a card in the page (step 1's knob panel); draw that card behind them
        ImageDraw.Draw(im).rounded_rectangle((0, 0, W - 1, top_pad + k.height + top_pad - 2), radius=12,
                                             fill=card, outline=line)
        im.paste(k, (kx, top_pad))
        im.paste(f, (0, top_pad + k.height + gap))
        _pointer(im, xy[0] - fclip["x"], xy[1] - knobs["y"] + top_pad)
        frames.append(im)
        dur.append(ms)

    def setv(sel, v):
        pg.evaluate("([s, v]) => { const r = document.getElementById(s); r.value = String(v);"
                    " r.dispatchEvent(new Event('input', {bubbles: true})); }", [sel, v])
        pg.wait_for_timeout(60)
        _fan_done(pg)

    here = None

    def sweep(sel, values, hold):
        nonlocal here
        to = slider(sel)
        for xy in _lerp(here, to, 3):
            grab(xy, FRAME_MS)
        for v in values:
            setv(sel, v)
            here = slider(sel)
            grab(here, 200)
        dur[-1] = hold

    week0 = int(pg.evaluate("() => +document.getElementById('k-week').value"))
    slope0 = float(pg.evaluate("() => +document.getElementById('k-slope').value"))
    here = slider("k-week")
    here = (here[0] + 60, here[1] + 26)
    grab(here, 1500)
    sweep("k-week", list(range(week0 + 3, 35, 3)), 800)                        # regain later
    sweep("k-week", [w for w in range(30, week_min - 1, -4)] + [week0], 800)   # earlier, back
    sweep("k-slope", [0.5, 0.7, 0.9], 800)                                     # steeper
    sweep("k-slope", [0.6, 0.35, 0.15, slope0], 900)                           # flatter, back
    for xy in _lerp(here, (here[0] + 60, here[1] - 20), 3):
        grab(xy, FRAME_MS)
    _assert_clean(pg)
    force = _page_colors(pg)
    pg.close()
    return frames, dur, 0, force


#: Citations the README step-6 caption says the answer table flags on the recorded case.
CAPTION_FLAGGED = 2


def gif_answers(browser) -> tuple[list, list, int, list]:
    """Step 6: select models one by one; the timeline rings the evidence each one cited."""
    from PIL import Image
    dpr, target = 2, 960
    pg = _open(browser, 1200, height=2000, dpr=dpr)
    _scroll_top_to(pg, "#tl-case-host", 40)
    tl, days = _box(pg, "#tl-case-host"), _box(pg, "#mx-slice")
    solvers = pg.evaluate("() => [...document.querySelectorAll('#mx tr[data-solver]')].map(r => r.dataset.solver)")
    slices = pg.evaluate("() => D.answers.slices")
    first, last, early = solvers[0], slices[-1], slices[min(2, len(slices) - 1)]
    # The README caption states how many citations the table flags (outside the ledger or
    # dated after the answer day) and that one model on one answer day makes them. Check it on
    # the page's own data, for every model and every answer day.
    stray = pg.evaluate("""() => { const ev = new Map((D.case.evidence || []).map(e => [e.id, e.t])), out = [];
        for (const [m, a] of Object.entries(D.answers.models)) for (const [t, s] of Object.entries(a.by_slice || {}))
          for (const id of (s.cited || [])) if (!ev.has(id) || ev.get(id) > +t) out.push(m + '@' + t + ':' + id);
        return out; }""")
    assert len(stray) == CAPTION_FLAGGED and len({x.split(":")[0] for x in stray}) == 1, \
        f"flagged citations differ from the README caption ({CAPTION_FLAGGED}, one model-day): {stray[:5]}"

    def summary_bottom():
        b = _box(pg, "#mx-sum")
        return b["y"] + b["height"] + 10

    # the summary under the table wraps on some days: size the canvas for the tallest one
    pg.locator(f'#mx-slice button[data-t="{early}"]').click()
    pg.wait_for_timeout(150)
    tallest = summary_bottom()
    pg.locator(f'#mx-slice button[data-t="{last}"]').click()
    pg.locator(f'#mx tr[data-solver="{first}"]').click()
    pg.wait_for_timeout(150)
    tallest = max(tallest, summary_bottom())
    top = {"x": tl["x"], "y": tl["y"] - 4, "width": tl["width"], "height": tl["height"] + 8}
    bot = {"x": tl["x"], "y": days["y"] - 6, "width": tl["width"], "height": tallest - days["y"] + 6}
    pad, gap = 16, 18
    W = tl["width"] + 2 * pad
    H = pad + top["height"] + gap + bot["height"] + pad
    s = target / W

    def to_canvas(x, y):                              # viewport CSS px -> output px
        if y < top["y"] + top["height"] + gap / 2:
            return (x - top["x"] + pad) * s, (y - top["y"] + pad) * s
        return (x - bot["x"] + pad) * s, (y - bot["y"] + pad + top["height"] + gap) * s

    frames, dur = [], []

    def grab(xy, ms):
        cut = dict(bot, height=summary_bottom() - bot["y"])   # nothing below the summary line
        a, b = _shot(pg, top), _shot(pg, cut)
        im = Image.new("RGB", (round(W * dpr), round(H * dpr)), "white")
        im.paste(a, (pad * dpr, pad * dpr))
        im.paste(b, (pad * dpr, round((pad + top["height"] + gap) * dpr)))
        im = im.resize((target, round(H * s)), Image.Resampling.LANCZOS)
        frames.append(_pointer(im, *to_canvas(*xy)))
        dur.append(ms)

    def row(solver):
        b = _box(pg, f'#mx tr[data-solver="{solver}"]')
        return b["x"] + 150, b["y"] + b["height"] / 2

    def day(t):
        b = _box(pg, f'#mx-slice button[data-t="{t}"]')
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    here = None

    def click(sel, to, hold):
        nonlocal here
        for xy in _lerp(here, to, 5):
            grab(xy, FRAME_MS)
        pg.locator(sel).click()
        pg.wait_for_timeout(150)
        grab(to, hold)
        here = to

    here = row(first)
    here = (here[0] + 70, here[1] + 14)
    grab(here, 1600)                  # the page opens on the last answer day, first model selected
    picks = [m for m in ("kimi-k3", "gemini-3.1-pro-preview", "minimax-m3") if m in solvers]
    assert len(picks) >= 2, f"the answer table no longer lists the models this animation selects: {solvers}"
    for m in picks:
        click(f'#mx tr[data-solver="{m}"]', row(m), 1500)
    click(f'#mx-slice button[data-t="{early}"]', day(early), 1800)
    click(f'#mx-slice button[data-t="{last}"]', day(last), 500)
    click(f'#mx tr[data-solver="{first}"]', row(first), 600)
    _assert_clean(pg)
    force = _page_colors(pg)
    pg.close()
    return frames, dur, 0, force


# ------------------------------------------------------------------ the terminal


def _redact(line: str, tree: pathlib.Path) -> str:
    """Machine-specific parts of a recorded line: the clone path, uv's interpreter, the batch stamp."""
    line = line.replace(str(tree), CLONE)
    line = re.sub(r"(interpreter at: )\S+", r"\1<python>", line)
    return re.sub(r"\b\d{8}-\d{6}\b", "<batch>", line)


def record(tree: pathlib.Path) -> None:
    """Run the four quick-start commands in `tree` under a pseudo-terminal; keep every line."""
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import termios
    tree = tree.expanduser().resolve()
    used = [d for d in (".venv", "results", "reports", "cases")
            if (tree / d).exists() and any((tree / d).iterdir())]
    if used:
        sys.exit(f"[gifs] {tree} is not a fresh clone ({', '.join(used)} already present)")
    cols, rows = 120, 40
    env = {k: v for k, v in os.environ.items() if k not in ("HAENV_OUTPUT_ROOT", "HAENV_DATA_ROOT")}
    env.update(NO_COLOR="1", COLUMNS=str(cols), LINES=str(rows), TERM="xterm-256color")
    runs = []
    for cmd in QUICKSTART:
        m, s = pty.openpty()
        fcntl.ioctl(s, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        t0 = time.time()
        p = subprocess.Popen(cmd, shell=True, cwd=tree, stdin=s, stdout=s, stderr=s, env=env,
                             close_fds=True)
        os.close(s)
        buf, lines = b"", []
        while True:
            r, _, _ = select.select([m], [], [], 0.2)
            if m in r:
                try:
                    chunk = os.read(m, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                buf += chunk
                *done, buf = buf.split(b"\n")
                now = round(time.time() - t0, 3)
                lines += [[now, d.decode("utf-8", "replace").rstrip("\r")] for d in done]
            elif p.poll() is not None:
                break
        if buf:
            lines.append([round(time.time() - t0, 3), buf.decode("utf-8", "replace").rstrip("\r")])
        rc = p.wait()
        os.close(m)
        lines = [[t, _redact(ln, tree)] for t, ln in lines]
        runs.append({"cmd": cmd, "exit": rc, "seconds": round(time.time() - t0, 3), "lines": lines})
        print(f"[gifs] {cmd} -> exit {rc}, {len(lines)} line(s)")
        if rc != 0:
            sys.exit(f"[gifs] `{cmd}` exited {rc}; the quick start is broken, not recorded")
    # Machine-local prefixes: the top-level directory the clone lives under, and home directories.
    top = pathlib.Path(tree).resolve().parts[1]
    local = re.compile("|".join(re.escape(p) for p in sorted({f"/{top}/", "/root/", "/home/"})))
    leaked = [ln for r in runs for _, ln in r["lines"] if local.search(ln)]
    if leaked:
        sys.exit(f"[gifs] the transcript would publish machine-local paths: {leaked[:3]}")
    TRANSCRIPT.write_text(json.dumps({
        "note": "Recorded by docs/scripts/make_readme_gifs.py --record in a fresh clone. "
                "Generated file, do not hand-edit.",
        "cwd": CLONE, "columns": cols, "runs": runs}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(f"[gifs] transcript -> {TRANSCRIPT.relative_to(ROOT)}")


_TERM_CSS = """
*{box-sizing:border-box} body{margin:0;background:#fff}
.win{width:960px;border-radius:10px;overflow:hidden;background:#0d1117;
     font:12.5px/17px "DejaVu Sans Mono","Noto Sans Mono","Liberation Mono",monospace}
.bar{height:30px;background:#1f242c;display:flex;align-items:center;padding:0 12px;gap:7px}
.bar i{width:11px;height:11px;border-radius:50%;display:inline-block}
.bar span{color:#8b949e;font-size:12px;margin-left:12px}
.scr{padding:10px 14px 12px;height:__H__px;overflow:hidden;white-space:pre;color:#c9d1d9}
.l{height:17px;overflow:hidden}
.p{color:#3fb950}.c{color:#f0f6fc;font-weight:bold}.info{color:#6e7681}.warn{color:#9e8a4f}
.err{color:#f07b72}.hv{color:#e6edf3}.gt{color:#b9a46a}.cur{background:#c9d1d9;color:#0d1117}
"""


def _classify(lines: list[str]) -> list[str]:
    """Colour class per line: log records by level (continuations inherit), else stdout."""
    out, prev = [], "hv"
    for ln in lines:
        m = re.match(r"(INFO|WARNING|ERROR|DEBUG) ", ln)
        if m:
            prev = {"INFO": "info", "DEBUG": "info", "WARNING": "warn", "ERROR": "err"}[m.group(1)]
        elif ln.startswith(" ") and prev in ("info", "warn", "err"):
            pass
        elif ln.startswith("[gates]"):
            prev = "gt"
        else:
            prev = "hv"
        out.append(prev)
    return out


def gif_quickstart(browser) -> tuple[list, list, int, list]:
    from PIL import Image
    if not TRANSCRIPT.is_file():
        sys.exit(f"[gifs] {TRANSCRIPT.relative_to(ROOT)} is missing; run with --record <fresh clone>")
    tr = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    bad = [r["cmd"] for r in tr["runs"] if r["exit"] != 0]
    assert not bad, f"the transcript records a failing command: {bad}"
    assert [r["cmd"] for r in tr["runs"]] == list(QUICKSTART), "transcript is not the README's quick start"
    cols, rows = tr["columns"], 24
    pg = browser.new_page(viewport={"width": 960, "height": 700}, device_scale_factor=1)
    pg.set_content("<style>" + _TERM_CSS.replace("__H__", str(rows * 17 + 22)) + "</style>"
                   '<div class="win" id="w"><div class="bar"><i style="background:#ff5f57"></i>'
                   '<i style="background:#febc2e"></i><i style="background:#28c840"></i>'
                   f'<span>{html.escape(tr["cwd"])}</span></div><div class="scr" id="s"></div></div>')
    screen: list[tuple[str, str]] = []
    prompt = f'<span class="p">{html.escape(tr["cwd"])} $</span> '

    def paint(extra: str | None = None):
        rowsv = [f'<div class="l {c}">{html.escape(t[:cols])}</div>' if c != "raw"
                 else f'<div class="l">{t}</div>' for c, t in screen]
        if extra is not None:
            rowsv.append(f'<div class="l">{extra}<span class="cur"> </span></div>')
        pg.evaluate("(h) => { document.getElementById('s').innerHTML = h; }", "".join(rowsv[-rows:]))
        return Image.open(io.BytesIO(pg.locator("#w").screenshot())).convert("RGB")

    frames, dur = [paint(prompt)], [500]
    for run in tr["runs"]:
        cmd = run["cmd"]
        for i in range(6, len(cmd) + 6, 6):                      # typing
            frames.append(paint(prompt + f'<span class="c">{html.escape(cmd[:i])}</span>'))
            dur.append(FRAME_MS)
        dur[-1] = 400
        screen.append(("raw", prompt + f'<span class="c">{html.escape(cmd)}</span>'))
        times = [t for t, _ in run["lines"]]
        texts = [t for _, t in run["lines"]]
        kinds = _classify(texts)
        n = len(texts)
        marks = (0.25, 0.55, 0.8, 1.0) if n > 30 else (0.5, 1.0)
        shown = 0
        for c in sorted({max(1, round(n * f)) for f in marks}):  # output, in bursts, in order
            screen.extend(zip(kinds[shown:c], texts[shown:c]))
            span = times[c - 1] - (times[shown - 1] if shown else 0.0)
            shown = c
            frames.append(paint())
            dur.append(int(min(450, max(FRAME_MS, span * 1000 / 4))))
        dur[-1] = 2600 if run is not tr["runs"][-1] else 1400
        screen.append(("raw", ""))
    frames.append(paint(prompt))
    dur.append(3500)
    pg.close()
    force = [_hex(c) for c in dict.fromkeys(re.findall(r"#[0-9a-f]{6}", _TERM_CSS))]
    return frames, dur, len(frames) - 1, force


# ------------------------------------------------------------------ main

GIFS = {"today": gif_today, "fan": gif_fan, "answers": gif_answers, "quickstart": gif_quickstart}
COLORS = {"today": 128, "fan": 96, "answers": 256, "quickstart": 32}


def _readme_matches_quickstart() -> None:
    """The quick-start block in both READMEs must list exactly the recorded commands."""
    for name in ("README.md", "README.zh-CN.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        shown = [re.sub(r"\s+#.*$", "", ln).rstrip() for ln in text.splitlines()
                 if ln.startswith("uv run haenv ") and "example-ew" in ln and "--models" not in ln]
        if [re.sub(r"\s+", " ", c) for c in shown[:4]] != [re.sub(r"\s+", " ", c) for c in QUICKSTART]:
            sys.exit(f"[gifs] {name}: quick-start commands {shown[:4]} differ from QUICKSTART")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", choices=sorted(GIFS), help="build only these")
    ap.add_argument("--record", metavar="CLONE", type=pathlib.Path,
                    help="first run the quick start in this fresh clone and save the transcript")
    a = ap.parse_args()
    _readme_matches_quickstart()
    if a.record:
        record(a.record)
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    with sync_playwright() as p:
        browser = _chrome(p)
        for name in a.only or list(GIFS):
            frames, dur, still, force = GIFS[name](browser)
            path = OUT / f"readme_{name}.gif"
            size = save_gif(frames, dur, path, COLORS[name], still, force)
            w, h = frames[0].size
            ok = size < MAX_BYTES and WIDTH[0] <= w <= WIDTH[1]
            print(f"[gifs] {path.relative_to(ROOT)}: {w}x{h}, {len(frames)} frames, "
                  f"{sum(dur) / 1000:.1f} s, {size:,} bytes{'' if ok else '  <-- over budget'}")
            if not ok:
                failed.append(name)
        browser.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
