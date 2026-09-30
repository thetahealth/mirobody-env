"""report_html.py (haenv_rare) — one self-contained HTML page for a rare_coding batch (no JS, cream/blue,
narrow-screen safe): rc_* aggregates per solver, the per-case table, sampled cases with their
gold vs answer, the batch's provenance, and the Markdown report folded underneath.

  uv run haenv-rare-html --batch results/joint_dx/<job>/<batch> [--out reports/.../eval.html]
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

RC = ["rc_coverage", "rc_hpo_strict", "rc_hpo_hier", "rc_polarity_ok", "rc_subject_ok", "rc_wrong_rate", "rc_negation_trap",
      "rc_narr_recall", "rc_narr_span_ok", "rc_narr_polarity_ok", "rc_narr_noise_abstain",
      "rc_orpha_top1", "rc_hgnc_ok", "rc_variant_hit", "rc_variant_gt_ok", "rc_variant_inh_ok", "rc_gene_from_variant",
      "rc_signal_index_ok", "rc_signal_phi_free"]
ZH = {"rc_coverage": "术语覆盖", "rc_hpo_strict": "HPO 严格", "rc_hpo_hier": "HPO 层级", "rc_polarity_ok": "否定/极性", "rc_subject_ok": "主体",
      "rc_wrong_rate": "错编率(越低越好)", "rc_negation_trap": "否定陷阱(越低越好)", "rc_orpha_top1": "ORPHA 首诊", "rc_hgnc_ok": "基因",
      "rc_variant_hit": "真值变异在列", "rc_variant_gt_ok": "基因型", "rc_variant_inh_ok": "遗传来源(trio)", "rc_gene_from_variant": "基因由变异证据定",
      "rc_signal_index_ok": "影像索引", "rc_signal_phi_free": "影像无 PHI",
      "rc_narr_recall": "病历术语召回", "rc_narr_span_ok": "病历位置对位", "rc_narr_polarity_ok": "病历否定/极性",
      "rc_narr_noise_abstain": "噪声句弃权"}
CSS = """:root{--bg:#fbf7ec;--fg:#1e3a8a;--ac:#1d4ed8;--line:#d9d2bf;--ok:#166534;--bad:#991b1b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif;padding:1rem}
main{max-width:64rem;margin:0 auto}h1{font-size:1.6rem;margin:.2rem 0}h2{font-size:1.15rem;color:var(--ac);border-bottom:2px solid var(--line);padding-bottom:.2rem;margin-top:1.6rem}
table{border-collapse:collapse;width:100%;font-size:.92rem;display:block;overflow-x:auto}th,td{border:1px solid var(--line);padding:.3rem .5rem;text-align:left;vertical-align:top}
th{background:#f3ecd8}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:.6rem}
.card{background:#fff9ea;border:1px solid var(--line);border-radius:.6rem;padding:.6rem .8rem}.card b{display:block;font-size:1.4rem;color:var(--ac)}
details{margin:.5rem 0}summary{cursor:pointer;color:var(--ac)}pre{white-space:pre-wrap;word-break:break-word;font-size:.8rem;background:#fff9ea;padding:.6rem;border-radius:.4rem}
.muted{opacity:.75;font-size:.9rem}@media(max-width:40rem){body{padding:.6rem}h1{font-size:1.3rem}}"""


def fmt(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _narrative_row(att: dict, r: dict, e) -> str:
    nar = att.get("narrative") or {}
    if not nar:
        return ""
    excerpt = ""
    try:
        md = Path(nar["path"]).read_text(encoding="utf-8")
        head = "## History of Present Illness" if "## History of Present Illness" in md else "## 现病史"
        body = md.split(head, 1)[-1].split("##", 1)[0].strip()
        excerpt = body[:180] + ("…" if len(body) > 180 else "")
    except OSError:
        excerpt = "(文件不在盘上)"
    return (f"<tr><td>叙述病历 {nar.get('n_gold_spans', 0)} 条金标句 + {nar.get('n_noise', 0)} 条噪声句<br>"
            f"<span class=muted>现病史节选:{e(excerpt)}</span></td>"
            f"<td>召回 {fmt(r.get('rc_narr_recall'))} · 位置对位 {fmt(r.get('rc_narr_span_ok'))} · 极性 {fmt(r.get('rc_narr_polarity_ok'))} · 噪声弃权 {fmt(r.get('rc_narr_noise_abstain'))}</td></tr>")


def load(batch: Path):
    rows = [json.loads(l) for l in (batch / "eval.jsonl").open()]
    cases = {}
    for l in (batch / "cases.jsonl").open():
        c = json.loads(l)
        cases[c["case_id"]] = c
    meta = json.loads((batch / "batch.json").read_text()) if (batch / "batch.json").exists() else {}
    return rows, cases, meta


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for r in rows:
        s = r.get("solver")
        for k in RC:
            v = r.get(k)
            if isinstance(v, (bool, int, float)) and v is not None:
                agg[s][k][0] += float(v)
                agg[s][k][1] += 1
    return agg


def narrative_errors(batch: Path, cases: dict, solver: str) -> dict:
    """Per gold span of the narrative: did `solver` code it with the right polarity / subject, and
    did it leave the noise sentences alone? Grouped by the sentence's first three words, so a
    phrasing the coder does not read (a negation cue, a kin word) shows as one row."""
    resp = {}
    rp = batch / "responses.jsonl"
    if not rp.exists():
        return {}
    for l in rp.open():
        r = json.loads(l)
        if r.get("solver") != solver:
            continue
        raw = r.get("raw") or ""
        try:
            resp[r["case"]] = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except ValueError:
            resp[r["case"]] = {}
    out = {"absent": defaultdict(lambda: [0, 0, 0]), "relative": defaultdict(lambda: [0, 0, 0]), "noise": []}
    for cid, c in cases.items():
        nar = ((c.get("case") or {}).get("adjudication", {}).get("rare", {}).get("attachments") or {}).get("narrative") or {}
        try:
            md = Path(nar["path"]).read_text(encoding="utf-8")
        except (KeyError, OSError):
            continue
        na = ((resp.get(cid) or {}).get("narrative") or {}).get("assertions") or []
        for sp in nar.get("spans") or []:
            sent = md[sp["start"]:sp["end"]]
            hits = [a for a in na if len(a.get("char_span") or []) == 2 and a["char_span"][0] < sp["end"] and a["char_span"][1] > sp["start"]]
            if sp["hpo_id"] is None:
                if hits:
                    out["noise"].append((cid, sent, [(a.get("codes") or {}).get("hpo") for a in hits]))
                continue
            mine = [a for a in hits if (a.get("codes") or {}).get("hpo") == sp["hpo_id"]]
            head = " ".join(sent.split()[:3]) if sent.isascii() else sent[:4]
            for axis, want, key in (("absent", "absent", "polarity"), ("relative", "relative", "subject")):
                if (axis == "absent" and sp["polarity"] != "absent") or (axis == "relative" and sp["subject"] != "relative"):
                    continue
                row = out[axis][head]
                if not mine:
                    row[2] += 1
                elif all(a.get(key) == want for a in mine):
                    row[0] += 1
                else:
                    row[1] += 1
    return out


def _errors_html(err: dict, e) -> str:
    if not err:
        return ""
    parts = ["<h2>叙述病历逐句失误</h2><p class=muted>按句子开头三个词归并:对 = 编码且极性/主体正确;错 = 编到了但极性/主体错;漏 = 没编到。</p>"]
    for axis, title in (("absent", "否定句(金标 absent)"), ("relative", "亲属句(金标 relative)")):
        rows = err[axis]
        tot = [sum(v[i] for v in rows.values()) for i in range(3)]
        parts.append(f"<h3>{e(title)}:对 {tot[0]} · 错 {tot[1]} · 漏 {tot[2]}</h3><table><tr><th>句子开头</th><th>对</th><th>错</th><th>漏</th></tr>")
        for head, (ok, bad, miss) in sorted(rows.items(), key=lambda kv: (-kv[1][1], -kv[1][2], kv[0])):
            parts.append(f"<tr><td>{e(head)}…</td><td>{ok}</td><td class={'bad' if bad else ''}>{bad}</td><td class={'bad' if miss else ''}>{miss}</td></tr>")
        parts.append("</table>")
    parts.append(f"<h3>噪声句被编码:{len(err['noise'])} 句</h3>")
    if err["noise"]:
        parts.append("<table><tr><th>case</th><th>句子</th><th>输出的码</th></tr>" + "".join(
            f"<tr><td>{e(c)}</td><td>{e(s)}</td><td>{e(', '.join(str(x) for x in codes))}</td></tr>" for c, s, codes in err["noise"]) + "</table>")
    return "".join(parts)


def _compare_html(agg, main: str, other: Path, e) -> str:
    rows2, cases2, meta2 = load(other)
    agg2 = aggregate(rows2)
    lang2 = _lang_of(cases2)
    out = [f"<h2>与 {e(other.parent.name)} 对照({e(main)})</h2><p class=muted>对照批次 {e(other.name)} · {len(cases2)} 例 · 语言 {e(lang2)}</p>"
           f"<table><tr><th>判据</th><th>本批</th><th>对照批</th><th>差</th></tr>"]
    for k in RC:
        a, b = agg[main][k], agg2[main][k]
        if not a[1] and not b[1]:
            continue
        va, vb = (a[0] / a[1] if a[1] else None), (b[0] / b[1] if b[1] else None)
        d = (va - vb) if va is not None and vb is not None else None
        worse = d is not None and ((d > 0.005) if k in ("rc_wrong_rate", "rc_negation_trap") else (d < -0.005))
        out.append(f"<tr><td>{e(ZH[k])} <span class=muted>{k}</span></td><td>{fmt(va)}</td><td>{fmt(vb)}</td>"
                   f"<td class={'bad' if worse else ''}>{'—' if d is None else f'{d:+.3f}'}</td></tr>")
    out.append("</table>")
    return "".join(out)


def _lang_of(cases: dict) -> str:
    langs = {str(((c.get("case") or {}).get("adjudication", {}).get("rare", {}) or {}).get("lang") or "zh") for c in cases.values()}
    return "/".join(sorted(langs)) or "zh"


def render(batch: Path, rows, cases, meta, sample_n: int = 5, compare: Path | None = None) -> str:
    agg = aggregate(rows)
    scored = [s for s in agg if s and any(agg[s][k][1] for k in RC)]
    # real solvers in the table; the degenerate baselines (haenv's floor/ceiling rows) only as a
    # one-line summary, otherwise 20 columns of 0/60 bury the reading
    solvers = [s for s in scored if "mirobody" in s or agg[s]["rc_coverage"][0] > 0]
    baselines = [s for s in scored if s not in solvers]
    e = html.escape
    out = [f"<!doctype html><html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>罕见病编码评测 · {e(batch.name)}</title><style>{CSS}</style></head><body><main>"
           f"<h1>罕见病编码评测报告</h1><p class=muted>批次 {e(batch.name)} · {len(cases)} 例 · 语言 {e(_lang_of(cases))} · world_sha {e(str(meta.get('world_sha', '?')))} · 生成器 {e(str(meta.get('generator', '?')))} · SYNTHETIC,仅评测用</p>"]
    if "en" in _lang_of(cases):
        out.append("<p class=muted>英文包:病历正文、带金标的症状句与编码任务说明为英文;evidence_ledger 里来自 haenv 共享世界层的日常干扰事件(良性事件池)仍为中文。"
                   "英文句式来自出题侧的固定模板(否定 5 种、家族 3 种、现病史 5 种):被测方若按这些模板调过规则,满分只说明链路通,不说明它读得懂一般英文病历。</p>")
    # 1 headline cards for the first real solver
    main = next((s for s in solvers if "mirobody" in s), solvers[0] if solvers else None)
    if main:
        out.append("<h2>结论</h2><div class=grid>")
        for k in ("rc_orpha_top1", "rc_hpo_strict", "rc_narr_span_ok", "rc_narr_noise_abstain", "rc_variant_hit", "rc_gene_from_variant", "rc_signal_index_ok", "rc_wrong_rate"):
            n, d = agg[main][k]
            out.append(f"<div class=card>{e(ZH[k])}<b>{fmt(n / d) if d else '—'}</b><span class=muted>{e(main)} · {int(n)}/{d}</span></div>")
        out.append("</div>")
    # 2 aggregate table
    out.append("<h2>判据汇总</h2><table><tr><th>判据</th>" + "".join(f"<th>{e(s)}</th>" for s in solvers) + "</tr>")
    for k in RC:
        cells = []
        for s in solvers:
            n, d = agg[s][k]
            if not d:
                cells.append("<td>—</td>")
            else:
                v = n / d
                good = (v <= 0.05) if k in ("rc_wrong_rate", "rc_negation_trap") else (v >= 0.95)
                cells.append(f"<td class={'ok' if good else 'bad'}>{v:.3f} <span class=muted>({int(n)}/{d})</span></td>")
        out.append(f"<tr><td>{e(ZH[k])} <span class=muted>{k}</span></td>{''.join(cells)}</tr>")
    out.append("</table>")
    if baselines:
        worst = max((agg[b]["rc_coverage"][0] / agg[b]["rc_coverage"][1]) for b in baselines if agg[b]["rc_coverage"][1]) if any(agg[b]["rc_coverage"][1] for b in baselines) else 0
        out.append(f"<p class=muted>退化基线 {len(baselines)} 个({e(', '.join(baselines[:6]))}…):术语覆盖最高 {worst:.3f} —— 判据把「不编码」与真实解法分得开。</p>")
    if main and compare:
        out.append(_compare_html(agg, main, compare, e))
    if main:
        out.append(_errors_html(narrative_errors(batch, cases, main), e))
    # 3 per-case table for the main solver
    if main:
        out.append(f"<h2>逐例({e(main)})</h2><table><tr><th>case</th><th>病种</th><th>基因</th><th>覆盖</th><th>HPO</th><th>首诊</th><th>变异</th><th>基因定</th><th>影像</th><th>病历召回</th><th>病历对位</th><th>噪声弃权</th><th>候选数</th></tr>")
        for r in sorted((r for r in rows if r.get("solver") == main), key=lambda r: r["case"]):
            rare = cases.get(r["case"], {}).get("case", {}).get("adjudication", {}).get("rare", {})
            def cell(k):
                v = r.get(k)
                if v is None:
                    return "<td>—</td>"
                return f"<td class={'ok' if float(v) >= 0.999 else 'bad'}>{fmt(v)}</td>"
            out.append(f"<tr><td>{e(r['case'])}</td><td>{e(str(rare.get('orpha', '')))}</td><td>{e(str(rare.get('gene') or '—'))}</td>"
                       + "".join(cell(k) for k in ("rc_coverage", "rc_hpo_strict", "rc_orpha_top1", "rc_variant_hit", "rc_gene_from_variant", "rc_signal_index_ok",
                                                   "rc_narr_recall", "rc_narr_span_ok", "rc_narr_noise_abstain"))
                       + f"<td>{fmt(r.get('rc_n_variants'))}</td></tr>")
        out.append("</table>")
        # 4 samples
        out.append(f"<h2>抽样({sample_n} 例:金标 vs 输出)</h2>")
        picked = sorted((r for r in rows if r.get("solver") == main), key=lambda r: r["case"])[:sample_n]
        for r in picked:
            rare = cases.get(r["case"], {}).get("case", {}).get("adjudication", {}).get("rare", {})
            gold = rare.get("hpo_gold") or []
            att = rare.get("attachments") or {}
            g = att.get("genome") or {}
            v = g.get("variant") or {}
            out.append(f"<details><summary>{e(r['case'])} · {e(str(rare.get('orpha')))} {e(str(rare.get('gene') or ''))} · 覆盖 {fmt(r.get('rc_coverage'))} · 变异 {fmt(r.get('rc_variant_hit'))}</summary>"
                       f"<table><tr><th>金标</th><th>输出/判据</th></tr>"
                       f"<tr><td>HPO {len(gold)} 条:{e(';'.join(h['label'] + ('(否)' if h['polarity'] == 'absent' else '') + ('/亲属' if h['subject'] == 'relative' else '') for h in gold[:8]))}{'…' if len(gold) > 8 else ''}</td>"
                       f"<td>断言 {fmt(r.get('rc_n_assert'))} 条 · 严格 {fmt(r.get('rc_hpo_strict'))} · 层级 {fmt(r.get('rc_hpo_hier'))} · 极性 {fmt(r.get('rc_polarity_ok'))} · 主体 {fmt(r.get('rc_subject_ok'))}</td></tr>"
                       f"<tr><td>变异 {e(str(v.get('chrom', '')))}:{e(str(v.get('pos', '')))} {e(str(v.get('ref', '')))}>{e(str(v.get('alt', '')))} · 遗传 {e(str(g.get('inheritance', '')))} · 诱饵 {len(g.get('decoys') or [])}</td>"
                       f"<td>候选 {fmt(r.get('rc_n_variants'))} · 命中 {fmt(r.get('rc_variant_hit'))} · 基因型 {fmt(r.get('rc_variant_gt_ok'))} · 来源 {fmt(r.get('rc_variant_inh_ok'))}</td></tr>"
                       f"<tr><td>影像 {len((att.get('imaging') or {}).get('series') or [])} 个序列({e(str((att.get('imaging') or {}).get('collection', '—')))} · {e(str((att.get('imaging') or {}).get('tier', '—')))})</td>"
                       f"<td>索引 {fmt(r.get('rc_signal_index_ok'))} · 无 PHI {fmt(r.get('rc_signal_phi_free'))}</td></tr>"
                       + _narrative_row(att, r, e) + "</table></details>")
    # 5 provenance
    out.append("<h2>环境与版本</h2><table>" + "".join(f"<tr><td>{e(k)}</td><td>{e(str(meta.get(k)))}</td></tr>" for k in ("batch_id", "world_sha", "judging_sha16", "generator", "created_at", "kernel_sha256") if k in meta) + "</table>")
    # 6 the markdown report, folded
    md = next(iter(sorted((batch.parent.parent.parent.parent / "reports" / batch.parent.name / batch.name).glob("eval-*.md"))), None)
    if md and md.exists():
        out.append(f"<h2>完整 Markdown 报告</h2><details><summary>{e(md.name)}</summary><pre>{e(md.read_text(encoding='utf-8'))}</pre></details>")
    out.append("</main></body></html>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--compare", default="", help="another batch dir: the main solver's rc_* side by side")
    a = ap.parse_args()
    batch = Path(a.batch)
    rows, cases, meta = load(batch)
    page = render(batch, rows, cases, meta, a.samples, Path(a.compare) if a.compare else None)
    out = Path(a.out) if a.out else Path("reports") / batch.parent.name / batch.name / f"eval-{batch.parent.name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(json.dumps({"out": str(out), "bytes": len(page.encode()), "solvers": sorted({r.get('solver') for r in rows if r.get('solver')})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
