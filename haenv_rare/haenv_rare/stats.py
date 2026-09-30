"""haenv-rare-stats -- do the synthetic case reports, PEDs and VCFs look like real ones?

    uv run haenv-rare-stats --jobs inputs/rare_coding-p6-zh.job.yaml inputs/rare_coding-p7-en.job.yaml \\
        --corpus /data/xfs_recovery/caill/mirobody_rare/data/rareDieaseCollect [--out table.md]

Real side: the case-description section of the 84 Chinese journal reports (《罕见病研究》: the
`1 临床资料` / `1 病例简介` section up to the next top-level heading) and of the PMC English case
reports (`Case report` / `Case presentation` up to the next heading). Synthetic side: every case
report a job points at. Per text: characters, sentences, negated sentences, subsections, family
history mentioned. Medians with the 10th-90th percentile. PED bytes and VCF record counts are
read from the files. SYNTHETIC evaluation data only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import statistics

import yaml

_ZH_SENT = re.compile(r"[^。！？!?\n]+[。！？!?]?")
_ZH_NEG = re.compile(r"未见|否认|阴性|未及|未闻及|未触及|无明显|无[^\s，。]")
_EN_NEG = re.compile(r"\b(no|not|denied|denies|negative|without|absent|unremarkable)\b", re.I)
_MD_NOISE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|<[^>]+>")


def text_metrics(md: str, lang: str) -> dict:
    """chars / sentences / negated sentences / subsections / family history, of one text."""
    body = _MD_NOISE.sub("", md)
    heads = re.findall(r"^#+\s", body, re.M)
    prose = "\n".join(l for l in body.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    if lang == "zh":
        sents = [s for s in _ZH_SENT.findall(prose) if s.strip()]
        neg = sum(1 for s in sents if _ZH_NEG.search(s))
        fam = "家族史" in body or "家族中" in body
        chars = len(re.sub(r"\s", "", prose))
    else:
        sents = [s for s in re.split(r"(?<=[.!?])\s+", prose.replace("\n", " ")) if s.strip()]
        neg = sum(1 for s in sents if _EN_NEG.search(s))
        fam = bool(re.search(r"family history|mother|father|sister|brother|sibling", body, re.I))
        chars = len(prose)
    return {"chars": chars, "sentences": len(sents), "negated": neg, "subsections": len(heads), "family": fam}


def _zh_case_section(md: str) -> str | None:
    m = re.search(r"^(#+)\s*1\s*[\.、]?\s*(临床资料|病例简介|病例资料|病例介绍).*$", md, re.M)
    if not m:
        return None
    rest = md[m.end():]
    nxt = re.search(rf"^{re.escape(m.group(1))}\s*[2-9]\s", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def _en_case_section(md: str) -> str | None:
    m = re.search(r"^(#+)\s*(\d+\.?\s*)?(case report|case presentation|case description)s?\s*$", md, re.M | re.I)
    if not m:
        return None
    rest = md[m.end():]
    nxt = re.search(rf"^#{{1,{len(m.group(1))}}}\s", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def corpus_texts(root: str | pathlib.Path) -> dict[str, list[str]]:
    root = pathlib.Path(root)
    zh, en = [], []
    for f in sorted((root / "罕见病研究期刊病例").glob("*/content.json")):
        sec = _zh_case_section(json.loads(f.read_text(encoding="utf-8")).get("body_markdown") or "")
        if sec and sec.strip():
            zh.append(sec)
    for f in sorted((root / "PMC罕见病病例").glob("*/content.md")):
        sec = _en_case_section(f.read_text(encoding="utf-8"))
        if sec and len(sec.strip()) > 200:
            en.append(sec)
    return {"zh": zh, "en": en}


def _dist(xs: list[float]) -> str:
    if not xs:
        return "-"
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return f"{statistics.median(xs):g} ({q(0.1):g}-{q(0.9):g})"


def summarize(rows: list[dict]) -> dict:
    return {"n": len(rows), **{k: _dist([r[k] for r in rows]) for k in ("chars", "sentences", "negated", "subsections")},
            "family": f"{sum(r['family'] for r in rows) / max(1, len(rows)):.0%}"}


def job_texts(job: str | pathlib.Path) -> tuple[str, list[str], list[dict]]:
    doc = yaml.safe_load(pathlib.Path(job).read_text(encoding="utf-8"))
    lang = "en" if any((c["latent"].get("rare_lang") == "en") for c in doc["cases"]) else "zh"
    texts, cases = [], []
    for c in doc["cases"]:
        att = c["latent"].get("rare_attachments") or {}
        nar = att.get("narrative") or {}
        if nar.get("path") and pathlib.Path(nar["path"]).exists():
            texts.append(pathlib.Path(nar["path"]).read_text(encoding="utf-8"))
        cases.append(c)
    return lang, texts, cases


def file_stats(cases: list[dict]) -> dict:
    ped, vcf = [], []
    for c in cases:
        gen = ((c["latent"].get("rare_attachments") or {}).get("genome") or {})
        if (gen.get("ped") or {}).get("path") and pathlib.Path(gen["ped"]["path"]).exists():
            ped.append(pathlib.Path(gen["ped"]["path"]).stat().st_size)
        f = (gen.get("files") or {}).get("proband") or {}
        p = pathlib.Path(f.get("path") or "")
        if p.is_file() and p.name.endswith(".gz"):
            with gzip.open(p, "rt") as fh:
                vcf.append(sum(1 for l in fh if not l.startswith("#")))
    return {"ped_bytes": _dist(ped), "vcf_records": _dist(vcf)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="haenv-rare-stats")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    real = corpus_texts(a.corpus)
    lines = ["| source | n | chars | sentences | negated sentences | subsections | family history | PED bytes | proband VCF records |",
             "|---|---|---|---|---|---|---|---|---|"]
    for lang, name in (("zh", "real: 《罕见病研究》 case section"), ("en", "real: PMC case report section")):
        s = summarize([text_metrics(t, lang) for t in real[lang]])
        lines.append(f"| {name} | {s['n']} | {s['chars']} | {s['sentences']} | {s['negated']} | {s['subsections']} | {s['family']} | - | - |")
    for j in a.jobs:
        lang, texts, cases = job_texts(j)
        s = summarize([text_metrics(t, lang) for t in texts])
        f = file_stats(cases)
        lines.append(f"| synthetic: {pathlib.Path(j).name.replace('.job.yaml', '')} ({lang}) | {s['n']} | {s['chars']} | {s['sentences']} | "
                     f"{s['negated']} | {s['subsections']} | {s['family']} | {f['ped_bytes']} | {f['vcf_records']} |")
    out = "\n".join(lines) + "\n"
    if a.out:
        pathlib.Path(a.out).write_text(out, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
