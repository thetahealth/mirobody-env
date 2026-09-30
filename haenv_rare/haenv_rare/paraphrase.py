"""paraphrase.py (haenv_rare) -- a HELD-OUT English case
report for every case of a rare job.

The deterministic English narratives (`narrative._render_en`) use a handful of templates, and a
parser tuned against them scores 1.000 there while failing ordinary English. This command asks
an LLM (`data/paraphrase.yaml:model`, overridable in config) to write each case report freely from the SAME gold
findings, so the gold, the attachments and the judges stay those of the source job and only the
phrasing is new. The model never places spans: it returns sentences tagged with the finding they
carry (or `noise`), and this module assembles the Markdown and computes every span itself.

Checked per case, else sent back with the reason (up to `max_retries`): every finding in exactly
one sentence, one finding per sentence, >=2 noise sentences, no diagnosis / alias / gene / ORPHA
name (whole-word, case-insensitive). The negation wording is deliberately NOT checked: checking
it against a cue list would pull the text back toward that list.

    uv run --with httpx --with tenacity --with tqdm haenv-rare-paraphrase \\
        --src inputs/rare_coding-p4-en.job.yaml --job-id rare_coding-p5-en-llm
    uv run haenv-rare-paraphrase --src inputs/rare_coding-p4-en.job.yaml \\
        --job-id rare_coding-p5-en-llm --cache-only       # rebuild from the cache, never call the model

Every accepted report is cached as `<attach-root>/<case>/llm.json`; a rerun only asks for the
cases without one. `--cache-only` opens no client at all and refuses (exit 1, nothing written)
when a case has no accepted cached report, so a rebuild cannot spend money or change the
held-out set. Held-out protocol: parser changes are tuned on cases < 60 (dev), cases >= 60 (test)
are measured once.
SYNTHETIC, evaluation only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

import yaml

from .gen import ROOT, _mentions

log = logging.getLogger("haenv_rare.paraphrase")

PROMPT = """You are writing a SYNTHETIC English clinical case report for a rare-disease benchmark.

Patient: {sex}, age {age}. Findings (use every one exactly once, each in its own sentence):
{findings}

Write the report as a clinician would, in these sections: Chief Complaint, History of Present Illness,
Past Medical History, Family History, Physical Examination, Investigations. Rules:
- Vary the wording naturally (synonyms, clinical phrasing, sentence structure) but keep each finding
  unambiguous: a clinician coding your sentence must arrive at the same HPO term.
- A finding marked ABSENT must be stated as absent / not observed / denied in that sentence.
- A finding of a RELATIVE must be attributed to that relative in that sentence, never to the patient.
- Add 2 or 3 sentences about unrelated everyday life events (noise); they must contain no symptom.
- Other sentences (vital signs, pending tests, scaffolding) are allowed but carry no finding.
- Never name or hint at a diagnosis, a disease, a syndrome, a gene or a genetic test result.
- Several sentences may share a section; do not use bullet points.

Return ONLY JSON: {{"sections": [{{"heading": "...", "sentences": [{{"text": "...", "finding": <number or null>, "noise": true|false}}]}}]}}
{feedback}"""


def _cfg() -> dict:
    """Writer settings: the package defaults (`data/paraphrase.yaml`), overridden key by key by
    a `rare_paraphrase:` section of config.yaml / config.local.yaml when one is present. The
    defaults ship with the plugin because the public config.yaml must not name a held-back one."""
    from haenv.cli import load_cfg
    cfg = dict(yaml.safe_load((Path(__file__).parent / "data" / "paraphrase.yaml").read_text(encoding="utf-8")))
    try:
        cfg.update((load_cfg() or {}).get("rare_paraphrase") or {})
    except FileNotFoundError:
        pass
    return cfg


def _key(cfg: dict) -> str:
    name = cfg["api_key_env"]
    if os.environ.get(name):
        return os.environ[name]
    env = Path(os.path.expanduser(cfg.get("env_file") or ""))
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(f"{name} not set and not in {env}")


def _findings(gold: list[dict]) -> str:
    rows = []
    for i, g in enumerate(gold, 1):
        who = f"RELATIVE ({g.get('relative')})" if g["subject"] == "relative" else "patient"
        rows.append(f"{i}. {g.get('label_en') or g['label']} — {'ABSENT' if g['polarity'] == 'absent' else 'present'} — {who}")
    return "\n".join(rows)


def _leak_terms(spec_id: str) -> list[str]:
    from .gen import Ontology, load_specs
    spec = load_specs()[spec_id]
    ont = Ontology.get()
    terms = [spec["diagnosis"], *spec["aliases"], spec.get("gene") or "", ont.orpha[str(spec["orpha"])]["name"]]
    return [t for t in terms if t and len(t) >= 3]


def check(doc: dict, n: int, leak: list[str]) -> list[str]:
    """Reasons to reject a report (empty list = accepted)."""
    errs, seen, noise = [], {}, 0
    for sec in doc.get("sections") or []:
        for s in sec.get("sentences") or []:
            f, txt = s.get("finding"), str(s.get("text") or "")
            if s.get("noise"):
                noise += 1
            if isinstance(f, int):
                if not 1 <= f <= n:
                    errs.append(f"finding {f} does not exist")
                seen[f] = seen.get(f, 0) + 1
            if "\n" in txt:
                errs.append("a sentence contains a line break")
    for i in range(1, n + 1):
        if seen.get(i, 0) != 1:
            errs.append(f"finding {i} appears {seen.get(i, 0)} times (must be exactly once)")
    if noise < 2:
        errs.append(f"only {noise} noise sentences (need 2 or 3)")
    blob = " ".join(str(s.get("text") or "") for sec in doc.get("sections") or [] for s in sec.get("sentences") or [])
    leaked = [t for t in leak if _mentions(blob, t, word=True)]
    if leaked:
        errs.append(f"names the diagnosis/gene: {leaked}")
    return errs


def assemble(case_id: str, doc: dict, gold: list[dict]) -> tuple[str, list[dict]]:
    """Markdown + spans computed here (sentences of a section share one paragraph line)."""
    md, spans = f"# Case {case_id}\n\n", []
    for sec in doc["sections"]:
        md += f"## {sec['heading'].strip()}\n"
        line = []
        for s in sec["sentences"]:
            txt = " ".join(str(s["text"]).split())
            start = len(md) + sum(len(x) + 1 for x in line)
            line.append(txt)
            f = s.get("finding")
            core_end = start + len(txt.rstrip(".").rstrip())
            if isinstance(f, int):
                g = gold[f - 1]
                spans.append({"hpo_id": g["hpo_id"], "polarity": g["polarity"], "subject": g["subject"],
                              "start": start, "end": core_end, "section": sec["heading"].strip(), "idx": g["idx"]})
            elif s.get("noise"):
                spans.append({"hpo_id": None, "polarity": None, "subject": None, "start": start, "end": core_end,
                              "section": sec["heading"].strip(), "idx": None})
        md += " ".join(line) + "\n\n"
    for sp in spans:
        assert md[sp["start"]:sp["end"]] == md[sp["start"]:sp["end"]].strip() and sp["end"] > sp["start"]
    return md, spans


def json_from(content: str) -> dict:
    """The JSON object of a model answer: as is, else the span from the first `{` to the last `}`
    (some models wrap it in a code fence or a sentence even when asked for JSON only)."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        i, j = content.find("{"), content.rfind("}")
        if i < 0 or j <= i:
            raise
        return json.loads(content[i:j + 1])


def cached(case: dict, out_dir: Path) -> dict | None:
    """The accepted report cached for this case, or None (absent, or failing today's check)."""
    p = out_dir / case["case_id"] / "llm.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    lat = case["latent"]
    errs = check(doc, len(lat["rare_hpo_gold"]), _leak_terms(lat["rare_spec_id"]))
    if errs:
        log.warning("[%s] cached report fails the check: %s", case["case_id"], errs[:3])
        return None
    return doc


def _client(cfg: dict):
    import httpx
    return httpx.AsyncClient(base_url=cfg["base_url"], headers={"Authorization": f"Bearer {_key(cfg)}"},
                             timeout=float(cfg["timeout_s"]))


async def one(client, cfg, sem, case: dict, out_dir: Path) -> tuple[str, dict | None]:
    import httpx
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
    cid = case["case_id"]
    lat = case["latent"]
    gold = lat["rare_hpo_gold"]
    doc = cached(case, out_dir)
    if doc is not None:
        return cid, doc
    leak = _leak_terms(lat["rare_spec_id"])
    raw = case["raw"]
    feedback = ""
    async with sem:
        for attempt in range(int(cfg["max_retries"]) + 1):
            prompt = PROMPT.format(sex="male" if raw.get("sex") == "M" else "female", age=raw.get("age_range") or "adult",
                                   findings=_findings(gold), feedback=feedback)
            async for r in AsyncRetrying(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30), reraise=True,
                                         retry=retry_if_exception_type((httpx.HTTPError, httpx.HTTPStatusError))):
                with r:
                    resp = await client.post("/chat/completions", json={
                        "model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}})
                    resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            try:
                doc = json_from(content)
            except json.JSONDecodeError as e:
                feedback, doc = f"\nYour previous answer was not valid JSON ({e}). Return only the JSON object.", None
                continue
            errs = check(doc, len(gold), leak)
            if not errs:
                p = out_dir / cid / "llm.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
                return cid, doc
            log.info("[%s] attempt %d rejected: %s", cid, attempt + 1, errs[:3])
            feedback = "\nYour previous answer was rejected: " + "; ".join(errs) + ". Fix these and return the full JSON again."
    log.warning("[%s] no acceptable report after %d attempts", cid, int(cfg["max_retries"]) + 1)
    return cid, None


async def _ask_all(cfg: dict, cases: list[dict], out_dir: Path) -> dict[str, dict]:
    from tqdm.asyncio import tqdm_asyncio
    sem = asyncio.Semaphore(int(cfg["concurrency"]))
    done: dict[str, dict] = {}
    async with _client(cfg) as client:
        for fut in tqdm_asyncio.as_completed([one(client, cfg, sem, c, out_dir) for c in cases], total=len(cases)):
            cid, doc = await fut
            if doc is not None:
                done[cid] = doc
    return done


def holdout_split(case_ids: list[str]) -> tuple[list[str], list[str]]:
    """Dev / test halves by case id: the first half of the ids in sorted order is dev."""
    ids = sorted(case_ids, key=lambda c: [int(x) if x.isdigit() else x for x in __import__("re").split(r"(\d+)", c)])
    k = len(ids) // 2
    return ids[:k], ids[k:]


def write_pack(src: dict, src_label: str, job_id: str, docs: dict[str, dict], out_dir: Path,
               writer: str, out: Path, split: bool = False) -> dict:
    """Write each case's Markdown next to its cache and the job file with the new narrative
    pointers; everything else in the source job is kept."""
    from haenv.canary import block as canary_block
    cases = []
    for c in src["cases"]:
        cid = c["case_id"]
        md, spans = assemble(cid, docs[cid], c["latent"]["rare_hpo_gold"])
        p = out_dir / cid / f"{cid}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        att = dict(c["latent"]["rare_attachments"])
        att["narrative"] = {"path": str(p), "sha256": hashlib.sha256(md.encode()).hexdigest(), "spans": spans,
                            "n_gold_spans": sum(1 for s in spans if s["hpo_id"]),
                            "n_noise": sum(1 for s in spans if not s["hpo_id"]), "writer": writer}
        cases.append({**c, "latent": {**c["latent"], "rare_attachments": att}})
    doc = {**src, "job_id": job_id, "report": f"eval-{job_id}.md", "cases": cases}
    head = (f"# Held-out English narratives written by {writer} from the gold of {src_label} —\n"
            f"# generated by haenv-rare-paraphrase (haenv_rare/haenv_rare/paraphrase.py), do not edit. "
            f"SYNTHETIC, evaluation only.\n")
    if split:
        dev, test = holdout_split([c["case_id"] for c in src["cases"]])
        head += (f"# Held-out split by case id: dev = {' '.join(dev)} | test = {' '.join(test)}\n"
                 f"# Parser changes are tuned on the dev half only; the test half is evaluated exactly once.\n")
    from .gen_job import BATCH_GATE_NOTE
    out.write_text(canary_block("# ") + head + BATCH_GATE_NOTE + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200),
                   encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="haenv-rare-paraphrase")
    ap.add_argument("--src", required=True, help="source job yaml (its gold, attachments and judges are kept)")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--attach-root", default="", help="where llm.json caches and the Markdown live "
                                                      "(default derived/rare_attachments-<job-id>)")
    ap.add_argument("--out", default="", help="job file to write (default inputs/<job-id>.job.yaml)")
    ap.add_argument("--cache-only", action="store_true",
                    help="rebuild from cached reports only; never open a client, fail if any case has none")
    ap.add_argument("--model", default="", help="writer model for this pack (overrides the settings)")
    ap.add_argument("--holdout-split", action="store_true",
                    help="record the dev / test halves by case id in the pack header")
    a = ap.parse_args(argv)
    cfg = _cfg()
    if a.model:
        cfg["model"] = a.model
    logging.basicConfig(level={0: logging.WARNING, 1: logging.INFO}.get(int(cfg.get("verbose", 1)), logging.DEBUG),
                        format="%(asctime)s %(levelname)s %(message)s")
    src_path = Path(a.src) if Path(a.src).is_absolute() else ROOT / a.src
    src = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    out_dir = Path(a.attach_root).resolve() if a.attach_root else ROOT / "derived" / f"rare_attachments-{a.job_id}"
    out = Path(a.out) if a.out else ROOT / "inputs" / f"{a.job_id}.job.yaml"
    if a.cache_only:
        done = {c["case_id"]: d for c in src["cases"] if (d := cached(c, out_dir)) is not None}
    else:
        done = asyncio.run(_ask_all(cfg, src["cases"], out_dir))
    missing = [c["case_id"] for c in src["cases"] if c["case_id"] not in done]
    if missing:
        log.error("no accepted report for %s%s", missing,
                  " (--cache-only: nothing written)" if a.cache_only else "; rerun to retry only these")
        return 1
    doc = write_pack(src, a.src, a.job_id, done, out_dir, cfg["model"], out, split=a.holdout_split)
    nar = [c["latent"]["rare_attachments"]["narrative"] for c in doc["cases"]]
    print(f"wrote {out} ({len(nar)} cases, {sum(x['n_gold_spans'] for x in nar)} gold spans + "
          f"{sum(x['n_noise'] for x in nar)} noise sentences, narratives by {cfg['model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
