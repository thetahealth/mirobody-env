"""Rare retrieval judge and leak probe: negative and positive controls
(docs/decisions/2026-09-30-rare-检索题.md §3).

    uv run python tools/rare_retrieval_selftest.py [--job inputs/rare_retrieval-p6p7.job.yaml]

On every question of the pack: the gold answer must grade correct (positive), and each kind's
corrupted answers (onset +1 s, genotype 0/1 -> 1/1, one image more, origin flipped, one row
dropped, no answer) must grade wrong (negative). The leak probe must drop a question whose answer
is written into the visible text, and the solver-visible block must carry no gold. Exit code 1
when any control misses. SYNTHETIC data, evaluation only.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "haenv_rare"))
from haenv_rare import retrieval as RR  # noqa: E402

_FLIP = {"maternal": "paternal", "paternal": "maternal", "biparental": "de_novo", "de_novo": "maternal"}


def corruptions(kind: str, g: dict) -> list[tuple[str, dict | None]]:
    """(label, a wrong answer) for this gold; each must grade wrong."""
    out: list[tuple[str, dict | None]] = [("no answer", None), ("not an object", "[]")]  # type: ignore[list-item]
    g = copy.deepcopy(g)
    if kind == "eeg_seizure":
        if g["seizures"]:
            a = copy.deepcopy(g); a["seizures"][0]["onset_sec"] += 1; out.append(("onset +1 s", a))
            a = copy.deepcopy(g); a["seizures"][0]["duration_sec"] += 1; out.append(("duration +1 s", a))
            a = copy.deepcopy(g); a["seizures"][0]["file"] = "other.edf"; out.append(("wrong file", a))
            a = copy.deepcopy(g); a["seizures"].pop(); out.append(("one seizure dropped", a))
        a = copy.deepcopy(g); a["seizures"].append({"file": "eeg-9.edf", "onset_sec": 1, "duration_sec": 1}); out.append(("one seizure added", a))
    elif kind == "sleep_stages":
        k = max(g, key=lambda x: g[x])
        a = dict(g); a[k] += 2; out.append((f"{k} +2 s", a))
        a = dict(g); a.pop(k); out.append((f"{k} missing", a))
    elif kind == "imaging":
        a = copy.deepcopy(g); a["series"][0]["images"] += 1; out.append(("one image more", a))
        a = copy.deepcopy(g); a["series"].append(dict(a["series"][0])); out.append(("one series more", a))
        a = copy.deepcopy(g); a["series"][0]["modality"] = "CT" if a["series"][0]["modality"] != "CT" else "MR"; out.append(("modality changed", a))
    elif kind == "variants":
        if g["variants"]:
            a = copy.deepcopy(g); v = a["variants"][0]; v["genotype"] = "1/1" if v["genotype"] != "1/1" else "0/1"; out.append(("genotype changed", a))
            a = copy.deepcopy(g); a["variants"][0]["pos"] += 1; out.append(("position +1", a))
            a = copy.deepcopy(g); a["variants"].pop(0); out.append(("planted call dropped", a))
    elif kind == "inheritance":
        a = copy.deepcopy(g); a["variants"][0]["origin"] = _FLIP[a["variants"][0]["origin"]]; out.append(("origin flipped", a))
        a = copy.deepcopy(g); a["variants"][0]["origin"] = "unknown"; out.append(("origin unknown", a))
    return out


def equivalents(kind: str, g: dict) -> list[tuple[str, dict]]:
    """(label, a differently written right answer); each must grade correct."""
    out = [("gold", copy.deepcopy(g))]
    if kind == "imaging":
        a = copy.deepcopy(g); a["series"] = [dict(s, modality="MRI" if s["modality"] == "MR" else s["modality"]) for s in reversed(a["series"])]
        out.append(("MRI spelling, reversed order", a))
    if kind == "variants" and g["variants"]:
        a = copy.deepcopy(g); v = a["variants"][0]; v["chrom"] = "chr" + v["chrom"]; v["genotype"] = v["genotype"].replace("/", "|")[::-1]
        a["variants"].append({"chrom": "1", "pos": 1, "ref": "A", "alt": "G", "genotype": "0/1"})
        out.append(("chr prefix, phased, one unplanted extra", a))
    if kind == "sleep_stages":
        out.append(("within 1 s", {k: v + 0.9 for k, v in g.items()}))
    if kind == "eeg_seizure" and g["seizures"]:
        a = copy.deepcopy(g); a["seizures"] = list(reversed(a["seizures"])); a["seizures"][0]["onset_sec"] += 0.4
        out.append(("reordered, onset within 0.5 s", a))
    return out


def leak_controls() -> list[tuple[str, bool]]:
    att = {"eeg": {"recordings": [{"path": "/x/eeg-1.edf", "phi_expected": False, "annotations": [
        {"onset_sec": 1234.0, "duration_sec": 56.0, "text": "Seizure"},
        {"onset_sec": 0.0, "duration_sec": 30.0, "text": "Sleep stage 2"}]}]},
        "genome": {"files": {r: {"path": f"/x/{r}.vcf.gz"} for r in ("proband", "father", "mother")},
                   "variant": {"chrom": "2", "pos": 166051974, "ref": "CA", "alt": "C"},
                   "genotypes": {"proband": "0/1", "father": "0/0", "mother": "0/0"}, "decoys": []}}
    clean = RR.build(att, "en", "no numbers here")
    kinds = {q["kind"] for q in clean["questions"]}
    res = [("clean text emits every kind", kinds == {"eeg_seizure", "sleep_stages", "variants", "inheritance"})]
    got = RR.build(att, "en", "a seizure at 1234 seconds")
    res.append(("seizure onset in the text drops the seizure question", "eeg_seizure" not in {q["kind"] for q in got["questions"]}))
    got = RR.build(att, "en", "position 166051974 on chromosome 2")
    res.append(("variant position in the text drops variants and inheritance",
                not {"variants", "inheritance"} & {q["kind"] for q in got["questions"]}))
    got = RR.build(att, "zh", "该变异为新发。")
    res.append(("an inheritance cue drops the inheritance question", "inheritance" not in {q["kind"] for q in got["questions"]}))
    got = RR.build(att, "en", "age 2, 1 sibling")
    res.append(("a number below 10 does not drop", {q["kind"] for q in got["questions"]} == kinds))
    return res


def judge_controls(case: dict) -> list[tuple[str, bool]]:
    rr = case["latent"]["rare_retrieval"]

    class VP:
        adjudication = {"rare": {"retrieval": rr}}

    class Out:
        _raw = {"retrieval_answers": {q["qid"]: {"answer": rr["gold"][q["qid"]], "tools": ["query_signal_index"]} for q in rr["questions"]}}
    r = RR.judge_rare_retrieval(Out, VP)
    res = [("all gold, tools used: rr_correct 1, tool_use 1", r["rr_correct"] == 1 and r["rr_tool_use"] == 1 and r["rr_parse_ok"] == 1)]
    Out._raw = {"retrieval_answers": {q["qid"]: {"answer": rr["gold"][q["qid"]], "tools": ["ls"]} for q in rr["questions"]}}
    r = RR.judge_rare_retrieval(Out, VP)
    res.append(("gold without a retrieval tool: tool_use 0", r["rr_tool_use"] == 0))
    Out._raw = {}
    r = RR.judge_rare_retrieval(Out, VP)
    res.append(("no answers: rr_correct 0, parse_ok 0", r["rr_correct"] == 0 and r["rr_parse_ok"] == 0))

    class VP0:
        adjudication = {"rare": {}}
    res.append(("no questions: not applicable", RR.judge_rare_retrieval(Out, VP0)["rr_na_reason"] == "no_retrieval_questions"))
    return res


def probe_controls(case: dict) -> list[tuple[str, bool]]:
    """The whole solver-visible side of a retrieval case, as haenv assembles it."""
    import haenv_rare as HR
    from haenv import external_gold as EG
    HR._ensure_task_registered()
    meta = {EG.SLOT: {k: v for k, v in case["latent"].items() if k.startswith("rare_")}}
    blk = EG.probe_blocks_for(meta)
    s = json.dumps(blk, ensure_ascii=False)
    rr = case["latent"]["rare_retrieval"]
    res = [("the questions sit at prediction_context.retrieval", len((blk.get("retrieval") or {}).get("questions") or []) == len(rr["questions"])),
           ("no gold, no dropped list, no coding task note, nothing under `rare`",
            '"gold"' not in s and '"dropped"' not in s and "coding_task" not in blk and "rare" not in blk)]
    for qid, g in rr["gold"].items():
        for v in (g.get("variants") or []):
            if str(v["pos"]) in s:
                res.append((f"{qid}: gold position {v['pos']} in the solver-visible side", False))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=str(ROOT / "inputs/rare_retrieval-p6p7.job.yaml"))
    a = ap.parse_args()
    cases = yaml.safe_load(Path(a.job).read_text(encoding="utf-8"))["cases"]
    n = miss = 0
    fails = []
    for c in cases:
        rr = c["latent"]["rare_retrieval"]
        for q in rr["questions"]:
            g = rr["gold"][q["qid"]]
            for label, ans in equivalents(q["kind"], g):
                n += 1
                if not RR.grade(q["kind"], g, ans)["correct"]:
                    miss += 1; fails.append(f"{c['case_id']} {q['qid']}: positive '{label}' graded wrong")
            for label, ans in corruptions(q["kind"], g):
                n += 1
                if RR.grade(q["kind"], g, ans)["correct"]:
                    miss += 1; fails.append(f"{c['case_id']} {q['qid']}: negative '{label}' graded correct")
    extra = leak_controls() + judge_controls(cases[0]) + probe_controls(cases[0])
    for label, ok in extra:
        n += 1
        if not ok:
            miss += 1; fails.append(f"control '{label}' failed")
    for f in fails[:30]:
        print("  MISS", f)
    print(f"rare retrieval selftest: {n} controls, {miss} missed")
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
