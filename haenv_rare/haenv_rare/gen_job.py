"""gen_job.py(haenv_rare)—— 由包内 `data/conditions_rare.yaml` 出罕见病编码题包(job.yaml)+ 附件。

    uv run haenv-rare-gen                                     # 写 inputs/rare_coding-p0.job.yaml + derived/rare_attachments/
    uv run haenv-rare-gen --variants 3 --job-id rare_coding-p2
    uv run haenv-rare-gen --refresh inputs/rare_coding-p1.job.yaml   # re-derive an existing pack, keeping attachments
    uv run haenv-rare-gen --job-id rare_coding-p4-en --lang en --variant 2 --attach-root derived/rare_attachments-p4-en ...

金标先行:每例的 HPO 术语集 / ORPHA / 基因 / 变异 / 遗传模式全由 `haenv_rare.gen` 按 case_id 确定性推出;
本脚本只是把它们排成 job.yaml 的 `raw` + `latent`。**本文件的产物由本脚本生成,不要手改。**

SYNTHETIC,仅评测用,非医疗建议。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

import haenv as _haenv                              # noqa: E402
# Repo root of the haenv checkout: jobs are written to its `inputs/`, as the tool did from `tools/`.
ROOT = Path(_haenv.__file__).resolve().parent.parent
from haenv import kernel_path as _kernel_path       # noqa: E402
if (_kp := _kernel_path()):
    sys.path.insert(0, str(_kp))

from haenv import rng                                # noqa: E402
from haenv.ddx import _course_end_for, _density_knobs, _t_index_for, variant_id  # noqa: E402
from haenv.demographics import doses_per_week, sample_profile                    # noqa: E402
from haenv_rare.gen import (Ontology, build_attachments, case_triage, gold_in_lang, load_specs,  # noqa: E402
                            render_symptoms, sample_gold)

HEADER = ("# 罕见病编码题包 —— 由 haenv-rare-gen(haenv_rare/haenv_rare/gen_job.py)从包内 data/conditions_rare.yaml 生成,**不要手改**。\n"
          "# 金标(HPO 术语集 / ORPHA / 基因 / 变异 / 遗传模式)由 haenv_rare/haenv_rare/gen.py 按 case_id 确定性推出;\n"
          "# 附件(VCF/PED/DICOM 指针)在 derived/rare_attachments/<case_id>/,latent.rare_attachments 记 sha256+路径。\n"
          "# 跑:  uv run haenv build inputs/<本文件> --gen deterministic\n"
          "# SYNTHETIC,仅评测用,非医疗建议。\n")

#: Every rare pack carries this note: the batch gate below is overridden when the pack is run,
#: and why. Written into the pack header so the reason travels with the file.
BATCH_GATE_NOTE = (
    "# Batch gate overridden when run (`haenv run ... --override-batch-gate`, recorded in batch.json):\n"
    "#   footprint_discriminates_real_symptom -- the HPO-sampled rare symptoms carry no\n"
    "#   registry/symptom_topics.yaml entry by construction, so only benign events get kernel\n"
    "#   footprints. Every per-case gate (haenv's and the plugin's R1-R12) still applies.\n")


def with_imaging(spec: dict, sid: str, idx: int, collections: list[str] | None) -> dict:
    """A spec whose `modalities.imaging` is `none` gets a TCIA collection from
    `rare_narrative.pair_imaging`: the matching collection where one exists (LFS ↔ sarcoma,
    neurological disorders ↔ brain MR, tier `exact_class` / `region_match`), otherwise an
    explicitly `unrelated_attachment` one — never a match claimed that is not there.
    A spec that already names a collection is returned untouched, and so is every spec when
    no collections are given: the registry stays the source of truth for the default job."""
    if not collections:
        return spec
    im = (spec.get("modalities") or {}).get("imaging")
    if im and im != "none":
        return spec
    from haenv_rare.narrative import pair_imaging
    out = dict(spec)
    out["modalities"] = {**(spec.get("modalities") or {}), "imaging": pair_imaging(sid, idx, collections)}
    return out


def case_entry(sid: str, spec: dict, anon: str, v: int, sex: str, gold: list[dict], symptoms: list[dict],
               prof: dict, att: dict, lang: str, T: int, weight: tuple[str, float, float] | None = None,
               extra_latent: dict | None = None) -> dict:
    """One job case (`raw` + `latent`) for a sampled gold set. `weight` overrides the spec's weight
    course (round 2: age-appropriate weights for paediatric cases); `extra_latent` adds keys."""
    ont = Ontology.get()
    kind, start, end = weight or spec["weight"]
    sc = 1.0 if (v == 1 or weight is not None) else 0.80 + 0.40 * rng.unit(anon, "variant_weight_scale")
    start, end = float(start) * sc, float(end) * sc
    mpw, sym_rate, life_rate = _density_knobs(anon, None, int(T))
    red_flag, urgency = case_triage(spec, gold)
    orpha = str(spec["orpha"])
    return {
        "case_id": anon,
        "raw": {**prof, "sex": sex,
                "start_weight": round(start, 1), "nadir_weight": round(min(start, end), 1),
                "symptoms": symptoms},
        "latent": {
            "index_time_T": int(T),
            "course_end_day": _course_end_for(anon, int(T)),
            "outcome": "regain" if kind == "up" else "maintain",
            **({"regain_end_kg": round(end, 1)} if kind == "up" else {}),
            "driver": "unknown_or_multifactorial",
            # ---- ddx shape (the built-in ddx judges run as well) ----
            "ddx_spec_id": sid,
            "ddx_diagnosis": spec["diagnosis"],
            "ddx_aliases": list(spec["aliases"]),
            "ddx_join_gold": "unified",
            "ddx_threads": None,
            "ddx_tests": list(spec["tests"]),
            "ddx_specialty": list(spec["specialty"]),
            "ddx_urgency": urgency,
            "ddx_red_flag": red_flag,
            "ddx_clinician_warranted": bool(spec["clinician_warranted"]),
            "ddx_outcome_label": spec["outcome_label"],
            # ---- rare coding gold ----
            "rare_spec_id": sid,
            "rare_orpha": f"ORPHA:{orpha}",
            "rare_omim": list(ont.orpha[orpha]["omim"]),
            "rare_icd10": list(ont.orpha[orpha]["icd10"]),
            "rare_gene": spec["gene"],   # None = acquired (NMOSD/MG) or no monogenic gene
            "rare_inheritance": spec["inheritance"],
            "rare_hpo_gold": gold,
            "rare_attachments": att,
            "rare_dx_in_text": False,
            # Written only for a non-default language, so the zh packs keep their keys.
            **({"rare_lang": lang} if lang != "zh" else {}),
            "rare_negation_density": sum(1 for g in gold if g["polarity"] == "absent") / max(1, len(gold)),
            "rare_family_noise": sum(1 for g in gold if g["subject"] == "relative"),
            **(extra_latent or {}),
            "event_density": {"measure_per_week": mpw,
                              "dosing_per_week": doses_per_week(prof["drug"]),
                              "symptom_rate": sym_rate, "life_event_rate": life_rate,
                              "clinical_symptoms_recorded": len(symptoms),
                              "course_weeks": round(_course_end_for(anon, int(T)) / 7.0, 1)},
        },
    }


def rare_case_specs(variants: int = 1, only: list[str] | None = None, attachments: bool = True,
                    imaging: list[str] | None = None, lang: str = "zh", variant_only: int | None = None) -> list[dict]:
    specs = load_specs()
    ont = Ontology.get()
    out: list[dict] = []
    for idx, (sid, spec) in enumerate(item for item in specs.items() if not only or item[0] in only):
        spec = with_imaging(spec, sid, idx, imaging)
        for v in ([variant_only] if variant_only else range(1, max(1, variants) + 1)):
            anon = variant_id(spec["case_id"], v)
            sex = rng.pick(["F", "M"], anon, "rare", "sex")
            # 需要 trio 骨架的例子:GIAB 三个 trio 先证者全是男性(HG001 是唯一女性单样本)
            gm = (spec.get("modalities") or {}).get("genome")
            if gm and gm != "none" and gm.get("skeleton") == "trio":
                sex = "M"
            T = _t_index_for(anon)
            gold = gold_in_lang(sample_gold(spec, anon, sex), lang)      # selection on zh labels; surface per lang
            symptoms = render_symptoms(gold, anon, int(T), lang=lang)
            prof = {k: val for k, val in sample_profile(anon, sex).items() if k != "sex"}
            att = build_attachments(spec, anon, sex, gold, profile={**prof, "sex": sex}, lang=lang) if attachments else {}
            out.append(case_entry(sid, spec, anon, v, sex, gold, symptoms, prof, att, lang, int(T)))
    return out


def refresh_cases(path: Path) -> tuple[dict, list[dict]]:
    """Re-derive every case of an existing rare job from the registry, in its order.
    `latent.rare_attachments` is carried over from the file: the attachments are built from
    VCF / DICOM sources on the packing machine and are not rebuilt here. Job-level fields are
    returned as they are in the file, and the pack keeps its language (`rare_lang`)."""
    old = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [c["case_id"] for c in old["cases"]]
    specs = load_specs()
    by_cid = {s["case_id"]: sid for sid, s in specs.items()}
    only = sorted({by_cid[i.split("v")[0]] for i in ids})
    variants = max(int(i.split("v")[1]) if "v" in i else 1 for i in ids)
    lang = str((old["cases"][0].get("latent") or {}).get("rare_lang") or "zh")
    fresh = {c["case_id"]: c for c in rare_case_specs(variants, only, attachments=False, lang=lang)}
    out = []
    for c in old["cases"]:
        n = fresh[c["case_id"]]
        n["latent"]["rare_attachments"] = c["latent"].get("rare_attachments") or {}
        out.append(n)
    return {k: v for k, v in old.items() if k != "cases"}, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default="rare_coding-p0")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-attachments", action="store_true")
    ap.add_argument("--imaging", default="", metavar="COLL,COLL",
                    help="give every spec without imaging a TCIA collection, round-robin over this list (03_dicom/<COLL>)")
    ap.add_argument("--refresh", default="", metavar="JOB",
                    help="re-derive the cases of an existing rare job in place, keeping its attachments")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="language of symptom sentences, narrative and task note")
    ap.add_argument("--variant", type=int, default=0, help="emit ONLY this variant of each spec (2 -> JD-50v2 ...): new patients, same diseases")
    ap.add_argument("--attach-root", default="", help="attachment directory (default derived/rare_attachments; give a separate one "
                                                         "for a package whose case ids already have attachments)")
    a = ap.parse_args()
    if a.attach_root:
        os.environ["HAENV_RARE_ATTACH_ROOT"] = str(Path(a.attach_root).resolve())
    if a.refresh:
        head, cases = refresh_cases(Path(a.refresh))
        job_id = head["job_id"]
    else:
        head = None
        job_id = a.job_id
        cases = rare_case_specs(a.variants, [x for x in a.only.split(",") if x], attachments=not a.no_attachments,
                                imaging=[x for x in a.imaging.split(",") if x], lang=a.lang, variant_only=a.variant or None)
    doc = {"job_id": job_id, "task_type": "joint_dx", "multiround": False,
           "include_baseline": True, "models": [], "sample_cases": 3,
           "report": f"eval-{job_id}.md",
           "plugins": ["haenv.judges.rare"],
           "cases": cases}
    if head is not None:
        doc = {**head, "cases": cases}
    path = ROOT / "inputs" / f"{job_id}.job.yaml"
    # Every answer-bearing file carries the canary block first (haenv.canary; guarded by
    # test_c2_every_answer_bearing_file_has_the_canary).
    from haenv.canary import block as canary_block
    path.write_text(canary_block("# ") + HEADER + BATCH_GATE_NOTE + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200),
                    encoding="utf-8")
    n_g = sum(len(c["latent"]["rare_hpo_gold"]) for c in cases)
    n_att = sum(1 for c in cases if (c["latent"]["rare_attachments"] or {}).get("genome"))
    n_img = sum(1 for c in cases if (c["latent"]["rare_attachments"] or {}).get("imaging"))
    n_nar = sum(1 for c in cases if (c["latent"]["rare_attachments"] or {}).get("narrative"))
    n_span = sum((c["latent"]["rare_attachments"] or {}).get("narrative", {}).get("n_gold_spans", 0) for c in cases)
    n_noise = sum((c["latent"]["rare_attachments"] or {}).get("narrative", {}).get("n_noise", 0) for c in cases)
    tiers = {}
    for c in cases:
        t = ((c["latent"]["rare_attachments"] or {}).get("imaging") or {}).get("tier")
        tiers[t] = tiers.get(t, 0) + 1
    print(f"写出 {path}({len(cases)} 例,{n_g} 条 gold 术语,{n_att} 例带基因组附件,{n_img} 例带影像 {tiers},"
          f"{n_nar} 例带叙述病历:{n_span} 条金标 span + {n_noise} 条噪声句)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
