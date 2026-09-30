"""haenv-rare-round2 -- round 2 of the rare-coding benchmark: 60 new gold cases (JD-100..JD-159)
from the staged rare-os data, written as rare_coding-p6-zh (40) and rare_coding-p7-en (40).

    uv run haenv-rare-round2 [--attach-root derived/rare_attachments-r2] [--workers 8] [--plan-only]

What is built comes from `data/round2.yaml`; `allocate()` turns it into one plan per case,
deterministically (case id = seed) and without writing anything: language, sex, age, EEG
recordings, background genome, imaging. Gold is sampled from the spec and case id before any
rendering. `build_case()` then writes the case's attachments; it is a pure function of the plan,
so cases are built in parallel, and a case whose `.plan.sha256` marker matches its plan is
reused on a rerun (only what is missing is rebuilt). Every file's sha256 goes into the job and
into `<attach-root>/SHA256SUMS`.
SYNTHETIC evaluation data only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor

import yaml

from haenv import rng

from . import edf as E
from . import genome_os as GO
from .gen import (ROOT, Ontology, RareSpecError, _genotypes, _sha256, gold_in_lang, load_specs,
                  pick_clean_dicom_patient, rare_data_root, render_symptoms, sample_gold, select_series,
                  write_ped)

_HERE = pathlib.Path(__file__).resolve().parent
PACKS = {"zh": "rare_coding-p6-zh", "en": "rare_coding-p7-en"}
EDFPLUS_STAGE = {"W": "Sleep stage W", "S1": "Sleep stage 1", "S2": "Sleep stage 2", "S3": "Sleep stage 3",
                 "S4": "Sleep stage 4", "R": "Sleep stage R", "MT": "Movement time"}
_SYNTH_NAMES = ("Test_Alder", "Test_Birch", "Test_Cedar", "Test_Dogwood", "Test_Elm", "Test_Fir", "Test_Gum")


def load_round() -> dict:
    return yaml.safe_load((_HERE / "data" / "round2.yaml").read_text(encoding="utf-8"))


def attach_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HAENV_RARE_R2_ROOT") or ROOT / "derived" / "rare_attachments-r2")


def age_band(age: float) -> str:
    """Five-year band of an age in years; under five reads `1-4` (the youngest staged patient is 1.5)."""
    lo = int(age) // 5 * 5
    return f"{max(1, lo)}-{lo + 4}"


def _eeg_inventory() -> dict:
    root = GO.rare_os_root() / "datasets"
    chb = [json.loads(l) for l in (root / "edf_chbmit" / "annotations.jsonl").open(encoding="utf-8")]
    cap = [json.loads(l) for l in (root / "edf_capslp" / "annotations.jsonl").open(encoding="utf-8")]
    patients: dict[str, dict] = {}
    for r in chb:
        p = patients.setdefault(r["patient"], {"patient": r["patient"], "age": float(r["patient_age_years"]),
                                               "sex": r["patient_sex"], "files": {}})
        p["files"][r["kind"]] = r
    return {"chbmit": patients, "capslp": {r["recording"]: r for r in cap}}


def _remind_patients() -> dict[str, dict]:
    import csv
    root = GO.rare_os_root() / "datasets" / "dicom_tcia_remind"
    out: dict[str, dict] = {}
    with open(root / "series.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            p = out.setdefault(r["patient"], {"patient": r["patient"], "histology": r["histopathology"],
                                              "who_grade": r["who_grade"], "idh_mutant": r["idh_mutant"],
                                              "age": r["age"], "sex": r["sex"], "series": []})
            p["series"].append({"path": str(root / r["zip_file"]), "sha256": r["zip_sha256"], "series_uid": r["series_uid"],
                                "modality": r["modality"], "n_instances": int(r["n_instances"]),
                                "bytes": int(r["zip_bytes"]), "role": r["series_role"]})
    return out


# ---------------------------------------------------------------- allocation (no writes)
def _sex_for(spec: dict, anon: str) -> str:
    if spec.get("sex"):
        return spec["sex"]
    if spec["inheritance"] == "XLR":
        return "M"
    return rng.pick(["F", "M"], anon, "rare", "sex")


def allocate(cfg: dict | None = None) -> list[dict]:
    """One plan per case, in id order. Pure: same data, same plans."""
    cfg = cfg or load_round()
    specs = load_specs()
    inv = _eeg_inventory()
    order: list[dict] = []
    i = 0
    for ent in cfg["specs"]:
        for _ in range(int(ent["n"])):
            anon = f"JD-{int(cfg['first_id']) + i}"
            order.append({"case_id": anon, "index": i, "nth": sum(1 for p in order if p["spec_id"] == ent["spec"]),
                          "spec_id": ent["spec"], "eeg_source": ent.get("eeg"),
                          "external": ent.get("external"), "lang": cfg["lang_cycle"][i % len(cfg["lang_cycle"])]})
            i += 1
    # ---- EEG: sex-pinned CHB-MIT specs choose first, so the female-only specs are not starved
    used_chb: set[str] = set()
    used_cap: set[str] = set()
    for plan in sorted((p for p in order if p["eeg_source"] == "chbmit"),
                       key=lambda p: (specs[p["spec_id"]].get("sex") is None, p["index"])):
        want = specs[plan["spec_id"]].get("sex")
        pool = sorted(k for k, v in inv["chbmit"].items() if k not in used_chb and (not want or v["sex"] == want))
        if not pool:
            raise RareSpecError(f"{plan['case_id']}: no CHB-MIT patient left for sex {want}")
        pt = inv["chbmit"][rng.pick(pool, plan["case_id"], "rare", "r2", "chbmit")]
        used_chb.add(pt["patient"])
        plan["sex"], plan["age"] = pt["sex"], pt["age"]
        plan["eeg"] = [{"dataset": "chbmit", "kind": k, "meta": pt["files"][k], "mode": cfg["eeg"]["chbmit_modes"][k]}
                       for k in ("seizure", "no_seizure")]
    for plan in (p for p in order if str(p["eeg_source"] or "").startswith("capslp/")):
        grp = plan["eeg_source"].split("/", 1)[1]
        pool = sorted(k for k, v in inv["capslp"].items() if v["diagnosis_group"] == grp and k not in used_cap)
        if not pool:
            raise RareSpecError(f"{plan['case_id']}: no CAP {grp} recording left")
        rec = inv["capslp"][rng.pick(pool, plan["case_id"], "rare", "r2", "capslp")]
        used_cap.add(rec["recording"])
        plan["sex"], plan["age"] = rec["patient_sex"], float(rec["patient_age_years"])
        plan["eeg"] = [{"dataset": "capslp", "kind": grp, "meta": rec, "mode": cfg["eeg"]["capslp_mode"]}]
    ctl = cfg["eeg"]["controls"]
    ctl_pool = sorted(k for k, v in inv["capslp"].items() if v["diagnosis_group"] in ctl["groups"] and k not in used_cap)
    for plan in (p for p in order if p["spec_id"] in ctl["specs"]):
        if not ctl_pool:
            break
        k = rng.pick(ctl_pool, plan["case_id"], "rare", "r2", "control")
        ctl_pool.remove(k)
        rec = inv["capslp"][k]
        plan["eeg"] = [{"dataset": "capslp", "kind": "control", "meta": rec, "mode": ctl["modes"][rec["diagnosis_group"]]}]
    k_rec = 0
    for plan in order:
        for r in plan.get("eeg") or []:
            k_rec += 1
            r["phi"] = k_rec % int(cfg["eeg"]["phi_every"]) == 0
    # ---- sex / age for the rest
    for plan in order:
        spec = specs[plan["spec_id"]]
        plan.setdefault("sex", _sex_for(spec, plan["case_id"]))
        plan.setdefault("age", None)
    # ---- genome background
    bgs = GO.backgrounds()
    use: dict[str, int] = {b["id"]: 0 for b in bgs}
    zh_sup = set(cfg["genome"]["zh_superpopulations"])
    for plan in order:
        spec = specs[plan["spec_id"]]
        gm = (spec.get("modalities") or {}).get("genome")
        if plan["external"] or not gm or gm == "none":
            plan["background"] = None
            continue
        east = plan["lang"] in ("both", "zh")
        pool = [b for b in bgs if (b["superpopulation"] in zh_sup) == east and b["child_sex"] == plan["sex"]]
        if not pool:
            raise RareSpecError(f"{plan['case_id']}: no background trio for sex {plan['sex']}")
        low = min(use[b["id"]] for b in pool)
        least = sorted(b["id"] for b in pool if use[b["id"]] == low)
        pick = rng.pick(least, plan["case_id"], "rare", "r2", "background")
        use[pick] += 1
        plan["background"] = next(b for b in bgs if b["id"] == pick)
    # ---- imaging
    rem = _remind_patients()
    used_img: set[tuple[str, str]] = set()
    region_pool = list(cfg["imaging"]["region_pool"])
    k_region = 0
    # exact (histology-matched) pairings first: they need specific patients
    for plan in order:
        sid = plan["spec_id"]
        if sid in cfg["imaging"]["exact"]:
            opts = cfg["imaging"]["exact"][sid]
            nth = sum(1 for p in order if p["spec_id"] == sid and p["index"] < plan["index"])
            plan["imaging"] = _pick_imaging(dict(opts[nth % len(opts)], tier="exact_class"), plan["case_id"], rem, used_img)
    for plan in order:
        sid = plan["spec_id"]
        choice = None
        if sid in cfg["imaging"]["exact"]:
            continue
        elif sid in cfg["imaging"]["region"]:
            # round-robin over the pool; a collection with no unused patient passes to the next one
            for j in range(len(region_pool)):
                coll = region_pool[(k_region + j) % len(region_pool)]
                try:
                    plan["imaging"] = _pick_imaging({"collection": coll, "tier": "region_match"}, plan["case_id"], rem, used_img)
                    break
                except RareSpecError:
                    continue
            else:
                raise RareSpecError(f"{plan['case_id']}: every region-pool collection is used up")
            k_region += 1
            continue
        if not choice:
            plan["imaging"] = None
            continue
        plan["imaging"] = _pick_imaging(choice, plan["case_id"], rem, used_img)
    return order


def _pick_imaging(choice: dict, anon: str, rem: dict, used: set) -> dict:
    coll = choice["collection"]
    if coll == "ReMIND":
        hist = choice.get("histology")
        pool = sorted(p for p, v in rem.items() if ("ReMIND", p) not in used
                      and (not hist or any(v["histology"].startswith(h) for h in hist)))
        if not pool:
            raise RareSpecError(f"{anon}: no ReMIND patient left for {hist}")
        pid = rng.pick(pool, anon, "rare", "r2", "remind")
        used.add(("ReMIND", pid))
        p = rem[pid]
        return {"tier": choice["tier"], "collection": "ReMIND", "tcia_patient_id": pid, "license": ["CC BY 4.0"],
                "deid_method": "DICOM PS3.15 Annex E (TCIA)",
                "histology": {k: p[k] for k in ("histology", "who_grade", "idh_mutant")},
                "series": [{k: s[k] for k in ("path", "sha256", "series_uid", "modality", "n_instances", "bytes")}
                           for s in p["series"]]}
    base = rare_data_root() / "03_dicom" / coll
    pats = sorted(x.name for x in base.iterdir() if x.is_dir())
    free = [x for x in pats if (coll, x) not in used]
    if not free:
        raise RareSpecError(f"{anon}: no {coll} patient left")
    pid = pick_clean_dicom_patient(base, free, rng.pick(free, anon, "rare", "r2", "dicom"), used=used)
    used.add((coll, pid))
    man = json.loads((rare_data_root() / "03_dicom" / "_manifests" / f"{coll}.series.json").read_text())
    lic = sorted({m.get("LicenseName", "") for m in man if m.get("PatientID") == pid})
    return {"tier": choice["tier"], "collection": coll, "tcia_patient_id": pid, "license": lic,
            "deid_method": "DICOM PS3.15 Annex E (TCIA)",
            "series": [{"path": str(z), "sha256": None, "series_uid": z.stem, "modality": m["modality"],
                        "n_instances": m["n"], "bytes": m["bytes"]} for z, m in select_series(base / pid)]}


# ---------------------------------------------------------------- gold (no writes)
def phenopacket_gold(anon: str) -> tuple[list[dict], dict]:
    """The Exomiser Pfeiffer example's phenotypes as gold (all present, proband), in a shuffled
    order like sampled gold; plus its source facts."""
    root = GO.rare_os_root() / "datasets" / "vcf_exomiser_examples"
    pp = json.loads((root / "pfeiffer-phenopacket.json").read_text(encoding="utf-8"))
    ont = Ontology.get()
    gold = []
    for f in pp["phenotypicFeatures"]:
        hp = ont.canon(f["type"]["id"]) or f["type"]["id"]
        gold.append({"hpo_id": hp, "label": ont.label(hp), "label_en": ont.name.get(hp, f["type"]["label"]),
                     "polarity": "present", "subject": "proband", "freq": "phenopacket"})
    order = sorted(range(len(gold)), key=lambda i: rng.counter_u32(anon, "rare", "order", gold[i]["hpo_id"], "proband"))
    gold = [gold[i] for i in order]
    for i, g in enumerate(gold):
        g["idx"] = i
    assembly = (pp.get("htsFiles") or [{}])[0].get("genomeAssembly")
    return gold, {"subject_sex": pp["subject"]["sex"], "assembly": assembly, "phenopacket": str(root / "pfeiffer-phenopacket.json")}


def plan_gold(plan: dict) -> list[dict]:
    spec = load_specs()[plan["spec_id"]]
    if plan["external"] == "exomiser_pfeiffer":
        return phenopacket_gold(plan["case_id"])[0]
    return sample_gold(spec, plan["case_id"], plan["sex"])


# ---------------------------------------------------------------- build (writes one case dir)
def _code_sha() -> str:
    """The builder code a cached case was made with: a code change rebuilds every case."""
    h = hashlib.sha256()
    for name in ("round2.py", "genome_os.py", "edf.py", "gen.py"):
        h.update((_HERE / name).read_bytes())
    h.update((_HERE / "data" / "conditions_rare.yaml").read_bytes())
    return h.hexdigest()


def _plan_key(plan: dict) -> str:
    view = {k: v for k, v in plan.items() if k not in ("index",)}
    return hashlib.sha256((_code_sha() + json.dumps(view, sort_keys=True, default=str)).encode()).hexdigest()


def _zygosity(inh: str, sex: str) -> str:
    return {"AR": "hom", "AD": "het", "de_novo": "het"}.get(inh) or ("hemi" if sex == "M" else "het")


def _norm_gt(gt: str) -> str:
    return gt.replace("|", "/")


def _pick(cands: list[dict], zyg: str, anon: str, tag: str, member_files: list[pathlib.Path],
          sex: str, avoid: set, spread: tuple[str, int] | None = None, male_het: bool = False) -> dict | None:
    """A rare candidate of `cands`. `spread=(spec_id, n)`: the n-th case of a spec starts n places
    after the spec's own pick, so two cases of one spec take different alleles when there are two.
    `male_het=True`: a male member (with a file) must be heterozygous here, so candidates where a
    male is haploid (non-PAR chrX / chrY) are not eligible."""
    free = GO.free_of_background([c for c in cands if (c["chrom"], c["pos"]) not in avoid], member_files)
    if male_het:
        free = [c for c in free if not GO.haploid_site(c["chrom"], c["pos"], "M")]
    if zyg == "hemi":
        free = [c for c in free if GO.haploid_site(c["chrom"], c["pos"], sex)]
    linked = [c for c in free if c["disease_linked"]] or free
    pool = sorted(linked, key=lambda c: int(c["vid"]) if c["vid"].isdigit() else 0)
    while pool:
        if spread:
            c = pool[(rng.below(len(pool), spread[0], "rare", "r2", tag) + spread[1]) % len(pool)]
        else:
            c = rng.pick(pool, anon, "rare", "r2", tag)
        g = GO.gnomad_af(c["chrom"], c["pos"], c["ref"], c["alt"])
        if GO.rare_enough(g, zyg):
            return {**c, "gnomad": g}
        pool = [x for x in pool if x is not c]
    return None


def _build_genome(plan: dict, spec: dict, out_dir: pathlib.Path, bg_cache: pathlib.Path, cfg: dict) -> dict:
    anon, sex, inh = plan["case_id"], plan["sex"], spec["inheritance"]
    bg = plan["background"]
    gts = _genotypes(inh, sex)
    trio = (spec["modalities"]["genome"]).get("skeleton") == "trio"
    roles = list(gts) if trio else ["proband"]
    src_sample = {"proband": bg["samples"]["child"], "father": bg["samples"]["father"], "mother": bg["samples"]["mother"]}
    ped_id = {"proband": f"{anon}-P", "father": f"{anon}-F", "mother": f"{anon}-M"}
    bg_files = {}
    for role in roles:
        cached = bg_cache / _cache_name(bg["id"], src_sample[role], trio)
        if not cached.exists():
            GO.member_background(bg["vcf"], src_sample[role], "SAMPLE", cached, joint=trio)
        bg_files[role] = cached
    style = GO.background_style(bg_files["proband"])
    ont = Ontology.get()
    orpha = str(spec["orpha"])
    zyg = _zygosity(inh, sex)
    cands = GO.clinvar_candidates(spec["gene"], orpha, ont.orpha[orpha]["omim"])
    tv = _pick(cands, zyg, anon, "variant", list(bg_files.values()), sex, set(), spread=(plan["spec_id"], plan["nth"]))
    if tv is None:
        raise RareSpecError(f"{anon}: no rare ClinVar P/LP >= 2-star variant of {spec['gene']} fits")
    # decoys: carriers of other AR genes of the table, and one rival-condition gene
    specs = load_specs()
    decoys = []
    avoid = {(tv["chrom"], tv["pos"])}
    ar_pool = sorted(s for s, sp in specs.items() if sp.get("gene") and sp["inheritance"] == "AR" and sp["gene"] != spec["gene"])
    # A decoy is heterozygous in the proband and its carrier parent. Where that parent is a father
    # with a file (trio), a non-PAR chrX / chrY candidate would need a heterozygous male: not
    # eligible, the decoy is drawn from the rest (a rival with no eligible candidate passes to the
    # next rival, in sorted order after the seeded one).
    for s_id in rng.subset(ar_pool, cfg["genome"]["decoys"]["carriers"], cfg["genome"]["decoys"]["carriers"], anon, "rare", "r2", "decoy"):
        d = specs[s_id]
        parent = rng.pick(["father", "mother"], anon, "rare", "r2", "decoy_parent", s_id)
        c = _pick(GO.clinvar_candidates(d["gene"], str(d["orpha"]), ont.orpha[str(d["orpha"])]["omim"]), "het",
                  anon, f"decoy/{s_id}", list(bg_files.values()), sex, avoid, male_het=trio and parent == "father")
        if c and not GO.haploid_site(c["chrom"], c["pos"], sex):
            decoys.append({**c, "gene": d["gene"], "kind": "carrier", "carrier_parent": parent})
            avoid.add((c["chrom"], c["pos"]))
    by_orpha = {str(sp["orpha"]): s for s, sp in specs.items()}
    rivals = [by_orpha[str(r)] for r in spec["rivals"] if str(r) in by_orpha]
    rivals = [s for s in rivals if specs[s].get("gene") and specs[s]["gene"] != spec["gene"]
              and not (specs[s]["inheritance"] == "XLR" and sex == "M")]
    if rivals and cfg["genome"]["decoys"]["rival"]:
        first = rng.pick(sorted(rivals), anon, "rare", "r2", "decoy_rival")
        order = [first] + [x for x in sorted(rivals) if x != first]
        for k, s_id in enumerate(order):
            d = specs[s_id]
            parent = ("mother" if d["inheritance"] == "XLR" else
                      rng.pick(["father", "mother"], anon, "rare", "r2", "rival_parent", *((s_id,) if k else ())))
            cands = GO.clinvar_candidates(d["gene"], str(d["orpha"]), ont.orpha[str(d["orpha"])]["omim"])
            c = _pick(cands, "het", anon, f"rival/{s_id}", list(bg_files.values()), sex, avoid,
                      male_het=trio and parent == "father")
            if c is None and trio and parent == "father" and \
                    _pick(cands, "het", anon, f"rival/{s_id}", list(bg_files.values()), sex, avoid) is not None:
                continue                                 # emptied by the male rule only: re-draw the next rival
            if c and not GO.haploid_site(c["chrom"], c["pos"], sex):
                decoys.append({**c, "gene": d["gene"], "kind": "rival", "rival_spec": s_id, "carrier_parent": parent})
            break
    files, genotypes = {}, {}
    for role in roles:
        role_sex = sex if role == "proband" else ("M" if role == "father" else "F")
        spikes = []
        g = gts[role]
        carries = any(a not in ("0", ".") for a in g.replace("|", "/").split("/"))
        haploid = GO.haploid_site(tv["chrom"], tv["pos"], role_sex)
        # Every member gets a row at every spiked locus (joint-call semantics, as round 1 wrote it):
        # the carrier's genotype, or an explicit hom-ref row (haploid `0` at a male's non-PAR chrX).
        if carries:
            spikes.append((tv, "hemi" if haploid else ("hom" if g in ("1/1",) else "het")))
        else:
            spikes.append(({**tv, "_haploid": haploid}, "ref"))
        genotypes[role] = (GO.gt_string(spikes[0][1], False, haploid) if carries else ("0" if haploid else "0/0"))
        for d in decoys:
            on = role == "proband" or role == d["carrier_parent"]
            d_hap = GO.haploid_site(d["chrom"], d["pos"], role_sex)
            if on and d_hap:
                raise RareSpecError(f"{anon}: decoy {d['chrom']}:{d['pos']} would make the {role} a heterozygous male")
            d.setdefault("genotypes", {})[role] = ("0" if d_hap else "0/0") if not on else "0/1"
            spikes.append((d, "het") if on else
                          ({**d, "_haploid": GO.haploid_site(d["chrom"], d["pos"], role_sex)}, "ref"))
        dst = out_dir / f"{role}.vcf.gz"
        GO.spike(bg_files[role], dst, spikes, style, sample=ped_id[role])
        files[role] = {"path": str(dst), "sha256": _sha256(dst), "sample": ped_id[role], "gt": genotypes[role]}
    ped = out_dir / f"{anon}.ped"
    write_ped(ped, anon, sex, inh, trio=len(roles) == 3)
    return {"source": "rare-os", "tier": spec["modalities"]["genome"]["tier"], "reference": "GRCh38",
            "background": {k: bg[k] for k in ("id", "dataset", "population", "superpopulation")}
            | {"samples": {r: src_sample[r] for r in roles}},
            "region": tv["chrom"], "inheritance": inh, "genotypes": genotypes, "style": style,
            "variant": {k: tv[k] for k in ("vid", "chrom", "pos", "ref", "alt", "hgvs", "clnsig", "rev", "stars",
                                           "disease_linked", "gnomad")},
            "decoys": decoys, "files": files, "ped": {"path": str(ped), "sha256": _sha256(ped)}}


def _synthetic_patient(anon: str, k: int, sex: str) -> tuple[str, list[str]]:
    code = f"P{100 + rng.below(900, anon, 'rare', 'r2', 'phi_code', k)}"
    y = 1940 + rng.below(70, anon, "rare", "r2", "phi_year", k)
    m = rng.below(12, anon, "rare", "r2", "phi_month", k)
    d = 1 + rng.below(28, anon, "rare", "r2", "phi_day", k)
    birth = f"{d:02d}-{E._MONTHS[m]}-{y}"
    name = rng.pick(list(_SYNTH_NAMES), anon, "rare", "r2", "phi_name", k)
    return E.edfplus_patient({"code": code, "sex": sex, "birthdate": birth, "name": name}), [name, birth]


def _eeg_annotations(r: dict) -> list[dict]:
    m = r["meta"]
    if r["dataset"] == "chbmit":
        return [{"onset_sec": float(a), "duration_sec": float(b) - float(a), "text": "Seizure"}
                for a, b in (m.get("seizure_intervals_s") or [])]
    dur = float(m["duration_s"])
    out = [{"onset_sec": float(s["onset_s"]), "duration_sec": float(s["duration_s"]),
            "text": EDFPLUS_STAGE.get(s["stage"], f"Sleep stage {s['stage']}")} for s in m.get("sleep_stages") or []]
    out += [{"onset_sec": float(e["onset_s"]), "duration_sec": float(e.get("duration_s") or 0), "text": str(e["event"])}
            for e in m.get("cap_events") or []]
    return [a for a in out if 0 <= a["onset_sec"] < dur]


def _build_eeg(plan: dict, out_dir: pathlib.Path) -> dict | None:
    recs = plan.get("eeg") or []
    if not recs:
        return None
    root = GO.rare_os_root() / "datasets"
    out = []
    for k, r in enumerate(recs, 1):
        m = r["meta"]
        src = root / ("edf_chbmit" if r["dataset"] == "chbmit" else "edf_capslp") / m["file"]
        dst = out_dir / f"eeg-{k}.edf"
        edfplus = r["mode"] == "edfplus" or r["phi"]
        h = E.read_header(src)
        phi_values: list[str] = []
        if r["phi"]:
            patient, phi_values = _synthetic_patient(plan["case_id"], k, plan["sex"] if r["kind"] != "control" else "X")
        elif edfplus:
            patient = "X X X X"
        else:
            patient = f"{plan['case_id'].replace('-', '')}R{k}"
        recording = E.edfplus_recording(h.startdate) if edfplus else h.recording
        ann = _eeg_annotations(r) if edfplus else None
        meta = E.write_edf(src, dst, patient=patient, recording=recording, annotations=ann, edfplus=edfplus)
        out.append({"path": str(dst), "sha256": _sha256(dst), "dataset": r["dataset"], "source": pathlib.Path(m["file"]).name,
                    "format": meta["format"], "duration_sec": meta["duration_sec"], "channel_count": meta["channel_count"],
                    "sfreq_hz": meta["sfreq_hz"], "annotations": meta["annotations"], "phi_expected": bool(r["phi"]),
                    **({"phi_values": phi_values} if phi_values else {}),
                    "kind": r["kind"], "source_sha256": m["sha256"], "cut_offset_s": m.get("cut_offset_s")})
    return {"recordings": out}


def _build_external(plan: dict, out_dir: pathlib.Path) -> dict:
    root = GO.rare_os_root() / "datasets" / "vcf_exomiser_examples"
    _, facts = phenopacket_gold(plan["case_id"])
    vcf = out_dir / "proband.vcf"
    ped = out_dir / f"{plan['case_id']}.ped"
    shutil.copyfile(root / "Pfeiffer.vcf", vcf)
    shutil.copyfile(root / "pfeiffer-singleton.ped", ped)
    return {"source": "exomiser", "build_refused": True, "reference": facts["assembly"],
            "tier": "external_benchmark", "variant": None, "decoys": [],
            "files": {"proband": {"path": str(vcf), "sha256": _sha256(vcf), "sample": "manuel"}},
            "ped": {"path": str(ped), "sha256": _sha256(ped)},
            "external": {"name": "Exomiser Pfeiffer example", "declared_assembly": facts["assembly"],
                         "vcf_source": str(root / "Pfeiffer.vcf"), "ped_source": str(root / "pfeiffer-singleton.ped"),
                         "phenopacket": facts["phenopacket"],
                         "note": "shipped as staged: the one record's INFO names the gene (GENE=FGFR2); "
                                 "the gold is the refusal, so no variant judge applies"}}


def build_case(plan: dict, root: str, cfg: dict) -> dict:
    """Write one case's attachments (genome, EEG; DICOM stays a pointer) and return them with the
    gold. Reused when `<case>/.plan.sha256` matches."""
    out_dir = pathlib.Path(root) / plan["case_id"]
    marker, saved = out_dir / ".plan.sha256", out_dir / ".built.json"
    key = _plan_key(plan)
    if marker.exists() and saved.exists() and marker.read_text().strip() == key:
        return json.loads(saved.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_specs()[plan["spec_id"]]
    gold = plan_gold(plan)
    att: dict = {"root": str(out_dir), "genome": None, "imaging": None, "narrative": None}
    if plan["external"] == "exomiser_pfeiffer":
        att["genome"] = _build_external(plan, out_dir)
    elif plan["background"]:
        att["genome"] = _build_genome(plan, spec, out_dir, pathlib.Path(root) / "_background_cache", cfg)
    if plan.get("imaging"):
        img = copy.deepcopy(plan["imaging"])
        for s in img["series"]:
            s["sha256"] = _sha256(pathlib.Path(s["path"]))
        att["imaging"] = img
    eeg = _build_eeg(plan, out_dir)
    if eeg:
        att["eeg"] = eeg
    built = {"gold": gold, "attachments": att}
    saved.write_text(json.dumps(built, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    marker.write_text(key)
    return built


# ---------------------------------------------------------------- jobs
def case_for_lang(plan: dict, built: dict, lang: str) -> dict:
    from haenv.demographics import sample_profile
    from haenv.ddx import _t_index_for
    from .gen_job import case_entry
    from .narrative import render_narrative, sha256_text
    spec = load_specs()[plan["spec_id"]]
    anon, sex = plan["case_id"], plan["sex"]
    T = int(_t_index_for(anon))
    gold = gold_in_lang(copy.deepcopy(built["gold"]), lang)
    symptoms = render_symptoms(gold, anon, T, lang=lang)
    prof = {k: v for k, v in sample_profile(anon, sex).items() if k != "sex"}
    if plan.get("age") is not None:
        prof["age_range"] = age_band(plan["age"])
    # The weight course stays the spec's: haenv's weight stream is an adult metabolic construct and
    # its physiology gate refuses < 40 kg (tried 2026-09-28: every paediatric case failed GV-1), so
    # a paediatric case carries an adult-range weight -- a known limitation of the skeleton.
    weight = None
    att = copy.deepcopy(built["attachments"])
    md, spans = render_narrative(anon, spec, gold, {**prof, "sex": sex}, att.get("imaging"), att.get("genome"),
                                 lang=lang, eeg=att.get("eeg"), style=2)
    mdp = pathlib.Path(att["root"]) / f"{anon}.{lang}.md"
    mdp.write_text(md, encoding="utf-8")
    att["narrative"] = {"path": str(mdp), "sha256": sha256_text(md), "spans": spans,
                        "n_gold_spans": sum(1 for x in spans if x["hpo_id"]), "n_noise": sum(1 for x in spans if not x["hpo_id"])}
    extra = {"rare_external": plan["external"]} if plan["external"] else None
    return case_entry(plan["spec_id"], spec, anon, 1, sex, gold, symptoms, prof, att, lang, T, weight=weight,
                      extra_latent=extra)


def write_job(job_id: str, cases: list[dict], note: str) -> pathlib.Path:
    from haenv.canary import block as canary_block
    from .gen_job import BATCH_GATE_NOTE
    doc = {"job_id": job_id, "task_type": "joint_dx", "multiround": False, "include_baseline": True,
           "models": [], "sample_cases": 3, "report": f"eval-{job_id}.md", "plugins": ["haenv.judges.rare"], "cases": cases}
    head = ("# Rare coding pack, round 2 -- written by haenv-rare-round2 (haenv_rare/haenv_rare/round2.py) from\n"
            "# haenv_rare/data/round2.yaml and the staged rare-os data; do not edit. Gold (HPO set / ORPHA / gene /\n"
            "# variant / inheritance / EEG annotations) is decided per case id before rendering.\n"
            f"# {note}\n# SYNTHETIC, evaluation only, not medical advice.\n")
    path = ROOT / "inputs" / f"{job_id}.job.yaml"
    path.write_text(canary_block("# ") + head + BATCH_GATE_NOTE + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200),
                    encoding="utf-8")
    return path


def _roles(plan: dict) -> list[str]:
    spec = load_specs()[plan["spec_id"]]
    gts = _genotypes(spec["inheritance"], plan["sex"])
    trio = spec["modalities"]["genome"].get("skeleton") == "trio"
    return [{"proband": "child"}.get(r, r) for r in (gts if trio else ["proband"])]


def _cache_name(bg_id: str, sample: str, joint: bool) -> str:
    return f"{bg_id}.{sample}.{'joint' if joint else 'carried'}.vcf.gz"


def _extract(job: tuple[str, str, bool, str]) -> str:
    vcf, sample, joint, out = job
    GO.member_background(vcf, sample, "SAMPLE", out, joint=joint)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="haenv-rare-round2")
    ap.add_argument("--attach-root", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true", help="print the allocation table and stop")
    a = ap.parse_args(argv)
    if a.attach_root:
        os.environ["HAENV_RARE_R2_ROOT"] = str(pathlib.Path(a.attach_root).resolve())
    cfg = load_round()
    plans = allocate(cfg)
    if a.plan_only:
        for p in plans:
            print(json.dumps(plan_row(p), ensure_ascii=False))
        return 0
    root = attach_root()
    root.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(8, a.workers))
    # every (background, member) once, before the cases (two cases may share a background)
    need = sorted({(p["background"]["id"], p["background"]["vcf"], p["background"]["samples"][m], len(_roles(p)) == 3)
                   for p in plans if p.get("background")
                   for m in _roles(p)})
    cache = root / "_background_cache"
    todo = [(v, smp, joint, str(cache / _cache_name(bid, smp, joint))) for bid, v, smp, joint in need
            if not (cache / _cache_name(bid, smp, joint)).exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_extract, todo))
        built = list(ex.map(build_case, plans, [str(root)] * len(plans), [cfg] * len(plans)))
    by_lang = {"zh": [], "en": []}
    for p, b in zip(plans, built):
        for lang in (("zh", "en") if p["lang"] == "both" else (p["lang"],)):
            by_lang[lang].append(case_for_lang(p, b, lang))
    for lang, cases in by_lang.items():
        note = ("Chinese journal-style case reports (40 cases: the 20 shared with p7-en + 20 zh-only)." if lang == "zh"
                else "English deterministic templates (40 cases: the 20 shared with p6-zh + 20 en-only).")
        path = write_job(PACKS[lang], cases, note)
        print(f"wrote {path} ({len(cases)} cases)")
    from .verify_attachments import main as verify
    return verify([str(ROOT / "inputs" / f"{PACKS[x]}.job.yaml") for x in ("zh", "en")] + ["--manifest", str(root / "SHA256SUMS")])


def plan_row(p: dict) -> dict:
    return {"case_id": p["case_id"], "lang": p["lang"], "spec": p["spec_id"], "sex": p["sex"], "age": p.get("age"),
            "background": (p.get("background") or {}).get("id"),
            "eeg": [f"{r['dataset']}:{pathlib.Path(r['meta']['file']).name}:{r['mode']}{':phi' if r['phi'] else ''}"
                    for r in p.get("eeg") or []],
            "imaging": (lambda i: i and f"{i['collection']}:{i['tcia_patient_id']}:{i['tier']}")(p.get("imaging")),
            "external": p.get("external")}


if __name__ == "__main__":
    raise SystemExit(main())
