"""Genome attachments from the staged rare-os data (round 2): real trio backgrounds, a ClinVar P/LP
spike that gnomAD v4.1 calls rare, per-member single-sample VCFs.

Sources (read-only, `rare_os_root()`, default /data/xfs_recovery/data/rare-os; see its README):
* backgrounds: `datasets/vcf_1000g_30x_trios/` (24 trios, `trios.tsv`) and `datasets/vcf_giab/`
  (2 trios); 3-sample VCFs over HPO-gene regions, child first;
* spike truth: `ref/clinvar_GRCh38/clinvar.vcf.gz` -- aggregate CLNSIG Pathogenic /
  Likely_pathogenic / Pathogenic/Likely_pathogenic, review status >= 2 stars, GENEINFO names the
  gene, small variant (REF and ALT <= 50 bp, one ALT); alleles whose CLNDISDB names the disease
  (Orphanet code or one of its OMIM ids) come first;
* rarity: `ref/gnomad_v4.1_exomes_subset/` -- popmax over the continental groups `CONTINENTAL`
  (bottlenecked asj / fin and `remaining` left out) must stay below `HET_MAX` for a heterozygous
  call and `HOM_MAX` for a homozygous / hemizygous one; an allele absent from the subset is absent
  from gnomAD exomes (AF 0).

What a solver could use to tell the spike from the background is removed or matched: the sample
column carries the PED id (a public 1kGP / GIAB sample id would let anyone diff against the
public panel), INFO is dropped on every record, the maintainer command lines are dropped from the
header, and each spiked record copies the background's majority ID pattern, QUAL, FILTER and
phasing. Output is BGZF + tabix index (pysam), byte-deterministic.
SYNTHETIC evaluation data only.
"""
from __future__ import annotations

import csv
import gzip
import os
import pathlib
from collections import Counter
from functools import lru_cache

CONTINENTAL = ("afr", "amr", "eas", "mid", "nfe", "sas")
HET_MAX, HOM_MAX = 0.01, 0.05
_PLP = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
_STARS = {"practice_guideline": 4, "reviewed_by_expert_panel": 3,
          "criteria_provided,_multiple_submitters,_no_conflicts": 2,
          "criteria_provided,_single_submitter": 1, "criteria_provided,_conflicting_classifications": 1}
MAX_ALLELE = 50
# GRCh38 pseudo-autosomal regions: diploid in males
_PAR_X = ((10_001, 2_781_479), (155_701_383, 156_030_895))
_PAR_Y = ((10_001, 2_781_479), (56_887_903, 57_217_415))


def rare_os_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HAENV_RARE_OS_ROOT") or "/data/xfs_recovery/data/rare-os")


def _bare(chrom: str) -> str:
    return str(chrom).replace("chr", "")


# ---------------------------------------------------------------- ClinVar
@lru_cache(maxsize=None)
def _gene_regions(gene: str, bed: str) -> tuple[tuple[str, int, int], ...]:
    out = []
    with open(bed, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4 and gene in f[3].split(","):
                out.append((f[0], int(f[1]), int(f[2])))
    return tuple(out)


def clinvar_candidates(gene: str, orpha: str, omim: list[str], *, clinvar: str | pathlib.Path | None = None,
                       bed: str | pathlib.Path | None = None) -> list[dict]:
    """P/LP, >= 2 stars, small, in `gene`; disease-linked first, then by ClinVar id."""
    import pysam
    root = rare_os_root()
    clinvar = str(clinvar or root / "ref" / "clinvar_GRCh38" / "clinvar.vcf.gz")
    bed = str(bed or root / "regions" / "hpo_gene_regions.GRCh38.bed")
    want = {f"Orphanet:{orpha}"} | {f"OMIM:{o}" for o in omim}
    seen, out = set(), []
    with pysam.VariantFile(clinvar) as cv:
        for chrom, start, end in _gene_regions(gene, bed):
            for r in cv.fetch(_bare(chrom), max(0, start), end):
                if r.id in seen or not r.alts or len(r.alts) != 1:
                    continue
                inf = r.info
                sig = "|".join(inf.get("CLNSIG") or ())
                if sig not in _PLP:
                    continue
                stars = _STARS.get(",".join(inf.get("CLNREVSTAT") or ()), 0)
                genes = [g.split(":")[0] for g in str(inf.get("GENEINFO") or "").split("|")]
                ref, alt = r.ref, r.alts[0]
                if stars < 2 or gene not in genes or len(ref) > MAX_ALLELE or len(alt) > MAX_ALLELE \
                        or not set(ref + alt) <= set("ACGT"):
                    continue
                disdb = ",".join(inf.get("CLNDISDB") or ())
                seen.add(r.id)
                out.append({"vid": str(r.id), "chrom": _bare(r.chrom), "pos": int(r.pos), "ref": ref, "alt": alt,
                            "clnsig": sig, "rev": ",".join(inf.get("CLNREVSTAT") or ()), "stars": stars,
                            "hgvs": ",".join(inf.get("CLNHGVS") or ()), "vc": str(inf.get("CLNVC") or ""),
                            "dn": ",".join(inf.get("CLNDN") or ()),
                            "disease_linked": any(w in disdb.replace("|", ",").split(",") for w in want)})
    return sorted(out, key=lambda c: (not c["disease_linked"], int(c["vid"]) if c["vid"].isdigit() else 0))


# ---------------------------------------------------------------- gnomAD
def gnomad_af(chrom: str, pos: int, ref: str, alt: str, *, root: str | pathlib.Path | None = None) -> dict | None:
    """gnomAD v4.1 exome frequencies of one allele, or None when the subset has no such record."""
    import pysam
    d = pathlib.Path(root or rare_os_root() / "ref" / "gnomad_v4.1_exomes_subset")
    p = d / f"gnomad.exomes.v4.1.sites.chr{_bare(chrom)}.clinvar_plp_positions.vcf.bgz"
    if not p.exists():
        return None
    with pysam.VariantFile(str(p)) as g:
        for r in g.fetch(f"chr{_bare(chrom)}", int(pos) - 1, int(pos)):
            if r.pos != int(pos) or r.ref != ref or not r.alts or alt not in r.alts:
                continue
            i = r.alts.index(alt)

            def val(k):
                v = r.info.get(k)
                if isinstance(v, tuple):
                    v = v[i] if i < len(v) else None
                return None if v is None else float(f"{float(v):.6g}")      # stored as float32
            groups = {grp: val(f"AF_{grp}") for grp in ("afr", "amr", "asj", "eas", "fin", "mid", "nfe", "remaining", "sas")}
            cont = {k: v for k, v in groups.items() if k in CONTINENTAL and v is not None}
            top = max(cont.items(), key=lambda kv: kv[1]) if cont else (None, 0.0)
            if not top[1]:
                top = (None, 0.0)                      # no carrier in any continental group
            gm = r.info.get("grpmax")
            return {"af": val("AF") or 0.0, "af_popmax": top[1], "popmax_group": top[0], "groups": groups,
                    "grpmax": (gm[i] if isinstance(gm, tuple) else gm), "af_grpmax": val("AF_grpmax"),
                    "faf95_max": val("fafmax_faf95_max"), "source": "gnomAD v4.1 exomes (ClinVar P/LP-position subset)"}
    return None


def rare_enough(g: dict | None, zygosity: str) -> bool:
    ceiling = HET_MAX if zygosity == "het" else HOM_MAX
    return g is None or float(g.get("af_popmax") or 0.0) < ceiling


# ---------------------------------------------------------------- backgrounds
def backgrounds() -> list[dict]:
    """The 26 staged trios: {id, dataset, vcf, samples{child,father,mother}, child_sex, population, superpopulation}."""
    root = rare_os_root() / "datasets"
    out = []
    with open(root / "vcf_1000g_30x_trios" / "trios.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out.append({"id": r["trio_id"], "dataset": "1kgp_30x", "vcf": str(root / "vcf_1000g_30x_trios" / r["vcf"]),
                        "samples": {"child": r["child"], "father": r["father"], "mother": r["mother"]},
                        "child_sex": "M" if r["child_sex"] == "male" else "F",
                        "population": r["population"], "superpopulation": r["superpopulation"]})
    for fam, pop, sup, (c, f, m) in (("AshkenazimTrio", "ASJ", "EUR", ("HG002", "HG003", "HG004")),
                                     ("ChineseTrio", "CHS", "EAS", ("HG005", "HG006", "HG007"))):
        vcf = root / "vcf_giab" / f"{fam}_{c}_{f}_{m}.GRCh38_v4.2.1.hpo_gene_regions.vcf.gz"
        out.append({"id": f"GIAB_{fam}", "dataset": "giab_v4.2.1", "vcf": str(vcf),
                    "samples": {"child": c, "father": f, "mother": m}, "child_sex": "M",
                    "population": pop, "superpopulation": sup})
    return out


def _keep_header(line: str) -> bool:
    return not (line.startswith("##INFO") or line.startswith("##bcftools") or line.startswith("##CL=")
                or line.startswith("##RUN-ID") or "/home/" in line or "/vepfs/" in line)


def _non_ref(gt: str) -> bool:
    return any(a not in ("0", ".") for a in gt.replace("|", "/").split("/"))


def member_background(trio_vcf: str | pathlib.Path, sample: str, new_name: str, out: str | pathlib.Path,
                      joint: bool = False) -> pathlib.Path:
    """One member of a trio VCF as a single-sample VCF, sample renamed to `new_name`, INFO and
    maintainer lines dropped, `##reference=GRCh38` present. BGZF + tabix.

    `joint=True` (a trio case): the member's column of the joint call -- a row at every site where
    ANY member is non-ref, carrying this member's own genotype (0|0 and ./. kept); only sites ref or
    missing in all members are dropped. A real trio is joint-called, and a parent's missing row reads
    as "unknown", not "hom-ref". `joint=False` (a singleton case): the sites this member carries."""
    import pysam
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".txt")
    with pysam.VariantFile(str(trio_vcf)) as v, open(tmp, "w", encoding="utf-8") as fo:
        samples = list(v.header.samples)
        k = samples.index(sample)
        head = [l for l in str(v.header).splitlines() if l.startswith("##") and _keep_header(l)]
        if not any(l.startswith("##reference") for l in head):
            head.insert(1, "##reference=GRCh38")
        fo.write("\n".join(head) + "\n")
        fo.write("\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", new_name]) + "\n")
        for line in v.fetch():
            f = str(line).rstrip("\n").split("\t")
            gts = [x.split(":")[0] for x in f[9:9 + len(samples)]]
            if not (any(_non_ref(g) for g in gts) if joint else _non_ref(gts[k])):
                continue
            fo.write("\t".join(f[:7] + [".", "GT", gts[k]]) + "\n")
    pysam.tabix_compress(str(tmp), str(out), force=True)
    pysam.tabix_index(str(out), preset="vcf", force=True)
    tmp.unlink()
    return out


def background_style(path: str | pathlib.Path, n: int = 2000) -> dict:
    """Majority ID pattern (`colon` = chrom:pos:ref:alt, `dot`), QUAL, FILTER and phasing of the
    first `n` records."""
    ids, quals, filts, phased = Counter(), Counter(), Counter(), Counter()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        k = 0
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            ids["colon" if f[2].count(":") == 3 else "dot"] += 1
            quals[f[5]] += 1
            filts[f[6]] += 1
            if len(f[9]) > 1 and "." not in f[9]:           # a missing call says nothing about phasing
                phased["|" in f[9]] += 1
            k += 1
            if k >= n:
                break
    return {"id": ids.most_common(1)[0][0], "qual": quals.most_common(1)[0][0],
            "filter": filts.most_common(1)[0][0], "phased": bool(phased.most_common(1)[0][0]) if phased else False}


def gt_string(zygosity: str, phased: bool, haploid: bool) -> str:
    """`het` / `hom` / `hemi` / `ref` (the explicit hom-ref row of a member who does not carry)."""
    if zygosity == "ref":
        return "0" if haploid else ("0|0" if phased else "0/0")
    if zygosity == "hemi":
        if not haploid:
            raise ValueError("a hemizygous call needs a haploid site (male chrX outside the PAR)")
        return "1"
    sep = "|" if phased else "/"
    if zygosity == "het":
        return f"0{sep}1"
    if zygosity == "hom":
        return f"1{sep}1"
    raise ValueError(f"unknown zygosity {zygosity!r}")


def haploid_site(chrom: str, pos: int, sex: str) -> bool:
    """A male at a non-PAR chrX / chrY position carries one allele."""
    c = _bare(chrom)
    par = {"X": _PAR_X, "Y": _PAR_Y}.get(c)
    return sex == "M" and par is not None and not any(a <= int(pos) <= b for a, b in par)


def male_het_rows(path: str | pathlib.Path) -> list[str]:
    """`chrom:pos gt` of every record at a non-PAR chrX / chrY position whose genotype has two
    different called alleles -- impossible in a male."""
    import pysam
    out = []
    with pysam.TabixFile(str(path)) as t:
        for c in (x for x in t.contigs if _bare(x) in ("X", "Y")):
            for line in t.fetch(c):
                f = line.split("\t", 10)
                if not haploid_site(c, int(f[1]), "M"):
                    continue
                al = [a for a in f[9].split(":")[0].replace("|", "/").split("/") if a != "."]
                if len(set(al)) > 1:
                    out.append(f"{f[0]}:{f[1]} {f[9].split(':')[0]}")
    return out


def spike(background: str | pathlib.Path, out: str | pathlib.Path, spikes: list[tuple[dict, str]], style: dict,
          sample: str | None = None) -> pathlib.Path:
    """`background` + spiked records, in contig order then position. `spikes` = [(variant, zygosity)];
    zygosity `het` / `hom` / `hemi` / `ref`; a variant with `_haploid: True` gets a haploid ref row.
    The records copy `style`; `sample` renames the sample column."""
    import pysam
    out = pathlib.Path(out)
    with gzip.open(background, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    head = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    if sample:
        head[-1] = "\t".join(head[-1].split("\t")[:9] + [sample])
    contigs = [l.split("ID=", 1)[1].split(",", 1)[0].rstrip(">") for l in head if l.startswith("##contig=")]
    order = {c: i for i, c in enumerate(contigs)}
    style_chr = "chr" if (body and body[0].startswith("chr")) or any(c.startswith("chr") for c in contigs) else ""
    new = []
    for v, zyg in spikes:
        chrom = style_chr + _bare(v["chrom"])
        vid = f"{_bare(v['chrom'])}:{v['pos']}:{v['ref']}:{v['alt']}" if style["id"] == "colon" else "."
        gt = gt_string(zyg, style["phased"], zyg == "hemi" or v.get("_haploid", False))
        new.append("\t".join([chrom, str(v["pos"]), vid, v["ref"], v["alt"], style["qual"], style["filter"], ".", "GT", gt]))
    key = lambda l: (order.get(l.split("\t", 1)[0], len(order)), int(l.split("\t", 2)[1]))
    merged = sorted(body + new, key=key)                    # stable: equal keys keep background first
    tmp = out.with_name(out.name + ".txt")
    tmp.write_text("\n".join(head + merged) + "\n", encoding="utf-8")
    pysam.tabix_compress(str(tmp), str(out), force=True)
    pysam.tabix_index(str(out), preset="vcf", force=True)
    tmp.unlink()
    return out


def readback(path: str | pathlib.Path, chrom: str, pos: int, ref: str, alt: str) -> str | None:
    """GT of that exact allele in a single-sample VCF, or None when the file has no such record."""
    import pysam
    with pysam.TabixFile(str(path)) as t:
        names = set(t.contigs)
        c = ("chr" + _bare(chrom)) if ("chr" + _bare(chrom)) in names else _bare(chrom)
        if c not in names:
            return None
        for line in t.fetch(c, int(pos) - 1, int(pos)):
            f = line.split("\t")
            if int(f[1]) == int(pos) and f[3] == ref and alt in f[4].split(","):
                return f[9].split(":")[0]
    return None


def record(path: str | pathlib.Path, chrom: str, pos: int, ref: str, alt: str) -> list[str] | None:
    """The tab-split record of that exact allele, or None."""
    import pysam
    with pysam.TabixFile(str(path)) as t:
        names = set(t.contigs)
        c = ("chr" + _bare(chrom)) if ("chr" + _bare(chrom)) in names else _bare(chrom)
        if c not in names:
            return None
        for line in t.fetch(c, int(pos) - 1, int(pos)):
            f = line.split("\t")
            if int(f[1]) == int(pos) and f[3] == ref and alt in f[4].split(","):
                return f
    return None


def free_of_background(cands: list[dict], member_files: list[str | pathlib.Path]) -> list[dict]:
    """Candidates whose position no background member already carries (any allele)."""
    import pysam
    tabs = [pysam.TabixFile(str(p)) for p in member_files]
    try:
        out = []
        for c in cands:
            hit = False
            for t in tabs:
                names = set(t.contigs)
                ch = ("chr" + c["chrom"]) if ("chr" + c["chrom"]) in names else c["chrom"]
                if ch in names and any(True for _ in t.fetch(ch, c["pos"] - 1, c["pos"] + max(len(c["ref"]), 1))):
                    hit = True
                    break
            if not hit:
                out.append(c)
        return out
    finally:
        for t in tabs:
            t.close()
