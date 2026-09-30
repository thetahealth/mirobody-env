"""rare-os genome path (`haenv_rare.genome_os`): background = one member of a real trio VCF,
spiked with a ClinVar P/LP allele that gnomAD v4.1 calls rare; everything the solver could use
to tell the spike apart from the background is controlled (sample ids, INFO, QUAL/FILTER/ID
style, phasing). Tiny synthetic inputs, built here with pysam."""
import gzip

import pytest

pysam = pytest.importorskip("pysam")

from haenv_rare import genome_os as G   # noqa: E402

CONTIGS = "##contig=<ID=chr1,length=1000000>\n##contig=<ID=chrX,length=156040895>\n"


def _bgz(tmp, name, text, preset="vcf"):
    raw = tmp / (name + ".txt")
    raw.write_text(text)
    out = tmp / name
    pysam.tabix_compress(str(raw), str(out), force=True)
    pysam.tabix_index(str(out), preset=preset, force=True)
    return out


@pytest.fixture
def world(tmp_path):
    trio = _bgz(tmp_path, "trio.vcf.gz",
                "##fileformat=VCFv4.2\n##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
                "##INFO=<ID=AC,Number=A,Type=Integer,Description=\"x\">\n"
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n" + CONTIGS
                + "##bcftools_viewCommand=view -s C,F,M /home/someone/work/x.vcf.gz\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tC\tF\tM\n"
                "chr1\t100\t1:100:A:G\tA\tG\t.\t.\tAC=1\tGT\t0|1\t0|0\t0|0\n"
                "chr1\t300\t1:300:C:T\tC\tT\t.\t.\tAC=2\tGT\t1|0\t1|0\t0|0\n"
                "chr1\t500\t1:500:G:A\tG\tA\t.\t.\tAC=1\tGT\t0|0\t0|0\t0|1\n"
                "chr1\t600\t1:600:A:T\tA\tT\t.\t.\tAC=0\tGT\t0|0\t0|0\t0|0\n"
                "chr1\t700\t1:700:C:G\tC\tG\t.\t.\tAC=0\tGT\t./.\t./.\t./.\n"
                "chr1\t800\t1:800:T:G\tT\tG\t.\t.\tAC=1\tGT\t./.\t0|1\t./.\n"
                "chrX\t400000\tX:400000:T:C\tT\tC\t.\t.\tAC=1\tGT\t1\t0\t0|1\n")
    clinvar = _bgz(tmp_path, "clinvar.vcf.gz",
                   "##fileformat=VCFv4.1\n##contig=<ID=1>\n##contig=<ID=X>\n"
                   "##INFO=<ID=CLNSIG,Number=.,Type=String,Description=\"x\">\n"
                   "##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description=\"x\">\n"
                   "##INFO=<ID=GENEINFO,Number=1,Type=String,Description=\"x\">\n"
                   "##INFO=<ID=CLNDISDB,Number=.,Type=String,Description=\"x\">\n"
                   "##INFO=<ID=CLNHGVS,Number=.,Type=String,Description=\"x\">\n"
                   "##INFO=<ID=CLNVC,Number=1,Type=String,Description=\"x\">\n"
                   "##INFO=<ID=CLNDN,Number=.,Type=String,Description=\"x\">\n"
                   "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                   "1\t200\t11\tC\tA\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=GENE1:1;CLNDISDB=Orphanet:33069;CLNHGVS=g1;CLNVC=single_nucleotide_variant;CLNDN=d\n"
                   "1\t300\t12\tC\tT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=GENE1:1;CLNDISDB=Orphanet:33069;CLNHGVS=g2;CLNVC=single_nucleotide_variant;CLNDN=d\n"
                   "1\t350\t13\tG\tT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_single_submitter;GENEINFO=GENE1:1;CLNDISDB=Orphanet:33069;CLNHGVS=g3;CLNVC=single_nucleotide_variant;CLNDN=d\n"
                   "1\t360\t14\tG\tC\t.\t.\tCLNSIG=Uncertain_significance;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=GENE1:1;CLNDISDB=Orphanet:33069;CLNHGVS=g4;CLNVC=single_nucleotide_variant;CLNDN=d\n"
                   "1\t370\t15\tA\tG\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;GENEINFO=GENE1:1;CLNDISDB=Orphanet:33069;CLNHGVS=g5;CLNVC=single_nucleotide_variant;CLNDN=d\n"
                   "1\t380\t16\tT\tA\t.\t.\tCLNSIG=Likely_pathogenic;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=OTHER:2;CLNDISDB=Orphanet:1;CLNHGVS=g6;CLNVC=single_nucleotide_variant;CLNDN=d\n"
                   "X\t400100\t17\tA\tT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=criteria_provided,_multiple_submitters,_no_conflicts;GENEINFO=GENEX:3;CLNDISDB=Orphanet:9;CLNHGVS=g7;CLNVC=single_nucleotide_variant;CLNDN=d\n")
    gd = tmp_path / "gnomad"
    gd.mkdir()
    _bgz(gd, "gnomad.exomes.v4.1.sites.chr1.clinvar_plp_positions.vcf.bgz",
         "##fileformat=VCFv4.2\n" + CONTIGS
         + "".join(f"##INFO=<ID={k},Number=A,Type=Float,Description=\"x\">\n" for k in
                   ("AF", "AF_afr", "AF_amr", "AF_asj", "AF_eas", "AF_fin", "AF_mid", "AF_nfe", "AF_remaining", "AF_sas", "AF_grpmax", "fafmax_faf95_max"))
         + "##INFO=<ID=grpmax,Number=A,Type=String,Description=\"x\">\n"
         "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
         "chr1\t200\t.\tC\tA\t.\tPASS\tAF=0.0002;AF_afr=0;AF_amr=0;AF_asj=0.2;AF_eas=0.0005;AF_fin=0;AF_mid=0;AF_nfe=0.0001;AF_remaining=0;AF_sas=0;grpmax=eas;AF_grpmax=0.0005;fafmax_faf95_max=0.0003\n"
         "chr1\t370\t.\tA\tG\t.\tPASS\tAF=0.02;AF_afr=0;AF_amr=0;AF_asj=0;AF_eas=0.03;AF_fin=0;AF_mid=0;AF_nfe=0.01;AF_remaining=0;AF_sas=0;grpmax=eas;AF_grpmax=0.03;fafmax_faf95_max=0.02\n")
    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t50\t450\tGENE1,GENE9\nchr1\t450\t600\tOTHER\nchrX\t399000\t401000\tGENEX\n")
    return {"trio": trio, "clinvar": clinvar, "gnomad": gd, "bed": bed, "tmp": tmp_path}


def test_candidates_are_plp_two_star_in_gene_and_disease_linked_first(world):
    c = G.clinvar_candidates("GENE1", "33069", [], clinvar=world["clinvar"], bed=world["bed"])
    assert [x["vid"] for x in c] == ["11", "12", "15"]          # 13 is one star, 14 is VUS, 16 other gene
    assert all(x["disease_linked"] for x in c) and c[0]["stars"] == 2 and c[2]["stars"] == 3


def test_gnomad_popmax_is_over_continental_groups(world):
    g = G.gnomad_af("1", 200, "C", "A", root=world["gnomad"])
    assert g["af_popmax"] == 0.0005 and g["popmax_group"] == "eas"   # asj 0.2 is a bottleneck group
    assert G.gnomad_af("1", 999, "C", "A", root=world["gnomad"]) is None
    assert G.rare_enough(g, "het") and G.rare_enough(None, "het")
    assert not G.rare_enough(G.gnomad_af("1", 370, "A", "G", root=world["gnomad"]), "het")
    assert G.rare_enough(G.gnomad_af("1", 370, "A", "G", root=world["gnomad"]), "hom")


def test_member_file_renames_the_sample_strips_info_and_keeps_only_carried_sites(world):
    out = world["tmp"] / "m.vcf.gz"
    G.member_background(world["trio"], "C", "JD-1-P", out)
    lines = gzip.open(out, "rt").read().splitlines()
    head = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    assert "##reference=GRCh38" in head and not any("bcftools" in l or "/home/" in l for l in head)
    assert not any(l.startswith("##INFO") for l in head)
    assert head[-1].split("\t")[9:] == ["JD-1-P"]
    assert [l.split("\t")[1] for l in body] == ["100", "300", "400000"]   # 500 / 800 not carried by C
    assert all(l.split("\t")[7] == "." for l in body)
    assert body[-1].split("\t")[9] == "1"                                  # haploid chrX kept as is


def test_spike_takes_the_background_style_and_lands_in_order(world):
    bg = world["tmp"] / "bg.vcf.gz"
    G.member_background(world["trio"], "C", "JD-1-P", bg)
    out = world["tmp"] / "proband.vcf.gz"
    style = G.background_style(bg)
    assert style == {"id": "colon", "qual": ".", "filter": ".", "phased": True}
    G.spike(bg, out, [({"chrom": "1", "pos": 200, "ref": "C", "alt": "A"}, "het"),
                      ({"chrom": "X", "pos": 400100, "ref": "A", "alt": "T"}, "hemi")], style)
    body = [l.rstrip("\n").split("\t") for l in gzip.open(out, "rt") if not l.startswith("#")]
    assert [r[1] for r in body] == ["100", "200", "300", "400000", "400100"]
    s = body[1]
    assert s[0] == "chr1" and s[2] == "1:200:C:A" and s[5:8] == [".", ".", "."] and s[9] == "0|1"
    assert body[4][9] == "1"
    assert pysam.TabixFile(str(out)).contigs == ["chr1", "chrX"]           # indexed
    assert G.readback(out, "1", 200, "C", "A") == "0|1"
    assert G.readback(out, "1", 250, "C", "A") is None


def test_gt_strings():
    assert G.gt_string("het", True, False) == "0|1"
    assert G.gt_string("hom", False, False) == "1/1"
    assert G.gt_string("hemi", True, True) == "1"
    with pytest.raises(ValueError):
        G.gt_string("hemi", True, False)


def test_a_candidate_on_a_background_site_is_skipped(world):
    bg = world["tmp"] / "bg2.vcf.gz"
    G.member_background(world["trio"], "C", "JD-2-P", bg)
    c = G.clinvar_candidates("GENE1", "33069", [], clinvar=world["clinvar"], bed=world["bed"])
    free = G.free_of_background(c, [bg])
    assert [x["vid"] for x in free] == ["11", "15"]                       # 12 sits on chr1:300


def test_real_data_smoke():
    """Against the staged data, if present on this machine."""
    if not G.rare_os_root().exists():
        pytest.skip(f"{G.rare_os_root()} absent: skipped, not passed")
    c = G.clinvar_candidates("SCN1A", "33069", ["607208"])
    assert c and all(x["stars"] >= 2 and x["disease_linked"] for x in c[:5])
    bgs = G.backgrounds()
    assert len(bgs) == 26 and sum(b["superpopulation"] == "EAS" for b in bgs) == 13


def test_spike_renames_the_sample_column(world):
    bg = world["tmp"] / "bg3.vcf.gz"
    G.member_background(world["trio"], "C", "SAMPLE", bg)
    out = world["tmp"] / "p3.vcf.gz"
    G.spike(bg, out, [], G.background_style(bg), sample="JD-9-P")
    head = [l for l in gzip.open(out, "rt") if l.startswith("#CHROM")]
    assert head[0].rstrip("\n").split("\t")[9:] == ["JD-9-P"]


def test_no_carrier_anywhere_has_no_popmax_group(world):
    import gzip as _gz
    g = G.gnomad_af("1", 200, "C", "A", root=world["gnomad"])
    assert g["popmax_group"] == "eas"
    # an all-zero record
    p = world["gnomad"] / "gnomad.exomes.v4.1.sites.chr1.clinvar_plp_positions.vcf.bgz"
    text = _gz.open(p, "rt").read().replace("AF=0.0002;AF_afr=0;AF_amr=0;AF_asj=0.2;AF_eas=0.0005",
                                            "AF=0;AF_afr=0;AF_amr=0;AF_asj=0;AF_eas=0").replace(
        "AF_nfe=0.0001;AF_remaining=0;AF_sas=0;grpmax=eas", "AF_nfe=0;AF_remaining=0;AF_sas=0;grpmax=eas")
    world["gnomad"].joinpath("z").mkdir()
    _bgz(world["gnomad"] / "z", p.name, text)
    z = G.gnomad_af("1", 200, "C", "A", root=world["gnomad"] / "z")
    assert z["af_popmax"] == 0.0 and z["popmax_group"] is None



def test_joint_member_file_keeps_every_site_any_member_carries(world):
    """A trio is joint-called: each member's file has a row wherever ANY member is non-ref, with
    that member's own genotype (0|0 and ./. kept); only all-ref / all-missing sites are dropped."""
    out = world["tmp"] / "f.vcf.gz"
    G.member_background(world["trio"], "F", "JD-1-F", out, joint=True)
    body = [l.rstrip("\n").split("\t") for l in gzip.open(out, "rt") if not l.startswith("#")]
    assert [(r[1], r[9]) for r in body] == [("100", "0|0"), ("300", "1|0"), ("500", "0|0"), ("800", "0|1"), ("400000", "0")]
    c = world["tmp"] / "c.vcf.gz"
    G.member_background(world["trio"], "C", "JD-1-P", c, joint=True)
    assert [r.rstrip("\n").split("\t")[9] for r in gzip.open(c, "rt") if not r.startswith("#")] == ["0|1", "1|0", "0|0", "./.", "1"]


def test_ref_rows_are_written_explicitly(world):
    bg = world["tmp"] / "m.vcf.gz"
    G.member_background(world["trio"], "M", "JD-1-M", bg, joint=True)
    out = world["tmp"] / "mother.vcf.gz"
    G.spike(bg, out, [({"chrom": "1", "pos": 200, "ref": "C", "alt": "A"}, "ref"),
                      ({"chrom": "X", "pos": 400100, "ref": "A", "alt": "T"}, "ref")], G.background_style(bg))
    assert G.readback(out, "1", 200, "C", "A") == "0|0"
    assert G.readback(out, "X", 400100, "A", "T") == "0|0"          # a female: diploid ref
    assert G.gt_string("ref", True, True) == "0" and G.gt_string("ref", False, False) == "0/0"


def test_missing_genotypes_do_not_decide_the_phasing_style(world):
    c = world["tmp"] / "c2.vcf.gz"
    G.member_background(world["trio"], "C", "JD-1-P", c, joint=True)
    assert G.background_style(c)["phased"] is True                  # ./. rows are not "unphased"


def _trio_gen(world, write_parent_rows):
    from haenv_rare.gen import _sha256
    v = {"chrom": "1", "pos": 200, "ref": "C", "alt": "A", "gnomad": None}
    gts = {"proband": "0/1", "father": "0/0", "mother": "0/0"}
    files, style = {}, None
    for role, smp, pid in (("proband", "C", "JD-1-P"), ("father", "F", "JD-1-F"), ("mother", "M", "JD-1-M")):
        bg = world["tmp"] / f"bg-{role}.vcf.gz"
        G.member_background(world["trio"], smp, pid, bg, joint=True)
        style = style or G.background_style(bg)
        out = world["tmp"] / f"{role}.vcf.gz"
        zyg = "het" if role == "proband" else "ref"
        G.spike(bg, out, [(v, zyg)] if (role == "proband" or write_parent_rows) else [], style)
        files[role] = {"path": str(out), "sha256": _sha256(out), "sample": pid, "gt": gts[role]}
    return {"source": "rare-os", "variant": v, "genotypes": gts, "decoys": [], "files": files, "style": style}


def test_gate_refuses_a_trio_whose_parents_have_no_row_at_the_spike(world):
    from haenv_rare.gen import _check_os_genome
    assert [h["kind"] for h in _check_os_genome(_trio_gen(world, True))] == []
    kinds = [h["kind"] for h in _check_os_genome(_trio_gen(world, False))]
    assert kinds.count("rare_spike_readback_failed") == 2                  # father and mother


def test_male_sex_chromosomes_outside_the_par_are_haploid():
    assert G.haploid_site("X", 3_000_100, "M") and not G.haploid_site("X", 3_000_100, "F")
    assert G.haploid_site("Y", 3_000_000, "M") and G.haploid_site("chrY", 3_000_000, "M")
    assert not G.haploid_site("X", 20_000, "M")                    # PAR1
    assert not G.haploid_site("Y", 20_000, "M")                    # PAR1 on Y
    assert not G.haploid_site("7", 3_000_100, "M")
    assert not G.haploid_site("X", 400_100, "M")                   # the fixture's chrX sites are PAR1


def test_gate_refuses_a_heterozygous_male_at_a_non_par_sex_chromosome_site(world):
    from haenv_rare.gen import _check_os_genome, _sha256
    gen = _trio_gen(world, True)
    assert _check_os_genome(gen, proband_sex="F") == []
    f = gen["files"]["father"]
    bad = world["tmp"] / "father-bad.vcf.gz"
    G.spike(pathlib_path(f["path"]), bad, [({"chrom": "X", "pos": 3_000_100, "ref": "A", "alt": "T"}, "het")],
            gen["style"])
    gen["files"]["father"] = {**f, "path": str(bad), "sha256": _sha256(bad)}
    kinds = [h["kind"] for h in _check_os_genome(gen, proband_sex="F")]
    assert "rare_male_het_nonpar" in kinds
    # the same row in a female member file is fine
    gen2 = _trio_gen(world, True)
    m = gen2["files"]["mother"]
    ok = world["tmp"] / "mother-x.vcf.gz"
    G.spike(pathlib_path(m["path"]), ok, [({"chrom": "X", "pos": 3_000_100, "ref": "A", "alt": "T"}, "het")], gen2["style"])
    gen2["files"]["mother"] = {**m, "path": str(ok), "sha256": _sha256(ok)}
    assert "rare_male_het_nonpar" not in [h["kind"] for h in _check_os_genome(gen2, proband_sex="F")]


def pathlib_path(p):
    import pathlib
    return pathlib.Path(p)
