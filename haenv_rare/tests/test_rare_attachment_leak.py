"""Solver-visible text attachments must not name the answer. The PED used to open with
`# haenv rare.py · <diagnosis> · <inheritance>`: every shipped PED told a solver that reads it the
diagnosis. The PED is now plain (a column header only), and gate R12 scans every solver-visible
text attachment (PED in full, VCF header lines) for the diagnosis / alias / gene / ORPHA name."""
import gzip

from haenv_rare.gen import attachment_text_leaks, write_ped


def test_ped_carries_no_diagnosis(tmp_path):
    p = tmp_path / "JD-1.ped"
    write_ped(p, "JD-1", "M", "AD", trio=True)
    text = p.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "#FamilyID\tIndividualID\tPaternalID\tMaternalID\tSex\tPhenotype"
    assert text.splitlines()[1] == "JD-1\tJD-1-P\tJD-1-F\tJD-1-M\t1\t2"
    assert "AD" not in text.replace("JD-1", "")                 # no inheritance either
    assert len(text.splitlines()) == 4


def test_single_ped(tmp_path):
    p = tmp_path / "JD-2.ped"
    write_ped(p, "JD-2", "F", "AR", trio=False)
    assert p.read_text(encoding="utf-8").splitlines()[1:] == ["JD-2\tJD-2-P\t0\t0\t2\t2"]


def test_gate_finds_a_name_in_a_ped_and_in_a_vcf_header(tmp_path):
    ped = tmp_path / "a.ped"
    ped.write_text("# haenv rare.py · Fabry disease · XLR\nF\tF-P\t0\t0\t1\t2\n", encoding="utf-8")
    vcf = tmp_path / "p.vcf.gz"
    with gzip.open(vcf, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n##reference=GRCh38\n##source=GLA panel\n"
                 "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n1\t5\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n")
    att = {"genome": {"files": {"proband": {"path": str(vcf)}}, "ped": {"path": str(ped)}}}
    hits = attachment_text_leaks(att, ["Fabry disease", "GLA", ""])
    assert sorted(h.split(":")[0] for h in hits) == ["a.ped", "p.vcf.gz"]


def test_clean_attachments_pass(tmp_path):
    ped = tmp_path / "b.ped"
    write_ped(ped, "JD-3", "M", "XLR", trio=True)
    vcf = tmp_path / "q.vcf.gz"
    with gzip.open(vcf, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n##reference=GRCh38\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n")
    att = {"genome": {"files": {"proband": {"path": str(vcf)}}, "ped": {"path": str(ped)}}}
    assert attachment_text_leaks(att, ["Fabry disease", "GLA"]) == []
    assert attachment_text_leaks({}, ["GLA"]) == []
