"""A spiked VCF is byte-reproducible: its sha256 goes into the gold and into the solver's pointer,
so regenerating an unchanged pack must not move it. gzip writes the wall-clock time into its
header unless told otherwise; that alone changed every VCF hash of rare_coding-p3 on
regeneration (content identical)."""
import gzip

import haenv_rare.gen as G

SKELETON = ("##fileformat=VCFv4.2\n##contig=<ID=chr1>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
            "chr1\t100\t.\tA\tG\t50\tPASS\t.\tGT\t0/1\nchr1\t300\t.\tC\tT\t50\tPASS\t.\tGT\t1/1\n")
SPIKE = ({"chrom": "1", "pos": 200, "ref": "G", "alt": "A"}, "0/1")


def _spike(tmp_path, name, clock, monkeypatch):
    sk = tmp_path / "sk.vcf.gz"
    sk.write_bytes(gzip.compress(SKELETON.encode(), mtime=0))
    monkeypatch.setattr(gzip.time, "time", lambda: clock)
    out = tmp_path / name / "proband.vcf.gz"
    out.parent.mkdir()
    G.spike_vcf(sk, out, [SPIKE])
    return out.read_bytes()


def test_same_input_same_bytes_at_different_times(tmp_path, monkeypatch):
    a = _spike(tmp_path, "a", 1_000_000_000.0, monkeypatch)
    b = _spike(tmp_path, "b", 1_790_000_000.0, monkeypatch)
    assert gzip.decompress(a) == gzip.decompress(b)          # control: the content is the same
    assert a == b


def test_the_spike_lands_in_position(tmp_path, monkeypatch):
    body = gzip.decompress(_spike(tmp_path, "c", 0.0, monkeypatch)).decode()
    pos = [int(l.split("\t")[1]) for l in body.splitlines() if not l.startswith("#")]
    assert pos == [100, 200, 300]
    assert "chr1\t200\t.\tG\tA\t900\tPASS\t.\tGT\t0/1" in body
