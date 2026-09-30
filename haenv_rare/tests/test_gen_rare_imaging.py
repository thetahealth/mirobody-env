"""`--imaging` on haenv-rare-gen (haenv_rare.gen_job): specs without a collection get one
from the corpus-informed pairing table; a spec that names its own is never touched."""
from haenv_rare.gen_job import with_imaging


def test_with_imaging_fills_only_missing_specs():
    none = {"modalities": {"genome": {"skeleton": "trio", "tier": "exact_variant"}, "imaging": "none"}}
    has = {"modalities": {"genome": {"skeleton": "single", "tier": "exact_variant"},
                          "imaging": {"collection": "Soft-tissue-Sarcoma", "tier": "exact_class"}}}
    colls = ["Soft-tissue-Sarcoma", "UPENN-GBM"]
    assert with_imaging(none, "RD-TSC", 0, colls)["modalities"]["imaging"] == {"collection": "UPENN-GBM", "tier": "region_match"}
    assert with_imaging(none, "RD-WILSON", 0, colls)["modalities"]["imaging"]["tier"] == "unrelated_attachment"
    assert with_imaging(has, "RD-LFS", 5, colls) is has             # registry entry untouched
    assert with_imaging(none, "RD-TSC", 0, None) is none            # default job unchanged
    assert none["modalities"]["imaging"] == "none"                 # no mutation of the input
