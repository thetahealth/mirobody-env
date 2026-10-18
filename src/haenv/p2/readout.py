"""P2 diagnostic dimensions and existing production score adapters."""
from functools import partial
from haenv.reporting.pack_readout import ReadoutPack
from haenv.reporting.pack_readout import path


def pack():
    from haenv.p2.plugin import pack as original
    from haenv.p2.score import composite
    return ReadoutPack(**vars(original()), raw_correct=lambda r: bool(r.get("p2_in_set")),
                       main_score=lambda rs: composite(rs)["composite"],
                       base_key=partial(path, dotted="adjudication.ddx.spec_id"),
                       dimensions={name: partial(path, dotted=f"adjudication.triage_p2.{key}")
                                   for name, key in {"family": "tier", "rule": "rule", "disposition": "floor", "stratum": "stratum"}.items()})
