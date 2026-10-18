"""P3 diagnostic dimensions and existing production score adapters."""
from functools import partial
from haenv.reporting.pack_readout import ReadoutPack
from haenv.reporting.pack_readout import path


def pack():
    from haenv.p3.plugin import pack as original
    from haenv.p3.score import pack_score
    return ReadoutPack(**vars(original()), raw_correct=lambda r: bool(r.get("p3_class_ok")),
                       main_score=lambda rs: pack_score(rs)["composite"],
                       base_key=partial(path, dotted="adjudication.ddx.spec_id"),
                       dimensions={name: partial(path, dotted=f"adjudication.meds_p3.{key}")
                                   for name, key in {"disposition": "class", "reason": "subtype", "drug": "drug", "axis": "axis"}.items()})
