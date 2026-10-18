"""P4 diagnostic dimensions and existing production score adapters."""
from functools import partial
from haenv.reporting.pack_readout import ReadoutPack
from haenv.reporting.pack_readout import path


def main_score(rows):
    from haenv.p4.score import cc
    rows = [r for r in rows if r.get("p4_class_gold") is not None]
    return cc([r["p4_class_gold"] for r in rows], [r.get("p4_class_said") for r in rows]) if rows else None


def pack():
    from haenv.p4.plugin import pack as original
    return ReadoutPack(**vars(original()), raw_correct=lambda r: bool(r.get("p4_class_ok")), main_score=main_score,
                       base_key=partial(path, dotted="adjudication.ddx.spec_id"),
                       dimensions={name: partial(path, dotted=f"adjudication.followup_p4.{key}")
                                   for name, key in {"class": "class", "reason": "cause", "analyte": "analyte", "next_step": "next_step"}.items()})
