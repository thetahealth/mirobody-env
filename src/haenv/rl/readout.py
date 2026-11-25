"""rl diagnostic dimensions for the pack readout (`haenv.pack_readouts`)."""
from functools import partial

from haenv.reporting.pack_readout import ReadoutPack
from haenv.reporting.pack_readout import path


def pack():
    from haenv.rl.plugin import pack as original
    return ReadoutPack(**vars(original()), base_key=partial(path, dotted="adjudication.ddx.spec_id"),
                       dimensions={name: partial(path, dotted=f"adjudication.redline_rl.{key}")
                                   for name, key in {"family": "family", "rule": "rule_id",
                                                     "question": "question_id"}.items()})
