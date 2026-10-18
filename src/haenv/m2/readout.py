"""M2 diagnostic dimensions and existing production score adapters."""
from functools import partial
from haenv.reporting.pack_readout import ReadoutPack
from haenv.reporting.pack_readout import path


def pack():
    from haenv.m2.plugin import pack as original
    from haenv.m2.score import solver_scores
    from haenv.reporting.readout_tools import tool
    raw_correct = tool('m2_raw_correct').raw_correct
    return ReadoutPack(**vars(original()), raw_correct=raw_correct,
                       main_score=lambda rs: solver_scores(rs)["composite_raw"],
                       base_key=partial(path, dotted="adjudication.ddx.spec_id"),
                       dimensions={"family": partial(path, dotted="adjudication.ddx.spec_id"),
                                   "join": partial(path, dotted="adjudication.ddx.join_gold"),
                                   "frame": partial(path, dotted="prediction_context.m2_frame.frame")})
