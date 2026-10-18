"""Optional acute triage pack readouts."""
from functools import partial
from haenv.reporting.pack_readout import ReadoutPack
from haenv.reporting.pack_readout import path


def pack():
    from haenv.pack2.plugin import pack as original
    from haenv.pack2.score import composite
    return ReadoutPack(**vars(original()), raw_correct=lambda r: r.get('pack2_disposition_ok'),
                       main_score=lambda rs: composite(rs)['composite'],
                       base_key=partial(path, dotted='adjudication.ddx.spec_id'),
                       dimensions={key: partial(path, dotted=f'adjudication.triage.{key}')
                                   for key in ('disposition', 'focal', 'prior_tier')})
