"""Select declared reporting capabilities without reading maintainer scripts."""
from functools import lru_cache


@lru_cache(maxsize=None)
def tool(name):
    if name == 'm2_raw_correct':
        from haenv.reporting.tools import m2_raw_correct
        return m2_raw_correct
    if name == 'reliability_passk':
        from haenv.reporting.tools import reliability_passk
        return reliability_passk
    raise ValueError(f'Unknown reporting capability: {name!r}')
