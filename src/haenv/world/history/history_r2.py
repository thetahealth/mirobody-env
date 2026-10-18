"""Compatibility entry points for the explicit history generation policy."""
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import install


def enable():
    """History policy is selected at call time; no execution functions are replaced."""
