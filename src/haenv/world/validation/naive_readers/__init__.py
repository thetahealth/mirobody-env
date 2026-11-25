"""Offline solver-payload readers and evidence ablations.

SYNTHETIC data, evaluation only, not medical advice.
"""
from importlib import import_module

PACKS = {key: f"{__name__}.{key}" for key in ("m2", "p2", "p3", "p4")}


def adapter(name):
    return import_module(PACKS[name])
