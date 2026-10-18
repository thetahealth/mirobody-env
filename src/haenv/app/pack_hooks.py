"""Application-owned hook state; the contracts package contains only data and interfaces."""
from haenv.contracts.hooks import HookRegistry
from haenv.contracts.hooks import Stage

HOOKS = {}
# Extensions formerly decorated history_dependency.install. Order is explicit data.
PACK_EXTENSIONS = {}


def register_hook(stage, name, callback):
    HookRegistry(HOOKS).register(stage, name, callback)


def call_hook(stage, base, *args, **kwargs):
    return HookRegistry(HOOKS).call(stage, base, *args, **kwargs)


def register_pack_extension(name, callback):
    PACK_EXTENSIONS.setdefault(name, callback)
