"""Ordered application stages. Registration stores callbacks; it never edits executors."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Callable


class Stage(str, Enum):
    CASE = "case"                       # scope around one case, including final snapshots
    GENERATED = "generated"             # after deterministic/LLM world generation
    OBSERVE = "observe"                 # measurements -> history -> delivery -> EMR
    PROMPT = "prompt"                   # pack framing -> frame suffix -> library/management
    PROTOCOL_QUESTION = "protocol_question"
    LOOKALIKES = "lookalikes"
    MENU = "menu"
    GATED_MENU = "gated_menu"
    SYNTH_TARGET = "synth_target"
    SYNTH_FINDING = "synth_finding"
    M2_FINDING = "m2_finding"
    RUN_GATED = "run_gated"


@dataclass
class HookRegistry:
    """A named callback takes `next_call, *args, **kwargs`.

    Calls enter in reverse registration order and unwind in registration order,
    matching the original wrappers. A call snapshots declarations; registration
    is idempotent by (stage, name). Independent registries share no run state.
    """
    hooks: dict[tuple[Stage, str], Callable] = field(default_factory=dict)

    def register(self, stage: Stage, name: str, callback: Callable) -> None:
        self.hooks.setdefault((stage, name), callback)

    def call(self, stage: Stage, base: Callable, *args, **kwargs):
        current = base
        for (registered_stage, _name), callback in tuple(self.hooks.items()):
            if registered_stage == stage:
                current = partial(callback, current)
        return current(*args, **kwargs)
