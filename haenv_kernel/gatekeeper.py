"""Gatekeeper world model: reveals only the observations the solver requests.

Holds the hidden state S (the solver payload's signals plus a cost table); `query()`
reveals one signal at a time and `advance()` moves the time pointer in multi-round loops.
"""
from __future__ import annotations

import logging

log = logging.getLogger("harness.gatekeeper")

# Simplified cost table (customizable): action -> cost
COST = {"ask": 1.0, "vitals": 1.0, "basic_lab": 5.0, "advanced_lab": 20.0,
        "imaging": 80.0, "cgm": 15.0, "referral": 10.0, "commit": 0.0}


class Gatekeeper:
    def __init__(self, solver_signals: dict[str, list[dict]], now_ptr: int):
        self._S = solver_signals          # hidden: the solver may not read this directly
        self.now = now_ptr                # current time pointer (day)
        self.spent = 0.0
        self.revealed: set[str] = set()

    def query(self, kind: str, target: str | None = None) -> dict:
        """Reveals on demand. target names a signal; a hit returns it as-is, a miss is handed to
        the caller's synthesis hook."""
        self.spent += COST.get(kind, 5.0)
        if kind == "commit":
            return {"ack": True, "cost": self.spent}
        if target and target in self._S:
            pts = [p for p in self._S[target] if p["ts"] <= self.now]
            self.revealed.add(target)
            return {"signal": target, "series": pts, "cost": self.spent}
        # Miss -> need_synth: the environment may synthesize a value consistent with S.
        return {"signal": target, "series": None, "need_synth": True, "cost": self.spent}

    def advance(self, days: int, future: dict[str, list[dict]]) -> dict:
        """Multi-round: advances the time pointer, returning newly arrived outcome observations
        (now < ts <= now+days)."""
        lo, hi = self.now, self.now + days
        obs = {sig: [p for p in pts if lo < p["ts"] <= hi] for sig, pts in future.items()}
        self.now = hi
        log.info("[gatekeeper] advance -> day %d", self.now)
        return {sig: v for sig, v in obs.items() if v}
