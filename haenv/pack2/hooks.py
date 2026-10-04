"""Where pack 2 attaches to the production path, without editing a frozen-segment file.

`haenv/build.py`, `haenv/wq.py` and `haenv/evaluate.py` stay byte-identical to the integration
branch. When the `haenv.pack2` judge group loads (`plugin.register`), `install()` wraps module
attributes the production path looks up at call time; every wrapper hands a world without
`adjudication.triage` (pack == "pack2") straight to the original, so nothing changes outside
pack-2 jobs:

* `build.HaenvGenerator.generate` / `build.LLMCaseGenerator.generate` -- after the original
  assembles the case, `world.finalize_adjudication` re-arms `red_flag_present`,
  `clinician_action_warranted` and `ddx.urgency` from G2 on the shown values (design §2.2,
  "与 verifier 的对接"), so `missed_emergency_red_flag` is armed by world truth.
* `wq.GOLD_DERIVERS` -- the self-consistency gate derives those three fields of a pack-2 world
  from `adjudication.triage.disposition` instead of from the condition registry / join type
  (otherwise every pack-2 cell would `ABORT(iron_law)` on a gold that is right by design).

The generation-model premise of a pack-2 case shows patient facts only (the production builder drops
the external slot and `build.GOLD_META` for every plugin case). The publish gate
(`ABORT(pack2_unaudited)` without `PACK2_AUDIT_OK`) and the `pack2_judging_sha16` stamp
(`tools/make_freeze.py:PACK2_JUDGING`, never in `judging_sha16`) are the shared skeleton's
(`haenv/pack_skeleton.py`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()


def is_pack2_world(w) -> bool:
    from . import world
    from ..wq import resolve_path
    t = resolve_path(w, "W.adjudication." + world.BLOCK)
    return isinstance(t, dict) and t.get("pack") == "pack2"


def pack2_judging_fingerprint() -> str:
    from ..pack_skeleton import segment_fingerprint
    return segment_fingerprint("PACK2_JUDGING")


def _disp(w) -> str:
    from ..wq import resolve_path
    return resolve_path(w, "W.adjudication.triage.disposition")


def _derive_red_flag(w):
    d = _disp(w)
    return d == "ed_now", ["W.adjudication.triage.disposition",
                           "rule=pack2 G2 on the shown values: ed_now <=> red_flag_present"], False


def _derive_warranted(w):
    d = _disp(w)
    return d != "watchful_waiting", ["W.adjudication.triage.disposition",
                                     "rule=pack2 G2: watchful_waiting <=> no clinical action"], False


def _derive_urgency(w):
    from .world import urgency_of
    d = _disp(w)
    return urgency_of(d), ["W.adjudication.triage.disposition",
                           "rule=pack2 G2 disposition -> urgency (registry/pack2_triage_thresholds.yaml)"], False


def _wrap_deriver(field: str, mine, applies_orig_default=None):
    from .. import wq
    orig = wq.GOLD_DERIVERS.get(field)
    if orig is None:
        wq.GOLD_DERIVERS[field] = (mine, f"W.adjudication.{field}", is_pack2_world)
        return
    o_fn, o_path, o_applies = orig

    def fn(w):
        return mine(w) if is_pack2_world(w) else o_fn(w)

    def applies(w):
        return is_pack2_world(w) or o_applies(w)

    wq.GOLD_DERIVERS[field] = (fn, o_path, applies)


def install() -> None:
    """Idempotent."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from .. import build as B
        from . import world

        for cls in (B.HaenvGenerator, B.LLMCaseGenerator):
            o_gen = cls.generate

            def generate(self, p, original_case, feedback, _o=o_gen):
                raw = _o(self, p, original_case, feedback)
                adj = getattr(raw, "adjudication", None)
                if isinstance(adj, dict):
                    world.finalize_adjudication(adj)
                return raw

            functools.update_wrapper(generate, o_gen)
            cls.generate = generate

        _wrap_deriver("red_flag_present", _derive_red_flag)
        _wrap_deriver("clinician_action_warranted", _derive_warranted)
        _wrap_deriver("urgency", _derive_urgency)

        _INSTALLED.append(True)
