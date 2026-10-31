"""Pack 2 question template (`PACK2_TRIAGE_PROMPT`) and its probe (`pack2.triage`).

Registered through `external_gold.register_framing` / `register_probe` by the `haenv.pack2`
plugin, so `evaluate.render_with_probe` renders a pack-2 item with the production renderer: the
template below plus the solver payload JSON, whose `prediction_context.triage` carries the visit
(complaint, vitals now, point-of-care labs, 72 h logs). The four options are stated once for every
item; the per-item text is scanned for disposition words by the `pack2_leak` gate. A4 means
"within 24 h" inside this pack only (design K8).

`FRAMING_SHA256` is pinned by hand (the probe declares it, `evaluate.load_probes` checks it), so a
template edit cannot slip through unnoticed. This file is in the PACK2_WORLD segment.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import L as _L

NAME = "PACK2_TRIAGE_PROMPT"
PROBE_ID = "pack2.triage"

PACK2_TRIAGE_PROMPT = _L('runtime.pack2.framing.module.1')

#: sha256(PACK2_TRIAGE_PROMPT)[:16], pinned (see the module docstring).
FRAMING_SHA256 = "39d83d73dd05dc0c"

PROBE = {
    "probe_id": PROBE_ID,
    "mode": "ddx",
    "difficulty_probe": "P0",
    "framing_ref": NAME,
    "framing_sha256": FRAMING_SHA256,
    "answer_space": "closed",
    "hint_level": 0,
    "budget": {"max_rounds": 1},
}
