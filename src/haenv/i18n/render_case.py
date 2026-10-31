"""Render built cases into the batch language (`haenv.i18n.text.render`).

`HAENV_I18N_AUDIT=<file>` writes, instead of failing on the first one, every text of the built
cases that does not render in the batch language -- and in zh every Chinese string that bypassed
the rendering layer (a plain `str`, not a `Msg`) -- as JSON lines `{case_id, path, text, reason}`.
A catalog tool; a normal build fails closed.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import dataclasses
import json
import os

from haenv.i18n.text import Msg, MissingText, has_cjk, render, render_msg, wrap
from haenv.i18n.identity import display

AUDIT_ENV = "HAENV_I18N_AUDIT"


def render_case(raw, lang: str):
    """A copy of `raw` (a `RawCase`) with every text rendered in `lang`."""
    return dataclasses.replace(raw, **{f.name: display(wrap(getattr(raw, f.name)), lang)
                                       for f in dataclasses.fields(raw)})


def gaps(obj, lang: str, path: str = "") -> list[dict]:
    """Texts that bypass the rendering layer (`plain`) or lack `lang` text (`missing`)."""
    out: list[dict] = []

    def walk(o, p):
        if isinstance(o, Msg):
            try:
                render_msg(o, lang, p)
            except MissingText as e:
                out.append({"path": p, "text": str(o), "reason": "missing", "what": e.what,
                            "key": o.key, "kind": o.kind})
        elif isinstance(o, str):
            if has_cjk(o):
                out.append({"path": p, "text": o, "reason": "plain"})
        elif isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and not isinstance(k, Msg) and has_cjk(k):
                    out.append({"path": p + "/<key>", "text": k, "reason": "plain"})
                walk(v, f"{p}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{p}[]")
    walk(obj, path)
    return out


def render_built(built: dict, lang: str) -> dict:
    audit = os.environ.get(AUDIT_ENV)
    if audit:
        with open(audit, "a", encoding="utf-8") as f:
            for cid, raw in built.items():
                for g in gaps(dataclasses.asdict(raw), lang):
                    f.write(json.dumps({"case_id": cid, **g}, ensure_ascii=False) + "\n")
    return {cid: render_case(raw, lang) for cid, raw in built.items()}
