"""Model-based judge: does a claimed exclusion land on a discriminative axis?

For each look-alike the model says it ruled out, a judge model sees the gold diagnosis,
the look-alike, its registered discriminator and the model's reason, and answers
DISCRIMINATIVE, GENERIC or IRRELEVANT. A rule cannot tell an empty "does not fit" from a
real differential argument.

Not in the core set: it runs only when registered (e.g. via the entry-point plugin in
`examples/llm_judge_demo`) and when the switch is on. Its category `llm` enters the
per-part fingerprint.

    HAENV_LLM_JUDGE=1               default judge model
    HAENV_LLM_JUDGE=<model key>     a model key from config.yaml
    unset / off / 0 / false / no    off (offline runs and CI)

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("haenv.judges.llm")

#: Default judge model (the verdict is a three-way choice, so a fast model suffices); a named
#: key in the switch takes precedence.
DEFAULT_JUDGE_MODEL = "gemini-3.8-flash"

_GENERIC_ON = ("1", "on", "true", "yes")


NAME = "llmj_rival_discriminator"

#: Output keys, all under one prefix so they cannot overwrite a built-in judge's keys.
KEYS = ("llmj_disc_status", "llmj_disc_via", "llmj_disc_n_rivals",
        "llmj_disc_n_claims", "llmj_disc_n_judged",
        "llmj_disc_n_discriminative", "llmj_disc_rate", "llmj_disc_verdicts")

#: Closed verdict set; anything else is recorded as unparseable (no fuzzy matching).
VERDICTS = ("DISCRIMINATIVE", "GENERIC", "IRRELEVANT")

STATUS = ("ok", "no_rival", "no_claim", "no_dispatch", "dispatch_failed", "unparsed")

ENV_SWITCH = "HAENV_LLM_JUDGE"

_OFF_WORDS = ("", "off", "0", "false", "no")

_OVERRIDE = None


def set_dispatch(fn):
    """Inject a dispatcher `(prompt) -> str` for tests; returns the previous one. `None` resets."""
    global _OVERRIDE
    prev, _OVERRIDE = _OVERRIDE, fn
    return prev


# ------------------------------------------------------------------ prompt
#: The judging prompt. It is part of the cache key, so any edit invalidates cached verdicts.
PROMPT = """你是临床评测的判分员。只判一件事,并且只输出规定格式。

金标诊断:{gold}
需要被排除的近名:{rival}
这两者的鉴别点(权威登记,自由文本):{discriminator}

受测模型给出的排除理由(原文,可能为中文或英文):
<<<
{reason}
>>>

问题:这条排除理由**有没有触及上面那条鉴别点所指的那个轴**
(同一生理量 / 同一项检查 / 同一形态学或时间学特征;措辞不必相同,换算或同义表述算触及)?

三选一:
  DISCRIMINATIVE —— 触及了那个轴,并据此把近名与金标分开;
  GENERIC        —— 只是泛泛之词(「不符合」「可能性低」「临床表现不支持」「病史不吻合」),
                    没有指向任何能分辨这两者的具体依据;
  IRRELEVANT     —— 指向了具体依据,但那个依据分不开这两个病(或与该鉴别点无关)。

输出**恰好两行**:
第一行:上面三个词之一,大写,不加任何其它字符。
第二行:不超过 40 字的理由。
"""


def build_prompt(gold: str, rival: str, discriminator: str, reason: str) -> str:
    """Build the judging prompt from four parts only (no case id or other gold fields), so cache
    entries are shared across cases and models.
    """
    return PROMPT.format(gold=str(gold or "?"), rival=str(rival or "?"),
                         discriminator=str(discriminator or "(未登记)"),
                         reason=str(reason or "").strip()[:800])


def parse_verdict(text) -> str | None:
    """Return the verdict, or `None` if unparseable. The first non-empty line, stripped of
    markdown and punctuation, must equal a verdict exactly.
    """
    for line in str(text or "").splitlines():
        tok = line.strip().strip("*`#-_ \t").rstrip(".。:,,;;!!").strip()
        if not tok:
            continue
        up = tok.upper()
        return up if up in VERDICTS else None
    return None


# ------------------------------------------------------------------ dispatch
def _resolve_dispatch():
    """Return (dispatcher, source label): an injected override, else the environment switch,
    else none. The dispatcher has its own disk cache.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE, "override"
    key = os.environ.get(ENV_SWITCH, "").strip()
    if key.lower() in _OFF_WORDS:
        return None, f"off:{ENV_SWITCH}"
    _via_default = key.lower() in _GENERIC_ON
    if _via_default:
        key = DEFAULT_JUDGE_MODEL
    try:
        from ..cli import ROOT, load_cfg          # the existing config entry point; no second one
        from ..llm import make_dispatcher
        return (make_dispatcher(load_cfg(), key, ROOT),
                f"{'default' if _via_default else 'cfg'}:{key}")
    except (ValueError, KeyError, OSError) as e:
        # Only config errors mean "unavailable"; other exceptions propagate.
        log.warning("[llmj] dispatcher construction failed (%s: %s) => this dimension recorded as no_dispatch for this batch",
                    type(e).__name__, e)
        return None, f"unavailable:{type(e).__name__}"


# ------------------------------------------- accessors
def rivals_of(vp) -> tuple:
    from ._helpers import _rivals_of
    return _rivals_of(vp)


def gold_name(vp) -> str:
    from ..wq import gold_of
    return str(gold_of(vp, "diagnosis", default="") or "")


def claims_of(out, rivals) -> list[dict]:
    """The registered look-alikes the model claims to have ruled out, with its reasons.

    Uses the scoring-side alias matcher (with negation and exclusion filters); a look-alike
    excluded under several candidates is judged once.
    """
    from ..tracks import _differential, alias_hit_asserted
    seen: set[str] = set()
    got: list[dict] = []
    for x in _differential(out):
        reason = str(x.get("ruled_out_by") or "").strip()
        if not reason:
            continue                       # no exclusion claimed: not this judge's object
        name = str(x.get("diagnosis") or "")
        for r in rivals:
            rn = str(r.get("name") or "")
            if rn in seen:
                continue
            names = tuple(r.get("aliases") or ()) + (rn,)
            if alias_hit_asserted(name, names):
                seen.add(rn)
                got.append({"rival": rn, "discriminator": str(r.get("discriminator") or ""),
                            "reason": reason, "claimed_as": name})
                break
    return got


# ------------------------------------------------------------------ the judge
def judge_rival_discriminator(out, vp, ctx=None) -> dict:
    """Judge whether each claimed exclusion lands on a discriminative axis.

    The denominator is the number of claimed exclusions; the number of registered look-alikes
    is reported alongside.
    """
    rivals = rivals_of(vp)
    row = {"llmj_disc_status": "no_rival", "llmj_disc_via": "-",
           "llmj_disc_n_rivals": len(rivals), "llmj_disc_n_claims": 0,
           "llmj_disc_n_judged": None, "llmj_disc_n_discriminative": None,
           "llmj_disc_rate": None, "llmj_disc_verdicts": None}
    if not rivals:
        return row                          # no look-alikes registered: not applicable, not 0

    claims = claims_of(out, rivals)
    row["llmj_disc_n_claims"] = len(claims)
    if not claims:
        row["llmj_disc_status"] = "no_claim"
        return row

    dispatch, via = _resolve_dispatch()
    row["llmj_disc_via"] = via
    if dispatch is None:
        row["llmj_disc_status"] = "no_dispatch"
        return row

    gold = gold_name(vp)
    verdicts, n_fail = [], 0
    for c in claims:
        prompt = build_prompt(gold, c["rival"], c["discriminator"], c["reason"])
        try:
            resp = dispatch(prompt)
        except (RuntimeError, OSError) as e:
            n_fail += 1
            log.warning("[llmj] %s dispatch failed: %s: %s", c["rival"], type(e).__name__, e)
            verdicts.append({"rival": c["rival"], "verdict": None, "note": "dispatch_failed"})
            continue
        v = parse_verdict(resp)
        verdicts.append({"rival": c["rival"], "verdict": v,
                         "raw": str(resp or "").strip()[:120]})

    judged = [v for v in verdicts if v.get("verdict") in VERDICTS]
    row["llmj_disc_verdicts"] = verdicts
    if not judged:
        row["llmj_disc_status"] = "dispatch_failed" if n_fail else "unparsed"
        return row
    n_disc = sum(1 for v in judged if v["verdict"] == "DISCRIMINATIVE")
    row.update({"llmj_disc_status": "ok", "llmj_disc_n_judged": len(judged),
                "llmj_disc_n_discriminative": n_disc,
                "llmj_disc_rate": round(n_disc / len(judged), 3)})
    return row


# ------------------------------------------------------- mounting and factory
def _ensure_mounted() -> None:
    """Mount on the single-shot and gated geometries (idempotent); slices get a recorded reason,
    multi falls under a wildcard reason.
    """
    from .. import mount_table as mt
    if NAME in mt.MOUNT:
        return
    mt.mount(NAME, {"single": mt.OUT, "gated": mt.OUT}, why_not={
        "slices": "In slice-by-slice answers the same exclusion reason recurs across slices, and "
                  "judging per slice would pay for the same sentence N times, with the same "
                  "near-miss counted repeatedly into `rate`'s denominator. Mounting it on slices "
                  "requires first deciding the convention 'take the last slice, or dedupe across "
                  "slices' (`dx_rival` takes `LAST` on slices, and this judge has not followed "
                  "suit) -- mounting it before that is settled would produce a number whose "
                  "denominator cannot be explained.",
    })


def judges():
    """Judge factory: mount, then return the judge (an empty list on a second call).

    Kinds and precondition match the rule judge (the three diagnosis kinds, cases with
    registered look-alikes), so the two denominators line up.
    """
    from . import JUDGES
    from ._helpers import _rivals_of
    from .safety import Judge
    _ensure_mounted()
    if any(j.name == NAME for j in JUDGES):
        return []
    return [Judge(NAME,
                  ("ddx:unified", "ddx:comorbidity", "ddx:independent"),
                  judge_rival_discriminator,
                  when=lambda vp: bool(_rivals_of(vp)),
                  category="llm")]
