"""Question templates (generation side, so a change moves the world stamp rather
than invalidating stored scores).

Variants are derived from the base prompt by explicit substitution; each
substitution's source text must appear exactly once, or import fails.
"""
from __future__ import annotations

from haenv.i18n.text import L as _L

import hashlib

import haenv.scoring.process as process
from haenv.interaction.prompts import PROMPT  # noqa: F401  (re-exported)
from haenv.contracts.external_gold import _BUILTIN_FRAMING_NAMES


#: The diagnosis prompt has no drivers block (no ddx judge scores it) and its
#: forecast block asks only for the risk category, which the trajectory judges read.
DDX_PROMPT = _L('runtime.framing.DDX_PROMPT')

#: Candidate certainty in `differential[].certainty` (the question above requires it).
DIFFERENTIAL_CERTAINTY: tuple[str, ...] = ("definite", "probable", "possible", "rule_out")

#: Answer-contract identifier of DDX_PROMPT and every framing derived from it. Recorded per case
#: in the Q-side ledger (`cases.jsonl` -> `question.injected_manifest.answer_contract`, read back
#: with `wq.injected_manifest(case_id)`), never in `adjudication`: it describes what the question
#: asked for, not the gold. A case without it was generated before these fields were asked for.
DDX_ANSWER_CONTRACT: dict = {
    "version": "ddx-answer-v2",
    # every `differential[]` candidate must carry `certainty` in DIFFERENTIAL_CERTAINTY
    "differential_certainty": "required",
    "certainty_values": list(DIFFERENTIAL_CERTAINTY),
    # `join_evidence`: >=2 evidence ids supporting `join_type`
    "join_evidence": "required",
    # the question states that only the first DIFFERENTIAL_SCORED_TOP candidates not ruled out
    # are scored (the judge's cap is gold threads + 3, never below this)
    "differential_scored_top": 4,
}


# The process trace is read from the raw response and never enters SolverOutput,
# so it earns no marks.
DDX_TRACE_PROMPT = DDX_PROMPT



# `ddx.scope`: `join_gold` is defined over the real symptoms, so the join question
# asks about the evidence the model itself marked as signal. Single-factor change;
# the anti-unified warning stays.
_SCOPE_EDITS: tuple[tuple[str, str], ...] = (
    (_L('runtime.world.generation.framings.module.2'), _L('runtime.world.generation.framings.module.3')),
    ("① `differential`",
     _L('runtime.world.generation.framings.module.4')),
    ("② `join_type` —— 这些跨时间症状之间是什么关系,三选一:",
     "③ `join_type` —— **只就你填进 `signal_evidence` 的那些 EV**,它们之间是什么关系,三选一:"),
    ("③ `tests_to_order`", "④ `tests_to_order`"),
    ('"join_type":"unified|comorbidity|independent",',
     '"signal_evidence":["EV-..."],"incidental_evidence":["EV-..."],\n'
     '"join_type":"unified|comorbidity|independent",'),
)


# `ddx.scope2`: the same edits, worded without presupposing a single diagnosis.
_SCOPE2_EDITS: tuple[tuple[str, str], ...] = tuple(
    (a, b.replace("你认为与本次诊断**相关**的,和你认为与本次诊断**无关**的",
                  "你认为**属于本次问题**的,和你认为**不属于本次问题**的"))
    for a, b in _SCOPE_EDITS
)

# Singular-framing word forms that must not appear in the scope2 prompt.
_SINGULAR_FRAMING_MARKERS: tuple[str, ...] = (
    "本次诊断", "该诊断", "这个诊断", "最终诊断", "单一诊断", "一个诊断",
)


def derive_scope2_prompt(base: str = "") -> str:
    """The `ddx.scope2` variant (P1'): referent anchoring, with no presupposition
    about how many diagnoses there are; differs from `ddx.scope` (P1) only in the
    wording of the relevance split.
    """
    t = base or DDX_PROMPT
    for a, b in _SCOPE2_EDITS:
        n = t.count(a)
        if n != 1:
            raise ValueError(f"cannot derive DDX_SCOPE2_PROMPT: {a!r} occurs {n} times in the base prompt (expected 1);"
                             " DDX_PROMPT changed, update the replacement table to match")
        t = t.replace(a, b)
    return t


def derive_scope_prompt(base: str = "") -> str:
    """Derive the referent-anchored variant from the base diagnosis prompt. Every
    replacement's source text must appear exactly once, or this raises.
    """
    t = base or DDX_PROMPT
    for a, b in _SCOPE_EDITS:
        n = t.count(a)
        if n != 1:
            raise ValueError(f"cannot derive DDX_SCOPE_PROMPT: {a!r} occurs {n} times in the base prompt (expected 1);"
                             f" DDX_PROMPT changed, update the replacement table to match")
        t = t.replace(a, b)
    return t


DDX_SCOPE_PROMPT = _L('runtime.framing.DDX_SCOPE_PROMPT')
DDX_SCOPE2_PROMPT = _L('runtime.framing.DDX_SCOPE2_PROMPT')


def _framings() -> dict[str, str]:
    """The question-template registry. Built-ins cannot be overridden by external
    templates.
    """
    from haenv.contracts.external_gold import external_framings as _ext_framings
    builtin = {"PROMPT": PROMPT, "DDX_PROMPT": DDX_PROMPT,
               "DDX_SCOPE_PROMPT": DDX_SCOPE_PROMPT,
               "DDX_SCOPE2_PROMPT": DDX_SCOPE2_PROMPT,
               "DDX_TRACE_PROMPT": DDX_TRACE_PROMPT}
    ext = _ext_framings()
    if not ext:
        return builtin
    # Built-ins go last, so an external template cannot override one.
    return {**ext, **builtin}


_FRAMINGS = _framings()


def framing_sha256(name: str) -> str:
    """Short fingerprint (first 16 hex digits) of a question template; probes declare
    it and `load_probes` checks it.
    """
    return hashlib.sha256(_framings()[name].encode()).hexdigest()[:16]
