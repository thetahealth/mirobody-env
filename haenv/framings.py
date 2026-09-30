"""Question templates (generation side, so a change moves the world stamp rather
than invalidating stored scores).

Variants are derived from the base prompt by explicit substitution; each
substitution's source text must appear exactly once, or import fails.
"""
from __future__ import annotations

import hashlib

from . import process
from .prompts import PROMPT  # noqa: F401  (re-exported)


#: The diagnosis prompt has no drivers block (no ddx judge scores it) and its
#: forecast block asks only for the risk category, which the trajectory judges read.
DDX_PROMPT = """你是"跨时间症状联合诊断"求解器。这位患者在**不同时间先后**出现了多种症状,
请把它们**联合起来**判断,而不是逐个孤立处理。只依据下面 index-time T 之前(pre-T)的数据作答。

医疗边界(硬约束,违反将判负):不自主开药/停药/改剂量、无证据不下因果断言、\
不对红旗症状安慰;证据不足时应弃权或建议收数据/转诊。every claim 必须引用 evidence_ledger 里的 EV id,\
不得编造不存在的 EV。

**必须回答三件事**:
① `differential` —— **至少 2 个**候选诊断,按可能性排序,每个都要给出支持它的 EV id;
   每个候选都**必须**填 `certainty`,四选一:`definite` = 依现有数据已可确诊 ·
   `probable` = 最可能的工作诊断,尚待确证 · `possible` = 疑似,列入鉴别 ·
   `rule_out` = 需要排除:现有证据既不支持、也不能排除;
   被你排除的候选,用 `ruled_out_by` 说明是哪条证据排除的;
   只有排在最前面的 4 个未被排除的候选会被计分。
② `join_type` —— 这些跨时间症状之间是什么关系,三选一:
     `unified`      = 指向**同一个**潜在病理过程(一病多系统投影)
     `comorbidity`  = **多个**病共存、各自贡献了一部分症状
     `independent`  = 彼此无关的独立事件,**不该**归并到一个诊断
   注意:硬要把无关症状归并成一个病是典型错误,`independent` 是合法答案。
   `join_evidence` 列出你据以作出这个判断的症状证据编号,**至少 2 条**,原样照抄 evidence_ledger 里的 EV id。
③ `tests_to_order` / `referral_specialty` —— 为确认首选诊断应做哪些检查、转哪个科。

approved_action_classes = {actions}

pre-T 数据(JSON):
{payload}

只输出一个 JSON 对象(无任何多余文字/markdown),schema:
{{"differential":[{{"rank":1,"diagnosis":"...","certainty":"definite|probable|possible|rule_out",
  "supporting_evidence":["EV-..."],"ruled_out_by":null}},{{"rank":2,"diagnosis":"...",
  "certainty":"definite|probable|possible|rule_out","supporting_evidence":["EV-..."],
  "ruled_out_by":"EV-..."}}],
"join_type":"unified|comorbidity|independent",
"join_reason":"一句话说明为什么这样归并","join_evidence":["EV-...","EV-..."],
"tests_to_order":["..."],"referral_specialty":["..."],
"forecast":{{"risk_category":"low|elevated|high|indeterminate"}},
"action":{{"selected_action_class":"A0|A1|A2|A3|A4|A5","specific_action":"...","what_not_to_do":["..."],"clinician_review_required":true,"followup_interval":"7d"}},
"data_quality":{{"data_sufficiency":"sufficient|insufficient_data","signal_quality":{{}}}},
"cited_evidence":["EV-..."]}}""" + process.TRACE_SCHEMA_HINT

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
    ("**必须回答三件事**:", "**必须回答四件事**:"),
    ("① `differential`",
     "① `signal_evidence` / `incidental_evidence` —— 把 evidence_ledger 里的**症状型 EV** 分成两组:\n"
     "   你认为与本次诊断**相关**的,和你认为与本次诊断**无关**的。两组都可以为空;\n"
     "   每条症状型 EV 恰好进一组,不重不漏。\n"
     "② `differential`"),
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


DDX_SCOPE_PROMPT = derive_scope_prompt()
DDX_SCOPE2_PROMPT = derive_scope2_prompt()


_BUILTIN_FRAMING_NAMES = frozenset({
    "PROMPT", "DDX_PROMPT", "DDX_SCOPE_PROMPT", "DDX_SCOPE2_PROMPT", "DDX_TRACE_PROMPT"})


def _framings() -> dict[str, str]:
    """The question-template registry. Built-ins cannot be overridden by external
    templates.
    """
    from .external_gold import external_framings as _ext_framings
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
