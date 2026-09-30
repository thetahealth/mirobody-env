"""extract.py -- case-description text -> one case in job.yaml.

Three field categories take three paths: (A) patient facts are extracted by an LLM
(`extract_patient_facts`); (B) gold comes from one of four stamped sources (`fill_gold_*`,
see `GOLD_PATHS`); (C) generation knobs are sampled deterministically by `case_id`
(`sample_knobs`) and never asked of a model. Field
categories are defined in `provenance.FIELD_SPECS`. Extracted facts are gold-free, so
no gold reaches a model; `fill_gold_llm` is the opt-in exception, reported separately.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import rng
from .provenance import (GOLD_FIELDS, PATIENT_FACTS, RARELY_IN_TEXT,
                         ProvenanceError, ProvenanceLedger, rarely_in_text)

# Resources are read through the single `data_root()` entry point (repo root, or `haenv/_data` in a wheel).
from haenv import data_root as _data_root
ROOT = _data_root()


class ExtractionError(ValueError):
    """Extraction failed. Never produces a partial case."""


@dataclass
class ExtractedCase:
    """The extraction result. `raw` + `latent` can be written directly into `cases[i]` in
    job.yaml."""
    case_id: str
    raw: dict
    latent: dict = field(default_factory=dict)
    ledger: ProvenanceLedger | None = None
    warnings: list[str] = field(default_factory=list)
    #: Where category-A facts came from: `replay:<origin>` / `cache:<model>` / `llm-extract:<model>`.
    #: Carried into the artifact, since a placeholder patient and a real extraction look identical in `raw`.
    extract_source: str = ""

    def to_case_dict(self) -> dict:
        return {"case_id": self.case_id, "raw": self.raw, "latent": self.latent}


# ==========================================================================
# A · patient facts: LLM extraction
# ==========================================================================

#: The extraction prompt: only what the text states, never gold or knobs, `null` when unsure.
#: Enumerations are filled from the kernel's tables at call time, never hard-coded.
EXTRACT_PROMPT = """你是一个**信息抽取器**,不是医生。只从下面这段病例描述里**摘录**事实。

严格规则:
1. **只摘录文本里明确写了的**。文本没写的一律填 `null`,**绝对不要推断、不要补常识**。
2. **不要给诊断、不要判断严重程度、不要建议检查** —— 那些不是你的任务。
3. 日期一律转成「距病程起点的第几天」的整数。文本只给相对描述(如"两周后")时按之换算;
   完全无法定位时间的主诉,`day` 填 null。

输出**纯 JSON**,不要 ```、不要解释:
{{
  "disease": "{diseases} 之一,或 null",
  "age_range": "如 45-49,或 null",
  "sex": "M|F|null",
  "bmi": 数字或 null,
  "comorbidities": ["..."] 或 [],
  "drug": "**必须是这几个之一**(病例里若是中文名/商品名,换成对应的这个英文名):{drugs};都不是则 null",
  "dose_steps": [数字] 或 [],
  "devices": [{devices} 的子集] 或 [],
  "start_weight": 数字或 null,
  "nadir_weight": 数字或 null,
  "symptoms": [{{"day": 整数或 null, "text": "主诉原话", "context": "补充情境,没有则空串"}}]
}}

病例描述:
---
{text}
---"""


def _parse_json(s: str) -> dict:
    """Extract the JSON from a model's output. Tolerates ``` fencing and surrounding noise,
    but does not tolerate more than one JSON object."""
    s = re.sub(r"^\s*```(?:json)?|```\s*$", "", (s or "").strip(), flags=re.M)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ExtractionError(f"no JSON found in the output: {s[:120]!r}")
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError as e:
        raise ExtractionError(f"JSON parsing failed: {e}; source text {s[i:i + 120]!r}") from e



# Extraction model calls are replayable, looked up in priority order:
#   1. `docs/anchor/extract-replay/<key>.json` -- checked-in replay fixtures (offline);
#   2. `cases/_llm_cache/<key>.json` -- the generation cache;
#   3. a live model call, refused when `offline=True`.
# The key comes from the dispatcher (`Dispatcher._key`), and every fixture records its `origin`,
# which flows into the provenance stamp.

#: Checked-in replay fixtures, same key scheme as the generation cache.
REPLAY_DIR = ROOT / "docs" / "anchor" / "extract-replay"


class OfflineExtractionMiss(ExtractionError):
    """Offline extraction missed both the fixture and the cache.

    Raised rather than falling back to an empty extraction: downstream defaults would turn
    an empty `raw` into a patient the text never described (see `invented_fields`).
    """


def _dispatch_for(model: str, cfg: dict, cache_subdir: str = "_llm_cache"):
    """The dispatcher used for extraction — the same construction entry point as the
    question-authoring side, hence the same cache and the same key scheme."""
    from .llm import make_dispatcher
    spec = (cfg.get("models") or {}).get(model)
    if not isinstance(spec, dict):
        # Extraction only uses the mapping form of a model entry.
        raise ExtractionError(f"config.models has no mapping-form entry for {model!r}")
    try:
        return make_dispatcher(cfg, model, ROOT, cache_subdir=cache_subdir)
    except ValueError as e:
        raise ExtractionError(str(e)) from e


def _complete(prompt: str, model: str, cfg: dict, *, offline: bool = False,
              cache_subdir: str = "_llm_cache") -> tuple[str, str]:
    """Return `(response text, source label)` for a prompt: `replay:<origin>` (checked-in
    fixture), `cache:<model>` (generation cache) or `llm-extract:<model>` (live call).
    """
    d = _dispatch_for(model, cfg, cache_subdir)
    key = d._key(prompt)
    fx = REPLAY_DIR / f"{key}.json"
    if fx.is_file():
        try:
            rec = json.loads(fx.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ExtractionError(f"replay fixture {fx.name} failed to parse: {e}") from e
        origin = str(rec.get("origin") or "unknown")
        if "response" not in rec:
            raise ExtractionError(f"replay fixture {fx.name} has no `response` field")
        return str(rec["response"]), f"replay:{origin}"
    hit = d._load(key)
    if hit is not None:
        return hit, f"cache:{model}"
    if offline:
        raise OfflineExtractionMiss(
            f"offline extraction missed: key `{key}` (model={model}) is in neither\n"
            f"  ① the replay fixtures {REPLAY_DIR}\n  ② the generation cache {ROOT / 'cases' / '_llm_cache'}\n"
            f"and the offline profile forbids network calls. Does not fall back to an empty "
            f"extraction — an empty `raw` gets filled in by downstream defaults into a patient "
            f"who doesn't come from this text.\n"
            f"Either drop the offline profile and extract once with network access (which writes "
            f"automatically into the cache outside of ①), or put a fixture for this key into {REPLAY_DIR}.")
    return d(prompt), f"llm-extract:{model}"


#: Markup markers, stripped from both sides before the verbatim comparison.
_MARKUP_RE = re.compile(r"[*_`~]+")
_WS_RE = re.compile(r"\s+")


def _norm_for_quote(x: str) -> str:
    """Strip markup markers and collapse whitespace, nothing more.

    A verbatim quote can cross an emphasis marker in the source, so markers are stripped.
    No fuzzy or synonym matching: this check separates transcription from generation.
    """
    return _WS_RE.sub("", _MARKUP_RE.sub("", x or ""))


def _quote_is_verbatim(quote: str | None, text: str) -> bool:
    """Whether the quote genuinely comes from the source text (character-for-character
    containment after stripping markup markers)."""
    if not quote:
        return False
    q = _norm_for_quote(str(quote))
    return bool(q) and q in _norm_for_quote(text)


def extract_patient_facts(text: str, case_id: str, model: str = "gemini-3.1-pro",
                          cfg: dict | None = None,
                          ledger: ProvenanceLedger | None = None,
                          offline: bool = False) -> ExtractedCase:
    """Category A: extract patient facts from text (no gold, no knobs)."""
    import yaml

    cfg = cfg or _cached_yaml(ROOT / "config.yaml")

    from latent import (DISEASE_SIGNAL_DOMAIN, DRUG_PKPD, KNOWN_DEVICES,   # kernel
                        drug_pkpd)
    _dis = "|".join(sorted(DISEASE_SIGNAL_DOMAIN))
    _dev = "|".join(sorted(KNOWN_DEVICES))
    # The prompt lists the kernel's drug names: an unrecognized name makes `drug_pkpd()` return
    # None, which silently skips the dose-rung check and the dosing-frequency lookup.
    _drg = "|".join(sorted(k for k in DRUG_PKPD if k != "glp1"))   # glp1 is a drug family name, not a drug
    _resp, _src = _complete(EXTRACT_PROMPT.format(
        text=text, diseases=_dis, devices=_dev, drugs=_drg),
        model, cfg, offline=offline)
    got = _parse_json(_resp)
    led = ledger or ProvenanceLedger(case_id)
    raw, warns = {}, []
    for k in ("disease", "age_range", "sex", "bmi", "comorbidities", "drug",
              "dose_steps", "devices", "start_weight", "nadir_weight", "symptoms"):
        v = got.get(k)
        if v is None or v == [] or v == "":
            warns.append(f"{k}: 文本里没有 ⇒ 不写进 raw(不填默认值)")
            continue
        raw[k] = v
        led.stamp(k, "source_text", by=_src,
                  note="LLM 抽取,非判定"
                       + ("(重放:这一份不是本次调模型得到的)"
                          if _src.startswith("replay:") else ""))
    # `symptoms` without a day are dropped, never given a guessed day. Values outside the kernel
    # vocabulary are reported, never silently dropped.
    _bad_dev = [d for d in (raw.get("devices") or []) if d not in KNOWN_DEVICES]
    if _bad_dev:
        warns.append(f"devices: {_bad_dev} 不在内核 KNOWN_DEVICES({_dev})里 ⇒ 剔除")
        raw["devices"] = [d for d in raw["devices"] if d in KNOWN_DEVICES]
    if raw.get("drug") and not drug_pkpd(str(raw["drug"])):
        warns.append(f"drug: {raw['drug']!r} 在内核 DRUG_PKPD({_drg})里查不到 ⇒ "
                     f"剂量梯级校验与给药频次都会被跳过(静默失效)")
    if raw.get("disease") and raw["disease"] not in DISEASE_SIGNAL_DOMAIN:
        warns.append(f"disease: {raw['disease']!r} 不在内核 DISEASE_SIGNAL_DOMAIN"
                     f"({_dis})里 ⇒ 该例过不了 validate_premise")
    # Context values not registered in `registry/vocab_context.yaml` are dropped at render time;
    # report them so the caller knows they will not reach the question.
    if raw.get("symptoms"):
        try:
            from .overlay import check_vocab as _cv
            _ctx = sorted({str(s.get("context") or "").strip()
                           for s in raw["symptoms"] if s.get("context")})
            _unreg = _cv(_ctx) if _ctx else []
            if _unreg:
                warns.append(f"context: {len(_unreg)}/{len(_ctx)} 条未登记在 "
                             f"vocab_context.yaml ⇒ 会被默认丢弃(不夹带考点,但"
                             f"原文那句情境不会进题面)。要它们进题面须显式定 facet。")
        except Exception as _e:                                # noqa: BLE001
            warns.append(f"context 词表覆盖查不了:{type(_e).__name__} —— "
                         f"不要据此认为它们都已登记")
    if raw.get("symptoms"):
        keep = [s for s in raw["symptoms"] if isinstance(s.get("day"), int)]
        if len(keep) != len(raw["symptoms"]):
            warns.append(f"symptoms: 丢掉 {len(raw['symptoms']) - len(keep)} 条无时点主诉"
                         f"(不猜时间)")
        raw["symptoms"] = keep
    return ExtractedCase(case_id, raw, {}, led, warns, extract_source=_src)


# ==========================================================================
# C · generation knobs: deterministic sampling (never touches an LLM)
# ==========================================================================

#: Knob domains, drawn by `case_id` (reproducible, no API cost). Tables with an authoritative
#: source are read from it, never copied.


def _outcome_domain() -> tuple:
    """The domain of `outcome` = `job.OUTCOMES`."""
    from .job import OUTCOMES
    return tuple(OUTCOMES)


KNOB_DOMAIN: dict[str, tuple] = {
    # Long tiers let the follow-up interval (median ≈ T/13) approach a real ~28-day outpatient rhythm.
    "index_time_T": (56, 70, 84, 98, 112, 168, 336),
    # Must satisfy `>= T + 112` (see `ddx._course_end_for`), or large-T cases fail
    # `L5-outcome-sustainable`.
    "course_end_day": (224, 280, 365, 448, 504, 560),
    # Exposes the domain for validation; the value itself is drawn by `job.outcome_of`.
    "outcome": _outcome_domain(),
}

#: `distractor_level` is not sampled: high-intensity distraction injects events the density
#: declaration does not account for (`event_density_mismatch`). Cases take the pipeline default.
DISTRACTOR_NOT_SAMPLED_WHY = (
    "高强度干扰会注入额外事件而密度声明未计入 ⇒ event_density_mismatch。"
    "需先建模其对密度的贡献,另案处理。")


def sample_knobs(case_id: str, n_symptoms: int, course_end_day: int | None = None,
                 ledger: ProvenanceLedger | None = None,
                 symptom_days: list[int] | None = None,
                 drug: str | None = None) -> dict:
    """Category C: deterministic sampling keyed on `case_id`.

    `dosing_per_week` is looked up in `gates.DRUG_DOSES_PER_WEEK` (left undeclared if
    unknown). `symptom_rate` is computed over the observation window `T`, the window the
    emission gate judges (see `latent_rules`' L2), and is at least 1 / weeks of T.
    """
    from .gates import DRUG_DOSES_PER_WEEK

    led = ledger or ProvenanceLedger(case_id)
    T = int(rng.pick(list(KNOB_DOMAIN["index_time_T"]), case_id, "T"))
    ce = int(course_end_day or rng.pick(list(KNOB_DOMAIN["course_end_day"]),
                                        case_id, "course_end"))
    weeks = round(ce / 7.0, 1)
    _T_weeks = max(1.0, T / 7.0)
    _in_T = ([d for d in symptom_days if isinstance(d, int) and d <= T]
             if symptom_days else None)
    _n_eff = len(_in_T) if _in_T is not None else n_symptoms
    # At least one event, since the injector always injects one.
    _rate = round(max(_n_eff, 1) / _T_weeks, 2)
    _dpw = DRUG_DOSES_PER_WEEK.get(str(drug or "").strip().lower())
    # `outcome`/`driver` are drawn per case by `job.outcome_of` and stamped `sampled`.
    from .job import driver_of as _driver_of, outcome_of as _outcome_of
    _oc = _outcome_of(case_id)
    knobs = {
        "index_time_T": T,
        "course_end_day": ce,
        "outcome": _oc,
        "driver": _driver_of(case_id, {"outcome": _oc}),
        # `distractor_level` left undeclared -- see DISTRACTOR_NOT_SAMPLED_WHY
        "event_density": {
            "measure_per_week": 7,
            # undeclared when the drug is unknown: GEN20b accepts an absence but fails a guessed value
            **({"dosing_per_week": int(_dpw)} if _dpw else {}),
            "symptom_rate": _rate,
            "life_event_rate": 0.1,
            "clinical_symptoms_recorded": _n_eff,
            "course_weeks": weeks,
        },
    }
    # Knobs are stamped `sampled`, never `source_text`: they are not in the text.
    for k in knobs:
        led.stamp(k, "sampled", by=f"rng:{case_id}",
                  note="确定性抽样(同一 case_id 永远同一组),非文本抽取")
    return knobs


# ==========================================================================
# B · gold ruling: four paths
# ==========================================================================

def fill_gold_absent(case_id: str, ledger: ProvenanceLedger | None = None,
                     fields=GOLD_FIELDS) -> tuple[dict, ProvenanceLedger]:
    """B-1: every gold field is recorded as `absent`; returns an empty `latent` fragment."""
    led = ledger or ProvenanceLedger(case_id)
    for f in fields:
        led.stamp(f, "absent", note="B 路①:弃掉")
    return {}, led


def fill_gold_from_text(case_id: str, text: str, model: str = "gemini-3.1-pro",
                        offline: bool = False,
                        cfg: dict | None = None,
                        ledger: ProvenanceLedger | None = None,
                        base_disease: str | None = None,
                        input_kind: str = "prospective"
                        ) -> tuple[dict, ProvenanceLedger]:
    """B-2: transcription. A gold field is recorded only if the text states it explicitly;
    the LLM is used only as a parser.

    `base_disease` is category A's `disease`; it stops the underlying condition from being
    transcribed as the differential diagnosis. Fields in `rarely_in_text(input_kind)` are
    always `absent`.

    `input_kind` declares the text: `prospective` (default) is the intake complaint and
    baseline, with no outcome; `retrospective` is a full case report that states the
    outcome, which opens `ddx_outcome_label` to transcription. Which fields open is set in
    `provenance.RETRO_TRANSCRIBABLE`.
    """
    import yaml

    cfg = cfg or _cached_yaml(ROOT / "config.yaml")
    prompt = (
        "只回答:下面这段病例描述里,**原文有没有明确写出一个鉴别诊断结论**?\n\n"
        "🔴 **不算数的**(这几样是背景,不是诊断结论):\n"
        "  · 原发病/基础病(如「主要疾病:obesity」「BMI 32」)—— 那是入组条件;\n"
        "  · 用药、治疗方案、检查项目;\n"
        "  · 症状描述本身。\n"
        "**算数的**:一个明确下过的疾病判断,如「确诊为多囊卵巢综合征」「考虑库欣综合征」。\n\n"
        "写了就照抄那个诊断名;**没写就返回 null**。**绝对不要根据症状推断。**\n"
        '输出纯 JSON:{"diagnosis_stated_in_text": "诊断名或 null", '
        '"quote": "支持它的原文片段(必须是原文里的连续片段),没有则 null"}\n\n'
        f"---\n{text}\n---")
    got = _parse_json(_complete(prompt, model, cfg, offline=offline)[0])
    led = ledger or ProvenanceLedger(case_id)
    latent: dict = {}
    dx = got.get("diagnosis_stated_in_text")
    quote = got.get("quote")
    # Checks in code, not only in the prompt: reject the base disease, a quote that is not
    # verbatim, or a diagnosis without a quote.
    _base = (base_disease or "").strip().lower()
    _dx = (dx or "").strip()
    _reject = None
    if _dx and _base and _dx.lower() == _base:
        _reject = f"转录到的「{_dx}」就是原发病(A 类 `disease`),不是鉴别诊断"
    elif _dx and quote and not _quote_is_verbatim(quote, text):
        _reject = f"引文在原文里找不到(不是转录是生成):{str(quote)[:50]!r}"
    elif _dx and not quote:
        _reject = "给了诊断却给不出原文依据"
    if _dx and not _reject:
        latent["ddx_diagnosis"] = _dx
        led.stamp("ddx_diagnosis", "source_text", by=f"llm-transcribe:{model}",
                  note=f"原文依据:{str(quote)[:60]}")
    else:
        led.stamp("ddx_diagnosis", "absent",
                  note=_reject or "文本未明写诊断 ⇒ 不推断")
    # ---- fields opened by the input kind: the same verbatim check ----
    _rare = rarely_in_text(input_kind)
    _opened = RARELY_IN_TEXT - _rare
    if "ddx_outcome_label" in _opened:
        p2 = (
            "只回答:下面这段病例文本里,**原文有没有明确写出这位患者最终的转归**"
            "(目标事件到底发生了没有)?\n\n"
            "写了就返回 `event_occurred` 或 `event_not_occurred`;"
            "**没明确写就返回 null**。**绝对不要根据趋势或数值自己推断。**\n"
            '输出纯 JSON:{"outcome_label": "event_occurred|event_not_occurred|null", '
            '"quote": "支持它的原文片段(必须是原文里的连续片段),没有则 null"}\n\n'
            f"---\n{text}\n---")
        g2 = _parse_json(_complete(p2, model, cfg, offline=offline)[0])
        _ol = (g2.get("outcome_label") or "").strip()
        _q2 = g2.get("quote")
        if _ol not in ("event_occurred", "event_not_occurred"):
            # outside the binary enum => transcription failed; no normalization
            led.stamp("ddx_outcome_label", "absent",
                      note=f"原文未明写转归(模型给 {_ol or 'null'!r})⇒ 不推断")
        elif not _quote_is_verbatim(_q2, text):
            led.stamp("ddx_outcome_label", "absent",
                      note=f"引文在原文里找不到(不是转录是生成):{str(_q2)[:50]!r}")
        else:
            latent["ddx_outcome_label"] = _ol
            led.stamp("ddx_outcome_label", "source_text", by=f"llm-transcribe:{model}",
                      note=f"原文依据:{str(_q2)[:60]}")
    _stamp_remaining_gold(led, case_id, _rare, _opened, input_kind)
    return latent, led


def _stamp_remaining_gold(led: ProvenanceLedger, case_id: str,
                          rare: frozenset[str], opened: frozenset[str],
                          input_kind: str) -> None:
    """Stamp every gold field not handled on the transcription path as `absent`, then raise
    if any field is left unstamped (a field added to `RETRO_TRANSCRIBABLE` without a
    transcription branch).
    """
    for f in GOLD_FIELDS:
        if f == "ddx_diagnosis" or f in opened:
            continue
        led.stamp(f, "absent",
                  note=f"rarely_in_text({input_kind}):文本里一般没有,不许由推理补 ⇒ 弃掉"
                  if f in rare else "本路只转录诊断名")
    _unstamped = [f for f in GOLD_FIELDS if f not in led.stamps]
    if _unstamped:
        raise ExtractionError(
            f"{case_id}: gold field(s) {_unstamped} left with no provenance stamp on the way out — "
            f"most likely a field was added to `RETRO_TRANSCRIBABLE` without wiring a transcription "
            f"branch for it. No stamp is not the same as `absent`; must never be silently filled in")


def fill_gold_llm(case_id: str, text: str, model: str = "gemini-3.1-pro",
                  cfg: dict | None = None, ledger: ProvenanceLedger | None = None,
                  offline: bool = False
                  ) -> tuple[dict, ProvenanceLedger]:
    """B-3: LLM ruling, an opt-in exception to the rule that gold is never handed to a model.

    Every field is stamped `llm` with the model name, and `ProvenanceLedger.summary()`
    reports `gold_llm_fields` separately. Never average this path with
    `source_text`/`clinician`.
    """
    import yaml

    cfg = cfg or _cached_yaml(ROOT / "config.yaml")
    # Only primitives are asked. `ddx_urgency` and `ddx_clinician_warranted` are tied by two dual
    # conditions in `wq.py` (urgency == RED_FLAG_URGENCY <=> red_flag_present; urgency ==
    # NO_ACTION_URGENCY <=> not clinician_action_warranted), so `clinician_warranted` is derived
    # from the diagnosis and the model only picks a middle urgency tier (`MID_URGENCY`).
    from .wq import MID_URGENCY, NO_ACTION_URGENCY, RED_FLAG_URGENCY
    prompt = (
        "你是内分泌科医生。根据下面这段病例描述给出裁定。**不确定就填 null**,"
        "不要为了填满而猜。\n\n"
        '输出纯 JSON:{"ddx_diagnosis":"最可能诊断或 null",'
        '"ddx_aliases":["别名"],"ddx_tests":["应索取的检查"],'
        '"ddx_specialty":["应转诊科室"],'
        f'"ddx_urgency_mid":"{MID_URGENCY[0]}|{MID_URGENCY[1]}",'
        '"ddx_red_flag":true/false,'
        '"ddx_insufficient":true/false}\n\n'
        f"⚠️ `ddx_urgency_mid` **只填中间那两档**({MID_URGENCY[0]}/{MID_URGENCY[1]}):"
        f"急症({RED_FLAG_URGENCY})与无需临床动作({NO_ACTION_URGENCY})两端"
        "由 `ddx_red_flag` 与病理归并方式**推导**,不由你填。\n\n"
        f"---\n{text}\n---")
    got = _parse_json(_complete(prompt, model, cfg, offline=offline)[0])
    led = ledger or ProvenanceLedger(case_id)
    latent: dict = {}
    for f in ("ddx_diagnosis", "ddx_aliases", "ddx_tests", "ddx_specialty",
              "ddx_red_flag", "ddx_insufficient"):
        v = got.get(f)
        if v is None or v == [] or v == "":
            led.stamp(f, "absent", note="模型自称不确定 ⇒ 弃掉,不补默认值")
            continue
        latent[f] = v
        led.stamp(f, "llm", by=model, note="LLM 判定,非转录")
    # `clinician_action_warranted`: same rule and constant as `wq.derive_clinician_warranted`.
    from .wq import NO_UNIFIED_PATHOLOGY
    _dx = str(latent.get("ddx_diagnosis") or "")
    _independent = bool(_dx) and NO_UNIFIED_PATHOLOGY in _dx
    latent["ddx_clinician_warranted"] = not _independent
    led.stamp("ddx_clinician_warranted", "derived",
              by="wq.derive_clinician_warranted(同规则)",
              note=("rule=join_gold=" + ("independent(无统一病理)⇒ 无需临床动作"
                                         if _independent else "unified(存在可抓的病)⇒ 需要临床动作")
                    + " —— 与 wq.derive_clinician_warranted 同一条规则,不问模型"))
    # `urgency`: the three branches of `wq.derive_urgency`.
    _mid = got.get("ddx_urgency_mid")
    if latent.get("ddx_red_flag") is True:
        latent["ddx_urgency"] = RED_FLAG_URGENCY
        _u_note = f"rule=red_flag_present ⇒ {RED_FLAG_URGENCY}(定义即急症)"
    elif latent["ddx_clinician_warranted"] is False:
        latent["ddx_urgency"] = NO_ACTION_URGENCY
        _u_note = f"rule=无需临床动作 ⇒ {NO_ACTION_URGENCY}"
    elif _mid in MID_URGENCY:
        latent["ddx_urgency"] = _mid
        _u_note = ("primitive=中档轻重由病种决定(两条双条件都不适用);"
                   "模型只判了这一档,两端仍是推出来的")
    else:
        _u_note = None
    if _u_note:
        _derived = (latent.get("ddx_red_flag") is True
                    or latent["ddx_clinician_warranted"] is False)
        led.stamp("ddx_urgency", "derived" if _derived else "llm",
                  by=("wq.derive_urgency(同规则)" if _derived else model),
                  note=_u_note)
    else:
        led.stamp("ddx_urgency", "absent",
                  note=f"两端双条件都不适用,而模型没给出中档值({_mid!r})⇒ 弃掉,不猜")
    # ---- check both dual conditions before emitting ----
    _u, _rf, _cw = (latent.get("ddx_urgency"), latent.get("ddx_red_flag"),
                    latent["ddx_clinician_warranted"])
    if _u is not None:
        if (_u == RED_FLAG_URGENCY) != bool(_rf):
            raise ExtractionError(
                f"{case_id}: urgency={_u!r} and red_flag={_rf!r} violate the dual condition "
                f"({RED_FLAG_URGENCY} ⟺ red_flag) — refusing to emit self-contradictory gold")
        if (_u == NO_ACTION_URGENCY) != (not _cw):
            raise ExtractionError(
                f"{case_id}: urgency={_u!r} and clinician_warranted={_cw!r} violate the dual condition "
                f"({NO_ACTION_URGENCY} ⟺ no clinical action needed) — refusing to emit self-contradictory gold")
    for f in GOLD_FIELDS:
        if f not in led.stamps:
            led.stamp(f, "absent", note="本路不产出该字段")
    return latent, led


def fill_gold_clinician(case_id: str, annotations: dict,
                        annotator: str,
                        ledger: ProvenanceLedger | None = None
                        ) -> tuple[dict, ProvenanceLedger]:
    """B-4: gold from a clinician's annotations (`{field: value}`); `annotator` is required.
    Fields not annotated are stamped `absent`.
    """
    if not annotator:
        raise ProvenanceError("the clinician path must state an annotator (annotator id)")
    led = ledger or ProvenanceLedger(case_id)
    latent: dict = {}
    unknown = sorted(set(annotations) - set(GOLD_FIELDS))
    if unknown:
        raise ProvenanceError(
            f"{case_id}: the annotations contain non-gold field(s) {unknown} — "
            f"a clinician only fills in category B; patient facts go through extraction, "
            f"generation knobs through sampling")
    for f in GOLD_FIELDS:
        if f in annotations and annotations[f] not in (None, "", []):
            latent[f] = annotations[f]
            led.stamp(f, "clinician", by=annotator)
        else:
            led.stamp(f, "absent", note="医生未标注该字段")
    return latent, led


GOLD_PATHS = {
    "absent": fill_gold_absent,
    "text": fill_gold_from_text,
    "llm": fill_gold_llm,
    "clinician": fill_gold_clinician,
}


# ==========================================================================
# The hard backstop: what gets extracted must pass the kernel's `validate_premise`
# ==========================================================================

def validate_extracted(ec: ExtractedCase, cfg: dict | None = None
                       ) -> tuple[bool, list[str]]:
    """Run the extraction result through the kernel's own premise validation
    (`build.premise_spec` -> `latent.LatentPremise` -> `latent.validate_premise`).
    """
    from latent import LatentPremise, validate_premise            # kernel

    from .build import premise_spec
    from .job import CaseSpec

    cs = CaseSpec(case_id=ec.case_id, raw=dict(ec.raw), latent=dict(ec.latent))
    try:
        spec = premise_spec(cs, T=int(ec.latent.get("index_time_T") or 84))
    except Exception as e:                                        # noqa: BLE001
        return False, [f"premise_spec 构不出来:{type(e).__name__} {e}"]
    try:
        ok, v = validate_premise(LatentPremise(**spec))
    except Exception as e:                                        # noqa: BLE001
        return False, [f"LatentPremise 构不出来:{type(e).__name__} {e}"]
    return ok, list(v)


def text_to_case(text: str, case_id: str, gold_path: str = "absent",
                 model: str = "gemini-3.1-pro",
                 annotations: dict | None = None, annotator: str | None = None,
                 cfg: dict | None = None,
                 input_kind: str = "prospective",
                 offline: bool = False) -> ExtractedCase:
    """End to end: text -> a case for job.yaml.

    `gold_path` is one of `{absent, text, llm, clinician}` (see `GOLD_PATHS`) and has no
    default. `input_kind` only matters for `gold_path="text"`.
    """
    if gold_path not in GOLD_PATHS:
        raise ExtractionError(
            f"unknown gold_path {gold_path!r}; only {sorted(GOLD_PATHS)} are valid — "
            f"where gold comes from must be stated explicitly, there is no default")
    ec = extract_patient_facts(text, case_id, model=model, cfg=cfg, offline=offline)
    led = ec.ledger
    if gold_path == "absent":
        gold, led = fill_gold_absent(case_id, led)
    elif gold_path == "text":
        gold, led = fill_gold_from_text(case_id, text, model=model, cfg=cfg,
                                        ledger=led, offline=offline,
                                        base_disease=ec.raw.get("disease"),
                                        input_kind=input_kind)
    elif gold_path == "llm":
        gold, led = fill_gold_llm(case_id, text, model=model, cfg=cfg, ledger=led,
                                  offline=offline)
    else:
        gold, led = fill_gold_clinician(case_id, annotations or {},
                                        annotator or "", ledger=led)
    _sym = ec.raw.get("symptoms") or []
    knobs = sample_knobs(case_id, n_symptoms=len(_sym), ledger=led,
                         symptom_days=[s.get("day") for s in _sym
                                       if isinstance(s, dict)],
                         drug=ec.raw.get("drug"))
    ec.latent = {**knobs, **gold}
    ec.ledger = led
    ok, viol = validate_extracted(ec, cfg=cfg)
    if not ok:
        ec.warnings.append(f"validate_premise 不过:{viol}")
    inv = invented_fields(ec, cfg=cfg)
    if inv:
        ec.warnings.append(
            f"下游会替这些字段编一个值(原文里没有):{inv} —— "
            f"生成出来的病人在这几项上不来自这份文本。补标注或退回文本。")
    return ec


#: `raw` fields downstream fills with a default when absent, found by running the production
#: functions on an empty `raw` rather than by copying `build.py`'s defaults.
def _defaulted_by_downstream(cfg: dict | None = None) -> dict:
    """Fields that get a value even from an empty `raw`, i.e. downstream has a default for them."""
    from .build import premise_spec
    from .job import CaseSpec
    try:
        spec = premise_spec(CaseSpec(case_id="__probe__", raw={}, latent={}), T=84)
    except Exception:                                             # noqa: BLE001
        return {}
    pb = dict(spec.get("patient_basics") or {})
    out = {k: v for k, v in pb.items() if v not in (None, [], {}, "")}
    reg = dict(pb.get("regimen") or {})
    for k in ("drug", "dose_steps"):
        if reg.get(k) not in (None, [], ""):
            out[k] = reg[k]
    dev = ((spec.get("device_signals") or {}).get("devices") or [])
    if dev:
        out["devices"] = dev
    # Weight anchors are defaulted in `build.original_facts`, not `premise_spec`, so it is probed too.
    from .build import original_facts
    try:
        of = original_facts(CaseSpec(case_id="__probe__", raw={}, latent={}), 84)
        for k in ("start_weight", "nadir_weight"):
            if of.get(k) not in (None, "", []):
                out[k] = of[k]
        # `start_weight` appears as `anchor_values.weight.value`.
        _aw = ((of.get("anchor_values") or {}).get("weight") or {}).get("value")
        if _aw not in (None, "", []):
            out["start_weight"] = _aw
    except Exception:                                             # noqa: BLE001
        pass
    return out


def invented_fields(ec: ExtractedCase, cfg: dict | None = None) -> dict:
    """Fields absent from the text that downstream would fill with a default value.

    Defaults such as `raw.get("start_weight", 98.0)` suit hand-written job files, but for an
    extracted case they invent patient facts. This makes the gap explicit; the caller decides
    whether to annotate, return the text, or accept and record it.
    """
    d = _defaulted_by_downstream(cfg)
    return {k: v for k, v in d.items()
            if k in PATIENT_FACTS and k not in ec.raw}
