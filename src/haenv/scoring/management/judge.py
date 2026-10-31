"""One anonymous atomic judgment per request, using shared semantic transport types."""
from __future__ import annotations

from haenv.i18n.runtime import localized_judge_version
from haenv.i18n.text import text_of as _rt
from copy import deepcopy

import json
import re

from haenv.scoring.management.rubric import VISIBLE_PREFIXES
from haenv.scoring.management.rubric import canonical
from haenv.scoring.management.rubric import pointer
from haenv.scoring.management.rubric import sha256
from haenv.scoring.semantic.judge import JudgeReply
from haenv.scoring.semantic.judge import JudgeRequest
from haenv.scoring.semantic.judge import JudgeTask
from haenv.scoring.semantic.judge import _unique_object
from haenv.scoring.semantic.judge import reply_from_response

LABELS = ("satisfied", "unmet", "contradicted")
PROTOCOL = "management-atomic-full-json-v3"


class EmptyJudgeResponse(ValueError):
    """An empty management judge answer, distinct from malformed nonempty output."""

    def __init__(self, raw=None, usage=None):
        super().__init__("Judge returned no answer content")
        self.raw, self.usage = raw, usage


def transport_reply(response, usage):
    """Tag retryable emptiness; preserve the shared normal-response checks."""
    choices = response.get("choices")
    if choices is None or choices == []:
        raise EmptyJudgeResponse(usage=usage)
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message")
        if choice.get("finish_reason") not in ("length", "content_filter") and (message is None or isinstance(message, dict)):
            raw = (message or {}).get("content")
            if raw is None or isinstance(raw, str) and not raw.strip():
                raise EmptyJudgeResponse(raw, usage)
    return reply_from_response(response, usage)


def answer_json(answer):
    """String controls remain supported; original solver JSON is never narrowed."""
    if isinstance(answer, str):
        return {"management_plan": answer}
    if not isinstance(answer, dict):
        raise ValueError("Management answer must be a JSON object")
    return answer


def answer_values(answer):
    """Preserve literal field strings (including newlines), for verified quotations."""
    if isinstance(answer, dict):
        return [text for value in answer.values() for text in answer_values(value)]
    if isinstance(answer, list):
        return [text for value in answer for text in answer_values(value)]
    return [answer] if isinstance(answer, str) else []


def empty_answer(answer):
    """A collected empty plan is free only when every answer field is empty."""
    if isinstance(answer, dict):
        return all(empty_answer(value) for value in answer.values())
    if isinstance(answer, list):
        return all(empty_answer(value) for value in answer)
    return answer is None or isinstance(answer, str) and not answer.strip()


def make_task(rubric, snapshot, answer, policy, *, run_id: str, items=None) -> JudgeTask:
    if rubric["version"] in {"management-rubric-v6", "management-rubric-v6.1", "management-rubric-v6.2", "management-rubric-v7"}:
        return _v6_task(rubric, answer, policy, run_id=run_id, items=items)
    selected = [i for i in rubric["items"] if not i.get("core_decision")
                and (items is None or i["id"] in set(items))]
    if not selected:
        raise ValueError("Empty management judgment task")
    refs = list(dict.fromkeys(ref for item in selected for ref in item["evidence_fact_ids"]))
    if any(not ref.startswith(VISIBLE_PREFIXES) for ref in refs):
        raise ValueError("A private source was selected for the judge")
    facts = {ref: pointer(snapshot, ref) for ref in refs}
    criteria = [{"id": i["id"], "polarity": i["polarity"], "criterion": i["text"],
                 "relevant_fact_ids": i["evidence_fact_ids"]} for i in selected]
    answer_text = canonical(answer_json(answer))
    prompt = (
        _rt('judge.scoring.management.judge.make_task.1', s0=canonical(criteria), s1=canonical(facts), s2=answer_text)
    )
    judge = policy["judge"]
    effort = judge.get("reasoning_effort") or canonical({"backend": judge["backend"],
                                                       "sampling": judge.get("sampling", {})})
    version = f"{rubric['version']}:{PROTOCOL}:{rubric['facts_sha256']}"
    # Preserve the historical Luna request identity exactly. Other sampling
    # conditions must never replay a receipt bought under different settings.
    if rubric["version"] == "management-rubric-v7":
        version += ":" + sha256(rubric["case_facts"])
    if judge.get("sampling") or judge["max_tokens"] != 12000:
        version += ":" + sha256({"sampling": judge.get("sampling"), "max_tokens": judge["max_tokens"]})
    return JudgeTask(run_id=run_id, item_ids=tuple(i["id"] for i in selected), prompt=prompt,
                     answer_text=answer_text, judge_model=policy["judge"]["model_id"],
                     rubric_version=localized_judge_version(version, policy), reasoning_effort=effort)


def _v6_task(rubric, answer, policy, *, run_id, items=None):
    selected = [i for i in rubric["items"] if items is None or i["id"] in items]
    if not selected:
        raise ValueError("Empty management judgment task")
    criteria = [{"id": i["id"], "polarity": i["polarity"], "criterion": i["text"]} for i in selected]
    facts = rubric["case_facts"]
    answer_text = canonical(answer_json(answer))
    writing_instruction = _rt('judge.scoring.management.judge._v6_task.1')
    if rubric["version"] in {"management-rubric-v6.1", "management-rubric-v6.2", "management-rubric-v7"}:
        writing_instruction = (
            _rt('judge.scoring.management.judge._v6_task.2'))
    quote_instruction = (_rt('judge.scoring.management.judge._v6_task.3')
                         if rubric["version"] in {"management-rubric-v6.2", "management-rubric-v7"} else "")
    if rubric["version"] == "management-rubric-v7":
        from haenv import i18n
        from haenv.i18n.text import templates
        # These JSON blobs are original archived data, not language assets.
        prompt = templates(i18n.current())["judge.scoring.management.judge._v6_task.4"].format(
            s0=writing_instruction, s1=quote_instruction, s2=canonical(criteria), s3=canonical(facts), s4=answer_text)
    else:
        prompt = _rt('judge.scoring.management.judge._v6_task.4', s0=writing_instruction,
                     s1=quote_instruction, s2=canonical(criteria), s3=canonical(facts), s4=answer_text)
    if rubric["version"] == "management-rubric-v7":
        prompt = _rt("judge.management.v7.instructions") + "\n" + prompt
    judge = policy["judge"]
    version = f"{rubric['version']}:{PROTOCOL}:{rubric['facts_sha256']}"
    if rubric["version"] == "management-rubric-v7":
        version += ":" + sha256(rubric["case_facts"])
    if judge.get("sampling") or judge["max_tokens"] != 12000:
        version += ":" + sha256({"sampling": judge.get("sampling"), "max_tokens": judge["max_tokens"]})
    effort = judge.get("reasoning_effort") or canonical({"backend": judge["backend"],
                                                       "sampling": judge.get("sampling", {})})
    return JudgeTask(run_id=run_id, item_ids=tuple(i["id"] for i in selected), prompt=prompt,
                     answer_text=answer_text, judge_model=policy["judge"]["model_id"],
                     rubric_version=localized_judge_version(version, policy), reasoning_effort=effort)


def request_for(task: JudgeTask, *, index=1) -> JudgeRequest:
    if index not in (1, 2):
        raise ValueError("Management requests allow only the first attempt and one retry")
    atom = {"type": "object", "additionalProperties": False,
            "required": ["label", "reason", "evidence"], "properties": {
                "label": {"type": "string", "enum": list(LABELS)},
                "reason": {"type": "string"},
                "evidence": {"type": "array", "maxItems": 1, "items": {"type": "string", "maxLength": 200}}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
              "properties": {"verdicts": {"type": "object", "additionalProperties": False,
                  "required": list(task.item_ids), "properties": {key: atom for key in task.item_ids}}}}
    return JudgeRequest(sample_id=f"{PROTOCOL}:{task.fingerprint}:{index}", index=index, prompt=task.prompt,
                        response_format={"type": "json_schema", "json_schema": {
                            "name": "haenv_management_verdicts", "strict": True, "schema": schema}})


def parse_reply(raw: str, items: list[dict], answer, *, ignore_core=False) -> dict:
    doc = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(doc, dict) or set(doc) != {"verdicts"}:
        raise ValueError("Invalid management response root")
    verdicts = doc["verdicts"]
    if ignore_core and isinstance(verdicts, dict):
        core_ids = {i["id"] for i in items if i.get("core_decision")}
        verdicts = {key: value for key, value in verdicts.items() if key not in core_ids}
        items = [i for i in items if not i.get("core_decision")]
    if not isinstance(verdicts, dict) or set(verdicts) != {i["id"] for i in items}:
        raise ValueError("Management rubric item ids differ")
    texts = answer_values(answer_json(answer)) + [canonical(answer_json(answer))]
    for item in items:
        value = verdicts[item["id"]]
        if not isinstance(value, dict) or set(value) != {"label", "reason", "evidence"}:
            raise ValueError("Invalid atomic management response")
        evidence = value["evidence"]
        if (value["label"] not in LABELS or not isinstance(value["reason"], str) or not value["reason"].strip()
                or not isinstance(evidence, list) or len(evidence) > 1
                or any(not isinstance(q, str) or not q or len(q) > 200 for q in evidence)):
            raise ValueError("Invalid management label, reason or verbatim quote")
        needs_quote = (value["label"] == "contradicted"
                       or value["label"] == "satisfied" and not item["stub"]["omitted_item"])
        # An assertion stands only on a quote found in the answer. A quote that cannot be
        # located demotes this one assertion to unmet; the other items of the case still count.
        if needs_quote and not (evidence and quote_found(evidence[0], texts)):
            verdicts[item["id"]] = {**value, "label": "unmet",
                                    "reason": "[quote not found in the answer; assertion not credited] " + value["reason"]}
    return verdicts


_ELLIPSIS = re.compile(r"…+|\.{3,}|⋯+")


def quote_found(quote: str, texts: list[str]) -> bool:
    """Verbatim, or verbatim fragments once whitespace, escaped newlines and the judge's
    ellipses are discounted; each fragment must sit inside one answer field."""
    if any(quote in text for text in texts):
        return True
    norm = lambda s: re.sub(r"\s+", "", s.replace("\\n", "\n"))
    fields = [norm(text) for text in texts]
    fragments = [norm(part) for part in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(len(f) >= 4 and any(f in field for field in fields) for f in fragments)


def _match(plan, phrase):
    """A deliberately narrow offline fact/phrase matcher, not a semantic score proxy."""
    norm = lambda s: re.sub(r"\s+", "", s).casefold()
    wanted = norm(phrase)
    if not wanted:
        return None
    # Retain a real excerpt from the original plan, including whitespace.
    pattern = r"\s*".join(re.escape(c) for c in phrase if not c.isspace())
    hit = re.search(pattern, plan, re.IGNORECASE)
    if hit is None:
        return None
    if norm(hit.group()) != wanted:
        return None
    return hit.group()[:200]


def _diagnostic_match(item, answer, matches):
    """Small explicit syntax probes; no medical inference or model-specific handling."""
    spec = item["stub"]
    structured = spec.get("structured")
    if structured and structured["field"] in answer:
        value = answer[structured["field"]]
        if isinstance(value, str):
            return ("satisfied" if value in structured["accepted"] else "contradicted", value[:200])
    diagnostic = spec.get("diagnostic", {})
    texts = answer_values(answer)
    kind = diagnostic.get("kind")
    if (kind == "regex" and (not diagnostic.get("decision_item") or matches.get(diagnostic["decision_item"]))
            and all(any(re.search(pattern, text, re.I) for text in texts)
                    for pattern in diagnostic["patterns"])
            and (not diagnostic.get("linked_patterns") or any(
                all(re.search(pattern, text, re.I) for pattern in diagnostic["linked_patterns"]) for text in texts))):
        quote = next(text for text in texts if re.search(diagnostic["patterns"][0], text, re.I))
        return "satisfied", quote[:200]
    if kind == "precision":
        # Exact offline control phrases only. No test counting or clinical inference.
        for phrase in diagnostic["violations"]:
            quote = next((q for text in texts if (q := _match(text, phrase))), None)
            if quote:
                return "contradicted", quote
        for phrase in diagnostic["decision_claims"]:
            quote = next((q for text in texts if (q := _match(text, phrase))), None)
            if quote:
                return "satisfied", quote
        structured = diagnostic.get("structured")
        if structured and answer.get(structured["field"]) in structured["choices"]:
            return "satisfied", answer[structured["field"]][:200]
        return "unmet", None
    return None


def stub_reply(items: list[dict], answer) -> JudgeReply:
    """Deterministic offline plumbing control. Never substituted for a real judge."""
    answer = answer_json(answer)
    texts = answer_values(answer)
    match = lambda phrase: next((q for text in texts if (q := _match(text, phrase)) is not None), None)
    verdicts = {}
    # Match the same control renderer as well as archived literal claims. This is
    # still exact phrase matching; rubric criteria and the real judge are unchanged.
    claims = lambda i: list(dict.fromkeys(i["stub"]["claims"] + [
        _control_claim(i, phrase) for phrase in i["stub"]["claims"]]))
    positive_matches = {i["id"]: next((q for phrase in claims(i)
                                      if (q := match(phrase)) is not None), None)
                        for i in items if i["polarity"] == "positive"}
    # Resolve concrete decisions before dependent precision, even in per-item requests.
    diagnostics = {}
    for item in items:
        if item["polarity"] == "positive" and item["stub"].get("diagnostic", {}).get("kind") != "precision":
            diagnostics[item["id"]] = _diagnostic_match(item, answer, positive_matches)
            if diagnostics[item["id"]]:
                positive_matches[item["id"]] = diagnostics[item["id"]][1] if diagnostics[item["id"]][0] == "satisfied" else None
    for item in items:
        spec = item["stub"]
        if item["polarity"] == "positive":
            contradictions = spec.get("contradictions", [])
            contradictions = contradictions + [_control_claim(item, p) for p in contradictions]
            contradiction = next((q for phrase in contradictions
                                  if (q := match(phrase)) is not None), None)
            diagnostic = diagnostics.get(item["id"]) or _diagnostic_match(item, answer, positive_matches)
            if contradiction is not None:
                label, quote = "contradicted", contradiction
            elif diagnostic is not None:
                label, quote = diagnostic
            else:
                quote = positive_matches[item["id"]]
                label = "satisfied" if quote is not None else "unmet"
        else:
            if spec.get("diagnostic", {}).get("kind") == "v6_safety":
                diagnostic = spec["diagnostic"]
                quote = next((q for phrase in diagnostic["violations"] if (q := match(phrase))), None)
                missing = any(not any(match(p) for p in group.split("|")) for group in diagnostic["required"])
                label = "satisfied" if quote or missing else "unmet"
                verdicts[item["id"]] = {"label": label, "reason": _rt('judge.scoring.management.judge.stub_reply.1'),
                                         "evidence": [quote] if quote else []}
                continue
            quote = match(spec["forbidden"]) if spec["forbidden"] else None
            if quote is not None:
                label = "satisfied"
            elif spec["omitted_item"]:
                # For per-item tasks the positive paired criterion must be supplied too.
                paired = next((i for i in items if i["id"] == spec["omitted_item"]), None)
                if paired is None:
                    raise ValueError("Offline omission matcher requires the paired positive criterion")
                label = "unmet" if positive_matches[spec["omitted_item"]] is not None else "satisfied"
            else:
                label = "unmet"
        verdicts[item["id"]] = {"label": label, "reason": _rt('judge.scoring.management.judge.stub_reply.2'),
                                "evidence": [quote] if quote is not None else []}
    return JudgeReply(raw=canonical({"verdicts": verdicts}), cached=False, usage={"offline": True, "cost": 0})


def _control_json(text):
    """Locate a saved fact embedded in an archived control claim."""
    pos = 0
    while pos < len(text):
        if text[pos] in '[{"':
            try:
                value, end = json.JSONDecoder().raw_decode(text[pos:])
            except ValueError:
                pass
            else:
                yield pos, pos + end, value
                pos += end
                continue
        pos += 1


def _clinical_text(value):
    """Render saved facts as prose, with units, rather than copying JSON."""
    names = {
        "heart_rate_bpm": _rt('judge.scoring.management.judge._clinical_text.1'), "resting_hr_bpm": _rt('judge.scoring.management.judge._clinical_text.2'), "respiratory_rate_per_min": _rt('judge.scoring.management.judge._clinical_text.3'),
        "systolic_bp_mmHg": _rt('judge.scoring.management.judge._clinical_text.4'), "diastolic_bp_mmHg": _rt('judge.scoring.management.judge._clinical_text.5'), "systolic_mmHg": _rt('judge.scoring.management.judge._clinical_text.6'),
        "diastolic_mmHg": _rt('judge.scoring.management.judge._clinical_text.7'), "temperature_c": _rt('judge.scoring.management.judge._clinical_text.8'), "spo2_pct": _rt('judge.scoring.management.judge._clinical_text.9'),
        "night_spo2_min_pct": _rt('judge.scoring.management.judge._clinical_text.10'), "potassium_mmol_L": _rt('judge.scoring.management.judge._clinical_text.11'),
        "creatinine_umol_L": _rt('judge.scoring.management.judge._clinical_text.12'), "glucose_mmol_L": _rt('judge.scoring.management.judge._clinical_text.13'), "min_mmol_L": _rt('judge.scoring.management.judge._clinical_text.14'),
        "dose_mg": _rt('judge.scoring.management.judge._clinical_text.15'), "dose_ug": _rt('judge.scoring.management.judge._clinical_text.16'), "kg": _rt('judge.scoring.management.judge._clinical_text.17'), "irregular_rhythm_alerts": _rt('judge.scoring.management.judge._clinical_text.18'),
        "urgent_symptoms": _rt('judge.scoring.management.judge._clinical_text.19'), "dizziness": _rt('judge.scoring.management.judge._clinical_text.20'), "mild_new_symptom": _rt('judge.scoring.management.judge._clinical_text.21'),
        "severe_headache_or_visual_disturbance": _rt('judge.scoring.management.judge._clinical_text.22'),
        "needed_help_for_low_glucose_last_3_days": _rt('judge.scoring.management.judge._clinical_text.23'),
        "urgent_screen_complete": _rt('judge.scoring.management.judge._clinical_text.24'), "on_oxygen": _rt('judge.scoring.management.judge._clinical_text.25'),
        "test": _rt('judge.scoring.management.judge._clinical_text.26'), "result": _rt('judge.scoring.management.judge._clinical_text.27'), "text": _rt('judge.scoring.management.judge._clinical_text.28'), "method": _rt('judge.scoring.management.judge._clinical_text.29'),
        "valid_from": _rt('judge.scoring.management.judge._clinical_text.30'), "valid_to": _rt('judge.scoring.management.judge._clinical_text.31'), "comparison_reference_method": _rt('judge.scoring.management.judge._clinical_text.32'),
        "target_spo2": _rt('judge.scoring.management.judge._clinical_text.33'), "clinical_scale": _rt('judge.scoring.management.judge._clinical_text.34'),
        "external_assistance": _rt('judge.scoring.management.judge._clinical_text.35'), "unable_to_self_treat": _rt('judge.scoring.management.judge._clinical_text.36'),
        "minutes_below_3_9": _rt('judge.scoring.management.judge._clinical_text.37'), "sleep_h": _rt('judge.scoring.management.judge._clinical_text.38'),
        "skin_temp_delta_c": _rt('judge.scoring.management.judge._clinical_text.39'), "workout_max_hr_bpm": _rt('judge.scoring.management.judge._clinical_text.40'),
        "condition": _rt('judge.scoring.management.judge._clinical_text.41'), "conclusion": _rt('judge.scoring.management.judge._clinical_text.42'), "history": _rt('judge.scoring.management.judge._clinical_text.43'), "symptom": _rt('judge.scoring.management.judge._clinical_text.44'),
        "scope": _rt('judge.scoring.management.judge._clinical_text.45'), "unit": _rt('judge.scoring.management.judge._clinical_text.46'), "value": _rt('judge.scoring.management.judge._clinical_text.47'), "details": _rt('judge.scoring.management.judge._clinical_text.48'),
        "note": _rt('judge.scoring.management.judge._clinical_text.49'), "time": _rt('judge.scoring.management.judge._clinical_text.50'), "hrv_ms": _rt('judge.scoring.management.judge._clinical_text.51'), "steps": _rt('judge.scoring.management.judge._clinical_text.52'),
    }
    units = {"potassium_mmol_L": " mmol/L", "glucose_mmol_L": " mmol/L", "min_mmol_L": " mmol/L",
             "creatinine_umol_L": " μmol/L", "dose_mg": " mg", "dose_ug": " μg", "kg": " kg",
             "heart_rate_bpm": _rt('judge.scoring.management.judge._clinical_text.53'), "resting_hr_bpm": _rt('judge.scoring.management.judge._clinical_text.54'), "respiratory_rate_per_min": _rt('judge.scoring.management.judge._clinical_text.55'),
             "temperature_c": "℃", "spo2_pct": "%", "night_spo2_min_pct": "%",
             "systolic_bp_mmHg": " mmHg", "diastolic_bp_mmHg": " mmHg",
             "systolic_mmHg": " mmHg", "diastolic_mmHg": " mmHg"}
    enums = {"hypercapnic_respiratory_failure": _rt('judge.scoring.management.judge._clinical_text.56'), "atrial_fibrillation": _rt('judge.scoring.management.judge._clinical_text.57'),
             "heart_failure": _rt('judge.scoring.management.judge._clinical_text.58'), "creatinine": _rt('judge.scoring.management.judge._clinical_text.59'), "breathless_at_rest": _rt('judge.scoring.management.judge._clinical_text.60'),
             "chest_pain_or_pressure": _rt('judge.scoring.management.judge._clinical_text.61'), "dose_related": _rt('judge.scoring.management.judge._clinical_text.62'),
             "unattributed": _rt('judge.scoring.management.judge._clinical_text.63')}
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            if key in {"source", "system_news2", "finding_id", "synthetic", "instrument", "input_stage", "missing_handling"}:
                continue
            if key in {"day", "night_of_day", "onset_day"}:
                parts.append(_rt('judge.scoring.management.judge._clinical_text.64', s0=val))
            elif key == "entries":
                parts.append(_clinical_text(val))
            else:
                parts.append(f"{names.get(key, key)}{_clinical_text(val)}{units.get(key, '')}")
        return _rt('judge.punctuation.judge.1').join(parts)
    if isinstance(value, list):
        return _rt('judge.punctuation.judge.2').join(_clinical_text(v) for v in value) if value else _rt('judge.scoring.management.judge._clinical_text.65')
    if isinstance(value, bool):
        return _rt('judge.scoring.management.judge._clinical_text.66') if value else _rt('judge.scoring.management.judge._clinical_text.67')
    if value is None:
        return _rt('judge.scoring.management.judge._clinical_text.68')
    return enums.get(str(value), str(value))


def _clinical_sorted(value):
    """Render facts in the historical canonical-JSON key order, without a JSON read-back."""
    def ordered(obj):
        if isinstance(obj, dict):
            return {k: ordered(obj[k]) for k in sorted(obj)}
        if isinstance(obj, list):
            return [ordered(v) for v in obj]
        return obj
    return _clinical_text(ordered(value))


def _human_claim(text):
    parts, cursor = [], 0
    for start, end, value in _control_json(text):
        if start < cursor:
            continue
        parts.extend((text[cursor:start], _clinical_text(value)))
        cursor = end
    return "".join(parts) + text[cursor:]


def _control_claim(item, claim):
    """One readable causal sentence per existing item; no new criterion."""
    key = item["id"]
    text = _human_claim(claim)
    if key == "protocol.matched":
        # Quote the condition, never the rule's alternative actions. Only the
        # selected decision appears, including in the intentionally wrong control.
        text = re.sub(r"→.*?；本次依据", _rt('judge.archive.1'), text)
    elif key == "management.target":
        unit = "μg" if "左甲状腺素" in text else "mg"
        text = re.sub(r"当前剂量([\d.]+)。(?!\d)", _rt('judge.archive.2', s0=unit), text)
    elif key.startswith("history.") and key.split(".")[-1].isdigit() and not key.startswith("history.notice."):
        fact = next((v for _, _, v in _control_json(claim) if isinstance(v, dict)), {})
        links = {
            "hypercapnic_respiratory_failure": _rt('judge.archive.3'),
            "atrial_fibrillation": _rt('judge.archive.4'),
            "heart_failure": _rt('judge.archive.5'),
        }
        link = links.get(fact.get("condition"), _rt('judge.archive.6'))
        text = text.replace("本次处置需考虑该记录", link)
    elif key == "history.dose":
        rows = next(v for _, _, v in _control_json(claim) if isinstance(v, list))
        unit = "μg" if "左甲状腺素" in text else "mg"
        doses = [r.get("dose_mg", r.get("dose_ug")) for r in rows]
        peak = max(doses)
        text = text.rstrip("。") + _rt('judge.archive.7', s0=peak, s1=unit, s2=doses[-1], s3=unit, s4=peak, s5=unit)
    elif key == "history.event":
        text = text.rstrip("。") + _rt('judge.archive.8')
    elif key == "history.causes":
        text = text.rstrip("。") + _rt('judge.archive.9')
    elif key == "history.help":
        text = text.rstrip("。") + _rt('judge.archive.10')
    elif key == "explanation.history":
        unit = "μg" if "左甲状腺素" in text else "mg"
        text = re.sub(r"涉及([\d.]+)剂量", _rt('judge.archive.11', s0=unit), text)
        text = text.replace("已结合上述事件证据解释本次所选方案", _rt('judge.archive.12'))
    elif key == "protocol.control" and "与原医嘱目标" in claim:
        start, target = claim.split("，与原医嘱目标", 1)
        target, conclusion = target.split("对照，据此", 1)
        target = target.split("本院目标：")[-1].strip()
        reading = re.search(r"为([\d.]+)", start)
        upper = re.search(r"<\s*([\d.]+)", target)
        band = re.search(r"([\d.]+)\s*[–—~～-]\s*([\d.]+)", target)
        if reading and upper:
            value, limit = float(reading[1]), float(upper[1])
            relation = _rt('judge.archive.13') if value < limit else _rt('judge.archive.14') if value > limit else _rt('judge.archive.15')
        elif reading and band:
            value = float(reading[1])
            relation = _rt('judge.archive.16') if value < float(band[1]) else _rt('judge.archive.17') if value > float(band[2]) else _rt('judge.archive.18')
        else:
            raise ValueError("Control reading has no supported saved target comparison")
        text = _rt('judge.archive.19', s0=start, s1=relation, s2=target, s3=conclusion)
    elif key == "protocol.elapsed":
        match = re.search(r"已(\d+)天.*?约([\d.]+)天", text)
        if match:
            elapsed, window = int(match[1]), float(match[2])
            state = _rt('judge.archive.20') if elapsed >= window else _rt('judge.archive.21')
            text = text.replace("时点判断", _rt('judge.archive.22', s0=state, s1=window))
    elif key == "protocol.adherence":
        text = text.replace("，据此支持", _rt('judge.archive.23'))
    elif key == "protocol.symptoms":
        text = text.replace("dose_related", _rt('judge.archive.24'))
        text = text.replace("unattributed", _rt('judge.archive.25'))
    elif key == "followup.efficacy":
        drug = re.search(r"(依那普利|阿托伐他汀|格列齐特|左甲状腺素钠)", text)
        days = re.search(r"约(\d+)天", text)
        if not drug or not days:
            raise ValueError("No saved drug/efficacy interval in control")
        anchor = _rt('judge.archive.26') if "方案调整后" in claim else _rt('judge.archive.27')
        text = _rt('judge.archive.28', s0=drug[1], s1=anchor, s2=days[1])
    elif key.startswith("history.notice."):
        text = text.rstrip("。") + _rt('judge.archive.29')
    return text


def _control_action(key, claim):
    patterns = {"management.disposition": _rt('judge.archive.30'), "management.decision": _rt('judge.archive.31'),
                "management.next_step": _rt('judge.archive.32'), "management.urgency": _rt('judge.archive.33')}
    match = re.search(patterns[key], claim) if key in patterns else None
    return match[1] if match else claim


def _triage_context(v):
    check = v["home_check_today"]
    news = check.get("system_news2")
    text = _rt('judge.archive.34', s0=_clinical_text(check))
    if news:
        top = max(news["parameter_scores"].values())
        text += _rt('judge.archive.35', s0=news['total'], s1=top, s2=news['spo2_scale'])
    return text


def _r14_context(v):
    """Summarize the saved observations that exclude preceding triage rules."""
    parts = []
    bp = v.get("home_bp_log", [])
    if bp:
        last = max(r["day"] for r in bp)
        pairs = _rt('judge.punctuation.judge.3').join(f"{r['systolic_mmHg']}/{r['diastolic_mmHg']}" for r in bp if r["day"] == last)
        parts.append(_rt('judge.archive.36', s0=pairs))
    weights = {r["day"]: r["kg"] for r in v.get("morning_weight", [])}
    if weights:
        last = max(weights)
        if last - 3 in weights:
            parts.append(_rt('judge.archive.37', s0=weights[last] - weights[last - 3]))
    glucose = v.get("cgm_nightly", [])
    if glucose:
        parts.append(_rt('judge.archive.38', s0=len(glucose), s1=min((r['min_mmol_L'] for r in glucose))))
    wearable = v.get("wearable_daily", [])
    if wearable:
        last = max(r["day"] for r in wearable)
        recent = [r for r in wearable if r["day"] >= last - 2]
        alerts = sum(r.get("irregular_rhythm_alerts", 0) for r in recent)
        parts.append(_rt('judge.archive.39', s0=alerts))
        night_oxygen = [r["night_spo2_min_pct"] for r in wearable if r.get("night_spo2_min_pct") is not None]
        if night_oxygen:
            parts.append(_rt('judge.archive.40', s0=min(night_oxygen), s1=v['home_check_today']['spo2_pct']))
        exercise = [r["workout_max_hr_bpm"] for r in wearable if r.get("workout_max_hr_bpm") is not None]
        if exercise:
            parts.append(_rt('judge.archive.41', s0=max(exercise), s1=v['home_check_today']['heart_rate_bpm']))
    return _rt('judge.punctuation.judge.4').join(parts) + _rt('judge.archive.42')


def _dose_history_context(items):
    by_id = {i["id"]: i for i in items}
    claim = by_id["history.dose"]["stub"]["claims"][0]
    doses = next(v for _, _, v in _control_json(claim) if isinstance(v, list))
    labs = next(v for _, _, v in _control_json(by_id["history.event"]["stub"]["claims"][0]) if isinstance(v, list))
    drug = claim.split("旧处方", 1)[0]
    unit = "μg" if "左甲状腺素" in drug else "mg"
    dose = lambda r: r.get("dose_mg", r.get("dose_ug"))
    peak = max(doses, key=dose)
    reduced = doses[-1]
    event_day = int(re.search(r"第(-?\d+)天", by_id["explanation.history"]["stub"]["claims"][0])[1])
    event = next(r for r in labs if r["day"] == event_day)
    causes = _human_claim(by_id["history.causes"]["stub"]["claims"][0]) if "history.causes" in by_id else ""
    return (_rt('judge.archive.43', s0=peak['day'], s1=drug, s2=dose(peak), s3=unit, s4=event_day, s5=_clinical_text({k: v for k, v in event.items() if k != 'day'}), s6=causes, s7=reduced['day'], s8=dose(reduced), s9=unit, s10=_clinical_text(labs[-1]), s11=dose(peak), s12=unit))


def _control_lines(rubric, replacements=()):
    items = [i for i in rubric["items"] if i["polarity"] == "positive"]
    selected = {i["id"]: i["stub"]["claims"][0] for i in items}
    for original, changed in replacements:
        found = [i for i in items if selected[i["id"]] == original]
        if len(found) != 1:
            raise ValueError("Control replacement must identify one selected item claim")
        selected[found[0]["id"]] = changed
    substitutions = {}
    for item in items:
        if item.get("core_decision") and selected[item["id"]] != item["stub"]["claims"][0]:
            changed = _control_action(item["id"], selected[item["id"]])
            substitutions.update({_control_action(item["id"], p): changed for p in item["stub"]["claims"]})
    lines = {i["id"]: _control_claim(i, selected[i["id"]]) for i in items}
    v = rubric["control_inventory"]["visible_facts"]
    if rubric["pack"] == "p2":
        context = _triage_context(v)
        # Explicitly exclude earlier rules for R14, rather than merely listing
        # normal vitals or copying the words 'none of the above'.
        if "符合R14" in lines["protocol.matched"]:
            context += _r14_context(v)
        lines["protocol.matched"] = context + lines["protocol.matched"]
        lines["explanation.disposition"] = context + lines["explanation.disposition"]
    elif rubric["pack"] == "p3" and "history.dose" in lines:
        lines["explanation.history"] = _dose_history_context(items) + lines["explanation.history"]
    if rubric["pack"] == "p3" and "protocol.symptoms" in lines:
        drug = selected["management.decision"].split("的方案", 1)[0]
        prescriptions = [r for r in v["prescription_record"] if drug in r.get("drug", "")]
        recent = max(prescriptions, key=lambda r: r["day"])
        lines["protocol.symptoms"] = (_rt('judge.archive.44', s0=recent['day'], s1=recent['prescription'], s2=_clinical_text(v['symptom_diary'])) + lines["protocol.symptoms"])
    # Replace all action mentions, including causal conclusions. No correct
    # action survives in the wrong control, even when there are multiple accepted tiers.
    for key, line in lines.items():
        for original, changed in sorted(substitutions.items(), key=lambda pair: -len(pair[0])):
            line = line.replace(original, changed)
        lines[key] = line
    if rubric["pack"] == "p2":
        followup = next(i for i in items if i["id"] == "followup.disposition")
        action = _control_action("management.disposition", selected["management.disposition"])
        choices = [p for p in followup["stub"]["claims"] if p.startswith(f"随访安排：落实所选{action}，")]
        if len(choices) != 1:
            raise ValueError("No registered control implementation for selected disposition")
        lines[followup["id"]] = choices[0]
    elif rubric["pack"] == "m2" and "followup.urgency" in lines:
        followup = next(i for i in items if i["id"] == "followup.urgency")
        action = _control_action("management.urgency", selected["management.urgency"])
        choices = [p for p in followup["stub"]["claims"] if p.startswith(f"本次落实所选{action}医疗评估：")]
        if len(choices) != 1:
            raise ValueError("No registered control implementation for selected M2 urgency")
        lines[followup["id"]] = choices[0]
    return lines


def controls(rubric: dict) -> dict:
    if rubric["version"] in {"management-rubric-v6", "management-rubric-v6.1", "management-rubric-v6.2", "management-rubric-v7"}:
        return v6_controls(rubric)
    lines = _control_lines(rubric)
    positive = "\n".join(lines.values())
    forbidden = next((i["stub"]["forbidden"] for i in rubric["items"]
                      if i["polarity"] == "negative" and i["stub"]["forbidden"]), None)
    # The long template is deliberately shared across every case. It does not copy a patient's facts.
    verbose = "\n".join([
        _rt('judge.scoring.management.judge.controls.1') + line for line in (
            _rt('judge.scoring.management.judge.controls.2'),
            _rt('judge.scoring.management.judge.controls.3'),
            _rt('judge.scoring.management.judge.controls.4'),
            _rt('judge.scoring.management.judge.controls.5'),
            _rt('judge.scoring.management.judge.controls.6') )])
    plans = {"gold_facts": positive, "empty": "", "generic": _rt('judge.scoring.management.judge.controls.7'),
             "verbose_unrelated": verbose, "gold_plus_contraindication": positive + "\n" + forbidden if forbidden else None}
    core = [lines[i["id"]] for i in rubric["items"] if i.get("core_decision")]
    basis = [lines[i["id"]] for i in rubric["items"]
             if i["polarity"] == "positive" and not i.get("core_decision")]
    plans["split_correct"] = {"management_plan": "\n".join(core), "reason": "\n".join(basis),
                              "evidence": {i["id"]: lines[i["id"]] for i in rubric["items"]
                                           if i["polarity"] == "positive" and not i.get("core_decision")}}
    plans["readings_only"] = canonical(rubric["control_inventory"])
    for name, replacements in rubric.get("control_replacements", {}).items():
        plans[name] = "\n".join(_control_lines(rubric, replacements).values())
    precision = next((i for i in rubric["items"] if i["type"] == "precision"), None)
    probes = precision["stub"]["diagnostic"]["controls"] if precision else {}
    for name, kind in (("gold_plus_initial_checks", "initial"),
                       ("gold_plus_not_recommended_check", "not_recommended"),
                       ("gold_plus_routine_monitoring", "monitoring")):
        plans[name] = positive + "\n" + probes[kind] if probes.get(kind) else None
    return plans


def v6_controls(rubric):
    """Fixed benign writing variants plus the original adverse plumbing probes."""
    c = rubric["control_context"]
    lines = {i["id"]: i["stub"]["claims"][0] for i in rubric["items"] if i["polarity"] == "positive"}
    positive = "\n".join(lines.values())
    safety = next(i for i in rubric["items"] if i["id"] == "safety")["stub"]["diagnostic"]
    forbidden = safety["violations"][0] if safety["violations"] else None
    plans = {"gold_facts": positive, "empty": "", "generic": _rt('judge.scoring.management.judge.v6_controls.1'),
             "readings_only": canonical(rubric["control_inventory"]),
             "verbose_unrelated": _rt('judge.scoring.management.judge.v6_controls.2'),
             "gold_plus_contraindication": positive + "\n" + forbidden if forbidden else None}
    structured = {"management_plan": lines["management"],
                  "reason": lines["explanation"], "evidence": {k: v for k, v in lines.items() if k != "management"}}
    plans["split_correct"] = structured
    plans["variant_structured"] = structured
    # Exact finite variant claims exercise acceptance without an NLP scoring proxy.
    brief = c.get("brief_explanation", lines["explanation"])
    brief_lines = {**lines, "explanation": brief}
    plans["variant_concise"] = "\n".join(brief_lines.values())
    plans["variant_general_explanation"] = "\n".join(brief_lines.values())
    timed = _rt('judge.scoring.management.judge.v6_controls.3')
    if rubric["version"] in {"management-rubric-v6.1", "management-rubric-v6.2", "management-rubric-v7"}:
        relative = _rt('judge.scoring.management.judge.v6_controls.4', s0=c['followup_project'])
        interval = _rt('judge.scoring.management.judge.v6_controls.5', s0=c['followup_project'])
        if rubric['version'] == 'management-rubric-v6.2':
            # Derive timing variants from this case, never add a universal interval.
            relative = c['followup_variants']['relative']
            interval = c['followup_variants']['range']
        plans["variant_relative_interval"] = "\n".join({**lines, "followup": relative}.values())
        plans["variant_range_interval"] = "\n".join({**lines, "followup": interval}.values())
        followup_item = next(i for i in rubric["items"] if i["id"] == "followup")
        for phrase in (relative, interval):
            if phrase not in followup_item["stub"]["claims"]:
                followup_item["stub"]["claims"].append(phrase)
        # Correct decision, no patient facts, named drug/check, or concrete follow-up project.
        plans["generic_correct"] = _rt('judge.scoring.management.judge.v6_controls.6', s0=c['generic_action'])
    if rubric["version"] == "management-rubric-v6.2":
        timed = c['followup_variants']['fixed']
    plans["variant_fixed_interval"] = "\n".join({**lines, "followup": timed}.values())
    initial = c["initial"]
    plans["variant_initial_check"] = positive + "\n" + rubric["writing_variants"]["reasonable_initial_check"]
    plans["gold_plus_initial_checks"] = positive + ("\n" + initial if initial else "\n" + rubric["writing_variants"]["reasonable_initial_check"])
    prohibited = next((r["control"] for r in rubric["guideline_controls"]["not_recommended"] if r["control"]), None)
    plans["gold_plus_not_recommended_check"] = positive + "\n" + prohibited if prohibited else None
    selected = c["selected"]
    choices = list(c["actions"])
    wrong = next((a for a in choices if a not in c.get("accepted", [selected])), None)
    if wrong is not None:
        plans["gold_wrong_decision"] = positive.replace(c["actions"][selected], c["actions"][wrong])
        rubric["control_decisions"]["gold_wrong_decision"] = {"correct": False, "selected": wrong,
                                                               "accepted": c.get("accepted", [selected])}
    for key in ("explanation", "management"):
        item = next(i for i in rubric["items"] if i["id"] == key)
        if key == "explanation" and brief not in item["stub"]["claims"]:
            item["stub"]["claims"].append(brief)
        if key == "management" and wrong is not None:
            claim = lines[key].replace(c["actions"][selected], c["actions"][wrong])
            if claim not in item["stub"]["claims"]:
                item["stub"]["claims"].append(claim)
    followup = next(i for i in rubric["items"] if i["id"] == "followup")
    if timed not in followup["stub"]["claims"]:
        followup["stub"]["claims"].append(timed)
    if wrong is not None:
        own_choice = lines["followup"].replace(c["actions"][selected], c["actions"][wrong])
        if own_choice not in followup["stub"]["claims"]:
            followup["stub"]["claims"].append(own_choice)
    if rubric.get("frozen_negative_controls"):
        plans.update(deepcopy(rubric["frozen_negative_controls"]))
        for name in rubric["frozen_negative_controls"]:
            if name in rubric.get("frozen_control_decisions", {}):
                rubric["control_decisions"][name] = deepcopy(rubric["frozen_control_decisions"][name])
    return plans
