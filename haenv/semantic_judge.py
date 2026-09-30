"""Strict adaptive two-plus-one consensus for anonymous semantic judgments.

This module performs no network calls by itself. A dispatcher must explicitly
return a fresh response for each sample; durable prior samples may be resumed.
Uncertain/invalid/failed responses never become negative scores or extra votes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .semantic_budget import BudgetExceeded

PROTOCOL = "semantic-consensus-2plus1-v2"
#: Same per-atom consensus; the third vote re-judges only the atoms the first valid
#: pair labelled differently (all atoms if either of the pair is invalid).
LEAN_PROTOCOL = "semantic-consensus-2plus1-disputed-atoms-lean-v4"
LABELS = frozenset({"yes", "no", "uncertain"})


@dataclass(frozen=True)
class JudgeTask:
    run_id: str
    item_ids: tuple[str, ...]
    prompt: str
    answer_text: str
    judge_model: str
    rubric_version: str
    reasoning_effort: str = "high"
    evidence_catalog: dict[str, str] | None = None
    #: Lean layout (semantic-v4): the prompt is head + one line per atom + tail, so
    #: an adaptive third vote can re-judge only the atoms the first pair disputed.
    atom_template: dict | None = None

    def __post_init__(self):
        if not self.run_id or not self.prompt or not self.answer_text:
            raise ValueError("A task requires run id, prompt and answer text")
        if not self.judge_model or not self.rubric_version or not self.reasoning_effort:
            raise ValueError("Judge model, effort and rubric version must be explicit")
        if not self.item_ids or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("Rubric items must be nonempty and unique")
        if any(not isinstance(k, str) or not k for k in self.item_ids):
            raise ValueError("Rubric item ids must be nonempty strings")
        if self.evidence_catalog is not None:
            if not isinstance(self.evidence_catalog, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                or v not in self.answer_text for k, v in self.evidence_catalog.items()):
                raise ValueError("Evidence catalogue must contain verbatim answer excerpts")
        if self.atom_template is not None:
            t = self.atom_template
            if (not isinstance(t, dict) or set(t) != {"head", "atoms", "tail"}
                    or not isinstance(t["atoms"], dict) or list(t["atoms"]) != list(self.item_ids)
                    or self.subset_prompt(self.item_ids) != self.prompt):
                raise ValueError("Atom template must reproduce the full prompt in item order")

    def subset_prompt(self, items) -> str:
        """The full prompt restricted to `items`, all shared context unchanged."""
        t = self.atom_template
        if t is None:
            raise ValueError("Only a lean-layout task can restrict its atoms")
        return t["head"] + "\n".join(t["atoms"][key] for key in self.item_ids if key in set(items)) + t["tail"]

    @property
    def protocol(self) -> str:
        if self.atom_template is not None:
            return LEAN_PROTOCOL
        return PROTOCOL if self.evidence_catalog is None else "semantic-consensus-2plus1-evidence-ids-v3"

    @property
    def fingerprint(self) -> str:
        payload = [self.protocol, self.run_id, self.item_ids, self.prompt,
                   self.answer_text, self.judge_model, self.rubric_version, self.reasoning_effort]
        if self.evidence_catalog is not None:
            payload.append(self.evidence_catalog)
        if self.atom_template is not None:
            payload.append(self.atom_template)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def task_record(task: JudgeTask) -> dict:
    """Serialized task; the lean-only field is omitted when absent, so verbatim-layout
    records keep exactly their original bytes."""
    from dataclasses import asdict
    record = asdict(task)
    if record.get("atom_template") is None:
        record.pop("atom_template", None)
    return record


@dataclass(frozen=True)
class JudgeRequest:
    sample_id: str
    index: int
    prompt: str
    fresh: bool = True
    response_format: dict | None = None


@dataclass(frozen=True)
class JudgeReply:
    raw: str
    cached: bool
    usage: dict | None


def reply_from_response(response: dict, usage) -> JudgeReply:
    """Which provider responses count as a judge answer (scoring logic, so it lives here).

    Exactly one choice, finished normally, with non-empty content; anything else is a
    failed call (`ValueError`), not a vote.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one judge response")
    choice = choices[0]
    if choice.get("finish_reason") in ("length", "content_filter"):
        raise ValueError("Judge response did not finish normally")
    raw = (choice.get("message") or {}).get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Judge returned no answer content")
    return JudgeReply(raw=raw, cached=False, usage=usage)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def parse_verdicts(raw: str, task: JudgeTask, items=None) -> dict:
    """Require exact atom ids, closed labels, reasons and verbatim answer evidence."""
    expected = set(task.item_ids if items is None else items)
    if not isinstance(raw, str):
        raise ValueError("response_not_text")
    try:
        doc = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(doc, dict) or set(doc) != {"verdicts"}:
        raise ValueError("invalid_root_schema")
    verdicts = doc["verdicts"]
    if not isinstance(verdicts, dict) or set(verdicts) != expected:
        raise ValueError("rubric_items_differ")
    evidence_field = "evidence_ids" if task.evidence_catalog is not None else "evidence"
    for key, value in verdicts.items():
        if not isinstance(value, dict) or set(value) != {"label", "reason", evidence_field}:
            raise ValueError("invalid_verdict_schema")
        label, reason, evidence = value["label"], value["reason"], value[evidence_field]
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError("invalid_verdict_label")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("missing_verdict_reason")
        if not isinstance(evidence, list) or any(not isinstance(q, str) or not q
                                                for q in evidence):
            raise ValueError("invalid_evidence_schema")
        if task.evidence_catalog is not None:
            if any(identifier not in task.evidence_catalog for identifier in evidence):
                raise ValueError("unknown_evidence_id")
            evidence = [task.evidence_catalog[identifier] for identifier in evidence]
            verdicts[key] = {**value, "evidence": evidence}
        if any(quote not in task.answer_text for quote in evidence):
            raise ValueError("evidence_not_in_answer")
    return verdicts


def _request(task: JudgeTask, index: int, items=None) -> JudgeRequest:
    items = tuple(task.item_ids) if items is None else tuple(k for k in task.item_ids if k in set(items))
    atom = {"type": "object", "additionalProperties": False,
            "required": ["label", "reason", "evidence"], "properties": {
                "label": {"type": "string", "enum": sorted(LABELS)},
                "reason": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
              "properties": {"verdicts": {
                  "type": "object", "additionalProperties": False,
                  "required": list(items),
                  "properties": {key: atom for key in items}}}}
    if task.evidence_catalog is not None:
        evidence = {"type": "array", "items": {"$ref": "#/$defs/evidence_id"}}
        if not task.evidence_catalog:evidence["maxItems"] = 0
        atomic = {"type": "object", "additionalProperties": False,
                  "required": ["label", "reason", "evidence_ids"], "properties": {
                      "label": {"type": "string", "enum": sorted(LABELS)},
                      "reason": {"type": "string"}, "evidence_ids": evidence}}
        evidence_id = {"type": "string"}
        if task.evidence_catalog:evidence_id["enum"] = list(task.evidence_catalog)
        schema["$defs"] = {"evidence_id": evidence_id, "atomic_verdict": atomic}
        schema["properties"]["verdicts"]["properties"] = {
            key: {"$ref": "#/$defs/atomic_verdict"} for key in items}
    fmt = {"type": "json_schema", "json_schema": {
        "name": "haenv_semantic_verdicts", "strict": True, "schema": schema}}
    prompt = task.prompt if items == tuple(task.item_ids) else task.subset_prompt(items)
    return JudgeRequest(sample_id=f"{task.fingerprint}:{index}", index=index,
                        prompt=prompt, response_format=fmt)


def _third_items(task: JudgeTask, first: dict, second: dict):
    """Lean protocol: atoms the first valid pair labelled differently; else every atom."""
    if task.atom_template is None or first["status"] != "ok" or second["status"] != "ok":
        return None
    return tuple(key for key in task.item_ids
                 if first["verdicts"][key]["label"] != second["verdicts"][key]["label"])


def _sample(task: JudgeTask, request: JudgeRequest, reply: JudgeReply, items=None) -> dict:
    base = {"sample_id": request.sample_id, "index": request.index,
            "task_sha256": task.fingerprint, "raw": reply.raw,
            "cached": reply.cached, "usage": reply.usage}
    if items is not None:
        base["items"] = list(items)
    try:
        if reply.cached is not False:
            raise ValueError("cached_response_not_independent")
        verdicts = parse_verdicts(reply.raw, task, items)
    except ValueError as exc:
        return {**base, "status": "invalid", "error": str(exc), "verdicts": None}
    return {**base, "status": "ok", "error": None, "verdicts": verdicts}


def _validate_prior(task: JudgeTask, prior_samples: list[dict]) -> list[dict]:
    if len(prior_samples) > 3:
        raise ValueError("At most three samples are permitted")
    samples = []
    for index, old in enumerate(prior_samples, 1):
        items = _third_items(task, samples[0], samples[1]) if index == 3 else None
        req = _request(task, index, items)
        if (old.get("sample_id") != req.sample_id or old.get("index") != index
                or old.get("task_sha256") != task.fingerprint
                or old.get("items", None) != (None if items is None else list(items))):
            raise ValueError("Duplicate, out-of-order or cross-task prior sample")
        if old.get("status") == "call_error":
            if old.get("raw") is not None or old.get("verdicts") is not None:
                raise ValueError("A failed call cannot hold a verdict")
            checked = dict(old)
        else:
            checked = _sample(task, req, JudgeReply(old.get("raw"), old.get("cached"), old.get("usage")),
                              items)
            if checked != old:
                raise ValueError("Persisted verdict differs from its raw evidence")
        samples.append(checked)
    return samples


def _disagree(first: dict, second: dict) -> bool | None:
    if first["status"] != "ok" or second["status"] != "ok":
        return None
    return any(first["verdicts"][key]["label"] != second["verdicts"][key]["label"]
               for key in first["verdicts"])


def evaluate_consensus(task: JudgeTask, dispatch: Callable[[JudgeRequest], JudgeReply], *,
                       prior_samples: list[dict] | None = None,
                       on_sample: Callable[[dict], None] | None = None,
                       parallel_first_pair: bool = False) -> dict:
    """Request two samples, then a third only on disagreement or an invalid first pair.

    Exact yes/no agreement by at least two independent valid samples resolves an
    item. Two identical uncertain responses stop after two calls but remain
    unresolved. The third request receives exactly the same prompt, never votes.
    The caller persists each sample before any following request is launched.
    """
    samples = _validate_prior(task, prior_samples or [])

    def make_sample(index, items=None):
        if len(samples) >= index:
            return None
        req = _request(task, index, items)
        try:
            response = dispatch(req)
            if not isinstance(response, JudgeReply):
                raise TypeError("Dispatcher must return JudgeReply")
        except BudgetExceeded:
            # A refused request is not a vote. Leave this index resumable and
            # propagate the accounting stop before another call is attempted.
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            # Exception text can contain backend URLs or keys; record the class only.
            record = {"sample_id": req.sample_id, "index": index,
                      "task_sha256": task.fingerprint, "raw": None, "cached": False,
                      "usage": None, "status": "call_error", "error": type(exc).__name__,
                      "verdicts": None}
            if items is not None:
                record["items"] = list(items)
        else:
            record = _sample(task, req, response, items)
        if on_sample is not None:
            pass
        return record

    def persist_sample(record):
        if record is None:
            return
        if on_sample is not None:
            on_sample(record)
        samples.append(record)

    def obtain(index, items=None):
        persist_sample(make_sample(index, items))

    if parallel_first_pair and not samples and len(prior_samples or []) == 0:
        # The first two votes are independent by protocol. Run them concurrently,
        # but persist them in index order and drain both futures before propagating
        # a budget stop. The adaptive third vote remains sequential.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(make_sample, index) for index in (1, 2)]
            records = []
            failure = None
            for future in futures:
                try:
                    records.append(future.result())
                except BudgetExceeded as exc:
                    failure = exc
                    records.append(None)
            for record in records:
                if record is None:
                    # Journals stay contiguous. A later sample that finished while
                    # an earlier one was refused keeps its durable provider receipt
                    # and is recovered on resume without another purchase.
                    break
                persist_sample(record)
            if failure is not None:
                raise failure
    else:
        obtain(1)
        obtain(2)
    disagreement = _disagree(samples[0], samples[1])
    third = _third_items(task, samples[0], samples[1])
    if disagreement is not False:
        obtain(3, third)
    elif len(samples) == 3:
        raise ValueError("An agreeing first pair must not have a third sample")
    final, votes = {}, {}
    for key in task.item_ids:
        counts = Counter(s["verdicts"][key]["label"] for s in samples
                         if s["status"] == "ok" and key in s["verdicts"])
        matches = [label for label in ("yes", "no") if counts[label] >= 2]
        final[key] = matches[0] if matches else None
        votes[key] = dict(sorted(counts.items()))
    unresolved = [key for key in task.item_ids if final[key] is None]
    result = {"protocol": task.protocol, "task_sha256": task.fingerprint,
              "status": "unresolved" if unresolved else "resolved", "verdicts": final,
              "unresolved": unresolved, "first_pair_disagreement": disagreement,
              "votes": votes, "samples": samples}
    if task.atom_template is not None:
        result["third_vote_items"] = None if len(samples) < 3 else (
            list(third) if third is not None else list(task.item_ids))
    return result
