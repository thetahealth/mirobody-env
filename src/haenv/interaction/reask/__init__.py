"""Deterministic diagnosis re-asking: archived results and neutral tool messages."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from haenv.interaction.query.gated import PURCHASE
from haenv.interaction.query.loop import FORCE_FINAL
from haenv.interaction.query.profile import returned_rows

ARMS = {"resample": "A", "decisive": "B", "placebo": "C", "half": "H"}


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def final_split(trace):
    """Remove the recorded answer and the optional trailing forced-final prompt."""
    messages = list(trace["messages"])
    if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("stored conversation does not end in a final assistant answer")
    prefix = messages[:-1]
    forced = bool(prefix and prefix[-1].get("role") == "user" and prefix[-1].get("content") == FORCE_FINAL)
    return prefix[:-1] if forced else prefix, messages[-1], forced


def injected(items):
    """One identical wrapper for every injected arm; evidence labels never reach the model."""
    if not items:
        return []
    calls = [{"id": f"inj{i}", "name": item["name"], "arguments": item["arguments"]}
             for i, item in enumerate(items)]
    return [{"role": "assistant", "content": "", "tool_calls": calls}] + [
        {"role": "tool", "tool_call_id": f"inj{i}", "name": item["name"], "content": item["content"]}
        for i, item in enumerate(items)]


def tool_items(trace):
    """Recorded result objects, in message order, paired with their original calls."""
    calls = {}
    for message in trace.get("messages") or ():
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or ():
                calls[call.get("id")] = call
        elif message.get("role") == "tool" and isinstance(message.get("content"), dict):
            call = calls.get(message.get("tool_call_id"), {})
            yield {"name": call.get("name", message.get("name")),
                   "arguments": call.get("arguments"), "content": message["content"]}


def keys_of(item, lookups):
    rows = list(returned_rows({"messages": [{"role": "tool", "name": item["name"],
                                           "content": item["content"]}]}))
    return sorted({lookup.what for lookup in lookups if any(lookup.seen(n, r) for n, r in rows)})


def candidates(items, lookups):
    """Deduplicate exactly as archived selection did; stable ties retain source order."""
    seen, out = set(), []
    for item in items:
        signature = (item["name"], json.dumps(item["arguments"], sort_keys=True, ensure_ascii=False),
                     json.dumps(item["content"], sort_keys=True, ensure_ascii=False))
        if signature not in seen:
            seen.add(signature)
            out.append({**item, "keys": keys_of(item, lookups)})
    return out


def cover(cand, need, *, environment=False):
    """Greedy missed-key coverage, preserving the archived selection's tie breaks."""
    need, selected = set(need), []
    while need:
        def rank(item):
            keys = set(item["keys"])
            common = (-len(need & keys), len(keys - need))
            if environment:
                return (*common, item["arguments"]["target"])
            return (item["name"] != PURCHASE, *common,
                    len(json.dumps(item["content"], ensure_ascii=False)),
                    json.dumps(item["arguments"], ensure_ascii=False))
        best = sorted((item for item in cand if need & set(item["keys"])), key=rank)
        if not best:
            break
        selected.append(best[0])
        need -= set(best[0]["keys"])
    return selected, sorted(need)


def placebo(cand, decisive, trace, model, question, repeat, seed=20261019):
    # Attempts count as bought even if a purchase was refused or unavailable.
    calls = list(trace.get("calls") or ())
    calls += [c for m in trace["messages"] if m.get("role") == "assistant"
              for c in m.get("tool_calls") or ()]
    bought = {(c.get("arguments") or {}).get("target") for c in calls
              if c.get("name") == PURCHASE and isinstance(c.get("arguments"), dict)}
    supplied = {(c.get("arguments") or {}).get("target") for c in decisive}
    pool = [c for c in cand if c["name"] == PURCHASE and not c["keys"]
            and c["content"].get("found") is not False and c["content"].get("ordered") is True
            and (c["arguments"] or {}).get("target") not in bought | supplied]
    pool.sort(key=lambda c: json.dumps(c["arguments"], ensure_ascii=False))
    random.Random(f"{seed}|{model}|{question}|{repeat}").shuffle(pool)
    if len(pool) < len(decisive):
        raise ValueError(f"{question}: placebo shortfall ({len(pool)} available, {len(decisive)} required); "
                         "include more recorded result sources with --sources")
    return pool[:len(decisive)]


def half(decisive, model, question, repeat, seed=20261020):
    if len(decisive) < 2:
        raise ValueError(f"{question}: half requires at least two missed results")
    chosen = set(random.Random(f"{seed}|{model}|{question}|{repeat}").sample(
        range(len(decisive)), len(decisive) // 2))
    return [c for i, c in enumerate(decisive) if i in chosen]


class Batch:
    """Read-only archived query batch, with the last trace/response per cell."""

    def __init__(self, path):
        from haenv.payloads import load_payloads
        self.path = Path(path)
        self.payloads = load_payloads(self.path / "payloads.jsonl")
        self.traces = {(r["case"], r["solver"]): r for r in read_rows(self.path / "query_trace.jsonl")}
        self.responses = {(r["case"], r["solver"]): r for r in read_rows(self.path / "responses.jsonl")}
        self.evals = {(r["case"], r["solver"]): r for r in read_rows(self.path / "eval.jsonl")}
        self._bought, self._receipts, self._cases = None, None, None
        self._payload_times, self._settings_receipts = None, None

    def world_identity(self, question):
        """Payload identity and T, supplemented by archived per-case fingerprints."""
        sp, vp = self.payloads[question]
        if self._payload_times is None:
            self._payload_times = {r["case_id"]: r.get("T") for r in read_rows(self.path / "payloads.jsonl")}
        stored_T = self._payload_times.get(question)
        if stored_T is not None and stored_T != sp.prediction_context["prediction_time_T"]:
            raise ValueError(f"{question}: source world mismatch: stored T disagrees with solver payload in {self.path}")
        payload_sha = digest(json.dumps([asdict(sp), asdict(vp)], ensure_ascii=False, sort_keys=True))
        fingerprints = {}
        for filename, key in (("batch.json", "cases_sha256"),
                              ("solver-accounting/manifest.json", "case_digests")):
            path = self.path / filename
            if path.is_file():
                value = (json.loads(path.read_text(encoding="utf-8")).get(key) or {}).get(question)
                if value is not None:
                    fingerprints[key] = value
        return {"payload_sha256": payload_sha, "T": sp.prediction_context["prediction_time_T"],
                "fingerprints": fingerprints}

    def original_answer_settings(self, question, model, prefix_sha):
        """Pin answer conditions from the batch identity and matching final receipts.

        Route-only differences use the runtime's canonical answer view. Missing or
        partial records cannot establish exact replay; contradictory records refuse.
        """
        from haenv.runtime.transport import ANSWER_KEYS, answer_view
        records = []
        path = self.path / "solver-accounting/manifest.json"
        if path.is_file():
            identity = json.loads(path.read_text(encoding="utf-8")).get("identity") or {}
            record = (identity.get("answer_settings") or identity.get("settings") or {}).get(model)
            if record:
                records.append(record)
        if self._settings_receipts is None:
            self._settings_receipts = defaultdict(list)
            for path in sorted((self.path / "solver-accounting/receipts").glob("*.json")):
                if path.name.endswith((".billing.json", ".inflight.json")):
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                cell, request = row.get("cell") or {}, row.get("request") or {}
                if request.get("settings"):
                    key = cell.get("case_id"), cell.get("solver"), request.get("prompt_sha256")
                    self._settings_receipts[key].append(request["settings"])
        records.extend(self._settings_receipts[question, model, prefix_sha])
        normalized, incomplete = [], False
        for record in records:
            required = set(ANSWER_KEYS) if "model_canonical" in record else {
                "backend", "model", "max_tokens", "reasoning_effort", "response_format"}
            if not required <= record.keys():
                incomplete = True
                continue
            view = answer_view(record)
            normalized.append({k: view[k] for k in ANSWER_KEYS})
        if normalized and any(value != normalized[0] for value in normalized[1:]):
            raise ValueError(f"{question}/{model}: conflicting source answer settings")
        return normalized[0] if normalized and not incomplete else None

    def store(self, question, model):
        from haenv.interaction.gated import withhold_signals
        from haenv.interaction.query.tools import RecordStore
        from haenv.m2.query import ADAPTER
        if self._bought is None:
            self._bought = defaultdict(dict)
            path = self.path / "trace.jsonl"
            if not path.is_file():
                path = (self.path / "query_trace.jsonl").resolve().parent / "trace.jsonl"
            calls = {}
            for event in read_rows(path):
                cell, data = (event.get("case"), event.get("solver")), event.get("data") or {}
                if event["type"] == "tool/call":
                    calls[cell, data.get("call_id")] = data.get("target")
                elif event["type"] == "tool/result" and data.get("revealed") and data.get("observation"):
                    target = calls.get((cell, data.get("call_id")))
                    if target:
                        self._bought[cell][target] = data["observation"]
        payload, _ = self.payloads[question]
        lean, _ = withhold_signals(payload)
        lean.longitudinal_data.update(self._bought.get((question, model), {}))
        trace = self.traces[question, model]
        limit = (trace.get("limits") or {}).get("max_points")
        if limit is None:
            configs = read_rows(self.path / "config_runs.jsonl")
            limit = ((configs[-1].get("resolved") or {}).get("query") or {}).get("max_points") if configs else None
        if limit is None:
            raise ValueError("stored trace/config is missing max_points")
        return RecordStore(lean, int(payload.prediction_context["prediction_time_T"]), ADAPTER, max_points=limit)

    def prefix_check(self, question, model):
        from haenv.interaction.query.gated import _specs
        from haenv.runtime.transport import make_envelope
        trace = self.traces[question, model]
        prefix, _, forced = final_split(trace)
        reqs = self.responses.get((question, model), {}).get("requests") or []
        want = reqs[-1].get("prompt_sha256") if reqs else None
        if want:
            wants, basis = {want}, "response_row"
        else:
            if self._receipts is None:
                self._receipts = defaultdict(set)
                for path in sorted((self.path / "solver-accounting/receipts").glob("*.json")):
                    if path.name.endswith(".billing.json"):
                        continue
                    row = json.loads(path.read_text(encoding="utf-8"))
                    cell, request = row.get("cell") or {}, row.get("request") or {}
                    if request.get("prompt_sha256"):
                        self._receipts[cell.get("case_id"), cell.get("solver")].add(request["prompt_sha256"])
            wants, basis = self._receipts[question, model], "receipts"
        store = self.store(question, model)
        messages = prefix + ([{"role": "user", "content": FORCE_FINAL}] if forced else [])
        for can_order in (True, False):
            specs = _specs(store, {"menu": [], "budget": 0, "spent": 0}, can_order)
            tools = [s.neutral() for s in specs]
            choice = "none" if forced else "auto"
            sha = digest(make_envelope(trace["system"], messages, tools, choice))
            if sha in wants:
                return {"tools": tools, "prefix_sha": sha, "basis": basis,
                        "forced": forced, "original_tool_choice": choice}
        raise ValueError(f"{question}/{model}: rebuilt final request does not match stored fingerprint")

    def environment_items(self, question, model, targets):
        """Use the environment's own purchase path for results absent from the archive."""
        from haenv.interaction import gated
        from haenv.interaction.query.gated import _purchase_result
        from haenv.interaction.query.tools import RecordStore
        from haenv.m2 import hooks
        from haenv.m2.query import ADAPTER
        from haenv.store import load_cases
        from haenv.world.generation.baselines import ConstantDdxSolver
        hooks.install()
        if self._cases is None:
            self._cases = load_cases(self.path / "cases.jsonl")
        raw = self._cases[question]
        observations, call_targets = {}, {}

        class Buyer:
            def __init__(self):
                self.n = 0

            def solve(self, payload):
                self.n += 1
                out = ConstantDdxSolver().solve(payload)
                if self.n == 1:
                    menu = {i["target"]: i.get("kind") for i in self.gated_context.get("menu", [])}
                    out._raw_text = json.dumps({"queries": [{"kind": menu.get(t, "test"), "target": t}
                                                          for t in targets], "commit": False}, ensure_ascii=False)
                    out._raw = {}
                return out

        class Trace:
            def append(self, typ, data, step=None):
                if typ == "tool/call":
                    call_targets[data["call_id"]] = data["target"]
                elif typ == "tool/result" and data.get("revealed") and data.get("observation"):
                    observations[call_targets[data["call_id"]]] = data["observation"]

        T = int(raw.prediction_context["prediction_time_T"])
        gated.run_gated(raw, T, Buyer(), budget=1e9, trace=Trace())
        lean, _ = gated.withhold_signals(self.payloads[question][0])
        lean.longitudinal_data.update(observations)
        store = RecordStore(lean, T, ADAPTER, max_points=self.store(question, model).max_points)
        return [{"name": PURCHASE, "arguments": {"target": t}, "content": _purchase_result(t, store, {}),
                 "src": "env_purchase"} for t in targets]


def build_unit(batch, question, model, repeat, source_items, *, arms=tuple(ARMS),
               seed=20261019, half_seed=20261020):
    """Reconstruct one unit, checking the pre-answer request before making any changes."""
    from haenv.m2.query import ADAPTER
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    trace = batch.traces[question, model]
    check = batch.prefix_check(question, model)
    prefix, _, _ = final_split(trace)
    _, verifier = batch.payloads[question]
    lookups = ADAPTER.key_lookups(verifier, None)
    if not lookups:
        raise ValueError(f"{question}: no diagnosis evidence list")
    rows = list(returned_rows({"messages": prefix}))
    missed = [lk.what for lk in lookups if not any(lk.seen(n, r) for n, r in rows)]
    if not missed:
        arms = tuple(arm for arm in arms if arm == "resample")
    cand = candidates(source_items, lookups)
    decisive, need = cover(cand, missed) if set(arms) - {"resample"} else ([], [])
    if need:
        menu = next((c["content"] for c in tool_items(trace) if c["name"] == "get_menu"), None)
        if menu is None:
            menu = next((c["content"] for c in cand if c["name"] == "get_menu"), {})
        env = candidates(batch.environment_items(question, model, [i["target"] for i in menu.get("items", [])]), lookups)
        added, need = cover(env, need, environment=True)
        decisive += added
    if need:
        raise ValueError(f"{question}: missed evidence cannot be supplied: {need}")
    items = {"resample": [], "decisive": decisive}
    if "placebo" in arms:
        items["placebo"] = placebo(cand, decisive, trace, model, question, repeat, seed)
    if "half" in arms:
        if len(decisive) >= 2:
            items["half"] = half(decisive, model, question, repeat, half_seed)
        # Ineligible units have no half request, rather than an empty injection labeled half.
    requests = {arm: {"system": trace["system"],
                      "messages": deepcopy(prefix + injected(items[arm]) + [{"role": "user", "content": FORCE_FINAL}]),
                      "tools": check["tools"], "items": items[arm]}
                for arm in arms if arm in items}
    return {"model": model, "question": question, "repeat": repeat,
            "kind": "target" if missed else "control", "missed": missed,
            **check, "requests": requests}
