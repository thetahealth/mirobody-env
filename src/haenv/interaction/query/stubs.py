"""Offline query stubs (no model calls). Each answers `chat(system, messages, tools, tool_choice)`
and runs through the same loop, guard, row builder and profile as a model.

* `q_reader`  -- reads everything: `list_sources` and `get_profile`, then every stream (following
  `earlier_page` until nothing is cut), every lab, medication and note, and every record section by
  name; it rebuilds the solver payload from the tool results alone (`rebuilt`) and answers with
  the pack's own visible-record reader on it (`adapter.read_answer`). Its rebuilt payload equals
  the payload the single-prompt geometry shows, so everything a decision needs is reachable
  through the tools, and its score is the reader's score.
* `q_nolook`  -- answers at once without a single call (`adapter.blind_answer`).
* `q_runaway` -- never stops calling tools; the loop closes the tools and the cell is scored on an
  empty answer.

Under the gated geometry (M2) the same three run through the purchase bridge: `q_reader` also
orders every menu item it can afford, cheapest first (`haenv/interaction/query/gated.py`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from haenv.interaction.query.loop import Turn
from haenv.interaction.query.solver import QueryMixin


def _results(messages) -> list[tuple[str, dict, dict]]:
    """`(tool name, arguments, result)` of every tool call so far, in order."""
    args = {}
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or ():
                args[c["id"]] = (c["name"], c.get("arguments") or {})
    out = []
    for m in messages:
        if m["role"] == "tool":
            name, a = args.get(m["tool_call_id"], (m.get("name"), {}))
            out.append((name, a, m["content"]))
    return out


def _call(i, tool, **args):
    return {"id": f"s{i}", "name": tool, "arguments": args}


class _StubBase(QueryMixin):
    backend, model, max_tokens, retries = "offline", "stub", None, 0
    query_mode_stub = True

    def __init__(self, q: dict):
        self.query = dict(q)
        self._turn = 0

    def solve_single(self, payload):
        from haenv.runtime.solvers import _to_output
        out = _to_output({}, payload)
        out._raw_text = ""
        return out

    def chat(self, system, messages, tools, tool_choice) -> Turn:
        self._turn += 1
        if tool_choice == "none":
            return self.final(messages)
        return self.step(messages, {t.name for t in tools})

    def final(self, messages) -> Turn:
        from haenv.scoring.management.plan import answer
        return Turn(content=json.dumps(answer(self.answer(messages)), ensure_ascii=False))

    def answer(self, messages) -> dict:
        return {}


class ReaderStub(_StubBase):
    """Reads everything, rebuilds the payload, answers with the pack's visible-record reader."""

    _streams: set = set()

    def on_case(self, adapter, payload) -> None:
        self._adapter, self._payload = adapter, payload
        pc = getattr(payload, "prediction_context", None) or {}
        blk = (pc.get(adapter.block) or {}) if adapter.block else {k: pc[k] for k in adapter.sections if k in pc}
        self._list_sections = {k for k, v in blk.items() if isinstance(v, list)}

    def step(self, messages, tools) -> Turn:
        got = _results(messages)
        names = {n for n, _a, _r in got}
        n0 = sum(1 for m in messages if m["role"] == "assistant") * 100
        if "list_sources" not in names:
            return Turn(tool_calls=[_call(n0, "list_sources"), _call(n0 + 1, "get_profile")]
                        + ([_call(n0 + 2, "get_menu")] if "get_menu" in tools else []))
        src = next(r for n, _a, r in got if n == "list_sources")
        calls = []
        if "order_test" in tools and "order_test" not in names:
            menu = next((r for n, _a, r in got if n == "get_menu"), None) or {}
            left = float(menu.get("remaining") or 0)
            for item in sorted(menu.get("items") or [], key=lambda i: (float(i["cost"]), i["target"])):
                if float(item["cost"]) <= left:
                    left -= float(item["cost"])
                    calls.append(_call(n0 + len(calls), "order_test", target=item["target"]))
            if calls:
                return Turn(tool_calls=calls)
        if not ({"get_series", "get_notes"} & names):
            self._streams = {s["name"] for s in src["streams"]}
            calls = [_call(n0 + i, "get_series", name=s["name"]) for i, s in enumerate(src["streams"])]
            calls += [_call(n0 + 50 + i, "get_record", section=r["section"]) for i, r in enumerate(src.get("records") or ())]
            calls += [_call(n0 + 90, "get_notes"), _call(n0 + 91, "get_labs"), _call(n0 + 92, "get_meds")]
            if "get_protocol" in tools:
                calls.append(_call(n0 + 93, "get_protocol"))
            return Turn(tool_calls=calls)
        # page back wherever a result was cut
        for n, a, r in got:
            page = ((r or {}).get("truncated") or {}).get("earlier_page")
            if page and not any(n2 == n and {**a, **page} == a2 for n2, a2, _r in got):
                calls.append({"id": f"s{n0 + len(calls)}", "name": n, "arguments": {**a, **page}})
        if calls:
            return Turn(tool_calls=calls)
        return self.final(messages)

    def rebuilt(self, messages, payload_like) -> SimpleNamespace:
        """The solver payload as the tool results show it (each resource unshaped to its own row)."""
        from haenv.interaction.query.fhir import source_of
        from haenv.interaction.query.fhir import unshape
        got = _results(messages)
        prof = next(({k: v for k, v in r.items() if k not in ("found", "resourceType", "id", "gender", "extension")}
                     for n, _a, r in got if n == "get_profile"), {})
        streams: dict[str, dict] = {}
        notes: dict[str, dict] = {}
        block: dict = {}
        for n, a, r in got:
            if not isinstance(r, dict) or not r.get("found"):
                continue
            if n == "get_series" and r.get("name") in self._streams:
                for e in r["entry"]:
                    p = unshape(e)
                    streams.setdefault(r["name"], {})[json.dumps(p, sort_keys=True, ensure_ascii=False)] = p
            elif n == "get_notes":
                for e in r["entry"]:
                    if source_of(e) == "evidence_ledger":
                        notes[e["evidence_id"]] = unshape(e)
            elif n == "get_record":
                if "value" in r:
                    block[r["section"]] = r["value"]
                else:
                    rows = block.setdefault(r["section"], {})
                    for e in r["entry"]:
                        rows[e["id"]] = unshape(e)
            elif n == "get_protocol":
                from haenv.world.protocols.library import catalog
                docs = [unshape(e) for e in r["entry"]]
                block["protocol_library"] = "本院规程库" if docs == catalog() else docs
        root_sections = ({k for k in self._adapter.sections if k in payload_like.prediction_context}
                         if self._adapter is not None else set())
        for k, v in list(block.items()):
            is_root_list = k in root_sections and isinstance(payload_like.prediction_context[k], list)
            if isinstance(v, dict) and (k in self._list_sections or is_root_list):
                block[k] = [v[i] for i in sorted(v, key=lambda i: int(str(i).rsplit("-", 1)[1]))]
        ld = {k: sorted(v.values(), key=lambda p: int(p["ts"])) for k, v in streams.items()}
        ctx = dict(payload_like.prediction_context)
        if self._adapter is not None and self._adapter.block:
            ctx = {k: v for k, v in ctx.items() if k not in root_sections}
            ctx.update({k: v for k, v in block.items() if k in root_sections})
            ctx[self._adapter.block] = {k: v for k, v in block.items() if k not in root_sections}
        elif self._adapter is not None:
            ctx = {k: v for k, v in ctx.items() if k not in self._adapter.sections}
            ctx.update(block)
        return SimpleNamespace(case_id=payload_like.case_id, user_profile=prof, prediction_context=ctx,
                               longitudinal_data=ld, evidence_ledger=sorted(
                                   notes.values(), key=lambda e: (int(e.get("source_timestamp", 0)), e["evidence_id"])))

    def answer(self, messages) -> dict:
        adapter = getattr(self, "_adapter", None)
        if adapter is None:
            return {}
        self.last_rebuilt = self.rebuilt(messages, self._payload)
        try:
            return adapter.read_answer(self.last_rebuilt)
        except (KeyError, TypeError, ValueError, IndexError):          # what it read does not decide
            return {}


class NoLookStub(_StubBase):
    def on_case(self, adapter, payload) -> None:
        self._adapter, self._payload = adapter, payload

    def step(self, messages, tools) -> Turn:
        return self.final(messages)

    def answer(self, messages) -> dict:
        return self._adapter.blind_answer(self._payload)


class ProtocolTitlesStub(NoLookStub):
    """Reads the patient-independent source catalog, without procedure bodies."""

    def step(self, messages, tools) -> Turn:
        if not _results(messages):
            return Turn(tool_calls=[_call(self._turn, "list_sources")])
        return self.final(messages)


class BlindSystemScoreStub(ReaderStub):
    """Reads everything but trusts the system's initial scoring inputs in P2."""

    def answer(self, messages) -> dict:
        answer = super().answer(messages)
        if self._adapter.pack != "p2" or not hasattr(self, "last_rebuilt"):
            return answer
        b = self.last_rebuilt.prediction_context[self._adapter.block]
        rec = b.get("home_check_today", {}).get("system_news2")
        if rec:
            from haenv.world.protocols.scoring import trusted_assessment
            a = trusted_assessment(b, rec)
            return {"disposition": a["set"][0], "rule": a["rule"].split("_", 1)[0],
                    "evidence": [], "reason": "采用系统自动评分的首次上传记录"}
        return answer


class RunawayStub(_StubBase):
    def step(self, messages, tools) -> Turn:
        return Turn(tool_calls=[_call(self._turn, "list_sources")])

    def final(self, messages) -> Turn:
        return Turn(tool_calls=[_call(self._turn, "list_sources")])


class TextPolicyStub(_StubBase):
    """A gated-protocol policy (`{"queries": [...]}` then an answer, e.g. `baselines.GatedShotgunSolver`)
    spoken through the query tools: its queries become `order_test` calls, its answer the final text.
    Its row must equal the policy's own row in the gated geometry (purchases, spend, atoms)."""

    def __init__(self, q: dict, policy):
        super().__init__(q)
        self.policy = policy

    def on_case(self, adapter, payload) -> None:
        self._payload = payload

    def step(self, messages, tools) -> Turn:
        self.policy.gated_context = getattr(self, "gated_context", None)
        out = self.policy.solve(self._payload)
        try:
            d = json.loads(out._raw_text)
        except (TypeError, ValueError):
            d = {}
        qs = d.get("queries") if isinstance(d, dict) else None
        if qs and "order_test" in tools:
            return Turn(tool_calls=[_call(f"{self._turn}.{i}", "order_test", target=q["target"])
                                    for i, q in enumerate(qs)])
        # an offline policy's answer lives on its output object; spoken as text it is the whole answer
        ans = {k: getattr(out, k) for k in ("forecast", "drivers", "action", "data_quality", "cited_evidence")
               if getattr(out, k, None)}
        return Turn(content=json.dumps({**ans, **(getattr(out, "_raw", {}) or {})}, ensure_ascii=False))

    def final(self, messages) -> Turn:
        return self.step(messages, set())


#: (name, factory(q)) pairs run by `--offline` on a job of a pack with an adapter.
QUERY_STUBS = (("q_reader", ReaderStub), ("q_nolook", NoLookStub), ("q_runaway", RunawayStub),
              ("q_protocol_titles", ProtocolTitlesStub), ("q_trust_system", BlindSystemScoreStub))
