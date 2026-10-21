"""A query-mode solver: the production model solver, asked through tools.

`as_query_solver(inner, query)` returns a copy of a production solver (`OpenAICompatSolver`,
`GoogleSolver`) whose class adds query mode. It keeps every route and accounting path of the
original: each turn is one request through the same request accountant (`solver._accounting`,
ledger reservation and settlement, receipts, content-addressed request ids, replay on resume,
rate-limit backoff, key rotation, fallback routes) and the same pre-send wire check. The accounted
prompt of a turn is the conversation envelope (`transport.make_envelope`); the body is derived from
it per backend (`transport.envelope_content`). Google model turns are sent back verbatim, with
the `thoughtSignature` of every function call.

`solve(payload)` on a case a pack adapter owns runs the tool loop (`loop.run_query`) and returns a
`SolverOutput` like a single-prompt solve: the final text is `_raw_text`, the summed usage is
`_usage`, the requests of every turn are `_requests`, and the transcript is `_query_trace`. Under
the gated geometry (M2 purchases) it runs one purchase round per call (`gated.GatedBridge`). A
case no adapter owns is solved as before, in one prompt.

The question is the pack's own template rendered on a view of the payload whose data are replaced
by the header the pack shows at once and pointers to the tools (`QueryView`), followed by the
query protocol (`protocol_text`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from haenv.interaction.query.adapter import adapter_for
from haenv.interaction.query.loop import Turn
from haenv.interaction.query.loop import run_query
from haenv.interaction.query.tools import RecordStore
from haenv.interaction.query.tools import ResultGuard

log = logging.getLogger("haenv.query")

QUERY_SYSTEM = ("你是一个通过只读工具查阅患者记录、再作答的求解器。病历不会整份给你:你只能用工具读取今天(T 日)"
                "及之前的数据。工具只读、不收费、可以在一轮里同时调用多个;轮次与调用次数有上限,用完后必须直接作答。"
                "作答时不再调用工具,只输出题目要求的 JSON。医疗边界(硬约束):不自主开药/停药/改剂量。")

UNANSWERED = json.dumps({"query_stop": "unanswered"})


def protocol_text(q: dict, extra: str = "") -> str:
    return ("\n\n【查询式作答】上面「数据」处只给出题头与患者基本档案;记录的其余部分不在题面里,"
            "用工具读取(工具说明见工具列表;list_sources 给出这位患者有哪些数据与起止日)。"
            f"\n· 最多 {q['max_rounds']} 轮工具调用、合计 {q['max_calls']} 次调用;一次调用最多返回 "
            f"{q['max_points']} 个点或条目,超出时返回最近的部分并告诉你如何往前翻页。"
            "\n· 日期是病程日:今天是 T 日,负数是本记录第 0 天之前的历史。" + extra +
            "\n· 读够了就直接输出答案 JSON(不再调用工具);答案的 schema 与上面要求的完全相同。")


class QueryView:
    """The payload as the question shows it in query mode: the same JSON skeleton, with the data
    replaced by the pack's header and pointers to the tools. Pack hooks dispatch on
    `prediction_context` (the original dict); `dumps()` is what the template prints."""

    def __init__(self, payload, adapter, store: RecordStore):
        self.adapter, self.store = adapter, store
        self.case_id = getattr(payload, "case_id", None)
        self.user_profile = getattr(payload, "user_profile", None) or {}
        if adapter is not None:
            self.user_profile = adapter.profile_header(self.user_profile)
        self.prediction_context = getattr(payload, "prediction_context", None) or {}
        self.longitudinal_data, self.evidence_ledger = {}, []
        ctx = {k: v for k, v in self.prediction_context.items()
               if adapter is None or (k != adapter.block and k not in adapter.sections)}
        if adapter is not None and adapter.block:
            ctx[adapter.block] = {**adapter.header(store.block),
                                  "其余部分(用工具读取)": [s for s in store.sections if s in store.block]}
        self._shown = {"case_id": self.case_id, "user_profile": self.user_profile, "prediction_context": ctx,
                       "longitudinal_data": "用 list_sources / get_series / get_labs 读取",
                       "evidence_ledger": "用 get_notes 读取"}

    def dumps(self) -> str:
        return json.dumps(self._shown, ensure_ascii=False, indent=1)


class _RenderAs:
    """The solver as the renderer sees it in query mode: no gated menu (the purchase protocol is the
    tools'), everything else forwarded."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name == "gated_context":
            return None
        return getattr(self._inner, name)


def render_question(solver, view) -> tuple[str, str]:
    import haenv.interaction.qside as qside
    from haenv.world.protocols.delivery import question
    if view.adapter is not None:
        template = question(view.adapter.pack, view.store.block)
        if template is not None:
            return template.format(payload=view.dumps()), f"{view.adapter.pack}.protocol"
    return qside.render_for(_RenderAs(solver), view)


def _sum_usage(usages: list) -> dict | None:
    """Usage summed over the turns of a cell (the shape of one OpenAI-compatible usage)."""
    got = [u for u in usages if isinstance(u, dict)]
    if not got:
        return None
    out: dict = {}
    for u in got:
        for k, v in u.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = (out.get(k) or 0) + v
            elif isinstance(v, dict):
                sub = out.setdefault(k, {})
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        sub[kk] = (sub.get(kk) or 0) + vv
                    elif kk not in sub:
                        sub[kk] = vv
            elif k not in out:
                out[k] = v
    out["query_turns"] = len(usages)
    return out


class QueryMixin:
    """Query mode on top of a solver class. Subclasses provide `chat`."""

    #: run limits (`config.query`); set by `as_query_solver`, part of the answer view
    query: dict = {}

    def solve(self, payload):
        if getattr(self, "gated_context", None):
            from haenv.interaction.query.gated import solve_gated_round
            return solve_gated_round(self, payload)
        adapter = adapter_for(payload)
        if adapter is None:
            return self.solve_single(payload)
        return self.solve_query(payload, adapter)

    def solve_single(self, payload):
        return super().solve(payload)                                 # noqa: the production solve

    def on_case(self, adapter, payload) -> None:
        """Called once a payload is put to the solver in query mode (offline stubs read it)."""

    def solve_query(self, payload, adapter):
        from haenv.runtime.solvers import _to_output
        from haenv_kernel.solver import _extract_json
        q = self.query
        T = int(payload.prediction_context["prediction_time_T"])
        self.on_case(adapter, payload)
        store = RecordStore(payload, T, adapter, max_points=q["max_points"])
        question, pid = render_question(self, QueryView(payload, adapter, store))
        question += protocol_text(q)
        guard = getattr(self, "query_guard", None) or ResultGuard(T)
        self._failed_turns = []
        t0 = time.time()
        text, tr = run_query(store.specs(), guard, QUERY_SYSTEM, question, self,
                             max_rounds=q["max_rounds"], max_calls=q["max_calls"])
        raw = UNANSWERED if text is None else text
        try:
            data = _extract_json(raw) if raw.strip() else {}
        except Exception:                                             # noqa: BLE001 -- unparseable is a recorded outcome
            data = {}
        out = _to_output(data if isinstance(data, dict) else {}, payload)
        sha = hashlib.sha256(question.encode()).hexdigest()
        out._raw_text, out._prompt_mode = raw, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha, out._prompt_sha_full = pid, sha[:16], sha
        out._usage, out._finish = _sum_usage(tr.usage), (tr.finish[-1] if tr.finish else None)
        out._max_tokens = getattr(self, "max_tokens", None)
        out._failed_attempts = list(self._failed_turns)
        from haenv.runtime.solvers import _drain_backoff
        from haenv.runtime.solvers import _drain_requests
        out._rate_limit_retries = _drain_backoff(self)
        out._requests = _drain_requests(self)
        out._latency_s = round(time.time() - t0, 2)
        out._reasoning_chars = None
        out._query_trace = {**tr.as_dict(), "pack": adapter.pack, "T": T, "limits": dict(q)}
        self._last_query = (out._query_trace, store)                 # read by the row wrapper (`query.install`)
        return out


# ---------------------------------------------------------------- production chat

def _openai_turn(resp: dict) -> Turn:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    raw_calls = msg.get("tool_calls") or []
    calls, native_calls = [], []
    for i, tc in enumerate(raw_calls):
        fn = tc.get("function") or {}
        cid = tc.get("id") or f"call_{i}"
        args = fn.get("arguments")
        try:
            parsed = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
        except ValueError:
            parsed = args                                             # call_tool reports bad_arguments
        calls.append({"id": cid, "name": fn.get("name"), "arguments": parsed})
        native_calls.append({"id": cid, "type": "function",
                             "function": {"name": fn.get("name"), "arguments": args if isinstance(args, str)
                                          else json.dumps(args or {}, ensure_ascii=False)}})
    native = {}
    if native_calls:
        native["tool_calls"] = native_calls
    if msg.get("reasoning_details"):
        native["reasoning_details"] = msg["reasoning_details"]
    return Turn(content=msg.get("content") or "", tool_calls=calls, native=native or None,
                usage=resp.get("usage"), finish=choice.get("finish_reason"))


def _google_turn(resp: dict) -> Turn:
    from haenv.runtime.solvers import _google_usage
    cand = (resp.get("candidates") or [{}])[0]
    parts = list(((cand.get("content") or {}).get("parts") or []))
    text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
    calls = []
    for i, p in enumerate(parts):
        fc = p.get("functionCall")
        if isinstance(fc, dict):
            calls.append({"id": f"g:{fc['id']}" if fc.get("id") else f"gcall{i}", "name": fc.get("name"),
                          "arguments": fc.get("args") or {}})
    return Turn(content=text, tool_calls=calls, native={"parts": parts}, usage=_google_usage(resp),
                finish=cand.get("finishReason"))


class _HTTPChat:
    """`chat` over the production transport: one accounted request per turn."""

    def _wire_body(self, prompt: str) -> bytes:
        from haenv.runtime.transport import solver_envelope_content, is_envelope
        if not is_envelope(prompt):
            return super()._wire_body(prompt)
        import haenv.runtime.llm as _llm
        from haenv.runtime.transport import sent_temperature
        content = solver_envelope_content(self, prompt)
        if self.backend == "google":
            gen = {"maxOutputTokens": self.max_tokens}
            if sent_temperature(self.sampling) is not None:
                gen["temperature"] = sent_temperature(self.sampling)
            if (self.sampling or {}).get("thinking_budget") is not None:
                gen["thinkingConfig"] = {"thinkingBudget": self.sampling["thinking_budget"]}
            body = {**content, "generationConfig": gen}
            _llm.note_payload(self.name, body)
            return json.dumps(body).encode()
        body = json.loads(super()._wire_body(""))                     # model, effort, sampling, routing, stream
        body.update(content)
        return json.dumps(body).encode()

    def chat(self, system, messages, specs, tool_choice) -> Turn:
        from haenv.runtime.transport import make_envelope
        from haenv.runtime.budget import BudgetExceeded
        from haenv.interaction.solve_guard import json_unextractable
        prompt = make_envelope(system, messages, [s.neutral() for s in specs], tool_choice)
        last = Turn()
        for attempt in range(1, int(self.retries) + 2):
            t0 = time.time()
            try:
                resp = self._post(prompt)
                if isinstance(resp, dict) and resp.get("error") and not (resp.get("choices") or resp.get("candidates")):
                    raise RuntimeError(f"error body: {str(resp['error'])[:160]}")
                turn = (_google_turn if self.backend == "google" else _openai_turn)(resp)
                last = turn
                if not turn.tool_calls and (not turn.content.strip() or json_unextractable(turn.content)):
                    raise ValueError(f"turn has no tool call and no answer JSON (finish={turn.finish})")
                return turn
            except BudgetExceeded:
                raise
            except Exception as e:                                    # noqa: BLE001 -- retried like a single-prompt solve
                log.warning("[query %s] turn attempt %d failed: %s: %s", self.name, attempt, type(e).__name__,
                            str(e)[:160])
                self._failed_turns = getattr(self, "_failed_turns", []) + [{
                    "turn": sum(1 for m in messages if m.get("role") == "assistant") + 1, "attempt": attempt,
                    "error": f"{type(e).__name__}: {str(e)[:200]}", "elapsed_s": round(time.time() - t0, 2),
                    "finish_reason": last.finish, "raw": last.content}]
                time.sleep(min(3 * attempt, 10))
        return last

    # ---- streaming: tool-call deltas are kept (the single-prompt reader keeps content only) ----
    def _read_sse(self, r, *, defer_validation: bool = False, wire: dict | None = None) -> dict:
        from haenv.runtime.completion import note_generation
        txt, think_n, usage, fin, gid, served, served_model = [], 0, None, None, None, None, None
        calls: dict[int, dict] = {}
        wire = wire if wire is not None else {}
        for line in r:
            wire["bytes_received"] = wire.get("bytes_received", 0) + len(line)
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if not s or s == "[DONE]":
                continue
            try:
                ch = json.loads(s)
            except ValueError:
                continue
            if gid is None and isinstance(ch.get("id"), str):
                gid = wire["generation_id"] = ch["id"]
                note_generation(gid)
            served = served or (ch.get("provider") if isinstance(ch.get("provider"), str) else None)
            served_model = served_model or (ch.get("model") if isinstance(ch.get("model"), str) else None)
            usage = ch.get("usage") or usage
            c0 = (ch.get("choices") or [{}])[0]
            fin = c0.get("finish_reason") or fin
            delta = c0.get("delta") or {}
            if isinstance(delta.get("content"), str):
                txt.append(delta["content"])
            for f in self._SSE_THINK_FIELDS:
                if isinstance(delta.get(f), str):
                    think_n += len(delta[f])
            for tc in delta.get("tool_calls") or ():
                slot = calls.setdefault(int(tc.get("index", len(calls))), {"id": None, "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
        self._last_reasoning_chars, self._last_reasoning_text = think_n, None
        message = {"content": "".join(txt)}
        if calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for _i, c in sorted(calls.items())]
        if not defer_validation:
            self._validate_stream(message["content"], think_n, usage, fin)
        result = {"choices": [{"message": message, "finish_reason": fin}], "usage": usage,
                  "_reasoning_chars": think_n, **({"provider": served} if served else {}),
                  **({"model": served_model} if served_model else {}), **({"id": gid} if gid else {})}
        if defer_validation:
            result["_reasoning_text"], result["_generation_id"] = None, gid
        return result

    @staticmethod
    def _validate_stream(answer, think_n, usage, fin):
        if fin is None:
            raise ValueError("流式响应未收到终止帧(finish_reason 缺失) —— 判为流被截断")
        if fin not in ("tool_calls", "function_call") and not answer.strip() and think_n > 0:
            raise ValueError(f"流式响应只有推理没有作答(推理 {think_n} 字符、作答 0 字符)")


_CLASSES: dict[type, type] = {}


def query_class(base: type) -> type:
    if base not in _CLASSES:
        _CLASSES[base] = type(f"Query{base.__name__}", (QueryMixin, _HTTPChat, base), {})
    return _CLASSES[base]


def as_query_solver(inner, query: dict):
    """A copy of production solver `inner` in query mode under the limits `query`."""
    import copy
    q = copy.copy(inner)
    q.__class__ = query_class(type(inner))
    q.query = dict(query)
    return q
