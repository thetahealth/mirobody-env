"""`FakeProvider`: a local HTTP server that answers as OpenRouter, Google and DashScope.

The provider is chosen by the request's `Host` header (the client keeps it when
`transport.rehearsal_url` redirects a request here). Per request the server draws a service
time from the profile, sleeps it (times `time_scale`), and answers with a recorded or
synthesized answer (`responder`), or injects one of the profiled errors. Every completion
carries a generation id and usage (with `cost` on OpenRouter), and OpenRouter's
`/generation`, `/models`, `/models/<id>/endpoints`, `/credits` and `/key` answer from the
profile, so pricing, settlement and reconcile run as against the real provider.

Each finished request is one line in the request log (JSONL): when it arrived and ended,
which model and answer kind, the outcome, the service time and the generation id.

A request that declares tools (query mode) is answered as a scripted tool user (`tool_turn`):
it calls the declared tools of `TOOL_SCRIPT` one turn at a time, then answers from the library;
with the tools closed (`tool_choice: none`, Google mode `NONE`) it answers at once. Its Google
function calls carry a `thoughtSignature`, and a later request whose model turn does not send a
call's signature back verbatim gets the 400 the real API returns.
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
import threading
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from haenv.runtime.loadtest.profile import Profile
from haenv.runtime.loadtest.responder import Answer
from haenv.runtime.loadtest.responder import Responder

OPENROUTER, GOOGLE, DASHSCOPE = "openrouter.ai", "generativelanguage.googleapis.com", "dashscope.aliyuncs.com"
#: Pieces a streamed answer is cut into.
STREAM_CHUNKS = 4
#: Seconds a 429 takes to come back (before scaling).
REJECT_S = 0.5


#: tool calls of a scripted query-mode turn, by turn: (tool name, arguments)
TOOL_SCRIPT = ((("list_sources", {}), ("get_profile", {})), (("get_labs", {}),), (("get_notes", {}),))


def tool_turn(fake, host: str, body: dict) -> list | None:
    """The calls of the next scripted turn, or None to answer. Raises `ValueError` on a Google model
    turn whose function call lost its signature."""
    if host == GOOGLE:
        contents = body.get("contents") or []
        for c in contents:
            for part in c.get("parts") or () if c.get("role") == "model" else ():
                if "functionCall" in part and part.get("thoughtSignature") not in fake.signatures:
                    raise ValueError("Function call is missing a thought_signature in functionCall parts.")
        closed = ((body.get("toolConfig") or {}).get("functionCallingConfig") or {}).get("mode") == "NONE"
        turns = sum(1 for c in contents if c.get("role") == "model")
        declared = {d["name"] for t in body.get("tools") or () for d in t.get("functionDeclarations") or ()}
    else:
        closed = body.get("tool_choice") == "none"
        turns = sum(1 for m in body.get("messages") or () if m.get("role") == "assistant")
        declared = {t["function"]["name"] for t in body.get("tools") or ()}
    if closed or turns >= len(TOOL_SCRIPT):
        return None
    calls = [(n, a) for n, a in TOOL_SCRIPT[turns] if n in declared]
    return calls or None


class FakeProvider:
    """Start with `start()`; `url` is the base URL to put in `transport.rehearsal_url`."""

    def __init__(self, profile: Profile, responder: Responder, log_path: Path, *,
                 time_scale: float = 1.0, seed: int = 0, host: str = "127.0.0.1", port: int = 0):
        self.profile, self.responder, self.time_scale = profile, responder, float(time_scale)
        self.rng = random.Random(seed)
        self.rng_lock = threading.Lock()
        self.generations: dict[str, dict] = {}
        self.signatures: set[str] = set()
        self.in_flight: dict[str, int] = {}
        self.lock = threading.Lock()
        self.logged = threading.Condition(self.lock)
        self.n_logged = 0
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", encoding="utf-8")
        self.httpd = ThreadingHTTPServer((host, port), _handler(self))
        self.httpd.daemon_threads = True
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeProvider":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._log.close()

    # ---- shared state ----
    def draw(self, fn):
        with self.rng_lock:
            return fn(self.rng)

    def log(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.logged:
            self._log.write(line)
            self._log.flush()
            self.n_logged += 1
            self.logged.notify_all()

    def wait_logged(self, n: int, timeout: float = 10.0) -> bool:
        """Block until `n` requests are in the log. A request is logged after its reply is
        sent, so a client can read its answer before the record is written."""
        with self.logged:
            return self.logged.wait_for(lambda: self.n_logged >= n, timeout)

    def enter(self, backend: str) -> bool:
        """Count a request in flight; False when the backend's concurrency limit is exceeded."""
        with self.lock:
            n = self.in_flight.get(backend, 0) + 1
            self.in_flight[backend] = n
        limit = self.profile.max_concurrency.get(backend)
        return limit is None or n <= limit

    def leave(self, backend: str) -> None:
        with self.lock:
            self.in_flight[backend] -= 1

    def cost(self, model: str, usage: dict) -> str:
        pricing = (self.profile.catalog.get(model) or {}).get("pricing") or {}
        rate = lambda k, d="0": Decimal(str(pricing.get(k, d)))  # noqa: E731
        fresh = max(0, usage["prompt"] - usage["cached"])
        total = (fresh * rate("prompt") + usage["cached"] * rate("input_cache_read", pricing.get("prompt", "0"))
                 + usage["completion"] * rate("completion"))
        return format(total, "f")


def _handler(fake: FakeProvider):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):          # the request log replaces stderr lines
            pass

        # ---- plumbing ----
        def _send_json(self, status: int, doc, headers: dict | None = None) -> None:
            body = json.dumps(doc, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _provider(self) -> str:
            return (self.headers.get("Host") or "").split(":")[0]

        # ---- GET: catalog, capacity, generation records ----
        def do_GET(self):
            url = urlsplit(self.path)
            host = self._provider()
            if host == GOOGLE:
                model = url.path.rsplit("/", 1)[-1]
                limits = fake.profile.catalog.get(f"google:{model}")
                return (self._send_json(200, {"name": f"models/{model}", **limits}) if limits
                        else self._send_json(404, {"error": {"code": 404, "message": "not found"}}))
            path = url.path.removeprefix("/api/v1")
            if path == "/models":
                return self._send_json(200, {"data": [m for k, m in fake.profile.catalog.items()
                                                      if ":" not in k]})
            if path.startswith("/models/") and path.endswith("/endpoints"):
                model = path[len("/models/"):-len("/endpoints")]
                meta = fake.profile.catalog.get(model) or {}
                return self._send_json(200, {"data": {"id": model, "endpoints": meta.get("endpoints") or [
                    {"provider_name": fake.profile.model(model).served_by, "pricing": meta.get("pricing") or {},
                     "supported_parameters": meta.get("supported_parameters") or []}]}})
            if path == "/generation":
                gid = (parse_qs(url.query).get("id") or [""])[0]
                record = fake.generations.get(gid)
                return (self._send_json(200, {"data": record}) if record
                        else self._send_json(404, {"error": {"code": 404, "message": "not found"}}))
            if path in ("/credits", "/key"):
                return self._send_json(200, {"data": {"total_credits": 1e6, "total_usage": 0,
                                                      "usage": 0, "limit": None}})
            return self._send_json(404, {"error": {"code": 404, "message": "unknown path"}})

        # ---- POST: completions ----
        def do_POST(self):
            t0 = time.time()
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            host = self._provider()
            backend = {OPENROUTER: "openrouter", GOOGLE: "google", DASHSCOPE: "dashscope"}.get(host, host)
            if host == GOOGLE:
                model = urlsplit(self.path).path.rsplit("/", 1)[-1].split(":")[0]
                prompt = "".join(p.get("text", "") for c in body.get("contents") or []
                                 for p in c.get("parts") or [])
            else:
                model = body.get("model", "")
                prompt = "".join(str(m.get("content", "")) for m in body.get("messages") or [])
            answer = fake.draw(lambda r: fake.responder.answer(
                model, prompt, body.get("response_format")))
            calls = None
            if body.get("tools"):
                try:
                    calls = tool_turn(fake, host, body)
                except ValueError as e:
                    return self._send_json(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                           "message": str(e)}})
            mp = fake.profile.model(model)
            service = fake.draw(lambda r: mp.draw_service_s(answer.kind, r))
            roll = fake.draw(lambda r: r.random())
            outcome, edge = "ok", 0.0
            for name in ("rate_limited", "disconnect", "extra_data"):
                edge += float(mp.errors.get(name, 0.0))
                if roll < edge:
                    outcome = name
                    break
            record = {"t0": round(t0, 3), "backend": backend, "model": model, "kind": answer.kind,
                      "prompt_sha16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                      "stream": bool(body.get("stream")), "service_s": service}
            within = fake.enter(backend)
            try:
                if not within:
                    outcome = "rate_limited_concurrency"
                if outcome.startswith("rate_limited"):
                    time.sleep(REJECT_S * fake.time_scale)
                    self._reject(host)
                    return
                time.sleep(service * fake.time_scale)
                if outcome == "disconnect":
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                if outcome == "extra_data":
                    answer = Answer(answer.kind, answer.text + "\n" + answer.text, answer.usage)
                record["gen_id"] = (self._tool_calls(host, model, body, prompt, answer, calls) if calls
                                    else self._complete(host, model, body, prompt, answer))
                record["tool_calls"] = [n for n, _a in calls or ()]
            except (BrokenPipeError, ConnectionResetError):
                outcome = "client_gone"
            finally:
                fake.leave(backend)
                fake.log({**record, "outcome": outcome, "t1": round(time.time(), 3)})

        def _reject(self, host: str) -> None:
            if host == GOOGLE:
                doc = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "rehearsal 429"}}
            else:
                doc = {"error": {"code": 429, "message": "Rate limit exceeded (rehearsal)"}}
            self._send_json(429, doc, {"Retry-After": "1"})

        def _tool_calls(self, host, model, body, prompt, answer: Answer, calls) -> str:
            """One scripted tool turn: function calls instead of an answer (streamed when asked)."""
            usage = {"prompt": max(1, len(prompt.encode()) // 4), "completion": 20 * len(calls), "cached": 0,
                     "reasoning": 0}
            if host == GOOGLE:
                gid = "rehearsal-" + secrets.token_hex(8)
                parts = []
                for i, (n, a) in enumerate(calls):
                    sig = secrets.token_hex(12)
                    fake.signatures.add(sig)
                    parts.append({"functionCall": {"name": n, "args": a, **({"id": f"fc{i}"} if i else {})},
                                  "thoughtSignature": sig})
                self._send_json(200, {
                    "candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": "STOP", "index": 0}],
                    "usageMetadata": {"promptTokenCount": usage["prompt"], "candidatesTokenCount": usage["completion"],
                                      "totalTokenCount": usage["prompt"] + usage["completion"]},
                    "modelVersion": model, "responseId": gid})
                return gid
            gid = "gen-rehearsal-" + secrets.token_hex(10)
            wire_usage = {"prompt_tokens": usage["prompt"], "completion_tokens": usage["completion"],
                          "total_tokens": usage["prompt"] + usage["completion"]}
            head = {"id": gid, "model": model}
            if host == OPENROUTER:
                cost = fake.cost(model, usage)
                wire_usage["cost"] = float(cost)
                served = fake.profile.model(model).served_by
                head["provider"] = served
                fake.generations[gid] = {"id": gid, "total_cost": float(cost), "model": model, "provider_name": served,
                                         "tokens_prompt": usage["prompt"], "tokens_completion": usage["completion"],
                                         "native_tokens_reasoning": 0, "streamed": bool(body.get("stream")),
                                         "cancelled": False, "finish_reason": "tool_calls"}
            tcs = [{"id": f"call_{secrets.token_hex(4)}", "type": "function",
                    "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
            if not body.get("stream"):
                self._send_json(200, {**head, "object": "chat.completion", "created": int(time.time()),
                                      "choices": [{"index": 0, "finish_reason": "tool_calls",
                                                   "message": {"role": "assistant", "content": None, "tool_calls": tcs}}],
                                      "usage": wire_usage}, {"X-Generation-Id": gid})
                return gid
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Generation-Id", gid)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            frames = [{**head, "choices": [{"index": 0, "delta": {"reasoning": "…"}}]}]
            for i, tc in enumerate(tcs):
                args = tc["function"]["arguments"]
                frames.append({**head, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"],
                                                                                   "arguments": args[:2]}}]}}]})
                frames.append({**head, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "function": {"arguments": args[2:]}}]}}]})
            frames += [{**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                       {**head, "choices": [], "usage": wire_usage}]
            for frame in frames:
                self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return gid

        def _complete(self, host, model, body, prompt, answer: Answer) -> str:
            usage = dict(answer.usage)
            # Never more input tokens than the prompt has bytes (the client's reservation bound).
            usage["prompt"] = max(1, min(usage["prompt"], len(prompt.encode()) // 2))
            usage["cached"] = min(usage["cached"], usage["prompt"])
            if host == GOOGLE:
                gid = "rehearsal-" + secrets.token_hex(8)
                self._send_json(200, {
                    "candidates": [{"content": {"parts": [{"text": answer.text}], "role": "model"},
                                    "finishReason": "STOP", "index": 0}],
                    "usageMetadata": {"promptTokenCount": usage["prompt"],
                                      "candidatesTokenCount": usage["completion"] - usage["reasoning"],
                                      "thoughtsTokenCount": usage["reasoning"],
                                      "cachedContentTokenCount": usage["cached"],
                                      "totalTokenCount": usage["prompt"] + usage["completion"]},
                    "modelVersion": model, "responseId": gid})
                return gid
            gid = "gen-rehearsal-" + secrets.token_hex(10)
            served = ((body.get("provider") or {}).get("only") or [None])[0] or \
                fake.profile.model(model).served_by
            wire_usage = {"prompt_tokens": usage["prompt"], "completion_tokens": usage["completion"],
                          "total_tokens": usage["prompt"] + usage["completion"],
                          "prompt_tokens_details": {"cached_tokens": usage["cached"]},
                          "completion_tokens_details": {"reasoning_tokens": usage["reasoning"]}}
            if host == OPENROUTER:
                cost = fake.cost(model, usage)
                wire_usage["cost"] = float(cost)
                fake.generations[gid] = {
                    "id": gid, "total_cost": float(cost), "model": model, "provider_name": served,
                    "tokens_prompt": usage["prompt"], "tokens_completion": usage["completion"],
                    "native_tokens_reasoning": usage["reasoning"], "streamed": bool(body.get("stream")),
                    "cancelled": False, "finish_reason": "stop"}
            head = {"id": gid, "model": model, **({"provider": served} if host == OPENROUTER else {})}
            if not body.get("stream"):
                self._send_json(200, {**head, "object": "chat.completion", "created": int(time.time()),
                                      "choices": [{"index": 0, "finish_reason": "stop",
                                                   "message": {"role": "assistant", "content": answer.text}}],
                                      "usage": wire_usage}, {"X-Generation-Id": gid})
                return gid
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Generation-Id", gid)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            step = max(1, len(answer.text) // STREAM_CHUNKS + 1)
            frames = [{**head, "choices": [{"index": 0, "delta": {"reasoning": "…"}}]}]
            frames += [{**head, "choices": [{"index": 0, "delta": {"content": answer.text[i:i + step]}}]}
                       for i in range(0, len(answer.text), step)]
            frames += [{**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                       {**head, "choices": [], "usage": wire_usage}]
            for frame in frames:
                self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return gid

    return Handler
