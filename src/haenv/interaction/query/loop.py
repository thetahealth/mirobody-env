"""The query-mode loop: native function calling, bounded rounds and calls, a guarded result per call.

A query solver implements `chat(system, messages, tools, tool_choice) -> Turn`. `messages` are
the neutral transcript (`transport.make_envelope`): `user` / `assistant` with `tool_calls` and the
provider's `native` turn / `tool` with `tool_call_id`; `tools` is the list of `ToolSpec`. An
assistant turn without tool calls is the final answer.

Termination: at most `max_rounds` tool turns, then one forced final turn on which the tools stay
declared and are closed (`tool_choice = none`). Calls past `max_calls` are answered with an
exhausted-budget result and the next turn is the forced final one. A model that still asks for
tools on the forced final turn has not answered (`stop = unanswered`); the cell is scored on an
empty answer, as a model that ran out of rounds in the gated geometry is.

A guard violation in a tool result raises `QueryLeak` before the model sees the result; the row
builder turns it into `ABORT(leak)`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from haenv.interaction.query.tools import QueryLeak
from haenv.interaction.query.tools import ResultGuard
from haenv.interaction.query.tools import ToolSpec
from haenv.interaction.query.tools import call_tool

FORCE_FINAL = "工具轮次或调用次数已用完,工具已关闭。请不要再调用工具,直接按要求的 schema 输出最终答案 JSON。"


@dataclass
class Turn:
    content: str = ""
    tool_calls: list = field(default_factory=list)      # [{"id", "name", "arguments": dict | str}]
    native: dict | None = None                          # provider turn to send back verbatim
    usage: dict | None = None
    finish: str | None = None


@dataclass
class QueryCall:
    round: int
    name: str
    arguments: object
    error: str | None              # unknown_tool / bad_arguments / over_budget / None
    found: bool | None             # the tool's own `found` flag (None when the tool has none)
    result_chars: int
    n_rows: int | None = None      # points or rows returned


@dataclass
class QueryTrace:
    calls: list = field(default_factory=list)
    rounds_used: int = 0
    stop: str = ""                 # answered / forced_final / unanswered
    protocol_errors: int = 0
    usage: list = field(default_factory=list)
    finish: list = field(default_factory=list)
    messages: list = field(default_factory=list)       # the whole transcript, persisted for re-scoring
    system: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["calls"] = [asdict(c) if not isinstance(c, dict) else c for c in self.calls]
        return d


def _n_rows(res) -> int | None:
    if not isinstance(res, dict):
        return None
    if isinstance(res.get("entry"), list):
        return len(res["entry"])
    return None


def run_query(specs: list[ToolSpec], guard: ResultGuard, system: str, user: str, solver, *,
              max_rounds: int, max_calls: int) -> tuple[str | None, QueryTrace]:
    """Run one cell; returns `(final text or None, trace)`. `guard` checks the question and every
    tool result before the model sees it."""
    by_name = {s.name: s for s in specs}
    tr = QueryTrace(system=system)
    for txt in (system, user):
        v = guard.for_prompt().check(txt)
        if v:
            raise QueryLeak(v)
    msgs: list[dict] = [{"role": "user", "content": user}]
    tr.messages = msgs
    n_calls = 0
    forced = False
    for r in range(1, max_rounds + 2):
        final_turn = r > max_rounds or forced
        if final_turn:
            msgs.append({"role": "user", "content": FORCE_FINAL})
        turn = solver.chat(system, msgs, specs, "none" if final_turn else "auto")
        tr.rounds_used = r
        tr.usage.append(turn.usage)
        tr.finish.append(turn.finish)
        calls = list(turn.tool_calls or ())
        msgs.append({"role": "assistant", "content": turn.content or "",
                     **({"tool_calls": [{"id": c.get("id") or f"c{r}.{i}", "name": str(c.get("name")),
                                         "arguments": c.get("arguments")} for i, c in enumerate(calls)]}
                        if calls else {}),
                     **({"native": turn.native} if turn.native else {})})
        if final_turn and calls:
            tr.protocol_errors += 1                  # asked for tools after the tools were closed
            tr.stop = "unanswered"
            return None, tr
        if not calls:
            tr.stop = "forced_final" if final_turn else "answered"
            return turn.content or "", tr
        for i, c in enumerate(calls):
            cid = c.get("id") or f"c{r}.{i}"
            if n_calls >= max_calls:
                res, err = {"error": "调用次数已用完"}, "over_budget"
                forced = True
            else:
                res, err = call_tool(by_name, c.get("name"), c.get("arguments"))
                n_calls += 1
                if err:
                    tr.protocol_errors += 1
            v = guard.check(res)
            if v:
                raise QueryLeak(v)
            body = json.dumps(res, ensure_ascii=False, separators=(",", ":"))
            tr.calls.append(QueryCall(r, str(c.get("name")), c.get("arguments"), err,
                                      res.get("found") if isinstance(res, dict) else None, len(body), _n_rows(res)))
            msgs.append({"role": "tool", "tool_call_id": cid, "name": str(c.get("name")), "content": res})
        if n_calls >= max_calls:
            forced = True
    tr.stop = "unanswered"                            # unreachable: the forced final turn always returns
    return None, tr
